"""Analyze a mem-profile CSV (see mem_profile.py): rank which counter's growth
tracks the RSS climb, and run the decisive revisit node-count check.

stdlib-only -- runs on the host (no kairos/numpy needed):

    python3 -m kairos_python.analyze_mem /path/to/mem.csv
    # or: python3 kairos/python/src/kairos_python/analyze_mem.py mem.csv
"""
import csv
import sys


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def _fmt(v):
    if isinstance(v, float):
        if v != v:  # nan
            return "  n/a"
        return f"{v:+.3f}"
    return str(v)


def main(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"empty CSV: {path}")
        return
    cols = [c for c in rows[0] if c not in ("step", "timestamp_ns")]

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (ValueError, KeyError):
                out.append(float("nan"))
        return out

    rss = col("vmrss_kb")
    n = len(rows)
    print(f"{path}: {n} samples, "
          f"steps {rows[0]['step']}..{rows[-1]['step']}")
    rss_mb0, rss_mb1 = rss[0] / 1024, rss[-1] / 1024
    print(f"RSS: {rss_mb0:.0f} -> {rss_mb1:.0f} MB "
          f"(+{rss_mb1 - rss_mb0:.0f} MB, x{rss_mb1 / max(rss_mb0,1e-9):.2f})")

    # Accumulation vs fragmentation: does the RSS climb show up as in-use heap
    # (real accumulation) or as free-but-retained heap (glibc fragmentation)?
    inuse, free = col("heap_inuse_kb"), col("heap_free_kb")
    if inuse[0] != -1 and inuse[-1] != -1:
        d_rss = rss[-1] - rss[0]
        d_in = inuse[-1] - inuse[0]
        d_free = free[-1] - free[0]
        print(f"heap in-use: {inuse[0]/1024:.0f} -> {inuse[-1]/1024:.0f} MB "
              f"(+{d_in/1024:.0f} MB) | free-retained: +{d_free/1024:.0f} MB")
        if d_rss > 0:
            frac = d_in / d_rss
            if frac > 0.7:
                verdict = "REAL ACCUMULATION (in-use heap explains the RSS climb) -> localize WHERE"
            elif frac < 0.3:
                verdict = "FRAGMENTATION / non-heap (in-use flat; RSS held by free heap or mmap) -> malloc_trim / per-scene-process"
            else:
                verdict = "MIXED (in-use explains part; rest is fragmentation/non-heap)"
            print(f"  in-use accounts for {frac*100:.0f}% of RSS growth -> {verdict}")
    print()

    # Per-series: first, last, delta, and correlation with RSS. A series whose
    # corr-with-RSS ~1 AND grows substantially is a culprit; flat series (corr
    # n/a because variance 0, or small delta) are exonerated.
    rows_out = []
    for c in cols:
        if c == "vmrss_kb":
            continue
        v = col(c)
        if all(x != x for x in v):  # all nan
            continue
        first, last = v[0], v[-1]
        # ignore Phase-2 placeholder columns left at -1
        if first == -1 and last == -1:
            continue
        corr = _pearson(rss, v)
        rows_out.append((c, first, last, last - first, corr))

    # sort by |corr| desc, then by delta magnitude
    rows_out.sort(key=lambda t: (abs(t[4]) if t[4] == t[4] else -1, abs(t[3])),
                  reverse=True)
    w = max(len(c) for c, *_ in rows_out)
    print(f"{'series'.ljust(w)}  {'first':>10} {'last':>10} {'delta':>10}  corr(RSS)")
    print("-" * (w + 46))
    for c, first, last, delta, corr in rows_out:
        print(f"{c.ljust(w)}  {first:>10.0f} {last:>10.0f} {delta:>+10.0f}  {_fmt(corr)}")

    # Duplication check at run thirds: in a fixed hall the distinct OCCUPIED
    # CELLS (0.5 m grid) should plateau after pass 1 (area is bounded). If node
    # count keeps climbing while occupied cells stay ~flat, the pipeline is
    # minting duplicate nodes on already-mapped ground (the growth mechanism).
    print("\nduplication check at run thirds  (nodes / occupied-cells):")
    idx = [0, n // 3, (2 * n) // 3, n - 1]
    for layer in ("l3_places", "l20_surface"):
        a, ar, oc = (col(f"{layer}_active"), col(f"{layer}_archived"),
                     col(f"{layer}_occupied_cells"))
        tot = [int(a[i] + ar[i]) for i in idx]
        cells = [int(oc[i]) if oc[i] == oc[i] else -1 for i in idx]
        if all(t == 0 for t in tot):
            continue
        print(f"  {layer:12s}: "
              + " -> ".join(f"{t}/{c}" for t, c in zip(tot, cells)))
        # compare growth from ~end-of-pass-1 (thirds[1]) to end
        if cells[1] > 0 and tot[1] > 0:
            node_g = tot[-1] / tot[1]
            cell_g = cells[-1] / cells[1]
            if node_g > 1.3 and cell_g < 1.3:
                print(f"    -> DUPLICATION: nodes x{node_g:.1f} while occupied "
                      f"cells ~flat (x{cell_g:.1f}) -> new nodes on mapped ground")
            elif cell_g >= 1.3:
                print(f"    -> still covering NEW ground (cells x{cell_g:.1f}) "
                      "-> not (only) duplication")

    # Global kairos dynamics voxel map: does it plateau (position-keyed, bounded
    # by area) or grow? 0 => no detections pushed (dynamics off/empty).
    fc = col("flow_cells")
    fcv = [int(fc[i]) if fc[i] == fc[i] else -1 for i in idx]
    if any(v > 0 for v in fcv):
        print("  flow_cells (global dynamics voxels): "
              + " -> ".join(str(v) for v in fcv))
    else:
        print("  flow_cells: 0  (no detections pushed -> dynamics map empty)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
