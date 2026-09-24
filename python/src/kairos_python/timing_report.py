#!/usr/bin/env python3
"""Timing and memory report for KAIROS / Hydra runs.

Reproduces the timing methodology of Hughes et al. 2024 ("Foundations of spatial
perception for robotics", IJRR) against a directory of KAIROS run outputs, and adds
the KAIROS-specific temporal-layer cost and a memory summary.

It is pure post-processing: it only reads the CSV/JSON that a run already writes
(``*_timing_raw.csv``, ``train.stats.json``, ``mem_scene_*.csv``). It does NOT load
the C++ bindings and needs no Docker, so it can run anywhere with a stdlib Python.
matplotlib is optional and only needed for the figures.

Usage
-----
    python3 timing_report.py --results results/kairos_timing
    python3 timing_report.py --results results/kairos_timing --budget-ms 200 --out /tmp/rep

The ``--results`` path may point at a single scene directory (one that contains
``frontend/spin_timing_raw.csv``) or at any parent directory; every scene found
underneath it is included and aggregated.

What it reports
---------------
* Per-component timing grouped into the Hydra tiers (mid-level frontend that must
  meet the keyframe budget; high-level backend that runs asynchronously and may lag).
  For each timer: mean +/- std, median, p95, max, fraction over the real-time budget,
  and the growth ratio (2nd-half mean / 1st-half mean) that exposes accumulation.
* The KAIROS temporal layer cost (add_observation / sharing / remap_cells), read
  from ``train.stats.json``'s ``flow_timing``.
* A memory summary (peak process RSS, and the RSS-vs-time curve if a mem_scene CSV
  is present).
* The data cadence (median inter-frame dt -> sensor rate) detected from the data,
  so the chosen real-time budget is always stated against the true frame rate.

See ``kairos/doc/timing.md`` for the full protocol and how to cite it.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Configuration: which timers make up the headline table, and their grouping.
# Edit this to match a different dataset's layer set; the full per-timer CSV is
# always emitted regardless, so nothing is ever silently dropped.
# --------------------------------------------------------------------------- #

# Timer folders Hydra writes (relative to a scene dir). reconstruction/ and places/
# are KAIROS additions to the same scheme.
TIMER_FOLDERS = ["reconstruction", "places", "frontend", "backend", "lcd"]

# (tier label) -> list of (timer key, display name). Timer key is "folder/stem".
HEADLINE_TIERS = [
    (
        "Mid-level / frontend (must meet keyframe budget)",
        [
            ("reconstruction/spin", "Reconstruction (TSDF+mesh)"),
            ("places/overall_update", "Places (GVD -> graph)"),
            ("frontend/objects_detection", "Objects"),
            ("frontend/spin", "Frontend total"),
        ],
    ),
    (
        "High-level / backend (asynchronous slow thread, may lag)",
        [
            ("backend/room_detection", "Rooms"),
            ("backend/update_layers", "Backend update/optimize"),
            ("backend/spin", "Backend total"),
        ],
    ),
]

# KAIROS temporal-layer timers, read from train.stats.json -> flow_timing.
# Each entry: (json prefix, display, cadence, count_field). Mean ms is
# total_us / count_field. Cadence groups them by the unit that makes them
# meaningful: "frame" steps compete for the keyframe budget; "event" steps run
# with the backend; "query" is eval-time only. ingest has no own count so it
# borrows process_frame_count.
FLOW_TIMERS = [
    # Per frame (compare to the keyframe budget). Every sub-row is divided by
    # process_frame_count so the group is additive: the sub-rows sum to the
    # total. Coupling does not run on every frame, so its per-frame figure is
    # amortized over all frames; its cost per run appears as a micro-stat below.
    ("process_frame", "Temporal per-frame total", "frame", "process_frame_count"),
    ("ingest", "  - observation ingest", "frame", "process_frame_count"),
    ("coupling", "  - edge coupling", "frame", "process_frame_count"),
    ("frame_other", "  - visibility + presence exposure", "frame", "process_frame_count"),
    # Per event (runs with the backend, not the per-frame loop).
    ("place_summaries", "Place summaries", "event", "place_summaries_count"),
    ("remap_cells", "Remap (loop closure)", "event", "remap_cells_count"),
    ("run_maintenance", "Maintenance", "event", "run_maintenance_count"),
    # Per query (evaluation only).
    ("score", "Scoring (per call)", "query", "score_count"),
    ("score_per_det", "Scoring (per detection)", "query", "score_detections"),
    # Supplementary micro-stats, not the headline. Each is divided by its own
    # count: sharing pools a neighbourhood when the weights are read, at a
    # cadence unrelated to the number of detections pushed in.
    ("add_observation", "Add observation (per detection)", "micro", "add_observation_count"),
    ("sharing", "Evidence sharing (per pooled neighbourhood)", "micro", "sharing_count"),
    ("coupling_per_run", "Edge coupling (per run, not per frame)", "micro", "coupling_count"),
]

# The timer used to detect the data cadence and to define the per-frame timeline.
CADENCE_TIMER = "frontend/spin"


# --------------------------------------------------------------------------- #
# Small numeric helpers (stdlib only).
# --------------------------------------------------------------------------- #


def _percentile(sorted_vals, q):
    """Linear-interpolation percentile (q in [0,1]) over an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class TimerStats:
    """Summary of one timer's elapsed times (seconds) for one scene."""

    key: str
    n: int = 0
    mean_s: float = float("nan")
    std_s: float = float("nan")
    median_s: float = float("nan")
    p95_s: float = float("nan")
    max_s: float = float("nan")
    frac_over_budget: float = float("nan")
    growth_ratio: float = float("nan")  # 2nd-half mean / 1st-half mean


def summarize_timer(key, stamps_ns, elapsed_s, budget_s):
    """Compute a TimerStats from parallel timestamp / elapsed lists."""
    n = len(elapsed_s)
    st = TimerStats(key=key, n=n)
    if n == 0:
        return st
    sorted_e = sorted(elapsed_s)
    st.mean_s = statistics.fmean(elapsed_s)
    st.std_s = statistics.pstdev(elapsed_s) if n > 1 else 0.0
    st.median_s = _percentile(sorted_e, 0.5)
    st.p95_s = _percentile(sorted_e, 0.95)
    st.max_s = sorted_e[-1]
    st.frac_over_budget = sum(1 for v in elapsed_s if v > budget_s) / n
    if n >= 4:
        h = n // 2
        first = statistics.fmean(elapsed_s[:h])
        second = statistics.fmean(elapsed_s[h:])
        st.growth_ratio = (second / first) if first > 0 else float("nan")
    return st


# --------------------------------------------------------------------------- #
# Loading.
# --------------------------------------------------------------------------- #


def load_raw_timer(path):
    """Read a *_timing_raw.csv -> (stamps_ns[list[int]], elapsed_s[list[float]])."""
    stamps, elapsed = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: timestamp(ns),elapsed(s)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                stamps.append(int(row[0]))
                elapsed.append(float(row[1]))
            except ValueError:
                continue
    return stamps, elapsed


def find_scene_dirs(results_path):
    """Return scene dirs (those containing the cadence timer) under results_path."""
    needle = os.path.join("frontend", "spin_timing_raw.csv")
    if os.path.isfile(os.path.join(results_path, needle)):
        return [results_path]
    found = []
    for p in glob.glob(os.path.join(results_path, "**", needle), recursive=True):
        found.append(os.path.dirname(os.path.dirname(p)))
    return sorted(set(found))


def scene_label(results_path, scene_dir):
    rel = os.path.relpath(scene_dir, results_path)
    return rel if rel != "." else os.path.basename(scene_dir.rstrip("/"))


def load_scene_timers(scene_dir):
    """Load every *_timing_raw.csv in a scene -> {key: (stamps_ns, elapsed_s)}."""
    timers = {}
    for folder in TIMER_FOLDERS:
        fdir = os.path.join(scene_dir, folder)
        if not os.path.isdir(fdir):
            continue
        for path in glob.glob(os.path.join(fdir, "*_timing_raw.csv")):
            stem = os.path.basename(path)[: -len("_timing_raw.csv")]
            timers[f"{folder}/{stem}"] = load_raw_timer(path)
    return timers


def apply_warmup(stamps_ns, elapsed_s, warmup_sec):
    """Drop all samples within warmup_sec of the first timestamp."""
    if not stamps_ns or warmup_sec <= 0:
        return stamps_ns, elapsed_s
    t0 = stamps_ns[0]
    cutoff = t0 + int(warmup_sec * 1e9)
    keep = [(t, e) for t, e in zip(stamps_ns, elapsed_s) if t >= cutoff]
    if not keep:
        return stamps_ns, elapsed_s
    return [t for t, _ in keep], [e for _, e in keep]


def detect_cadence(stamps_ns):
    """Median inter-frame dt (s) and rate (Hz), excluding recording gaps."""
    if len(stamps_ns) < 3:
        return float("nan"), float("nan")
    s = sorted(stamps_ns)
    dts = [(s[i] - s[i - 1]) / 1e9 for i in range(1, len(s))]
    med = statistics.median(dts)
    # Exclude gaps (e.g. paused recording) before reporting the working cadence.
    gap_thresh = max(0.5, 5.0 * med)
    good = [d for d in dts if d <= gap_thresh]
    med_good = statistics.median(good) if good else med
    rate = 1.0 / med_good if med_good > 0 else float("nan")
    return med_good, rate


def load_flow_and_memory(scene_dir):
    """Read train.stats.json -> (flow_timing, memory, flow_stats, elapsed_s)."""
    for name in ("train.stats.json", "preds.stats.json"):
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                d = json.load(f)
            return (
                d.get("flow_timing", {}) or {},
                d.get("memory", {}) or {},
                d.get("flow_stats", {}) or {},
                d.get("elapsed_s"),
            )
    return {}, {}, {}, None


def find_mem_scene_csv(results_path, scene_dir):
    """Locate the mem_scene_*.csv that a run writes next to the scene dir."""
    base = os.path.basename(scene_dir.rstrip("/"))  # scene_tbd_<...>
    cand = os.path.join(os.path.dirname(scene_dir), f"mem_{base}.csv")
    return cand if os.path.isfile(cand) else None


def load_mem_series(path):
    """Read mem_scene_*.csv -> dict of column -> list (numeric where possible)."""
    cols = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    val = float("nan")
                cols.setdefault(k, []).append(val)
    return cols


# --------------------------------------------------------------------------- #
# Aggregation across scenes.
# --------------------------------------------------------------------------- #


@dataclass
class SceneReport:
    label: str
    cadence_dt_s: float
    cadence_hz: float
    timers: dict = field(default_factory=dict)  # key -> TimerStats
    pooled: dict = field(default_factory=dict)  # key -> list elapsed_s (post-warmup)
    flow: dict = field(default_factory=dict)  # display -> mean_ms
    cells: int = 0          # persistent temporal cells at end of scene
    n_frames: int = 0       # process_frame_count (frames the worker processed)
    peak_rss_mb: float = float("nan")
    elapsed_s: float = None
    mem_path: str = None


def build_scene_report(results_path, scene_dir, budget_s, warmup_sec):
    label = scene_label(results_path, scene_dir)
    raw = load_scene_timers(scene_dir)

    timers, pooled = {}, {}
    for key, (stamps, elapsed) in raw.items():
        stamps, elapsed = apply_warmup(stamps, elapsed, warmup_sec)
        timers[key] = summarize_timer(key, stamps, elapsed, budget_s)
        pooled[key] = elapsed

    cadence_stamps = raw.get(CADENCE_TIMER, ([], []))[0]
    dt, hz = detect_cadence(cadence_stamps)

    flow_raw, mem, flow_stats, elapsed_s = load_flow_and_memory(scene_dir)
    flow = {}
    for prefix, disp, _cadence, count_field in FLOW_TIMERS:
        # Most timers expose {prefix}_total_us; ingest and per-detection scoring
        # borrow another timer's total. Mean ms = total_us / count_field.
        if prefix == "ingest":
            total_us = flow_raw.get("ingest_total_us")
        elif prefix == "score_per_det":
            total_us = flow_raw.get("score_total_us")
        elif prefix == "coupling_per_run":
            total_us = flow_raw.get("coupling_total_us")
        elif prefix == "frame_other":
            # processFrame spans the visible-voxel snapshot, ingest, the
            # visibility write-back, presence exposure, and coupling. Only
            # ingest and coupling carry their own timer, so the remainder is
            # reported as a residual rather than left out of the group.
            total = flow_raw.get("process_frame_total_us")
            parts = [flow_raw.get("ingest_total_us"), flow_raw.get("coupling_total_us")]
            total_us = None
            if total is not None and all(p is not None for p in parts):
                total_us = max(0, total - sum(parts))
        else:
            total_us = flow_raw.get(f"{prefix}_total_us")
        count = flow_raw.get(count_field)
        if count and total_us is not None:
            flow[disp] = (total_us / count) / 1000.0  # us -> ms
    peak_rss = mem.get("peak_rss_bytes")
    peak_rss_mb = peak_rss / (1024 * 1024) if peak_rss else float("nan")

    rep = SceneReport(
        label=label,
        cadence_dt_s=dt,
        cadence_hz=hz,
        timers=timers,
        pooled=pooled,
        flow=flow,
        cells=int(flow_stats.get("cells", 0) or 0),
        n_frames=int(flow_raw.get("process_frame_count", 0) or 0),
        peak_rss_mb=peak_rss_mb,
        elapsed_s=elapsed_s,
        mem_path=find_mem_scene_csv(results_path, scene_dir),
    )
    return rep


def aggregate(reports, key):
    """Aggregate one timer across scenes: scene-averaged mean +/- between-scene std,
    pooled median/p95/max, and mean fraction-over-budget and growth."""
    means = [r.timers[key].mean_s for r in reports if key in r.timers and r.timers[key].n]
    if not means:
        return None
    overs = [r.timers[key].frac_over_budget for r in reports if key in r.timers and r.timers[key].n]
    growths = [
        r.timers[key].growth_ratio
        for r in reports
        if key in r.timers and not math.isnan(r.timers[key].growth_ratio)
    ]
    pooled = []
    for r in reports:
        pooled.extend(r.pooled.get(key, []))
    pooled.sort()
    return {
        "scene_mean_s": statistics.fmean(means),
        "scene_std_s": statistics.pstdev(means) if len(means) > 1 else 0.0,
        "pooled_median_s": _percentile(pooled, 0.5),
        "pooled_p95_s": _percentile(pooled, 0.95),
        "pooled_max_s": pooled[-1] if pooled else float("nan"),
        "mean_frac_over": statistics.fmean(overs) if overs else float("nan"),
        "mean_growth": statistics.fmean(growths) if growths else float("nan"),
        "n_scenes": len(means),
    }


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #


def _ms(x):
    return x * 1000.0 if x is not None and not math.isnan(x) else float("nan")


def print_console(reports, budget_s, args):
    rates = [r.cadence_hz for r in reports if not math.isnan(r.cadence_hz)]
    dts = [r.cadence_dt_s for r in reports if not math.isnan(r.cadence_dt_s)]
    print("=" * 78)
    print(f"KAIROS timing report  ({len(reports)} scene(s))")
    print(f"  results : {args.results}")
    if rates:
        print(
            f"  cadence : median dt {statistics.median(dts)*1000:.1f} ms "
            f"-> {statistics.median(rates):.2f} Hz (detected from data)"
        )
    print(
        f"  budget  : {budget_s*1000:.0f} ms per frame "
        f"({1/budget_s:.1f} Hz keyframe target; Hughes2024 protocol)"
    )
    print(f"  warmup  : first {args.warmup_sec:.0f} s of each scene dropped")
    print("=" * 78)

    hdr = f"{'component':36s}{'mean+/-std ms':>18s}{'med':>8s}{'p95':>8s}{'max':>9s}{'>bud':>7s}{'grow':>7s}"
    for tier, items in HEADLINE_TIERS:
        print(f"\n[{tier}]")
        print(hdr)
        for key, disp in items:
            agg = aggregate(reports, key)
            if agg is None:
                print(f"{disp:36s}{'(missing)':>18s}")
                continue
            print(
                f"{disp:36s}"
                f"{_ms(agg['scene_mean_s']):8.2f} +/-{_ms(agg['scene_std_s']):6.2f}"
                f"{_ms(agg['pooled_median_s']):8.2f}"
                f"{_ms(agg['pooled_p95_s']):8.2f}"
                f"{_ms(agg['pooled_max_s']):9.1f}"
                f"{100*agg['mean_frac_over']:6.0f}%"
                f"{agg['mean_growth']:7.2f}"
            )

    # KAIROS temporal layer, grouped by the cadence that makes each meaningful.
    cadence_titles = {
        "frame": "per frame (compare to keyframe budget)",
        "event": "per event (runs with backend)",
        "query": "per query (eval-time only)",
        "micro": "supplementary micro-stat (per detection)",
    }
    print("\n[KAIROS temporal layer (scene-averaged mean ms, from flow_timing)]")
    for cad in ("frame", "event", "query", "micro"):
        rows = [(d, c) for (_p, d, c, _cf) in FLOW_TIMERS if c == cad]
        printed_header = False
        for disp, _c in rows:
            vals = [r.flow[disp] for r in reports if disp in r.flow]
            if not vals:
                continue
            if not printed_header:
                budget_note = (f"   [budget {budget_s*1000:.0f} ms]"
                               if cad == "frame" else "")
                print(f"  {cadence_titles[cad]}{budget_note}")
                printed_header = True
            mean_ms = statistics.fmean(vals)
            over = (f"  ({'OVER' if mean_ms > budget_s*1000 else 'within'} budget)"
                    if cad == "frame" else "")
            print(f"    {disp:38s}{mean_ms:10.4f} ms  (n={len(vals)}){over}")

    # Memory
    print("\n[Memory]")
    rss = [r.peak_rss_mb for r in reports if not math.isnan(r.peak_rss_mb)]
    if rss:
        print(
            f"  peak process RSS: {statistics.fmean(rss):.0f} MB (scene-avg), "
            f"max {max(rss):.0f} MB   [whole process incl. ORB vocab/models]"
        )
    n_mem = sum(1 for r in reports if r.mem_path)
    print(f"  mem_scene CSV (RSS-vs-time) available for {n_mem}/{len(reports)} scenes")
    print("  per-structure map MB (TSDF/semantics/GVD): NOT logged in these runs")
    print("    -> requires re-enabling getMemorySize() in volumetric_map.cpp + rerun")
    print("=" * 78)


def write_per_scene_csv(reports, path):
    fields = [
        "scene", "timer", "n", "mean_ms", "std_ms", "median_ms", "p95_ms",
        "max_ms", "frac_over_budget", "growth_ratio",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in reports:
            for key in sorted(r.timers):
                s = r.timers[key]
                w.writerow([
                    r.label, key, s.n, f"{_ms(s.mean_s):.4f}", f"{_ms(s.std_s):.4f}",
                    f"{_ms(s.median_s):.4f}", f"{_ms(s.p95_s):.4f}", f"{_ms(s.max_s):.4f}",
                    f"{s.frac_over_budget:.4f}", f"{s.growth_ratio:.4f}",
                ])


def write_aggregate_csv(reports, path):
    keys = set()
    for r in reports:
        keys.update(r.timers)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timer", "n_scenes", "scene_mean_ms", "scene_std_ms", "pooled_median_ms",
            "pooled_p95_ms", "pooled_max_ms", "mean_frac_over_budget", "mean_growth_ratio",
        ])
        for key in sorted(keys):
            agg = aggregate(reports, key)
            if agg is None:
                continue
            w.writerow([
                key, agg["n_scenes"], f"{_ms(agg['scene_mean_s']):.4f}",
                f"{_ms(agg['scene_std_s']):.4f}", f"{_ms(agg['pooled_median_s']):.4f}",
                f"{_ms(agg['pooled_p95_s']):.4f}", f"{_ms(agg['pooled_max_s']):.4f}",
                f"{agg['mean_frac_over']:.4f}", f"{agg['mean_growth']:.4f}",
            ])


def write_latex_table(reports, budget_s, path):
    lines = [
        "% Auto-generated by timing_report.py - KAIROS timing (Hughes2024 protocol)",
        "\\begin{tabular}{l r r r r}",
        "\\toprule",
        " & mean$\\pm$std [ms] & median [ms] & p95 [ms] & $>$budget \\\\",
        "\\midrule",
    ]
    for tier, items in HEADLINE_TIERS:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{tier}}}}} \\\\")
        for key, disp in items:
            agg = aggregate(reports, key)
            if agg is None:
                continue
            lines.append(
                f"\\quad {disp} & "
                f"${_ms(agg['scene_mean_s']):.1f}\\pm{_ms(agg['scene_std_s']):.1f}$ & "
                f"${_ms(agg['pooled_median_s']):.1f}$ & "
                f"${_ms(agg['pooled_p95_s']):.1f}$ & "
                f"${100*agg['mean_frac_over']:.0f}\\%$ \\\\"
            )
    # flow tier
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\emph{KAIROS temporal layer (mean ms)}} \\\\")
    for _prefix, disp, _cadence, _cf in FLOW_TIMERS:
        vals = [r.flow[disp] for r in reports if disp in r.flow]
        if vals:
            safe = disp.replace("-", "--").strip()
            lines.append(
                f"\\quad {safe} & ${statistics.fmean(vals):.4f}$ & -- & -- & -- \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"% real-time budget = {budget_s*1000:.0f} ms ({1/budget_s:.1f} Hz keyframe)",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Figures (optional; require matplotlib).
# --------------------------------------------------------------------------- #


def plot_trends(scene_dir, budget_s, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    raw = load_scene_timers(scene_dir)

    def series(key):
        st, el = raw.get(key, ([], []))
        if not st:
            return [], []
        t0 = st[0]
        return [(t - t0) / 1e9 for t in st], el

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Panel 1: mid-level frontend.
    for key, lbl in [
        ("reconstruction/spin", "reconstruction (TSDF+mesh)"),
        ("frontend/spin", "frontend total"),
    ]:
        x, y = series(key)
        if x:
            axes[0].plot(x, y, lw=0.7, label=lbl)
    axes[0].axhline(budget_s, ls="--", c="k")
    axes[0].text(0.99, budget_s, "real-time (keyframe period)  ", va="bottom", ha="right",
                 transform=axes[0].get_yaxis_transform(), fontsize=8)
    axes[0].set_ylabel("elapsed [s]")
    axes[0].set_title("Mid-level / frontend")
    axes[0].legend(loc="upper left", fontsize=8)

    # Panel 2: high-level backend.
    x, y = series("backend/spin")
    if x:
        axes[1].plot(x, y, lw=0.7, c="tab:red", label="backend total")
    axes[1].axhline(budget_s, ls="--", c="k")
    axes[1].text(0.99, budget_s, "real-time (keyframe period)  ", va="bottom", ha="right",
                 transform=axes[1].get_yaxis_transform(), fontsize=8)
    axes[1].set_ylabel("elapsed [s]")
    axes[1].set_xlabel("trajectory time [s]")
    axes[1].set_title("High-level / backend")
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_memory(mem_path, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cols = load_mem_series(mem_path)
    if "timestamp_ns" not in cols or "vmrss_kb" not in cols:
        return False
    t0 = cols["timestamp_ns"][0]
    x = [(t - t0) / 1e9 for t in cols["timestamp_ns"]]
    rss_mb = [v / 1024.0 for v in cols["vmrss_kb"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    l1, = ax.plot(x, rss_mb, lw=1.2, c="tab:green",
                  label="process RSS (grows: persistent graph)")
    ax.set_xlabel("trajectory time [s]")
    ax.set_ylabel("RSS [MB]")
    ax.set_title("Process memory over time")
    handles = [l1]
    if "l3_places_active" in cols:
        ax2 = ax.twinx()
        l2, = ax2.plot(x, cols["l3_places_active"], lw=0.8, c="tab:blue", alpha=0.7,
                       label="L3 places active (bounded by active window)")
        ax2.set_ylabel("L3 places (active)")
        handles.append(l2)
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _scene_sort_key(label):
    """Order scenes chronologically by the timestamp in scene_tbd_<ts>."""
    base = label.split("/")[-1]
    marker = "scene_tbd_"
    return base[base.find(marker) + len(marker):] if marker in base else base


def write_temporal_by_session(reports, path):
    """Per-session temporal cost in chronological order. The temporal map is
    loaded from the previous session and grown on top (stop-and-go), so this
    sequence reveals whether the per-frame temporal cost stays flat as the
    persistent map accumulates across sessions."""
    fields = [
        "order", "scene", "cells_end", "frames",
        "temporal_per_frame_ms", "ingest_ms", "coupling_ms", "visibility_presence_ms",
        "score_per_det_ms",
    ]
    ordered = sorted(reports, key=lambda r: _scene_sort_key(r.label))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, r in enumerate(ordered):
            w.writerow([
                i, r.label, r.cells, r.n_frames,
                f"{r.flow.get('Temporal per-frame total', float('nan')):.4f}",
                f"{r.flow.get('  - observation ingest', float('nan')):.4f}",
                f"{r.flow.get('  - edge coupling', float('nan')):.4f}",
                f"{r.flow.get('  - visibility + presence exposure', float('nan')):.4f}",
                f"{r.flow.get('Scoring (per detection)', float('nan')):.4f}",
            ])


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="scene dir or any parent dir of run outputs")
    ap.add_argument("--out", default=None,
                    help="output dir for CSV/LaTeX/figures (default: <results>/timing_report)")
    ap.add_argument("--budget-ms", type=float, default=200.0,
                    help="real-time budget per frame in ms (default 200 = 5 Hz, Hughes2024)")
    ap.add_argument("--warmup-sec", type=float, default=10.0,
                    help="drop the first N seconds of each scene (default 10)")
    ap.add_argument("--plot-scene", default=None,
                    help="scene label/substring to plot (default: first scene)")
    ap.add_argument("--no-plot", action="store_true", help="skip figures")
    args = ap.parse_args()

    budget_s = args.budget_ms / 1000.0
    out_dir = args.out or os.path.join(args.results, "timing_report")
    os.makedirs(out_dir, exist_ok=True)

    scene_dirs = find_scene_dirs(args.results)
    if not scene_dirs:
        raise SystemExit(f"No scenes (frontend/spin_timing_raw.csv) under {args.results}")

    reports = [build_scene_report(args.results, d, budget_s, args.warmup_sec)
               for d in scene_dirs]

    print_console(reports, budget_s, args)

    per_scene = os.path.join(out_dir, "timing_per_scene.csv")
    agg_csv = os.path.join(out_dir, "timing_aggregate.csv")
    tex = os.path.join(out_dir, "timing_table.tex")
    by_session = os.path.join(out_dir, "temporal_by_session.csv")
    write_per_scene_csv(reports, per_scene)
    write_aggregate_csv(reports, agg_csv)
    write_latex_table(reports, budget_s, tex)
    write_temporal_by_session(reports, by_session)
    print(f"\nwrote: {per_scene}\n       {agg_csv}\n       {tex}\n       {by_session}")

    if not args.no_plot:
        sel = scene_dirs[0]
        if args.plot_scene:
            for d in scene_dirs:
                if args.plot_scene in d:
                    sel = d
                    break
        sel_label = scene_label(args.results, sel).replace("/", "_")
        trend_png = os.path.join(out_dir, f"trends_{sel_label}.png")
        if plot_trends(sel, budget_s, trend_png):
            print(f"       {trend_png}")
        mem_path = find_mem_scene_csv(args.results, sel)
        if mem_path:
            mem_png = os.path.join(out_dir, f"memory_{sel_label}.png")
            if plot_memory(mem_path, mem_png):
                print(f"       {mem_png}")


if __name__ == "__main__":
    main()
