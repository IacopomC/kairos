"""Per-layer memory instrumentation for ``kairos run``.

Samples process RSS alongside scene-graph / dynamics counters every N frames
into a CSV, to pin *which layer's* growth tracks the RSS climb over multiple
passes of a fixed environment (the within-scene RAM question in
``HANDOFF_gate_and_memory_2026-06-13.md``).

Phase 1 (this module, **no rebuild**): everything reachable from Python today --
DSG node counts per layer split active/archived (the decisive triple-or-flat
check on revisit), object trajectory-history lengths (the uncapped per-node
history), and flow dynamics size. The mesh-archive (PGMO ``DeltaCompression``)
and TSDF-block counters need small pybind additions (Phase 2) and are emitted as
``-1`` until those bindings exist.

Usage: pass ``--mem-profile-csv PATH`` (and optionally ``--mem-profile-every N``,
default 50) to ``kairos run``. Plot each column against ``vmrss_kb`` over the run;
the series whose slope matches RSS is the culprit.
"""
import csv

# DSG layer ids -- see hydra-kairos/doc/scene_graph_overview.md. Plain ints so we
# don't depend on the DsgLayers enum being importable in every env.
_LAYERS = {2: "l2_objects", 3: "l3_places", 20: "l20_surface"}


def _read_vmrss_kb():
    """Resident set size in kB from /proc, or -1 if unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # already in kB
    except OSError:
        pass
    return -1


_LIBC = None


def _malloc_info_kb():
    """Ground truth on the heap via glibc malloc_info(), which reports ALL
    arenas. Hydra is multithreaded, so a single-arena reading undercounts.

    Returns (total_kb, free_kb, inuse_kb): bytes glibc holds from the OS
    (sbrk+mmap), bytes free-but-retained (fragmentation), and in-use. If RSS
    climbs but inuse is flat while free grows -> fragmentation, not a leak.
    (-1, -1, -1) if unavailable.
    """
    global _LIBC
    import ctypes
    import re
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
            _LIBC.fopen.restype = ctypes.c_void_p
            _LIBC.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            _LIBC.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
            _LIBC.fclose.argtypes = [ctypes.c_void_p]
        import os
        path = f"/tmp/kairos_minfo_{os.getpid()}.xml".encode()
        fp = _LIBC.fopen(path, b"w")
        if not fp:
            return -1, -1, -1
        _LIBC.malloc_info(0, fp)
        _LIBC.fclose(fp)
        with open(path) as f:
            xml = f.read()

        # The aggregate totals are the LAST occurrence of each tag (after all
        # per-arena <heap> blocks).
        def last(tag, typ):
            ms = re.findall(rf'<{tag} type="{typ}"[^>]*size="(\d+)"', xml)
            return int(ms[-1]) if ms else 0

        system = last("system", "current")  # sbrk bytes from OS (all arenas)
        mmap_ = last("total", "mmap")        # mmap'd bytes
        free = last("total", "fast") + last("total", "rest")
        total = system + mmap_
        return total // 1024, free // 1024, (total - free) // 1024
    except Exception:
        return -1, -1, -1


class MemProfiler:
    """Append one CSV row every ``every`` frames. Instrumentation-only:
    every accessor is defensively wrapped so a profiling error can never abort
    the run."""

    def __init__(self, csv_path, every=50, dsg_dump_dir=None):
        self.every = max(1, int(every))
        self.path = str(csv_path)
        self.dsg_dump_dir = dsg_dump_dir
        if dsg_dump_dir:
            import os
            os.makedirs(dsg_dump_dir, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        cols = ["step", "timestamp_ns", "vmrss_kb",
                "heap_total_kb", "heap_free_kb", "heap_inuse_kb"]
        for name in _LAYERS.values():
            cols += [f"{name}_active", f"{name}_archived"]
        cols += [
            "obj_traj_len_sum",
            "mesh_vertices",  # total published geometry (live + committed)
            "mesh_faces",
            "flow_cells",
            "flow_total_components",
            "flow_cells_valid_nudft",
            # Active-window voxel-layer footprint via map_memory_stats(). Block
            # counts are the bounded-memory signal (flat = active window works);
            # tsdf_bytes is exact, semantic_bytes is a lower bound (excludes the
            # per-voxel likelihood heap). -1 if the binding predates this field.
            "tsdf_blocks",
            "semantic_blocks",
            "tsdf_bytes",
            "semantic_bytes",
        ]
        self._w.writerow(cols)
        self._fh.flush()

    def maybe_sample(self, pipeline, step_idx, timestamp_ns):
        if step_idx % self.every == 0:
            self.sample(pipeline, step_idx, timestamp_ns)

    def sample(self, pipeline, step_idx, timestamp_ns):
        heap_total, heap_free, heap_inuse = _malloc_info_kb()
        row = [step_idx, int(timestamp_ns), _read_vmrss_kb(),
               heap_total, heap_free, heap_inuse]

        # DSG per-layer counts via the C++ dsg_stats() binding (plain ints --
        # avoids the cross-module pybind issue with returning the graph object).
        dsg = {}
        try:
            dsg = pipeline.dsg_stats()
        except Exception as e:
            if not getattr(self, "_warned_dsg", False):
                import sys
                print(f"[mem-profile] pipeline.dsg_stats() failed: {e!r} "
                      "(DSG columns -> -1; rebuild _kairos_bindings)",
                      file=sys.stderr)
                self._warned_dsg = True
        for layer_id in _LAYERS:
            row += [dsg.get(f"l{layer_id}_active", -1),
                    dsg.get(f"l{layer_id}_archived", -1)]
        row.append(dsg.get("obj_traj_len_sum", -1))
        row += [dsg.get("mesh_vertices", -1), dsg.get("mesh_faces", -1)]

        cells = comps = valid = -1
        try:
            s = pipeline.flow_stats()
            cells = s.get("cells", -1)
            comps = s.get("total_components", -1)
            valid = s.get("cells_with_valid_nudft", -1)
        except Exception:
            pass
        row += [cells, comps, valid]

        # Active-window voxel-layer footprint (bounded by the window).
        tsdf_blocks = sem_blocks = tsdf_bytes = sem_bytes = -1
        try:
            mm = pipeline.map_memory_stats()
            tsdf_blocks = mm.get("tsdf_blocks", -1)
            sem_blocks = mm.get("semantic_blocks", -1)
            tsdf_bytes = mm.get("tsdf_bytes", -1)
            sem_bytes = mm.get("semantic_bytes", -1)
        except Exception:
            if not getattr(self, "_warned_mapmem", False):
                import sys
                print("[mem-profile] pipeline.map_memory_stats() unavailable "
                      "(map memory columns -> -1; rebuild _kairos_bindings)",
                      file=sys.stderr)
                self._warned_mapmem = True
        row += [tsdf_blocks, sem_blocks, tsdf_bytes, sem_bytes]

        self._w.writerow(row)
        self._fh.flush()

        # Snapshot the backend DSG (JSON + mesh) for visual inspection.
        if self.dsg_dump_dir:
            try:
                import os
                pipeline.dump_dsg(
                    os.path.join(self.dsg_dump_dir, f"dsg_{step_idx:06d}.json"),
                    True)
            except Exception as e:
                if not getattr(self, "_warned_dump", False):
                    import sys
                    print(f"[mem-profile] dump_dsg failed: {e!r}", file=sys.stderr)
                    self._warned_dump = True

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
