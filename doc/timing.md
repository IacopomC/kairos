# Timing and memory analysis

How to measure and report the runtime and memory of a KAIROS run on any dataset. The
protocol follows Hughes et al. 2024 ("Foundations of spatial perception for robotics",
IJRR) and adds the KAIROS-specific temporal-layer cost.

---

## 1. What "real-time" means here

Offline replay feeds frames back-to-back with no pacing, so the end-to-end
iterations/second of a run measure *throughput*, and are reported as throughput.

Real-time, following Hydra, is a **per-component, per-keyframe** statement: each module
must process one keyframe within the keyframe budget. Hydra organises work into logical
threads that run concurrently ("thinking fast and slow"):

* **Early perception** (frame rate): feature tracking / VIO, 2D semantic segmentation,
  stereo depth. In KAIROS these are **provided externally** (precomputed depth + labels +
  odometry/tracks), so the timing reports them as "external".
* **Mid-level / frontend** (sub-second): mesh + places + objects reconstruction and the
  scene-graph frontend. **This must meet the keyframe budget.**
* **High-level / backend** (slow, infrequent): loop-closure detection, scene-graph
  optimization, room detection. Runs **asynchronously on its own thread and is allowed to
  lag**; its cost grows with map size and spikes when loop closures fire.

The real-time budget is the **keyframe period**. Hughes 2024 use 5 Hz on a workstation
(**200 ms**) and ~1 Hz on an embedded Xavier. We adopt the 5 Hz / 200 ms workstation
budget. The script auto-detects the actual data cadence and prints it next to the chosen
budget.

> **Step mode vs threaded deployment.** The offline pipeline runs
> `active_window -> frontend -> backend` **synchronously in one thread** (step mode); the
> deployed system runs the frontend and backend on **separate threads**. Each timer wraps
> only its own code region. (i) Step mode has *less* core contention than concurrent
> deployment, so per-component numbers are best-case; (ii) the serial sum of
> frontend+backend per frame overstates the deployed per-frame latency, because the
> backend overlaps the frontend on its own thread.

---

## 2. What the pipeline already records

Every run writes, under each scene output directory:

* **Per-component timing** — `*_timing_raw.csv` in `reconstruction/`, `places/`,
  `frontend/`, `backend/`, `lcd/`. Each row is `timestamp(ns),elapsed(s)` for one call of
  that timer. Produced by Hydra's `ScopedTimer` (wall-clock
  `std::chrono::high_resolution_clock`; see `hydra/utils/timing_utilities.h`).
* **Aggregate timing** — `timing_stats.csv`: one row per timer with mean/min/max/std (s).
* **KAIROS temporal-layer timing** — `train.stats.json` `flow_timing`, each timer
  reported at its natural cadence as `<comp>_count` +
  `<comp>_total_us` (and `<comp>_mean_ms` derived by `run_stats.py`):
  * *Per frame* (compete for the keyframe budget): `process_frame` (whole
    per-frame temporal cost = ingest + coupling), `ingest` (folding the frame's
    detections; `ingest_total_us`, mean uses `process_frame_count`), `coupling`
    (the O(used²) edge-coupling pass).
  * *Per event* (run with the backend): `place_summaries` (flow → DSG
    aggregation), `remap_cells` (loop-closure remap), `run_maintenance`.
  * *Per query* (eval-time only): `score` (`scoreObservations`; per-call mean,
    and per-detection via `score_total_us / score_detections`).
  * *Supplementary micro-stat*: `add_observation` (µs per detection), `sharing`
    (µs per pooled neighbourhood, counted each time the weights are read, in
    training and in evaluation).
  Plus `memory` (`peak_rss_bytes`, `rss_bytes`, `vms_bytes`).
* **Memory-vs-time (opt-in)** — `mem_<scene>.csv` when the run is given
  `--mem-profile-csv`. Columns include `vmrss_kb` (process RSS over time), DSG
  node counts per layer (`l3_places_active`, ...), `mesh_vertices`, `flow_cells`,
  and the active-window voxel-layer footprint `tsdf_blocks`, `semantic_blocks`,
  `tsdf_bytes`, `semantic_bytes` (via the `map_memory_stats()` binding).

The key timers, grouped into Hydra tiers:

| tier | timer key | meaning |
|---|---|---|
| mid-level | `reconstruction/spin` | TSDF + mesh update inside the active window |
| mid-level | `places/overall_update` | GVD computation and sparsification into the places graph |
| mid-level | `frontend/objects_detection` | object extraction |
| mid-level | `frontend/spin` | **frontend total** (must meet the budget) |
| high-level | `backend/room_detection` | room clustering + classification |
| high-level | `backend/update_layers` | scene-graph backend optimization / layer update |
| high-level | `backend/spin` | **backend total** (async slow thread) |

---

## 3. Generate the data and the report (reproducible)

### 3a. Generate the runs

Run the TBD scenes with the dataset config and the flow state carried from one scene
to the next (stop-and-go): the **3DSG is rebuilt every
scene but the temporal map persists**.

```bash
# Training sessions in chronological order (the TRAIN_SCENES list of
# kairos-suite/evaluation/run_tbd.sh, paths relative to /data/tbd); each one
# resumes the previous session's flow state.
prev=""
for s in $TRAIN_SCENES; do
  name=$(basename "$s")
  mkdir -p /out/kairos_timing/train/$name
  kairos run /data/tbd/$s -c tbd -l ade20k_outdoor -o /out/kairos_timing/train \
      --tracks /data/tbd/$s/tracks.csv --no-publish --no-enable-lcd \
      --mem-profile-csv /out/kairos_timing/train/$name/mem_$name.csv \
      ${prev:+--load-flow-state $prev}
  prev=/out/kairos_timing/train/$name/temporal/flow_state.bin
done
```

`kairos run` writes each session under `<-o>/<session name>/`, and its flow state
to `<-o>/<session name>/temporal/flow_state.bin`.

The config keeps `enable_timing: true`. Loop-closure detection stays off on TBD. Because the temporal map is carried
over, read the temporal cost **in chronological session order** (see
`temporal_by_session.csv`).

### 3b. Run the report (one command)

`timing_report.py` is pure post-processing of the CSV/JSON above. It needs only a
standard-library Python; matplotlib is needed only for the figures.

```bash
# A whole run (every scene under the path is found and aggregated):
python3 kairos/python/src/kairos_python/timing_report.py \
    --results results/kairos_timing --budget-ms 200

# A single scene:
python3 .../timing_report.py --results results/kairos_timing/train/scene_tbd_2022-12-07-12-15-36

# Another dataset: point --results at its output path.
```

Options: `--budget-ms` (default 200 = 5 Hz keyframe budget), `--warmup-sec` (default 10;
drops the empty-map transient at the start of each scene), `--plot-scene <substring>`
(which scene to draw; default first), `--no-plot`, `--out <dir>`.

Outputs (in `<results>/timing_report/`):

* `timing_per_scene.csv` — every timer, every scene: n, mean, std, median, p95, max,
  fraction-over-budget, growth ratio (2nd-half mean / 1st-half mean).
* `timing_aggregate.csv` — each timer aggregated across scenes (scene-averaged
  mean ± between-scene std, pooled median / p95 / max, mean fraction-over-budget, mean
  growth).
* `timing_table.tex` — the per-tier LaTeX table (incl. the KAIROS temporal layer
  grouped by cadence).
* `temporal_by_session.csv` — per-session temporal cost in chronological order
  (cells, frames, per-frame temporal ms, ingest, coupling, per-detection evaluation).
* `trends_<scene>.png` — two panels (mid-level frontend; high-level backend), elapsed vs
  trajectory time, with the dashed real-time line.
* `memory_<scene>.png` — process RSS over time, with active L3-places overlaid.

**Reporting conventions** (the timing protocol of Hughes 2024):

* Report per-component **mean ± std** in ms, plus **median and p95** (the distributions
  are heavy-tailed; p95-vs-budget is the real-time statement).
* Report the **hardware** (CPU model and core count) and the data **cadence**.
* Drop a warm-up window.
* Read the **growth ratio** column to separate *bounded* components (≈1.0) from
  *accumulating* ones (>1).

---

## 4. Memory: the bounded part vs the persistent part

The **active-window map structures** (TSDF, semantic labels, dense GVD) are
**bounded** — flat as the explored area grows. The persistent scene graph grows with
explored area.

Two distinct memory numbers:

* **Process RSS** (`peak_rss_bytes`; `vmrss_kb` over time). This **grows** over a scene
  (e.g. ~230 MB → ~1.8 GB on the first TBD scene) because of the persistent graph + mesh
  + flow cells, and includes fixed overhead (the ~145 MB ORB vocabulary, DBoW, models).
  Report it as the whole-process footprint.
* **Active-window voxel layers** — logged via the `map_memory_stats()` binding into
  the mem CSV: `tsdf_blocks` / `semantic_blocks` (flat block counts mean the active
  window is bounded) and `tsdf_bytes` (exact:
  `blocks × voxels_per_side³ × sizeof(TsdfVoxel)`) / `semantic_bytes` (a lower bound — it
  counts the voxel containers but not each voxel's variable-length likelihood heap).
  `VolumetricMap::memoryStats()` computes these from the layer block counts
  (block count × struct size).
* **GVD bytes** are outside these numbers: the GVD/places layer is owned by a separate
  module with no size accessor. The flat `places/*` timing (growth ≈ 1.0) indicates the
  GVD is also bounded.

---

## 5. References

* Hughes, Chang, Hosseini, Rosinol, Carlone et al., *Foundations of spatial perception
  for robotics: Hierarchical representations and real-time systems*, IJRR 2024.
  Runtime evaluation §Runtime,
  Table "timing breakdown", Fig. "timing", and the active-window memory discussion.
* `hydra-kairos/eval/python/hydra_eval/timing.py` — Hydra's own timing helpers
  (frontend/backend collation, real-time threshold line); `timing_report.py` follows the
  same grouping and adds the KAIROS temporal layer and memory summary.
