# Temporal flow dynamics (kairos extension)

The temporal flow dynamics layer extends Hydra with per-voxel motion
modeling and periodic forecasting. While base hydra builds a 3D scene
graph from RGB-D and poses, this layer additionally ingests pedestrian
detections, fits a semi-wrapped Gaussian mixture and an NUDFT predictor
per active voxel, and writes flow attributes onto place nodes that the
visualizer renders as direction-and-speed arrows.

This document describes how to operate the layer: its inputs, its
configuration, its CLI and programmatic interfaces, and the architectural
boundaries to know about when extending it.

The layer implements a per-voxel
fixed-component semi-wrapped Gaussian mixture whose mixing weights are
forecast by spectral (NUDFT) predictors with per-cell order selection, a
Gamma-Poisson presence model with the dwell correction, predictive
uncertainty, evidence sharing across a voxel's traversable neighbourhood,
pairwise edge coupling, and the lifting of all of it onto the scene graph,
kept consistent under loop-closure corrections.

Unless another package is named, file paths are relative to the `kairos/` package.

---

## 1. What the layer produces

Each active voxel — defined as a TSDF voxel with weight at least
`min_tsdf_weight_for_visibility` — maintains a flow cell state with a
semi-wrapped Gaussian mixture over `(theta, rho)`, where `theta` is the
observed heading and `rho` the observed speed of incoming detections. The
mixture uses a fixed eight-cardinal-direction model whose mixing weights
evolve over time; each slot also keeps its own running mean speed. On top of
every mixing weight the layer fits an incremental NUDFT predictor over a
configurable set of candidate periods.
A Poisson model with a Gamma prior tracks per-voxel arrival rate and
yields a `p_present(t)` estimate.

After each backend update, voxel-level state is aggregated to scene-graph
place nodes by partitioning the flow voxels across places (each voxel
goes to its nearest place, capped by `place_support_max_distance_m`),
querying every supporting voxel's flow snapshot, and writing the result
to the node's metadata as the JSON fields `flow_components`,
`flow_dominant_theta`, `flow_dominant_rho`, `flow_confidence`,
`flow_probability`, `flow_support`, `flow_t_latest`,
and `flow_num_components`. Each component
in `flow_components` additionally carries `voxel_count` and `R`
(slot-level mean resultant length). Edge-flow scores (forward/reverse
alignment with the place-place edge direction, plus bidirectionality)
are written once per edge into the edge's own
`EdgeAttributes::metadata` under a `flow` key, oriented along
`edge.source -> edge.target`. See
[`place_aggregation.md`](https://github.com/IacopomC/hydra-kairos/blob/main/doc/place_aggregation.md) in `hydra-kairos` for a conceptual walk-through of the partition
choice and what each summary field means.

The visualizer reads exactly those fields. The arrow rendering is
implemented in
`kairos-ros/src/plugins/flow_place_arrow_plugin.cpp`
and `scene_graph_renderer.cpp` (`makeFlowArrowMarkers`); both publish to
`/hydra_dsg_visualizer/dsg_markers`, the same topic the static DSG uses.

---

## 2. Architecture

The dynamics module follows the same layering as the rest of hydra-kairos:
the non-ROS core lives in `kairos/`, and a thin ROS adapter lives in
`kairos-ros/`.

`kairos::FlowTemporalModule`
(`kairos/include/kairos/temporal/flow_temporal_module.{h,cpp}`) is a `Module`
that owns the worker thread, the `FlowMap`, the active-voxel set, the
deformation-alignment logic, and the place-summary writer. Detection observations enter via `pushDetectionFrame` (or
the convenience `pushDetections`, which takes world-frame xyz and
performs the voxel-grid lookup internally). Map updates from the active
window enter via `handleMapUpdate`, and backend updates via
`handleBackendUpdate`.

`kairos::FlowTemporalRosBridge`
(`kairos-ros/{include/kairos_ros,src}/flow_temporal_ros_bridge.{h,cpp}`)
composes a `FlowTemporalModule` and adds the ROS surface: a subscriber on
the configured `OdometryArray` topic, a `query_flow` service, marker and
metrics publishers, and a TF lookup. Both `HydraRosPipeline` (live ROS)
and `KairosPipeline` (offline) construct an instance of the module and
register `handleMapUpdate` / `handleBackendUpdate` as sinks on the active
window and backend respectively. The only difference between the two
paths is the entry point for detections: `OdometryArray` callback in ROS,
`pipeline.push_detections(...)` in Python.

---

## 3. Inputs

The frame stream — RGB, depth, labels, poses — is the base pipeline's.
Refer to [`non_ros_usage.md`](https://github.com/IacopomC/hydra-kairos/blob/main/doc/non_ros_usage.md) in `hydra-kairos` for the `FileDataLoader` contract
and to [`data_generation.md`](https://github.com/IacopomC/hydra-kairos/blob/main/doc/data_generation.md) for converting raw datasets into that layout.

Detections are an additional per-frame stream. The on-disk format is a
sorted CSV with one header row:

```
t_ns,ped_id,x_world,y_world,z_world,theta_rad,rho_mps
```

`t_ns` is the detection timestamp in nanoseconds, in the same epoch as
`poses.csv`. `x_world,y_world,z_world` is the position in the world frame
the poses live in (in meters). `theta_rad` is the heading angle wrapped
to `[-pi, pi]`, computed as `atan2(vy, vx)`. `rho_mps` is the speed in
m/s; rows with `rho <= 0` are dropped at ingestion. `ped_id` groups a
track's consecutive detections in one voxel into a single crossing, which
feeds the weights forecast one sample; rows without a track id each feed
their own sample. The
file must be sorted by `t_ns`; the runner streams it in order and drains
all rows with `t_ns <= frame_ts_ns` before each `pipeline.step(...)`.

Datasets that ship annotated trajectories (TBD) reduce to a mechanical
conversion. Other datasets need an upstream detect-and-track pipeline
(e.g. a person detector on RGB, a tracker maintaining IDs, a 2D-to-3D
lift via depth at the bounding-box foot), which lives outside this
repository.

---

## 4. Command-line workflow

The standard `kairos run` command serves both the
no-dynamics and with-dynamics paths. Adding `--tracks PATH` opts in to
the dynamics layer:

```bash
kairos run <scene_dir> \
    -c <config_name> -l <labelspace> \
    -m <max_frames> \
    -o <output_dir> \
    --tracks <scene_dir>/tracks.csv \
    --auto-align-tracks
```

The relevant flags:

- `--tracks PATH` enables detection ingestion. If omitted, behaviour is
  identical to base hydra — the flow module is still constructed when
  the YAML enables it, but its detection queue stays empty and place
  nodes acquire no `flow_*` attributes.
- `--auto-align-tracks` shifts every track timestamp so the first track
  aligns with the first frame in the dataloader. Useful when the CSV
  was produced with an approximate session epoch; off by default.
- `--tracks-time-shift-ns N` adds a manual offset (nanoseconds) to every
  track timestamp, applied on top of `--auto-align-tracks`.

The dataset YAML must set `enable_flow_temporal_module: true` for the
flag to do anything. The runner prints the track span, the offset
between the first track and first frame, and a final tally of pushed
samples on exit; a zero tally is the first thing to check when arrows
fail to appear.

On the live ROS path, `FlowTemporalRosBridge` subscribes to
`/people_detections_odom` (`odom_array_msgs::OdometryArray`); the launch
arguments live in
`hydra-kairos-ros/hydra_ros/launch/pipeline_config.launch` under the
`flow_temporal_*` prefix.

---

## 5. Configuration

The flow temporal module is configured via a `flow_temporal:` block at
the top level of the dataset YAML, plus an `enable_flow_temporal_module:`
toggle. `config/datasets/tbd.yaml` is the reference example; copy it as
a starting point for new datasets.

If `enable_flow_temporal_module` is missing or set to `false`, the
module is not constructed and `pipeline.push_detections(...)` becomes a
no-op returning `False`.

### 5.1 Module-level parameters

Defined in `kairos::FlowTemporalModule::Config`
(`kairos/include/kairos/temporal/flow_temporal_module.h`).

| Key | Default | Effect |
| --- | --- | --- |
| `publish_metrics` | `true` | (ROS-only) emit metrics on the configured topic. Set `false` for offline configs. |
| `publish_voxel_markers` | `true` | (ROS-only) publish CUBE_LIST markers for active and NUDFT-valid voxels. |
| `min_tsdf_weight_for_visibility` | `0.001` | TSDF weight threshold for a voxel to count as observable. Detections in voxels below threshold are ignored. |
| `worker_rate_hz` | `5.0` | How often the worker thread drains the detection queue and updates the per-voxel models. |
| `heavy_maintenance_rate_hz` | `0.0333` | How often deformation alignment and reliability checks run. |
| `max_queue_size` | `10` | Maximum buffered detection frames. Oldest are dropped on overflow. |
| `prediction_horizon_s` | `2.0` | Future horizon used for `p_present(t)` and place summaries. |
| `place_summary_max_nodes` | `2000` | Cap on place / mesh-place nodes summarized per backend update. |
| `max_components_per_place` | `3` | Maximum mixture components retained in `flow_components` metadata. |
| `component_min_weight` | `0.05` | Drop place components with normalized weight below this. |
| `component_min_voxel_count` | `0` | Drop place components backed by fewer than this many voxels. `0` disables the gate. |
| `place_support_max_distance_m` | `0.5` | Fallback admission radius (m) for assigning a flow voxel to its nearest place, used when the place's free-space distance is missing or non-positive. Normally a voxel belongs to its nearest place when it lies inside that place's free-space sphere. |
| `deformation_knn` | `4` | KNN used by the deformation graph point deformation. |
| `deformation_tolerance_s` | `60.0` | Temporal tolerance for matching deformation graph poses to flow cells. |

The three dataset configs set `prediction_horizon_s: 1.0`,
`max_components_per_place: 8`, `component_min_weight: 0.0` and
`deformation_tolerance_s: 10.0`.

The module embeds the full FlowMap state in the DSG itself: each
backend update writes `graph.metadata["flow"] = {voxel_size, map}`,
and `BackendModule::save` then serializes it as part of
`dsg_with_mesh.json`. The offline visualizer reads the same
`dsg_with_mesh.json` and renders the voxels; see the post-hoc voxel viz
section below.

### 5.2 `FlowMapConfig` (nested under `flow_temporal.flow`)

Defined in `kairos::FlowMapConfig` (`include/kairos/flow/config.h`).

Every key, with its default and effect, is listed in section 6 of
[`params.md`](params.md). The dataset-scale keys (`candidate_periods`,
`presence_rate_window_s`) and `nudft_order` / `nudft_min_obs` are set in each
dataset config.

### 5.3 Tuning notes

For smoke tests where arrows must appear within tens of seconds of
detection start, set `nudft_min_obs: 1` and `nudft_order: 1`. To capture daily and sub-daily structure
on long sessions, extend `candidate_periods` to include the relevant
scales (e.g. `[86400, 43200, 28800, 14400, 10800, 7200, 3600]`) and
keep `nudft_order` (the highest order the per-cell gate may select) above
the number of candidate periods, as the shipped configs do, so the gate
chooses the order freely. Profiling is supported via `enable_timing: true`, which produces
`[FlowTiming]` glog lines from the per-call sites in
`flow_map.cpp`, `cell_state.cpp`, and `flow_temporal_module.cpp`.

---

## 6. Visualization

The flow plugins run in the base pipeline's visualizer and read node
metadata directly, so any DSG carrying `flow_*` attributes renders
arrows automatically with the standard topics, displays, and RViz
configuration.

### Post-hoc voxel visualization

The same launch command renders the flow voxels embedded in the saved
DSG:

```bash
roslaunch hydra_visualizer hydra_visualizer.launch \
    scene_graph:=<out>/backend/dsg_with_mesh.json \
    color_mesh_by_label:=false
```

A small Python node (`flow_voxel_publisher.py`) reads
`graph.metadata.flow.map` from the DSG JSON and publishes the same
CUBE_LIST markers the live ROS bridge does (`flow_observed_voxels` in
blue, `flow_nudft_valid_voxels` in red) on
`/hydra_dsg_visualizer/flow_voxels`. Add a MarkerArray display in RViz
pointing at that topic if your saved config doesn't already include
one. Disable with `show_flow_voxels:=false`; stride/cap tunable via
`flow_voxel_stride` and `flow_voxel_max_cells`.

Each place node carrying a non-empty `flow_components` array gets one
arrow per retained component, anchored at the node's position. Direction
is taken from the component's `theta`; arrow length scales with
`weight * rho` clamped to a sensible range; colour is modulated by
`flow_confidence`.

There are two visualization paths. For live streaming, launch
`hydra_streaming_visualizer.launch` (or the `hydra visualize` shortcut)
in one terminal and run the pipeline in another with `--publish` (the
default). The visualizer subscribes to the ZMQ stream the pipeline emits
on `tcp://127.0.0.1:8001` and arrows appear on place nodes as soon as
the flow module has accumulated enough observations and one heavy
maintenance tick has fired (~30 seconds with default cadence). For
post-hoc replay, run the pipeline with `-o <out>` and load the saved DSG
into the file-based visualizer:

```bash
roslaunch hydra_visualizer hydra_visualizer.launch \
    scene_graph:=<out>/<scene_stem>/backend/dsg_with_mesh.json \
    color_mesh_by_label:=false
```

Always load `dsg_with_mesh.json`, not `dsg.json`: the arrow plugin skips
rendering when the mesh is absent.

---

## 7. Programmatic API

For driving the pipeline from a Python script, the binding exposes the relevant entry points alongside the standard `step`
/ `save` / `graph` from base hydra:

```python
from kairos_python._kairos_bindings import KairosPipeline

pipeline = KairosPipeline.from_file(
    "<path to YAML>", sensor, robot_id=0, zmq_url="tcp://127.0.0.1:8001"
)
pipeline.start()                                    # starts the flow worker

pipeline.step(timestamp_ns, t, q, depth, labels, rgb)

ok = pipeline.push_detections(
    timestamp_s,                                    # seconds (float)
    [(x1, y1, z1), (x2, y2, z2), ...],              # world-frame xyz
    [theta1, theta2, ...],                          # heading rad
    [rho1, rho2, ...],                              # speed m/s
)

pipeline.stop()
pipeline.save()
```

`push_detections` is thread-safe: the queue is drained by the worker at
`worker_rate_hz`. Calling more often than the worker rate buffers up to
`max_queue_size` frames; older frames are dropped on overflow. The call
returns `False` if the module is not enabled or if the voxel grid has
not yet been seeded by a `handleMapUpdate` (i.e. before the first
`pipeline.step` that produces an active TSDF).

Predicted flow at arbitrary world-frame positions is queried on the C++
side through `FlowTemporalModule::queryFlowAtPosition`; to call it from
Python, add a binding alongside `push_detections`.

---

## 8. Worked example: TBD pedestrian dataset

End-to-end recipe for a single TBD session. Mirror this structure for
new datasets that ship annotated trajectories.

```bash
# 1) RGB / depth / poses → FileDataLoader scene (base hydra converter)
python3 <kairos_ws>/src/hydra-kairos/scripts/tbd_to_filedataloader.py \
    --data3d     <tbd_root>/Robot_data/2022-12/cabot_2022-12-07-12-15-36_data3d \
    --trajectory <tbd_root>/Robot_data/2022-12/cabot_2022-12-07-12-15-36-trajectory.yaml \
    --output     ~/Downloads/scene_tbd_2022-12-07-12-15-36

# 2) Semantic labels (replaces the all-zero placeholders the converter wrote)
python3 <kairos_ws>/src/hydra-kairos/scripts/generate_labels.py \
    ~/Downloads/scene_tbd_2022-12-07-12-15-36 --overwrite

# 3) Pedestrian labels (per-segment, timestamped) → tracks.csv
python3 <kairos_ws>/src/hydra-kairos/scripts/tbd_pedestrian_to_detections.py \
    --ped-labels-dir    <tbd_root>/Robot_data/ped_labels_w_time \
    --sync-file         <tbd_root>/Robot_data/zed_sync_times.txt \
    --cabot-time-file   <tbd_root>/Robot_data/2022-12/cabot_2022-12-07-12-15-36time.txt \
    --useful-times-file <tbd_root>/Robot_data/svo_useful_times.txt \
    --scene-dir         ~/Downloads/scene_tbd_2022-12-07-12-15-36 \
    --output            ~/Downloads/scene_tbd_2022-12-07-12-15-36/tracks.csv

# 4) Run the pipeline on the scene with its tracks
kairos run ~/Downloads/scene_tbd_2022-12-07-12-15-36 \
    -c tbd -l ade20k_outdoor \
    --tracks ~/Downloads/scene_tbd_2022-12-07-12-15-36/tracks.csv \
    -o ~/Downloads/tbd_dynamics --no-publish

# 5) Visualize (live streaming or post-hoc replay — see §6)
```

`tbd_pedestrian_to_detections.py` reads the timestamped per-segment label
files `<sess>_<seg>_t.txt`, keeps the session selected by `--sync-file` and the
intervals in `--useful-times-file`, moves the positions from the overhead-camera
frame into the robot's map frame, and computes heading and speed by finite
differences with a small smoothing window. The TBD trajectories are 2D
(annotated from overhead cameras), so `z_world` is set to `--z-floor`
(default `0.0`). Timestamps come from `--cabot-time-file`, the same wall clock
as the scene's frames, so the tracks line up with the scene as written.

---

## 9. Extending

Adding a new dataset requires producing a `FileDataLoader` scene
(see [`data_generation.md`](https://github.com/IacopomC/hydra-kairos/blob/main/doc/data_generation.md) in `hydra-kairos`), producing a `tracks.csv` from whatever
detection or annotation source the dataset provides, and adding a
`<dataset>.yaml` under `config/datasets/` with
`enable_flow_temporal_module: true` plus a `flow_temporal:` block tuned
for the sensor cadence and observation density.

A new in-process detection source — for instance, a live detector
running in the same Python interpreter as the pipeline — calls
`pipeline.push_detections(...)` directly with the same signature; voxel-grid lookup happens inside the binding
from world-frame coordinates.

Adding new place-node attributes requires aligned changes on both sides
of the `dsg_markers` topic. Append the new field to the metadata write
in `updatePlaceSummaries` (`src/temporal/flow_temporal_module.cpp`) and
add a matching read in the visualizer
(`hydra-kairos-ros/hydra_visualizer/src/scene_graph_renderer.cpp`,
`makeFlowArrowMarkers` or a sibling). There is no schema validation, so
naming mismatches silently render nothing.

---

## 10. Verifying behaviour

The runner prints a single-line tally at exit:

```
[tracks] pushed N samples across M frame-times
```

`N == 0` is the first thing to check when arrows fail to appear; the
two usual causes are (a) tracks and frames not sharing an epoch — visible
in the startup `offset (frame - track) = …s` line, and resolved by
regenerating `tracks.csv` with `--scene-dir` or by passing
`--auto-align-tracks` for a smoke-test alignment — and (b) the YAML not
enabling the module, in which case `push_detections` returns `False`
silently.

When `N > 0` but no arrows render, check that the detection positions
fall in the same world frame as `poses.csv`. The simplest sanity check
is to overlay the first few CSV positions against the reconstruction
mesh in RViz; misaligned frames produce detections in voxels that no
place node supports, so no place ever accumulates flow. Other readily
checkable causes: `nudft_min_obs` higher than the observations any
single voxel has accumulated (drop to `1` for smoke tests);
`min_tsdf_weight_for_visibility` higher than the typical TSDF weight at
the ages of voxel pedestrians cross (the threshold filters out
freshly-integrated voxels, so very short runs are most affected); the
visualizer loaded `dsg.json` instead of `dsg_with_mesh.json` (the arrow
plugin skips rendering without a mesh).

When arrows render but go stale after a loop closure, confirm that
`deformation_tolerance_s`, the time window for selecting deformation
control points (60 s by default, 10 s in the dataset configs), covers the
time between a cell's first observation and the nearest control point.

Profiling is supported via `enable_timing: true` under
`flow_temporal.flow`, which logs per-call durations from
`addObservation`, `processFrame`, `updatePlaceSummaries`,
`runDeformationAlignment`, and `runMaintenance`. If the worker is
saturating, either increase `worker_rate_hz` (catches up faster, more
CPU) or rate-limit detections upstream.

---

## 11. Reference

Code organization, in dependency order:

- `include/kairos/flow/`, `src/flow/` — non-ROS primitives. `FlowMap`
  manages per-voxel state (`flow_map.h`). `FlowCellState`
  (`cell_state.h`) wraps the SWGMM and NUDFT predictors per voxel.
  `swgmm.h` defines `FixedComponents`. `nudft.h`
  defines the incremental Fourier predictor. `poisson_model.h` defines
  the presence model. `config.h` defines `FlowMapConfig`.
- `kairos/include/kairos/temporal/flow_temporal_module.h`,
  `src/temporal/flow_temporal_module.cpp` — the orchestrating module.
  Owns the worker thread, ingestion queue, deformation alignment,
  place-summary writer, and the public ingestion entry points
  (`pushDetectionFrame`, `pushDetections`, `queryFlowAtPosition`).
- `kairos/python/bindings/src/kairos_pipeline.cpp` — `KairosPipeline`
  (subclass of Hydra's base pipeline) wires in `FlowTemporalModule` and binds
  `start`, `stop`, `push_detections`, evaluation, flow-state I/O.
- `kairos/python/src/kairos_python/commands/run.py` — `kairos run` CLI entry,
  including `--tracks`, `--auto-align-tracks`, and `--tracks-time-shift-ns`.
- `kairos/config/datasets/tbd.yaml` — reference YAML enabling the module.
- `kairos_ros/{include/kairos_ros,src}/flow_temporal_ros_bridge.{h,cpp}`
  — ROS adapter (`OdometryArray` decode, metrics and voxel-marker
  publishers, `query_flow` service).
- `src/plugins/flow_place_arrow_plugin.cpp` in the `kairos-ros` repo — arrow-marker
  construction (rviz plugin).
