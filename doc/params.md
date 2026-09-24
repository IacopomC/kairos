# FlowTemporalModule & FlowMap Parameters

This document covers the YAML parameters for the temporal flow stack:
the `FlowTemporalModule` (worker thread, deformation alignment, place
summaries) and the `FlowMap` (per-cell statistical model).

The relevant YAML block lives at `flow_temporal:` inside a dataset config
(see `config/datasets/tbd.yaml`).

```
flow_temporal:        # → FlowTemporalModule::Config
  ...module-level fields...
  flow:               # → FlowMapConfig
    ...flow-map fields...
```

The C++ structs are at:

- `include/kairos/temporal/flow_temporal_module.h` — `FlowTemporalModule::Config`
- `include/kairos/flow/config.h` — `FlowMapConfig`

The yaml↔struct binding is in
`declare_config` in `src/temporal/flow_temporal_module.cpp` and
`src/flow/config.cpp` (FlowMap fields).

---

## 1. Worker / scheduling

These control the threading model. The module spawns one worker thread on
`start()` (`src/temporal/flow_temporal_module.cpp`) that pulls detection
frames and runs deformation alignment.

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `worker_rate_hz` | `float` | `5.0` | Wakeup frequency of the worker thread. Each tick: drain `frame_queue_`, run `processFrame`, optionally run heavy maintenance. |
| `heavy_maintenance_rate_hz` | `float` | `1/30` | Rate at which the worker runs `runDeformationAlignment` + `flow_map_.runMaintenance`. Set well below `worker_rate_hz`; the alignment is the expensive part. |
| `max_queue_size` | `int` | `10` | Soft cap on the detection frame queue. Frames beyond this drop the oldest. |
| `prediction_horizon_s` | `float` | `2.0` | How far ahead `queryFlowAtPosition` predicts by default (the per-call `horizon_s` argument overrides this). |

**Threading invariant.** `handleMapUpdate` and `handleBackendUpdate` are
called by the active-window / backend threads (under each module's own
mutex). `pushDetectionFrame`, `pushDetections`, and `setTickCallback` are
called from the user thread (or ROS spinner). The worker thread reads from
the frame queue and the backend dgraph snapshot. Mutations to internal
state are guarded by `mutex_` (frame queue, voxel grid, dgraph pointer/
snapshot) and `summaries_mutex_` (place summaries + counters).

---

## 2. Deformation alignment

These control how the FlowMap's voxel cells get re-positioned after a loop
closure / pose-graph optimisation. Alignment always runs on a backend update.

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `deformation_knn` | `int` | `4` | Number of nearest control points used by `kimera_pgmo::deformation::interpPoint` to interpolate each cell's new position. Higher = smoother, more averaging; lower = more local. |
| `deformation_tolerance_s` | `double` | `60.0` (yaml: `10.0`) | Time window (s) used by `deformPoints` to bracket eligible control points around each cell's stamp. Bigger = considers more control points, including temporally-distant ones; smaller = strict locality. |

**Cost.** `runDeformationAlignment` iterates over every flow cell with an
original position. For each cell: octree NN query + `deformation_knn`
weighted averages. Per-cell cost is logged at INFO level
(`[FlowTiming] runDeformationAlignment ...`).

**Trigger.** The worker checks `dgraph_updated_` once per
`heavy_maintenance_rate_hz` tick. The flag is set by `handleBackendUpdate`,
which is registered as a `pre_update_sink` on the backend
(`python/bindings/src/kairos_pipeline.cpp`). So one alignment runs
per backend update cycle (post-optimization), throttled to the maintenance
rate.

**Threading note.** `handleBackendUpdate` is called on the backend thread
inside the backend's mutex (`hydra-kairos/src/backend/backend_module.cpp` from
`callUpdateFunctions`, which is invoked from `optimize()` after
`deformation_graph_->optimize()`). This is the synchronization point at
which the dgraph is consistent. The worker thread reads the dgraph
asynchronously and must therefore use a snapshot (or share the backend
mutex) to avoid racing with the next backend update.

**TBD-style hall.** The defaults suit it. If you see "not enough valid control
points" warnings, raise `deformation_tolerance_s`. If you see the
deformation warping cells across the hall (cross-room collisions), lower
it.

---

## 3. Visibility / cell admission

Controls which TSDF voxels get tracked as flow cells.

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `min_tsdf_weight_for_visibility` | `float` | `1e-3` | Minimum TSDF integration weight for a voxel to be considered "active". Below this the voxel is skipped on `handleMapUpdate`. Raise to filter noisy near-empty voxels; lower to track sparse regions. |

The active set is rebuilt each `handleMapUpdate` from
`map.getTsdfLayer().allocatedBlockIndices()`.

---

## 4. Place summaries (PlaceFlowSummary)

These build per-place flow summaries from the cell-level mixture model.
They feed `placeSummariesCopy()` and the `archetypes` backend functor.

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `place_summary_max_nodes` | `int` | `2000` | Hard cap on places summarised per pass. Older / lower-priority places drop out when exceeded. |
| `max_components_per_place` | `int` | `3` (yaml: `8`) | Top-K mixture components retained per place summary. |
| `component_min_weight` | `float` | `0.05` (yaml: `0.0`) | Components below this mixture weight are pruned from the summary. |
| `component_min_voxel_count` | `int` | `0` | Components with fewer cells than this are pruned. |
| `place_support_max_distance_m` | `float` | `0.5` | Fallback admission distance for "voxel belongs to place" when the place's GVD distance is missing/non-positive. The primary criterion is `dist(voxel, place) <= place.distance` (see `place_support_max_distance_m` in `flow_temporal_module.h`). |

---

## 5. Visualization / publishing

The ROS adapter reads these; the offline (python) pipeline ignores most of
them, and the dataset configs set the publishing flags to `false`.

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `publish_metrics` | `bool` | `true` (yaml: `false`) | Whether the ROS adapter publishes per-tick metrics. |
| `publish_voxel_markers` | `bool` | `true` (yaml: `false`) | Whether the ROS adapter publishes voxel marker arrays. |
| `voxel_markers_stride` | `int` | `1` | Take every Nth cell when publishing markers. Keeps RViz responsive on dense maps. |
| `voxel_markers_max_cells` | `int` | `50000` | Hard cap on cells per marker message. |
| `debug_metrics_nodes_only` | `bool` | `false` | Limit debug metric output to a small node subset. |
| `debug_metrics_max_nodes` | `int` | `10` | Cap when `debug_metrics_nodes_only` is on. |

---

## 6. FlowMap (`flow:` sub-block)

Per-cell statistical model. Defaults are the C++ header values; the shipped
dataset configs set the dataset-scale ones explicitly.

### 6.1 Mixture model (fixed K=8 cardinal slots)

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `fixed_mu_rho` | `double` | `1.1` | Speed (m/s) a slot reports before any speed has been observed anywhere in the map. |
| `fixed_sigma_theta` | `double` | `0.4` | Angular σ (rad) of every slot. |
| `fixed_sigma_rho` | `double` | `0.3` | Radial σ (m/s) of every slot; also the prior spread of the speed uncertainty. |
| `fixed_sigma_cross` | `double` | `0.0` | θ-ρ cross covariance. |

### 6.2 Spectral predictors and order gate

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `candidate_periods` | `vector<float>` | weekly→hourly grid (yaml: per dataset) | Candidate periods (s) the spectral predictors fit. Set to the dataset's recurrence scale. |
| `nudft_order` | `int` | `3` (yaml: `10`) | Highest spectral order the per-cell prequential gate may select for a mixing-weight or presence predictor. |
| `nudft_min_obs` | `int` | `20` (yaml: `10`) | Samples a predictor needs before it is trusted; below this it reports its mean term. |

### 6.3 Mixing weights (one sample per crossing)

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `visit_gap_s` | `double` | `2.0` | A track's open visit to a cell closes after this much silence. |
| `visit_max_s` | `double` | `10.0` | Hard cap on a visit's span; a lingering agent opens a new visit. |

Each closed visit feeds one weights sample (its mean responsibility);
detections without a track id feed one sample each.

### 6.4 Per-slot speed

Each slot's speed is the running mean of the speeds routed to it; until the slot has `nudft_min_obs` observations it reports the
running mean of speed over the whole map.

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `speed_scatter_prior_obs` | `double` | `5.0` | Weight, in observations, of `fixed_sigma_rho` in the speed uncertainty, which blends it with the slot's measured speed scatter. |

### 6.5 Presence

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `presence_rate_window_s` | `double` | `120.0` (yaml: per dataset) | Exposure window that feeds the presence predictor one windowed-rate sample. Must sit well below the shortest candidate period. |
| `voxel_size_m` | `double` | `0.4` | Flow voxel side (m); sets the dwell time for the presence dwell correction. Must match the reconstruction voxel size. |

### 6.6 Evidence sharing

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `sharing_lent_crossings` | `double` | `3.0` | Crossings of neighbourhood evidence lent to a voxel's mixing weights. The neighbourhood holds a share m/(C+m) of the reported weight. |

### 6.7 Predictive uncertainty

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `weight_uncertainty_noise_sigma` | `double` | `0.3` | Observation-noise σ assumed by the mixing-weight predictors' posterior; sets how fast the weight variance shrinks. |

### 6.8 Edge coupling

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `edge_coupling_connectivity` | `int` | `6` | Neighbourhood of a coupled pair: 6 = face-adjacent, 26 = full cube. |
| `edge_coupling_delta_t_coh` | `double` | `1.0` | Two detections within this many seconds form a co-occurring pair. |
| `edge_coupling_samples_per_period` | `int` | `12` | Log-PMI samples per cycle of the shortest candidate period; sets the sampling window. |
| `edge_coupling_window_min_pairs` | `int` | `2` | Pairs a sampling window must hold before it may close. |

### 6.9 Diagnostics

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `stability_window` | `int` | `5` | Consecutive observations with an unchanged dominant slot after which a cell counts as stable. |
| `enable_timing` | `bool` | `false` (yaml: `true`) | Logs `[FlowTiming] ...` per major phase. |

---

## 7. How a cell flows through the pipeline

1. **Admission**: `handleMapUpdate` walks TSDF voxels, admits those above
   `min_tsdf_weight_for_visibility` into `active_voxels_`.
2. **Detection ingest**: ROS / python pushes a `DetectionFrame` (per-track
   θ, ρ at voxel indices); the worker pops one per tick and routes
   observations to the flow cells.
3. **Cell update**: `FlowMap::addObservation` updates the per-cell
   mixture. Cells stay at the fixed K=8 cardinal-direction geometry; only
   the π predictors (fed once per crossing), the per-slot speed means and the
   presence model accumulate state.
4. **Place rollup**: `updatePlaceSummaries` aggregates cells under each
   place node into a `PlaceFlowSummary` (top-`max_components_per_place`,
   threshold by `component_min_weight` / `component_min_voxel_count`).
5. **Backend update**: when the backend finishes an optimize pass, it calls
   `handleBackendUpdate`. The worker takes a dgraph snapshot and, on the
   next heavy-maintenance tick, runs `runDeformationAlignment` to remap
   cells to their corrected world positions.

---

## 8. Tuning recipes

### Make alignment less aggressive (debug)

- `deformation_knn: 1` and `deformation_tolerance_s: 5.0` — minimal
  smoothing, strict temporal locality. Useful for diagnosing whether
  cross-cell merges come from over-aggressive interpolation.

### Faster flow convergence (debug runs)

- `nudft_min_obs: 1` — trust every predictor from its first sample.

### Reduce place summary churn

- Raise `max_components_per_place` to 5–8, but raise `component_min_weight`
  to 0.05–0.10 to prune noise components.
- Lower `place_summary_max_nodes` if the summary loop is hot.

### Single-hall (TBD)

The defaults plus the `tbd.yaml` overrides suit a single hall. If you lower
`nudft_min_obs` to 1 for a debug run, every cell publishes a spectral
estimate from a single sample; keep it at 5–10 otherwise.
