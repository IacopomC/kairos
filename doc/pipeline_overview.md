# KAIROS flow pipeline overview

What the flow layer in this repo does, step by step, mapped to the source files
that implement it.

Scope: the **flow layer** in this repo (`kairos`). The 3DSG it builds on
(TSDF integration, GVD places, mesh, pose-graph backend) lives in `hydra` and is
documented separately.

> **API reference.** The headers under `include/kairos/` carry Doxygen comments.
> Run `doxygen Doxyfile` from the repo root to generate a browsable reference at
> `doc/api/html/index.html`.

---

## Pipeline at a glance

\dot KAIROS flow layer: how a detection frame updates the dynamics map and is evaluated
digraph kairos_flow {
  rankdir=TB;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=10];
  edge [fontname="Helvetica", fontsize=9];

  map [label="map update (TSDF)"];
  det [label="detections (θ, ρ)"];

  subgraph cluster_p1 {
    label="Phase 1: per-voxel temporal modelling";
    style=dashed; color="#9bb7d4";
    ingest [label="handleMapUpdate\n(active voxel set)"];
    buffer [label="pushDetections\n(buffer frames)"];
    worker [label="processFrame\n(cell: mixture, predictors, presence)"];
    deform [label="runDeformationAlignment\n(re-key cells after backend opt)"];
  }
  subgraph cluster_p2 {
    label="Phase 2: aggregation into the 3DSG";
    style=dashed; color="#9bb7d4";
    summ  [label="updatePlaceSummaries"];
    edges [label="annotateFlowEdges"];
  }
  arch  [label="archetype layer\n(built in hydra backend)", style="rounded,dashed", fillcolor="#ffffff"];
  score [label="scoreObservations /\nqueryFlowAtPosition"];
  dsg   [label="scene graph\n(flow metadata)", shape=note, fillcolor="#fff7e6"];

  map -> ingest;
  det -> buffer;
  ingest -> worker;
  buffer -> worker;
  worker -> deform;
  worker -> summ;
  deform -> summ;
  summ -> edges;
  edges -> dsg;
  summ -> arch [style=dashed];
  worker -> score [label="held-out detections"];
}
\enddot

Phase 1 models each voxel's dynamics from streamed detections; phase 2 aggregates
that into place summaries and edge-flow attributes on the scene graph. The
archetype layer is built downstream, in the `hydra` backend. Evaluation reads
a cell snapshot at each held-out detection's own timestamp.

---

## Code layout

**Orchestrator** (`src/temporal/flow_temporal_module.cpp`, `FlowTemporalModule`).
A scene-graph module. It receives map/backend updates and per-frame detections,
drives per-frame processing on a worker thread, runs periodic maintenance, and
writes results back into the scene graph. This file is control flow only.

**Algorithm pieces** (`src/flow/`):
- `flow_map.{h,cpp}`: the collection of per-voxel **cells** (the dynamics map).
- `cell_state.{h,cpp}`: one voxel's state, namely its mixture, predictors, and presence model.
- `swgmm.cpp` / `swnd.cpp`: the semi-wrapped Gaussian **mixture** and its density.
- `nudft.cpp`: incremental spectral (NUDFT) **predictor** for a time-varying scalar.
- `poisson_model.cpp`: the Gamma/Poisson **presence** model.
- `edge_coupling_map.cpp`: **pairwise coupling** between adjacent voxels.
- `config.cpp`: configuration.

**Entry point** (`python/bindings/src/kairos_pipeline.cpp`). It wraps the
scene-graph pipeline together with this module; `kairos_python run` drives it
frame by frame.

---

## How a frame flows (execution order)

### Phase 1: per-voxel temporal modelling
1. **Ingest geometry.** `handleMapUpdate()` rebuilds the active-voxel set from
   the input map's TSDF blocks (keeping voxels above a visibility weight) and
   refreshes the voxel-index grid. *(flow_temporal_module.cpp)*
2. **Ingest detections.** `pushDetections()` / `pushDetectionFrame()` map each
   detection to a voxel and buffer the per-frame `(θ, ρ)` observations on a
   bounded queue. *(flow_temporal_module.cpp)*
3. **Worker loop.** `workerSpin()` drains queued detection frames, calls
   `processFrame`, and triggers periodic maintenance (including deformation
   alignment). *(flow_temporal_module.cpp)*
4. **Per-frame update.** `processFrame()` filters detections to the active voxel
   set and updates each touched voxel's cell: its mixture, its spectral
   predictors, and its presence model. It also updates the edge-coupling map
   for adjacent observed voxels. A track's detections in one cell are grouped
   into a single crossing, which feeds the weights forecast one sample.
   Evidence sharing is applied when the weights are read (see `design-choices.md` §4). *(flow_temporal_module.cpp
   → flow_map → cell_state → swgmm / nudft / poisson_model; edge_coupling_map.cpp)*
5. **Deformation alignment.** `handleBackendUpdate()` snapshots the optimized
   deformation graph. `runDeformationAlignment()`, run from the worker loop,
   re-keys cells to the corrected positions, statistically merging cells that
   collide on a target voxel, so accumulated statistics follow the corrected
   map. Each run is recorded as a loop-closure event. *(flow_temporal_module.cpp)*

### Phase 2: aggregation into the 3DSG
6. **Place summaries.** `updatePlaceSummaries()` assigns flow voxels to their
   nearest place (within its free-space radius) and aggregates the per-voxel cell
   state into place-node-level flow summaries embedded in the place metadata.
   *(flow_temporal_module.cpp → flow_map)*
7. **Edge flow annotation.** `annotateFlowEdges()` reads the two endpoints' place
   summaries and writes flow attributes (forward/reverse alignment, speed,
   bidirectionality) onto the scene-graph edge metadata. *(flow_temporal_module.cpp)*
8. **Archetype layer.** Built downstream of this module, which summarises each
   place independently. The archetype node/attribute types are defined in
   the `spark-dsg` fork, and archetypes are constructed in the `hydra` backend
   (a dedicated `update_archetypes_functor`). Documented with those repos.

### Inference / evaluation
9. **Evaluation.** `scoreObservations()` returns `std::vector<ObservationScore>` and
   does trajectory-anchored evaluation of a batch of held-out detections. For each
   `(t, xyz, θ, ρ)` it looks up the voxel, takes a cell snapshot **at that
   detection's own timestamp**, then evaluates the predicted mixture density at
   `(θ, ρ)` (plus the per-channel marginals and presence). *(flow_temporal_module.cpp
   → cell_state → swgmm / swnd / nudft / poisson_model)*. The single-point
   `queryFlowAtPosition()` (returning `QueryResult`) is the live-query counterpart
   for one `(xyz, t)`.

### Persistence
`saveFlowState()` / `loadFlowState()` serialize and restore the full per-cell
state, used for train-once / evaluate-many. *(flow_temporal_module.cpp)*
