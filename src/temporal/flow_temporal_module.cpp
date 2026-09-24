/**
 * @file flow_temporal_module.cpp
 * @brief Orchestrator of the flow pipeline: owns the per-voxel dynamics map and
 *        drives per-frame updates, maintenance, and scene-graph aggregation.
 *
 * Execution order of a frame (see doc/pipeline_overview.md for the walkthrough):
 *   Phase 1 (per-voxel temporal modelling)
 *     handleMapUpdate()       ingest TSDF active-voxel set + mesh
 *     pushDetections()        buffer per-frame pedestrian (theta, rho) obs
 *     workerSpin()            worker loop: drain frames + periodic maintenance
 *     processFrame()          filter dets to active voxels; update each cell
 *     handleBackendUpdate() -> runDeformationAlignment()  re-key cells on correction
 *   Phase 2 (aggregation into the 3DSG)
 *     updatePlaceSummaries()  per-voxel state -> place-node flow summaries
 *     annotateFlowEdges()     write flow attributes onto scene-graph edges
 *   Inference:   scoreObservations()  query a voxel's predicted mixture at (x,y,t)
 *   Persistence: saveFlowState() / loadFlowState()
 *
 * Algorithm pieces live in src/flow/ (flow_map, cell_state, swgmm/swnd, nudft,
 * poisson_model, edge_coupling_map); this file is control flow only.
 */
#include "kairos/temporal/flow_temporal_module.h"

#include <algorithm>
#include <chrono>
#include <mutex>

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <kimera_pgmo/deformation_graph.h>
#include <kimera_pgmo/mesh_deformation.h>
#include <kimera_pgmo/pcl_mesh_traits.h>
#include <kimera_pgmo/utils/common_structs.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spark_dsg/node_attributes.h>

#include "kairos/flow/poisson_model.h"
#include "kairos/flow/swnd.h"
#include "hydra/places/nearest_voxel_utilities.h"
#include "hydra/utils/log_utilities.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

double wallClockSeconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// Definition matches the forward-declaration in flow_temporal_module.h. The
// three vector/Values members hold an immutable copy of the slice
// runDeformationAlignment reads from kimera_pgmo::DeformationGraph; the
// gtsam::Values copy is a deep clone (each Value is virtually cloned), so
// the snapshot is self-contained and safe to read from the worker thread
// after the backend mutex has been released.
struct DgraphSnapshot {
  uint64_t timestamp_ns = 0;
  char prefix = 0;
  std::vector<gtsam::Point3> control_points;
  std::vector<kimera_pgmo::Timestamp> vertex_stamps;
  gtsam::Values gtsam_values;
};

double FlowTemporalModule::PlaceFlowSummary::presence(double delta_t) const {
  return presenceFromOccupancy(occupancy, dwell_s, delta_t);
}

void declare_config(FlowTemporalModule::Config& config) {
  using namespace config;
  name("FlowTemporalModule::Config");
  field(config.publish_metrics, "publish_metrics");
  field(config.min_tsdf_weight_for_visibility, "min_tsdf_weight_for_visibility");
  check(config.min_tsdf_weight_for_visibility, GE, 0.0f, "min_tsdf_weight_for_visibility");
  field(config.worker_rate_hz, "worker_rate_hz");
  check(config.worker_rate_hz, GT, 0.0f, "worker_rate_hz");
  field(config.heavy_maintenance_rate_hz, "heavy_maintenance_rate_hz");
  check(config.heavy_maintenance_rate_hz, GT, 0.0f, "heavy_maintenance_rate_hz");
  field(config.max_queue_size, "max_queue_size");
  check(config.max_queue_size, GT, 0, "max_queue_size");
  field(config.prediction_horizon_s, "prediction_horizon_s");
  check(config.prediction_horizon_s, GT, 0.0f, "prediction_horizon_s");
  field(config.place_summary_max_nodes, "place_summary_max_nodes");
  check(config.place_summary_max_nodes, GE, 0, "place_summary_max_nodes");
  field(config.max_components_per_place, "max_components_per_place");
  check(config.max_components_per_place, GE, 1, "max_components_per_place");
  field(config.component_min_weight, "component_min_weight");
  check(config.component_min_weight, GE, 0.0f, "component_min_weight");
  check(config.component_min_weight, LE, 1.0f, "component_min_weight");
  field(config.component_min_voxel_count, "component_min_voxel_count");
  check(config.component_min_voxel_count, GE, 0, "component_min_voxel_count");
  field(config.place_support_max_distance_m, "place_support_max_distance_m");
  check(config.place_support_max_distance_m, GT, 0.0f, "place_support_max_distance_m");
  field(config.deformation_knn, "deformation_knn");
  check(config.deformation_knn, GE, 1, "deformation_knn");
  field(config.deformation_tolerance_s, "deformation_tolerance_s");
  check(config.deformation_tolerance_s, GT, 0.0, "deformation_tolerance_s");
  field(config.debug_metrics_nodes_only, "debug_metrics_nodes_only");
  field(config.debug_metrics_max_nodes, "debug_metrics_max_nodes");
  check(config.debug_metrics_max_nodes, GE, 1, "debug_metrics_max_nodes");
  field(config.publish_voxel_markers, "publish_voxel_markers");
  field(config.voxel_markers_stride, "voxel_markers_stride");
  check(config.voxel_markers_stride, GE, 1, "voxel_markers_stride");
  field(config.voxel_markers_max_cells, "voxel_markers_max_cells");
  check(config.voxel_markers_max_cells, GE, 0, "voxel_markers_max_cells");
  field(config.flow, "flow");
}

FlowTemporalModule::FlowTemporalModule(const Config& config)
    : config(config::checkValid(config)), flow_map_(config.flow) {
  // share the frequency vector so edge-coupling NUDFTs match cell NUDFTs
  const auto shared_omegas = flow_map_.sharedOmegas();
  // Derive the phi sampling window from the shortest rhythm being fitted, so
  // the rate is expressed against candidate_periods rather than restated as
  // a duration each time those periods change. The config checks guarantee a
  // non-empty period set and at least two samples per period.
  const double shortest_period_s = static_cast<double>(*std::min_element(
      config.flow.candidate_periods.begin(), config.flow.candidate_periods.end()));
  const double phi_window_s =
      shortest_period_s / config.flow.edge_coupling_samples_per_period;
  edge_coupling_map_ = std::make_unique<EdgeCouplingMap>(
      shared_omegas,
      config.flow.nudft_min_obs,
      phi_window_s,
      config.flow.edge_coupling_window_min_pairs);
  LOG(INFO) << "[FlowTemporal] edge coupling ("
            << config.flow.edge_coupling_connectivity << "-connected, Δt_coh="
            << config.flow.edge_coupling_delta_t_coh << "s, phi sampled "
            << config.flow.edge_coupling_samples_per_period
            << "x per shortest period -> window=" << phi_window_s
            << "s, min pairs=" << config.flow.edge_coupling_window_min_pairs << ")";
}

FlowTemporalModule::~FlowTemporalModule() { stop(); }

void FlowTemporalModule::start() {
  if (started_) {
    return;
  }

  should_shutdown_ = false;
  started_ = true;

  worker_thread_ = std::make_unique<std::thread>(&FlowTemporalModule::workerSpin, this);
}

void FlowTemporalModule::stop() {
  if (!started_) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    should_shutdown_ = true;
  }
  cv_.notify_all();

  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }
  worker_thread_.reset();

  started_ = false;
}

void FlowTemporalModule::save(const LogSetup& setup) {
  // Persist the full per-cell flow state alongside the DSG so reload-and-
  // score works across process restarts. The DSG JSON only carries the
  // per-place flow summary (a snapshot at save time); the live NUDFT /
  // Poisson state is what this binary carries and what is needed for
  // time-anchored scoring at arbitrary t.
  const std::string log_dir = setup.getLogDir("temporal");
  if (log_dir.empty()) {
    LOG(WARNING) << "[FlowTemporal] no temporal log dir; skipping flow_state save";
    return;
  }
  flow_map_.flushVisits();  // feed open per-track visits so no vote is lost
  const std::string path = log_dir + "/flow_state.bin";
  std::string err;
  if (!saveFlowState(path, &err)) {
    LOG(WARNING) << "[FlowTemporal] flow_state save failed: " << err;
  } else {
    LOG(INFO) << "[FlowTemporal] flow_state saved to " << path;
  }
}

bool FlowTemporalModule::saveFlowState(const std::string& path,
                                       std::string* error) const {
  nlohmann::json record;
  record["schema"] = "kairos_flow_state";
  // Schema v2 added edge-coupling. v3 adds explicit grid_origin +
  // world_frame_id to the header; both are asserted to match on load.
  // grid_origin is currently always (0,0,0) — spatial_hash::IndexGrid is
  // origin-less and indexes off world coords directly — but recording it
  // explicitly future-proofs the format against a deliberate world-origin
  // shift.
  record["version"] = 3;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    record["voxel_size"] = voxel_size_;
  }
  record["grid_origin"]    = std::array<double, 3>{0.0, 0.0, 0.0};
  record["world_frame_id"] = "map";
  // toJson() takes the FlowMap's own mutex internally; do not hold mutex_
  // here to avoid an A/B lock-order with FlowMap operations elsewhere.
  record["flow_map"] = flow_map_.toJson();
  record["cell_count"] = static_cast<int64_t>(flow_map_.numCells());

  // Persist the edge coupling state so checkpointed runs preserve the per-edge
  // co-occurrence / marginals / PMI-predictor state. Empty record when the map
  // isn't enabled.
  if (edge_coupling_map_) {
    record["edge_coupling"] = edge_coupling_map_->toJson();
  }

  std::vector<std::uint8_t> bytes;
  try {
    bytes = nlohmann::json::to_msgpack(record);
  } catch (const std::exception& e) {
    if (error) *error = std::string("msgpack encode failed: ") + e.what();
    return false;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out.good()) {
    if (error) *error = "failed to open " + path + " for writing";
    return false;
  }
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out.good()) {
    if (error) *error = "write failed on " + path;
    return false;
  }
  LOG(INFO) << "[FlowTemporal] saved " << flow_map_.numCells() << " cells, "
            << bytes.size() << " bytes (msgpack) to " << path;
  return true;
}

bool FlowTemporalModule::loadFlowState(const std::string& path,
                                       std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    if (error) *error = "failed to open " + path + " for reading";
    return false;
  }
  std::vector<std::uint8_t> bytes(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());

  nlohmann::json record;
  try {
    record = nlohmann::json::from_msgpack(bytes);
  } catch (const std::exception& e) {
    if (error) *error = std::string("msgpack decode failed: ") + e.what();
    return false;
  }
  if (!record.is_object()) {
    if (error) *error = "flow_state root must be a map";
    return false;
  }
  if (!record.contains("version") || !record.at("version").is_number_integer()) {
    if (error) *error = "missing integer version";
    return false;
  }
  const int version = record.at("version").get<int>();
  // v1, v2, v3 all readable. v3 adds optional header keys (grid_origin,
  // world_frame_id) — older files without them default to (0,0,0) / "map",
  // which is what every existing v1/v2 file implicitly used.
  if (version != 1 && version != 2 && version != 3) {
    if (error) *error = "unsupported flow_state version " + std::to_string(version);
    return false;
  }

  // v3 cross-baseline header assertions (informational for kairos: the
  // voxel grid is origin-less so a non-zero saved origin is suspect, and
  // a non-"map" frame id is a hint that the saved session ran in a
  // different world frame). Warn rather than abort so existing TBD
  // checkpoints (all "map", all origin 0) round-trip silently.
  if (record.contains("grid_origin") && record.at("grid_origin").is_array()) {
    const auto& go = record.at("grid_origin");
    if (go.size() == 3) {
      const double gx = go.at(0).get<double>();
      const double gy = go.at(1).get<double>();
      const double gz = go.at(2).get<double>();
      if (std::abs(gx) > 1e-9 || std::abs(gy) > 1e-9 || std::abs(gz) > 1e-9) {
        LOG(WARNING) << "[FlowTemporal] loaded flow_state grid_origin = ("
                     << gx << "," << gy << "," << gz << ") is non-zero; "
                     << "current run assumes (0,0,0).";
      }
    }
  }
  if (record.contains("world_frame_id") && record.at("world_frame_id").is_string()) {
    const std::string fid = record.at("world_frame_id").get<std::string>();
    if (fid != "map") {
      LOG(WARNING) << "[FlowTemporal] loaded flow_state world_frame_id = '"
                   << fid << "' (expected 'map').";
    }
  }

  // Restore voxel grid first so subsequent scoring calls have a working
  // world->index translator without waiting for a fresh map update.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    voxel_size_ = record.value("voxel_size", 0.0f);
    if (voxel_size_ > 0.0f) {
      voxel_grid_ = std::make_unique<spatial_hash::IndexGrid>(voxel_size_);
    } else {
      voxel_grid_.reset();
      LOG(WARNING) << "[FlowTemporal] loaded flow_state has zero voxel_size; "
                      "voxel grid will be rebuilt on first map update";
    }
  }

  std::string fm_err;
  if (!flow_map_.fromJson(record.at("flow_map"), &fm_err)) {
    if (error) *error = "flow_map load failed: " + fm_err;
    return false;
  }

  // Restore the edge coupling state when the checkpoint carries one.
  if (record.contains("edge_coupling")) {
    std::string ec_err;
    if (!edge_coupling_map_->fromJson(record.at("edge_coupling"), &ec_err)) {
      if (error) *error = "edge_coupling load failed: " + ec_err;
      return false;
    }
    LOG(INFO) << "[FlowTemporal] loaded edge coupling map ("
              << edge_coupling_map_->edgeCount() << " edges)";
  }

  LOG(INFO) << "[FlowTemporal] loaded " << flow_map_.numCells()
            << " cells from " << path
            << " (voxel_size=" << voxel_size_ << ")";
  return true;
}

std::string FlowTemporalModule::printInfo() const {
  return config::toString(config);
}

void FlowTemporalModule::handleMapUpdate(uint64_t,
                                         const VolumetricMap& map,
                                         const ActiveWindowOutput&) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Collect the active voxels from the active-window TSDF: this update is the
  // current ground truth, so voxels that dropped out of the active window are
  // forgotten (the reset below replaces the set wholesale).
  GlobalIndexSet active;
  const auto block_indices = map.getTsdfLayer().allocatedBlockIndices();
  for (const auto& index : block_indices) {
    const auto& block = map.getTsdfLayer().getBlock(index);
    for (size_t i = 0; i < block.numVoxels(); ++i) {
      const auto& voxel = block.getVoxel(i);
      if (voxel.weight < config.min_tsdf_weight_for_visibility) {
        continue;
      }
      active.insert(block.getGlobalVoxelIndex(i));
    }
  }
  resetActiveVoxelsLocked(map.config.voxel_size, std::move(active));
}

void FlowTemporalModule::resetActiveVoxelsLocked(float voxel_size,
                                                 GlobalIndexSet active) {
  voxel_size_ = voxel_size;
  if (!voxel_grid_ || std::abs(voxel_grid_->voxel_size - voxel_size_) > 1.0e-6f) {
    voxel_grid_ = std::make_unique<spatial_hash::IndexGrid>(voxel_size_);
  }
  active_voxels_ = std::move(active);
}

void FlowTemporalModule::setVisibleVoxels(
    float voxel_size, const std::vector<Eigen::Vector3d>& xyz_world) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Voxelize the supplied points with a grid at the same resolution the reset
  // will install. spatial_hash::IndexGrid is origin-less (floor(xyz / size)), so
  // a local grid yields the same indices the member grid would.
  spatial_hash::IndexGrid grid(voxel_size);
  GlobalIndexSet active;
  for (const auto& p : xyz_world) {
    active.insert(grid.toIndex(p.cast<float>()));
  }
  resetActiveVoxelsLocked(voxel_size, std::move(active));
}

void FlowTemporalModule::handleBackendUpdate(
    uint64_t timestamp_ns,
    const DynamicSceneGraph& graph,
    const kimera_pgmo::DeformationGraph& dgraph) {
  // Called on the backend thread under BackendModule::mutex_, after
  // deformation_graph_->optimize() in callUpdateFunctions(). At this point
  // the dgraph is post-optimization and not being mutated by anyone else,
  // so we can take a self-consistent copy of the read slice. The previous
  // implementation stored only a raw pointer to the live dgraph, which the
  // worker thread later dereferenced without synchronization — that race
  // produced the SIGSEGV-in-__dynamic_cast inside interpPoint when the
  // backend was mid-mutating gtsam::Values on the next update.
  const auto snap_start = std::chrono::steady_clock::now();
  auto snap = std::make_shared<DgraphSnapshot>();
  snap->timestamp_ns = timestamp_ns;
  const auto prefix = GlobalInfo::instance().getRobotPrefix().vertex_key;
  snap->prefix = prefix;
  if (dgraph.hasVertexKey(prefix)) {
    snap->control_points = dgraph.getInitialPositionsVertices(prefix);
    snap->vertex_stamps = dgraph.getVertexStamps(prefix);
    snap->gtsam_values = dgraph.getGtsamValues();
  }
  if (config.flow.enable_timing) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - snap_start).count();
    LOG(INFO) << "[FlowTiming] dgraph_snapshot=" << elapsed_us
              << "us values=" << snap->gtsam_values.size()
              << " control_points=" << snap->control_points.size();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_snapshot_ = std::move(snap);
    dgraph_updated_ = true;
  }
  auto& mutable_graph = const_cast<DynamicSceneGraph&>(graph);
  const double timestamp_s =
      timestamp_ns > 0 ? static_cast<double>(timestamp_ns) * 1.0e-9 : wallClockSeconds();
  updatePlaceSummaries(timestamp_s, mutable_graph);

  // Embed the flow map in the graph's top-level metadata so it round-
  // trips with the DSG, just like the mesh. BackendModule::save calls
  // graph->save(...), which serializes graph.metadata; the offline
  // visualizer then reads `metadata.flow` directly from the DSG JSON
  // without needing a sidecar file.
  if (!mutable_graph.metadata.is_object()) {
    mutable_graph.metadata = nlohmann::json::object();
  }
  flow_map_.flushVisits();  // feed open per-track visits before serializing
  mutable_graph.metadata["flow"] = {
      {"voxel_size", voxel_size_},
      {"map", flow_map_.toJson()},
  };
}

void FlowTemporalModule::pushDetectionFrame(
    double timestamp_s, std::vector<FrameObservation> observations) {
  DetectionFrame frame;
  frame.timestamp_s = timestamp_s;
  frame.observations = std::move(observations);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_queue_.push_back(std::move(frame));
    while (static_cast<int>(frame_queue_.size()) > config.max_queue_size) {
      frame_queue_.pop_front();
    }
  }

  cv_.notify_one();
}

bool FlowTemporalModule::pushDetections(double timestamp_s,
                                        const std::vector<Eigen::Vector3d>& xyz_world,
                                        const std::vector<double>& theta,
                                        const std::vector<double>& rho,
                                        const std::vector<int64_t>& track_ids) {
  if (xyz_world.size() != theta.size() || xyz_world.size() != rho.size() ||
      (!track_ids.empty() && track_ids.size() != xyz_world.size())) {
    LOG(WARNING) << "[FlowTemporal] pushDetections: size mismatch (xyz="
                 << xyz_world.size() << ", theta=" << theta.size()
                 << ", rho=" << rho.size() << ", ids=" << track_ids.size() << ")";
    return false;
  }

  std::unique_ptr<spatial_hash::IndexGrid> local_grid;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!voxel_grid_) {
      return false;
    }
    local_grid = std::make_unique<spatial_hash::IndexGrid>(*voxel_grid_);
  }

  std::vector<FrameObservation> obs;
  obs.reserve(xyz_world.size());
  for (size_t i = 0; i < xyz_world.size(); ++i) {
    if (rho[i] <= 0.0) {
      continue;
    }
    FrameObservation o;
    o.index = local_grid->toIndex(xyz_world[i].cast<float>());
    o.theta = theta[i];
    o.rho = rho[i];
    o.track_id = track_ids.empty() ? -1 : track_ids[i];
    obs.push_back(o);
  }

  pushDetectionFrame(timestamp_s, std::move(obs));
  return true;
}

std::optional<FlowTemporalModule::QueryResult>
FlowTemporalModule::queryFlowAtPosition(const Eigen::Vector3f& pos,
                                        double query_time_s,
                                        double horizon_s) const {
  GlobalIndex index;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!voxel_grid_) {
      return std::nullopt;
    }
    index = voxel_grid_->toIndex(pos);
  }

  QueryResult result;
  const auto snap = flow_map_.tryGetCellSnapshot(index, query_time_s, horizon_s);
  if (!snap.has_value() || snap->components.empty()) {
    return result;
  }

  double best_weight = -1.0;
  for (const auto& comp : snap->components) {
    if (comp.weight > best_weight) {
      best_weight = comp.weight;
      result.dominant_theta = comp.theta;
      result.dominant_rho = comp.rho;
    }
  }
  result.probability = snap->p_present;
  result.flag = static_cast<uint8_t>(snap->flag);
  result.num_components = static_cast<int32_t>(snap->components.size());
  return result;
}

std::vector<FlowTemporalModule::ObservationScore>
FlowTemporalModule::scoreObservations(
    const std::vector<double>& timestamps_s,
    const std::vector<Eigen::Vector3d>& xyz_world,
    const std::vector<double>& theta,
    const std::vector<double>& rho,
    double horizon_s) const {
  const bool timing_enabled = config.flow.enable_timing;
  const auto t_score_start = std::chrono::steady_clock::now();
  const size_t N = timestamps_s.size();
  std::vector<ObservationScore> out(N);
  if (xyz_world.size() != N || theta.size() != N || rho.size() != N) {
    LOG_FIRST_N(WARNING, 3) << "[FlowScore] size mismatch: t=" << N
                            << " xyz=" << xyz_world.size()
                            << " theta=" << theta.size()
                            << " rho=" << rho.size()
                            << "; returning zeroed rows";
    return out;
  }

  // Pull out a local copy of the voxel grid under mutex_, then release it
  // before the per-observation loop. The grid pointer is stable across
  // active-window updates (only swapped under mutex_), so a snapshotted
  // copy stays valid for the duration of this call without contending with
  // processFrame. Per-row flow_map_.tryGetCellSnapshot takes its own lock.
  std::unique_ptr<spatial_hash::IndexGrid> grid_local;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (voxel_grid_) {
      grid_local = std::make_unique<spatial_hash::IndexGrid>(*voxel_grid_);
    }
  }
  if (!grid_local) {
    LOG_FIRST_N(WARNING, 3) << "[FlowScore] voxel grid not initialized; "
                               "returning zero matched rows (call after at "
                               "least one map update)";
    return out;
  }

  for (size_t i = 0; i < N; ++i) {
    const Eigen::Vector3f pos = xyz_world[i].cast<float>();
    const GlobalIndex index = grid_local->toIndex(pos);
    const auto snap = flow_map_.tryGetCellSnapshot(index, timestamps_s[i], horizon_s);
    if (!snap.has_value() || snap->components.empty()) {
      continue;
    }
    out[i].matched = true;
    out[i].flag = static_cast<uint8_t>(snap->flag);
    out[i].p_present = snap->p_present;
    out[i].n_crossings = snap->n_crossings;

    // p(theta, rho) = sum_k pi_k * SW-N((theta, rho) | mu_k, Sigma_k).
    // We rebuild SWNDComponents from the snapshot tuples so the winding-sum
    // evaluator in swnd.cpp can be reused (matches the C++ density math
    // everywhere else in the codebase and avoids drift between scoring and
    // the model's own density evaluation).
    double density = 0.0;
    double density_theta = 0.0;
    double density_rho = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    double srho = 0.0;
    double sw = 0.0;
    // Mixing-weight-weighted variance accumulators. std_theta / std_rho are
    // NaN-safe: a component with zero weight or zero std contributes 0.
    double var_theta_sum = 0.0;
    double var_rho_sum = 0.0;
    out[i].slot_weights.reserve(snap->components.size());
    for (const auto& comp : snap->components) {
      SWNDComponent c;
      c.mu_theta = comp.theta;
      c.mu_rho = comp.rho;
      c.Sigma << comp.sigma_theta, comp.sigma_cross, comp.sigma_cross, comp.sigma_rho;
      precompute(c);
      density += comp.weight * evaluate(c, theta[i], rho[i], 2);
      // Same components, the other variable integrated out (heading/speed
      // channel marginals). Shares the regularized Sigma from precompute(c).
      density_theta += comp.weight * evaluateThetaMarginal(c, theta[i], 2);
      density_rho += comp.weight * evaluateRhoMarginal(c, rho[i]);
      out[i].slot_weights.push_back(comp.weight);
      // Weight-side uncertainty rides along per slot (not pooled across
      // components like std_theta/std_rho: each slot's weight interval is a
      // separate statement). std_pi is 0.0 throughout when tracking is off,
      // and the vectors stay aligned with slot_weights by construction.
      out[i].slot_weight_stds.push_back(comp.std_pi);
      out[i].slot_weight_clamped.push_back(comp.pi_clamped ? 1 : 0);
      out[i].slot_weight_orders.push_back(comp.pi_order);
      if (comp.weight > 0.0) {
        sx += comp.weight * std::cos(comp.theta);
        sy += comp.weight * std::sin(comp.theta);
        srho += comp.weight * comp.rho;
        sw += comp.weight;
        var_theta_sum += comp.weight * comp.std_theta * comp.std_theta;
        var_rho_sum += comp.weight * comp.std_rho * comp.std_rho;
      }
    }
    out[i].log_p_joint = std::log(std::max(density, 1.0e-10));
    out[i].log_p_heading = std::log(std::max(density_theta, 1.0e-10));
    out[i].log_p_speed = std::log(std::max(density_rho, 1.0e-10));
    if (sw > 0.0) {
      double t_mean = std::atan2(sy, sx);
      if (t_mean < 0.0) {
        t_mean += 2.0 * M_PI;
      }
      out[i].mean_theta = t_mean;
      out[i].mean_rho = srho / sw;
      out[i].std_theta = std::sqrt(std::max(0.0, var_theta_sum / sw));
      out[i].std_rho = std::sqrt(std::max(0.0, var_rho_sum / sw));
    }
  }
  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - t_score_start)
                                .count();
    std::lock_guard<std::mutex> lk(timing_mutex_);
    module_timing_.score_count++;
    module_timing_.score_detections += static_cast<uint64_t>(N);
    module_timing_.score_total_us += static_cast<uint64_t>(elapsed_us);
    module_timing_.score_samples_us.push_back(static_cast<uint32_t>(elapsed_us));
  }
  return out;
}

std::vector<FlowMap::PresenceGridEntry> FlowTemporalModule::presenceGrid(
    double t_seconds, const std::vector<double>& horizons_s) const {
  GlobalIndexSet visible;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    visible = active_voxels_;
  }
  return flow_map_.presenceGrid(visible, t_seconds, horizons_s);
}

std::vector<FlowMap::FlowGridEntry> FlowTemporalModule::flowGrid(double t_seconds) const {
  GlobalIndexSet visible;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    visible = active_voxels_;
  }
  return flow_map_.flowGrid(visible, t_seconds);
}

void FlowTemporalModule::setTickCallback(TickCallback cb) {
  std::lock_guard<std::mutex> lock(mutex_);
  tick_callback_ = std::move(cb);
}

std::unordered_map<NodeId, FlowTemporalModule::PlaceFlowSummary>
FlowTemporalModule::placeSummariesCopy() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return place_summaries_;
}

std::vector<FlowTemporalModule::LcEvent>
FlowTemporalModule::lcEventsCopy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lc_events_;
}

std::string FlowTemporalModule::couplingJson() const {
  if (!edge_coupling_map_) {
    return "{}";
  }
  return edge_coupling_map_->toJson().dump();
}

FlowTemporalModule::ModuleTiming FlowTemporalModule::moduleTiming() const {
  std::lock_guard<std::mutex> lk(timing_mutex_);
  return module_timing_;
}

size_t FlowTemporalModule::lastFrameObservations() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return last_frame_observations_;
}

size_t FlowTemporalModule::lastFrameUsedObservations() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return last_frame_used_observations_;
}

size_t FlowTemporalModule::placeNodesSeen() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return place_nodes_seen_;
}

size_t FlowTemporalModule::placeNodesWithSupport() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return place_nodes_with_support_;
}

size_t FlowTemporalModule::placeNodesWithComponents() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return place_nodes_with_components_;
}

size_t FlowTemporalModule::placeSummariesPlacesCount() const {
  std::lock_guard<std::mutex> lock(summaries_mutex_);
  return place_summaries_places_count_;
}

void FlowTemporalModule::workerSpin() {
  const auto wait_ms = std::max(
      1, static_cast<int>(std::round(1000.0 / std::max(0.1f, config.worker_rate_hz))));
  last_heavy_maintenance_time_ = std::chrono::steady_clock::now();
  const double maintenance_period_s = 1.0 / config.heavy_maintenance_rate_hz;

  while (true) {
    std::vector<DetectionFrame> frames;
    TickCallback tick_cb_local;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this]() {
        return should_shutdown_ || !frame_queue_.empty();
      });

      if (should_shutdown_) {
        break;
      }

      while (!frame_queue_.empty()) {
        frames.push_back(std::move(frame_queue_.front()));
        frame_queue_.pop_front();
      }
      worker_processing_ = !frames.empty();

      tick_cb_local = tick_callback_;
    }

    double last_ts = 0.0;
    for (const auto& frame : frames) {
      processFrame(frame);
      last_ts = frame.timestamp_s;
    }
    if (last_ts > 0.0) {
      last_detection_ts_ = last_ts;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - last_heavy_maintenance_time_).count();

    if (elapsed_s >= maintenance_period_s) {
      const double ts = last_ts > 0.0 ? last_ts : wallClockSeconds();
      std::shared_ptr<const DgraphSnapshot> local_snapshot;
      bool should_deform = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        local_snapshot = latest_snapshot_;
        should_deform = dgraph_updated_;
        dgraph_updated_ = false;
      }

      if (local_snapshot && should_deform) {
        runDeformationAlignment(*local_snapshot);
      }

      flow_map_.runMaintenance(ts);
      last_heavy_maintenance_time_ = now;
    }

    const double current_ts = last_ts > 0.0 ? last_ts : wallClockSeconds();
    if (tick_cb_local) {
      tick_cb_local(current_ts);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_processing_ = false;
    }
    cv_.notify_all();  // wake any flush() waiter now the batch is done
  }
}

size_t FlowTemporalModule::pendingFrames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return frame_queue_.size();
}

void FlowTemporalModule::flush() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() {
    return frame_queue_.empty() && !worker_processing_;
  });
}

void FlowTemporalModule::applyDeformationSnapshot(
    double timestamp_s,
    const std::vector<Eigen::Vector3d>& points_before,
    const std::vector<double>& stamps_s,
    const std::vector<Eigen::Vector3d>& points_after,
    const std::vector<double>& yaw_after) {
  if (points_before.size() != stamps_s.size() ||
      points_before.size() != points_after.size()) {
    LOG(ERROR) << "[FlowDeform] applyDeformationSnapshot size mismatch: "
               << points_before.size() << "/" << stamps_s.size() << "/"
               << points_after.size();
    return;
  }
  if (!yaw_after.empty() && yaw_after.size() != points_before.size()) {
    LOG(ERROR) << "[FlowDeform] applyDeformationSnapshot yaw size mismatch: "
               << yaw_after.size() << " against " << points_before.size()
               << " control points; ignoring the rotation.";
    return;
  }
  // deformPoints sweeps its control-point octree forward by stamp, so the
  // snapshot arrays must be stamp-sorted; keys are assigned by index AFTER
  // sorting so control_points[i] and Symbol(prefix, i) stay paired.
  std::vector<size_t> order(points_before.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return stamps_s[a] < stamps_s[b];
  });
  // The prefix only has to be consistent between control_points and the
  // gtsam keys built here; no backend robot prefix is involved.
  constexpr char kPrefix = 'v';
  DgraphSnapshot snap;
  snap.timestamp_ns = static_cast<uint64_t>(timestamp_s * 1.0e9);
  snap.prefix = kPrefix;
  snap.control_points.reserve(points_before.size());
  snap.vertex_stamps.reserve(stamps_s.size());
  for (size_t i = 0; i < order.size(); ++i) {
    const auto src = order[i];
    snap.control_points.emplace_back(points_before[src]);
    snap.vertex_stamps.push_back(
        static_cast<kimera_pgmo::Timestamp>(stamps_s[src] * 1.0e9));
    // Per-control-point correction: interpPoint computes R * (v - g) + t per
    // neighbour, so t is the control point's own corrected location and R the
    // rotation the correction turns the frame through there. Identity rotation
    // reduces it to the pure translation v + (t - g).
    const auto rotation =
        yaw_after.empty() ? gtsam::Rot3() : gtsam::Rot3::Yaw(yaw_after[src]);
    snap.gtsam_values.insert(
        gtsam::Symbol(kPrefix, i),
        gtsam::Pose3(rotation, gtsam::Point3(points_after[src])));
  }
  runDeformationAlignment(snap);
}

void FlowTemporalModule::runDeformationAlignment(
    const DgraphSnapshot& snapshot) {
  const bool timing_enabled = config.flow.enable_timing;
  const auto start = std::chrono::steady_clock::now();

  if (!voxel_grid_ || flow_map_.numCells() == 0) {
    return;
  }

  const auto prefix = snapshot.prefix;
  if (snapshot.control_points.empty()) {
    // hasVertexKey was false at snapshot time — nothing to align against.
    return;
  }

  const auto inputs = flow_map_.cellRemapInputs();
  if (inputs.empty()) {
    return;
  }

  // Save the cells' KEYs in cells_ (post all prior remaps) — these are the
  // "from" side of the (old, new) pairs we hand to remapCells. The DEFORMATION
  // INPUT is something different: each cell's *original* index + stamp, so the
  // deformation graph windows control points around the original observation
  // in space-time. Feeding back the current corrected position (as the
  // pre-fix code did) is what produced the cross-room collisions: an
  // already-corrected kitchen cell would, on the next graph update, be
  // re-deformed by whatever control points happened to fall near its
  // *current* position, which after a long run can be living-room control
  // points whose translations have nothing to do with the kitchen.
  const float vs = voxel_size_;
  pcl::PointCloud<pcl::PointXYZ> originals;
  std::vector<uint64_t> stamps_ns;
  std::vector<size_t> deformable_indices;  // indices into `inputs` that we deform
  originals.reserve(inputs.size());
  stamps_ns.reserve(inputs.size());
  deformable_indices.reserve(inputs.size());
  size_t skipped_no_origin = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& in = inputs[i];
    if (!in.original_set) {
      // No first-observation stamp/index recorded (cell from old checkpoint
      // or only ensureCells'd, never observed). Don't deform — passing it
      // through with stamp=0 would let the stamped overload pull in the
      // earliest control points regardless of where the cell physically is.
      ++skipped_no_origin;
      continue;
    }
    pcl::PointXYZ p;
    p.x = (static_cast<float>(in.original_index.x()) + 0.5f) * vs;
    p.y = (static_cast<float>(in.original_index.y()) + 0.5f) * vs;
    p.z = (static_cast<float>(in.original_index.z()) + 0.5f) * vs;
    originals.push_back(p);
    stamps_ns.push_back(in.creation_timestamp_ns);
    deformable_indices.push_back(i);
  }

  if (originals.empty()) {
    LOG(INFO) << "[FlowDeform.stats] cells=" << inputs.size()
              << " skipped_no_origin=" << skipped_no_origin
              << " (nothing to deform; all cells lack original-position metadata).";
    return;
  }

  // Stamps must be non-decreasing for the windowed control-point selection
  // in deformation::deformPoints (it sweeps an octree forward by stamp). Sort
  // the parallel arrays jointly without disturbing `deformable_indices`'
  // mapping back into `inputs`.
  std::vector<size_t> order(originals.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return stamps_ns[a] < stamps_ns[b];
  });
  pcl::PointCloud<pcl::PointXYZ> originals_sorted;
  std::vector<uint64_t> stamps_sorted;
  std::vector<size_t> deformable_sorted;
  originals_sorted.reserve(originals.size());
  stamps_sorted.reserve(originals.size());
  deformable_sorted.reserve(originals.size());
  for (const auto idx : order) {
    originals_sorted.push_back(originals[idx]);
    stamps_sorted.push_back(stamps_ns[idx]);
    deformable_sorted.push_back(deformable_indices[idx]);
  }
  originals = std::move(originals_sorted);
  stamps_ns = std::move(stamps_sorted);
  deformable_indices = std::move(deformable_sorted);

  // The stamped namespace overload of `kimera_pgmo::deformation::deformPoints`
  // is the one we want. The DeformationGraph member `deformPoints` mutates
  // an internal `last_calculated_vertices_[prefix]` cache that the mesh path
  // also writes into — calling it from here would corrupt the mesh's cache.
  // The namespace overload has no such side effect.
  const auto& gtsam_values = snapshot.gtsam_values;
  const auto& control_points = snapshot.control_points;
  const auto& vertex_stamps = snapshot.vertex_stamps;

  pcl::PointCloud<pcl::PointXYZ> new_points = originals;  // initialised to identity
  kimera_pgmo::ConstStampedCloud<pcl::PointXYZ> cloud_in{originals, stamps_ns};
  std::vector<std::set<size_t>> control_point_map;  // scratch
  kimera_pgmo::deformation::deformPoints(new_points,
                                         control_point_map,
                                         cloud_in,
                                         prefix,
                                         control_points,
                                         vertex_stamps,
                                         gtsam_values,
                                         static_cast<size_t>(config.deformation_knn),
                                         config.deformation_tolerance_s,
                                         nullptr);

  // Re-expose `old_indices` aligned with `originals` so the rest of the
  // diagnostics + remap_pair construction is symmetric with the original
  // code path. `inputs[deformable_indices[i]].current_index` is the cells_
  // KEY for the i-th deformed point.
  std::vector<GlobalIndex> old_indices;
  pcl::PointCloud<pcl::PointXYZ> old_points;
  old_indices.reserve(originals.size());
  old_points.reserve(originals.size());
  for (size_t i = 0; i < originals.size(); ++i) {
    old_indices.push_back(inputs[deformable_indices[i]].current_index);
    old_points.push_back(originals[i]);  // for diagnostic norms vs deformed
  }

  // Per-cell yaw of the correction. A rotating correction displaces the cell
  // and turns the bearings stored in it through the same angle, so the two
  // have to be read off the same control points. deformPoints reports, per
  // deformed point, the control points it interpolated over; the cell's yaw is
  // the circular mean of their yaws. That is an equal-weight mean over the
  // neighbour set where the position used distance weights over the same set,
  // an approximation of at most the spread of yaw across one neighbourhood,
  // far below the width of a direction slot at any drift rate a backend
  // corrects.
  std::vector<double> cell_yaw;
  double max_ctrl_yaw = 0.0;
  for (const auto& key_value : gtsam_values) {
    const auto& pose = gtsam_values.at<gtsam::Pose3>(key_value.key);
    max_ctrl_yaw = std::max(max_ctrl_yaw, std::abs(pose.rotation().yaw()));
  }
  const bool rotating = max_ctrl_yaw > 0.0;
  if (rotating && control_point_map.size() != originals.size()) {
    // The stamped overload skips points it cannot interpolate, leaving the
    // neighbour sets misaligned with the cloud. Correcting positions is still
    // right; rotating headings against the wrong neighbour set is not.
    LOG(WARNING) << "[FlowDeform] control_point_map has "
                 << control_point_map.size() << " entries for "
                 << originals.size()
                 << " deformed points; skipping the heading rotation this pass.";
  } else if (rotating) {
    cell_yaw.assign(originals.size(), 0.0);
    // The equal-weight mean differs from the distance-weighted one by at most
    // the spread of yaw across a neighbourhood, so that spread is the bound on
    // the approximation and is tracked here rather than assumed small. The
    // control points of one neighbourhood are consecutive samples of the same
    // trajectory segment, so the spread is the drift accrued over a few
    // seconds of travel; anything above a few degrees means the neighbourhood
    // straddles corrections of genuinely different frames and the equal-weight
    // reading no longer stands in for the weighted one.
    double max_neighbourhood_spread = 0.0;
    for (size_t i = 0; i < originals.size(); ++i) {
      double sum_cos = 0.0;
      double sum_sin = 0.0;
      double min_yaw = std::numeric_limits<double>::max();
      double max_yaw = std::numeric_limits<double>::lowest();
      for (const auto j : control_point_map[i]) {
        const auto symbol = gtsam::Symbol(prefix, j);
        if (!gtsam_values.exists(symbol)) {
          continue;
        }
        const double yaw = gtsam_values.at<gtsam::Pose3>(symbol).rotation().yaw();
        sum_cos += std::cos(yaw);
        sum_sin += std::sin(yaw);
        min_yaw = std::min(min_yaw, yaw);
        max_yaw = std::max(max_yaw, yaw);
      }
      if (sum_cos != 0.0 || sum_sin != 0.0) {
        cell_yaw[i] = std::atan2(sum_sin, sum_cos);
      }
      if (max_yaw >= min_yaw) {
        max_neighbourhood_spread =
            std::max(max_neighbourhood_spread, max_yaw - min_yaw);
      }
    }
    constexpr double kMaxSpreadRad = 5.0 * M_PI / 180.0;
    if (max_neighbourhood_spread > kMaxSpreadRad) {
      LOG(WARNING) << "[FlowDeform] control-point yaw spreads by up to "
                   << max_neighbourhood_spread * 180.0 / M_PI
                   << " deg within one neighbourhood. The per-cell yaw is an "
                      "equal-weight mean over the neighbours the position "
                      "blend weighted by distance, and that spread bounds the "
                      "difference between the two.";
    }
  }

  // Quantitative diagnostics on the deformation field. The cross-room-merge
  // complaint (kitchen voxel paired with living-room voxel after loop closure)
  // can come from either (a) deformation legitimately mapping a drifted-frame
  // cell onto its physically-correct location across the room, in which case
  // displacements are *consistent* across nearby cells, or (b) the kNN /
  // tolerance setup producing wildly different corrections for adjacent
  // cells, in which case displacements are *inconsistent*. The stats below
  // surface both cases. We also flag the top-K cells by displacement so the
  // exact (old_pos -> new_pos) jumps are visible in the log.
  struct Disp {
    double norm_m = 0.0;
    Eigen::Vector3f old_p;
    Eigen::Vector3f new_p;
    GlobalIndex old_idx;
    GlobalIndex new_idx;
  };
  std::vector<Disp> displacements;
  displacements.reserve(old_indices.size());

  std::vector<std::pair<GlobalIndex, GlobalIndex>> remap_pairs;
  remap_pairs.reserve(old_indices.size());
  size_t identity_index_count = 0;
  size_t zero_displacement_count = 0;
  double sum_norm_m = 0.0;
  double sum_norm_sq_m = 0.0;
  double max_norm_m = 0.0;
  double sum_dx = 0.0, sum_dy = 0.0, sum_dz = 0.0;  // mean drift correction
  for (size_t i = 0; i < old_indices.size(); ++i) {
    const Eigen::Vector3f old_p(
        old_points[i].x, old_points[i].y, old_points[i].z);
    const Eigen::Vector3f new_p(
        new_points[i].x, new_points[i].y, new_points[i].z);
    const auto new_index = voxel_grid_->toIndex(new_p);
    remap_pairs.emplace_back(old_indices[i], new_index);

    if (new_index.x() == old_indices[i].x() &&
        new_index.y() == old_indices[i].y() &&
        new_index.z() == old_indices[i].z()) {
      ++identity_index_count;
    }

    const Eigen::Vector3f delta = new_p - old_p;
    const double norm = static_cast<double>(delta.norm());
    if (norm < 1.0e-9) {
      ++zero_displacement_count;
    }
    sum_norm_m += norm;
    sum_norm_sq_m += norm * norm;
    sum_dx += static_cast<double>(delta.x());
    sum_dy += static_cast<double>(delta.y());
    sum_dz += static_cast<double>(delta.z());
    if (norm > max_norm_m) {
      max_norm_m = norm;
    }
    Disp d;
    d.norm_m = norm;
    d.old_p = old_p;
    d.new_p = new_p;
    d.old_idx = old_indices[i];
    d.new_idx = new_index;
    displacements.push_back(d);
  }

  const double n = static_cast<double>(old_indices.size());
  const double mean_norm_m = n > 0 ? sum_norm_m / n : 0.0;
  const double rms_norm_m = n > 0 ? std::sqrt(sum_norm_sq_m / n) : 0.0;
  // Percentiles via partial sort.
  auto percentile = [&](double q) {
    if (displacements.empty()) {
      return 0.0;
    }
    const size_t k = std::min(displacements.size() - 1,
                              static_cast<size_t>(q * (displacements.size() - 1)));
    std::nth_element(displacements.begin(),
                     displacements.begin() + k,
                     displacements.end(),
                     [](const Disp& a, const Disp& b) {
                       return a.norm_m < b.norm_m;
                     });
    return displacements[k].norm_m;
  };
  // Compute percentiles before re-sorting for top-K (nth_element disturbs
  // order, so call them in increasing q so each partial sort uses the prior
  // partition as a starting point — order-independent for our purposes).
  const double p50 = percentile(0.50);
  const double p95 = percentile(0.95);
  const double p99 = percentile(0.99);

  // Non-zero displacements only, for a cleaner mean correction direction.
  const double mean_dx = n > 0 ? sum_dx / n : 0.0;
  const double mean_dy = n > 0 ? sum_dy / n : 0.0;
  const double mean_dz = n > 0 ? sum_dz / n : 0.0;
  const double mean_correction_norm =
      std::sqrt(mean_dx * mean_dx + mean_dy * mean_dy + mean_dz * mean_dz);

  LOG(INFO) << "[FlowDeform.stats] cells_total=" << inputs.size()
            << " deformed=" << old_indices.size()
            << " skipped_no_origin=" << skipped_no_origin
            << " identity_idx=" << identity_index_count
            << " zero_disp=" << zero_displacement_count
            << " disp_m: mean=" << mean_norm_m << " rms=" << rms_norm_m
            << " p50=" << p50 << " p95=" << p95 << " p99=" << p99
            << " max=" << max_norm_m
            << " mean_correction=(" << mean_dx << "," << mean_dy << "," << mean_dz
            << ") |mean_correction|=" << mean_correction_norm
            << " voxel_size=" << voxel_size_
            << " knn=" << config.deformation_knn
            << " tolerance_s=" << config.deformation_tolerance_s
            << " gtsam_values=" << gtsam_values.size()
            << " control_points=" << control_points.size()
            << " (disp_m is |deformed - original|, the total drift "
               "correction).";

  // Top-K largest displacements, full pos -> pos info so cross-room jumps
  // are inspectable in the log without a bag.
  if (!displacements.empty()) {
    std::sort(displacements.begin(),
              displacements.end(),
              [](const Disp& a, const Disp& b) {
                return a.norm_m > b.norm_m;
              });
    const size_t k_top = std::min<size_t>(8, displacements.size());
    for (size_t i = 0; i < k_top; ++i) {
      const auto& d = displacements[i];
      LOG(INFO) << "[FlowDeform.top] rank=" << i << " norm_m=" << d.norm_m
                << " old_pos=(" << d.old_p.x() << "," << d.old_p.y() << ","
                << d.old_p.z() << ") new_pos=(" << d.new_p.x() << ","
                << d.new_p.y() << "," << d.new_p.z() << ")"
                << " old_idx=(" << d.old_idx.x() << "," << d.old_idx.y() << ","
                << d.old_idx.z() << ") new_idx=(" << d.new_idx.x() << ","
                << d.new_idx.y() << "," << d.new_idx.z() << ")";
    }
  }

  // Inconsistency check: large variance in displacement direction across
  // physically-adjacent cells points at the kNN/tolerance failure mode.
  // We sample by comparing each cell's displacement to the population mean.
  if (n > 0 && rms_norm_m > 0.0) {
    double sum_sq_dev = 0.0;
    for (const auto& d : displacements) {
      const double dev_x = static_cast<double>(d.new_p.x() - d.old_p.x()) - mean_dx;
      const double dev_y = static_cast<double>(d.new_p.y() - d.old_p.y()) - mean_dy;
      const double dev_z = static_cast<double>(d.new_p.z() - d.old_p.z()) - mean_dz;
      sum_sq_dev += dev_x * dev_x + dev_y * dev_y + dev_z * dev_z;
    }
    const double rms_dev = std::sqrt(sum_sq_dev / n);
    LOG(INFO) << "[FlowDeform.consistency] rms_dev_from_mean_correction_m="
              << rms_dev << " (large vs mean_correction="
              << mean_correction_norm
              << " means deformation is *not* a coherent rigid drift; cells "
                 "are being moved in inconsistent directions, which can "
                 "produce cross-room collisions).";
  }

  if (!cell_yaw.empty()) {
    double sum_abs_yaw = 0.0;
    double max_abs_yaw = 0.0;
    for (const auto yaw : cell_yaw) {
      sum_abs_yaw += std::abs(yaw);
      max_abs_yaw = std::max(max_abs_yaw, std::abs(yaw));
    }
    LOG(INFO) << "[FlowDeform.rotation] cells=" << cell_yaw.size()
              << " mean|yaw|_deg="
              << (sum_abs_yaw / static_cast<double>(cell_yaw.size())) * 180.0 / M_PI
              << " max|yaw|_deg=" << max_abs_yaw * 180.0 / M_PI
              << " max control-point |yaw|_deg=" << max_ctrl_yaw * 180.0 / M_PI
              << " (the directional state of each cell is rotated by its own "
                 "value before the remap).";
  }

  // cell_yaw is built in the same order as remap_pairs and is empty for a
  // purely translational correction, which selects the translation-only path.
  flow_map_.remapCells(remap_pairs, cell_yaw);

  // Record this alignment as a loop-closure event so downstream evaluation can
  // bracket detections around it.
  //
  // Timestamp priority (all in seconds since Unix epoch, matching the
  // prediction log):
  //   1. last_detection_ts_ — the most recent detection frame seen by the
  //      worker thread. Updated every iteration; matches the prediction-log
  //      timestamps exactly because both come from the input stream's stamps.
  //   2. snapshot.timestamp_ns — the backend's last frame timestamp in ns.
  //      Can be 0 when the backend fires without new input (end-of-run
  //      cleanup), which would otherwise cause a wallclock fallback that
  //      puts the event years away from the detection stream (bug).
  //   3. wallclock — last resort, should never be reached in practice.
  {
    const double event_t_s =
        last_detection_ts_ > 0.0
            ? last_detection_ts_
            : (snapshot.timestamp_ns > 0
                   ? static_cast<double>(snapshot.timestamp_ns) * 1.0e-9
                   : std::chrono::duration<double>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count());
    std::lock_guard<std::mutex> lock(mutex_);
    lc_events_.push_back({event_t_s, old_indices.size(), remap_pairs.size()});
  }

  GlobalIndexMap<GlobalIndex> remap_lookup;
  remap_lookup.reserve(remap_pairs.size());
  for (const auto& [old_index, new_index] : remap_pairs) {
    remap_lookup[old_index] = new_index;
  }

  // Rebuild last_visibility_update_s_ from the remap. Held under mutex_
  // so updatePlaceSummaries on the backend thread never finds()s a
  // half-rehashed map (same race that crashed at SIGSEGV / std::_Hashtable::find).
  {
    std::lock_guard<std::mutex> lock(mutex_);
    GlobalIndexMap<double> remapped_last_visibility;
    remapped_last_visibility.reserve(last_visibility_update_s_.size());
    for (const auto& [old_index, timestamp_s] : last_visibility_update_s_) {
      const auto lookup_iter = remap_lookup.find(old_index);
      const auto target_index =
          lookup_iter == remap_lookup.end() ? old_index : lookup_iter->second;

      auto iter = remapped_last_visibility.find(target_index);
      if (iter == remapped_last_visibility.end() || timestamp_s > iter->second) {
        remapped_last_visibility[target_index] = timestamp_s;
      }
    }
    last_visibility_update_s_ = std::move(remapped_last_visibility);
  }

  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    const double per_cell_us = old_indices.empty()
                                   ? 0.0
                                   : static_cast<double>(elapsed_us) /
                                         static_cast<double>(old_indices.size());
    LOG(INFO) << "[FlowTiming] runDeformationAlignment total=" << elapsed_us
              << " us, cells=" << old_indices.size() << ", avg=" << per_cell_us
              << " us/cell";
  }
}

void FlowTemporalModule::processFrame(const DetectionFrame& frame) {
  const bool timing_enabled = config.flow.enable_timing;
  const auto start = std::chrono::steady_clock::now();

  // Take mutex_ once at the start to (a) snapshot active_voxels_ and (b)
  // pre-fetch only the visibility timestamps we'll need this frame.
  // Previously each observation took mutex_ separately; under contention
  // with updatePlaceSummaries' snapshot copy that turned a constant-time
  // per-obs cost into ms-scale stalls (avg=2000us/obs in the long run).
  GlobalIndexSet visible_voxels;
  GlobalIndexMap<double> visibility_view;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    visible_voxels = active_voxels_;
    for (const auto& obs : frame.observations) {
      if (!visible_voxels.count(obs.index)) {
        continue;
      }
      auto it = last_visibility_update_s_.find(obs.index);
      if (it != last_visibility_update_s_.end()) {
        visibility_view[obs.index] = it->second;
      }
    }
  }

  // Ingest = folding this frame's detections into their cells. Timed as one
  // per-frame block (the real-time-relevant unit), not per detection.
  const auto t_ingest_start = std::chrono::steady_clock::now();
  size_t used = 0;
  for (const auto& obs : frame.observations) {
    if (!visible_voxels.count(obs.index)) {
      continue;
    }

    const GlobalIndex matched_index = obs.index;

    // visibility_view doubles as the in-frame write log: same semantics
    // as the previous per-obs writes (a cell observed twice in one frame
    // sees its updated value on the second hit).
    double delta_t_visible = 0.0;
    auto it = visibility_view.find(matched_index);
    if (it != visibility_view.end() && frame.timestamp_s >= it->second) {
      delta_t_visible = frame.timestamp_s - it->second;
    }
    visibility_view[matched_index] = frame.timestamp_s;

    flow_map_.addObservation(matched_index, obs.theta, obs.rho, frame.timestamp_s,
                             delta_t_visible, obs.track_id);
    ++used;
  }

  // Apply the in-frame updates back to the global map under one lock.
  // max() merge so we never regress a timestamp written by another thread
  // (e.g., runDeformationAlignment's remap) since the read snapshot.
  if (!visibility_view.empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [idx, ts] : visibility_view) {
      auto& current = last_visibility_update_s_[idx];
      if (ts > current) {
        current = ts;
      }
    }
  }

  // --- Presence exposure ---
  // Accumulate this frame's visible time into every visible cell's presence
  // (Poisson) rate denominator, so the rate is events per unit *visible* time
  // rather than per inter-detection gap. Runs after ingest so cells created
  // this frame are included; never-observed voxels have no cell and are skipped.
  if (prev_frame_timestamp_s_ >= 0.0) {
    const double exposure_dt = frame.timestamp_s - prev_frame_timestamp_s_;
    if (exposure_dt > 0.0) {
      // Clamp so a long inter-frame gap (e.g. a dropped span) cannot dominate
      // the denominator; normal ~10 Hz frames are far below this.
      constexpr double kMaxExposureDtS = 1.0;
      flow_map_.addPoissonExposure(visible_voxels, std::min(exposure_dt, kMaxExposureDtS),
                                   frame.timestamp_s);
    }
  }
  prev_frame_timestamp_s_ = frame.timestamp_s;

  // Coupling pass begins here; the ingest block ends at this instant.
  const auto t_coupling_start = std::chrono::steady_clock::now();
  bool coupling_ran = false;

  // --- Edge coupling update ---
  // Pair each used observation with adjacent observations occurring within the
  // coherence window Δt_coh and update the EdgeCouplingMap with their
  // per-detection responsibilities (the soft assignment of heading/speed to the
  // cardinal slots). Pairing spans nearby frames, not only the current one, so
  // near-simultaneous detections in adjacent cells are captured even when they
  // arrive in consecutive frames. The coupling signal is how that assignment
  // co-varies across neighbouring detections, so the per-observation
  // responsibility is used rather than the cell's time-averaged mixing weights.
  if (edge_coupling_map_) {
    const int connectivity = config.flow.edge_coupling_connectivity;
    const double dt_coh = config.flow.edge_coupling_delta_t_coh;
    const double now_s = frame.timestamp_s;

    // Drop buffered observations that have fallen outside the coherence window.
    while (!coupling_window_.empty() &&
           now_s - coupling_window_.front().t_seconds > dt_coh) {
      coupling_window_.pop_front();
    }

    const auto adjacent = [connectivity](const GlobalIndex& ia, const GlobalIndex& ib) {
      const int dx = std::abs(ia.x() - ib.x());
      const int dy = std::abs(ia.y() - ib.y());
      const int dz = std::abs(ia.z() - ib.z());
      return (connectivity == 6) ? (dx + dy + dz == 1) : (std::max({dx, dy, dz}) == 1);
    };

    // Current frame's used observations with their per-detection responsibility.
    std::vector<CouplingSample> current;
    current.reserve(used);
    for (const auto& obs : frame.observations) {
      if (!visible_voxels.count(obs.index)) {
        continue;
      }
      auto resp = flow_map_.responsibilities(obs.index, obs.theta, obs.rho);
      if (!resp.empty()) {
        current.push_back({obs.index, std::move(resp), now_s});
      }
    }
    if (!current.empty()) {
      coupling_ran = true;
    }

    // Pairs within the current frame (each unordered pair once).
    for (size_t a = 0; a < current.size(); ++a) {
      for (size_t b = a + 1; b < current.size(); ++b) {
        if (adjacent(current[a].index, current[b].index)) {
          edge_coupling_map_->update(current[a].index, current[a].resp,
                                     current[b].index, current[b].resp, now_s);
        }
      }
    }
    // Cross-frame pairs against earlier observations still inside the window.
    // Counted once, when the later observation (this frame) arrives.
    for (const auto& c : current) {
      for (const auto& p : coupling_window_) {
        if (adjacent(c.index, p.index)) {
          edge_coupling_map_->update(c.index, c.resp, p.index, p.resp, now_s);
        }
      }
    }
    // Retain the current frame for pairing against subsequent frames.
    for (auto& c : current) {
      coupling_window_.push_back(std::move(c));
    }
  }

  // Coupling pass ends here.
  const auto t_coupling_end = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(summaries_mutex_);
    last_frame_observations_ = frame.observations.size();
    last_frame_used_observations_ = used;
  }

  if (timing_enabled) {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    const auto elapsed_us =
        duration_cast<microseconds>(t_coupling_end - start).count();
    const auto ingest_us =
        duration_cast<microseconds>(t_coupling_start - t_ingest_start).count();
    const auto coupling_us =
        duration_cast<microseconds>(t_coupling_end - t_coupling_start).count();
    {
      // Per-frame accumulators; compared against the keyframe budget downstream.
      std::lock_guard<std::mutex> lk(timing_mutex_);
      module_timing_.process_frame_count++;
      module_timing_.process_frame_total_us += static_cast<uint64_t>(elapsed_us);
      module_timing_.process_frame_samples_us.push_back(static_cast<uint32_t>(elapsed_us));
      module_timing_.ingest_total_us += static_cast<uint64_t>(ingest_us);
      module_timing_.ingest_samples_us.push_back(static_cast<uint32_t>(ingest_us));
      if (coupling_ran) {
        module_timing_.coupling_count++;
        module_timing_.coupling_total_us += static_cast<uint64_t>(coupling_us);
        module_timing_.coupling_samples_us.push_back(static_cast<uint32_t>(coupling_us));
      }
    }
    const double per_obs_us = frame.observations.empty()
                                  ? 0.0
                                  : static_cast<double>(elapsed_us) /
                                        static_cast<double>(frame.observations.size());
    const size_t edge_count =
        edge_coupling_map_ ? edge_coupling_map_->edgeCount() : 0u;
    LOG_EVERY_N(INFO, 2000)
        << "[FlowTiming] processFrame total=" << elapsed_us
        << " us (ingest=" << ingest_us << ", coupling=" << coupling_us
        << "), observations=" << frame.observations.size() << ", used=" << used
        << ", visible_voxels=" << visible_voxels.size() << ", avg=" << per_obs_us
        << " us/obs, edge_coupling_edges=" << edge_count;
  }
}

void FlowTemporalModule::updatePlaceSummariesExternal(double timestamp_s,
                                                      DynamicSceneGraph& graph) {
  // Thin public entry point: reuse the exact aggregation the backend path runs,
  // just over a caller-supplied graph. place_summaries_ is populated as usual.
  updatePlaceSummaries(timestamp_s, graph);
}

std::vector<FlowMap::CellCount> FlowTemporalModule::flowCells() const {
  return flow_map_.cellObservationCounts();
}

void FlowTemporalModule::updatePlaceSummaries(double timestamp_s,
                                              DynamicSceneGraph& graph) {
  const bool timing_enabled = config.flow.enable_timing;
  const auto start = std::chrono::steady_clock::now();

  // Runs on the backend thread (called from handleBackendUpdate). The
  // members read below are mutated by other threads (processFrame /
  // runDeformationAlignment / handleMapUpdate). voxel_grid_ + voxel_size_
  // are tiny so we copy them up-front. last_visibility_update_s_ can be
  // hundreds of thousands of entries in long runs, so a full copy here
  // was holding mutex_ long enough to stall processFrame on every
  // observation. Defer it: pre-fetch only the entries we'll actually
  // look up after bucket_by_place is built (~thousands, not 150K+).
  GlobalIndexMap<double> last_visibility_local;
  std::unique_ptr<spatial_hash::IndexGrid> voxel_grid_local;
  float voxel_size_local = 0.0f;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!voxel_grid_) {
      return;
    }
    voxel_grid_local = std::make_unique<spatial_hash::IndexGrid>(*voxel_grid_);
    voxel_size_local = voxel_size_;
  }

  std::lock_guard<std::mutex> summaries_lock(summaries_mutex_);
  // place_summaries_ is per-iteration: it holds the summaries computed this
  // call, used by annotateFlowEdges below. Voxels are the source of truth
  // for flow state (their flow_map_ persists across iterations); each
  // backend update we re-derive every place's summary from its voxels.
  //
  // Snapshot the previous iteration's summaries so we can quantify arrow
  // stickiness (lost / gained / kept counts and dominant-theta jumps for
  // kept places) below. The complaint is "arrows flicker on/off" — that's
  // exactly the lost/gained transition we're measuring.
  const auto prev_place_summaries = place_summaries_;
  place_summaries_.clear();
  place_nodes_seen_ = 0;
  place_nodes_with_support_ = 0;
  place_nodes_with_components_ = 0;
  place_summaries_places_count_ = 0;

  auto write_summary_to_metadata = [&](nlohmann::json& metadata,
                                       const PlaceFlowSummary& s) {
    metadata["flow_occupancy"] = s.occupancy;
    metadata["flow_dwell_s"] = s.dwell_s;
    metadata["flow_extent_m"] = s.extent_m;
    metadata["flow_mean_rho"] = s.mean_rho;
    metadata["flow_confidence"] = s.confidence;
    metadata["flow_support"] = s.support;
    metadata["flow_t_latest"] = s.t_latest;
    metadata["flow_dominant_theta"] = s.dominant_theta;
    metadata["flow_dominant_rho"] = s.dominant_rho;
    metadata["flow_num_components"] = s.components.size();
    nlohmann::json components_json = nlohmann::json::array();
    for (const auto& c : s.components) {
      components_json.push_back({{"theta", c.theta},
                                 {"rho", c.rho},
                                 {"sigma_theta", c.sigma_theta},
                                 {"sigma_rho", c.sigma_rho},
                                 {"sigma_cross", c.sigma_cross},
                                 {"weight", c.weight},
                                 {"voxel_count", c.voxel_count},
                                 {"R", c.R}});
    }
    metadata["flow_components"] = components_json;
  };

  auto erase_flow_metadata = [](nlohmann::json& metadata) {
    metadata.erase("flow_occupancy");
    metadata.erase("flow_dwell_s");
    metadata.erase("flow_extent_m");
    metadata.erase("flow_mean_rho");
    metadata.erase("flow_confidence");
    metadata.erase("flow_support");
    metadata.erase("flow_t_latest");
    metadata.erase("flow_dominant_theta");
    metadata.erase("flow_dominant_rho");
    metadata.erase("flow_num_components");
    metadata.erase("flow_components");
  };

  auto update_layer = [&](LayerId layer_id,
                          auto support_getter,
                          size_t& nodes_seen,
                          size_t& nodes_with_support,
                          size_t& nodes_with_components,
                          size_t& place_summary_count) {
    if (!graph.hasLayer(layer_id)) {
      return;
    }

    const auto& layer = graph.getLayer(layer_id);
    int node_count = 0;
    for (const auto& id_node_pair : layer.nodes()) {
      if (config.place_summary_max_nodes > 0 &&
          node_count >= config.place_summary_max_nodes) {
        break;
      }
      ++node_count;
      ++nodes_seen;

      const auto node_id = id_node_pair.first;
      const auto& node = *id_node_pair.second;
      const auto support_indices = support_getter(node);
      if (!support_indices.empty()) {
        ++nodes_with_support;
      }

      struct AggSlot {
        double weight_sum = 0.0;
        double rho_sum = 0.0;
        double cos_sum = 0.0;
        double sin_sum = 0.0;
        double sigma_theta_sum = 0.0;
        double sigma_rho_sum = 0.0;
        double sigma_cross_sum = 0.0;
        double ref_theta = 0.0;
        int voxel_count = 0;
        double t_latest = 0.0;
      };

      std::vector<AggSlot> agg_slots;
      GlobalIndexSet matched_support_cells;
      // Every cross-cell quantity below is weighted by the cell's occupancy,
      // never by its presence probability: the probability saturates toward
      // one at every cell as the prediction horizon grows, which would erase
      // the differences between cells and drive the aggregate mixture toward
      // uniform. The occupancy carries no horizon and stays proportional to
      // the traffic each cell sees.
      double occupancy_sum = 0.0;
      double confidence_weighted_sum = 0.0;
      double speed_weighted_sum = 0.0;
      double speed_weight_sum = 0.0;
      int support = 0;
      double t_latest = 0.0;
      // Support-cell centres in the plane, kept so the region's extent can be
      // measured along the dominant heading once that heading is known.
      std::vector<Eigen::Vector2d> support_positions_2d;
      support_positions_2d.reserve(support_indices.size());

      // Fixed-component regime: K=8 cardinal slots indexed identically across
      // all cells. Aggregate by component index, no heading-matching needed.
      auto find_or_create_slot = [&](size_t index_hint, double /*theta*/) -> size_t {
        if (index_hint >= agg_slots.size()) {
          agg_slots.resize(index_hint + 1);
        }
        return index_hint;
      };

      for (const auto& index : support_indices) {
        // A horizon of zero: nothing read off the snapshot below depends on
        // one. The occupancy and the crossing time are horizon-free, and the
        // node stores them so a reader can pick its own horizon later.
        const auto snapshot = flow_map_.tryGetCellSnapshot(index, timestamp_s, 0.0);
        if (!snapshot || snapshot->flag == CellFlag::FALLBACK ||
            snapshot->components.empty()) {
          continue;
        }

        if (matched_support_cells.count(index)) {
          continue;
        }
        matched_support_cells.insert(index);

        const auto& snapshot_ref = *snapshot;
        const double cell_occupancy = std::max(0.0, snapshot_ref.l_hat);

        occupancy_sum += cell_occupancy;
        confidence_weighted_sum += cell_occupancy * snapshot_ref.confidence;
        // The cell's dwell is its size over its pooled mean speed, so the
        // speed is recoverable from it. Cells without a speed estimate carry a
        // non-positive dwell and sit out the region's mean.
        if (snapshot_ref.dwell_s > 0.0 && voxel_size_local > 0.0f) {
          const double cell_speed =
              static_cast<double>(voxel_size_local) / snapshot_ref.dwell_s;
          speed_weighted_sum += cell_occupancy * cell_speed;
          speed_weight_sum += cell_occupancy;
        }

        if (voxel_grid_local) {
          const auto centre = voxel_grid_local->toPoint(index);
          support_positions_2d.emplace_back(static_cast<double>(centre.x()),
                                            static_cast<double>(centre.y()));
        }

        double cell_t_latest = 0.0;
        const auto last_iter = last_visibility_local.find(index);
        if (last_iter != last_visibility_local.end()) {
          cell_t_latest = last_iter->second;
        }
        if (cell_t_latest > t_latest) {
          t_latest = cell_t_latest;
        }

        for (size_t i = 0; i < snapshot_ref.components.size(); ++i) {
          const auto& component = snapshot_ref.components[i];
          const double weighted = component.weight * cell_occupancy;
          if (weighted <= 0.0) {
            continue;
          }

          const size_t slot = find_or_create_slot(i, component.theta);
          agg_slots[slot].weight_sum += weighted;
          agg_slots[slot].rho_sum += weighted * component.rho;
          agg_slots[slot].cos_sum += weighted * std::cos(component.theta);
          agg_slots[slot].sin_sum += weighted * std::sin(component.theta);
          agg_slots[slot].sigma_theta_sum += weighted * component.sigma_theta;
          agg_slots[slot].sigma_rho_sum += weighted * component.sigma_rho;
          agg_slots[slot].sigma_cross_sum += weighted * component.sigma_cross;
          agg_slots[slot].voxel_count += 1;
          if (cell_t_latest > agg_slots[slot].t_latest) {
            agg_slots[slot].t_latest = cell_t_latest;
          }
        }

        ++support;
      }

      auto attrs_clone = node.attributes().clone();
      auto& metadata = attrs_clone->metadata;
      if (!metadata.is_object()) {
        metadata = nlohmann::json::object();
      }

      if (support == 0) {
        // No support voxels with valid snapshots at this timestamp.
        // Erase any stale flow metadata; the next backend iteration will
        // re-evaluate from current voxel state.
        erase_flow_metadata(metadata);
        graph.setNodeAttributes(node_id, std::move(attrs_clone));
        continue;
      }

      constexpr double kOccupancyEps = 1.0e-9;
      PlaceFlowSummary summary;
      // The region's occupancy is the sum of its cells' occupancies, the joint
      // intensity being additive over the region. It is emphatically not the
      // union of the cells' independent presences, which counts one target
      // once per cell it transits.
      summary.occupancy = occupancy_sum;
      summary.mean_rho =
          speed_weight_sum > kOccupancyEps ? speed_weighted_sum / speed_weight_sum : 0.0;
      summary.confidence = occupancy_sum > kOccupancyEps
                               ? confidence_weighted_sum / occupancy_sum
                               : 0.0;
      summary.support = support;
      summary.t_latest = t_latest;

      double total_component_weight = 0.0;
      for (const auto& slot : agg_slots) {
        total_component_weight += slot.weight_sum;
      }

      if (total_component_weight > 0.0) {
        for (size_t i = 0; i < agg_slots.size(); ++i) {
          const auto& slot = agg_slots[i];
          if (slot.weight_sum <= 0.0) {
            continue;
          }

          PlaceFlowComponentSummary component_summary;
          component_summary.weight = slot.weight_sum / total_component_weight;
          component_summary.rho = slot.rho_sum / slot.weight_sum;
          // Circular mean heading: the angle of the weight-summed unit vectors
          // (sum of w*cos, w*sin), which averages angles correctly across the
          // 0/2pi wrap.
          component_summary.theta = std::atan2(slot.sin_sum, slot.cos_sum);
          component_summary.sigma_theta = slot.sigma_theta_sum / slot.weight_sum;
          component_summary.sigma_rho = slot.sigma_rho_sum / slot.weight_sum;
          component_summary.sigma_cross = slot.sigma_cross_sum / slot.weight_sum;
          component_summary.voxel_count = slot.voxel_count;
          // R is the circular concentration: the resultant length of those unit
          // vectors normalised by total weight. It lies in [0, 1], where values
          // near 1 mean the contributing headings strongly agree and values
          // near 0 mean they cancel out.
          const double resultant =
              std::sqrt(slot.cos_sum * slot.cos_sum + slot.sin_sum * slot.sin_sum);
          component_summary.R = resultant / slot.weight_sum;

          if (component_summary.weight < config.component_min_weight) {
            continue;
          }
          if (component_summary.voxel_count < config.component_min_voxel_count) {
            continue;
          }

          summary.components.push_back(component_summary);
        }
      }

      std::sort(summary.components.begin(),
                summary.components.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.weight > rhs.weight; });

      if (static_cast<int>(summary.components.size()) > config.max_components_per_place) {
        summary.components.resize(static_cast<size_t>(config.max_components_per_place));
      }

      if (!summary.components.empty()) {
        double weight_sum = 0.0;
        double rho_sum = 0.0;
        double cos_sum = 0.0;
        double sin_sum = 0.0;
        for (const auto& component : summary.components) {
          const double weight = std::max(0.0, component.weight);
          if (weight <= 0.0) {
            continue;
          }

          weight_sum += weight;
          rho_sum += weight * component.rho;
          cos_sum += weight * std::cos(component.theta);
          sin_sum += weight * std::sin(component.theta);
        }

        if (weight_sum > 0.0) {
          summary.dominant_theta = std::atan2(sin_sum, cos_sum);
          summary.dominant_rho = rho_sum / weight_sum;
        }
      }

      // The region's extent is the distance a target crossing it actually
      // covers, so it is measured along the prevailing heading rather than
      // across the widest span of the support set: a long corridor node is as
      // long as people walk it, not as wide as it is at its widest. The span
      // is between cell centres, so one cell of width is added back to reach
      // the region's true edges, and a single-cell region reduces to the cell.
      if (!support_positions_2d.empty()) {
        const Eigen::Vector2d heading(std::cos(summary.dominant_theta),
                                      std::sin(summary.dominant_theta));
        double min_projection = std::numeric_limits<double>::max();
        double max_projection = std::numeric_limits<double>::lowest();
        for (const auto& position : support_positions_2d) {
          const double projection = heading.dot(position);
          min_projection = std::min(min_projection, projection);
          max_projection = std::max(max_projection, projection);
        }
        summary.extent_m = (max_projection - min_projection) +
                           static_cast<double>(voxel_size_local);
      }

      // The region's crossing time closes the loop between its extent and the
      // speed at which it is traversed. Without a speed estimate it stays
      // non-positive and the presence readout drops its arrival term, exactly
      // as it does per cell.
      summary.dwell_s = (summary.mean_rho > 0.0 && summary.extent_m > 0.0)
                            ? summary.extent_m / summary.mean_rho
                            : -1.0;

      write_summary_to_metadata(metadata, summary);

      place_summaries_[node_id] = summary;
      ++place_summary_count;
      if (!summary.components.empty()) {
        ++nodes_with_components;
      }
      graph.setNodeAttributes(node_id, std::move(attrs_clone));
    }
  };

  // Voronoi assignment of flow voxels to place nodes. We build a KDTree
  // over place positions (in voxel-index coordinates) and run 1-NN per
  // flow cell. Acceptance is per-place: a voxel is accepted into place P
  // iff its distance to P does not exceed P's GVD free-space radius
  // (PlaceNodeAttributes::distance), with a small voxel-quantization
  // slack and a fallback to `place_support_max_distance_m` when GVD
  // distance is missing or non-positive. The fixed cap is no longer the
  // primary criterion: tight doorways stay tight (small distance), open
  // rooms accept far voxels (large distance), and the assignment respects
  // the actual free-space topology Hydra has already computed.
  std::unordered_map<NodeId, GlobalIndexSet> bucket_by_place;
  if (graph.hasLayer(DsgLayers::PLACES)) {
    const auto& places_layer = graph.getLayer(DsgLayers::PLACES);
    GlobalIndices place_indices;
    std::vector<NodeId> place_node_ids;
    std::vector<int64_t> place_max_sq_voxels;  // per-place sq-radius gate, voxel² units
    std::vector<double> place_radius_m;        // per-place radius in metres (for log only)
    place_indices.reserve(places_layer.numNodes());
    place_node_ids.reserve(places_layer.numNodes());
    place_max_sq_voxels.reserve(places_layer.numNodes());
    place_radius_m.reserve(places_layer.numNodes());

    const float voxel_size = voxel_size_local > 0.0f ? voxel_size_local : 1.0f;
    // The kdtree compares floored integer indices, not true world centres.
    // Add one voxel of slack on the radius (squared in voxel units) so a
    // voxel whose centre is exactly on the radius isn't excluded by
    // quantisation drift up to ~half-voxel-per-axis.
    constexpr double kQuantisationSlackVoxels = 1.0;
    size_t fallback_count = 0;

    for (const auto& [node_id, node_ptr] : places_layer.nodes()) {
      const auto& attrs = node_ptr->attributes<PlaceNodeAttributes>();
      place_indices.push_back(
          voxel_grid_local->toIndex(attrs.position.cast<float>()));
      place_node_ids.push_back(node_id);

      double radius_m;
      if (attrs.distance > 0.0) {
        radius_m = attrs.distance;
      } else {
        radius_m = static_cast<double>(config.place_support_max_distance_m);
        ++fallback_count;
      }
      place_radius_m.push_back(radius_m);

      const double radius_v =
          radius_m / static_cast<double>(voxel_size) + kQuantisationSlackVoxels;
      const int64_t r_int = static_cast<int64_t>(std::ceil(radius_v));
      place_max_sq_voxels.push_back(r_int * r_int);
    }

    if (!place_indices.empty()) {
      places::NearestVoxelFinder finder(place_indices);
      const auto cell_indices = flow_map_.cellIndices();
      const size_t cells_total = cell_indices.size();
      size_t cells_assigned = 0;
      size_t cells_rejected_by_radius = 0;
      for (const auto& cell_index : cell_indices) {
        bool assigned_here = false;
        bool rejected_here = false;
        finder.find(cell_index,
                    1,
                    [&](const GlobalIndex&,
                        size_t nn_idx,
                        int64_t sq_distance) {
                      if (sq_distance > place_max_sq_voxels[nn_idx]) {
                        rejected_here = true;
                        return;
                      }
                      bucket_by_place[place_node_ids[nn_idx]].insert(cell_index);
                      assigned_here = true;
                    });
        if (assigned_here) {
          ++cells_assigned;
        } else if (rejected_here) {
          ++cells_rejected_by_radius;
        }
      }

      // Per-place radius statistics so the user can see the new
      // adaptive-radius distribution at a glance and verify it's not
      // collapsing back to the fallback.
      double r_min = std::numeric_limits<double>::infinity();
      double r_max = 0.0;
      double r_sum = 0.0;
      std::vector<double> r_sorted = place_radius_m;
      std::sort(r_sorted.begin(), r_sorted.end());
      for (const auto r : place_radius_m) {
        r_min = std::min(r_min, r);
        r_max = std::max(r_max, r);
        r_sum += r;
      }
      const double r_mean =
          place_radius_m.empty() ? 0.0 : r_sum / static_cast<double>(place_radius_m.size());
      const double r_median =
          r_sorted.empty() ? 0.0 : r_sorted[r_sorted.size() / 2];

      LOG(INFO) << "[PlaceArrows.bucket] places=" << place_indices.size()
                << " cells=" << cells_total
                << " cells_assigned=" << cells_assigned
                << " cells_rejected_by_radius=" << cells_rejected_by_radius
                << " cells_unassigned=" << (cells_total - cells_assigned)
                << " gvd_radius_m: min=" << (place_radius_m.empty() ? 0.0 : r_min)
                << " median=" << r_median
                << " mean=" << r_mean
                << " max=" << r_max
                << " gvd_fallback_count=" << fallback_count
                << " fallback_radius_m=" << config.place_support_max_distance_m
                << " (acceptance is now per-place: dist <= P.distance + 1 voxel; "
                   "fallback only when P.distance <= 0).";

      // Restrict the coupling neighbourhood to traversable adjacency. A voxel
      // pair is coupled only when both voxels lie in the support of one place
      // node, or in the supports of two place nodes joined by a places-layer
      // edge. Because place support is gated by each node's GVD free-space
      // radius, a pair separated by geometry falls into supports that are not
      // topologically adjacent, and is dropped. Index adjacency alone admits
      // such pairs: two voxels can share a face while an obstacle thinner than
      // one voxel sits between them, and no continuous agent motion carries
      // flow across it. The prune runs here, on the backend thread, where the
      // graph and the support buckets already exist, so the per-frame path is
      // untouched.
      GlobalIndexMap<NodeId> place_of_voxel;
      for (const auto& [place_id, cells] : bucket_by_place) {
        for (const auto& cell_idx : cells) {
          place_of_voxel.emplace(cell_idx, place_id);
        }
      }

      // Publish the same relation to the flow map, which evidence sharing reads
      // at query time. Sharing takes it unnarrowed: a cell draws on every voxel
      // its place supports and on those of the places an edge joins to it, so
      // the lenders are the cells an agent can reach it from, however far across
      // the region they sit. The coupling prune below narrows the same relation
      // to face-sharing pairs, which is the contiguity its dependence needs.
      {
        FlowMap::SharingTopology topology;
        topology.place_of_voxel = place_of_voxel;
        for (const auto& [place_id, cells] : bucket_by_place) {
          auto& support = topology.support_of_place[place_id];
          support.reserve(cells.size());
          for (const auto& cell_idx : cells) {
            support.push_back(cell_idx);
          }
          const auto node = places_layer.findNode(place_id);
          if (!node) {
            continue;
          }
          auto& siblings = topology.siblings_of_place[place_id];
          siblings.reserve(node->siblings().size());
          for (const auto sibling : node->siblings()) {
            siblings.push_back(sibling);
          }
        }
        // How far out a voxel may draw is bounded by its place's clearance, the
        // same bound that sets the support set. Expressing that bound in edges
        // rather than metres makes it independent of how coarse the layer is: a
        // layer whose nodes sit about a clearance apart is covered in one step,
        // while a layer authored at voxel resolution needs several to reach the
        // same distance. The spacing is measured from the layer's own edges.
        double spacing_m = 0.0;
        {
          std::vector<double> edge_lengths;
          edge_lengths.reserve(places_layer.numEdges());
          for (const auto& [edge_id, edge] : places_layer.edges()) {
            const auto source = places_layer.findNode(edge.source);
            const auto target = places_layer.findNode(edge.target);
            if (!source || !target) {
              continue;
            }
            edge_lengths.push_back(
                (source->attributes<PlaceNodeAttributes>().position -
                 target->attributes<PlaceNodeAttributes>().position)
                    .norm());
          }
          if (!edge_lengths.empty()) {
            const size_t mid = edge_lengths.size() / 2;
            std::nth_element(edge_lengths.begin(), edge_lengths.begin() + mid,
                             edge_lengths.end());
            spacing_m = edge_lengths[mid];
          }
        }
        // Below one voxel the spacing carries no information, so fall back to the
        // voxel size and let a single step stand for the whole neighbourhood.
        const double step_m = spacing_m > static_cast<double>(voxel_size_local)
                                  ? spacing_m
                                  : static_cast<double>(voxel_size_local);
        constexpr int kMaxHops = 8;  // bounds the walk on a fine layer
        int hops_min = kMaxHops;
        int hops_max = 1;
        for (size_t i = 0; i < place_node_ids.size(); ++i) {
          const int hops = std::clamp(
              static_cast<int>(std::lround(place_radius_m[i] / step_m)), 1, kMaxHops);
          topology.hops_of_place[place_node_ids[i]] = hops;
          hops_min = std::min(hops_min, hops);
          hops_max = std::max(hops_max, hops);
        }

        const size_t place_count = topology.support_of_place.size();
        const size_t voxel_count = topology.place_of_voxel.size();
        flow_map_.setSharingTopology(std::move(topology));
        LOG(INFO) << "[EvidenceSharing.topology] " << voxel_count << " voxels over "
                  << place_count << " places, node spacing " << spacing_m
                  << " m, clearance reached in " << hops_min << "-" << hops_max
                  << " edges";
      }

      if (edge_coupling_map_) {
        const auto traversably_adjacent = [&](const GlobalIndex& i,
                                              const GlobalIndex& j) {
          const auto it_i = place_of_voxel.find(i);
          const auto it_j = place_of_voxel.find(j);
          // A voxel with no place support yet (newly observed, or outside every
          // node's free-space radius) carries no topology to test, so the edge
          // is retained rather than silently discarded; it is re-tested on the
          // next pass once the graph covers it.
          if (it_i == place_of_voxel.end() || it_j == place_of_voxel.end()) {
            return true;
          }
          if (it_i->second == it_j->second) {
            return true;
          }
          const auto node_i = places_layer.findNode(it_i->second);
          if (!node_i) {
            return true;
          }
          return node_i->siblings().count(it_j->second) > 0;
        };
        const auto report = edge_coupling_map_->pruneEdges(traversably_adjacent);
        if (report.edges_dropped > 0) {
          const size_t edges_total = report.edges_dropped + report.edges_kept;
          const int64_t paired_total = report.paired_dropped + report.paired_kept;
          LOG(INFO) << "[EdgeCoupling.prune] dropped " << report.edges_dropped
                    << "/" << edges_total << " edges ("
                    << (100.0 * report.edges_dropped / static_cast<double>(edges_total))
                    << "%) not traversably adjacent, holding "
                    << report.paired_dropped << "/" << paired_total
                    << " paired observations ("
                    << (100.0 * static_cast<double>(report.paired_dropped) /
                        static_cast<double>(std::max<int64_t>(1, paired_total)))
                    << "%)";
        }
      }
    }
  }

  // Targeted pre-fetch: fill last_visibility_local with entries only for
  // the cells we'll iterate inside update_layer. ~3K cells (cells_assigned
  // in [PlaceArrows.bucket]) vs the 150K+ full visibility map — ~50x
  // cheaper, and processFrame only blocks on this short lock instead of a
  // full-map copy.
  if (!bucket_by_place.empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [place_id, cells] : bucket_by_place) {
      for (const auto& cell_idx : cells) {
        auto vit = last_visibility_update_s_.find(cell_idx);
        if (vit != last_visibility_update_s_.end()) {
          last_visibility_local[cell_idx] = vit->second;
        }
      }
    }
  }

  update_layer(DsgLayers::PLACES,
               [&](const SceneGraphNode& node) {
                 const auto it = bucket_by_place.find(node.id);
                 if (it == bucket_by_place.end()) {
                   return GlobalIndexSet{};
                 }
                 return it->second;
               },
               place_nodes_seen_,
               place_nodes_with_support_,
               place_nodes_with_components_,
               place_summaries_places_count_);

  // Arrow stickiness: compare to the previous iteration. A place "lost"
  // arrows if it had components last call but doesn't now (visualizer will
  // clear that node's arrows); "gained" if newly populated; "kept" if both
  // call had components — those should be stable, with small dominant-theta
  // deltas. Big delta-theta in the kept set is a sign that voxel support
  // changed enough to flip which component dominates.
  size_t lost = 0, gained = 0, kept = 0;
  size_t kept_dtheta_gt_45deg = 0;
  size_t kept_dtheta_gt_90deg = 0;
  double sum_abs_dtheta = 0.0;
  double max_abs_dtheta = 0.0;
  for (const auto& [node_id, prev] : prev_place_summaries) {
    const bool prev_had = !prev.components.empty();
    auto it = place_summaries_.find(node_id);
    const bool now_has = it != place_summaries_.end() && !it->second.components.empty();
    if (prev_had && !now_has) {
      ++lost;
    } else if (prev_had && now_has) {
      ++kept;
      double d = it->second.dominant_theta - prev.dominant_theta;
      while (d > M_PI) d -= 2.0 * M_PI;
      while (d < -M_PI) d += 2.0 * M_PI;
      const double abs_d = std::abs(d);
      sum_abs_dtheta += abs_d;
      if (abs_d > max_abs_dtheta) max_abs_dtheta = abs_d;
      if (abs_d > M_PI / 4.0) ++kept_dtheta_gt_45deg;
      if (abs_d > M_PI / 2.0) ++kept_dtheta_gt_90deg;
    }
  }
  for (const auto& [node_id, summary] : place_summaries_) {
    if (summary.components.empty()) continue;
    auto it = prev_place_summaries.find(node_id);
    const bool prev_had = it != prev_place_summaries.end() && !it->second.components.empty();
    if (!prev_had) {
      ++gained;
    }
  }
  const double mean_abs_dtheta =
      kept > 0 ? sum_abs_dtheta / static_cast<double>(kept) : 0.0;
  LOG(INFO) << "[PlaceArrows.lifecycle] places_with_components="
            << place_nodes_with_components_ << " lost=" << lost
            << " gained=" << gained << " kept=" << kept
            << " kept_dtheta>45deg=" << kept_dtheta_gt_45deg
            << " kept_dtheta>90deg=" << kept_dtheta_gt_90deg
            << " mean|dtheta|_kept_rad=" << mean_abs_dtheta
            << " max|dtheta|_kept_rad=" << max_abs_dtheta
            << " (high lost+gained ≈ flicker; high kept_dtheta = arrows reorient between calls).";

  annotateFlowEdges(graph, DsgLayers::PLACES);

  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    {
      // Per-event accumulator (runs on the backend thread).
      std::lock_guard<std::mutex> lk(timing_mutex_);
      module_timing_.place_summaries_count++;
      module_timing_.place_summaries_total_us += static_cast<uint64_t>(elapsed_us);
      module_timing_.place_summaries_samples_us.push_back(static_cast<uint32_t>(elapsed_us));
    }
    const double per_node_us = place_nodes_seen_ == 0
                                   ? 0.0
                                   : static_cast<double>(elapsed_us) /
                                         static_cast<double>(place_nodes_seen_);
    LOG_EVERY_N(INFO, 50) << "[FlowTiming] updatePlaceSummaries total=" << elapsed_us
              << " us, places: seen=" << place_nodes_seen_
              << " support=" << place_nodes_with_support_
              << " comps=" << place_nodes_with_components_
              << ", avg=" << per_node_us << " us/node";
  }
}

void FlowTemporalModule::annotateFlowEdges(DynamicSceneGraph& graph,
                                           LayerId layer_id) {
  if (!graph.hasLayer(layer_id)) {
    return;
  }

  const auto& layer = graph.getLayer(layer_id);
  for (const auto& [edge_key, edge] : layer.edges()) {
    const auto it_src = place_summaries_.find(edge.source);
    const auto it_tgt = place_summaries_.find(edge.target);
    if (it_src == place_summaries_.end() || it_tgt == place_summaries_.end()) {
      continue;
    }

    const auto& src_summary = it_src->second;
    const auto& tgt_summary = it_tgt->second;

    if (src_summary.components.empty() && tgt_summary.components.empty()) {
      continue;
    }

    const auto& src_pos =
        graph.getNode(edge.source).attributes<PlaceNodeAttributes>().position;
    const auto& tgt_pos =
        graph.getNode(edge.target).attributes<PlaceNodeAttributes>().position;
    const Eigen::Vector3d edge_vec = tgt_pos - src_pos;
    const double edge_length = edge_vec.norm();
    if (edge_length < 1.0e-6) {
      continue;
    }

    const Eigen::Vector2d edge_dir_2d(edge_vec.x(), edge_vec.y());
    const double edge_dir_norm = edge_dir_2d.norm();
    if (edge_dir_norm < 1.0e-6) {
      continue;
    }

    const Eigen::Vector2d edge_unit = edge_dir_2d / edge_dir_norm;

    auto compute_alignment = [&](const PlaceFlowSummary& summary,
                                 double& forward,
                                 double& reverse,
                                 double& avg_speed) {
      forward = 0.0;
      reverse = 0.0;
      avg_speed = 0.0;
      double total_weight = 0.0;

      for (const auto& comp : summary.components) {
        const Eigen::Vector2d flow_dir(std::cos(comp.theta), std::sin(comp.theta));
        const double projection = edge_unit.dot(flow_dir);
        // Scaled by the endpoint's occupancy rather than by a presence
        // probability, so the score stays linear in the traffic the endpoint
        // carries and remains comparable across edges at any horizon.
        const double weighted = comp.weight * summary.occupancy;

        if (projection > 0.0) {
          forward += weighted * projection;
        } else {
          reverse += weighted * (-projection);
        }

        avg_speed += comp.weight * comp.rho;
        total_weight += comp.weight;
      }

      if (total_weight > 0.0) {
        avg_speed /= total_weight;
      }
    };

    double src_fwd = 0.0, src_rev = 0.0, src_speed = 0.0;
    double tgt_fwd = 0.0, tgt_rev = 0.0, tgt_speed = 0.0;
    compute_alignment(src_summary, src_fwd, src_rev, src_speed);
    compute_alignment(tgt_summary, tgt_fwd, tgt_rev, tgt_speed);

    const double forward_flow = 0.5 * (src_fwd + tgt_fwd);
    const double reverse_flow = 0.5 * (src_rev + tgt_rev);
    const double total_flow = forward_flow + reverse_flow;
    const double avg_occupancy = 0.5 * (src_summary.occupancy + tgt_summary.occupancy);
    const double avg_speed = 0.5 * (src_speed + tgt_speed);
    const double bidirectionality = (total_flow > 1.0e-9)
                                        ? std::min(forward_flow, reverse_flow) / total_flow
                                        : 0.5;

    // forward_flow / reverse_flow are oriented along edge.source -> edge.target.
    auto& edge_metadata = edge.attributes().metadata;
    if (!edge_metadata.is_object()) {
      edge_metadata = nlohmann::json::object();
    }
    edge_metadata["flow"] = {{"forward_flow", forward_flow},
                             {"reverse_flow", reverse_flow},
                             {"occupancy", avg_occupancy},
                             {"speed", avg_speed},
                             {"bidirectionality", bidirectionality}};
  }
}

}  // namespace kairos
