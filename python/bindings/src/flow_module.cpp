/* -----------------------------------------------------------------------------
 * KAIROS: 4D Scene Graphs for Periodic Flow Dynamics.
 *
 * FlowModule: Python handle on the flow/temporal module running in Hydra's pipeline.
 * -------------------------------------------------------------------------- */
#include "kairos/bindings/flow_module.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "kairos/archetypes/update_archetypes_functor.h"
#include "kairos/temporal/flow_module_registry.h"
#include "kairos/flow/swgmm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <set>
#include <utility>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_layer.h>
#include <spark_dsg/scene_graph_node.h>
#include <spark_dsg/scene_graph_types.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>

namespace kairos::python {

namespace {

spark_dsg::DynamicSceneGraph::Ptr graphFromBinary(const pybind11::bytes& contents) {
  const auto view = static_cast<std::string_view>(contents);
  return spark_dsg::io::binary::readGraph(reinterpret_cast<const uint8_t*>(view.data()),
                                          view.size());
}

}  // namespace

FlowHandle::FlowHandle(const std::string& name) : name(name) {}

FlowTemporalModule* FlowHandle::module() const { return FlowModuleRegistry::get(name); }

bool FlowHandle::pushDetections(double timestamp_s,
                                    const std::vector<Eigen::Vector3d>& xyz_world,
                                    const std::vector<double>& theta,
                                    const std::vector<double>& rho,
                                    const std::vector<int64_t>& track_ids) {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return false;
  }
  return flow_temporal_->pushDetections(timestamp_s, xyz_world, theta, rho, track_ids);
}

void FlowHandle::setVisibleVoxels(
    float voxel_size_m, const std::vector<Eigen::Vector3d>& xyz_world) {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return;
  }
  flow_temporal_->setVisibleVoxels(voxel_size_m, xyz_world);
}

void FlowHandle::applyDeformationSnapshot(
    double timestamp_s,
    const std::vector<Eigen::Vector3d>& points_before,
    const std::vector<double>& stamps_s,
    const std::vector<Eigen::Vector3d>& points_after,
    const std::vector<double>& yaw_after) {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return;
  }
  flow_temporal_->flush();
  flow_temporal_->applyDeformationSnapshot(timestamp_s, points_before,
                                           stamps_s, points_after, yaw_after);
}

size_t FlowHandle::pendingDetectionFrames() const {
  auto* flow_temporal_ = module();
  return flow_temporal_ ? flow_temporal_->pendingFrames() : 0;
}

void FlowHandle::flushDetections() {
  auto* flow_temporal_ = module();
  if (flow_temporal_) {
    flow_temporal_->flush();
  }
}

std::vector<uint64_t> FlowHandle::setNavigationalLayer(
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<double>& radii_m,
    const std::vector<std::array<size_t, 2>>& edges) {
  auto* flow_temporal_ = module();
  // Build a fresh private PLACES layer from the caller's nodes. The default
  // DynamicSceneGraph ctor already creates layer 3 (PLACES).
  nav_graph_ = std::make_shared<spark_dsg::DynamicSceneGraph>();
  nav_node_ids_.clear();
  const size_t n = std::min(positions.size(), radii_m.size());
  nav_node_ids_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto attrs = std::make_unique<spark_dsg::PlaceNodeAttributes>();
    attrs->position =
        Eigen::Vector3d(positions[i][0], positions[i][1], positions[i][2]);
    attrs->distance = radii_m[i];  // per-node support radius (read at aggregation)
    const spark_dsg::NodeId id = spark_dsg::NodeSymbol('p', i);
    nav_graph_->emplaceNode(spark_dsg::DsgLayers::PLACES, id, std::move(attrs));
    nav_node_ids_.push_back(id);
  }
  // Sibling edges in the PLACES layer. These carry the traversability the
  // coupling neighbourhood is defined over, so a layer built without them
  // leaves every pair of distinct nodes non-adjacent.
  size_t inserted = 0;
  for (const auto& e : edges) {
    if (e[0] >= n || e[1] >= n || e[0] == e[1]) {
      continue;
    }
    if (nav_graph_->insertEdge(nav_node_ids_[e[0]], nav_node_ids_[e[1]])) {
      ++inserted;
    }
  }
  LOG(INFO) << "[NavLayer] nodes=" << n << " edges_requested=" << edges.size()
            << " edges_inserted=" << inserted;
  return {nav_node_ids_.begin(), nav_node_ids_.end()};
}

void FlowHandle::updatePlaceSummaries(double t) {
  auto* flow_temporal_ = module();
  if (flow_temporal_ && nav_graph_) {
    flow_temporal_->updatePlaceSummariesExternal(t, *nav_graph_);
  }
}

std::vector<FlowTemporalModule::PlaceFlowSummary> FlowHandle::placeSummaries()
    const {
  auto* flow_temporal_ = module();
  std::vector<FlowTemporalModule::PlaceFlowSummary> out;
  if (!flow_temporal_) {
    return out;
  }
  const auto by_id = flow_temporal_->placeSummariesCopy();
  out.reserve(nav_node_ids_.size());
  for (const auto id : nav_node_ids_) {
    const auto it = by_id.find(id);
    out.push_back(it == by_id.end() ? FlowTemporalModule::PlaceFlowSummary{}
                                    : it->second);
  }
  return out;
}

std::vector<FlowMap::CellCount> FlowHandle::flowVoxels() const {
  auto* flow_temporal_ = module();
  return flow_temporal_ ? flow_temporal_->flowCells()
                        : std::vector<FlowMap::CellCount>{};
}

std::vector<FlowTemporalModule::ObservationScore> FlowHandle::scoreObservations(
    const std::vector<double>& timestamps_s,
    const std::vector<Eigen::Vector3d>& xyz_world,
    const std::vector<double>& theta,
    const std::vector<double>& rho,
    double horizon_s) const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->scoreObservations(
      timestamps_s, xyz_world, theta, rho, horizon_s);
}

std::vector<FlowMap::PresenceGridEntry> FlowHandle::presenceGrid(
    double t_seconds, const std::vector<double>& horizons_s) const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->presenceGrid(t_seconds, horizons_s);
}

std::vector<FlowMap::FlowGridEntry> FlowHandle::flowGrid(double t_seconds) const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->flowGrid(t_seconds);
}

std::pair<bool, std::string> FlowHandle::saveFlowState(
    const std::string& path) const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {false, "flow temporal module not enabled"};
  }
  std::string err;
  const bool ok = flow_temporal_->saveFlowState(path, &err);
  return {ok, err};
}

std::pair<bool, std::string> FlowHandle::loadFlowState(const std::string& path) {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {false, "flow temporal module not enabled"};
  }
  std::string err;
  const bool ok = flow_temporal_->loadFlowState(path, &err);
  return {ok, err};
}

FlowMap::StateStats FlowHandle::flowStats() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->flowMap().stateStats();
}

FlowMap::TimingStats FlowHandle::flowTiming() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->flowMap().timingStats();
}

FlowTemporalModule::ModuleTiming FlowHandle::moduleTiming() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->moduleTiming();
}

hydra::VolumetricMap::MemoryStats FlowHandle::mapMemoryStats() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->mapMemoryStats();
}

std::vector<FlowMap::StabilityRecord> FlowHandle::flowStability() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->flowMap().stabilityRecords();
}

std::vector<FlowTemporalModule::LcEvent> FlowHandle::lcEvents() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return {};
  }
  return flow_temporal_->lcEventsCopy();
}

std::string FlowHandle::couplingJson() const {
  auto* flow_temporal_ = module();
  if (!flow_temporal_) {
    return "{}";
  }
  return flow_temporal_->couplingJson();
}

namespace flow_module {

using namespace pybind11::literals;
namespace py = pybind11;

void addBindings(pybind11::module_& m) {
  // Handle on a running flow module, found by name. Hydra's pipeline
  // (hydra_python) owns the module; this exposes the flow-specific surface.
  py::class_<FlowHandle>(m, "FlowModule")
      .def(py::init<const std::string&>(), "name"_a = "flow_temporal")
      .def_readonly("name", &FlowHandle::name)
      .def_property_readonly("running",
                             [](const FlowHandle& handle) { return handle.module() != nullptr; })
      .def(
          "push_detections",
          [](FlowHandle& pipeline,
             double timestamp_s,
             const std::vector<Eigen::Vector3d>& xyz_world,
             const std::vector<double>& theta,
             const std::vector<double>& rho,
             const std::vector<int64_t>& track_ids) {
            return pipeline.pushDetections(timestamp_s, xyz_world, theta, rho, track_ids);
          },
          "timestamp_s"_a,
          "xyz_world"_a,
          "theta"_a,
          "rho"_a,
          "track_ids"_a = std::vector<int64_t>{})
      .def(
          "set_visible_voxels",
          [](FlowHandle& pipeline,
             float voxel_size_m,
             const std::vector<Eigen::Vector3d>& xyz_world) {
            pipeline.setVisibleVoxels(voxel_size_m, xyz_world);
          },
          "voxel_size_m"_a,
          "xyz_world"_a)
      .def(
          "apply_deformation_snapshot",
          [](FlowHandle& pipeline,
             double timestamp_s,
             const std::vector<Eigen::Vector3d>& points_before,
             const std::vector<double>& stamps_s,
             const std::vector<Eigen::Vector3d>& points_after,
             const std::vector<double>& yaw_after) {
            pipeline.applyDeformationSnapshot(timestamp_s, points_before,
                                              stamps_s, points_after,
                                              yaw_after);
          },
          "timestamp_s"_a,
          "points_before"_a,
          "stamps_s"_a,
          "points_after"_a,
          "yaw_after"_a = std::vector<double>{})
      .def("pending_detection_frames",
           [](const FlowHandle& pipeline) {
             return pipeline.pendingDetectionFrames();
           })
      .def("flush_detections",
           [](FlowHandle& pipeline) { pipeline.flushDetections(); })
      .def(
          "set_navigational_layer",
          [](FlowHandle& pipeline,
             const std::vector<std::array<double, 3>>& positions,
             const std::vector<double>& radii_m,
             const std::vector<std::array<size_t, 2>>& edges) {
            return pipeline.setNavigationalLayer(positions, radii_m, edges);
          },
          "positions"_a, "radii_m"_a, "edges"_a = std::vector<std::array<size_t, 2>>{})
      .def(
          "update_place_summaries",
          [](FlowHandle& pipeline, double t) {
            pipeline.updatePlaceSummaries(t);
          },
          "t"_a)
      .def("place_summaries",
           [](const FlowHandle& pipeline, double presence_horizon_s) {
             const auto sums = pipeline.placeSummaries();
             const size_t N = sums.size();
             std::vector<double> occupancy(N), dwell_s(N), extent_m(N), mean_rho(N),
                 probability(N), confidence(N), dominant_theta(N), dominant_rho(N),
                 t_latest(N);
             std::vector<int> support(N);
             std::vector<std::vector<double>> c_theta(N), c_rho(N), c_weight(N),
                 c_R(N), c_sigma_theta(N), c_sigma_rho(N), c_sigma_cross(N);
             std::vector<std::vector<int>> c_voxel_count(N);
             for (size_t i = 0; i < N; ++i) {
               const auto& s = sums[i];
               occupancy[i] = s.occupancy;
               dwell_s[i] = s.dwell_s;
               extent_m[i] = s.extent_m;
               mean_rho[i] = s.mean_rho;
               probability[i] = s.presence(presence_horizon_s);
               confidence[i] = s.confidence;
               dominant_theta[i] = s.dominant_theta;
               dominant_rho[i] = s.dominant_rho;
               support[i] = s.support;
               t_latest[i] = s.t_latest;
               for (const auto& c : s.components) {
                 c_theta[i].push_back(c.theta);
                 c_rho[i].push_back(c.rho);
                 c_weight[i].push_back(c.weight);
                 c_R[i].push_back(c.R);
                 c_sigma_theta[i].push_back(c.sigma_theta);
                 c_sigma_rho[i].push_back(c.sigma_rho);
                 c_sigma_cross[i].push_back(c.sigma_cross);
                 c_voxel_count[i].push_back(c.voxel_count);
               }
             }
             py::dict out;
             out["occupancy"] = occupancy;
             out["dwell_s"] = dwell_s;
             out["extent_m"] = extent_m;
             out["mean_rho"] = mean_rho;
             out["probability"] = probability;
             out["confidence"] = confidence;
             out["dominant_theta"] = dominant_theta;
             out["dominant_rho"] = dominant_rho;
             out["support"] = support;
             out["t_latest"] = t_latest;
             out["comp_theta"] = c_theta;
             out["comp_rho"] = c_rho;
             out["comp_weight"] = c_weight;
             out["comp_R"] = c_R;
             out["comp_sigma_theta"] = c_sigma_theta;
             out["comp_sigma_rho"] = c_sigma_rho;
             out["comp_sigma_cross"] = c_sigma_cross;
             out["comp_voxel_count"] = c_voxel_count;
             return out;
           },
           // A place stores its occupancy, so the probability reported here is
           // derived at a horizon the caller names. Zero asks whether the place
           // is occupied now.
           "presence_horizon_s"_a = 0.0)
      .def("flow_voxels",
           [](const FlowHandle& pipeline) {
             const auto cells = pipeline.flowVoxels();
             const size_t N = cells.size();
             std::vector<int> vx(N), vy(N), vz(N), count(N);
             for (size_t i = 0; i < N; ++i) {
               vx[i] = cells[i].index.x();
               vy[i] = cells[i].index.y();
               vz[i] = cells[i].index.z();
               count[i] = cells[i].observations;
             }
             py::dict out;
             out["vx"] = vx; out["vy"] = vy; out["vz"] = vz;
             out["count"] = count;   // world = (v + 0.5) * voxel_size
             return out;
           })
      .def(
          "score_observations",
          [](FlowHandle& pipeline,
             const std::vector<double>& timestamps_s,
             const std::vector<Eigen::Vector3d>& xyz_world,
             const std::vector<double>& theta,
             const std::vector<double>& rho,
             double horizon_s) {
            const auto scores = pipeline.scoreObservations(
                timestamps_s, xyz_world, theta, rho, horizon_s);
            const size_t N = scores.size();
            std::vector<int> matched(N);
            std::vector<double> log_p_joint(N);
            std::vector<double> log_p_heading(N);
            std::vector<double> log_p_speed(N);
            std::vector<double> mean_theta(N);
            std::vector<double> mean_rho(N);
            std::vector<double> p_present(N);
            std::vector<int> flag(N);
            std::vector<double> std_theta(N);
            std::vector<double> std_rho(N);
            std::vector<std::vector<double>> slot_weights(N);
            std::vector<std::vector<double>> slot_weight_stds(N);
            std::vector<std::vector<int>> slot_weight_clamped(N);
            std::vector<std::vector<int>> slot_weight_orders(N);
            std::vector<int> n_crossings(N);
            for (size_t i = 0; i < N; ++i) {
              matched[i] = scores[i].matched ? 1 : 0;
              log_p_joint[i] = scores[i].log_p_joint;
              log_p_heading[i] = scores[i].log_p_heading;
              log_p_speed[i] = scores[i].log_p_speed;
              mean_theta[i] = scores[i].mean_theta;
              mean_rho[i] = scores[i].mean_rho;
              p_present[i] = scores[i].p_present;
              flag[i] = static_cast<int>(scores[i].flag);
              std_theta[i] = scores[i].std_theta;
              std_rho[i] = scores[i].std_rho;
              slot_weights[i] = scores[i].slot_weights;
              slot_weight_stds[i] = scores[i].slot_weight_stds;
              slot_weight_clamped[i].assign(scores[i].slot_weight_clamped.begin(),
                                            scores[i].slot_weight_clamped.end());
              slot_weight_orders[i] = scores[i].slot_weight_orders;
              n_crossings[i] = static_cast<int>(scores[i].n_crossings);
            }
            py::dict out;
            out["matched"] = matched;
            out["log_p_joint"] = log_p_joint;
            out["log_p_heading"] = log_p_heading;
            out["log_p_speed"] = log_p_speed;
            out["mean_theta"] = mean_theta;
            out["mean_rho"] = mean_rho;
            out["p_present"] = p_present;
            out["flag"] = flag;
            out["std_theta"] = std_theta;
            out["std_rho"] = std_rho;
            out["slot_weights"] = slot_weights;
            out["slot_weight_stds"] = slot_weight_stds;
            out["slot_weight_clamped"] = slot_weight_clamped;
            out["slot_weight_orders"] = slot_weight_orders;
            out["n_crossings"] = n_crossings;
            return out;
          },
          "timestamps_s"_a,
          "xyz_world"_a,
          "theta"_a,
          "rho"_a,
          "horizon_s"_a = 0.0)
      .def(
          "responsibilities",
          [](const FlowHandle& pipeline,
             const std::vector<double>& theta,
             const std::vector<double>& rho) {
            // Soft slot assignments of raw (theta, rho) observations under the
            // configured fixed component layout. Model-independent: pure
            // component likelihoods normalized, no learned state — the same
            // values the mixing-weight predictors are trained on, so a caller
            // can build observed shares from the identical definition instead
            // of reimplementing the density.
            if (rho.size() != theta.size()) {
              throw std::invalid_argument("theta and rho must have equal length");
            }
            const auto module = pipeline.module();
            if (!module) {
              throw std::runtime_error("flow module '" + pipeline.name + "' is not running");
            }
            FixedComponents components(module->config.flow);
            std::vector<std::vector<double>> out(theta.size());
            for (size_t i = 0; i < theta.size(); ++i) {
              out[i] = components.responsibilities(theta[i], rho[i]);
            }
            return out;
          },
          "theta"_a,
          "rho"_a)
      .def(
          "presence_grid",
          [](const FlowHandle& pipeline, double t_seconds,
             const std::vector<double>& horizons_s) {
            const auto grid = pipeline.presenceGrid(t_seconds, horizons_s);
            const size_t N = grid.size();
            const size_t H = horizons_s.size();
            std::vector<int> vx(N), vy(N), vz(N);
            std::vector<std::vector<double>> ps(H, std::vector<double>(N));
            for (size_t i = 0; i < N; ++i) {
              vx[i] = grid[i].index.x();
              vy[i] = grid[i].index.y();
              vz[i] = grid[i].index.z();
              for (size_t h = 0; h < H; ++h) {
                ps[h][i] = grid[i].p_present[h];
              }
            }
            py::dict out;
            out["vx"] = vx;
            out["vy"] = vy;
            out["vz"] = vz;
            // One key per horizon, p<h>, where <h> drops a trailing ".0" so the
            // default horizons give the keys p5 / p10.
            for (size_t h = 0; h < H; ++h) {
              std::ostringstream tag;
              tag << horizons_s[h];
              out[py::str("p" + tag.str())] = ps[h];
            }
            return out;
          },
          "t_seconds"_a,
          "horizons_s"_a = std::vector<double>{5.0, 10.0})
      .def(
          "flow_grid",
          [](const FlowHandle& pipeline, double t_seconds) {
            const auto grid = pipeline.flowGrid(t_seconds);
            const size_t N = grid.size();
            std::vector<int> vx(N), vy(N), vz(N);
            std::vector<double> lam(N);
            std::vector<std::vector<double>> pi(N);
            for (size_t i = 0; i < N; ++i) {
              vx[i] = grid[i].index.x();
              vy[i] = grid[i].index.y();
              vz[i] = grid[i].index.z();
              lam[i] = grid[i].lambda;
              pi[i] = grid[i].pi;
            }
            py::dict out;
            out["vx"] = vx;
            out["vy"] = vy;
            out["vz"] = vz;
            out["lambda"] = lam;
            out["pi"] = pi;
            return out;
          },
          "t_seconds"_a)
      .def(
          "save_flow_state",
          [](FlowHandle& pipeline, const std::string& path) {
            const auto [ok, err] = pipeline.saveFlowState(path);
            return py::make_tuple(ok, err);
          },
          "path"_a)
      .def(
          "load_flow_state",
          [](FlowHandle& pipeline, const std::string& path) {
            const auto [ok, err] = pipeline.loadFlowState(path);
            return py::make_tuple(ok, err);
          },
          "path"_a)
      .def("flow_stats",
           [](const FlowHandle& pipeline) {
             const auto s = pipeline.flowStats();
             py::dict out;
             out["cells"] = s.cells;
             out["total_components"] = s.total_components;
             out["cells_with_valid_nudft"] = s.cells_with_valid_nudft;
             out["total_pi_nudft_obs"] = s.total_pi_nudft_obs;
             return out;
           })
      .def("flow_timing",
           [](const FlowHandle& pipeline) {
             // Nearest-rank percentile (us -> ms) over per-event samples.
             auto pct = [](std::vector<uint32_t> v, double q) -> double {
               if (v.empty()) return 0.0;
               std::sort(v.begin(), v.end());
               const double pos = q * (static_cast<double>(v.size()) - 1.0);
               const auto idx = static_cast<std::size_t>(pos);
               return static_cast<double>(v[idx]) / 1000.0;
             };
             // Population standard deviation over the per-event samples
             // (us -> ms), consistent with the mean = total_us / count.
             auto sd = [](const std::vector<uint32_t>& v) -> double {
               if (v.size() < 2) return 0.0;
               double mean = 0.0;
               for (const uint32_t x : v) mean += static_cast<double>(x);
               mean /= static_cast<double>(v.size());
               double acc = 0.0;
               for (const uint32_t x : v) {
                 const double d = static_cast<double>(x) - mean;
                 acc += d * d;
               }
               return std::sqrt(acc / static_cast<double>(v.size())) / 1000.0;
             };
             const auto t = pipeline.flowTiming();
             py::dict out;
             out["add_observation_count"] = t.add_observation_count;
             out["add_observation_total_us"] = t.add_observation_total_us;
             out["add_observation_median_ms"] = pct(t.add_observation_samples_us, 0.5);
             out["add_observation_p95_ms"] = pct(t.add_observation_samples_us, 0.95);
             out["add_observation_std_ms"] = sd(t.add_observation_samples_us);
             out["ensure_cells_count"] = t.ensure_cells_count;
             out["ensure_cells_total_us"] = t.ensure_cells_total_us;
             out["remap_cells_count"] = t.remap_cells_count;
             out["remap_cells_total_us"] = t.remap_cells_total_us;
             out["remap_cells_median_ms"] = pct(t.remap_cells_samples_us, 0.5);
             out["remap_cells_p95_ms"] = pct(t.remap_cells_samples_us, 0.95);
             out["remap_cells_std_ms"] = sd(t.remap_cells_samples_us);
             out["run_maintenance_count"] = t.run_maintenance_count;
             out["run_maintenance_total_us"] = t.run_maintenance_total_us;
             out["sharing_count"] = t.sharing_count;
             out["sharing_total_us"] = t.sharing_total_us;
             out["sharing_median_ms"] = pct(t.sharing_samples_us, 0.5);
             out["sharing_p95_ms"] = pct(t.sharing_samples_us, 0.95);
             out["sharing_std_ms"] = sd(t.sharing_samples_us);
             // Module-level temporal steps, timed at their natural cadence:
             // per-frame (process_frame / ingest / coupling), per-event
             // (place_summaries), and per-query (score).
             const auto m = pipeline.moduleTiming();
             out["process_frame_count"] = m.process_frame_count;
             out["process_frame_total_us"] = m.process_frame_total_us;
             out["process_frame_median_ms"] = pct(m.process_frame_samples_us, 0.5);
             out["process_frame_p95_ms"] = pct(m.process_frame_samples_us, 0.95);
             out["process_frame_std_ms"] = sd(m.process_frame_samples_us);
             out["ingest_total_us"] = m.ingest_total_us;
             out["ingest_median_ms"] = pct(m.ingest_samples_us, 0.5);
             out["ingest_p95_ms"] = pct(m.ingest_samples_us, 0.95);
             out["ingest_std_ms"] = sd(m.ingest_samples_us);
             out["coupling_count"] = m.coupling_count;
             out["coupling_total_us"] = m.coupling_total_us;
             out["coupling_median_ms"] = pct(m.coupling_samples_us, 0.5);
             out["coupling_p95_ms"] = pct(m.coupling_samples_us, 0.95);
             out["coupling_std_ms"] = sd(m.coupling_samples_us);
             out["place_summaries_count"] = m.place_summaries_count;
             out["place_summaries_total_us"] = m.place_summaries_total_us;
             out["place_summaries_median_ms"] = pct(m.place_summaries_samples_us, 0.5);
             out["place_summaries_p95_ms"] = pct(m.place_summaries_samples_us, 0.95);
             out["place_summaries_std_ms"] = sd(m.place_summaries_samples_us);
             out["score_count"] = m.score_count;
             out["score_detections"] = m.score_detections;
             out["score_total_us"] = m.score_total_us;
             out["score_median_ms"] = pct(m.score_samples_us, 0.5);
             out["score_p95_ms"] = pct(m.score_samples_us, 0.95);
             out["score_std_ms"] = sd(m.score_samples_us);
             return out;
           })
      .def("map_memory_stats",
           [](const FlowHandle& pipeline) {
             const auto m = pipeline.mapMemoryStats();
             py::dict out;
             out["voxels_per_side"] = m.voxels_per_side;
             out["tsdf_blocks"] = m.tsdf_blocks;
             out["semantic_blocks"] = m.semantic_blocks;
             out["tsdf_bytes"] = m.tsdf_bytes;
             out["semantic_bytes"] = m.semantic_bytes;
             return out;
           })
      .def("flow_stability",
           [](const FlowHandle& pipeline) {
             const auto records = pipeline.flowStability();
             py::list out;
             for (const auto& r : records) {
               py::dict d;
               d["total_observations"] = r.total_observations;
               d["n_obs_to_stable"] = r.n_obs_to_stable;
               out.append(std::move(d));
             }
             return out;
           })
      .def("lc_events",
           [](const FlowHandle& pipeline) {
             const auto events = pipeline.lcEvents();
             py::list out;
             for (const auto& e : events) {
               py::dict d;
               d["t_seconds"] = e.t_seconds;
               d["n_cells"] = e.n_cells;
               d["n_remap_pairs"] = e.n_remap_pairs;
               out.append(std::move(d));
             }
             return out;
           })
      .def("flow_coupling_json",
           [](const FlowHandle& pipeline) { return pipeline.couplingJson(); });

  m.def(
      "nav_snapshot",
      [](const py::bytes& graph_binary) {
             // Extract the PLACES layer as primitives (positions, edges, and the
             // flow-summary metadata the backend wrote) so no spark_dsg type has
             // to cross the module boundary. For the viz recorder.
             py::dict out;
             std::vector<uint64_t> ids, esrc, etgt;
             std::vector<double> x, y, dtheta, drho;
             std::vector<int> support, arch;
             std::vector<std::vector<double>> ctheta, cweight;
             const auto graph = graphFromBinary(graph_binary);
             if (graph && graph->hasLayer(spark_dsg::DsgLayers::PLACES)) {
               const auto& layer = graph->getLayer(spark_dsg::DsgLayers::PLACES);
               for (const auto& [node_id, node_ptr] : layer.nodes()) {
                 const auto& a =
                     node_ptr->attributes<spark_dsg::PlaceNodeAttributes>();
                 ids.push_back(node_id);
                 x.push_back(a.position.x()); y.push_back(a.position.y());
                 // Archetype is READ from the C++ classification (the ARCHETYPES
                 // layer UpdateArchetypesFunctor writes), never recomputed here:
                 // the place's archetype-layer parent carries the type. 0
                 // (STATIC) when no archetype parent exists (functor disabled).
                 int arch_type = 0;
                 for (const auto parent_id : node_ptr->parents()) {
                   const auto* parent = graph->findNode(parent_id);
                   if (!parent) continue;
                   if (const auto* aa =
                           parent->tryAttributes<spark_dsg::ArchetypeNodeAttributes>()) {
                     arch_type = static_cast<int>(aa->archetype_type);
                     break;
                   }
                 }
                 arch.push_back(arch_type);
                 const auto& md = a.metadata();
                 int sup = 0; double dt = 0.0, dr = 0.0;
                 std::vector<double> ct, cw;
                 if (md.is_object()) {          // null/absent until a summary is written
                   sup = md.value("flow_support", 0);
                   dt = md.value("flow_dominant_theta", 0.0);
                   dr = md.value("flow_dominant_rho", 0.0);
                   if (md.contains("flow_components") && md["flow_components"].is_array()) {
                     for (const auto& c : md["flow_components"]) {
                       if (!c.is_object()) continue;
                       ct.push_back(c.value("theta", 0.0));
                       cw.push_back(c.value("weight", 0.0));
                     }
                   }
                 }
                 support.push_back(sup); dtheta.push_back(dt); drho.push_back(dr);
                 ctheta.push_back(std::move(ct)); cweight.push_back(std::move(cw));
               }
               for (const auto& [edge_key, edge] : layer.edges()) {
                 (void)edge_key;
                 esrc.push_back(edge.source); etgt.push_back(edge.target);
               }
             }
             out["ids"] = ids; out["x"] = x; out["y"] = y;
             out["support"] = support; out["dtheta"] = dtheta; out["drho"] = drho;
             out["arch"] = arch;
             out["comp_theta"] = ctheta; out["comp_weight"] = cweight;
             out["edge_src"] = esrc; out["edge_tgt"] = etgt;
             return out;
      },
      "graph_binary"_a);

  m.def(
      "mesh_snapshot",
      [](const py::bytes& graph_binary) {
             // Dump the Hydra reconstruction mesh (world-frame vertices + faces,
             // optional per-vertex colour) so the viz can draw real geometry
             // instead of just the sparse place footprint. Read-only; empty dict
             // when the graph has no mesh. Flattened arrays (the recorder
             // reshapes to (V,3)/(F,3)) to avoid list-of-list overhead.
             py::dict out;
             const auto graph = graphFromBinary(graph_binary);
             if (!graph) return out;
             const auto mesh = graph->mesh();
             if (!mesh) return out;
             const size_t V = mesh->numVertices();
             const size_t F = mesh->numFaces();
             std::vector<float> vx(V), vy(V), vz(V);
             for (size_t i = 0; i < V; ++i) {
               const auto& p = mesh->pos(i);
               vx[i] = p.x(); vy[i] = p.y(); vz[i] = p.z();
             }
             std::vector<int> f0(F), f1(F), f2(F);
             for (size_t i = 0; i < F; ++i) {
               const auto& f = mesh->face(i);
               f0[i] = static_cast<int>(f[0]);
               f1[i] = static_cast<int>(f[1]);
               f2[i] = static_cast<int>(f[2]);
             }
             out["vx"] = vx; out["vy"] = vy; out["vz"] = vz;
             out["f0"] = f0; out["f1"] = f1; out["f2"] = f2;
             if (mesh->has_colors) {
               std::vector<int> r(V), g(V), b(V);
               for (size_t i = 0; i < V; ++i) {
                 const auto& c = mesh->color(i);
                 r[i] = c.r; g[i] = c.g; b[i] = c.b;
               }
               out["r"] = r; out["g"] = g; out["b"] = b;
             }
             return out;
      },
      "graph_binary"_a);

  // Per-layer DSG node counts (active vs archived) + summed object
  // trajectory-history length, computed in C++ and returned as plain ints.
  // Used by mem_profile.py to pin which layer's growth tracks the RSS
  // climb. The graph arrives serialized (DynamicSceneGraph.to_binary) and
  // ints come back: the spark_dsg DynamicSceneGraph python type is registered
  // in a different pybind module and can't be converted across the
  // _kairos_bindings boundary.
  m.def(
      "dsg_stats",
      [](const py::bytes& graph_binary) {
             py::dict out;
             const auto graph = graphFromBinary(graph_binary);
             if (!graph) {
               return out;
             }
             std::size_t traj_total = 0;
             // keys keep the layer numbers of the ROS1 version (objects 2, places 3,
             // mesh places 20) so existing readers are unaffected
             const std::vector<std::pair<std::string, std::string>> layers{
                 {spark_dsg::DsgLayers::OBJECTS, "2"},
                 {spark_dsg::DsgLayers::PLACES, "3"},
                 {spark_dsg::DsgLayers::MESH_PLACES, "20"}};
             for (const auto& [layer_name, layer_label] : layers) {
               std::size_t active = 0;
               std::size_t archived = 0;
               if (const auto layer_ptr = graph->findLayer(layer_name)) {
                 const auto& layer = *layer_ptr;
                 for (const auto& [node_id, node] : layer.nodes()) {
                   const auto* attrs = &node->attributes();
                   if (attrs && attrs->is_active) {
                     ++active;
                   } else {
                     ++archived;
                   }
                   if (const auto* k =
                           node->tryAttributes<spark_dsg::KhronosObjectAttributes>()) {
                     traj_total += k->trajectory_timestamps.size();
                   }
                 }
               }
               const std::string base = "l" + layer_label;
               out[(base + "_active").c_str()] = active;
               out[(base + "_archived").c_str()] = archived;
             }
             out["obj_traj_len_sum"] = traj_total;
             // Total published mesh geometry (live + committed) as a proxy for
             // mesh memory. Grows with re-covered ground if archived geometry is
             // retained; complements the (bounded) live TSDF window.
             if (const auto mesh = graph->mesh()) {
               out["mesh_vertices"] = mesh->numVertices();
               out["mesh_faces"] = mesh->numFaces();
             }
             return out;
      },
      "graph_binary"_a);

  // Standalone access to the SAME archetype rule the backend functor applies,
  // for the offline / ATC path that classifies from place_summaries() (which
  // returns probability, support, and per-component weights) rather than from a
  // populated ARCHETYPES layer. Thresholds are passed in by the caller (read
  // from the dataset config) so there is one rule and one set of parameters.
  m.def(
      "classify_archetype",
      [](double probability, int support, const std::vector<double>& weights,
         double dominant_weight_threshold, int min_flow_support,
         double min_probability) {
        return static_cast<int>(classifyArchetype(
            probability, support, weights, dominant_weight_threshold,
            min_flow_support, min_probability));
      },
      "probability"_a, "support"_a, "weights"_a,
      "dominant_weight_threshold"_a, "min_flow_support"_a, "min_probability"_a);
}

}  // namespace flow_module

}  // namespace kairos::python
