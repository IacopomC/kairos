/* -----------------------------------------------------------------------------
 * KAIROS: 4D Scene Graphs for Periodic Flow Dynamics.
 *
 * KairosPipeline subclasses Hydra's base PythonPipeline to add the flow /
 * temporal module. Hydra has no knowledge of KAIROS; the flow wiring is
 * injected through the base class's protected virtual hooks.
 * -------------------------------------------------------------------------- */
#pragma once

#include <pybind11/pybind11.h>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <spark_dsg/dynamic_scene_graph.h>

#include "hydra/bindings/python_pipeline.h"
#include "kairos/flow/flow_map.h"
#include "kairos/temporal/flow_temporal_module.h"

namespace kairos::python {

class KairosPipeline : public hydra::python::PythonPipeline {
 public:
  struct Config : hydra::python::PythonPipeline::Config {
    bool enable_flow_temporal_module = false;
    FlowTemporalModule::Config flow_temporal;
  } const config;

  KairosPipeline(const Config& config,
                 const hydra::Sensor::Ptr& sensor,
                 int robot_id = 0,
                 int config_verbosity = 0,
                 bool freeze_global_info = true,
                 bool step_mode_only = true,
                 std::string zmq_url = "");

  ~KairosPipeline() override;

  bool pushDetections(double timestamp_s,
                      const std::vector<Eigen::Vector3d>& xyz_world,
                      const std::vector<double>& theta,
                      const std::vector<double>& rho,
                      const std::vector<int64_t>& track_ids = {});

  /// Set the active voxel set directly from a supplied point set, for use when
  /// no reconstruction frontend is running (e.g. driving the flow layer from a
  /// simulated observer). Forwards to FlowTemporalModule::setVisibleVoxels.
  void setVisibleVoxels(float voxel_size_m,
                        const std::vector<Eigen::Vector3d>& xyz_world);

  /// Apply a caller-supplied deformation correction, for harnesses with no
  /// SLAM backend: flushes pending detection frames, then forwards to
  /// FlowTemporalModule::applyDeformationSnapshot (the standard alignment
  /// pass, run synchronously on the supplied stamped control-point warp).
  /// @p yaw_after carries the per-control-point yaw of a rotating correction,
  /// in radians, and is empty for a purely translational one.
  void applyDeformationSnapshot(double timestamp_s,
                                const std::vector<Eigen::Vector3d>& points_before,
                                const std::vector<double>& stamps_s,
                                const std::vector<Eigen::Vector3d>& points_after,
                                const std::vector<double>& yaw_after = {});

  /// Detection frames still queued for the flow worker (0 if flow disabled).
  size_t pendingDetectionFrames() const;
  /// Block until the flow worker has drained the queue and finished the
  /// in-flight batch. For offline batch driving with no frame loop to pace.
  void flushDetections();

  std::vector<FlowTemporalModule::ObservationScore> scoreObservations(
      const std::vector<double>& timestamps_s,
      const std::vector<Eigen::Vector3d>& xyz_world,
      const std::vector<double>& theta,
      const std::vector<double>& rho,
      double horizon_s) const;

  /// Additive hook: supply an external navigational-node layer so Kairos can run
  /// its per-node dynamics aggregation on it without a reconstruction frontend
  /// (the nodes are held in the spark_dsg PLACES layer). `positions` are
  /// world-frame node centres, `radii_m` the per-node
  /// support radius (PlaceNodeAttributes::distance), and `edges` the index pairs
  /// into `positions` that are traversably connected. The edges become
  /// PLACES-layer sibling edges, which the traversable-adjacency neighbourhood
  /// of the spatial coupling reads; passing none leaves every node isolated.
  /// Returns the assigned NodeIds in insertion order. Requires
  /// set_visible_voxels/a map update beforehand.
  std::vector<uint64_t> setNavigationalLayer(
      const std::vector<std::array<double, 3>>& positions,
      const std::vector<double>& radii_m,
      const std::vector<std::array<size_t, 2>>& edges = {});
  /// Aggregate flow onto the injected navigational layer at time `t`.
  void updatePlaceSummaries(double t);
  /// Per-node flow summaries, aligned to the order returned by
  /// setNavigationalLayer (empty summary for nodes with no support).
  std::vector<FlowTemporalModule::PlaceFlowSummary> placeSummaries() const;

  /// Live flow cells (voxel index + observation count) for the visualizer.
  std::vector<FlowMap::CellCount> flowVoxels() const;

  std::vector<FlowMap::PresenceGridEntry> presenceGrid(
      double t_seconds, const std::vector<double>& horizons_s = {5.0, 10.0}) const;
  std::vector<FlowMap::FlowGridEntry> flowGrid(double t_seconds) const;

  std::pair<bool, std::string> saveFlowState(const std::string& path) const;
  std::pair<bool, std::string> loadFlowState(const std::string& path);

  FlowMap::StateStats flowStats() const;
  FlowMap::TimingStats flowTiming() const;
  FlowTemporalModule::ModuleTiming moduleTiming() const;
  /// Active-window voxel-layer footprint (TSDF/semantic block counts + bytes),
  /// for the bounded-memory analysis. Zeroed before the first map update.
  hydra::VolumetricMap::MemoryStats mapMemoryStats() const;
  std::vector<FlowMap::StabilityRecord> flowStability() const;
  std::vector<FlowTemporalModule::LcEvent> lcEvents() const;
  std::string couplingJson() const;

 protected:
  void initExtraModules() override;
  void startExtraWorkers() override;
  void stopExtraWorkers() override;
  void resetExtraModules() override;

  std::shared_ptr<FlowTemporalModule> flow_temporal_;

  // Caller-supplied navigational-node layer for the additive summary hook.
  // A private DSG (never the backend graph), so the frontend/backend path is
  // untouched. nav_node_ids_ maps injection index -> NodeId.
  spark_dsg::DynamicSceneGraph::Ptr nav_graph_;
  std::vector<spark_dsg::NodeId> nav_node_ids_;
};

void declare_config(KairosPipeline::Config& config);

namespace kairos_pipeline {
void addBindings(pybind11::module_& m);
}

}  // namespace kairos::python
