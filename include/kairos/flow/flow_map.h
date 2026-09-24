#pragma once

/**
 * @file flow_map.h
 * @brief The per-voxel dynamics map: the collection of flow cells keyed by voxel
 *        index, with lookup/insertion, JSON (de)serialization, and the debug /
 *        merge-event helper types. Owns no algorithm logic itself — each cell's
 *        model lives in cell_state.
 */

#include <array>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hydra/reconstruction/voxel_types.h>
#include <spark_dsg/scene_graph_types.h>

#include "kairos/flow/cell_state.h"
#include "kairos/flow/config.h"

namespace kairos {
using namespace hydra;  // base-Hydra types (temporary; tighten to explicit using-decls)

/// One sampled cell's predicted state, for debug visualization/export.
struct FlowDebugSample {
  GlobalIndex index;
  double theta = 0.0;
  double rho = 0.0;
  double p_present = 0.0;
  double confidence = 0.0;
  CellFlag flag = CellFlag::FALLBACK;
};

/// One sampled cell's provenance flag only, for lightweight flag-distribution views.
struct FlowCellFlagSample {
  GlobalIndex index;
  CellFlag flag = CellFlag::FALLBACK;
};

/// Logged when two cells collide at the same target index after a loop-closure
/// remap. Emitted by FlowMap::drainMergeEvents() so a consumer can visualize
/// and quantify loop-closure-induced merges.
struct MergeEvent {
  GlobalIndex source_a;
  GlobalIndex source_b;
  GlobalIndex target;
  int n_a = 0;
  int n_b = 0;
  double timestamp = 0.0;
};

/// Per-cell snapshot consumed by the deformation-alignment step.
/// `current_index` is the cell's key in the map (after all prior remaps);
/// `original_index` and `creation_timestamp_ns` are the *invariant* original
/// observation hash and stamp, fed into the stamped deformation so the
/// deformation graph windows control points by time relative to the original
/// observation, not relative to the current corrected position.
struct CellRemapInput {
  GlobalIndex current_index;
  GlobalIndex original_index;
  uint64_t creation_timestamp_ns = 0;
  bool original_set = false;
};

/// The per-voxel dynamics map: a thread-safe collection of flow cells keyed by
/// voxel index. Handles cell creation, observation ingestion, time-stamped
/// prediction/snapshot queries, loop-closure remapping and merging,
/// maintenance, sampling for visualization, and JSON persistence. Each cell's
/// model lives in FlowCellState; this class owns no per-cell algorithm itself.
class FlowMap {
 public:
  explicit FlowMap(const FlowMapConfig& cfg);

  /// Create any cells in @p indices that do not exist yet (no observation).
  void ensureCells(const GlobalIndexSet& indices);

  /// Fold one observation (heading @p theta, speed @p rho) into the cell at
  /// @p index at time @p t_seconds, covering @p delta_t_visible of visibility.
  void addObservation(const GlobalIndex& index,
                      double theta,
                      double rho,
                      double t_seconds,
                      double delta_t_visible,
                      int64_t track_id = -1);

  /// Close and feed every open per-track visit (one weights sample per
  /// crossing). Called before checkpointing and
  /// before deformation re-keying so no accumulated vote is lost or keyed to
  /// a moved cell; also safe to call at any time.
  void flushVisits();

  /// Predict the cell at @p index at @p t_seconds over a window @p delta_t.
  FlowPrediction predict(const GlobalIndex& index, double t_seconds, double delta_t) const;

  /// Accumulate @p dt_visible seconds of exposure into the presence (Poisson)
  /// rate denominator of every cell in @p visible_indices that exists. Called
  /// once per visible frame so the presence rate is events per unit visible
  /// time rather than per inter-detection gap. Indices without a cell (e.g.
  /// never-observed voxels) are skipped. @p t_seconds is the frame time; it
  /// drives each cell's spectral rate window (see PoissonModel).
  void addPoissonExposure(const GlobalIndexSet& visible_indices, double dt_visible,
                          double t_seconds);

  /// One in-frustum cell's predicted presence probabilities for the presence
  /// grid, one value per requested horizon, index-aligned with the horizons
  /// argument of presenceGrid().
  struct PresenceGridEntry {
    GlobalIndex index;
    std::vector<double> p_present;  ///< P(occupied within horizon).
  };
  /// For each index in @p indices that has a cell, the presence probability at
  /// @p t_seconds, one value per horizon in @p horizons_s (seconds). Used to
  /// build the presence grid; indices without a cell are skipped.
  std::vector<PresenceGridEntry> presenceGrid(
      const GlobalIndexSet& indices, double t_seconds,
      const std::vector<double>& horizons_s = {5.0, 10.0}) const;

  /// One in-frustum cell's forecast state: the occupancy rate lambda (events
  /// per visible second) and the K-slot mixing weights at the query time.
  struct FlowGridEntry {
    GlobalIndex index;
    double lambda = 0.0;     ///< Forecast rate (the MAP rate until the
                             ///< predictor is valid).
    std::vector<double> pi;  ///< Forecast weights, normalized (mean-term
                             ///< mix until the predictors are valid).
  };
  /// For each index in @p indices that has a cell, its FlowGridEntry at
  /// @p t_seconds. Indices without a cell are skipped.
  std::vector<FlowGridEntry> flowGrid(const GlobalIndexSet& indices,
                                      double t_seconds) const;

  /// Per-observation responsibility (soft slot assignment) of (@p theta, @p rho)
  /// under the cell at @p index. In the fixed-component regime this depends only
  /// on the observation and the shared cardinal geometry, so it carries the
  /// per-detection directional signal the edge coupling needs (unlike the
  /// time-averaged predicted mixing weights). Empty if no cell exists at @p index.
  std::vector<double> responsibilities(const GlobalIndex& index,
                                       double theta,
                                       double rho) const;

  /// Full snapshot of the cell at @p index, or nullopt if no such cell exists.
  std::optional<FlowCellSnapshot> tryGetCellSnapshot(const GlobalIndex& index,
                                                     double t_seconds,
                                                     double delta_t) const;

  /// Keys of all cells currently in the map.
  std::vector<GlobalIndex> cellIndices() const;

  /// One live cell's voxel index and observation count, for dumping the flow
  /// field to a visualizer.
  struct CellCount {
    GlobalIndex index;
    int observations = 0;
  };
  /// (voxel index, #observations) for every cell currently in the map.
  std::vector<CellCount> cellObservationCounts() const;

  /// The traversability relation evidence sharing draws on, supplied by the
  /// temporal module from the places layer: which place node supports each
  /// voxel, which voxels each place node supports, and which place nodes an
  /// intra-layer edge joins. A voxel's neighbourhood is its own place's support
  /// set plus the support sets of that place's siblings, so it reaches the whole
  /// region the place covers rather than the face-adjacent cells alone.
  /// Refreshed on the backend pass; empty until the graph first covers the map,
  /// in which case sharing reports each cell's own estimate.
  struct SharingTopology {
    GlobalIndexMap<spark_dsg::NodeId> place_of_voxel;
    std::unordered_map<spark_dsg::NodeId, std::vector<GlobalIndex>> support_of_place;
    std::unordered_map<spark_dsg::NodeId, std::vector<spark_dsg::NodeId>> siblings_of_place;
    /// How many edges out from its own place a voxel may draw on, per place. The
    /// neighbourhood is bounded by the place's clearance either way: where place
    /// nodes are spaced about a clearance apart, one step already covers it and
    /// this is 1; where the layer is authored at voxel resolution, a place holds
    /// a single voxel and the same clearance is reached by walking several
    /// edges. One rule, so the neighbourhood does not silently depend on how
    /// coarse a dataset's graph happens to be.
    std::unordered_map<spark_dsg::NodeId, int> hops_of_place;
  };
  void setSharingTopology(SharingTopology topology);
  /// True once a topology has been supplied and evidence sharing can act.
  bool hasSharingTopology() const;

  /// Returns one entry per cell so the deformation step can feed the *original*
  /// (pre-deformation) position + stamp back into the deformation each call.
  std::vector<CellRemapInput> cellRemapInputs() const;

  /// Apply loop-closure index remaps; colliding cells are statistically merged.
  void remapCells(const std::vector<std::pair<GlobalIndex, GlobalIndex>>& remap_pairs);

  /// Apply loop-closure index remaps where the correction also rotates the
  /// frame. @p yaw_per_pair holds, per entry of @p remap_pairs, the yaw the
  /// correction applies at that cell, in radians. Each cell's directional state
  /// is rotated by its own yaw (FlowCellState::rotateHeading) before the cell
  /// moves, so colliding cells are merged in a common frame. A pure translation
  /// correction passes zeros, which is the single-argument overload.
  void remapCells(const std::vector<std::pair<GlobalIndex, GlobalIndex>>& remap_pairs,
                  const std::vector<double>& yaw_per_pair);

  /// Drains the queue of MergeEvents accumulated by remapCells. Logically a
  /// mutating operation, but exposed as const so a consumer can call it through
  /// a const reference. Mirrors the `mutable mutex_` pattern in this class.
  std::vector<MergeEvent> drainMergeEvents() const;

  /// Aggregate observation counts across all cells immediately before and after
  /// the most recent remapCells call. Under correct statistical merging the two
  /// are equal; exposed so downstream metrics can check that conservation
  /// invariant.
  int64_t totalObservationsPreRemap() const;
  int64_t totalObservationsPostRemap() const;

  /// Periodic maintenance hook at @p t_seconds. The per-cell models update
  /// incrementally during addObservation, so this currently only records the
  /// cell count and per-cycle timing rather than doing a batch refit.
  void runMaintenance(double t_seconds);

  /// Sample up to @p max_samples cells (every @p stride-th) returning only the
  /// provenance flag of each.
  std::vector<FlowCellFlagSample> cellFlagSamples(double t_seconds,
                                                  double delta_t,
                                                  int stride,
                                                  int max_samples) const;

  /// Sample up to @p max_samples cells (every @p stride-th) returning their full
  /// predicted debug state.
  std::vector<FlowDebugSample> debugSamples(double t_seconds,
                                            double delta_t,
                                            int stride,
                                            int max_samples) const;

  /// Count cells by provenance flag, indexed by CellFlag value.
  std::array<size_t, 3> countFlags(double t_seconds, double delta_t) const;

  /// Serialize / restore the whole map to / from JSON.
  nlohmann::json toJson() const;
  bool fromJson(const nlohmann::json& record, std::string* error = nullptr);

  /// Serialize / restore the whole map to / from a file. Return false on error.
  bool saveToFile(const std::string& filepath) const;
  bool loadFromFile(const std::string& filepath, std::string* error = nullptr);

  /// Number of cells in the map.
  [[nodiscard]] size_t numCells() const;

  /// Returns the shared frequency vector used by all spectral predictors in
  /// this map, so EdgeCouplingMap can use the same set of frequencies.
  std::shared_ptr<const std::vector<double>> sharedOmegas() const {
    return shared_omegas_;
  }

  /// Aggregate state statistics for runtime/memory reporting. Cheap (single
  /// mutex acquisition, summation over cells).
  struct StateStats {
    size_t cells = 0;
    size_t total_components = 0;       ///< Sum of component count across all cells.
    size_t cells_with_valid_nudft = 0;
    size_t total_pi_nudft_obs = 0;     ///< Sum of mixing-weight predictor obs across all (cell, slot).
  };
  [[nodiscard]] StateStats stateStats() const;

  /// Per-component timing accumulators. Populated only when timing is enabled
  /// in the config; counts and totals are zero otherwise. Lets per-component
  /// cost be reported quantitatively rather than only logged.
  struct TimingStats {
    uint64_t add_observation_count = 0;
    uint64_t add_observation_total_us = 0;
    uint64_t ensure_cells_count = 0;
    uint64_t ensure_cells_total_us = 0;
    uint64_t remap_cells_count = 0;
    uint64_t remap_cells_total_us = 0;
    uint64_t run_maintenance_count = 0;
    uint64_t run_maintenance_total_us = 0;
    /// Evidence sharing is timed per pooled neighbourhood, on the read path,
    /// so its cost is reported independently of the query it serves.
    uint64_t sharing_count = 0;
    uint64_t sharing_total_us = 0;
    /// Per-event microsecond samples, kept so median/p95 can be reported
    /// alongside the mean (total_us / count). Populated only when timing is on.
    std::vector<uint32_t> add_observation_samples_us;
    std::vector<uint32_t> remap_cells_samples_us;
    std::vector<uint32_t> sharing_samples_us;
  };
  [[nodiscard]] TimingStats timingStats() const;

  /// Per-cell time-to-stability records, one entry per cell that has received
  /// at least one observation. `n_obs_to_stable` is -1 for cells that have not
  /// yet reached stability (so "still-noisy" cells can be counted separately
  /// from "stable" ones). Cells with zero observations are omitted; including
  /// them would dilute the bucketing.
  struct StabilityRecord {
    int total_observations = 0;
    int n_obs_to_stable = -1;
  };
  [[nodiscard]] std::vector<StabilityRecord> stabilityRecords() const;

 private:
  std::unique_ptr<SWGMMBase> makeModel() const;

  FlowMapConfig cfg_;
  /// Single allocation of angular frequencies, shared with every cell this map
  /// ever creates. Pointer-identical across cells so cross-cell predictor
  /// merges are not silently refused.
  std::shared_ptr<const std::vector<double>> shared_omegas_;
  /// Map-wide running mean of observed speed, shared (as a read-only view) with
  /// every cell as the cold-start speed prior. Maintained in addObservation and
  /// saved with the checkpoint.
  std::shared_ptr<GlobalSpeedPrior> global_speed_prior_ =
      std::make_shared<GlobalSpeedPrior>();
  mutable std::mutex mutex_;
  GlobalIndexMap<FlowCellState> cells_;
  SharingTopology sharing_topology_;

  /// One track's open visit to one cell: the running sum of per-detection
  /// responsibilities, its count, and the visit's time span. Closed (fed to
  /// the cell as a single averaged weights sample at the span's mid-time) on
  /// cell change, on a silence longer than visit_gap_s, on a span longer than
  /// visit_max_s, and by flushVisits().
  struct OpenVisit {
    GlobalIndex cell;
    std::vector<double> sum_r;
    double n = 0.0;
    double t_first = 0.0;
    double t_last = 0.0;
  };
  std::unordered_map<int64_t, OpenVisit> open_visits_;

  /// Close one visit (caller holds mutex_). Feeds the visit's mean
  /// responsibility to its cell as one weights sample if the cell still exists.
  void closeVisitLocked(OpenVisit& visit);

  /// Pool the evidence of one cell's traversable neighbourhood for read-time
  /// sharing (caller holds mutex_). Returns an unusable pool when the lent
  /// crossings are zero, when the place graph does not yet cover the cell, or when no
  /// neighbour has a crossing of its own.
  NeighbourhoodEvidence neighbourhoodEvidenceLocked(const GlobalIndex& index) const;
  /// neighbourhoodEvidenceLocked() with the read-path timing sub-step attached.
  NeighbourhoodEvidence timedNeighbourhoodEvidenceLocked(const GlobalIndex& index) const;
  mutable std::vector<MergeEvent> merge_events_;
  int64_t total_observations_pre_remap_ = 0;
  int64_t total_observations_post_remap_ = 0;
  /// Updated under mutex_ from the top-level entry points; read by timingStats()
  /// which acquires mutex_ once for a snapshot copy.
  /// Mutable so the read paths can attribute their evidence-sharing sub-step
  /// from const query methods. Mirrors the `mutable mutex_` pattern above.
  mutable TimingStats timing_;
};

}  // namespace kairos
