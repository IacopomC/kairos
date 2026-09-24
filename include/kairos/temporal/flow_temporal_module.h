#pragma once

#include <hydra/active_window/active_window_module.h>
#include <hydra/backend/backend_module.h>
#include <hydra/common/module.h>
#include <hydra/reconstruction/volumetric_map.h>
#include <hydra/reconstruction/voxel_types.h>
#include <spatial_hash/grid.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "kairos/flow/edge_coupling_map.h"
#include "kairos/flow/flow_map.h"

namespace kairos {
using namespace hydra;  // base-Hydra types (temporary; tighten to explicit using-decls)

/// Immutable post-optimization snapshot of the slice of the deformation graph
/// that runDeformationAlignment reads. Built on the backend thread inside
/// handleBackendUpdate (which fires under the backend mutex, after the graph is
/// optimized), then handed to the worker via shared_ptr. Decouples the worker's
/// reads from the backend's next mutation so the value lookup in interpPoint can
/// no longer race with backend insert/optimize. Defined in the .cpp.
struct DgraphSnapshot;

/// Orchestrator for the temporal flow pipeline. Ingests volumetric-map updates
/// and detection frames, folds observations into the per-voxel FlowMap on a
/// background worker thread, maintains the optional edge-coupling map,
/// re-keys cells against the optimized deformation graph on backend updates,
/// aggregates per-place summaries into the scene graph, and answers flow
/// queries and trajectory-anchored scoring. Also persists and restores the
/// full flow state.
class FlowTemporalModule : public Module {
 public:
  /// Tunable parameters for the module. Validated by declare_config.
  struct Config {
    bool publish_metrics = true;
    float min_tsdf_weight_for_visibility = 1.0e-3f;  ///< Min reconstruction weight for a voxel to count as visible.
    float worker_rate_hz = 5.0f;                     ///< Rate at which the worker thread drains the frame queue.
    float heavy_maintenance_rate_hz = 1.0f / 30.0f;  ///< Rate of periodic heavy maintenance (deformation, summaries).
    int max_queue_size = 10;                         ///< Max queued detection frames before the oldest are dropped.
    float prediction_horizon_s = 2.0f;               ///< Default look-ahead (s) for presence prediction.
    int place_summary_max_nodes = 2000;              ///< Cap on place nodes summarized per pass (0 = no cap).
    int max_components_per_place = 3;                ///< Max mixture components kept per place summary.
    float component_min_weight = 0.05f;              ///< Drop place-summary components below this mixing weight.
    int component_min_voxel_count = 0;               ///< Drop place-summary components backed by fewer voxels.
    /// Fallback radius (metres) used when a place's free-space distance is
    /// missing or non-positive. A flow voxel is normally assigned to its
    /// nearest place when it sits inside that place's free-space sphere (with a
    /// small voxel-quantisation slack); this value is consulted only when the
    /// distance is unavailable, so cells at the active-window boundary still
    /// have a sane drop threshold.
    float place_support_max_distance_m = 0.5f;
    int deformation_knn = 4;                         ///< Neighbours used when interpolating a deformed position.
    double deformation_tolerance_s = 60.0;           ///< Time window (s) for selecting deformation control points.
    bool debug_metrics_nodes_only = false;
    int debug_metrics_max_nodes = 10;
    bool publish_voxel_markers = true;
    int voxel_markers_stride = 1;                    ///< Publish every Nth voxel marker.
    int voxel_markers_max_cells = 50000;             ///< Cap on published voxel markers.
    FlowMapConfig flow;                              ///< Configuration for the underlying flow map and cells.
  } const config;

  /// One decoded detection: the voxel it falls in and its (heading, speed).
  struct FrameObservation {
    GlobalIndex index;
    double theta = 0.0;
    double rho = 0.0;
    int64_t track_id = -1;  ///< Tracker identity; -1 = anonymous (no passage grouping).
  };

  /// A timestamped batch of detections handed to the worker queue.
  struct DetectionFrame {
    double timestamp_s = 0.0;
    std::vector<FrameObservation> observations;
  };

  /// One aggregated mixture component for a place-node summary: its mean
  /// (heading, speed), covariance, mixing weight, supporting voxel count, and
  /// circular concentration R.
  struct PlaceFlowComponentSummary {
    double theta = 0.0;
    double rho = 0.0;
    double sigma_theta = 0.0;
    double sigma_rho = 0.0;
    double sigma_cross = 0.0;
    double weight = 0.0;
    int voxel_count = 0;
    double R = 0.0;
  };

  /// Per-place flow summary embedded in the scene graph: the region occupancy,
  /// its crossing geometry, a directional concentration, the dominant
  /// (heading, speed), the supporting voxel count, the latest contributing
  /// timestamp, and the component breakdown.
  ///
  /// The occupancy is stored rather than the presence probability it induces,
  /// so no stored attribute is tied to a prediction horizon: a reader supplies
  /// its own horizon and calls presence() to obtain a probability. Weighting
  /// the aggregation by a probability instead would flatten every summary
  /// toward uniform at long horizons, since each cell's probability saturates
  /// toward one and the cells stop being distinguishable by the traffic they
  /// carry.
  struct PlaceFlowSummary {
    /// Region occupancy: the mean number of targets present across the support
    /// set, the sum of the supporting cells' occupancies. Grows with the size
    /// of the region, as the question it answers ("is anyone anywhere in
    /// here") should.
    double occupancy = 0.0;
    /// Mean occupancy of a single supporting cell. Unlike the region occupancy
    /// this is free of the region's size, so it stays comparable between a
    /// small place and a large one carrying the same density of traffic, and
    /// between datasets. It is the quantity to threshold when classifying how
    /// busy a place is; the region occupancy is the one to read when asking
    /// whether the region is occupied.
    double cell_occupancy = 0.0;
    /// Occupancy-weighted mean of the supporting cells' crossing times, the
    /// per-cell counterpart of dwell_s. Pairs with cell_occupancy to give a
    /// per-cell presence probability at any horizon.
    double cell_dwell_s = -1.0;
    /// Time one target takes to cross the region (its extent over its
    /// occupancy-weighted mean speed), or non-positive when no supporting cell
    /// has a speed estimate.
    double dwell_s = -1.0;
    /// Extent of the support region along the dominant heading, in metres: the
    /// distance a target travelling the prevailing direction covers in
    /// crossing it.
    double extent_m = 0.0;
    /// Occupancy-weighted mean speed over the support set.
    double mean_rho = 0.0;
    /// Directional concentration: the occupancy-weighted mean, over the
    /// support set, of each cell's dominant-slot weight. Runs from 1/K where
    /// every cell spreads its mass evenly over the headings to 1 where each
    /// commits to a single slot.
    double confidence = 0.0;
    double dominant_theta = 0.0;
    double dominant_rho = 0.0;
    int support = 0;
    double t_latest = 0.0;
    std::vector<PlaceFlowComponentSummary> components;

    /// Probability that at least one target occupies the region at some point
    /// over the horizon @p delta_t. Evaluated at read time from the stored
    /// occupancy and crossing time, so the caller sets the horizon.
    double presence(double delta_t) const;

    /// Probability that a single supporting cell is occupied at some point over
    /// the horizon @p delta_t, the size-free counterpart of presence(). Use
    /// this where a threshold has to carry the same meaning across places of
    /// different sizes and across datasets.
    double cellPresence(double delta_t) const;
  };

  /// Result of a single world-frame flow query: the dominant (heading, speed),
  /// occupancy probability, provenance flag, and number of mixture components.
  struct QueryResult {
    double dominant_theta = 0.0;
    double dominant_rho = 0.0;
    double probability = 0.0;
    uint8_t flag = 0;
    int32_t num_components = 0;
  };

  /// Per-observation score returned by scoreObservations. `matched` is false
  /// when no flow cell exists at the queried voxel (a model coverage gap); in
  /// that case the rest of the fields are zero and the caller should treat the
  /// observation as model-uncovered. `log_p_joint` is the joint log-density of
  /// (heading, speed) under the predicted mixture at the queried position and
  /// time. `mean_theta` / `mean_rho` are weight-weighted vector means across
  /// components — the right summary for circular error on multimodal mixtures.
  struct ObservationScore {
    bool matched = false;
    double log_p_joint = 0.0;
    /// Channel marginals of the joint predictive: the per-axis log-densities of
    /// heading and of speed, each obtained by analytically integrating the
    /// other axis out of the mixture. Zero when matched is false.
    double log_p_heading = 0.0;
    double log_p_speed = 0.0;
    double mean_theta = 0.0;
    double mean_rho = 0.0;
    double p_present = 0.0;
    uint8_t flag = 0;
    /// Predictive standard deviation propagated from the spectral predictor's
    /// posterior variance: mixing-weight-weighted RMS across components
    /// (std^2 = sum_k pi_k * sigma_k^2). Zero when uncertainty tracking is off
    /// or the cell isn't model-based.
    double std_theta = 0.0;
    double std_rho = 0.0;
    /// Per-slot mixing weights at query time, in component order. Empty when
    /// matched is false. Lets the coupling query use predicted weights rather
    /// than running-average ones.
    std::vector<double> slot_weights;
    /// Per-slot parameter standard deviation on the normalized mixing weight,
    /// aligned with slot_weights: the weight predictors' coefficient posterior
    /// propagated through the normalization at the effective order the
    /// reported weight used. Empty when matched is false; 0.0 throughout when
    /// uncertainty tracking is off; NaN entries mark a loaded state without
    /// variance (checkpoint written before tracking), which a consumer should
    /// refuse rather than read as certainty.
    std::vector<double> slot_weight_stds;
    /// Per slot, nonzero when the pre-normalization weight prediction sat
    /// outside the normalizer's clamp, where the propagated std above does
    /// not describe the reported weight. Aligned with slot_weights; empty
    /// when matched is false.
    std::vector<uint8_t> slot_weight_clamped;
    /// Per slot, the effective spectral order the reported weight used (0 =
    /// DC term alone, whether gated off or below the observation threshold).
    /// Aligned with slot_weights; empty when matched is false; all zeros when
    /// uncertainty tracking is off.
    std::vector<int> slot_weight_orders;
    /// Crossings (weights samples) behind the matched cell's mixing weights:
    /// its training evidence. 0 when matched is false.
    int32_t n_crossings = 0;
  };

  using TickCallback = std::function<void(double timestamp_s)>;

  explicit FlowTemporalModule(const Config& config);
  ~FlowTemporalModule() override;

  void start() override;
  void stop() override;
  void save(const LogSetup& setup) override;
  std::string printInfo() const override;

  /**
   * @brief Rebuild the active voxel set from the current TSDF.
   *
   * A voxel becomes "active" once its weight passes
   * min_tsdf_weight_for_visibility; only detections that fall in an active
   * voxel are kept. Re-creates the voxel index grid when the map resolution
   * changes.
   */
  void handleMapUpdate(uint64_t timestamp_ns,
                       const VolumetricMap& map,
                       const ActiveWindowOutput& msg);

  /**
   * @brief React to a completed pose-graph backend optimization.
   *
   * Snapshots the post-optimization deformation graph (for
   * runDeformationAlignment to re-key cells against), refreshes the place-node
   * summaries, and embeds the flow map in the graph metadata so it round-trips
   * with the saved scene graph.
   */
  void handleBackendUpdate(uint64_t timestamp_ns,
                           const DynamicSceneGraph& graph,
                           const kimera_pgmo::DeformationGraph& dgraph);

  /// Push a pre-decoded detection frame onto the worker queue. Called by the
  /// live adapter and by the offline entry point. Voxel indices in
  /// `observations` are assumed to be in the active voxel grid's coordinate
  /// system.
  void pushDetectionFrame(double timestamp_s,
                          std::vector<FrameObservation> observations);

  /// Convenience: push detections given world-frame positions plus per-track
  /// (heading, speed). Returns false if the voxel grid hasn't been initialized
  /// yet (i.e. no handleMapUpdate has fired). Sizes of all vectors must match.
  bool pushDetections(double timestamp_s,
                      const std::vector<Eigen::Vector3d>& xyz_world,
                      const std::vector<double>& theta,
                      const std::vector<double>& rho,
                      const std::vector<int64_t>& track_ids = {});

  /// Set the active voxel set directly from a supplied point set, for use when
  /// no reconstruction frontend is running. The voxels containing `xyz_world`
  /// (at `voxel_size_m`) become the active set that gates detections and
  /// receives per-frame presence exposure --- exactly the role the TSDF-derived
  /// set plays in handleMapUpdate. Both entry points funnel through the same
  /// private reset, so the only thing that differs is the source of the indices.
  void setVisibleVoxels(float voxel_size_m,
                        const std::vector<Eigen::Vector3d>& xyz_world);

  /// Apply a caller-supplied deformation correction, for harnesses with no
  /// SLAM backend (e.g. the grounded simulation): builds the same snapshot
  /// handleBackendUpdate() takes from the optimized deformation graph and
  /// runs the standard alignment pass synchronously. `points_before` are
  /// control points in the frame the map was built in, with their timestamps
  /// (seconds, same clock as detection timestamps); `points_after` are the
  /// corrected positions of the same points. Points may be passed in any
  /// order (they are sorted by stamp internally). Callers must ensure no
  /// detection batch is in flight (flush() first).
  ///
  /// @p yaw_after gives, per control point, the yaw the correction applies
  /// there, in radians about the vertical axis. Pass it empty for a purely
  /// translational correction. A yaw error rotates every bearing the sensor
  /// reports, so a rotating correction moves the cells and rotates the
  /// directional state they hold; both follow from the same control-point
  /// poses.
  void applyDeformationSnapshot(double timestamp_s,
                                const std::vector<Eigen::Vector3d>& points_before,
                                const std::vector<double>& stamps_s,
                                const std::vector<Eigen::Vector3d>& points_after,
                                const std::vector<double>& yaw_after = {});

  /// Number of detection frames still queued for the worker. Lets an offline
  /// driver pace its pushes so the bounded queue never drops frames.
  [[nodiscard]] size_t pendingFrames() const;

  /// Block until the worker has drained the queue and finished the in-flight
  /// batch. For offline batch driving where there is no frame loop to pace the
  /// pushes; a no-op wait when already idle. Observation/synchronisation only —
  /// does not change how frames are processed.
  void flush();

  /// Query a world-frame position for predicted flow. Returns nullopt if the
  /// voxel grid isn't initialized; otherwise returns a result with zeros if the
  /// queried voxel has no usable model.
  std::optional<QueryResult> queryFlowAtPosition(const Eigen::Vector3f& pos,
                                                 double query_time_s,
                                                 double horizon_s) const;

  /// Trajectory-anchored scoring of a batch of held-out detections. For each
  /// (timestamp, xyz, heading, speed) tuple, look up the voxel at xyz, take a
  /// snapshot at that timestamp (evaluate the predictors at the detection's own
  /// time, not at a shared snapshot time), and evaluate the predicted mixture
  /// density at (heading, speed). Each detection is thus scored at its own time
  /// anchor, with no externally chosen evaluation times. Returns the same
  /// length as the inputs; size mismatches yield all-default rows.
  std::vector<ObservationScore> scoreObservations(
      const std::vector<double>& timestamps_s,
      const std::vector<Eigen::Vector3d>& xyz_world,
      const std::vector<double>& theta,
      const std::vector<double>& rho,
      double horizon_s = 0.0) const;

  /// Run the per-navigational-node dynamics aggregation over a caller-supplied
  /// scene graph's PLACES layer (the spark_dsg layer the nav nodes live in), for
  /// use when no reconstruction frontend is running (e.g. an externally-authored
  /// navigational graph). Identical aggregation to the
  /// backend path; only the graph source differs. Requires a voxel grid
  /// (setVisibleVoxels or a map update) to have been installed first. Results
  /// are readable afterwards via placeSummariesCopy().
  void updatePlaceSummariesExternal(double timestamp_s, DynamicSceneGraph& graph);

  /// Dump the live flow cells (voxel index + observation count) for the
  /// visualizer's incremental voxel-field layer.
  std::vector<FlowMap::CellCount> flowCells() const;

  /// Snapshot the currently in-frustum voxels (the same active_voxels_ signal
  /// that drives presence exposure) and return each existing cell's predicted
  /// presence probabilities at @p t_seconds, one value per horizon in
  /// @p horizons_s (seconds). Used to build the presence grid during a
  /// test-scene replay (call at the desired cadence).
  std::vector<FlowMap::PresenceGridEntry> presenceGrid(
      double t_seconds, const std::vector<double>& horizons_s = {5.0, 10.0}) const;
  /// Forecast grid over the currently active voxels: per covered cell, the
  /// occupancy rate and mixing weights at @p t_seconds.
  std::vector<FlowMap::FlowGridEntry> flowGrid(double t_seconds) const;

  /// Persist the per-cell flow state (spectral predictors, mixture components,
  /// presence state, timestamps, counters) plus the voxel grid's resolution to
  /// a binary msgpack file. On load, before any map update fires, the saved
  /// voxel size is reused to reconstruct the index grid so scoreObservations
  /// works without warm-up. The schema is versioned; load aborts on version
  /// mismatch. The format is msgpack (compact binary, roughly 3x smaller than
  /// equivalent JSON text), so the on-disk file size is a meaningful proxy for
  /// the model's state size.
  ///
  /// Save is performed under the module's mutex; the call is best invoked when
  /// no other thread is writing to the flow map (e.g. after stop() or between
  /// processFrame ticks). Returns true on success; on failure writes the
  /// reason to *error if non-null.
  bool saveFlowState(const std::string& path, std::string* error = nullptr) const;
  bool loadFlowState(const std::string& path, std::string* error = nullptr);

  /// Hook called after each worker tick (the post-processing publish point).
  /// Used by a wrapper to publish markers/metrics at the worker rate.
  void setTickCallback(TickCallback cb);

  /// @name Read-only accessors used by the publish layer.
  /// @{
  const FlowMap& flowMap() const { return flow_map_; }
  float voxelSize() const { return voxel_size_; }
  std::unordered_map<NodeId, PlaceFlowSummary> placeSummariesCopy() const;
  size_t lastFrameObservations() const;
  size_t lastFrameUsedObservations() const;
  size_t placeNodesSeen() const;
  size_t placeNodesWithSupport() const;
  size_t placeNodesWithComponents() const;
  size_t placeSummariesPlacesCount() const;
  /// @}

  /// One loop-closure event records the timestamp at which
  /// runDeformationAlignment dispatched a remapCells call, plus how many cells
  /// and index pairs participated. Exposed so external evaluation can measure
  /// the effect of each alignment without the C++ side scoring detections.
  struct LcEvent {
    double t_seconds = 0.0;
    size_t n_cells = 0;
    size_t n_remap_pairs = 0;
  };
  std::vector<LcEvent> lcEventsCopy() const;

  /// Export the full EdgeCouplingMap state as a JSON string for external pair
  /// scoring. Returns "{}" when edge coupling is disabled. The JSON schema is
  /// documented in EdgeCouplingMap::toJson().
  std::string couplingJson() const;

  /// Module-level timing for the temporal steps that are NOT owned by FlowMap,
  /// each accumulated at its natural cadence so it can be compared against the
  /// keyframe budget (per-frame steps) or reported as a per-event / per-query
  /// cost. Populated only when config.flow.enable_timing is set; all counters
  /// stay zero otherwise. Counts let mean cost be recovered as total_us / count.
  struct ModuleTiming {
    /// Per frame (the real-time mapping loop; compare to the keyframe budget).
    uint64_t process_frame_count = 0;     ///< processFrame calls (= frames).
    uint64_t process_frame_total_us = 0;  ///< whole frame: ingest + coupling.
    uint64_t ingest_total_us = 0;         ///< folding this frame's detections.
    uint64_t coupling_count = 0;          ///< frames the coupling pass ran on.
    uint64_t coupling_total_us = 0;       ///< edge-coupling pass (O(used^2)).
    /// Per event (runs with the backend, not the per-frame loop).
    uint64_t place_summaries_count = 0;
    uint64_t place_summaries_total_us = 0;
    /// Per query (evaluation time only, not part of online mapping).
    uint64_t score_count = 0;             ///< scoreObservations invocations.
    uint64_t score_detections = 0;        ///< detections scored across all calls.
    uint64_t score_total_us = 0;
    /// Per-event microsecond samples, kept so median/p95 can be reported
    /// alongside the mean. Populated only when timing is enabled.
    std::vector<uint32_t> process_frame_samples_us;
    std::vector<uint32_t> ingest_samples_us;
    std::vector<uint32_t> coupling_samples_us;
    std::vector<uint32_t> place_summaries_samples_us;
    std::vector<uint32_t> score_samples_us;
  };
  [[nodiscard]] ModuleTiming moduleTiming() const;

 private:
  /// @brief Worker-thread loop: drains queued detection frames through
  /// processFrame and runs periodic maintenance (deformation alignment,
  /// place summaries).
  void workerSpin();
  /// @brief Core per-frame update: keep detections that fall in an active voxel
  /// and fold each into its cell (mixture + spectral predictors + presence model).
  void processFrame(const DetectionFrame& frame);
  /// @brief Re-key/move cells to follow a pose-graph correction (from a
  /// deformation-graph snapshot) so accumulated per-cell statistics stay aligned.
  void runDeformationAlignment(const DgraphSnapshot& snapshot);
  /// @brief Aggregate per-voxel cell state into place-node-level flow summaries.
  void updatePlaceSummaries(double timestamp_s, DynamicSceneGraph& graph);
  /// @brief Write per-edge flow attributes onto the scene graph at @p layer_id.
  void annotateFlowEdges(DynamicSceneGraph& graph, LayerId layer_id);
  /// @brief Set the grid resolution and replace the active-voxel set. The single
  /// writer of voxel_size_, voxel_grid_, and active_voxels_; both handleMapUpdate
  /// (indices from the TSDF) and setVisibleVoxels (indices from a supplied point
  /// set) funnel through here, so the active set is populated identically no
  /// matter its source. Caller must hold mutex_.
  void resetActiveVoxelsLocked(float voxel_size, GlobalIndexSet active);

 private:
  mutable std::mutex mutex_;
  mutable std::mutex summaries_mutex_;  // guards place_summaries_ and counters
  mutable std::mutex timing_mutex_;     // guards module_timing_ (cross-thread)
  mutable ModuleTiming module_timing_;  // mutable: scoreObservations is const
  std::condition_variable cv_;
  std::unique_ptr<std::thread> worker_thread_;
  bool should_shutdown_ = false;
  bool started_ = false;
  bool dgraph_updated_ = false;
  bool worker_processing_ = false;  ///< True while the worker owns an in-flight
                                    ///< batch (queue drained but not yet done);
                                    ///< read by flush(). Guarded by mutex_.

  TickCallback tick_callback_;

  FlowMap flow_map_;
  /// Pairwise edge coupling map. Owned here rather than in FlowMap or the scene
  /// graph because the per-edge tensors are heavy (hundreds of floats each) and
  /// the graph schema is not designed for dense edge payloads. Null when edge
  /// coupling is disabled.
  std::unique_ptr<EdgeCouplingMap> edge_coupling_map_;
  /// Loop-closure event log: one entry per runDeformationAlignment dispatch.
  /// Cheap to store (a few dozen events in a typical session) and exposed for
  /// loop-closure-event evaluation. Guarded by mutex_.
  std::vector<LcEvent> lc_events_;
  float voxel_size_ = 0.0f;
  GlobalIndexSet active_voxels_;
  std::unique_ptr<spatial_hash::IndexGrid> voxel_grid_;
  std::deque<DetectionFrame> frame_queue_;
  std::shared_ptr<const DgraphSnapshot> latest_snapshot_;
  GlobalIndexMap<double> last_visibility_update_s_;
  std::unordered_map<NodeId, PlaceFlowSummary> place_summaries_;
  size_t place_nodes_seen_ = 0;
  size_t place_nodes_with_support_ = 0;
  size_t place_nodes_with_components_ = 0;
  size_t place_summaries_places_count_ = 0;
  std::chrono::steady_clock::time_point last_heavy_maintenance_time_;
  size_t last_frame_observations_ = 0;

  /// One buffered detection (voxel index, per-observation responsibility, time)
  /// retained for the edge-coupling coherence window Δt_coh, so adjacent
  /// detections in nearby frames are paired, not only within a single frame.
  /// Worker-thread only; transient (not serialized).
  struct CouplingSample {
    GlobalIndex index;
    std::vector<double> resp;
    double t_seconds = 0.0;
  };
  std::deque<CouplingSample> coupling_window_;

  /// Timestamp of the previously processed frame, used to accumulate per-frame
  /// visible time into each visible cell's presence (Poisson) denominator.
  /// Negative until the first frame; reset per run.
  double prev_frame_timestamp_s_ = -1.0;
  size_t last_frame_used_observations_ = 0;
  /// Tracks the most recent detection-stream timestamp seen by the worker.
  /// Used as the preferred loop-closure-event timestamp in
  /// runDeformationAlignment so the logged t_seconds matches the input stream's
  /// timestamps, rather than the snapshot stamp which can be 0 when the backend
  /// fires without new input at end-of-run.
  double last_detection_ts_ = 0.0;
};

void declare_config(FlowTemporalModule::Config& config);

}  // namespace kairos
