#pragma once

/**
 * @file cell_state.h
 * @brief One voxel's flow state — its directional mixture, the spectral
 *        predictors that make its parameters time-varying, and its presence
 *        model — together with the snapshot/prediction types returned when the
 *        cell is queried at a given time.
 */

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "kairos/flow/config.h"
#include "kairos/flow/nudft.h"
#include "kairos/flow/poisson_model.h"
#include "kairos/flow/swgmm.h"
#include "hydra/reconstruction/voxel_types.h"

namespace kairos {
using namespace hydra;  // base-Hydra types (temporary; tighten to explicit using-decls)

/// Provenance of a prediction: a hard-coded FALLBACK default, a COLD_START
/// estimate from too few observations, or a full MODEL_BASED prediction.
enum class CellFlag { FALLBACK, COLD_START, MODEL_BASED };

/// Online running mean of observed agent speed across the whole map, owned by
/// the parent FlowMap and shared (by shared_ptr) with every cell. It is the
/// speed a slot reports until it has enough observations of its own. Saved
/// with the checkpoint (see FlowMap::toJson / fromJson).
struct GlobalSpeedPrior {
  double sum = 0.0;
  long long count = 0;
  void add(double value) {
    sum += value;
    ++count;
  }
  double mean(double fallback) const {
    return count > 0 ? sum / static_cast<double>(count) : fallback;
  }
};

/// Lightweight prediction for one cell at a query time: the mixing weights over
/// the direction slots, the probability the cell is occupied, and the
/// provenance flag.
struct FlowPrediction {
  std::vector<double> pi;
  /// Parameter variance of each normalized mixing weight, from the weight
  /// predictors' coefficient posterior propagated (first-order) through the
  /// normalization, at the same per-slot effective order the mean used.
  std::vector<double> var_pi;
  /// Per slot, nonzero when the pre-normalization weight prediction sat
  /// outside the normalizer's [0.001, 1] clamp (never-observed slot or
  /// negative harmonic reconstruction). Where the clamp binds the propagated
  /// variance ignores it, so these slots are separable downstream.
  std::vector<uint8_t> pi_clamped;
  /// Per slot, the effective spectral order the reported weight used: the
  /// gated order, 0 wherever the readout fell back to the DC term. At order 0
  /// the parameter variance is the DC-term posterior alone.
  std::vector<int> pi_order;
  double p_present = 0.0;
  /// Occupancy: the mean number of targets present in the cell at the query
  /// time, before any prediction horizon is applied. p_present is derived from
  /// it, so this is the same quantity stripped of the horizon.
  double l_hat = 0.0;
  /// Time one target takes to cross the cell (cell size / mean speed), or a
  /// non-positive value when the cell has no speed estimate yet. Reported so a
  /// consumer can re-derive p_present at a horizon of its own choosing.
  double dwell_s = -1.0;
  CellFlag flag = CellFlag::FALLBACK;
};

/// One mixture component's parameters as reconstructed at a query time, with the
/// predictive standard deviations on its heading and speed.
struct FlowComponentSnapshot {
  double theta = 0.0;
  double rho = 0.0;
  double sigma_theta = 0.0;
  double sigma_rho = 0.0;
  double sigma_cross = 0.0;
  double weight = 0.0;
  /// Predictive standard deviation of the slot's speed: its measured speed
  /// scatter blended with the prior spread (see
  /// FlowMapConfig::speed_scatter_prior_obs). std_theta stays zero because
  /// heading is pinned in the fixed regime.
  double std_theta = 0.0;
  double std_rho = 0.0;
  /// Parameter standard deviation on the normalized mixing weight (see
  /// FlowPrediction::var_pi).
  double std_pi = 0.0;
  /// See FlowPrediction::pi_clamped.
  bool pi_clamped = false;
  /// See FlowPrediction::pi_order.
  int pi_order = 0;
};

/// Full snapshot of a cell at a query time: every component's parameters plus
/// the cell-level presence probability, a confidence score, and the provenance
/// flag.
struct FlowCellSnapshot {
  std::vector<FlowComponentSnapshot> components;
  double p_present = 0.0;
  /// See FlowPrediction::l_hat. The horizon-free weight to aggregate cells by.
  double l_hat = 0.0;
  /// See FlowPrediction::dwell_s.
  double dwell_s = -1.0;
  double confidence = 0.0;
  CellFlag flag = CellFlag::FALLBACK;
  /// Crossings (weights samples) behind this cell's mixing weights at query
  /// time — the training evidence the weight estimates are a mean over.
  int n_crossings = 0;
};

/// Pooled evidence from a voxel's traversable neighbourhood, formed by FlowMap
/// at query time for read-time evidence sharing. Holds the count-weighted sum of
/// the neighbours' per-slot mean weights and the crossings behind it, so the
/// neighbourhood estimate for slot k is weighted_sum[k] / crossings: the value a
/// single voxel holding the neighbourhood's pooled crossings would carry.
struct NeighbourhoodEvidence {
  std::vector<double> weighted_sum;  ///< Sum over neighbours of C_j * gamma0_j^(k).
  double crossings = 0.0;            ///< Sum over neighbours of C_j.
  bool usable() const { return crossings > 0.0 && !weighted_sum.empty(); }
};

/// The complete persistent state of one voxel: its directional mixture, the
/// spectral predictors that make the mixing weights and per-slot speeds
/// time-varying, its presence model, the bookkeeping for loop-closure remaps
/// and time-to-stability, and the update / predict / snapshot / merge
/// operations that act on them.
struct FlowCellState {
  /// shared_omegas is owned by the parent FlowMap and reused across all cells
  /// so cross-cell predictor merges find a pointer-identical omegas_ptr (the
  /// merge path checks pointer identity, not value equality). Pass the
  /// FlowMap's shared_ptr in so all cells in one map agree by construction.
  FlowCellState(std::unique_ptr<SWGMMBase> model,
                const FlowMapConfig& cfg,
                std::shared_ptr<const std::vector<double>> shared_omegas,
                std::shared_ptr<const GlobalSpeedPrior> global_speed_prior = nullptr);

  std::shared_ptr<const std::vector<double>> shared_omegas_;  ///< Frequencies shared across the parent map's cells.
  /// Map-wide running mean speed, shared across all cells of the parent map.
  /// Null when no prior was supplied (e.g. unit tests) → cold starts fall back
  /// to fixed_mu_rho.
  std::shared_ptr<const GlobalSpeedPrior> global_speed_prior_;

  std::unique_ptr<SWGMMBase> swgmm;  ///< This cell's directional mixture.

  std::vector<NUDFTModel> nudft_models;  ///< Mixing-weight predictors, one per cardinal slot. Always populated.
  /// Per-slot speed models, one per cardinal slot. The speed is not forecast:
  /// only each model's running mean (gamma0), second moment (m2) and count are
  /// read, for the slot's speed and its scatter.
  std::vector<NUDFTModel> nudft_mu_rho_models;
  bool nudft_valid = false;
  double nudft_fit_timestamp = 0.0;

  double t_previous_observation = 0.0;

  PoissonModel poisson;  ///< This cell's presence model.
  int total_observations = 0;
  /// Crossings (weights samples) folded into the mixing-weight predictors: one
  /// per agent passage, and one per detection without a track id, so it counts
  /// the samples the weight estimate is a mean over. Kept
  /// on the cell rather than read off a predictor because a merge may refuse
  /// individual channels, leaving their sample counts unequal across slots.
  int n_crossings = 0;

  /// Time-to-stability tracking. The cell is "stable" once `last_dominant_slot`
  /// has been unchanged across `stability_window` consecutive observations.
  /// `n_obs_to_stable` records the observation count at which the streak began
  /// (the smallest N for which the dominant slot at N, ..., N + window - 1 all
  /// coincide). -1 until the cell first reaches stability; persists once set so
  /// the cold-start cost can be read directly.
  int last_dominant_slot = -1;
  int stability_streak = 0;
  int n_obs_to_stable = -1;

  /// Original (pre-deformation) hash index of this cell, captured on first
  /// observation. The cell's key in the map moves with each loop-closure remap;
  /// this field does NOT — it is the input fed back into the stamped
  /// deformation every cycle so the deformation graph interpolates the
  /// *correct* control points (those near the original observation in
  /// space-time) instead of re-deforming an already-deformed position.
  /// Preserved through mergeFrom by taking the originator with the most
  /// observations.
  GlobalIndex original_index{0, 0, 0};
  /// Timestamp (nanoseconds) of the cell's first observation. Used as the
  /// per-point stamp for time-windowed control-point selection in the
  /// deformation step.
  uint64_t creation_timestamp_ns = 0;
  /// False until the first observation lands on this cell; cells materialised
  /// by ensureCells() but never observed have no meaningful original position
  /// and are not deformed.
  bool original_set = false;
  /// Total rotation (radians) already applied to this cell's directional state,
  /// measured against the frame its observations were accumulated in. The
  /// positional correction is recomputed from `original_index` on every
  /// optimization, so applying it repeatedly lands the cell in the same place;
  /// the rotation is made idempotent the same way. Each correction rotates by
  /// the difference between the angle it asks for and the angle already
  /// carried, so a long-lived cell corrected at every loop closure ends up
  /// rotated once by its own epoch's error rather than once per closure.
  double heading_rotation_applied = 0.0;

  /// Fold one observation (heading @p theta, speed @p rho) at @p t_seconds,
  /// covering @p delta_t_visible of visibility, into this cell.
  void update(double theta,
              double rho,
              double t_seconds,
              double delta_t_visible,
              const FlowMapConfig& cfg,
              std::vector<double>* r_out = nullptr);

  /// Observation-count-weighted mean of this cell's per-slot speed means (the
  /// gamma0 of the speed predictors), i.e. the cell's overall mean observed
  /// speed. Returns a negative value when the cell has no speed observations.
  /// Sets the cell's crossing time for the presence dwell correction.
  double cellSpeedMean() const;

  /// Predict this cell / produce a full snapshot at @p t_seconds over @p delta_t.
  /// @p neighbourhood supplies the pooled neighbour evidence for read-time
  /// sharing; null reports the cell's own estimate.
  FlowPrediction predict(double t_seconds,
                         double delta_t,
                         const FlowMapConfig& cfg,
                         const NeighbourhoodEvidence* neighbourhood = nullptr) const;
  /// Fold one weights sample (a responsibility vector, typically the mean over
  /// a closed track visit) into the per-slot mixing-weight predictors at
  /// @p t_seconds. FlowMap calls it when a track's visit closes, and once per
  /// detection that carries no track id.
  void feedWeightsSample(double t_seconds, const std::vector<double>& r,
                         const FlowMapConfig& cfg);
  FlowCellSnapshot snapshot(double t_seconds,
                            double delta_t,
                            const FlowMapConfig& cfg,
                            const NeighbourhoodEvidence* neighbourhood = nullptr) const;

  /// The mean term slot @p k reports: its own running mean shrunk toward the
  /// neighbourhood estimate (evidence sharing). Public so
  /// FlowMap's grid readouts can substitute it at a chosen spectral order.
  double sharedMean(int k, const FlowMapConfig& cfg,
                    const NeighbourhoodEvidence* neighbourhood) const;

  /// Statistically merge another cell into this one (used on loop-closure
  /// collisions). Fixed-component geometry is identical across cells: each
  /// channel combines via count-weighted sufficient statistics.
  void mergeFrom(FlowCellState&& other, const FlowMapConfig& cfg);

  /// Rotate this cell's directional state by @p d_psi radians, the yaw part of
  /// a pose correction. A yaw error rotates every bearing the sensor reports,
  /// so a correction that only moves the cell leaves its headings wrong by the
  /// same angle; this re-expresses the stored distribution in the corrected
  /// frame. The mixture components stay pinned to their cardinal layout, so the
  /// rotation resamples the per-slot state onto the same fixed slots.
  ///
  /// The mixing-weight predictors are resampled by linear interpolation between
  /// adjacent slots. Every slot's predictor is fed at the same timestamps with
  /// the same sample count (feedWeightsSample folds one responsibility vector
  /// per crossing), and gamma0 and the AC coefficients are all
  /// linear in the fed value, so the interpolated coefficients are exactly
  /// those the rotated responsibilities would have produced. The posterior
  /// variances depend only on the sample times, shared across slots, so they
  /// are invariant. The second moment `m2` and the prequential `gate_err` are
  /// quadratic in the fed value and are interpolated as an approximation;
  /// convexity makes both slightly conservative (the interpolated value is at
  /// or above the true one).
  ///
  /// The per-slot speed predictors are permuted by the nearest whole slot
  /// rather than interpolated: each is fed only at the crossings whose heading
  /// won its slot, so the slots carry different sample sets and different
  /// counts, and blending them would fabricate a series no observation
  /// supports. The residual error is at most half a slot.
  ///
  /// The presence model is direction-free and is left untouched.
  void rotateHeading(double d_psi);

 private:
  /// The normalized mixing weights at @p t_seconds. When @p var_pi_out /
  /// @p clamped_out are non-null they receive, per slot, the propagated
  /// parameter variance of the normalized weight and the clamp indicator
  /// (semantics of FlowPrediction::var_pi / pi_clamped). The variance is
  /// evaluated at each slot's effective spectral order — the gated order,
  /// falling to 0 wherever the mean readout fell back to its DC term — so
  /// mean and variance always describe the same reconstruction.
  std::vector<double> queryPi(double t_seconds,
                              const FlowMapConfig& cfg,
                              const NeighbourhoodEvidence* neighbourhood,
                              std::vector<double>* var_pi_out = nullptr,
                              std::vector<uint8_t>* clamped_out = nullptr,
                              std::vector<int>* order_out = nullptr) const;
};

}  // namespace kairos
