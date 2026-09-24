#pragma once

/**
 * @file nudft.h
 * @brief Incremental spectral predictor for one time-varying scalar. A
 *        NUDFTModel accumulates a non-uniform discrete Fourier transform of a
 *        scalar signal sampled at irregular timestamps over a fixed set of
 *        angular frequencies, predicts the signal's value at an arbitrary time,
 *        can track per-coefficient posterior variance, and supports
 *        statistical merging and JSON (de)serialization.
 */

#include <complex>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace kairos {

struct NUDFTModel {
  std::shared_ptr<const std::vector<double>> omegas_ptr;  ///< Shared angular frequencies.
  std::vector<std::complex<double>> gammas;  ///< One complex AC coefficient per frequency.
  double gamma0 = 0.0;                        ///< DC term (running mean of the signal).
  double m2 = 0.0;                            ///< Running mean of value^2. With gamma0 gives the cumulative observed-value variance (m2 - gamma0^2) in O(1) / constant memory: the online aleatoric scatter. Equal-weight, not recency-weighted.
  int n = 0;                                  ///< Number of observations accumulated.
  int min_observations = 20;                  ///< Observations required before predict() is trusted.

  /// @name Prequential order gating
  /// With `gate_enabled` (configured, not serialized), every update() first
  /// scores each candidate order's one-step-ahead prediction against the
  /// incoming value BEFORE folding it in, accumulating per-order squared
  /// error. bestOrder() then returns the order with the lowest accumulated
  /// error: harmonics engage only where they have lowered held-out prediction
  /// error, per cell and signal. Until min_observations errors accrue the
  /// gate stays at order 0 (static).
  /// @{
  bool gate_enabled = false;      ///< Configured; off = fixed order.
  std::vector<double> gate_err;   ///< Accumulated squared error per order 0..F.
  int gate_n = 0;                 ///< Number of scored predictions.
  /// @}

  /// @name Posterior variance tracking
  /// Maintained by updateVariance() for the mixing-weight predictors. The diagonal
  /// Sherman-Morrison update maintains an O(F) approximation of the posterior
  /// covariance of a Bayesian linear regression whose design vector is
  /// [1, cos(omega_1 t + phi_1), ..., cos(omega_F t + phi_F)]. The tracked
  /// parameters are therefore the *amplitudes* of that basis, i.e. the
  /// a_f = 2|gammas[f]| that predict() multiplies by cos(omega_f t + phi_f),
  /// not the complex coefficients gammas[f] themselves. predictionVariance()
  /// propagates them as var_gamma0 + sum_f var_gammas[f] * cos^2(.), which is
  /// exactly the variance of predict()'s reconstruction under that reading.
  /// Initialized to 1.0 (uninformative) on the first updateVariance() call;
  /// empty on a predictor that has never been fed a variance update.
  /// @{
  double var_gamma0 = 0.0;         ///< Posterior variance of the DC term.
  std::vector<double> var_gammas;  ///< Posterior variance of each AC amplitude a_f = 2|gammas[f]|.
  /// @}

  /// Fold one observed @p value at time @p t_seconds into the coefficients.
  void update(double t_seconds, double value);
  /// Call immediately after update() to track posterior variance. Uses the
  /// same design vector as update() to apply a diagonal Sherman-Morrison step.
  void updateVariance(double t_seconds, double sigma_noise);

  /// Reconstruct the signal value at @p t_seconds using @p order frequencies.
  double predict(double t_seconds, int order) const;
  /// predict() with the mean term replaced by @p mean, leaving the harmonics as
  /// fitted. Used by read-time evidence sharing, which shrinks the mean toward a
  /// neighbourhood estimate without touching the periodic structure.
  double predictWithMean(double t_seconds, int order, double mean) const;
  /// The prequentially-validated order in [0, max_order]: lowest accumulated
  /// one-step-ahead error, ties to the lower order; 0 until min_observations
  /// errors accrue. With the gate disabled returns @p max_order.
  int bestOrder(int max_order) const;
  /// predict() at bestOrder(max_order): the gated forecast.
  double predictGated(double t_seconds, int max_order) const;
  /// predictWithMean() at bestOrder(max_order): the gated forecast on a
  /// substituted mean term.
  double predictGatedWithMean(double t_seconds, int max_order, double mean) const;
  /// Posterior variance of predict()'s reconstruction at @p t_seconds using
  /// @p order frequencies: var_gamma0 plus the amplitude variances of the
  /// top-@p order frequencies by current coefficient magnitude, each scaled by
  /// cos^2 of its phase-shifted argument. Pass the same effective order the
  /// mean readout used (0 when the mean fell back to the DC term), or the
  /// variance describes a different reconstruction than the mean. 0 when no
  /// variance state has been initialized.
  double predictionVariance(double t_seconds, int order) const;
  /// True once updateVariance() has initialized posterior-variance state.
  bool hasVarianceState() const { return !var_gammas.empty(); }

  /// Clear all coefficients and counts back to the empty state.
  void reset();

  /// Statistically merge another model into this one (used on loop-closure cell
  /// collisions). gamma0 merge is exact (running mean of values); gammas[i]
  /// merge is a count-weighted linear combination, exact only if both streams
  /// shared the same DC trajectory at every step. Worst-case bias on gammas[i]
  /// is bounded by |dgamma0| * n_other / (n + n_other). Requires omegas_ptr to
  /// be pointer-equal between both models; on mismatch the merge is skipped.
  bool mergeFrom(const NUDFTModel& other);
};

/// JSON helpers shared between FlowMap and EdgeCouplingMap serialization so
/// the on-disk schema for a NUDFT model is identical in both contexts.
/// fromJson leaves the variance state empty when the record carries none.
nlohmann::json nudftToJson(const NUDFTModel& model);
bool nudftFromJson(const nlohmann::json& record,
                   NUDFTModel& model,
                   std::string* error = nullptr);

}  // namespace kairos