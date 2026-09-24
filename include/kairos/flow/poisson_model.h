#pragma once

/**
 * @file poisson_model.h
 * @brief Gamma-Poisson presence model for one cell: a conjugate Gamma posterior
 *        over the occupancy rate updated from observed visible-time, made
 *        time-varying through an embedded spectral predictor, exposing the
 *        predicted rate and the probability that the cell is occupied at a given
 *        time. Supports statistical merging of two models.
 */

#include <utility>
#include <vector>

#include "kairos/flow/nudft.h"

namespace kairos {

/// Probability that at least one target occupies a region over a horizon
/// @p delta_t, given the region's occupancy @p occupancy (the mean number
/// present) and the time @p dwell_seconds it takes one target to cross it.
/// Those already present contribute @p occupancy; new arrivals enter at the
/// flux rate occupancy / dwell (Little's law), so the expected count over the
/// horizon is occupancy * (1 + delta_t / dwell). A non-positive dwell means no
/// usable speed estimate and drops the arrival term, leaving the instantaneous
/// occupancy probability.
///
/// Shared by the per-cell and the region-level presence: a region is one cell
/// writ large, with its own occupancy (the sum over its cells) and its own
/// crossing time (its extent over its mean speed). Applying this once at
/// region scale counts a target crossing the region once, where a union over
/// the cells' independent presences would count it once per cell it transits.
double presenceFromOccupancy(double occupancy, double dwell_seconds, double delta_t);

struct PoissonModel {
  static constexpr double kPriorAlpha = 1.0;  ///< Gamma prior shape.
  static constexpr double kPriorBeta = 1.0;   ///< Gamma prior rate.

  double alpha = kPriorAlpha;  ///< Posterior Gamma shape (prior + observed counts).
  double beta = kPriorBeta;    ///< Posterior Gamma rate (prior + observed visible-time).
  double n_frames = 0.0;       ///< Count of visible frames (exposure events), so the
                               ///< mean number present = (observed counts) / n_frames.
  NUDFTModel nudft;            ///< Spectral predictor for the time-varying rate.
  bool nudft_valid = false;    ///< True once the spectral predictor has enough data.

  /// @name Windowed-rate spectral feeding
  /// The spectral predictor receives one sample per exposure window: the
  /// windowed instantaneous rate (events per visible second inside the window)
  /// timestamped at the window's mid-time. A window closes once
  /// `rate_window_s` of wall-clock time has elapsed since it opened, or on a
  /// visibility gap longer than that, so observed-but-empty spans contribute
  /// zero-rate samples and every observed phase of a cycle votes once per
  /// window.
  /// @{
  double rate_window_s = 120.0;  ///< Window length (> 0); configured, not serialized.
  double win_t0 = -1.0;        ///< Time the open window started (<0 = no window).
  double win_last = -1.0;      ///< Last exposure time folded into the window.
  double win_events = 0.0;     ///< Detections inside the open window.
  double win_visible = 0.0;    ///< Visible seconds inside the open window.
  /// @}

  /// Record one detection event at @p t_seconds: increments the count (alpha)
  /// and votes into the open rate window. Exposure (visible time) is
  /// accumulated separately via addExposure(), so the posterior rate is events
  /// per unit *visible* time, not per inter-detection gap.
  void recordEvent(double t_seconds);
  /// Accumulate @p dt_visible seconds of exposure (the cell was visible) into
  /// the rate denominator (beta), independent of whether a detection occurred.
  /// @p t_seconds is the frame time driving the spectral rate window.
  void addExposure(double dt_visible, double t_seconds);
  /// Convenience: one observation that both consumes @p delta_t_visible of
  /// exposure and records an event. Equivalent to addExposure() then
  /// recordEvent(). Retained for callers/tests that fold both at once.
  void update(double t_seconds, double delta_t_visible);
  /// Predicted occupancy rate lambda at @p t_seconds: the spectral forecast
  /// once the predictor is valid, the posterior MAP estimate until then.
  double predictLambda(double t_seconds, int order, int min_obs) const;
  /// Predicted occupancy L at @p t_seconds: the M/G/inf mean number of targets
  /// present in the cell, L = lambda * (visible-time / n_frames). Unlike
  /// predictPresence() this carries no prediction horizon, which is what makes
  /// it the correct weight for aggregating cells into a region: a probability
  /// saturates toward one at every cell as the horizon grows, flattening the
  /// aggregate, whereas occupancies stay proportional to the traffic each cell
  /// carries and are additive over a region.
  double predictOccupancy(double t_seconds, int order, int min_obs) const;
  /// Probability the cell is occupied at some point over a window @p delta_t at
  /// @p t_seconds, under an M/G/inf occupancy model. The mean number present
  /// L = lambda * (visible-time / n_frames) accounts for targets currently
  /// lingering; given a positive @p dwell_seconds (= cell size / mean speed)
  /// new arrivals enter at the flux rate lambda_arr = L / dwell, so
  /// p = 1 - exp(-(L + lambda_arr * delta_t)). With @p dwell_seconds <= 0 (no
  /// usable speed) it reports the instantaneous occupancy p = 1 - exp(-L).
  double predictPresence(double t_seconds, double delta_t, double dwell_seconds, int order,
                         int min_obs) const;

  /// Statistically merge another Poisson model into this one. Exact for the
  /// conjugate update: cancels one copy of the prior so n_AB = n_A + n_B and
  /// dt_AB = dt_A + dt_B. Internal NUDFT is merged via NUDFTModel::mergeFrom.
  void mergeFrom(const PoissonModel& other);

 private:
  /// Feed the open rate window's mean rate to the spectral predictor at the
  /// mid-time of [win_t0, @p t_end] and reset the window. Windows with no
  /// visible time are discarded without feeding.
  void closeRateWindow(double t_end);
};

}  // namespace kairos
