#include "kairos/flow/poisson_model.h"

#include <algorithm>
#include <cmath>

namespace kairos {

namespace {
constexpr double kMinWindowVisibleS = 1.0e-6;
}  // namespace

double presenceFromOccupancy(double occupancy, double dwell_seconds, double delta_t) {
  const double L = std::max(0.0, occupancy);
  const double horizon = std::max(delta_t, 0.0);
  if (dwell_seconds > 0.0) {
    const double lambda_arr = L / dwell_seconds;
    return 1.0 - std::exp(-(L + lambda_arr * horizon));
  }
  return 1.0 - std::exp(-L);
}

void PoissonModel::closeRateWindow(double t_end) {
  if (win_t0 < 0.0 || win_visible < kMinWindowVisibleS) {
    win_t0 = -1.0;
    win_last = -1.0;
    win_events = 0.0;
    win_visible = 0.0;
    return;
  }
  const double t_mid = 0.5 * (win_t0 + t_end);
  nudft.update(t_mid, std::max(0.0, win_events / win_visible));
  if (!nudft_valid && nudft.n >= nudft.min_observations) {
    nudft_valid = true;
  }
  win_t0 = -1.0;
  win_last = -1.0;
  win_events = 0.0;
  win_visible = 0.0;
}

void PoissonModel::addExposure(double dt_visible, double t_seconds) {
  if (dt_visible <= 0.0) {
    return;
  }
  beta += dt_visible;
  n_frames += 1.0;  // one visible frame -> denominator for mean number present

  // A visibility gap longer than the window means the open window's span no
  // longer reflects contiguous observation: close it at its own last exposure
  // so the sample's mid-time stays inside observed time.
  if (win_t0 >= 0.0 && (t_seconds - win_last) > rate_window_s) {
    closeRateWindow(win_last);
  }
  if (win_t0 < 0.0) {
    win_t0 = t_seconds;
  }
  win_last = t_seconds;
  win_visible += dt_visible;
  if ((t_seconds - win_t0) >= rate_window_s) {
    closeRateWindow(t_seconds);
  }
}

void PoissonModel::recordEvent(double t_seconds) {
  alpha += 1.0;

  // The event votes into the open rate window; the spectral predictor is fed
  // on window close (see addExposure).
  if (win_t0 < 0.0) {
    win_t0 = t_seconds;
    win_last = t_seconds;
  }
  win_events += 1.0;
}

void PoissonModel::update(double t_seconds, double delta_t_visible) {
  addExposure(delta_t_visible, t_seconds);
  recordEvent(t_seconds);
}

double PoissonModel::predictLambda(double t_seconds, int order, int min_obs) const {
  if (nudft_valid && nudft.n >= min_obs) {
    return std::max(0.0, nudft.predictGated(t_seconds, order));
  }

  return (alpha > 1.0 && beta > 0.0) ? std::max(0.0, (alpha - 1.0) / beta) : 0.0;
}

double PoissonModel::predictOccupancy(double t_seconds,
                                      int order,
                                      int min_obs) const {
  // Mean number of targets present = occupancy rate * mean per-frame exposure.
  // This is the M/G/inf number-in-system, not a detection rate: a lingering
  // target inflates the detection count but contributes one unit of presence.
  const double lambda_t = predictLambda(t_seconds, order, min_obs);
  const double visible_time = beta - kPriorBeta;
  return (n_frames > 0.0 && visible_time > 0.0)
             ? std::max(0.0, lambda_t * (visible_time / n_frames))
             : 0.0;
}

double PoissonModel::predictPresence(double t_seconds,
                                     double delta_t,
                                     double dwell_seconds,
                                     int order,
                                     int min_obs) const {
  return presenceFromOccupancy(
      predictOccupancy(t_seconds, order, min_obs), dwell_seconds, delta_t);
}

void PoissonModel::mergeFrom(const PoissonModel& other) {
  alpha += other.alpha - kPriorAlpha;
  beta += other.beta - kPriorBeta;
  n_frames += other.n_frames;
  nudft.mergeFrom(other.nudft);
  nudft_valid = nudft_valid || other.nudft_valid;
  // Fold the other model's open rate window into this one's counters. The
  // merged span is approximate (the two windows may not overlap), but the
  // pooled events/visible-time keep the next fed sample's rate unbiased.
  if (other.win_t0 >= 0.0) {
    win_t0 = (win_t0 < 0.0) ? other.win_t0 : std::min(win_t0, other.win_t0);
    win_last = std::max(win_last, other.win_last);
    win_events += other.win_events;
    win_visible += other.win_visible;
  }
}

}  // namespace kairos
