#include <gtest/gtest.h>
#include <kairos/flow/poisson_model.h>

#include <cmath>

namespace kairos {

TEST(PoissonModel, InitialState) {
  PoissonModel model;
  EXPECT_NEAR(model.alpha, 1.0, 1.0e-15);
  EXPECT_NEAR(model.beta, 1.0, 1.0e-15);
  EXPECT_NEAR(model.n_frames, 0.0, 1.0e-15);
  EXPECT_FALSE(model.nudft_valid);
  EXPECT_EQ(model.nudft.n, 0);
}

TEST(PoissonModel, UpdateIncrementsAlphaBeta) {
  PoissonModel model;
  model.nudft.omegas_ptr =
      std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.nudft.min_observations = 5;
  model.nudft.reset();
  model.update(100.0, 5.0);

  EXPECT_NEAR(model.alpha, 2.0, 1.0e-15);
  EXPECT_NEAR(model.beta, 6.0, 1.0e-15);
  EXPECT_EQ(model.nudft.n, 0);  // the event waits in the open rate window
  EXPECT_FALSE(model.nudft_valid);
}

TEST(PoissonModel, LambdaMapEstimate) {
  PoissonModel model;
  for (int i = 0; i < 10; ++i) {
    model.update(static_cast<double>(i) * 10.0, 2.0);
  }

  const double expected_lambda = (model.alpha - 1.0) / model.beta;
  const double predicted = model.predictLambda(0.0, 3, 100);
  EXPECT_NEAR(predicted, expected_lambda, 1.0e-10);
}

TEST(PoissonModel, UpdateClampsNegativeVisibility) {
  PoissonModel model;
  model.update(0.0, -5.0);

  EXPECT_NEAR(model.beta, 1.0, 1.0e-15);
}

TEST(PoissonModel, AddExposureGrowsBetaNotAlpha) {
  PoissonModel model;
  model.addExposure(3.0, 0.0);
  model.addExposure(2.0, 0.1);
  EXPECT_NEAR(model.alpha, 1.0, 1.0e-15);    // exposure is not an event
  EXPECT_NEAR(model.beta, 6.0, 1.0e-15);     // prior 1 + 5 s visible time
  EXPECT_NEAR(model.n_frames, 2.0, 1.0e-15); // two visible frames
}

TEST(PoissonModel, RecordEventGrowsAlphaNotBeta) {
  PoissonModel model;
  model.recordEvent(0.0);
  model.recordEvent(1.0);
  EXPECT_NEAR(model.alpha, 3.0, 1.0e-15);    // prior 1 + 2 events
  EXPECT_NEAR(model.beta, 1.0, 1.0e-15);     // no exposure recorded
  EXPECT_NEAR(model.n_frames, 0.0, 1.0e-15); // events are not frames
}

TEST(PoissonModel, RateIsEventsPerVisibleTime) {
  // 9 detections over 100 s of accumulated visible time -> the MAP rate is
  // events / visible-time, independent of how exposure was split across frames.
  PoissonModel model;
  for (int i = 0; i < 100; ++i) {
    model.addExposure(1.0, static_cast<double>(i));  // 100 s visible
  }
  for (int i = 0; i < 9; ++i) {
    model.recordEvent(static_cast<double>(i));  // 9 events
  }
  // alpha = 1 + 9, beta = 1 + 100 -> (alpha-1)/beta = 9/101
  EXPECT_NEAR(model.predictLambda(0.0, 3, 1000), 9.0 / 101.0, 1.0e-9);
}

TEST(PoissonModel, PredictPresenceRange) {
  PoissonModel model;
  for (int i = 0; i < 5; ++i) {
    model.update(static_cast<double>(i), 1.0);
  }

  const double p = model.predictPresence(10.0, 2.0, /*dwell=*/1.0, 3, 100);
  EXPECT_GE(p, 0.0);
  EXPECT_LE(p, 1.0);
}

TEST(PoissonModel, PredictPresenceZeroHorizonIsInstantaneousOccupancy) {
  // At a zero horizon the occupancy model reports the probability the cell is
  // occupied *now* (the lingering term), which is positive when the cell has
  // been occupied -- unlike a detection-arrival rate, which would vanish. The
  // value is the dwell-independent 1 - exp(-L).
  PoissonModel model;
  for (int i = 0; i < 5; ++i) {
    model.update(static_cast<double>(i), 1.0);
  }

  const double p = model.predictPresence(10.0, 0.0, /*dwell=*/1.0, 3, 100);
  EXPECT_GT(p, 0.0);
  EXPECT_LT(p, 1.0);
  // Independent of dwell at zero horizon (only the lingering term remains).
  const double p_other_dwell = model.predictPresence(10.0, 0.0, /*dwell=*/5.0, 3, 100);
  EXPECT_NEAR(p, p_other_dwell, 1.0e-12);
}

TEST(PoissonModel, PredictPresenceIncreasesWithHorizon) {
  PoissonModel model;
  for (int i = 0; i < 10; ++i) {
    model.update(static_cast<double>(i) * 10.0, 1.0);
  }

  const double p_short = model.predictPresence(100.0, 0.5, /*dwell=*/1.0, 3, 100);
  const double p_long = model.predictPresence(100.0, 10.0, /*dwell=*/1.0, 3, 100);
  EXPECT_LT(p_short, p_long);
}

TEST(PoissonModel, PredictPresenceNoSpeedIsHorizonIndependent) {
  // With no usable speed (dwell <= 0) there is no new-arrival term, so presence
  // is the instantaneous occupancy regardless of the horizon.
  PoissonModel model;
  for (int i = 0; i < 10; ++i) {
    model.update(static_cast<double>(i) * 10.0, 1.0);
  }

  const double p_short = model.predictPresence(100.0, 0.5, /*dwell=*/-1.0, 3, 100);
  const double p_long = model.predictPresence(100.0, 10.0, /*dwell=*/-1.0, 3, 100);
  EXPECT_NEAR(p_short, p_long, 1.0e-12);
  EXPECT_GT(p_short, 0.0);
}

TEST(PoissonModel, NudftValidityNeedsMinObs) {
  PoissonModel model;
  model.nudft.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.nudft.min_observations = 20;
  model.nudft.reset();

  for (int i = 0; i < 5; ++i) {
    model.update(static_cast<double>(i) * 100.0, 1.0);
  }

  EXPECT_FALSE(model.nudft_valid);
}

TEST(PoissonModel, NudftValidityWithEnoughObs) {
  PoissonModel model;
  model.nudft.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.nudft.min_observations = 5;
  model.nudft.reset();

  for (int i = 0; i < 30; ++i) {
    model.update(static_cast<double>(i) * 100.0, 1.0);
  }

  EXPECT_TRUE(model.nudft_valid);
}

TEST(PoissonModel, WindowedFeedingEmitsOneSamplePerWindow) {
  PoissonModel model;
  model.nudft.omegas_ptr =
      std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.nudft.min_observations = 2;
  model.nudft.reset();
  model.rate_window_s = 10.0;

  // 25 s of contiguous 1 Hz exposure with one event per second: windows close
  // at 10 s and 20 s (two fed samples); the tail stays open.
  for (int i = 0; i < 25; ++i) {
    model.addExposure(1.0, static_cast<double>(i));
    model.recordEvent(static_cast<double>(i));
  }

  EXPECT_EQ(model.nudft.n, 2);          // one sample per closed window
  EXPECT_GT(model.win_visible, 0.0);    // tail window still open
  // Constant one-event-per-visible-second signal -> fed rate is ~1.
  EXPECT_NEAR(model.nudft.gamma0, 1.0, 0.15);
  // Conjugate counters are unaffected by the windowing.
  EXPECT_NEAR(model.alpha, 26.0, 1.0e-12);
  EXPECT_NEAR(model.beta, 26.0, 1.0e-12);
}

TEST(PoissonModel, WindowedFeedingClosesOnVisibilityGap) {
  PoissonModel model;
  model.nudft.omegas_ptr =
      std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.nudft.min_observations = 1;
  model.nudft.reset();
  model.rate_window_s = 10.0;

  model.addExposure(1.0, 0.0);
  model.recordEvent(0.0);
  model.addExposure(1.0, 1.0);
  // A gap far longer than the window: the stale window must close at its own
  // last exposure (not span the unobserved hours) before the new one opens.
  model.addExposure(1.0, 5000.0);

  EXPECT_EQ(model.nudft.n, 1);
  EXPECT_NEAR(model.win_t0, 5000.0, 1.0e-12);  // fresh window
  EXPECT_NEAR(model.win_events, 0.0, 1.0e-12);
}

TEST(PoissonModel, WindowedFeedingRecoversSinusoidalRate) {
  // Ground truth: a rate oscillating with period 1 h, always observed. The
  // windowed signal must give the spectral predictor a usable day-shape:
  // prediction at the peak phase clearly above prediction at the trough phase.
  const double period_s = 3600.0;
  const auto omegas = std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / period_s});

  PoissonModel windowed;
  windowed.nudft.omegas_ptr = omegas;
  windowed.nudft.min_observations = 5;
  windowed.nudft.reset();
  windowed.rate_window_s = 60.0;

  // 4 hours of 1 Hz exposure; events arrive where the sinusoid is high
  // (deterministic thinning of a 1 Hz event stream).
  double acc_w = 0.0;
  for (int i = 0; i < 4 * 3600; ++i) {
    const double t = static_cast<double>(i);
    const double rate = 0.5 + 0.4 * std::sin(2.0 * M_PI * t / period_s);
    windowed.addExposure(1.0, t);
    acc_w += rate;
    if (acc_w >= 1.0) {
      acc_w -= 1.0;
      windowed.recordEvent(t);
    }
  }

  const double t_peak = 4.0 * period_s + 0.25 * period_s;    // sin = +1
  const double t_trough = 4.0 * period_s + 0.75 * period_s;  // sin = -1
  const double swing_true = 0.8;  // 0.9 - 0.1

  const double w_peak = windowed.predictLambda(t_peak, 1, 5);
  const double w_trough = windowed.predictLambda(t_trough, 1, 5);
  EXPECT_GT(w_peak - w_trough, 0.5 * swing_true);  // captures most of the swing
}

}  // namespace kairos
