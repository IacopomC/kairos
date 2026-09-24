#include <gtest/gtest.h>
#include <kairos/flow/nudft.h>

#include <memory>

#include <cmath>

namespace kairos {

TEST(NUDFT, ResetClearsState) {
  NUDFTModel model;
  model.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{1.0, 2.0});
  model.min_observations = 2;
  model.reset();

  EXPECT_EQ(model.n, 0);
  EXPECT_NEAR(model.gamma0, 0.0, 1.0e-15);
  ASSERT_EQ(model.gammas.size(), 2u);
  EXPECT_NEAR(std::abs(model.gammas[0]), 0.0, 1.0e-15);
}

TEST(NUDFT, UpdateTracksDC) {
  NUDFTModel model;
  model.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.min_observations = 1;
  model.reset();

  model.update(0.0, 5.0);
  EXPECT_NEAR(model.gamma0, 5.0, 1.0e-10);
  EXPECT_EQ(model.n, 1);

  model.update(100.0, 3.0);
  EXPECT_NEAR(model.gamma0, 4.0, 1.0e-10);
  EXPECT_EQ(model.n, 2);
}

TEST(NUDFT, PredictReturnsDCWhenBelowMinObs) {
  NUDFTModel model;
  model.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{2.0 * M_PI / 3600.0});
  model.min_observations = 20;
  model.reset();

  for (int i = 0; i < 10; ++i) {
    model.update(static_cast<double>(i) * 100.0, 1.0);
  }

  EXPECT_NEAR(model.predict(500.0, 3), model.gamma0, 1.0e-15);
}

TEST(NUDFT, PredictRecoversSinusoid) {
  const double period = 3600.0;
  const double omega = 2.0 * M_PI / period;
  const double amplitude = 0.3;
  const double dc = 0.5;

  NUDFTModel model;
  model.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{omega});
  model.min_observations = 5;
  model.reset();

  // Train over several full periods so the online coefficient estimate (whose
  // residual uses the running mean, biased during cold start) converges, as it
  // does over multi-cycle deployment. A single period leaves a startup bias.
  const int cycles = 6;
  const int N = 1200;
  for (int i = 0; i < N; ++i) {
    const double t = static_cast<double>(i) * (cycles * period) / static_cast<double>(N);
    const double value = dc + amplitude * std::cos(omega * t);
    model.update(t, value);
  }

  EXPECT_NEAR(model.gamma0, dc, 0.05);

  for (int i = 0; i < 10; ++i) {
    const double t = static_cast<double>(i) * period / 10.0;
    const double expected = dc + amplitude * std::cos(omega * t);
    const double predicted = model.predict(t, 1);
    EXPECT_NEAR(predicted, expected, 0.1) << "at t=" << t;
  }
}

TEST(NUDFT, PredictWithMultipleFrequencies) {
  const double omega1 = 2.0 * M_PI / 3600.0;
  const double omega2 = 2.0 * M_PI / 7200.0;

  NUDFTModel model;
  model.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{omega1, omega2});
  model.min_observations = 5;
  model.reset();

  const int N = 400;
  for (int i = 0; i < N; ++i) {
    const double t = static_cast<double>(i) * 7200.0 / static_cast<double>(N);
    const double value = 0.5 + 0.2 * std::cos(omega1 * t) + 0.1 * std::cos(omega2 * t);
    model.update(t, value);
  }

  const double predicted = model.predict(0.0, 2);
  EXPECT_NEAR(predicted, 0.5 + 0.2 + 0.1, 0.15);
}

TEST(NUDFT, PredictWithZeroOrderReturnsDC) {
  NUDFTModel model;
  model.omegas_ptr = std::make_shared<const std::vector<double>>(std::vector<double>{1.0});
  model.min_observations = 1;
  model.reset();

  for (int i = 0; i < 30; ++i) {
    model.update(static_cast<double>(i), static_cast<double>(i % 5));
  }

  EXPECT_NEAR(model.predict(100.0, 0), model.gamma0, 1.0e-15);
}

TEST(NUDFTModel, GateSelectsOrderZeroOnStationarySignal) {
  NUDFTModel m;
  m.omegas_ptr = std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / 3600.0});
  m.min_observations = 20;
  m.reset();
  m.gate_enabled = true;
  for (int i = 0; i < 300; ++i) {
    // Stationary mean with deterministic, period-free wobble.
    m.update(i * 12.0, 0.5 + ((i % 2) ? 0.05 : -0.05));
  }
  EXPECT_EQ(m.bestOrder(1), 0);
  EXPECT_NEAR(m.predictGated(123.0, 1), m.gamma0, 1.0e-12);
}

TEST(NUDFTModel, GateSelectsHarmonicOnPeriodicSignal) {
  const double period = 3600.0;
  NUDFTModel m;
  m.omegas_ptr = std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / period});
  m.min_observations = 20;
  m.reset();
  m.gate_enabled = true;
  for (int i = 0; i < 600; ++i) {
    const double t = i * 12.0;
    m.update(t, 0.5 + 0.4 * std::sin(2.0 * M_PI * t / period));
  }
  EXPECT_EQ(m.bestOrder(1), 1);
  // The gated forecast tracks the oscillation: clearly above the mean at the
  // peak phase, clearly below at the trough.
  const double peak = m.predictGated(2.0 * period + 0.25 * period, 1);
  const double trough = m.predictGated(2.0 * period + 0.75 * period, 1);
  EXPECT_GT(peak - trough, 0.4);
}

}  // namespace kairos
