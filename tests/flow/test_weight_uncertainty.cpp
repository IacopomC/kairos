#include <gtest/gtest.h>
#include <kairos/flow/cell_state.h>
#include <kairos/flow/swgmm.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

FlowMapConfig weightConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.nudft_order = 1;
  cfg.candidate_periods = {3600.0f};
  cfg.weight_uncertainty_noise_sigma = 0.3;
  return cfg;
}

std::shared_ptr<const std::vector<double>> testOmegas(const FlowMapConfig& cfg) {
  std::vector<double> omegas;
  omegas.reserve(cfg.candidate_periods.size());
  for (const auto period : cfg.candidate_periods) {
    if (period > 0.0f) {
      omegas.push_back(2.0 * M_PI / static_cast<double>(period));
    }
  }
  return std::make_shared<const std::vector<double>>(std::move(omegas));
}

/// One detection without a track id, fed the way FlowMap feeds it: the cell
/// update, then one weights sample from the detection's responsibilities.
void observe(FlowCellState& cell, double theta, double rho, double t_seconds,
             const FlowMapConfig& cfg) {
  std::vector<double> r;
  cell.update(theta, rho, t_seconds, 1.0, cfg, &r);
  if (!r.empty()) {
    cell.feedWeightsSample(t_seconds, r, cfg);
  }
}

void feedModelBased(FlowCellState& cell, const FlowMapConfig& cfg, int n = 25) {
  for (int i = 0; i < n; ++i) {
    observe(cell, 0.5, 1.1, static_cast<double>(i) * 100.0, cfg);
  }
}

}  // namespace

TEST(WeightUncertainty, VarianceVectorsAlignedFiniteNonNegative) {
  auto cfg = weightConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));
  feedModelBased(cell, cfg);

  const auto pred = cell.predict(2500.0, 1.0, cfg);
  ASSERT_EQ(pred.pi.size(), 8u);
  ASSERT_EQ(pred.var_pi.size(), 8u);
  ASSERT_EQ(pred.pi_clamped.size(), 8u);
  for (const auto v : pred.var_pi) {
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GE(v, 0.0);
  }
}

// The propagated variance must describe the same reconstruction the reported
// weight comes from: same per-slot gated order, same shared-mean substitution,
// with the delta-method combination of the per-slot posterior variances. The
// recomputation below goes through the public model state, so any divergence
// between the mean path and the variance path (e.g. a fixed-order variance
// under a gated-order mean) fails this test.
TEST(WeightUncertainty, VarianceMatchesDeltaFormulaAtEffectiveOrder) {
  const auto cfg = weightConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));
  feedModelBased(cell, cfg, 40);

  const double t = 4200.0;
  const auto pred = cell.predict(t, 1.0, cfg);
  ASSERT_EQ(pred.var_pi.size(), 8u);

  std::vector<double> raw(8), v(8);
  double S = 0.0;
  double sum_v = 0.0;
  for (int k = 0; k < 8; ++k) {
    const auto& m = cell.nudft_models[static_cast<size_t>(k)];
    raw[static_cast<size_t>(k)] =
        m.predictGatedWithMean(t, cfg.nudft_order, cell.sharedMean(k, cfg, nullptr));
    const int eff = m.n >= m.min_observations ? m.bestOrder(cfg.nudft_order) : 0;
    v[static_cast<size_t>(k)] = m.predictionVariance(t, eff);
    S += std::clamp(raw[static_cast<size_t>(k)], 1.0e-3, 1.0);
    sum_v += v[static_cast<size_t>(k)];
  }
  for (int k = 0; k < 8; ++k) {
    const double p = pred.pi[static_cast<size_t>(k)];
    const double expected =
        std::max(0.0, ((1.0 - 2.0 * p) * v[static_cast<size_t>(k)] + p * p * sum_v) / (S * S));
    EXPECT_NEAR(pred.var_pi[static_cast<size_t>(k)], expected, 1.0e-12) << "slot " << k;
  }
}

// Every vote in one direction: the antipodal slots accumulate means far below
// the normalizer's 0.001 floor, so their clamp indicator must be set, while
// the dominant slot's must not.
TEST(WeightUncertainty, ClampFlagSetOnStarvedSlots) {
  auto cfg = weightConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));
  for (int i = 0; i < 40; ++i) {
    observe(cell, 0.0, 1.1, static_cast<double>(i) * 100.0, cfg);
  }

  const auto pred = cell.predict(4200.0, 1.0, cfg);
  ASSERT_EQ(pred.pi_clamped.size(), 8u);
  EXPECT_EQ(pred.pi_clamped[0], 0);  // dominant (theta = 0) slot
  EXPECT_NE(pred.pi_clamped[4], 0);  // antipodal (theta = pi) slot
}

TEST(WeightUncertainty, SnapshotStdIsSqrtOfPredictionVariance) {
  auto cfg = weightConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));
  feedModelBased(cell, cfg);

  const double t = 2500.0;
  const auto pred = cell.predict(t, 1.0, cfg);
  const auto snap = cell.snapshot(t, 1.0, cfg);
  ASSERT_EQ(snap.components.size(), 8u);
  for (int k = 0; k < 8; ++k) {
    EXPECT_NEAR(snap.components[static_cast<size_t>(k)].std_pi,
                std::sqrt(pred.var_pi[static_cast<size_t>(k)]), 1.0e-12);
  }
  EXPECT_EQ(snap.n_crossings, cell.n_crossings);
  EXPECT_GT(snap.n_crossings, 0);
}

// The variance shrink is driven by the weight-channel noise scale: a smaller
// weight sigma shrinks the weight posterior faster.
TEST(WeightUncertainty, WeightNoiseScaleDrivesShrink) {
  auto cfg_small = weightConfig();
  cfg_small.weight_uncertainty_noise_sigma = 0.05;
  auto cfg_large = weightConfig();
  cfg_large.weight_uncertainty_noise_sigma = 1.0;

  FlowCellState a(std::make_unique<FixedComponents>(cfg_small), cfg_small,
                  testOmegas(cfg_small));
  FlowCellState b(std::make_unique<FixedComponents>(cfg_large), cfg_large,
                  testOmegas(cfg_large));
  feedModelBased(a, cfg_small);
  feedModelBased(b, cfg_large);

  const double va = a.nudft_models[0].predictionVariance(2500.0, 0);
  const double vb = b.nudft_models[0].predictionVariance(2500.0, 0);
  EXPECT_LT(va, vb);
  EXPECT_GT(va, 0.0);
}

}  // namespace kairos
