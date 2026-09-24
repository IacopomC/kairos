#include <gtest/gtest.h>
#include <kairos/flow/cell_state.h>
#include <kairos/flow/swgmm.h>

#include <cmath>
#include <memory>
#include <vector>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

FlowMapConfig fixedConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.candidate_periods = {3600.0f};
  return cfg;
}

// Tests construct cells in isolation; in production each FlowMap owns one
// shared_omegas across all its cells. Build a one-off allocation here so the
// constructor signature matches.
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

}  // namespace

TEST(FlowCellState, DiscardZeroSpeed) {
  auto cfg = fixedConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));

  observe(cell, 1.0, 0.0, 0.0, cfg);
  observe(cell, 1.0, -1.0, 1.0, cfg);
  EXPECT_EQ(cell.total_observations, 0);
}

TEST(FlowCellState, PredictFallbackWithNoObs) {
  auto cfg = fixedConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));

  const auto pred = cell.predict(0.0, 1.0, cfg);
  EXPECT_EQ(pred.flag, CellFlag::FALLBACK);
}

TEST(FlowCellState, PredictColdStartBeforeNudft) {
  auto cfg = fixedConfig();
  cfg.nudft_min_obs = 1000;
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));

  for (int i = 0; i < 20; ++i) {
    observe(cell, 1.0, 1.0, static_cast<double>(i), cfg);
  }

  const auto pred = cell.predict(20.0, 1.0, cfg);
  EXPECT_EQ(pred.flag, CellFlag::COLD_START);
}

TEST(FlowCellState, PredictionPiSumsToOne) {
  auto cfg = fixedConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));

  for (int i = 0; i < 20; ++i) {
    observe(cell, 0.5, 1.0, static_cast<double>(i), cfg);
  }

  const auto pred = cell.predict(20.0, 1.0, cfg);
  double sum = 0.0;
  for (const auto v : pred.pi) {
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0, 1.0e-6);
}

TEST(FlowCellState, SnapshotHasComponents) {
  auto cfg = fixedConfig();
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));

  for (int i = 0; i < 5; ++i) {
    observe(cell, 0.5, 1.0, static_cast<double>(i), cfg);
  }

  const auto snap = cell.snapshot(5.0, 1.0, cfg);
  EXPECT_EQ(snap.components.size(), 8u);
}

TEST(FlowCellState, NudftBecomesModelBasedAfterMinObs) {
  auto cfg = fixedConfig();
  cfg.nudft_min_obs = 5;
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));

  for (int i = 0; i < 25; ++i) {
    observe(cell, 0.5, 1.1, static_cast<double>(i) * 100.0, cfg);
  }

  EXPECT_TRUE(cell.nudft_valid);
  const auto pred = cell.predict(2500.0, 1.0, cfg);
  EXPECT_EQ(pred.flag, CellFlag::MODEL_BASED);
}

}  // namespace kairos
