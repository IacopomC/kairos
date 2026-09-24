#include <gtest/gtest.h>
#include <kairos/flow/edge_coupling_map.h>
#include <kairos/flow/flow_map.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

constexpr int K = 8;
// Sampling window of the coupling predictors: the 3600 s candidate period at
// 12 samples per period, as the shipped configs derive it.
constexpr double kWindowS = 300.0;
constexpr int kWindowMinPairs = 2;

std::shared_ptr<const std::vector<double>> sharedOmegas() {
  return std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / 3600.0});
}

std::vector<double> oneHot(int slot) {
  std::vector<double> v(K, 0.0);
  v[static_cast<size_t>(slot)] = 1.0;
  return v;
}

// Empirical marginal of the synthetic corridor below: half the observations in
// slot 2, half in slot 6. conditionalFlow weights its result by this marginal,
// so a query conditioned on slot 2 starts at p=0.5 there and only the coupling
// can push it higher.
std::vector<double> marginalTwoSix() {
  std::vector<double> v(K, 0.0);
  v[2] = 0.5;
  v[6] = 0.5;
  return v;
}

int argmax(const std::vector<double>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

FlowMapConfig fixedConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.candidate_periods = {3600.0f};
  return cfg;
}

}  // namespace

// A perfectly co-moving corridor: the two adjacent cells always occupy the same
// slot (alternating between 2 and 6 so the marginals are non-degenerate). The
// learned coupling must then sharpen the conditional-flow query toward the
// conditioning slot, well above its 0.5 marginal weight.
TEST(EdgeCoupling, LearnsCoMovementFromResponsibilities) {
  auto omegas = sharedOmegas();
  EdgeCouplingMap cmap(omegas, /*nudft_min_obs=*/5, kWindowS, kWindowMinPairs);
  const GlobalIndex i(0, 0, 0), j(1, 0, 0);

  for (int n = 0; n < 200; ++n) {
    const int slot = (n % 2 == 0) ? 2 : 6;  // both cells move the same way
    cmap.update(i, oneHot(slot), j, oneHot(slot), static_cast<double>(n) * 100.0);
  }

  auto q2 = cmap.conditionalFlow(i, /*k_i=*/2, j, marginalTwoSix(), 20000.0, /*order=*/1);
  ASSERT_TRUE(q2.has_value());
  EXPECT_EQ(argmax(*q2), 2);
  EXPECT_GT((*q2)[2], 0.9) << "coupling did not sharpen toward the co-moving slot";
  EXPECT_LT((*q2)[6], 0.1);

  auto q6 = cmap.conditionalFlow(i, 6, j, marginalTwoSix(), 20000.0, 1);
  ASSERT_TRUE(q6.has_value());
  EXPECT_EQ(argmax(*q6), 6);
  EXPECT_GT((*q6)[6], 0.9);
}

// Regression guard for the responsibilities-vs-pi bug. Feeding the SAME
// time-averaged vector every step (what a cell's predicted mixing weights pi
// would supply) carries no per-observation co-movement, so the co-occurrence
// tensor equals the outer product of the marginals and the learned coupling is
// flat. The conditional query must then just return the marginal (~0.5), not a
// sharpened distribution. This is exactly the failure that shipped.
TEST(EdgeCoupling, AveragedInputLearnsNoCoupling) {
  auto omegas = sharedOmegas();
  EdgeCouplingMap cmap(omegas, 5, kWindowS, kWindowMinPairs);
  const GlobalIndex i(0, 0, 0), j(1, 0, 0);

  const std::vector<double> averaged = marginalTwoSix();  // identical every step
  for (int n = 0; n < 200; ++n) {
    cmap.update(i, averaged, j, averaged, static_cast<double>(n) * 100.0);
  }

  auto q = cmap.conditionalFlow(i, 2, j, marginalTwoSix(), 20000.0, 1);
  ASSERT_TRUE(q.has_value());
  EXPECT_NEAR((*q)[2], 0.5, 0.1)
      << "averaged input produced spurious coupling; the conditional should "
         "stay at the marginal";
}

// The seam itself: FlowMap must expose the PER-OBSERVATION responsibility,
// peaked at the observed heading's slot. This is the quantity the coupling
// update must consume; the bug fed the time-averaged predicted weights instead.
TEST(EdgeCoupling, FlowMapResponsibilityIsPerObservationPeak) {
  FlowMap map(fixedConfig());
  const GlobalIndex idx(0, 0, 0);
  const double theta = M_PI / 2.0;  // slot 2 (k*pi/4, k=2)
  for (int n = 0; n < 10; ++n) {
    map.addObservation(idx, theta, /*rho=*/1.1, static_cast<double>(n) * 100.0, 0.1);
  }

  auto resp = map.responsibilities(idx, theta, 1.1);
  ASSERT_EQ(static_cast<int>(resp.size()), K);
  EXPECT_EQ(argmax(resp), 2) << "responsibility is not peaked at the observed heading";
  EXPECT_GT(resp[2], 0.4);

  // An unmodelled cell yields an empty vector, never a fabricated one.
  EXPECT_TRUE(map.responsibilities(GlobalIndex(9, 9, 9), theta, 1.1).empty());
}

// Serialization must preserve the learned coupling: a conditional query after a
// round trip matches the pre-save query, so a silently reset tensor is caught.
TEST(EdgeCoupling, RoundTripPreservesLearnedCoupling) {
  auto omegas = sharedOmegas();
  EdgeCouplingMap cmap(omegas, 5, kWindowS, kWindowMinPairs);
  const GlobalIndex i(0, 0, 0), j(1, 0, 0);
  for (int n = 0; n < 200; ++n) {
    const int slot = (n % 2 == 0) ? 2 : 6;
    cmap.update(i, oneHot(slot), j, oneHot(slot), static_cast<double>(n) * 100.0);
  }
  auto before = cmap.conditionalFlow(i, 2, j, marginalTwoSix(), 20000.0, 1);
  ASSERT_TRUE(before.has_value());

  EdgeCouplingMap restored(omegas, 5, kWindowS, kWindowMinPairs);
  std::string err;
  ASSERT_TRUE(restored.fromJson(cmap.toJson(), &err)) << err;
  auto after = restored.conditionalFlow(i, 2, j, marginalTwoSix(), 20000.0, 1);
  ASSERT_TRUE(after.has_value());
  for (int k = 0; k < K; ++k) {
    EXPECT_NEAR((*before)[static_cast<size_t>(k)], (*after)[static_cast<size_t>(k)], 1e-9);
  }
}

}  // namespace kairos
