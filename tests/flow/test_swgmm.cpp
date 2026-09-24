#include <gtest/gtest.h>
#include <kairos/flow/swgmm.h>

#include <cmath>
#include <numeric>

namespace kairos {

FlowMapConfig makeDefaultConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.fixed_sigma_cross = 0.0;
  return cfg;
}

TEST(FixedComponents, HasEightComponents) {
  FixedComponents model(makeDefaultConfig());
  EXPECT_EQ(model.numComponents(), 8);
}

TEST(FixedComponents, ComponentsAt45DegreeIntervals) {
  FixedComponents model(makeDefaultConfig());
  for (int k = 0; k < 8; ++k) {
    const auto& c = model.component(k);
    EXPECT_NEAR(c.mu_theta, static_cast<double>(k) * M_PI / 4.0, 1.0e-10);
    EXPECT_NEAR(c.mu_rho, 1.1, 1.0e-10);
  }
}

TEST(FixedComponents, ResponsibilitiesSumToOne) {
  FixedComponents model(makeDefaultConfig());
  const auto r = model.responsibilities(1.0, 1.1);

  ASSERT_EQ(r.size(), 8u);
  const double sum = std::accumulate(r.begin(), r.end(), 0.0);
  EXPECT_NEAR(sum, 1.0, 1.0e-10);
}

TEST(FixedComponents, ResponsibilitiesPeakNearMatchingComponent) {
  FixedComponents model(makeDefaultConfig());
  const double target_theta = M_PI / 4.0;
  const auto r = model.responsibilities(target_theta, 1.1);

  double max_r = 0.0;
  int best_k = -1;
  for (size_t k = 0; k < r.size(); ++k) {
    if (r[k] > max_r) {
      max_r = r[k];
      best_k = static_cast<int>(k);
    }
  }
  EXPECT_EQ(best_k, 1);
}

TEST(FixedComponents, LogLikelihoodIsFinite) {
  FixedComponents model(makeDefaultConfig());
  std::vector<double> pi(8, 1.0 / 8.0);
  const double ll = model.logLikelihood(1.0, 1.1, pi);
  EXPECT_TRUE(std::isfinite(ll));
}

TEST(FixedComponents, OffDiagonalCovariance) {
  auto cfg = makeDefaultConfig();
  cfg.fixed_sigma_cross = 0.05;
  FixedComponents model(cfg);

  const auto r = model.responsibilities(0.5, 1.1);
  const double sum = std::accumulate(r.begin(), r.end(), 0.0);
  EXPECT_NEAR(sum, 1.0, 1.0e-10);
}

}  // namespace kairos
