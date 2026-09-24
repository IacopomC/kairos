#include <gtest/gtest.h>
#include <kairos/flow/swnd.h>

#include <cmath>

namespace kairos {

TEST(SWND, PrecomputeSetsDeterminantAndInverse) {
  SWNDComponent c;
  c.mu_theta = 0.0;
  c.mu_rho = 1.0;
  c.Sigma << 0.16, 0.0, 0.0, 0.09;
  precompute(c);

  // precompute adds a 1e-8 diagonal stability floor, so the determinant differs
  // from the bare 0.16*0.09 by ~2.5e-9; tolerate that intentional perturbation.
  EXPECT_NEAR(c.SigmaDet, 0.16 * 0.09, 1.0e-6);
  EXPECT_NEAR((c.Sigma * c.SigmaInv - Eigen::Matrix2d::Identity()).norm(), 0.0, 1.0e-10);
}

TEST(SWND, PrecomputeEnforcesSymmetry) {
  SWNDComponent c;
  c.Sigma << 0.16, 0.01, 0.02, 0.09;
  precompute(c);

  EXPECT_NEAR(c.Sigma(0, 1), c.Sigma(1, 0), 1.0e-15);
}

TEST(SWND, PrecomputeHandlesTinyDiagonal) {
  SWNDComponent c;
  c.Sigma << 0.0, 0.0, 0.0, 0.0;
  precompute(c);

  EXPECT_GT(c.SigmaDet, 0.0);
  EXPECT_TRUE(std::isfinite(c.SigmaInv(0, 0)));
}

TEST(SWND, EvaluateAtMeanIsPositive) {
  SWNDComponent c;
  c.mu_theta = 1.0;
  c.mu_rho = 1.5;
  c.Sigma << 0.16, 0.0, 0.0, 0.09;
  precompute(c);

  const double value = evaluate(c, 1.0, 1.5, 2);
  EXPECT_GT(value, 0.0);
}

TEST(SWND, EvaluateIsMaxAtMean) {
  SWNDComponent c;
  c.mu_theta = M_PI;
  c.mu_rho = 1.0;
  c.Sigma << 0.16, 0.0, 0.0, 0.09;
  precompute(c);

  const double at_mean = evaluate(c, M_PI, 1.0, 2);
  const double offset = evaluate(c, M_PI + 0.5, 1.3, 2);
  EXPECT_GT(at_mean, offset);
}

TEST(SWND, EvaluateWrapsAround2Pi) {
  SWNDComponent c;
  c.mu_theta = 0.1;
  c.mu_rho = 1.0;
  c.Sigma << 0.16, 0.0, 0.0, 0.09;
  precompute(c);

  const double near_zero = evaluate(c, 0.1, 1.0, 2);
  const double near_2pi = evaluate(c, 2.0 * M_PI + 0.1, 1.0, 2);
  EXPECT_NEAR(near_zero, near_2pi, 1.0e-10);
}

TEST(SWND, EvaluateWrappingSymmetry) {
  SWNDComponent c;
  c.mu_theta = 0.0;
  c.mu_rho = 1.0;
  c.Sigma << 0.16, 0.0, 0.0, 0.09;
  precompute(c);

  const double below = evaluate(c, -0.3, 1.0, 2);
  const double above = evaluate(c, 0.3, 1.0, 2);
  const double wrapped_below = evaluate(c, 2.0 * M_PI - 0.3, 1.0, 2);
  EXPECT_NEAR(below, above, 1.0e-10);
  EXPECT_NEAR(below, wrapped_below, 1.0e-10);
}

TEST(SWND, EvaluateWithOffDiagonalCovariance) {
  SWNDComponent c;
  c.mu_theta = 1.0;
  c.mu_rho = 1.0;
  c.Sigma << 0.16, 0.05, 0.05, 0.09;
  precompute(c);

  const double value = evaluate(c, 1.0, 1.0, 2);
  EXPECT_GT(value, 0.0);
  EXPECT_TRUE(std::isfinite(value));
}

TEST(SWND, EvaluateFarFromMeanIsSmall) {
  SWNDComponent c;
  c.mu_theta = 0.0;
  c.mu_rho = 1.0;
  c.Sigma << 0.04, 0.0, 0.0, 0.04;
  precompute(c);

  const double at_mean = evaluate(c, 0.0, 1.0, 2);
  const double far_away = evaluate(c, M_PI, 5.0, 2);
  EXPECT_GT(at_mean / std::max(far_away, 1.0e-300), 1.0e6);
}

}  // namespace kairos
