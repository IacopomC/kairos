#include "kairos/flow/swnd.h"

#include <cmath>

namespace kairos {

void precompute(SWNDComponent& c) {
  Eigen::Matrix2d sigma = c.Sigma;
  // .eval() forces a temporary: M = 0.5*(M + M.transpose()) aliases otherwise
  // (Eigen overwrites entries it still needs to read), corrupting off-diagonals.
  sigma = (0.5 * (sigma + sigma.transpose())).eval();
  sigma(0, 0) = std::max(sigma(0, 0), 1.0e-6);
  sigma(1, 1) = std::max(sigma(1, 1), 1.0e-6);
  sigma += 1.0e-8 * Eigen::Matrix2d::Identity();

  c.Sigma = sigma;
  c.SigmaDet = std::max(c.Sigma.determinant(), 1.0e-12);
  c.SigmaInv = c.Sigma.inverse();
}

double evaluate(const SWNDComponent& c, double theta, double rho, int winding_terms) {
  const double norm = 1.0 / (2.0 * M_PI * std::sqrt(std::max(c.SigmaDet, 1.0e-12)));
  double total = 0.0;

  for (int w = -winding_terms; w <= winding_terms; ++w) {
    const Eigen::Vector2d x(theta + 2.0 * M_PI * static_cast<double>(w), rho);
    const Eigen::Vector2d mu(c.mu_theta, c.mu_rho);
    const Eigen::Vector2d d = x - mu;
    const double exp_term = -0.5 * (d.transpose() * c.SigmaInv * d).value();
    total += norm * std::exp(exp_term);
  }

  return total;
}

double evaluateThetaMarginal(const SWNDComponent& c, double theta, int winding_terms) {
  const double var = std::max(c.Sigma(0, 0), 1.0e-12);
  const double norm = 1.0 / std::sqrt(2.0 * M_PI * var);
  double total = 0.0;
  for (int w = -winding_terms; w <= winding_terms; ++w) {
    const double d = theta + 2.0 * M_PI * static_cast<double>(w) - c.mu_theta;
    total += norm * std::exp(-0.5 * d * d / var);
  }
  return total;
}

double evaluateRhoMarginal(const SWNDComponent& c, double rho) {
  const double var = std::max(c.Sigma(1, 1), 1.0e-12);
  const double norm = 1.0 / std::sqrt(2.0 * M_PI * var);
  const double d = rho - c.mu_rho;
  return norm * std::exp(-0.5 * d * d / var);
}

}  // namespace kairos
