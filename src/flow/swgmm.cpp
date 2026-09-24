#include "kairos/flow/swgmm.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace kairos {

namespace {

constexpr double kMinLikelihood = 1.0e-300;

// Scale a vector of non-negative values so it sums to 1. If the total is zero,
// fall back to a uniform distribution.
std::vector<double> normalize(const std::vector<double>& values) {
  std::vector<double> out = values;
  double sum = std::accumulate(out.begin(), out.end(), 0.0);
  if (sum <= 0.0) {
    const double uniform = out.empty() ? 0.0 : 1.0 / static_cast<double>(out.size());
    std::fill(out.begin(), out.end(), uniform);
    return out;
  }

  for (auto& value : out) {
    value /= sum;
  }

  return out;
}

}  // namespace

std::vector<SWNDComponent> SWGMMBase::exportComponents() const {
  std::vector<SWNDComponent> components;
  const int K = std::max(0, numComponents());
  components.reserve(static_cast<size_t>(K));
  for (int k = 0; k < K; ++k) {
    components.push_back(component(k));
  }

  return components;
}

FixedComponents::FixedComponents(const FlowMapConfig& cfg) {
  components_.resize(8);
  const double var_theta = cfg.fixed_sigma_theta * cfg.fixed_sigma_theta;
  const double var_rho = cfg.fixed_sigma_rho * cfg.fixed_sigma_rho;
  const double cov = cfg.fixed_sigma_cross;

  for (int k = 0; k < 8; ++k) {
    auto& c = components_[static_cast<size_t>(k)];
    c.mu_theta = static_cast<double>(k) * M_PI / 4.0;
    c.mu_rho = cfg.fixed_mu_rho;
    c.Sigma << var_theta, cov, cov, var_rho;
    precompute(c);
  }
}

int FixedComponents::numComponents() const { return static_cast<int>(components_.size()); }

std::vector<double> FixedComponents::responsibilities(double theta, double rho) const {
  std::vector<double> likelihoods(components_.size(), kMinLikelihood);
  for (size_t k = 0; k < components_.size(); ++k) {
    likelihoods[k] = std::max(evaluate(components_[k], theta, rho, 2), kMinLikelihood);
  }

  return normalize(likelihoods);
}

double FixedComponents::logLikelihood(double theta,
                                     double rho,
                                     const std::vector<double>& pi) const {
  const double uniform = components_.empty() ? 0.0 : 1.0 / static_cast<double>(components_.size());
  double value = 0.0;
  for (size_t k = 0; k < components_.size(); ++k) {
    const double weight = k < pi.size() ? pi[k] : uniform;
    value += std::max(weight, 1.0e-9) *
             std::max(evaluate(components_[k], theta, rho, 2), kMinLikelihood);
  }

  return std::log(std::max(value, kMinLikelihood));
}

const SWNDComponent& FixedComponents::component(int k) const {
  return components_.at(static_cast<size_t>(k));
}

}  // namespace kairos
