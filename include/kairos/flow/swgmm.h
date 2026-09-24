#pragma once

/**
 * @file swgmm.h
 * @brief Semi-wrapped Gaussian mixture over the (heading, speed) plane: the
 *        SWGMMBase interface that scores an observation against a set of mixture
 *        components and reports their responsibilities, plus the FixedComponents
 *        variant whose component headings are pinned to a fixed cardinal layout
 *        so only the mixing weights vary over time.
 */

#include <vector>

#include "kairos/flow/config.h"
#include "kairos/flow/swnd.h"

namespace kairos {

/// Interface for a semi-wrapped Gaussian mixture over (heading, speed).
class SWGMMBase {
 public:
  virtual ~SWGMMBase() = default;

  /// Number of mixture components.
  virtual int numComponents() const = 0;
  /// Per-component responsibility (soft assignment) of the observation (theta, rho).
  virtual std::vector<double> responsibilities(double theta, double rho) const = 0;
  /// Log-likelihood of (theta, rho) under the mixture with mixing weights @p pi.
  virtual double logLikelihood(double theta,
                               double rho,
                               const std::vector<double>& pi) const = 0;
  /// Access the k-th component's parameters.
  virtual const SWNDComponent& component(int k) const = 0;

  /// Copy out all components (e.g. for serialization).
  std::vector<SWNDComponent> exportComponents() const;
};

/// Mixture whose component means are fixed at construction (a cardinal heading
/// layout with config-supplied speed and covariance), so only the mixing
/// weights — tracked elsewhere — vary over time.
class FixedComponents : public SWGMMBase {
 public:
  explicit FixedComponents(const FlowMapConfig& cfg);

  int numComponents() const override;
  std::vector<double> responsibilities(double theta, double rho) const override;
  double logLikelihood(double theta, double rho, const std::vector<double>& pi) const override;
  const SWNDComponent& component(int k) const override;

 private:
  std::vector<SWNDComponent> components_;
};

}  // namespace kairos
