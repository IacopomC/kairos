#pragma once

/**
 * @file swnd.h
 * @brief A single semi-wrapped normal component over (heading, speed): the
 *        component parameters (mean, covariance and its cached inverse /
 *        determinant) and the routines that evaluate its bivariate density and
 *        the two 1D marginals. "Semi-wrapped" because the heading axis is
 *        circular (wrapped over 2*pi) while the speed axis is linear.
 */

#include <Eigen/Core>
#include <Eigen/LU>

namespace kairos {

/// One semi-wrapped normal component over (heading, speed). Sigma is the
/// covariance; SigmaInv and SigmaDet are its cached inverse and determinant,
/// (re)computed by precompute().
struct SWNDComponent {
  double mu_theta = 0.0;                                 ///< Mean heading (rad, circular).
  double mu_rho = 0.0;                                   ///< Mean speed (m/s, linear).
  Eigen::Matrix2d Sigma = Eigen::Matrix2d::Identity();   ///< Covariance over (theta, rho).
  Eigen::Matrix2d SigmaInv = Eigen::Matrix2d::Identity();  ///< Cached inverse of Sigma.
  double SigmaDet = 1.0;                                 ///< Cached determinant of Sigma.
};

/// Recompute the cached SigmaInv / SigmaDet from Sigma. Call before evaluate().
void precompute(SWNDComponent& c);
/// Bivariate density at (theta, rho); @p winding_terms wraps the heading axis.
double evaluate(const SWNDComponent& c, double theta, double rho, int winding_terms = 2);

/// Marginal densities of the bivariate semi-wrapped normal, obtained by
/// integrating out the other variable. The marginal over rho leaves a 1D
/// wrapped normal in theta with variance Sigma(0,0); the marginal over theta
/// leaves a 1D Gaussian in rho with variance Sigma(1,1) (marginal variances are
/// the diagonal of Sigma regardless of the cross-covariance). Call precompute(c)
/// first.
double evaluateThetaMarginal(const SWNDComponent& c, double theta, int winding_terms = 2);
double evaluateRhoMarginal(const SWNDComponent& c, double rho);

}  // namespace kairos
