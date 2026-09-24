#include "kairos/flow/config.h"

#include <config_utilities/validation.h>

#include <string>
#include <vector>

namespace kairos {

void declare_config(FlowMapConfig& config) {
  using namespace config;
  name("FlowMapConfig");
  field(config.presence_rate_window_s, "presence_rate_window_s");
  check(config.presence_rate_window_s, GT, 0.0, "presence_rate_window_s");
  field(config.visit_gap_s, "visit_gap_s");
  check(config.visit_gap_s, GT, 0.0, "visit_gap_s");
  field(config.visit_max_s, "visit_max_s");
  check(config.visit_max_s, GT, 0.0, "visit_max_s");

  field(config.fixed_mu_rho, "fixed_mu_rho");
  check(config.fixed_mu_rho, GT, 0.0, "fixed_mu_rho");
  field(config.fixed_sigma_theta, "fixed_sigma_theta");
  check(config.fixed_sigma_theta, GT, 0.0, "fixed_sigma_theta");
  field(config.fixed_sigma_rho, "fixed_sigma_rho");
  check(config.fixed_sigma_rho, GT, 0.0, "fixed_sigma_rho");
  field(config.fixed_sigma_cross, "fixed_sigma_cross");

  field(config.speed_scatter_prior_obs, "speed_scatter_prior_obs");
  check(config.speed_scatter_prior_obs, GE, 0.0, "speed_scatter_prior_obs");

  field(config.enable_timing, "enable_timing");
  field(config.nudft_order, "nudft_order");
  check(config.nudft_order, GE, 0, "nudft_order");
  field(config.nudft_min_obs, "nudft_min_obs");
  check(config.nudft_min_obs, GE, 1, "nudft_min_obs");

  field(config.voxel_size_m, "voxel_size_m");
  check(config.voxel_size_m, GT, 0.0, "voxel_size_m");

  field(config.candidate_periods, "candidate_periods");
  check(config.candidate_periods.size(), GT, 0u, "candidate_periods_size");

  field(config.sharing_lent_crossings, "sharing_lent_crossings");
  check(config.sharing_lent_crossings, GE, 0.0, "sharing_lent_crossings");

  field(config.weight_uncertainty_noise_sigma, "weight_uncertainty_noise_sigma");
  check(config.weight_uncertainty_noise_sigma, GT, 0.0, "weight_uncertainty_noise_sigma");

  field(config.edge_coupling_connectivity, "edge_coupling_connectivity");
  field(config.edge_coupling_delta_t_coh, "edge_coupling_delta_t_coh");
  check(config.edge_coupling_delta_t_coh, GT, 0.0, "edge_coupling_delta_t_coh");
  field(config.edge_coupling_samples_per_period, "edge_coupling_samples_per_period");
  check(config.edge_coupling_samples_per_period, GE, 2, "edge_coupling_samples_per_period");
  field(config.edge_coupling_window_min_pairs, "edge_coupling_window_min_pairs");
  check(config.edge_coupling_window_min_pairs, GE, 1, "edge_coupling_window_min_pairs");

  field(config.stability_window, "stability_window");
  check(config.stability_window, GE, 1, "stability_window");
}

}  // namespace kairos
