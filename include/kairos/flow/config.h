#pragma once

/**
 * @file config.h
 * @brief Configuration for the flow map and the per-cell models it owns. A
 *        single FlowMapConfig struct collects the numeric parameters for the
 *        directional mixture, the spectral weight and presence predictors,
 *        evidence sharing, predictive uncertainty, edge coupling and the
 *        stability metric, together with the declare_config hook used to load
 *        and validate them.
 */

#include <config_utilities/config.h>

#include <vector>

namespace kairos {

struct FlowMapConfig {
  /// Wall-clock length (seconds) of the exposure window that feeds the presence
  /// spectral predictor one windowed-rate sample (events per visible second) at
  /// the window mid-time. Must be positive and sit well below the shortest
  /// candidate period so each cycle is sampled at many phases. Dataset-scale
  /// parameter: set per dataset next to `candidate_periods`. The default suits
  /// the default one-hour shortest period.
  double presence_rate_window_s = 120.0;

  /// The mixing-weight spectral models are fed one sample per (track, voxel)
  /// visit: the mean responsibility over the crossing, timestamped at the
  /// visit's mid-time. Detections without a track id are fed one sample each.
  /// A track's open visit closes when this much time passes without a new
  /// detection from it (it left the cell or the field of view).
  double visit_gap_s = 2.0;

  /// Hard cap on a visit's span: a mover lingering in one cell longer than
  /// this closes its visit and opens a new one, so one agent cannot hold a
  /// vote open indefinitely.
  double visit_max_s = 10.0;

  /// Fixed initial component parameters for the directional mixture, applied to
  /// every cardinal slot at creation. mu_rho is the prior mean speed (m/s);
  /// sigma_theta / sigma_rho are the prior heading (rad) and speed (m/s)
  /// standard deviations; sigma_cross is their prior covariance.
  double fixed_mu_rho = 1.1;
  double fixed_sigma_theta = 0.4;
  double fixed_sigma_rho = 0.3;
  double fixed_sigma_cross = 0.0;

  /// Weight, in observations, of the prior spread `fixed_sigma_rho` in a
  /// slot's speed uncertainty. The reported speed variance blends that prior
  /// with the slot's measured speed scatter as
  /// (kappa * prior + n * scatter) / (kappa + n), so a new slot reports the
  /// prior spread and a well-observed one its own.
  double speed_scatter_prior_obs = 5.0;

  /// Accumulate per-component timing statistics for the map's hot paths. Adds a
  /// small per-call overhead; all timing counters stay zero when false.
  bool enable_timing = false;

  /// Highest spectral order (frequency components beyond the DC term) the
  /// order gate may select for a mixing-weight or presence predictor. Each
  /// predictor selects its own order online, by accumulated one-step-ahead
  /// (prequential) error, so harmonics engage only where they lower held-out
  /// error and stationary cells stay at order 0.
  int nudft_order = 3;

  /// Minimum number of observations a spectral predictor needs before it is
  /// trusted; below this it falls back to the running mean.
  int nudft_min_obs = 20;

  /// Physical side length of a flow voxel (metres). Converts a cell's mean
  /// speed into a dwell time (cell / speed) for the occupancy presence model.
  /// MUST match the voxel size of the reconstruction map this flow runs on;
  /// kept here as a separate field because the flow map works in integer voxel
  /// coordinates and never otherwise receives the physical size.
  double voxel_size_m = 0.4;

  /// Candidate periods (seconds) whose angular frequencies the spectral
  /// predictors are evaluated over — here one week down to one hour.
  std::vector<float> candidate_periods = {604800.0f,
                                          259200.0f,
                                          172800.0f,
                                          86400.0f,
                                          43200.0f,
                                          28800.0f,
                                          21600.0f,
                                          14400.0f,
                                          10800.0f,
                                          7200.0f,
                                          3600.0f};

  /// Read-time evidence sharing on the mixing weights. The weight a cell
  /// reports is its own running mean shrunk toward the count-weighted mean of
  /// its traversable neighbourhood, which lends it sharing_lent_crossings
  /// crossings of evidence. The neighbourhood therefore holds a share m/(C+m)
  /// of the reported weight, falling to zero as the cell accumulates crossings
  /// C of its own. Nothing is written into the stored state.
  /// Crossings of neighbourhood evidence lent to a cell (m). At the default the
  /// neighbourhood holds about a quarter of the reported weight by nudft_min_obs
  /// crossings, the point from which a cell forecasts its own periodic
  /// structure, and under a twentieth once it has ten times that.
  double sharing_lent_crossings = 3.0;


  /// Observation-noise standard deviation assumed by the mixing-weight
  /// predictors' posterior. The weight targets are soft responsibilities in
  /// [0, 1], a different scale from the speed channel's m/s, so the two
  /// channels carry separate noise fields; this one sets the shrink rate of
  /// the weight coefficient variances alone. Set it per dataset from the
  /// training-scene scatter of responsibilities about the fitted weights.
  double weight_uncertainty_noise_sigma = 0.3;

  /// Pairwise coupling between the mixtures of adjacent voxels. connectivity
  /// selects the neighbourhood: 6 = face-adjacent only, 26 = full cube.
  /// edge_coupling_delta_t_coh is the time window (seconds) within which two
  /// detections count as co-occurring for coupling estimation.
  int edge_coupling_connectivity = 6;
  double edge_coupling_delta_t_coh = 1.0;

  /// How many log-PMI samples an edge's spectral predictors receive per cycle
  /// of the shortest entry in `candidate_periods`. The sampling window follows
  /// from it, window_s = min(candidate_periods) / this, so the rate is stated
  /// against the rhythms being fitted rather than as a raw duration that has
  /// to be re-derived whenever the candidate periods change. Each closed
  /// window contributes one sample, computed from that window's own
  /// co-occurrences and marginals and timestamped at its mid-time, so the
  /// stream is a sequence of time-local measurements of the coupling.
  /// Values below 2 cannot resolve the shortest period at all; 8 to 12 leaves
  /// margin.
  int edge_coupling_samples_per_period = 12;
  /// Minimum paired observations a window must hold before it may close. A
  /// window with a single pair yields phi == 0 identically, carrying no
  /// information, so short-changing this dilutes the sample stream. Windows
  /// below the threshold stay open and keep accumulating past their nominal
  /// length.
  int edge_coupling_window_min_pairs = 2;

  /// Time-to-stability metric. A cell is declared "stable" once its dominant
  /// slot (the slot with the largest DC speed term) is unchanged across this
  /// many consecutive observations; the cell then records the observation count
  /// at which that streak began.
  int stability_window = 5;
};

void declare_config(FlowMapConfig& config);

}  // namespace kairos
