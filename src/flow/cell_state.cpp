#include "kairos/flow/cell_state.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace kairos {

namespace {

// Clamp each weight into [1e-3, 1] and renormalize so they sum to 1, with a
// uniform fallback when the total is non-positive.
std::vector<double> normalizePi(std::vector<double> pi) {
  for (auto& value : pi) {
    value = std::clamp(value, 1.0e-3, 1.0);
  }
  const double sum = std::accumulate(pi.begin(), pi.end(), 0.0);
  if (sum <= 0.0) {
    const double uniform = pi.empty() ? 0.0 : 1.0 / static_cast<double>(pi.size());
    std::fill(pi.begin(), pi.end(), uniform);
    return pi;
  }
  for (auto& value : pi) {
    value /= sum;
  }
  return pi;
}

}  // namespace

FlowCellState::FlowCellState(std::unique_ptr<SWGMMBase> model,
                             const FlowMapConfig& cfg,
                             std::shared_ptr<const std::vector<double>> shared_omegas,
                             std::shared_ptr<const GlobalSpeedPrior> global_speed_prior)
    : shared_omegas_(std::move(shared_omegas)),
      global_speed_prior_(std::move(global_speed_prior)),
      swgmm(std::move(model)) {
  const int K = std::max(1, swgmm ? swgmm->numComponents() : 1);
  const size_t K_sz = static_cast<size_t>(K);

  nudft_models.resize(K_sz);
  nudft_mu_rho_models.resize(K_sz);

  // shared_omegas_ is owned by the parent FlowMap; every cell in the same
  // FlowMap holds the same shared_ptr by construction, so cross-cell merges
  // see pointer-identical omegas_ptr and the merge succeeds.
  const auto configure_models = [&](std::vector<NUDFTModel>& models) {
    for (auto& m : models) {
      m.omegas_ptr = shared_omegas_;
      m.min_observations = cfg.nudft_min_obs;
      m.reset();
    }
  };

  configure_models(nudft_models);
  configure_models(nudft_mu_rho_models);

  // The mixing-weight and presence predictors select their spectral order per
  // cell through the prequential gate.
  poisson.nudft.omegas_ptr = shared_omegas_;
  poisson.nudft.min_observations = cfg.nudft_min_obs;
  poisson.rate_window_s = cfg.presence_rate_window_s;
  poisson.nudft.gate_enabled = true;
  for (auto& m : nudft_models) {
    m.gate_enabled = true;
  }
  poisson.nudft.reset();
}

void FlowCellState::update(double theta,
                           double rho,
                           double t_seconds,
                           double delta_t_visible,
                           const FlowMapConfig& cfg,
                           std::vector<double>* r_out) {
  if (rho <= 0.0 || !swgmm) {
    return;
  }

  ++total_observations;
  // Count the detection only. Exposure (visible time) is accumulated per
  // visible frame via FlowMap::addPoissonExposure, so the rate denominator is
  // true visible time rather than the inter-detection gap.
  poisson.recordEvent(t_seconds);
  (void)delta_t_visible;

  const auto r = swgmm->responsibilities(theta, rho);
  if (static_cast<int>(r.size()) != swgmm->numComponents()) {
    return;
  }
  if (r_out) {
    *r_out = r;
  }

  // mu_theta/Sigma are pinned by construction in the fixed regime; only the
  // per-slot speed is updated here. The weights predictors are NOT fed here:
  // FlowMap aggregates responsibilities over each track's visit and calls
  // feedWeightsSample() when the visit closes, so one crossing casts one vote
  // (FlowMap also routes id-less detections there one detection at a time).
  if (!r.empty()) {
    // Per-slot speed: route the observed rho to the slot most responsible
    // for the heading. At K=8 cardinal slots with sigma_theta ~ 0.4 rad the
    // responsibility is sharply peaked, so hard assignment is a close
    // approximation of the soft target.
    size_t k_best = 0;
    double r_best = r[0];
    for (size_t k = 1; k < r.size(); ++k) {
      if (r[k] > r_best) {
        r_best = r[k];
        k_best = k;
      }
    }
    nudft_mu_rho_models[k_best].update(t_seconds, rho);
  }


  // Time-to-stability tracking. Recompute the current dominant slot from the
  // mixing-weight DC terms (gamma0 of nudft_models), independent of the
  // mu_rho speed channel. The streak resets whenever the dominant slot changes and the
  // cell is declared stable as soon as the streak reaches stability_window.
  if (n_obs_to_stable < 0 && !nudft_models.empty()) {
    int dominant_slot = 0;
    double best = nudft_models[0].gamma0;
    for (size_t k = 1; k < nudft_models.size(); ++k) {
      if (nudft_models[k].gamma0 > best) {
        best = nudft_models[k].gamma0;
        dominant_slot = static_cast<int>(k);
      }
    }
    if (dominant_slot == last_dominant_slot) {
      ++stability_streak;
    } else {
      stability_streak = 1;
      last_dominant_slot = dominant_slot;
    }
    if (stability_streak >= cfg.stability_window) {
      // The streak began stability_window observations ago, so the cell
      // first reached stability at this earlier point.
      n_obs_to_stable = total_observations - cfg.stability_window + 1;
    }
  }

  t_previous_observation = t_seconds;
}

void FlowCellState::feedWeightsSample(double t_seconds,
                                      const std::vector<double>& r,
                                      const FlowMapConfig& cfg) {
  if (r.size() != nudft_models.size()) {
    return;
  }
  ++n_crossings;
  for (size_t k = 0; k < r.size(); ++k) {
    nudft_models[k].update(t_seconds, r[k]);
    nudft_models[k].updateVariance(t_seconds, cfg.weight_uncertainty_noise_sigma);
  }
  // Flip nudft_valid once the pi NUDFTs have enough samples (crossings).
  if (!nudft_valid) {
    bool all_ready = !nudft_models.empty();
    for (size_t k = 0; k < nudft_models.size(); ++k) {
      if (nudft_models[k].n < cfg.nudft_min_obs) {
        all_ready = false;
        break;
      }
    }
    if (all_ready) {
      nudft_valid = true;
      nudft_fit_timestamp = t_seconds;
    }
  }
}

FlowPrediction FlowCellState::predict(double t_seconds,
                                      double delta_t,
                                      const FlowMapConfig& cfg,
                                      const NeighbourhoodEvidence* neighbourhood) const {
  FlowPrediction prediction;
  prediction.pi = queryPi(t_seconds, cfg, neighbourhood, &prediction.var_pi,
                          &prediction.pi_clamped, &prediction.pi_order);
  const double speed = cellSpeedMean();  // -1 if the cell has no observations yet
  // A positive speed gives a finite dwell (cell / speed); a near-stationary
  // target then yields a long dwell and a vanishing arrival term, collapsing to
  // the instantaneous occupancy on its own. No speed (<= 0) -> instantaneous.
  const double dwell = (speed > 0.0) ? cfg.voxel_size_m / speed : -1.0;
  prediction.dwell_s = dwell;
  prediction.l_hat = poisson.predictOccupancy(t_seconds, cfg.nudft_order, cfg.nudft_min_obs);
  prediction.p_present = presenceFromOccupancy(prediction.l_hat, dwell, delta_t);

  if (total_observations == 0) {
    prediction.flag = CellFlag::FALLBACK;
    return prediction;
  }

  if (!nudft_valid) {
    prediction.flag = CellFlag::COLD_START;
    return prediction;
  }

  prediction.flag = CellFlag::MODEL_BASED;
  return prediction;
}

FlowCellSnapshot FlowCellState::snapshot(double t_seconds,
                                         double delta_t,
                                         const FlowMapConfig& cfg,
                                         const NeighbourhoodEvidence* neighbourhood) const {
  FlowCellSnapshot out;
  if (!swgmm) {
    return out;
  }

  const auto prediction = predict(t_seconds, delta_t, cfg, neighbourhood);
  out.p_present = prediction.p_present;
  out.l_hat = prediction.l_hat;
  out.dwell_s = prediction.dwell_s;
  out.flag = prediction.flag;
  out.n_crossings = n_crossings;

  if (prediction.flag == CellFlag::FALLBACK) {
    return out;
  }

  if (prediction.pi.empty() || swgmm->numComponents() <= 0) {
    return out;
  }

  out.components.reserve(static_cast<size_t>(swgmm->numComponents()));
  double max_weight = 0.0;

  // Cold-start speed prior (identical across slots): the map-wide running mean
  // of observed speed, which a slot reports until it has enough observations
  // of its own.
  const double cell_prior = global_speed_prior_
                                ? global_speed_prior_->mean(cfg.fixed_mu_rho)
                                : cfg.fixed_mu_rho;

  for (int k = 0; k < swgmm->numComponents(); ++k) {
    const auto& component = swgmm->component(k);
    const double weight = k < static_cast<int>(prediction.pi.size())
                              ? prediction.pi[static_cast<size_t>(k)]
                              : 0.0;
    double theta = component.mu_theta;
    double rho = cell_prior;
    double sigma_theta = component.Sigma(0, 0);
    double sigma_rho = component.Sigma(1, 1);
    double sigma_cross = component.Sigma(0, 1);

    // Per-slot speed: the slot's running mean of observed speed (not forecast),
    // once the cell's weights are valid and the slot has nudft_min_obs
    // observations of its own; the cold-start prior until then. The pinned
    // cardinal heading and configured covariance are kept.
    if (static_cast<size_t>(k) < nudft_mu_rho_models.size()) {
      const auto& m = nudft_mu_rho_models[static_cast<size_t>(k)];
      if (nudft_valid && m.n >= cfg.nudft_min_obs) {
        rho = m.gamma0;
      }
    }

    FlowComponentSnapshot snapshot_component;
    snapshot_component.theta = theta;
    snapshot_component.rho = rho;
    snapshot_component.sigma_theta = std::max(1.0e-6, sigma_theta);
    snapshot_component.sigma_rho = std::max(1.0e-6, sigma_rho);
    // Independent NUDFT predictors on Sigma_00, Sigma_11, Sigma_01 do not
    // jointly enforce PSD; clamp |Sigma_01| <= sqrt(Sigma_00 * Sigma_11) at
    // query time using the post-clamp diagonals so the bound matches the
    // values written downstream. Sign of the cross is
    // preserved. The (1 - kPSDRidge) factor keeps det(Sigma) strictly positive
    // so downstream Gaussian density evaluation (held-out log-likelihood) does
    // not produce -inf at the PSD frontier; bias on the cross is < 0.1%.
    constexpr double kPSDRidge = 1.0e-3;
    const double cross_bound = (1.0 - kPSDRidge) *
                               std::sqrt(snapshot_component.sigma_theta *
                                         snapshot_component.sigma_rho);
    const double clamped_cross =
        std::clamp(sigma_cross, -cross_bound, cross_bound);
    snapshot_component.sigma_cross = clamped_cross;
    snapshot_component.weight = std::clamp(weight, 0.0, 1.0);
    snapshot_component.std_theta = 0.0;
    // Speed uncertainty. The speed is a running mean, so the interval is its
    // aleatoric term alone: the slot's measured speed scatter, the cumulative
    // online variance (m2 - gamma0^2) in constant memory, blended with the
    // prior spread fixed_sigma_rho carrying the weight of
    // speed_scatter_prior_obs observations. A new slot reports the prior spread
    // and a well-observed one its own scatter; old observations are never
    // down-weighted.
    double aleatoric_var = sigma_rho * sigma_rho;
    if (static_cast<size_t>(k) < nudft_mu_rho_models.size()) {
      const auto& speed_model = nudft_mu_rho_models[static_cast<size_t>(k)];
      const double emp_var =
          std::max(0.0, speed_model.m2 - speed_model.gamma0 * speed_model.gamma0);
      const double kappa = cfg.speed_scatter_prior_obs;
      const double prior_var = cfg.fixed_sigma_rho * cfg.fixed_sigma_rho;
      const double nobs = static_cast<double>(speed_model.n);
      if (kappa + nobs > 0.0) {
        aleatoric_var = (kappa * prior_var + nobs * emp_var) / (kappa + nobs);
      }
    }
    snapshot_component.std_rho = std::sqrt(aleatoric_var);
    // Weight-side parameter std, propagated by queryPi at the effective order
    // the reported weight used.
    if (static_cast<size_t>(k) < prediction.var_pi.size()) {
      const double var_pi = prediction.var_pi[static_cast<size_t>(k)];
      snapshot_component.std_pi = std::sqrt(var_pi);
      snapshot_component.pi_clamped =
          static_cast<size_t>(k) < prediction.pi_clamped.size() &&
          prediction.pi_clamped[static_cast<size_t>(k)] != 0;
      snapshot_component.pi_order =
          static_cast<size_t>(k) < prediction.pi_order.size()
              ? prediction.pi_order[static_cast<size_t>(k)]
              : 0;
    }
    out.components.push_back(snapshot_component);
    max_weight = std::max(max_weight, snapshot_component.weight);
  }

  out.confidence = max_weight;
  return out;
}

double FlowCellState::cellSpeedMean() const {
  // gamma0 is the cumulative mean of the speeds routed to a slot, so gamma0 * n
  // is that slot's observed-speed sum. Pooling over slots recovers the cell's
  // overall mean observed speed (each observation lands in exactly one slot).
  double sum = 0.0;
  long long count = 0;
  for (const auto& m : nudft_mu_rho_models) {
    if (m.n > 0) {
      sum += m.gamma0 * static_cast<double>(m.n);
      count += m.n;
    }
  }
  return count > 0 ? sum / static_cast<double>(count) : -1.0;
}

void FlowCellState::mergeFrom(FlowCellState&& other, const FlowMapConfig& cfg) {
  if (other.total_observations <= 0) {
    VLOG(2) << "[FlowMerge] mergeFrom skipped: other has no observations";
    return;
  }
  const int n_a_pre = total_observations;
  const int n_b_pre = other.total_observations;

  // Preserve the original-position metadata. This metadata is what feeds the
  // next stamped deformation; if we lose it, the merged cell will be
  // re-deformed from a stale corrected position and may end up cross-room
  // again.
  auto adopt_origin_from = [&](FlowCellState& src) {
    original_index = src.original_index;
    creation_timestamp_ns = src.creation_timestamp_ns;
    original_set = src.original_set;
    // The carried rotation is measured against the origin, so it travels with
    // it; the two sides of a collision are neighbours corrected by the same
    // control points and agree to well under a slot in any case.
    heading_rotation_applied = src.heading_rotation_applied;
  };
  if (other.original_set && (!original_set || other.total_observations > total_observations)) {
    adopt_origin_from(other);
  }

  if (!swgmm) {
    swgmm = std::move(other.swgmm);
    nudft_models = std::move(other.nudft_models);
    nudft_mu_rho_models = std::move(other.nudft_mu_rho_models);
    poisson = std::move(other.poisson);
    total_observations = other.total_observations;
    n_crossings = other.n_crossings;
    nudft_valid = other.nudft_valid;
    nudft_fit_timestamp = other.nudft_fit_timestamp;
    t_previous_observation = other.t_previous_observation;
    return;
  }

  poisson.mergeFrom(other.poisson);

  total_observations += other.total_observations;
  n_crossings += other.n_crossings;
  t_previous_observation = std::max(t_previous_observation, other.t_previous_observation);
  nudft_fit_timestamp = std::max(nudft_fit_timestamp, other.nudft_fit_timestamp);

  // Fixed-component geometry is identical across cells: merge each NUDFT
  // channel pairwise.
  const size_t K = nudft_models.size();
  int channels_attempted = 0;
  int channels_refused = 0;
  const auto merge_vec = [&](std::vector<NUDFTModel>& a,
                             const std::vector<NUDFTModel>& b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t k = 0; k < n; ++k) {
      ++channels_attempted;
      if (!a[k].mergeFrom(b[k])) {
        ++channels_refused;
      }
    }
  };
  merge_vec(nudft_models, other.nudft_models);
  merge_vec(nudft_mu_rho_models, other.nudft_mu_rho_models);

  // Recheck nudft_valid: pooling may have crossed min_observations.
  if (!nudft_valid) {
    bool all_ready = !nudft_models.empty();
    for (size_t k = 0; k < K; ++k) {
      if (nudft_models[k].n < cfg.nudft_min_obs) {
        all_ready = false;
        break;
      }
    }
    if (all_ready) {
      nudft_valid = true;
      nudft_fit_timestamp = t_previous_observation;
    }
  }
  nudft_valid = nudft_valid || other.nudft_valid;
  if (channels_refused > 0) {
    LOG(WARNING) << "[FlowMerge.cell_lost_nudft] Fixed cell merge dropped "
                 << channels_refused << "/" << channels_attempted
                 << " NUDFT channels from other (n_a=" << n_a_pre
                 << " n_b=" << n_b_pre
                 << "). Spectral state from `other` was discarded.";
  }
  LOG(INFO) << "[FlowMerge.cell] n_a=" << n_a_pre << " n_b=" << n_b_pre
            << " n_total=" << total_observations
            << " nudft_valid=" << (nudft_valid ? "1" : "0")
            << " nudft_refused=" << channels_refused << "/"
            << channels_attempted;
}

// The mean term one slot reports. With no neighbourhood to draw on, this is the
// cell's own running mean. Otherwise it is that mean shrunk toward the
// neighbourhood estimate, which lends the cell m crossings of
// evidence, so the neighbourhood holds a share m/(C+m) of the result and that
// share falls to zero as the cell accumulates crossings of its own.
double FlowCellState::sharedMean(int k,
                                 const FlowMapConfig& cfg,
                                 const NeighbourhoodEvidence* neighbourhood) const {
  const double own = std::max(0.0, nudft_models[static_cast<size_t>(k)].gamma0);
  const double m = cfg.sharing_lent_crossings;
  if (m <= 0.0 || !neighbourhood ||
      !neighbourhood->usable() ||
      static_cast<int>(neighbourhood->weighted_sum.size()) <= k) {
    return own;
  }
  const double neighbourhood_mean =
      std::max(0.0, neighbourhood->weighted_sum[static_cast<size_t>(k)]) /
      neighbourhood->crossings;
  const double own_crossings = static_cast<double>(n_crossings);
  return (own_crossings * own + m * neighbourhood_mean) / (own_crossings + m);
}

// Return the mixing weights at the query time. Before the predictors are ready
// it reports the mean terms alone; once ready it adds the harmonics on top of
// them. Sharing substitutes the mean term only, so the periodic structure the
// coefficients carry is unaffected. The result is always clamped and normalized.
std::vector<double> FlowCellState::queryPi(double t_seconds,
                                           const FlowMapConfig& cfg,
                                           const NeighbourhoodEvidence* neighbourhood,
                                           std::vector<double>* var_pi_out,
                                           std::vector<uint8_t>* clamped_out,
                                           std::vector<int>* order_out) const {
  const int K = swgmm ? swgmm->numComponents() : 1;
  std::vector<double> pi(static_cast<size_t>(K), 1.0 / static_cast<double>(K));
  if (var_pi_out) {
    var_pi_out->assign(static_cast<size_t>(K), 0.0);
  }
  if (clamped_out) {
    clamped_out->assign(static_cast<size_t>(K), 0);
  }
  if (order_out) {
    order_out->assign(static_cast<size_t>(K), 0);
  }
  if (static_cast<int>(nudft_models.size()) != K) {
    return pi;
  }
  const bool sharing = cfg.sharing_lent_crossings > 0.0 && neighbourhood &&
                       neighbourhood->usable();

  // Raw per-slot values and, alongside each, the effective spectral order the
  // mean readout used (0 wherever it fell back to the DC term), so a variance
  // computed below describes the same reconstruction as the mean.
  std::vector<double> raw = pi;
  std::vector<int> eff_order(static_cast<size_t>(K), 0);
  bool normalized = true;
  if (!nudft_valid) {
    // The mean terms alone give a meaningful non-uniform mix before the
    // predictors are valid. A cell with no crossings of its own still reports
    // the neighbourhood estimate.
    bool has_any = false;
    for (int k = 0; k < K; ++k) {
      if (nudft_models[static_cast<size_t>(k)].n > 0 || sharing) {
        raw[static_cast<size_t>(k)] = std::max(0.0, sharedMean(k, cfg, neighbourhood));
        has_any = true;
      }
    }
    // With no evidence anywhere the uniform mix passes through un-normalized.
    normalized = has_any;
  } else {
    for (int k = 0; k < K; ++k) {
      // Each slot uses its prequentially-validated order.
      const auto& model = nudft_models[static_cast<size_t>(k)];
      raw[static_cast<size_t>(k)] = model.predictGatedWithMean(
          t_seconds, cfg.nudft_order, sharedMean(k, cfg, neighbourhood));
      // predictWithMean emits the mean term alone below min_observations, so
      // the effective order is 0 there regardless of what the gate selected.
      eff_order[static_cast<size_t>(k)] =
          model.n >= model.min_observations ? model.bestOrder(cfg.nudft_order) : 0;
    }
  }
  pi = normalized ? normalizePi(raw) : raw;

  if (order_out) {
    for (int k = 0; k < K; ++k) {
      (*order_out)[static_cast<size_t>(k)] = eff_order[static_cast<size_t>(k)];
    }
  }

  if (clamped_out) {
    // Where the normalizer's clamp binds, the reported weight no longer moves
    // with the raw prediction and the propagated variance below is wrong;
    // flag those slots so a consumer can separate them. Covers both a
    // negative harmonic reconstruction and a never-observed slot's zero mean.
    for (int k = 0; k < K; ++k) {
      const double r = raw[static_cast<size_t>(k)];
      (*clamped_out)[static_cast<size_t>(k)] = (r < 1.0e-3 || r > 1.0) ? 1 : 0;
    }
  }

  if (var_pi_out) {
    // Per-slot parameter variance of the raw prediction. A slot that has never
    // received a vote carries the coefficient prior (variance 1.0, DC term
    // only).
    std::vector<double> v(static_cast<size_t>(K), 0.0);
    double sum_v = 0.0;
    for (int k = 0; k < K; ++k) {
      const auto& model = nudft_models[static_cast<size_t>(k)];
      v[static_cast<size_t>(k)] =
          model.hasVarianceState()
              ? model.predictionVariance(t_seconds, eff_order[static_cast<size_t>(k)])
              : 1.0;
      sum_v += v[static_cast<size_t>(k)];
    }
    if (normalized) {
      // First-order propagation of the per-slot variances (treated as
      // independent) through pi_k = x_k / S at the clamped values the
      // normalizer actually divides:
      //   var(pi_k) = [ (1 - 2 pi_k) v_k + pi_k^2 sum_j v_j ] / S^2.
      // The clamp itself is not differentiated; see the flag above.
      double S = 0.0;
      for (int k = 0; k < K; ++k) {
        S += std::clamp(raw[static_cast<size_t>(k)], 1.0e-3, 1.0);
      }
      for (int k = 0; k < K; ++k) {
        const double p = pi[static_cast<size_t>(k)];
        const double var =
            ((1.0 - 2.0 * p) * v[static_cast<size_t>(k)] + p * p * sum_v) / (S * S);
        (*var_pi_out)[static_cast<size_t>(k)] = std::max(0.0, var);
      }
    } else {
      // Uniform passthrough (no evidence anywhere): report the unpropagated
      // prior variances.
      *var_pi_out = v;
    }
  }
  return pi;
}

void FlowCellState::rotateHeading(double d_psi) {
  const size_t K = nudft_models.size();
  if (K == 0 || !std::isfinite(d_psi)) {
    return;
  }

  const double slot_width = 2.0 * M_PI / static_cast<double>(K);
  const double shift_slots = d_psi / slot_width;
  // A shift of a whole number of slots (within a rounding epsilon) is a pure
  // permutation; only a fractional shift needs the interpolation below.
  const double shift_wrapped =
      shift_slots - std::floor(shift_slots / static_cast<double>(K)) * static_cast<double>(K);
  if (std::abs(shift_wrapped) < 1.0e-9 ||
      std::abs(shift_wrapped - static_cast<double>(K)) < 1.0e-9) {
    return;
  }

  // Slot k of the rotated cell holds what used to sit at heading theta_k -
  // d_psi, i.e. at the fractional slot position k - shift_slots. Reading the
  // source at a fractional position is the linear resample onto the fixed
  // cardinal layout.
  auto wrap = [K](long long index) {
    const long long k = static_cast<long long>(K);
    return static_cast<size_t>(((index % k) + k) % k);
  };

  const double source_pos_offset = -shift_slots;
  const long long floor_offset =
      static_cast<long long>(std::floor(source_pos_offset));
  const double frac = source_pos_offset - static_cast<double>(floor_offset);

  // Mixing weights: interpolate the coefficients of the two straddling slots.
  const auto weights_before = nudft_models;
  for (size_t k = 0; k < K; ++k) {
    const size_t lo = wrap(static_cast<long long>(k) + floor_offset);
    const size_t hi = wrap(static_cast<long long>(k) + floor_offset + 1);
    const auto& a = weights_before[lo];
    const auto& b = weights_before[hi];
    auto& out = nudft_models[k];

    out.gamma0 = (1.0 - frac) * a.gamma0 + frac * b.gamma0;
    // Quadratic in the fed value, so the blend is an approximation; convexity
    // keeps it at or above the true second moment.
    out.m2 = (1.0 - frac) * a.m2 + frac * b.m2;
    const size_t f_count = std::min(a.gammas.size(), b.gammas.size());
    out.gammas.assign(f_count, std::complex<double>(0.0, 0.0));
    for (size_t f = 0; f < f_count; ++f) {
      out.gammas[f] = (1.0 - frac) * a.gammas[f] + frac * b.gammas[f];
    }
    // The posterior variances are driven by the sample times alone, identical
    // across slots, so interpolating them is the identity in exact arithmetic.
    if (a.var_gammas.size() == b.var_gammas.size()) {
      out.var_gammas.assign(a.var_gammas.size(), 0.0);
      for (size_t f = 0; f < a.var_gammas.size(); ++f) {
        out.var_gammas[f] = (1.0 - frac) * a.var_gammas[f] + frac * b.var_gammas[f];
      }
    }
    out.var_gamma0 = (1.0 - frac) * a.var_gamma0 + frac * b.var_gamma0;
    // Accumulated squared one-step-ahead error per candidate order: quadratic,
    // blended on the same conservative footing as m2. The gate then selects an
    // order for the rotated signal from the rotated error budget.
    if (a.gate_err.size() == b.gate_err.size()) {
      out.gate_err.assign(a.gate_err.size(), 0.0);
      for (size_t o = 0; o < a.gate_err.size(); ++o) {
        out.gate_err[o] = (1.0 - frac) * a.gate_err[o] + frac * b.gate_err[o];
      }
    }
    // n, gate_n, min_observations, the shared frequencies and the configured
    // flags are common to every slot and survive the resample unchanged.
  }

  // Per-slot speeds: nearest whole slot, see the header for why these are
  // permuted rather than blended.
  if (!nudft_mu_rho_models.empty() && nudft_mu_rho_models.size() == K) {
    const long long nearest = static_cast<long long>(std::llround(shift_slots));
    if (wrap(nearest) != 0) {
      const auto speeds_before = nudft_mu_rho_models;
      for (size_t k = 0; k < K; ++k) {
        nudft_mu_rho_models[k] =
            speeds_before[wrap(static_cast<long long>(k) - nearest)];
      }
    }
  }

  if (last_dominant_slot >= 0) {
    last_dominant_slot = static_cast<int>(
        wrap(static_cast<long long>(last_dominant_slot) +
             static_cast<long long>(std::llround(shift_slots))));
  }
}

}  // namespace kairos
