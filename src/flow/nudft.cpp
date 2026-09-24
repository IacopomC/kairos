#include "kairos/flow/nudft.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace kairos {

namespace {

bool setNudftError(std::string* error, const std::string& message) {
  if (error) {
    *error = message;
  }
  return false;
}

}  // namespace

void NUDFTModel::update(double t_seconds, double value) {
  if (!omegas_ptr) return;
  const auto& omegas = *omegas_ptr;

  if (omegas.size() != gammas.size()) {
    gammas.assign(omegas.size(), std::complex<double>(0.0, 0.0));
  }

  // Prequential gate: score every candidate order against this value BEFORE
  // folding it in, so each sample is genuinely held out with respect to the
  // state that predicts it.
  if (gate_enabled) {
    const size_t F = omegas.size();
    if (gate_err.size() != F + 1) {
      gate_err.assign(F + 1, 0.0);
    }
    for (size_t o = 0; o <= F; ++o) {
      const double e = predict(t_seconds, static_cast<int>(o)) - value;
      gate_err[o] += e * e;
    }
    ++gate_n;
  }

  const double n_double = static_cast<double>(n);
  const double gamma0_old = gamma0;
  // DC term: incremental running mean of the observed value.
  gamma0 = (n_double * gamma0 + value) / (n_double + 1.0);

  // Second moment: incremental running mean of value^2. Combined with gamma0 this
  // yields the cumulative observed-value variance (m2 - gamma0^2) in O(1) and constant
  // memory -- the online aleatoric scatter. Equal-weight over all history (same scheme
  // as gamma0), so it is not recency-weighted and does not forget old observations.
  m2 = (n_double * m2 + value * value) / (n_double + 1.0);

  // Each AC coefficient: running mean of the residual (value - mean) projected
  // onto its frequency.
  const std::complex<double> j(0.0, 1.0);
  for (size_t i = 0; i < omegas.size(); ++i) {
    const auto term = std::exp(-j * omegas[i] * t_seconds);
    gammas[i] = (n_double * gammas[i] + (value - gamma0_old) * term) / (n_double + 1.0);
  }

  ++n;
}

double NUDFTModel::predict(double t_seconds, int order) const {
  return predictWithMean(t_seconds, order, gamma0);
}

double NUDFTModel::predictWithMean(double t_seconds, int order, double mean) const {
  if (!omegas_ptr) return mean;
  const auto& omegas = *omegas_ptr;

  if (n < min_observations || order <= 0 || omegas.empty() || gammas.empty()) {
    return mean;
  }

  // Rank frequencies by coefficient magnitude so the strongest `order` of them
  // are used.
  std::vector<size_t> indices(omegas.size());
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

  std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
    return std::abs(gammas[a]) > std::abs(gammas[b]);
  });

  // Start from the mean and add each selected component's cosine contribution.
  double value = mean;
  const int use_count = std::min(order, static_cast<int>(indices.size()));
  for (int i = 0; i < use_count; ++i) {
    const size_t idx = indices[static_cast<size_t>(i)];
    // gammas[idx] is the one-sided coefficient (A/2)e^{j*phi} of a real signal;
    // the conjugate negative-frequency term contributes equally, so the real
    // reconstruction is 2*Re(gamma e^{j*omega*t}) = 2|gamma| cos(omega t + arg).
    value += 2.0 * std::abs(gammas[idx]) *
             std::cos(omegas[idx] * t_seconds + std::arg(gammas[idx]));
  }

  return value;
}

int NUDFTModel::bestOrder(int max_order) const {
  if (!gate_enabled) {
    return max_order;
  }
  if (gate_n < min_observations || gate_err.empty() || max_order <= 0) {
    return 0;
  }
  const int hi = std::min<int>(max_order, static_cast<int>(gate_err.size()) - 1);
  int best = 0;
  for (int o = 1; o <= hi; ++o) {
    if (gate_err[static_cast<size_t>(o)] < gate_err[static_cast<size_t>(best)]) {
      best = o;  // strict <: ties resolve to the lower (simpler) order
    }
  }
  return best;
}

double NUDFTModel::predictGated(double t_seconds, int max_order) const {
  return predict(t_seconds, bestOrder(max_order));
}

double NUDFTModel::predictGatedWithMean(double t_seconds, int max_order, double mean) const {
  return predictWithMean(t_seconds, bestOrder(max_order), mean);
}

void NUDFTModel::updateVariance(double t_seconds, double sigma_noise) {
  if (!omegas_ptr) {
    return;
  }
  const auto& omegas = *omegas_ptr;
  const size_t F = omegas.size();

  // Initialise variances on first call (uninformative prior = 1.0 per coeff).
  if (var_gammas.size() != F) {
    var_gammas.assign(F, 1.0);
  }
  if (var_gamma0 == 0.0) {
    var_gamma0 = 1.0;
  }

  // Diagonal Sherman-Morrison step for a Bayesian linear regression with
  // design vector A = [1, cos(ω_1 t + φ_1), ..., cos(ω_F t + φ_F)]. The
  // regressed parameters are the amplitudes a_f = 2|gammas[f]| of that basis,
  // so var_gammas[f] is var(a_f) and predictionVariance() propagates it
  // without any further factor.
  // denom = σ²_noise + v_0 * A_0² + Σ_f v_f * A_f²
  // v_i ← v_i - v_i² * A_i² / denom
  const double sig2 = sigma_noise * sigma_noise;
  double denom = sig2 + var_gamma0;
  std::vector<double> A(F);
  for (size_t i = 0; i < F; ++i) {
    const double phase = gammas.size() > i ? std::arg(gammas[i]) : 0.0;
    A[i] = std::cos(omegas[i] * t_seconds + phase);
    denom += var_gammas[i] * A[i] * A[i];
  }
  if (denom <= 0.0) {
    return;
  }
  var_gamma0 -= (var_gamma0 * var_gamma0) / denom;
  var_gamma0 = std::max(0.0, var_gamma0);
  for (size_t i = 0; i < F; ++i) {
    var_gammas[i] -= (var_gammas[i] * var_gammas[i] * A[i] * A[i]) / denom;
    var_gammas[i] = std::max(0.0, var_gammas[i]);
  }
}

double NUDFTModel::predictionVariance(double t_seconds, int order) const {
  if (var_gammas.empty() || !omegas_ptr) {
    return 0.0;
  }
  const auto& omegas = *omegas_ptr;
  const size_t F = omegas.size();

  // Propagate posterior variance through the prediction formula. var_gammas[f]
  // is the posterior variance of the *amplitude* a_f = 2|gammas[f]| (the
  // parameter of the cos basis that updateVariance() regresses on), so the
  // variance of predict()'s reconstruction gamma0 + Σ_f a_f cos(ω_f t + φ_f) is
  // var_pred = var_gamma0 + Σ_f var_f * cos²(ω_f t + φ_f)
  // with no further scaling: the factor 2 is already inside a_f.
  double var = var_gamma0;
  const int use_count = std::min(order, static_cast<int>(F));
  if (use_count > 0 && !gammas.empty()) {
    // Sort by magnitude as in predict() so we use the same top-order freqs.
    std::vector<size_t> indices(F);
    for (size_t i = 0; i < F; ++i) {
      indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
      return std::abs(gammas[a]) > std::abs(gammas[b]);
    });
    for (int i = 0; i < use_count; ++i) {
      const size_t idx = indices[static_cast<size_t>(i)];
      if (idx >= var_gammas.size()) {
        continue;
      }
      const double phase = std::arg(gammas[idx]);
      const double c = std::cos(omegas[idx] * t_seconds + phase);
      var += var_gammas[idx] * c * c;
    }
  }
  return std::max(0.0, var);
}

void NUDFTModel::reset() {
  const size_t sz = omegas_ptr ? omegas_ptr->size() : 0;
  gammas.assign(sz, std::complex<double>(0.0, 0.0));
  gamma0 = 0.0;
  n = 0;
  var_gamma0 = 0.0;
  var_gammas.clear();
  gate_err.clear();
  gate_n = 0;
}

bool NUDFTModel::mergeFrom(const NUDFTModel& other) {
  if (other.n == 0) {
    return true;
  }
  if (n == 0) {
    gamma0 = other.gamma0;
    m2 = other.m2;
    gammas = other.gammas;
    n = other.n;
    min_observations = std::min(min_observations, other.min_observations);
    if (!omegas_ptr) {
      omegas_ptr = other.omegas_ptr;
    }
    // Inherit posterior variances when other tracks them; matches the
    // gamma inheritance above.
    var_gamma0 = other.var_gamma0;
    var_gammas = other.var_gammas;
    gate_err = other.gate_err;
    gate_n = other.gate_n;
    return true;
  }

  // The merge is valid as long as the two NUDFTs are defined over the same
  // frequency set. FlowMap routes one shared_omegas_ through every cell so
  // pointer identity normally holds, but we gate on *content* equality so the
  // merge stays correct under deserialization or any future path that
  // accidentally constructs separate-but-identical omegas vectors.
  const size_t this_n = omegas_ptr ? omegas_ptr->size() : 0;
  const size_t other_n = other.omegas_ptr ? other.omegas_ptr->size() : 0;
  if (!omegas_ptr || !other.omegas_ptr || this_n != other_n) {
    LOG_FIRST_N(WARNING, 5)
        << "[FlowMerge] NUDFT mergeFrom refused: omegas vectors missing or "
           "size mismatch (this="
        << this_n << ", other=" << other_n << ", this_n_obs=" << n
        << ", other_n_obs=" << other.n << "). Keeping this model unchanged.";
    return false;
  }
  const bool pointer_identical = (omegas_ptr.get() == other.omegas_ptr.get());
  if (!pointer_identical) {
    for (size_t i = 0; i < this_n; ++i) {
      if (std::abs((*omegas_ptr)[i] - (*other.omegas_ptr)[i]) > 1.0e-12) {
        LOG_FIRST_N(WARNING, 5)
            << "[FlowMerge] NUDFT mergeFrom refused: omegas content drift at "
               "index "
            << i << " (this=" << (*omegas_ptr)[i]
            << ", other=" << (*other.omegas_ptr)[i]
            << "). Keeping this model unchanged.";
        return false;
      }
    }
    LOG_FIRST_N(INFO, 5) << "[FlowMerge] NUDFT mergeFrom: omegas pointers "
                            "differ but contents match — proceeding with "
                            "count-weighted pool.";
  }

  const double n_a = static_cast<double>(n);
  const double n_b = static_cast<double>(other.n);
  const double inv = 1.0 / (n_a + n_b);

  gamma0 = (n_a * gamma0 + n_b * other.gamma0) * inv;
  m2 = (n_a * m2 + n_b * other.m2) * inv;  // count-weighted pool, mirrors gamma0

  if (gammas.size() != other.gammas.size()) {
    // Should not happen when omegas content matches, but guard anyway.
    gammas.resize(std::max(gammas.size(), other.gammas.size()),
                  std::complex<double>(0.0, 0.0));
  }
  const size_t F = std::min(gammas.size(), other.gammas.size());
  for (size_t i = 0; i < F; ++i) {
    gammas[i] = (n_a * gammas[i] + n_b * other.gammas[i]) * inv;
  }

  // Pool posterior variances. Count-weighted linear pool mirrors the gamma
  // pooling above: not Bayesian-exact
  // (the strict identity is 1/v_ab ≈ 1/v_a + 1/v_b − 1/v_prior) but
  // numerically robust, monotone in n, and matches the same approximation
  // used for the means. When only one side has variance state, it is kept.
  if (!var_gammas.empty() && !other.var_gammas.empty()) {
    var_gamma0 = (n_a * var_gamma0 + n_b * other.var_gamma0) * inv;
    const size_t Fv = std::min(var_gammas.size(), other.var_gammas.size());
    for (size_t i = 0; i < Fv; ++i) {
      var_gammas[i] = (n_a * var_gammas[i] + n_b * other.var_gammas[i]) * inv;
    }
  } else if (var_gammas.empty() && !other.var_gammas.empty()) {
    var_gamma0 = other.var_gamma0;
    var_gammas = other.var_gammas;
  }

  // Gate errors are additive sums of per-sample squared errors.
  if (!other.gate_err.empty()) {
    if (gate_err.size() < other.gate_err.size()) {
      gate_err.resize(other.gate_err.size(), 0.0);
    }
    for (size_t i = 0; i < other.gate_err.size(); ++i) {
      gate_err[i] += other.gate_err[i];
    }
  }
  gate_n += other.gate_n;

  n += other.n;
  min_observations = std::min(min_observations, other.min_observations);
  VLOG(2) << "[FlowMerge] NUDFT pooled: n_a=" << static_cast<int>(n_a)
          << " n_b=" << static_cast<int>(n_b) << " n_ab=" << n
          << " (pointer_identical=" << (pointer_identical ? "yes" : "no")
          << ")";
  return true;
}

nlohmann::json nudftToJson(const NUDFTModel& model) {
  nlohmann::json gammas = nlohmann::json::array();
  for (const auto& value : model.gammas) {
    gammas.push_back({{"real", value.real()}, {"imag", value.imag()}});
  }
  nlohmann::json var_gammas = nlohmann::json::array();
  for (const auto& v : model.var_gammas) {
    var_gammas.push_back(v);
  }
  nlohmann::json gate_err = nlohmann::json::array();
  for (const auto& v : model.gate_err) {
    gate_err.push_back(v);
  }
  return {{"gamma0", model.gamma0},
          {"m2", model.m2},
          {"n", model.n},
          {"min_observations", model.min_observations},
          {"gammas", gammas},
          {"var_gamma0", model.var_gamma0},
          {"var_gammas", var_gammas},
          {"gate_err", gate_err},
          {"gate_n", model.gate_n}};
}

bool nudftFromJson(const nlohmann::json& record,
                   NUDFTModel& model,
                   std::string* error) {
  if (!record.is_object()) {
    return setNudftError(error, "invalid NUDFT record");
  }

  model.gamma0 = record.value("gamma0", 0.0);
  model.m2 = record.value("m2", 0.0);
  model.n = record.value("n", 0);
  model.min_observations = record.value("min_observations", 1);
  model.gammas.clear();

  if (record.contains("gammas")) {
    const auto& gammas = record.at("gammas");
    if (!gammas.is_array()) {
      return setNudftError(error, "invalid NUDFT gamma array");
    }
    model.gammas.reserve(gammas.size());
    for (const auto& gamma : gammas) {
      if (!gamma.is_object() || !gamma.contains("real") || !gamma.contains("imag") ||
          !gamma.at("real").is_number() || !gamma.at("imag").is_number()) {
        return setNudftError(error, "invalid NUDFT gamma entry");
      }
      model.gammas.emplace_back(gamma.at("real").get<double>(),
                                gamma.at("imag").get<double>());
    }
  }

  // Gate accumulators (gate_enabled itself is configured, not serialized).
  // Absent on predictors that never gate, such as the coupling predictors.
  model.gate_n = record.value("gate_n", 0);
  model.gate_err.clear();
  if (record.contains("gate_err") && record.at("gate_err").is_array()) {
    for (const auto& v : record.at("gate_err")) {
      if (!v.is_number()) {
        return setNudftError(error, "invalid NUDFT gate_err entry");
      }
      model.gate_err.push_back(v.get<double>());
    }
  }

  // Variance state; absent on predictors that never track it.
  model.var_gamma0 = record.value("var_gamma0", 0.0);
  model.var_gammas.clear();
  if (record.contains("var_gammas") && record.at("var_gammas").is_array()) {
    const auto& vg = record.at("var_gammas");
    model.var_gammas.reserve(vg.size());
    for (const auto& v : vg) {
      if (!v.is_number()) {
        return setNudftError(error, "invalid NUDFT var_gammas entry");
      }
      model.var_gammas.push_back(v.get<double>());
    }
  }

  return true;
}

}  // namespace kairos