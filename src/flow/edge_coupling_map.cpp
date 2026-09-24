#include "kairos/flow/edge_coupling_map.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <tuple>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

void initNudft(NUDFTModel& m,
               const std::shared_ptr<const std::vector<double>>& omegas,
               int min_obs) {
  m.omegas_ptr = omegas;
  m.min_observations = min_obs;
  m.reset();
}

}  // namespace

EdgeCouplingMap::EdgeCouplingMap(std::shared_ptr<const std::vector<double>> shared_omegas,
                                 int nudft_min_obs,
                                 double window_s,
                                 int window_min_pairs)
    : shared_omegas_(std::move(shared_omegas)),
      nudft_min_obs_(nudft_min_obs),
      window_s_(window_s),
      window_min_pairs_(std::max(1, window_min_pairs)) {}

void EdgeCouplingMap::closeWindowLocked(EdgeCouplingState& state, double t_end) {
  if (state.win_t0 < 0.0 || state.win_n <= 0) {
    state.win_t0 = -1.0;
    state.win_last = -1.0;
    state.win_n = 0;
    return;
  }

  // The PMI estimator evaluated on this window's counts alone: observed
  // co-occurrence against the product of this window's
  // marginals. With N_w = win_n pairs, the expected count under independence
  // is (win_a[ka]/N_w) * (win_b[kb]/N_w) * N_w = win_a[ka]*win_b[kb]/N_w.
  constexpr double kEps = 1.0e-6;
  const double n_w = static_cast<double>(state.win_n);
  const double t_mid = 0.5 * (state.win_t0 + t_end);
  const int K_a = static_cast<int>(state.win_co.rows());
  const int K_b = static_cast<int>(state.win_co.cols());
  for (int ka = 0; ka < K_a; ++ka) {
    for (int kb = 0; kb < K_b; ++kb) {
      const double c_obs = state.win_co(ka, kb);
      const double c_exp = state.win_a[static_cast<size_t>(ka)] *
                           state.win_b[static_cast<size_t>(kb)] / n_w;
      const double phi_hat = std::log((c_obs + kEps) / (c_exp + kEps));
      state.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)].update(t_mid,
                                                                               phi_hat);
    }
  }

  state.win_co.setZero();
  std::fill(state.win_a.begin(), state.win_a.end(), 0.0);
  std::fill(state.win_b.begin(), state.win_b.end(), 0.0);
  state.win_n = 0;
  state.win_t0 = -1.0;
  state.win_last = -1.0;
}

void EdgeCouplingMap::update(const GlobalIndex& i,
                             const std::vector<double>& resp_i,
                             const GlobalIndex& j,
                             const std::vector<double>& resp_j,
                             double t_seconds) {
  if (resp_i.empty() || resp_j.empty()) {
    return;
  }
  const auto key = EdgeKey::make(i, j);
  // Determine which side is A (the canonical 'a' index) so resp_A/B align.
  const bool i_is_a = (key.a == i);
  const std::vector<double>& resp_a = i_is_a ? resp_i : resp_j;
  const std::vector<double>& resp_b = i_is_a ? resp_j : resp_i;
  const int K_a = static_cast<int>(resp_a.size());
  const int K_b = static_cast<int>(resp_b.size());

  std::lock_guard<std::mutex> lock(mutex_);
  auto& state = edges_[key];

  if (!state.initialised || state.co_occurrence.rows() != K_a ||
      state.co_occurrence.cols() != K_b) {
    state.co_occurrence = Eigen::MatrixXd::Zero(K_a, K_b);
    state.pi_a.assign(static_cast<size_t>(K_a), 0.0);
    state.pi_b.assign(static_cast<size_t>(K_b), 0.0);
    state.n_paired = 0;
    state.win_co = Eigen::MatrixXd::Zero(K_a, K_b);
    state.win_a.assign(static_cast<size_t>(K_a), 0.0);
    state.win_b.assign(static_cast<size_t>(K_b), 0.0);
    state.win_n = 0;
    state.win_t0 = -1.0;
    state.win_last = -1.0;
    state.nudft_phi.assign(static_cast<size_t>(K_a),
                           std::vector<NUDFTModel>(static_cast<size_t>(K_b)));
    for (int ka = 0; ka < K_a; ++ka) {
      for (int kb = 0; kb < K_b; ++kb) {
        initNudft(state.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)],
                  shared_omegas_,
                  nudft_min_obs_);
      }
    }
    if (state.creation_timestamp_ns == 0 && t_seconds > 0.0) {
      state.creation_timestamp_ns = static_cast<uint64_t>(t_seconds * 1.0e9);
    }
    state.initialised = true;
  }

  // Update the all-history running marginals and paired count. Running
  // average: new = old + (x - old)/N.
  state.n_paired += 1;
  const double inv_N = 1.0 / static_cast<double>(state.n_paired);
  for (int ka = 0; ka < K_a; ++ka) {
    state.pi_a[static_cast<size_t>(ka)] +=
        (resp_a[static_cast<size_t>(ka)] - state.pi_a[static_cast<size_t>(ka)]) * inv_N;
  }
  for (int kb = 0; kb < K_b; ++kb) {
    state.pi_b[static_cast<size_t>(kb)] +=
        (resp_b[static_cast<size_t>(kb)] - state.pi_b[static_cast<size_t>(kb)]) * inv_N;
  }

  // Accumulate the joint co-occurrence tensor, all-history and for the open
  // sampling window. Each closed window yields one log-PMI sample per slot pair
  //   φ̂_ij^(k_i, k_j) = log((C_w + ε) / (π̄_a,w · π̄_b,w · N_w + ε)),
  // which the per-pair NUDFTs track: under independence numerator and
  // denominator agree and φ̂ ≈ 0; positive φ̂ = co-occurrence above chance,
  // negative = below chance. conditionalFlow multiplies the predicted φ
  // against the target marginal.

  // A gap longer than the window means the open window no longer spans
  // contiguous observation, so close it at its own last pair to keep the
  // sample's mid-time inside observed time.
  if (state.win_t0 >= 0.0 && (t_seconds - state.win_last) > window_s_) {
    closeWindowLocked(state, state.win_last);
  }
  if (state.win_t0 < 0.0) {
    state.win_t0 = t_seconds;
  }

  for (int ka = 0; ka < K_a; ++ka) {
    for (int kb = 0; kb < K_b; ++kb) {
      const double incr =
          resp_a[static_cast<size_t>(ka)] * resp_b[static_cast<size_t>(kb)];
      state.co_occurrence(ka, kb) += incr;
      state.win_co(ka, kb) += incr;
    }
  }

  for (int ka = 0; ka < K_a; ++ka) {
    state.win_a[static_cast<size_t>(ka)] += resp_a[static_cast<size_t>(ka)];
  }
  for (int kb = 0; kb < K_b; ++kb) {
    state.win_b[static_cast<size_t>(kb)] += resp_b[static_cast<size_t>(kb)];
  }
  state.win_n += 1;
  state.win_last = t_seconds;
  // A window holding a single pair yields φ̂ = 0 identically, so hold it open
  // past its nominal length until it carries enough pairs to be informative.
  if ((t_seconds - state.win_t0) >= window_s_ && state.win_n >= window_min_pairs_) {
    closeWindowLocked(state, t_seconds);
  }
}

std::optional<std::vector<double>> EdgeCouplingMap::conditionalFlow(
    const GlobalIndex& i,
    int k_i,
    const GlobalIndex& j,
    const std::vector<double>& marginal_j,
    double t_seconds,
    int nudft_order) const {
  if (marginal_j.empty()) {
    return std::nullopt;
  }

  const auto key = EdgeKey::make(i, j);
  const bool i_is_a = (key.a == i);

  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = edges_.find(key);
  if (it == edges_.end() || !it->second.initialised) {
    return std::nullopt;
  }
  const auto& state = it->second;

  // Map k_i to the canonical A/B side index.
  const int K_a = static_cast<int>(state.co_occurrence.rows());
  const int K_b = static_cast<int>(state.co_occurrence.cols());

  const int k_a = i_is_a ? k_i : -1;
  const int k_b_query = i_is_a ? -1 : k_i;
  (void)k_b_query;

  if (i_is_a) {
    if (k_a < 0 || k_a >= K_a) {
      return std::nullopt;
    }
    const int K_j = static_cast<int>(marginal_j.size());
    if (K_j != K_b) {
      return std::nullopt;
    }
    // p(k_j | k_i, t) ∝ marginal_j[k_j] * exp(φ[k_a][k_j](t))
    std::vector<double> result(static_cast<size_t>(K_b), 0.0);
    double sum = 0.0;
    for (int kb = 0; kb < K_b; ++kb) {
      const double phi = state.nudft_phi[static_cast<size_t>(k_a)][static_cast<size_t>(kb)]
                             .predict(t_seconds, nudft_order);
      result[static_cast<size_t>(kb)] = marginal_j[static_cast<size_t>(kb)] * std::exp(phi);
      sum += result[static_cast<size_t>(kb)];
    }
    if (sum > 0.0) {
      for (auto& v : result) {
        v /= sum;
      }
    }
    return result;
  } else {
    // i is the B side; query is p(k_j | k_b, t) where j is A.
    if (k_i < 0 || k_i >= K_b) {
      return std::nullopt;
    }
    const int K_j = static_cast<int>(marginal_j.size());
    if (K_j != K_a) {
      return std::nullopt;
    }
    std::vector<double> result(static_cast<size_t>(K_a), 0.0);
    double sum = 0.0;
    for (int ka = 0; ka < K_a; ++ka) {
      const double phi = state.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(k_i)]
                             .predict(t_seconds, nudft_order);
      result[static_cast<size_t>(ka)] = marginal_j[static_cast<size_t>(ka)] * std::exp(phi);
      sum += result[static_cast<size_t>(ka)];
    }
    if (sum > 0.0) {
      for (auto& v : result) {
        v /= sum;
      }
    }
    return result;
  }
}

void EdgeCouplingMap::mergeEdge(const EdgeKey& key, const EdgeCouplingState& other) {
  if (!other.initialised) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto& state = edges_[key];
  if (!state.initialised) {
    state = other;
    return;
  }

  const int K_a = static_cast<int>(state.co_occurrence.rows());
  const int K_b = static_cast<int>(state.co_occurrence.cols());
  if (other.co_occurrence.rows() != K_a || other.co_occurrence.cols() != K_b) {
    LOG_FIRST_N(WARNING, 5)
        << "[EdgeCoupling] mergeEdge skipped: K mismatch (this=(" << K_a << "," << K_b
        << "), other=(" << other.co_occurrence.rows() << "," << other.co_occurrence.cols()
        << "))";
    return;
  }

  // Pool sufficient statistics: co-occurrence is additive, marginals are
  // count-weighted averages. n_paired aggregates so the next update()
  // call's PMI denominator reflects the combined sample size.
  const double n_a = static_cast<double>(state.n_paired);
  const double n_b = static_cast<double>(other.n_paired);
  const double n_ab = n_a + n_b;
  state.co_occurrence += other.co_occurrence;
  if (n_ab > 0.0) {
    const double inv = 1.0 / n_ab;
    for (int ka = 0; ka < K_a; ++ka) {
      const double pa = static_cast<size_t>(ka) < state.pi_a.size()
                            ? state.pi_a[static_cast<size_t>(ka)]
                            : 0.0;
      const double pb_other = static_cast<size_t>(ka) < other.pi_a.size()
                                  ? other.pi_a[static_cast<size_t>(ka)]
                                  : 0.0;
      if (static_cast<size_t>(ka) < state.pi_a.size()) {
        state.pi_a[static_cast<size_t>(ka)] = (n_a * pa + n_b * pb_other) * inv;
      }
    }
    for (int kb = 0; kb < K_b; ++kb) {
      const double pb = static_cast<size_t>(kb) < state.pi_b.size()
                            ? state.pi_b[static_cast<size_t>(kb)]
                            : 0.0;
      const double pb_other = static_cast<size_t>(kb) < other.pi_b.size()
                                  ? other.pi_b[static_cast<size_t>(kb)]
                                  : 0.0;
      if (static_cast<size_t>(kb) < state.pi_b.size()) {
        state.pi_b[static_cast<size_t>(kb)] = (n_a * pb + n_b * pb_other) * inv;
      }
    }
  }
  state.n_paired += other.n_paired;
  for (int ka = 0; ka < K_a; ++ka) {
    for (int kb = 0; kb < K_b; ++kb) {
      state.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)].mergeFrom(
          other.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)]);
    }
  }
}

size_t EdgeCouplingMap::edgeCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return edges_.size();
}

namespace {

nlohmann::json indexArray(const GlobalIndex& idx) {
  return nlohmann::json::array({idx.x(), idx.y(), idx.z()});
}

bool indexFromArray(const nlohmann::json& j, GlobalIndex& out, std::string* error) {
  if (!j.is_array() || j.size() != 3 || !j[0].is_number_integer() ||
      !j[1].is_number_integer() || !j[2].is_number_integer()) {
    if (error) *error = "invalid EdgeCouplingMap index";
    return false;
  }
  out = GlobalIndex(j[0].get<int>(), j[1].get<int>(), j[2].get<int>());
  return true;
}

}  // namespace

nlohmann::json EdgeCouplingMap::toJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  nlohmann::json record;
  record["schema"] = "edge_coupling_map";
  record["version"] = 1;
  if (shared_omegas_) {
    record["omegas"] = *shared_omegas_;
  }
  record["edges"] = nlohmann::json::array();

  for (const auto& [key, state] : edges_) {
    if (!state.initialised) {
      continue;
    }
    nlohmann::json edge_json;
    edge_json["a"] = indexArray(key.a);
    edge_json["b"] = indexArray(key.b);

    const int K_a = static_cast<int>(state.co_occurrence.rows());
    const int K_b = static_cast<int>(state.co_occurrence.cols());
    edge_json["K_a"] = K_a;
    edge_json["K_b"] = K_b;
    edge_json["n_paired"] = state.n_paired;
    edge_json["creation_timestamp_ns"] = state.creation_timestamp_ns;

    nlohmann::json co = nlohmann::json::array();
    for (int ka = 0; ka < K_a; ++ka) {
      for (int kb = 0; kb < K_b; ++kb) {
        co.push_back(state.co_occurrence(ka, kb));
      }
    }
    edge_json["co_occurrence"] = co;
    edge_json["pi_a"] = state.pi_a;
    edge_json["pi_b"] = state.pi_b;

    nlohmann::json nudft = nlohmann::json::array();
    for (int ka = 0; ka < K_a; ++ka) {
      for (int kb = 0; kb < K_b; ++kb) {
        nudft.push_back(nudftToJson(
            state.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)]));
      }
    }
    edge_json["nudft_phi"] = nudft;

    record["edges"].push_back(std::move(edge_json));
  }
  return record;
}

bool EdgeCouplingMap::fromJson(const nlohmann::json& record, std::string* error) {
  if (!record.is_object() || !record.contains("edges") || !record.at("edges").is_array()) {
    if (error) *error = "invalid EdgeCouplingMap record";
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  edges_.clear();

  for (const auto& edge_json : record.at("edges")) {
    if (!edge_json.is_object()) {
      if (error) *error = "invalid EdgeCouplingMap edge entry";
      return false;
    }
    GlobalIndex a, b;
    if (!indexFromArray(edge_json.at("a"), a, error)) return false;
    if (!indexFromArray(edge_json.at("b"), b, error)) return false;
    const EdgeKey key = EdgeKey::make(a, b);

    EdgeCouplingState state;
    const int K_a = edge_json.value("K_a", 0);
    const int K_b = edge_json.value("K_b", 0);
    if (K_a <= 0 || K_b <= 0) {
      continue;
    }
    state.co_occurrence = Eigen::MatrixXd::Zero(K_a, K_b);
    state.pi_a.assign(static_cast<size_t>(K_a), 0.0);
    state.pi_b.assign(static_cast<size_t>(K_b), 0.0);
    // The open window is transient and never serialized, but its accumulators
    // must be sized here: update() only sizes them on an edge's first sight,
    // and a restored edge skips that path.
    state.win_co = Eigen::MatrixXd::Zero(K_a, K_b);
    state.win_a.assign(static_cast<size_t>(K_a), 0.0);
    state.win_b.assign(static_cast<size_t>(K_b), 0.0);
    state.win_n = 0;
    state.win_t0 = -1.0;
    state.win_last = -1.0;
    state.n_paired = edge_json.value("n_paired", static_cast<int64_t>(0));
    state.creation_timestamp_ns =
        edge_json.value("creation_timestamp_ns", static_cast<uint64_t>(0));

    if (edge_json.contains("co_occurrence") && edge_json.at("co_occurrence").is_array()) {
      const auto& co = edge_json.at("co_occurrence");
      if (static_cast<int>(co.size()) == K_a * K_b) {
        for (int ka = 0; ka < K_a; ++ka) {
          for (int kb = 0; kb < K_b; ++kb) {
            state.co_occurrence(ka, kb) = co.at(ka * K_b + kb).get<double>();
          }
        }
      }
    }
    if (edge_json.contains("pi_a") && edge_json.at("pi_a").is_array()) {
      const auto& pa = edge_json.at("pi_a");
      for (int ka = 0; ka < K_a && ka < static_cast<int>(pa.size()); ++ka) {
        state.pi_a[static_cast<size_t>(ka)] = pa.at(ka).get<double>();
      }
    }
    if (edge_json.contains("pi_b") && edge_json.at("pi_b").is_array()) {
      const auto& pb = edge_json.at("pi_b");
      for (int kb = 0; kb < K_b && kb < static_cast<int>(pb.size()); ++kb) {
        state.pi_b[static_cast<size_t>(kb)] = pb.at(kb).get<double>();
      }
    }

    state.nudft_phi.assign(static_cast<size_t>(K_a),
                           std::vector<NUDFTModel>(static_cast<size_t>(K_b)));
    const bool have_nudfts = edge_json.contains("nudft_phi") &&
                             edge_json.at("nudft_phi").is_array() &&
                             static_cast<int>(edge_json.at("nudft_phi").size()) == K_a * K_b;
    for (int ka = 0; ka < K_a; ++ka) {
      for (int kb = 0; kb < K_b; ++kb) {
        auto& m = state.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)];
        if (have_nudfts) {
          std::string err;
          if (!nudftFromJson(edge_json.at("nudft_phi").at(ka * K_b + kb), m, &err)) {
            if (error) *error = "EdgeCouplingMap NUDFT: " + err;
            return false;
          }
        }
        // Bind shared omegas + min_obs without resetting state. initNudft
        // would clobber the just-loaded gamma/var via reset(), so set the
        // metadata directly here.
        m.omegas_ptr = shared_omegas_;
        m.min_observations = nudft_min_obs_;
      }
    }

    state.initialised = true;
    edges_[key] = std::move(state);
  }
  return true;
}

EdgeCouplingMap::PruneReport EdgeCouplingMap::pruneEdges(
    const std::function<bool(const GlobalIndex&, const GlobalIndex&)>& keep) {
  PruneReport report;
  if (!keep) {
    return report;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  // Collect first, erase after: erasing while iterating an unordered_map
  // invalidates the iterator being held.
  std::vector<EdgeKey> to_erase;
  for (const auto& [key, state] : edges_) {
    if (keep(key.a, key.b)) {
      ++report.edges_kept;
      report.paired_kept += state.n_paired;
      continue;
    }
    ++report.edges_dropped;
    report.paired_dropped += state.n_paired;
    to_erase.push_back(key);
  }
  for (const auto& key : to_erase) {
    edges_.erase(key);
  }
  return report;
}

void EdgeCouplingMap::remapIndex(const GlobalIndex& old_index, const GlobalIndex& new_index) {
  if (old_index == new_index) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<EdgeKey> to_erase;
  std::vector<std::pair<EdgeKey, EdgeCouplingState>> to_insert;

  for (auto& [key, state] : edges_) {
    const bool a_match = (key.a == old_index);
    const bool b_match = (key.b == old_index);
    if (!a_match && !b_match) {
      continue;
    }
    const GlobalIndex new_a = a_match ? new_index : key.a;
    const GlobalIndex new_b = b_match ? new_index : key.b;
    to_erase.push_back(key);
    to_insert.emplace_back(EdgeKey::make(new_a, new_b), std::move(state));
  }

  for (const auto& k : to_erase) {
    edges_.erase(k);
  }
  for (auto& [new_key, incoming] : to_insert) {
    auto& dest = edges_[new_key];
    if (!dest.initialised) {
      dest = std::move(incoming);
      continue;
    }
    // Target already exists: merge inline (mutex already held, cannot call
    // the public mergeEdge which would deadlock).
    if (!incoming.initialised) {
      continue;
    }
    const int K_a = static_cast<int>(dest.co_occurrence.rows());
    const int K_b = static_cast<int>(dest.co_occurrence.cols());
    if (incoming.co_occurrence.rows() != K_a || incoming.co_occurrence.cols() != K_b) {
      LOG_FIRST_N(WARNING, 5) << "[EdgeCoupling] remapIndex merge skipped: K mismatch";
      continue;
    }
    // Same statistical pool as mergeEdge: additive co-occurrence,
    // count-weighted marginals, n_paired sum. Inline here because mutex_
    // is already held and the public mergeEdge would deadlock.
    const double n_a = static_cast<double>(dest.n_paired);
    const double n_b = static_cast<double>(incoming.n_paired);
    const double n_ab = n_a + n_b;
    dest.co_occurrence += incoming.co_occurrence;
    if (n_ab > 0.0) {
      const double inv = 1.0 / n_ab;
      for (int ka = 0; ka < K_a; ++ka) {
        const double pa = static_cast<size_t>(ka) < dest.pi_a.size()
                              ? dest.pi_a[static_cast<size_t>(ka)]
                              : 0.0;
        const double pa_other = static_cast<size_t>(ka) < incoming.pi_a.size()
                                    ? incoming.pi_a[static_cast<size_t>(ka)]
                                    : 0.0;
        if (static_cast<size_t>(ka) < dest.pi_a.size()) {
          dest.pi_a[static_cast<size_t>(ka)] = (n_a * pa + n_b * pa_other) * inv;
        }
      }
      for (int kb = 0; kb < K_b; ++kb) {
        const double pb = static_cast<size_t>(kb) < dest.pi_b.size()
                              ? dest.pi_b[static_cast<size_t>(kb)]
                              : 0.0;
        const double pb_other = static_cast<size_t>(kb) < incoming.pi_b.size()
                                    ? incoming.pi_b[static_cast<size_t>(kb)]
                                    : 0.0;
        if (static_cast<size_t>(kb) < dest.pi_b.size()) {
          dest.pi_b[static_cast<size_t>(kb)] = (n_a * pb + n_b * pb_other) * inv;
        }
      }
    }
    dest.n_paired += incoming.n_paired;
    for (int ka = 0; ka < K_a; ++ka) {
      for (int kb = 0; kb < K_b; ++kb) {
        dest.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)].mergeFrom(
            incoming.nudft_phi[static_cast<size_t>(ka)][static_cast<size_t>(kb)]);
      }
    }
  }
}

}  // namespace kairos
