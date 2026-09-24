#pragma once

/**
 * @file edge_coupling_map.h
 * @brief Pairwise coupling between the directional mixtures of adjacent voxels.
 *        For each undirected voxel-pair edge it tracks the empirical
 *        co-occurrence of the two cells' mixture components, per-side running
 *        marginals, and a spectral predictor on the log point-wise mutual
 *        information so the coupling stays centred at zero under independence.
 *        From these it answers conditional flow queries (the distribution over
 *        one cell's components given the other's). Owns the canonical edge-key
 *        type and the container of per-edge state.
 */

#include <Eigen/Core>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hydra/reconstruction/voxel_types.h>

#include "kairos/flow/config.h"
#include "kairos/flow/nudft.h"

namespace kairos {
using namespace hydra;  // base-Hydra types (temporary; tighten to explicit using-decls)

/// Canonical key for an undirected voxel-pair edge: always stores the
/// lexicographically smaller GlobalIndex first so that (i,j) and (j,i) hash
/// to the same entry.
struct EdgeKey {
  GlobalIndex a;
  GlobalIndex b;

  bool operator==(const EdgeKey& other) const {
    return a == other.a && b == other.b;
  }

  static EdgeKey make(const GlobalIndex& i, const GlobalIndex& j) {
    if (std::make_tuple(i.x(), i.y(), i.z()) <= std::make_tuple(j.x(), j.y(), j.z())) {
      return {i, j};
    }
    return {j, i};
  }
};

/// Hash functor for EdgeKey, mixing all six voxel coordinates.
struct EdgeKeyHash {
  size_t operator()(const EdgeKey& k) const noexcept {
    // Combine hashes of all six coordinates.
    size_t h = 0;
    auto mix = [&h](int v) {
      h ^= std::hash<int>{}(v) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
    };
    mix(k.a.x()); mix(k.a.y()); mix(k.a.z());
    mix(k.b.x()); mix(k.b.y()); mix(k.b.z());
    return h;
  }
};

/// Per-edge state for the pairwise coupling between two cells' mixture
/// components. Tracks the empirical co-occurrence tensor over the two cells'
/// components, per-side running marginal mixing weights, and a spectral
/// predictor per component pair on the *log point-wise mutual information*
/// phi = log((C + eps) / (pi_a * pi_b * N + eps)), sampled once per closed
/// window from that window's own counts. Modelling log-PMI rather
/// than raw co-occurrence keeps the coupling centred at zero under
/// independence and bounded as the observation count N grows, so the
/// conditional query stays well-behaved.
struct EdgeCouplingState {
  /// Co-occurrence counts; outer index = component on side A, inner = side B.
  Eigen::MatrixXd co_occurrence;                          ///< K_A x K_B
  std::vector<std::vector<NUDFTModel>> nudft_phi;         ///< log-PMI predictor per [K_A][K_B]
  /// Running per-side marginal mixing weights, one running average per side.
  /// pi_a is the marginal of EdgeKey::a, pi_b of EdgeKey::b; each is the
  /// running average of the per-observation responsibility vectors seen in
  /// update().
  std::vector<double> pi_a;
  std::vector<double> pi_b;
  /// Total paired observations on this edge. Distinct from any single
  /// predictor's count (those are per component pair).
  int64_t n_paired = 0;
  uint64_t creation_timestamp_ns = 0;  ///< Time of this edge's first observation (ns).

  bool initialised = false;  ///< True once at least one update has been applied.

  /// @name Open sample window
  /// Accumulators for the window currently being filled. They hold
  /// this window's co-occurrence mass and per-side responsibility sums, from
  /// which one time-local log-PMI sample per component pair is formed when the
  /// window closes. Transient by design: an open window is in-flight
  /// measurement, so it is neither serialized nor merged, and at most one
  /// window's pairs are lost across a save/load boundary.
  /// @{
  Eigen::MatrixXd win_co;      ///< K_A x K_B co-occurrence mass inside the window.
  std::vector<double> win_a;   ///< Per-slot responsibility sums, side A.
  std::vector<double> win_b;   ///< Per-slot responsibility sums, side B.
  int64_t win_n = 0;           ///< Paired observations inside the window.
  double win_t0 = -1.0;        ///< Window start time (s); negative when closed.
  double win_last = -1.0;      ///< Time of the last pair folded in (s).
  /// @}
};

/// Owns the voxel-pair coupling tensors. Lives in FlowTemporalModule, not in
/// the scene graph (coupling data is hundreds of floats per edge — too heavy
/// for the graph schema). Thread safety: the public API is protected by an
/// internal mutex; callers must not hold FlowMap's mutex while calling these
/// methods, to avoid lock inversion.
class EdgeCouplingMap {
 public:
  /// @param shared_omegas Candidate angular frequencies, shared with the cell
  ///        predictors so both fit the same periods.
  /// @param nudft_min_obs Samples a per-pair predictor needs before it is
  ///        trusted.
  /// @param window_s Length (s, > 0) of the window over which one log-PMI
  ///        sample is formed: the shortest candidate period divided by
  ///        FlowMapConfig::edge_coupling_samples_per_period.
  /// @param window_min_pairs Paired observations a window must hold before it
  ///        may close.
  explicit EdgeCouplingMap(std::shared_ptr<const std::vector<double>> shared_omegas,
                           int nudft_min_obs,
                           double window_s,
                           int window_min_pairs = 2);

  /// Increment the co-occurrence tensor for edge (i,j) using per-component
  /// responsibility vectors resp_i and resp_j. Resizes tensors on first call
  /// for this edge. Also updates the time-varying predictor per (k_i, k_j).
  void update(const GlobalIndex& i,
              const std::vector<double>& resp_i,
              const GlobalIndex& j,
              const std::vector<double>& resp_j,
              double t_seconds);

  /// Conditional flow query for the distribution over side-j components given
  /// side-i component k_i at time t: proportional to marginal_j[k_j] *
  /// exp(phi_ij(t)). Returns nullopt if the edge has no data or the query
  /// components are out of range. @p marginal_j is the target cell's mixing
  /// weights at time t (from its own predictor).
  std::optional<std::vector<double>> conditionalFlow(const GlobalIndex& i,
                                                     int k_i,
                                                     const GlobalIndex& j,
                                                     const std::vector<double>& marginal_j,
                                                     double t_seconds,
                                                     int nudft_order) const;

  /// Merge another state into the edge @p key, used on loop-closure collisions.
  /// Count-weighted pool of co-occurrence counts and predictor coefficients.
  void mergeEdge(const EdgeKey& key, const EdgeCouplingState& other);

  /// Number of edges currently tracked.
  size_t edgeCount() const;

  /// Remove all edges that reference @p old_index and re-insert them under
  /// @p new_index. Used during FlowMap::remapCells.
  void remapIndex(const GlobalIndex& old_index, const GlobalIndex& new_index);

  /// Drop every edge for which @p keep returns false, and return how many were
  /// dropped along with the paired observations they held. The predicate sees
  /// the two voxel indices of each edge. Used to restrict the coupling
  /// neighbourhood to traversable adjacency: the caller supplies the test
  /// against the place topology, which this class does not know about.
  struct PruneReport {
    size_t edges_dropped = 0;    ///< Edges removed by the predicate.
    size_t edges_kept = 0;       ///< Edges that satisfied the predicate.
    int64_t paired_dropped = 0;  ///< Paired observations held by dropped edges.
    int64_t paired_kept = 0;     ///< Paired observations held by kept edges.
  };
  PruneReport pruneEdges(
      const std::function<bool(const GlobalIndex&, const GlobalIndex&)>& keep);

  /// Serialize / restore the full coupling state (co-occurrence, marginals,
  /// predictors). The lock is taken internally, so these are safe to call
  /// while update() / conditionalFlow() run on other threads. fromJson clears
  /// existing state first.
  nlohmann::json toJson() const;
  bool fromJson(const nlohmann::json& record, std::string* error = nullptr);

 private:
  /// Form one log-PMI sample per component pair from the open window's own
  /// counts and feed it to the corresponding predictor at the window's
  /// mid-time, then clear the window. Caller must hold mutex_.
  void closeWindowLocked(EdgeCouplingState& state, double t_end);

  mutable std::mutex mutex_;
  std::unordered_map<EdgeKey, EdgeCouplingState, EdgeKeyHash> edges_;
  std::shared_ptr<const std::vector<double>> shared_omegas_;
  int nudft_min_obs_ = 20;
  double window_s_ = 300.0;
  int window_min_pairs_ = 2;
};

}  // namespace kairos
