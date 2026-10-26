/**
 * @file bow_agent_solver.h
 * @brief Agent-layer registration solver that runs entirely on data already
 *        present in the scene graph. It scores a bag-of-words similarity
 *        between the query and matched agent nodes and, when the match is
 *        accepted, emits a relative-pose constraint derived from the two
 *        agents' stored world poses rather than from fresh visual features.
 */
#pragma once

#include <config_utilities/factory.h>
#include <cstdint>
#include <unordered_map>

#include <hydra/loop_closure/registration.h>

namespace kairos {

/// @brief Resolves agent-layer loop closures from stored bag-of-words
/// descriptors without invoking an external visual registration service.
///
/// On each call the solver computes an L1 bag-of-words similarity between the
/// query and matched agent nodes. A score below the configured threshold
/// returns an invalid solution so the detection cascade can fall through to
/// the object and place layers. When the score passes and the match is not
/// suppressed by the deduplication window, it returns a valid solution whose
/// relative pose is computed from the two agents' world poses. That pose adds
/// a constraint to the backend without supplying an independent drift
/// estimate, which keeps a false match from warping the trajectory.
class BowAgentSolver : public hydra::lcd::DsgRegistrationSolver {
 public:
  /// @brief Parameters controlling acceptance and deduplication.
  struct Config {
    double min_score = 0.05;       ///< Minimum bag-of-words similarity in [0, 1] to accept a match.
    double dedup_window_s = 30.0;  ///< Suppress repeat closures against the same match anchor within this many seconds.
  };

  /// @brief Construct the solver with the given configuration.
  /// @param config Acceptance threshold and deduplication window.
  explicit BowAgentSolver(const Config& config);

  ~BowAgentSolver() override = default;

  /// @brief Score the candidate match and produce a registration solution.
  /// @param dsg Scene graph holding the agent attributes.
  /// @param match Candidate match describing the query and matched roots.
  /// @param query_agent_id Identifier of the query agent node.
  /// @return A valid solution when the match passes scoring and deduplication, otherwise an invalid solution.
  hydra::lcd::RegistrationSolution solve(const spark_dsg::DynamicSceneGraph& dsg,
                                         const hydra::lcd::DsgRegistrationInput& match,
                                         spark_dsg::NodeId query_agent_id) const override;

 private:
  Config config_;  ///< Acceptance threshold and deduplication window.
  /// Maps each matched anchor to the timestamp (ns) at which a closure last
  /// fired against it. Marked mutable so the const solve() can record fires;
  /// it grows by at most one entry per distinct anchor.
  mutable std::unordered_map<spark_dsg::NodeId, uint64_t> last_fired_against_;

};

void declare_config(BowAgentSolver::Config& config);

}  // namespace kairos
