/* -----------------------------------------------------------------------------
 * BowAgentSolver implementation. See bow_agent_solver.h.
 * -------------------------------------------------------------------------- */
#include "kairos/loop_closure/bow_agent_solver.h"

#include <config_utilities/factory.h>

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

#include <glog/logging.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/printing.h>

#include <Eigen/Geometry>
#include <cmath>

namespace kairos {

using hydra::lcd::DsgRegistrationInput;
using hydra::lcd::RegistrationSolution;
using spark_dsg::AgentNodeAttributes;
using spark_dsg::DynamicSceneGraph;
using spark_dsg::NodeId;
using spark_dsg::NodeSymbol;

namespace {

const auto registration =
    config::RegistrationWithConfig<hydra::lcd::DsgRegistrationSolver,
                                   BowAgentSolver,
                                   BowAgentSolver::Config>("BowAgentSolver");

}  // namespace

namespace {

// DBoW2-style L1 similarity for sparse BoW vectors stored as parallel
// (word_ids, word_values) arrays sorted ascending by word_id (which is
// what DBoW3::Vocabulary::transform produces).
//
//   score(v, w) = 1 - 0.5 * sum_{i in v∪w} |v[i] - w[i]|
//
// Range [0, 1]; identical vectors score 1.0, disjoint vectors with
// L1-normalised weights score 0.0.
double bowL1Similarity(const AgentNodeAttributes::BowIdVector& a_ids,
                       const Eigen::VectorXf& a_vals,
                       const AgentNodeAttributes::BowIdVector& b_ids,
                       const Eigen::VectorXf& b_vals) {
  Eigen::Index i = 0;
  Eigen::Index j = 0;
  double sum = 0.0;
  while (i < a_ids.size() && j < b_ids.size()) {
    if (a_ids(i) < b_ids(j)) {
      sum += std::abs(static_cast<double>(a_vals(i)));
      ++i;
    } else if (a_ids(i) > b_ids(j)) {
      sum += std::abs(static_cast<double>(b_vals(j)));
      ++j;
    } else {
      sum += std::abs(static_cast<double>(a_vals(i)) -
                      static_cast<double>(b_vals(j)));
      ++i;
      ++j;
    }
  }
  while (i < a_ids.size()) {
    sum += std::abs(static_cast<double>(a_vals(i)));
    ++i;
  }
  while (j < b_ids.size()) {
    sum += std::abs(static_cast<double>(b_vals(j)));
    ++j;
  }
  return 1.0 - 0.5 * sum;
}

}  // namespace

void declare_config(BowAgentSolver::Config& config) {
  using namespace config;
  name("BowAgentSolver::Config");
  field(config.min_score, "min_score");
  field(config.dedup_window_s, "dedup_window_s");
  check(config.min_score, GE, 0.0, "min_score");
  check(config.min_score, LE, 1.0, "min_score");
  check(config.dedup_window_s, GE, 0.0, "dedup_window_s");
}

BowAgentSolver::BowAgentSolver(const Config& config)
    : config_(config::checkValid(config)) {
  LOG(INFO) << "[BowAgentSolver] min_score=" << config_.min_score
            << " dedup_window_s=" << config_.dedup_window_s;
}

RegistrationSolution BowAgentSolver::solve(const DynamicSceneGraph& dsg,
                                           const DsgRegistrationInput& match,
                                           NodeId query_agent_id) const {
  // For the agent layer, `match.match_root` is the *place* root (parent of
  // the matched agent leaf), and the matched agent ID lives in
  // `match.match_nodes`. Casting match_root to AgentNodeAttributes would throw
  // std::bad_cast, so the matched agent must be taken from match_nodes.
  if (match.match_nodes.empty()) {
    return {};
  }
  const NodeId match_agent_id = *match.match_nodes.begin();

  if (!dsg.hasNode(query_agent_id) || !dsg.hasNode(match_agent_id)) {
    LOG(WARNING) << "[BowAgentSolver] query or match node missing in DSG: "
                 << NodeSymbol(query_agent_id).getLabel() << " -> "
                 << NodeSymbol(match_agent_id).getLabel();
    return {};
  }

  const auto& query_attrs =
      dsg.getNode(query_agent_id).attributes<AgentNodeAttributes>();
  const auto& match_attrs =
      dsg.getNode(match_agent_id).attributes<AgentNodeAttributes>();

  if (query_attrs.dbow_ids.size() == 0 || match_attrs.dbow_ids.size() == 0) {
    LOG(INFO) << "[BowAgentSolver] missing BoW on "
              << NodeSymbol(query_agent_id).getLabel() << "(words="
              << query_attrs.dbow_ids.size() << ") or "
              << NodeSymbol(match_agent_id).getLabel() << "(words="
              << match_attrs.dbow_ids.size() << ")";
    return {};
  }

  // Similarity in [0, 1] between the two agents' bag-of-words (higher = more
  // similar); this is the appearance evidence for the loop closure.
  const double score = bowL1Similarity(query_attrs.dbow_ids,
                                       query_attrs.dbow_values,
                                       match_attrs.dbow_ids,
                                       match_attrs.dbow_values);

  // Promoted to INFO so we can see the score distribution without bumping
  // glog --v. Once tuned, downgrade to VLOG(3).
  LOG(INFO) << "[BowAgentSolver] solve "
            << NodeSymbol(query_agent_id).getLabel() << " -> "
            << NodeSymbol(match_agent_id).getLabel() << " score=" << score
            << " threshold=" << config_.min_score;

  if (score < config_.min_score) {
    return {};
  }

  // Suppress repeated closures against the same match anchor within a time
  // window, so a stationary or slowly-moving robot revisiting one place
  // doesn't fire a burst of near-identical loop closures.
  const uint64_t now_ns = query_attrs.timestamp.count();
  const auto dedup_window_ns =
      static_cast<uint64_t>(config_.dedup_window_s * 1e9);

  auto last_iter = last_fired_against_.find(match_agent_id);
  if (last_iter != last_fired_against_.end() &&
      now_ns >= last_iter->second &&
      (now_ns - last_iter->second) < dedup_window_ns) {
    LOG(INFO) << "[BowAgentSolver] dedup-suppressed "
              << NodeSymbol(query_agent_id).getLabel() << " -> "
              << NodeSymbol(match_agent_id).getLabel() << " (score=" << score
              << ", last fired " << (now_ns - last_iter->second) * 1e-9
              << "s ago)";
    return {};
  }

  // Compute odometry-derived match_T_query so the backend factor agrees with
  // existing odometry edges (no drift correction, but agent-LCD short-circuits
  // the cascade so places-LCD doesn't run on the same revisit).
  Eigen::Isometry3d world_T_query = Eigen::Isometry3d::Identity();
  world_T_query.linear() = query_attrs.world_R_body.toRotationMatrix();
  world_T_query.translation() = query_attrs.position;

  Eigen::Isometry3d world_T_match = Eigen::Isometry3d::Identity();
  world_T_match.linear() = match_attrs.world_R_body.toRotationMatrix();
  world_T_match.translation() = match_attrs.position;

  const Eigen::Isometry3d match_T_query = world_T_match.inverse() * world_T_query;
  Eigen::Quaterniond match_q_query(match_T_query.linear());
  match_q_query.normalize();

  last_fired_against_[match_agent_id] = now_ns;

  LOG(INFO) << "[BowAgentSolver] valid LC "
            << NodeSymbol(query_agent_id).getLabel() << " -> "
            << NodeSymbol(match_agent_id).getLabel() << " (score=" << score
            << ")";

  return {true,
          query_agent_id,
          match_agent_id,
          match_T_query.translation(),
          match_q_query,
          -1};
}

}  // namespace kairos
