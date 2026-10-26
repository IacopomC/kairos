/* -----------------------------------------------------------------------------
 * Copyright 2024 Iacopo Catalano.
 * All Rights Reserved
 * -------------------------------------------------------------------------- */
#include "kairos/archetypes/update_archetypes_functor.h"

#include <config_utilities/factory.h>

#include <config_utilities/config.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_symbol.h>

#include <cmath>
#include <queue>

#include <hydra/utils/timing_utilities.h>

#include "kairos/flow/poisson_model.h"

namespace kairos {

using hydra::SharedDsgInfo;
using hydra::UpdateInfo;
using hydra::timing::ScopedTimer;
using spark_dsg::ArchetypeNodeAttributes;
using spark_dsg::ArchetypeType;
using spark_dsg::DynamicSceneGraph;
using spark_dsg::NodeId;
using spark_dsg::NodeSymbol;
using spark_dsg::SceneGraphLayer;
using spark_dsg::SceneGraphNode;

namespace {

const auto registration =
    config::RegistrationWithConfig<hydra::UpdateFunctor,
                                   UpdateArchetypesFunctor,
                                   UpdateArchetypesFunctor::Config>(
        "UpdateArchetypesFunctor");

}  // namespace

void declare_config(UpdateArchetypesFunctor::Config& config) {
  using namespace config;
  name("UpdateArchetypesFunctor::Config");
  field(config.enable, "enable");
  field(config.dominant_weight_threshold, "dominant_weight_threshold");
  field(config.min_flow_support, "min_flow_support");
  field(config.min_probability, "min_probability");
  field(config.presence_horizon_s, "presence_horizon_s");
  check(config.presence_horizon_s, GE, 0.0, "presence_horizon_s");
  field(config.places_layer, "places_layer");
  field(config.archetype_layer, "archetype_layer");
  field<CharConversion>(config.prefix, "prefix");
}

UpdateArchetypesFunctor::UpdateArchetypesFunctor(const Config& config)
    : config(config::checkValid(config)) {
  LOG(INFO) << "[Archetypes] UpdateArchetypesFunctor constructed";
}

ArchetypeType classifyArchetype(double probability,
                                int support,
                                const std::vector<double>& weights,
                                double dominant_weight_threshold,
                                int min_flow_support,
                                double min_probability) {
  if (probability < min_probability) {
    return ArchetypeType::STATIC;
  }
  if (support < min_flow_support) {
    return ArchetypeType::STATIC;
  }

  int dominant_count = 0;
  for (const double weight : weights) {
    if (weight >= dominant_weight_threshold) {
      ++dominant_count;
    }
  }

  if (dominant_count == 0) {
    // Evidence present (the probability and support gates passed) but no slot
    // reaches the dominance threshold: mass is spread across many headings with
    // none dominant.
    return ArchetypeType::DIFFUSE;
  } else if (dominant_count == 1) {
    return ArchetypeType::UNIMODAL;
  } else if (dominant_count == 2) {
    return ArchetypeType::BIMODAL;
  } else {
    return ArchetypeType::MULTIMODAL;
  }
}

ArchetypeType UpdateArchetypesFunctor::classifyPlace(
    const SceneGraphNode& node) const {
  // Pull the flow evidence out of the place metadata, then defer to the shared
  // classifier so the backend and the bindings apply one identical rule.
  const auto& meta = node.attributes().metadata();

  if (!meta.contains("flow_num_components") || !meta.contains("flow_occupancy")) {
    return ArchetypeType::STATIC;
  }
  if (!meta.contains("flow_support")) {
    return ArchetypeType::STATIC;
  }

  // Places store an occupancy, which carries no horizon. Turn it into a
  // probability here, at the functor's own horizon, so the STATIC gate keeps
  // comparing like with like against min_probability.
  const double occupancy = meta["flow_occupancy"].get<double>();
  const double dwell_s =
      meta.contains("flow_dwell_s") ? meta["flow_dwell_s"].get<double>() : -1.0;
  const double probability =
      presenceFromOccupancy(occupancy, dwell_s, config.presence_horizon_s);
  const int support = meta["flow_support"].get<int>();

  std::vector<double> weights;
  if (meta.contains("flow_components") && meta["flow_components"].is_array()) {
    for (const auto& comp : meta["flow_components"]) {
      if (comp.contains("weight")) {
        weights.push_back(comp["weight"].get<double>());
      }
    }
  }

  return classifyArchetype(probability,
                           support,
                           weights,
                           config.dominant_weight_threshold,
                           config.min_flow_support,
                           config.min_probability);
}

std::map<size_t, std::set<NodeId>> UpdateArchetypesFunctor::clusterByArchetype(
    const SceneGraphLayer& places,
    const std::map<NodeId, ArchetypeType>& classifications) const {
  // Group places into connected components where every member shares the same
  // archetype type and is reachable through same-type sibling edges (BFS flood).
  std::map<size_t, std::set<NodeId>> clusters;
  std::set<NodeId> visited;
  size_t cluster_id = 0;

  for (const auto& [node_id, node] : places.nodes()) {
    if (visited.count(node_id)) {
      continue;
    }

    auto class_it = classifications.find(node_id);
    if (class_it == classifications.end()) {
      continue;
    }

    const ArchetypeType type = class_it->second;

    // BFS to find connected component of same archetype type
    std::set<NodeId> component;
    std::queue<NodeId> frontier;
    frontier.push(node_id);
    visited.insert(node_id);

    while (!frontier.empty()) {
      const NodeId current = frontier.front();
      frontier.pop();
      component.insert(current);

      const auto* current_node = places.findNode(current);
      if (!current_node) {
        continue;
      }

      for (const auto sibling_id : current_node->siblings()) {
        if (visited.count(sibling_id)) {
          continue;
        }

        auto sib_class = classifications.find(sibling_id);
        if (sib_class == classifications.end() || sib_class->second != type) {
          continue;
        }

        visited.insert(sibling_id);
        frontier.push(sibling_id);
      }
    }

    if (!component.empty()) {
      clusters[cluster_id] = std::move(component);
      ++cluster_id;
    }
  }

  return clusters;
}

void UpdateArchetypesFunctor::rewriteArchetypes(
    DynamicSceneGraph& graph,
    const std::map<size_t, std::set<NodeId>>& clusters,
    const std::map<NodeId, ArchetypeType>& classifications) const {
  // Step 1: Remove all old archetype nodes (same pattern as rewriteRooms)
  const auto archetypes = graph.findLayer(config.archetype_layer);
  if (!archetypes) {
    LOG_FIRST_N(ERROR, 1) << "[Archetypes] graph has no layer '" << config.archetype_layer
                          << "': add it to the graph layers";
    return;
  }

  {
    std::vector<NodeId> to_remove;
    for (const auto& [id, node] : archetypes->nodes()) {
      to_remove.push_back(id);
    }
    for (const auto node_id : to_remove) {
      graph.removeNode(node_id);
    }
  }

  // Step 2: Create new archetype nodes
  for (const auto& [cluster_id, members] : clusters) {
    if (members.empty()) {
      continue;
    }

    const ArchetypeType type = classifications.at(*members.begin());

    // Summarize the cluster into a single archetype node: centroid position,
    // mean confidence, and the pooled dominant flow directions of its members.
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    double confidence_sum = 0.0;
    int confidence_count = 0;
    std::vector<double> all_thetas;
    std::vector<double> all_rhos;

    for (const auto member_id : members) {
      const auto* node = graph.findNode(member_id);
      if (!node) {
        continue;
      }

      centroid += node->attributes().position;

      const auto& meta = node->attributes().metadata();
      if (meta.contains("flow_confidence")) {
        confidence_sum += meta["flow_confidence"].get<double>();
        ++confidence_count;
      }

      if (meta.contains("flow_components") && meta["flow_components"].is_array()) {
        for (const auto& comp : meta["flow_components"]) {
          if (comp.contains("weight") &&
              comp["weight"].get<double>() >= config.dominant_weight_threshold) {
            if (comp.contains("theta")) {
              all_thetas.push_back(comp["theta"].get<double>());
            }
            if (comp.contains("rho")) {
              all_rhos.push_back(comp["rho"].get<double>());
            }
          }
        }
      }
    }

    centroid /= static_cast<double>(members.size());

    auto attrs = std::make_unique<ArchetypeNodeAttributes>();
    attrs->position = centroid;
    attrs->archetype_type = type;
    attrs->member_count = static_cast<uint32_t>(members.size());
    attrs->confidence =
        confidence_count > 0 ? confidence_sum / confidence_count : 0.0;
    attrs->dominant_thetas = std::move(all_thetas);
    attrs->dominant_rhos = std::move(all_rhos);

    switch (type) {
      case ArchetypeType::STATIC:
        attrs->color = spark_dsg::Color(128, 128, 128);
        attrs->name = "static";
        break;
      case ArchetypeType::UNIMODAL:
        attrs->color = spark_dsg::Color(0, 200, 0);
        attrs->name = "unimodal";
        break;
      case ArchetypeType::BIMODAL:
        attrs->color = spark_dsg::Color(0, 100, 255);
        attrs->name = "bimodal";
        break;
      case ArchetypeType::MULTIMODAL:
        attrs->color = spark_dsg::Color(255, 50, 0);
        attrs->name = "multimodal";
        break;
      case ArchetypeType::DIFFUSE:
        attrs->color = spark_dsg::Color(230, 200, 0);
        attrs->name = "diffuse";
        break;
    }

    const NodeId archetype_id = NodeSymbol(config.prefix, cluster_id);
    graph.emplaceNode(config.archetype_layer, archetype_id, std::move(attrs));
  }

  // Step 3: Create archetype -> place parent edges
  for (const auto& [cluster_id, members] : clusters) {
    const NodeId archetype_id = NodeSymbol(config.prefix, cluster_id);
    for (const auto member_id : members) {
      // add edge enforcing parent invariants
      graph.insertEdge(archetype_id, member_id, nullptr, true);
    }
  }
}

void UpdateArchetypesFunctor::call(const DynamicSceneGraph&,
                                   SharedDsgInfo& dsg,
                                   const UpdateInfo::ConstPtr& info) const {
  LOG(INFO) << "[Archetypes] call() invoked, enable=" << config.enable;
  if (!config.enable) {
    return;
  }

  ScopedTimer timer(
      "backend/archetype_classification", info->timestamp_ns, true, 1, false);

  if (!dsg.graph->hasLayer(config.places_layer)) {
    return;
  }

  const auto& places = dsg.graph->getLayer(config.places_layer);

  // Step 1: Classify each place by dominant mode count
  std::map<NodeId, ArchetypeType> classifications;
  for (const auto& [node_id, node] : places.nodes()) {
    if (NodeSymbol(node_id).category() != 'p') {
      continue;  // skip frontier nodes
    }
    classifications[node_id] = classifyPlace(*node);
  }

  std::map<ArchetypeType, int> type_counts;
  std::map<std::string, int> static_reasons;
  for (const auto& [node_id, type] : classifications) {
    type_counts[type] += 1;
    if (type != ArchetypeType::STATIC) {
      continue;
    }
    const auto& meta = places.getNode(node_id).attributes().metadata();
    if (!meta.is_object() || !meta.contains("flow_components")) {
      static_reasons["no_metadata"] += 1;
    } else if (!meta.contains("flow_occupancy") ||
               presenceFromOccupancy(
                   meta["flow_occupancy"].get<double>(),
                   meta.contains("flow_dwell_s") ? meta["flow_dwell_s"].get<double>()
                                                 : -1.0,
                   config.presence_horizon_s) < config.min_probability) {
      static_reasons["low_probability"] += 1;
    } else if (!meta.contains("flow_support") ||
               meta["flow_support"].get<int>() < config.min_flow_support) {
      static_reasons["low_support"] += 1;
    } else {
      static_reasons["other"] += 1;
    }
  }
  LOG(INFO) << "[Archetypes] places with prefix 'p': " << classifications.size()
            << " (total in layer: " << places.numNodes() << ") types: STATIC="
            << type_counts[ArchetypeType::STATIC]
            << " UNIMODAL=" << type_counts[ArchetypeType::UNIMODAL]
            << " BIMODAL=" << type_counts[ArchetypeType::BIMODAL]
            << " MULTIMODAL=" << type_counts[ArchetypeType::MULTIMODAL]
            << " DIFFUSE=" << type_counts[ArchetypeType::DIFFUSE]
            << " | static_reasons: no_metadata=" << static_reasons["no_metadata"]
            << " low_probability=" << static_reasons["low_probability"]
            << " low_support=" << static_reasons["low_support"]
            << " other=" << static_reasons["other"]
            << " | thresholds: min_prob=" << config.min_probability
            << " min_support=" << config.min_flow_support
            << " dom_w=" << config.dominant_weight_threshold;

  // Step 2: Cluster spatially adjacent places with same archetype type
  auto clusters = clusterByArchetype(places, classifications);

  LOG(INFO) << "[Archetypes] clusters formed: " << clusters.size();

  // Step 3: Update the archetype layer
  rewriteArchetypes(*dsg.graph, clusters, classifications);

  if (dsg.graph->hasLayer(config.archetype_layer)) {
    LOG(INFO) << "[Archetypes] archetype nodes: "
              << clusters.size() << " clusters, "
              << dsg.graph->getLayer(config.archetype_layer).numNodes()
              << " nodes in layer";
  }
}

}  // namespace kairos
