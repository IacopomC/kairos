/* -----------------------------------------------------------------------------
 * Copyright 2024 Iacopo Catalano.
 * All Rights Reserved
 * -------------------------------------------------------------------------- */
#pragma once
#include <config_utilities/factory.h>
#include <hydra/backend/update_functions.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/scene_graph_types.h>

#include <map>
#include <set>
#include <string>
#include <vector>

/**
 * @file update_archetypes_functor.h
 * @brief Update functor that builds the archetype layer. Each place is
 *        classified into a motion archetype from its per-place flow-component
 *        metadata, places sharing an archetype are clustered, and one archetype
 *        node is written per cluster.
 */

namespace kairos {

/// @brief Classifies one place's flow evidence into a motion archetype.
///
/// Single source of truth for the archetype rule, shared by the backend functor
/// and any other consumer (e.g. the python bindings' offline/ATC path) so the
/// classification can never drift into a re-implementation. Below the
/// probability/support gates the place is STATIC; otherwise the count of
/// components whose mixture weight reaches @p dominant_weight_threshold maps to
/// UNIMODAL/BIMODAL/MULTIMODAL, and evidence with no dominant component is
/// DIFFUSE.
spark_dsg::ArchetypeType classifyArchetype(double probability,
                                           int support,
                                           const std::vector<double>& weights,
                                           double dominant_weight_threshold,
                                           int min_flow_support,
                                           double min_probability);

/// @brief Classifies places into motion archetypes and writes archetype nodes.
struct UpdateArchetypesFunctor : public hydra::UpdateFunctor {
  struct Config {
    bool enable = true;                    ///< Whether the functor runs at all.
    double dominant_weight_threshold = 0.20;  ///< Minimum mixture weight for a flow component to count as dominant.
    int min_flow_support = 1;              ///< Minimum flow support before a place can be non-static.
    double min_probability = 0.05;         ///< Minimum flow probability before a place can be non-static.
    /// Horizon, in seconds, at which a place's stored occupancy is turned into
    /// the presence probability the STATIC gate compares against
    /// min_probability. Places store an occupancy rather than a probability, so
    /// the horizon that fixes the meaning of the gate is named here instead of
    /// being inherited from whatever horizon last wrote the metadata. Every
    /// reader of the gate, this functor and the python bindings' offline path
    /// alike, takes the horizon from the dataset config, so one classification
    /// horizon governs the layer. Zero asks whether the place is occupied now.
    double presence_horizon_s = 0.0;
    //! Layer of the places to classify
    std::string places_layer = spark_dsg::DsgLayers::PLACES;
    //! Layer the archetype nodes are written to
    std::string archetype_layer = spark_dsg::DsgLayers::ARCHETYPES;
    //! Node symbol prefix of the archetype nodes
    char prefix = 'A';
  } const config;

  /// @brief Constructs the functor from a validated configuration.
  explicit UpdateArchetypesFunctor(const Config& config);

  /// @brief Classifies and clusters places, then rewrites the archetype layer.
  /// @param unmerged Graph holding the original, unmerged nodes.
  /// @param dsg Shared graph whose archetype layer is rewritten.
  /// @param info Per-iteration context.
  void call(const spark_dsg::DynamicSceneGraph& unmerged,
            hydra::SharedDsgInfo& dsg,
            const hydra::UpdateInfo::ConstPtr& info) const override;

 private:
  using ArchetypeType = spark_dsg::ArchetypeType;

  /// @brief Classifies a single place into a motion archetype from its flow metadata.
  ArchetypeType classifyPlace(const spark_dsg::SceneGraphNode& node) const;

  /// @brief Groups connected places that share an archetype into clusters.
  std::map<size_t, std::set<spark_dsg::NodeId>> clusterByArchetype(
      const spark_dsg::SceneGraphLayer& places,
      const std::map<spark_dsg::NodeId, ArchetypeType>& classifications) const;

  /// @brief Replaces the archetype layer with one node per cluster.
  void rewriteArchetypes(
      spark_dsg::DynamicSceneGraph& graph,
      const std::map<size_t, std::set<spark_dsg::NodeId>>& clusters,
      const std::map<spark_dsg::NodeId, ArchetypeType>& classifications) const;

};

/// @brief Declares the configuration fields for UpdateArchetypesFunctor.
void declare_config(UpdateArchetypesFunctor::Config& config);

}  // namespace kairos
