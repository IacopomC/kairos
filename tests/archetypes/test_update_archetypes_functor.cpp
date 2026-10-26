#include <gtest/gtest.h>
#include <hydra/common/shared_dsg_info.h>
#include <kairos/archetypes/update_archetypes_functor.h>
#include <spark_dsg/node_symbol.h>

namespace kairos {

using spark_dsg::ArchetypeNodeAttributes;
using spark_dsg::ArchetypeType;
using spark_dsg::DsgLayers;
using spark_dsg::NodeSymbol;
using spark_dsg::PlaceNodeAttributes;

namespace {

// a place with flow metadata as the flow module writes it
void addPlace(spark_dsg::DynamicSceneGraph& graph,
              size_t index,
              double x,
              const std::vector<double>& weights) {
  auto attrs = std::make_unique<PlaceNodeAttributes>();
  attrs->position = Eigen::Vector3d(x, 0.0, 0.0);
  auto components = nlohmann::json::array();
  for (const auto weight : weights) {
    components.push_back({{"weight", weight}, {"theta", 0.5}, {"rho", 1.2}});
  }

  attrs->metadata.set({{"flow_num_components", weights.size()},
                       {"flow_occupancy", 1.0},
                       {"flow_dwell_s", 2.0},
                       {"flow_support", 5},
                       {"flow_confidence", 0.8},
                       {"flow_components", components}});
  graph.emplaceNode(DsgLayers::PLACES, NodeSymbol('p', index), std::move(attrs));
}

}  // namespace

TEST(UpdateArchetypesFunctor, ClustersPlacesBySharedArchetype) {
  hydra::SharedDsgInfo dsg(
      {{{DsgLayers::PLACES, 3}, {DsgLayers::ARCHETYPES, 4}, {DsgLayers::ROOMS, 5}}});
  auto& graph = *dsg.graph;

  // two connected one-way places, then a two-way place
  addPlace(graph, 0, 0.0, {0.9});
  addPlace(graph, 1, 1.0, {0.8, 0.1});
  addPlace(graph, 2, 2.0, {0.5, 0.5});
  graph.insertEdge(NodeSymbol('p', 0), NodeSymbol('p', 1));
  graph.insertEdge(NodeSymbol('p', 1), NodeSymbol('p', 2));

  UpdateArchetypesFunctor functor(UpdateArchetypesFunctor::Config{});
  auto info = std::make_shared<hydra::UpdateInfo>();
  functor.call(graph, dsg, info);

  const auto& archetypes = graph.getLayer(DsgLayers::ARCHETYPES);
  ASSERT_EQ(archetypes.numNodes(), 2u);

  const auto p0_parent = graph.getNode(NodeSymbol('p', 0)).getParent();
  const auto p2_parent = graph.getNode(NodeSymbol('p', 2)).getParent();
  ASSERT_TRUE(p0_parent && p2_parent);
  EXPECT_EQ(p0_parent, graph.getNode(NodeSymbol('p', 1)).getParent());
  EXPECT_NE(p0_parent, p2_parent);

  const auto& one_way = graph.getNode(*p0_parent).attributes<ArchetypeNodeAttributes>();
  EXPECT_EQ(one_way.archetype_type, ArchetypeType::UNIMODAL);
  EXPECT_EQ(one_way.member_count, 2u);
  const auto& two_way = graph.getNode(*p2_parent).attributes<ArchetypeNodeAttributes>();
  EXPECT_EQ(two_way.archetype_type, ArchetypeType::BIMODAL);
}

TEST(UpdateArchetypesFunctor, ClassifiesByDominantComponents) {
  EXPECT_EQ(classifyArchetype(0.01, 5, {0.9}, 0.2, 1, 0.05), ArchetypeType::STATIC);
  EXPECT_EQ(classifyArchetype(0.5, 0, {0.9}, 0.2, 1, 0.05), ArchetypeType::STATIC);
  EXPECT_EQ(classifyArchetype(0.5, 5, {0.1, 0.1}, 0.2, 1, 0.05), ArchetypeType::DIFFUSE);
  EXPECT_EQ(classifyArchetype(0.5, 5, {0.9}, 0.2, 1, 0.05), ArchetypeType::UNIMODAL);
  EXPECT_EQ(classifyArchetype(0.5, 5, {0.5, 0.5}, 0.2, 1, 0.05), ArchetypeType::BIMODAL);
  EXPECT_EQ(classifyArchetype(0.5, 5, {0.3, 0.3, 0.3}, 0.2, 1, 0.05),
            ArchetypeType::MULTIMODAL);
}

}  // namespace kairos
