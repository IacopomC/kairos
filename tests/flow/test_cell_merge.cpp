#include <gtest/gtest.h>
#include <kairos/flow/cell_state.h>
#include <kairos/flow/flow_map.h>
#include <kairos/flow/nudft.h>
#include <kairos/flow/poisson_model.h>

#include <cmath>
#include <random>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

FlowMapConfig fixedConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.candidate_periods = {3600.0f};
  return cfg;
}

GlobalIndex makeIndex(int x, int y, int z) { return GlobalIndex(x, y, z); }

NUDFTModel makeNudft() {
  NUDFTModel m;
  m.omegas_ptr = std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / 3600.0, 2.0 * M_PI / 86400.0});
  m.gammas.assign(m.omegas_ptr->size(), {0.0, 0.0});
  m.min_observations = 5;
  return m;
}

NUDFTModel makeNudftSharing(const NUDFTModel& other) {
  NUDFTModel m;
  m.omegas_ptr = other.omegas_ptr;
  m.gammas.assign(m.omegas_ptr->size(), {0.0, 0.0});
  m.min_observations = other.min_observations;
  return m;
}

}  // namespace

TEST(CellMerge, NudftGamma0ExactMean) {
  auto a = makeNudft();
  auto b = makeNudftSharing(a);
  auto reference = makeNudftSharing(a);

  std::vector<std::pair<double, double>> stream_a, stream_b;
  for (int i = 0; i < 30; ++i) {
    stream_a.push_back({static_cast<double>(i), 0.5 + 0.01 * i});
  }
  for (int i = 0; i < 17; ++i) {
    stream_b.push_back({100.0 + static_cast<double>(i), 0.7 - 0.005 * i});
  }
  for (const auto& [t, v] : stream_a) {
    a.update(t, v);
    reference.update(t, v);
  }
  for (const auto& [t, v] : stream_b) {
    b.update(t, v);
    reference.update(t, v);
  }

  ASSERT_TRUE(a.mergeFrom(b));
  EXPECT_EQ(a.n, reference.n);
  EXPECT_NEAR(a.gamma0, reference.gamma0, 1.0e-12);
}

TEST(CellMerge, NudftGammaFBoundedBias) {
  auto a = makeNudft();
  auto b = makeNudftSharing(a);
  auto reference = makeNudftSharing(a);

  std::mt19937 rng(7u);
  std::uniform_real_distribution<double> v_dist(0.0, 1.0);
  for (int i = 0; i < 40; ++i) {
    const double t = static_cast<double>(i);
    const double v = v_dist(rng);
    a.update(t, v);
    reference.update(t, v);
  }
  const double gamma0_a_final = a.gamma0;
  for (int i = 0; i < 25; ++i) {
    const double t = static_cast<double>(40 + i);
    const double v = v_dist(rng);
    b.update(t, v);
    reference.update(t, v);
  }
  const double gamma0_b_final = b.gamma0;
  const int n_a = a.n;
  const int n_b = b.n;

  ASSERT_TRUE(a.mergeFrom(b));
  // Bound: |γ_f^merged − γ_f^ref| ≤ |Δγ₀| · n_b / (n_a + n_b)
  const double bound =
      std::abs(gamma0_a_final - gamma0_b_final) * static_cast<double>(n_b) /
      static_cast<double>(n_a + n_b);
  for (size_t f = 0; f < a.gammas.size(); ++f) {
    const double err = std::abs(a.gammas[f] - reference.gammas[f]);
    EXPECT_LE(err, bound + 1.0e-9) << "frequency index " << f;
  }
}

// mergeFrom gates on frequency-content equality, not pointer identity. A merge
// across DIFFERENT frequency content must be refused and leave this model intact.
TEST(CellMerge, NudftRefuseMergeOnOmegasContentMismatch) {
  auto a = makeNudft();  // periods 3600 s, 86400 s
  NUDFTModel b;
  b.omegas_ptr = std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / 1800.0, 2.0 * M_PI / 43200.0});  // different content
  b.gammas.assign(b.omegas_ptr->size(), {0.0, 0.0});
  b.min_observations = 5;

  for (int i = 0; i < 5; ++i) {
    a.update(static_cast<double>(i), 0.5);
    b.update(static_cast<double>(i), 0.7);
  }
  const double a_g0_pre = a.gamma0;
  const int a_n_pre = a.n;
  EXPECT_FALSE(a.mergeFrom(b));
  EXPECT_EQ(a.n, a_n_pre);
  EXPECT_DOUBLE_EQ(a.gamma0, a_g0_pre);
}

// Conversely, separate-but-identical frequency vectors (e.g. after a reload)
// must merge: the gate is on content, so the counts pool.
TEST(CellMerge, NudftMergeSucceedsOnIdenticalContentDistinctPointers) {
  auto a = makeNudft();
  NUDFTModel b;
  b.omegas_ptr = std::make_shared<const std::vector<double>>(
      std::vector<double>{2.0 * M_PI / 3600.0, 2.0 * M_PI / 86400.0});  // same content, new alloc
  b.gammas.assign(b.omegas_ptr->size(), {0.0, 0.0});
  b.min_observations = 5;

  for (int i = 0; i < 5; ++i) {
    a.update(static_cast<double>(i), 0.5);
    b.update(static_cast<double>(i), 0.7);
  }
  const int n_total = a.n + b.n;
  EXPECT_TRUE(a.mergeFrom(b));
  EXPECT_EQ(a.n, n_total);
}

TEST(CellMerge, PoissonConjugateExact) {
  PoissonModel a;
  PoissonModel b;
  PoissonModel reference;
  a.nudft = makeNudft();
  b.nudft = makeNudftSharing(a.nudft);
  reference.nudft = makeNudftSharing(a.nudft);

  for (int i = 0; i < 12; ++i) {
    a.update(static_cast<double>(i), 0.4);
    reference.update(static_cast<double>(i), 0.4);
  }
  for (int i = 0; i < 9; ++i) {
    const double t = static_cast<double>(12 + i);
    b.update(t, 0.6);
    reference.update(t, 0.6);
  }

  a.mergeFrom(b);
  // α and β are exactly additive in (n_obs, Δt_vis) under the conjugate update.
  EXPECT_DOUBLE_EQ(a.alpha, reference.alpha);
  EXPECT_DOUBLE_EQ(a.beta, reference.beta);
}

TEST(CellMerge, PoissonPriorOnlyMergeIsPrior) {
  PoissonModel a;
  PoissonModel b;
  a.nudft = makeNudft();
  b.nudft = makeNudftSharing(a.nudft);
  a.mergeFrom(b);
  EXPECT_DOUBLE_EQ(a.alpha, PoissonModel::kPriorAlpha);
  EXPECT_DOUBLE_EQ(a.beta, PoissonModel::kPriorBeta);
}

TEST(CellMerge, FlowMapTotalObservationsConserved) {
  FlowMap map(fixedConfig());
  const auto idx_a = makeIndex(0, 0, 0);
  const auto idx_b = makeIndex(1, 0, 0);
  const auto target = makeIndex(5, 5, 5);

  for (int i = 0; i < 12; ++i) {
    map.addObservation(idx_a, 0.5, 1.0, static_cast<double>(i), 0.5);
  }
  for (int i = 0; i < 7; ++i) {
    map.addObservation(idx_b, -0.5, 1.0, static_cast<double>(i), 0.5);
  }

  std::vector<std::pair<GlobalIndex, GlobalIndex>> remap = {
      {idx_a, target}, {idx_b, target}};
  map.remapCells(remap);

  EXPECT_EQ(map.totalObservationsPreRemap(), 19);
  EXPECT_EQ(map.totalObservationsPostRemap(), 19);
}

TEST(CellMerge, FlowMapEmitsMergeEvent) {
  FlowMap map(fixedConfig());
  const auto idx_a = makeIndex(0, 0, 0);
  const auto idx_b = makeIndex(1, 0, 0);
  const auto target = makeIndex(5, 5, 5);

  for (int i = 0; i < 8; ++i) {
    map.addObservation(idx_a, 0.5, 1.0, static_cast<double>(i), 0.5);
    map.addObservation(idx_b, -0.5, 1.0, static_cast<double>(i), 0.5);
  }

  map.remapCells({{idx_a, target}, {idx_b, target}});
  const auto events = map.drainMergeEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events.front().target, target);
  EXPECT_GT(events.front().n_a, 0);
  EXPECT_GT(events.front().n_b, 0);
  EXPECT_EQ(map.drainMergeEvents().size(), 0u);
}

TEST(CellMerge, FlowMapNoCollisionPreservesBehavior) {
  FlowMap map(fixedConfig());
  const auto idx_a = makeIndex(0, 0, 0);
  const auto idx_b = makeIndex(1, 0, 0);
  for (int i = 0; i < 6; ++i) {
    map.addObservation(idx_a, 0.5, 1.0, static_cast<double>(i), 0.5);
    map.addObservation(idx_b, 1.0, 1.0, static_cast<double>(i), 0.5);
  }
  map.remapCells({{idx_a, makeIndex(2, 0, 0)}, {idx_b, makeIndex(3, 0, 0)}});
  EXPECT_EQ(map.numCells(), 2u);
  EXPECT_TRUE(map.drainMergeEvents().empty());
}

TEST(CellMerge, FlowMapEmptyCellMergedIntoPopulated) {
  FlowMap map(fixedConfig());
  const auto idx_a = makeIndex(0, 0, 0);
  const auto idx_b = makeIndex(1, 0, 0);
  const auto target = makeIndex(5, 5, 5);

  GlobalIndexSet indices;
  indices.insert(idx_b);
  map.ensureCells(indices);
  for (int i = 0; i < 8; ++i) {
    map.addObservation(idx_a, 0.5, 1.0, static_cast<double>(i), 0.5);
  }

  map.remapCells({{idx_a, target}, {idx_b, target}});
  EXPECT_EQ(map.totalObservationsPreRemap(), 8);
  EXPECT_EQ(map.totalObservationsPostRemap(), 8);
}

}  // namespace kairos
