#include <gtest/gtest.h>
#include <kairos/flow/flow_map.h>

#include <cmath>
#include <filesystem>
#include <thread>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

FlowMapConfig defaultConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.candidate_periods = {3600.0f};
  return cfg;
}

GlobalIndex makeIndex(int x, int y, int z) { return GlobalIndex(x, y, z); }

}  // namespace

TEST(FlowMap, EnsureCellsCreatesEntries) {
  FlowMap map(defaultConfig());
  GlobalIndexSet indices;
  indices.insert(makeIndex(0, 0, 0));
  indices.insert(makeIndex(1, 2, 3));

  map.ensureCells(indices);
  EXPECT_EQ(map.numCells(), 2u);
}

TEST(FlowMap, AddObservationCreatesCellOnDemand) {
  FlowMap map(defaultConfig());
  map.addObservation(makeIndex(5, 5, 5), 1.0, 1.0, 0.0, 1.0);
  EXPECT_EQ(map.numCells(), 1u);
}

TEST(FlowMap, PredictForUnknownCellReturnsFallback) {
  FlowMap map(defaultConfig());
  const auto pred = map.predict(makeIndex(0, 0, 0), 0.0, 1.0);
  EXPECT_EQ(pred.flag, CellFlag::FALLBACK);
}

TEST(FlowMap, AddAndPredict) {
  FlowMap map(defaultConfig());
  const auto idx = makeIndex(1, 1, 1);

  for (int i = 0; i < 10; ++i) {
    map.addObservation(idx, 0.5, 1.1, static_cast<double>(i), 1.0);
  }

  const auto pred = map.predict(idx, 10.0, 1.0);
  EXPECT_FALSE(pred.pi.empty());
  EXPECT_GE(pred.p_present, 0.0);
}

TEST(FlowMap, TryGetCellSnapshotNullForUnknown) {
  FlowMap map(defaultConfig());
  const auto snap = map.tryGetCellSnapshot(makeIndex(0, 0, 0), 0.0, 1.0);
  EXPECT_FALSE(snap.has_value());
}

TEST(FlowMap, RemapCellsMovesCells) {
  FlowMap map(defaultConfig());
  GlobalIndexSet indices;
  const auto old_idx = makeIndex(0, 0, 0);
  const auto new_idx = makeIndex(10, 10, 10);
  indices.insert(old_idx);
  map.ensureCells(indices);

  for (int i = 0; i < 5; ++i) {
    map.addObservation(old_idx, 1.0, 1.0, static_cast<double>(i), 1.0);
  }

  std::vector<std::pair<GlobalIndex, GlobalIndex>> remap = {{old_idx, new_idx}};
  map.remapCells(remap);

  const auto snap_old = map.tryGetCellSnapshot(old_idx, 5.0, 1.0);
  const auto snap_new = map.tryGetCellSnapshot(new_idx, 5.0, 1.0);
  EXPECT_FALSE(snap_old.has_value());
  EXPECT_TRUE(snap_new.has_value());
}

TEST(FlowMap, CellIndicesReturnsAll) {
  FlowMap map(defaultConfig());
  GlobalIndexSet indices;
  indices.insert(makeIndex(0, 0, 0));
  indices.insert(makeIndex(1, 1, 1));
  indices.insert(makeIndex(2, 2, 2));
  map.ensureCells(indices);

  const auto result = map.cellIndices();
  EXPECT_EQ(result.size(), 3u);
}

TEST(FlowMap, CountFlagsMatchesCellCount) {
  FlowMap map(defaultConfig());
  GlobalIndexSet indices;
  indices.insert(makeIndex(0, 0, 0));
  indices.insert(makeIndex(1, 1, 1));
  map.ensureCells(indices);

  const auto flags = map.countFlags(0.0, 1.0);
  size_t total = 0;
  for (const auto count : flags) {
    total += count;
  }
  EXPECT_EQ(total, 2u);
}

TEST(FlowMap, ConcurrentAccessDoesNotCrash) {
  FlowMap map(defaultConfig());
  GlobalIndexSet indices;
  const auto idx = makeIndex(0, 0, 0);
  indices.insert(idx);
  map.ensureCells(indices);

  auto writer = [&]() {
    for (int i = 0; i < 100; ++i) {
      map.addObservation(idx, 0.5, 1.0, static_cast<double>(i), 0.1);
    }
  };

  auto reader = [&]() {
    for (int i = 0; i < 100; ++i) {
      map.predict(idx, static_cast<double>(i), 1.0);
    }
  };

  std::thread t1(writer);
  std::thread t2(reader);
  t1.join();
  t2.join();
}

TEST(FlowMap, CheckpointRoundTripFixedMode) {
  auto cfg = defaultConfig();

  FlowMap map(cfg);
  const auto idx_a = makeIndex(1, 2, 3);
  const auto idx_b = makeIndex(-2, 0, 4);
  for (int i = 0; i < 30; ++i) {
    map.addObservation(idx_a, 0.25 + 0.01 * i, 1.1, static_cast<double>(i), 0.5);
    map.addObservation(idx_b, -0.5 + 0.02 * i, 0.9, static_cast<double>(i), 0.5);
  }

  const auto temp_file =
      std::filesystem::temp_directory_path() / "hydra_flow_map_fixed_checkpoint_test.json";
  ASSERT_TRUE(map.saveToFile(temp_file.string()));

  FlowMap restored(cfg);
  std::string error;
  ASSERT_TRUE(restored.loadFromFile(temp_file.string(), &error)) << error;
  EXPECT_EQ(restored.numCells(), map.numCells());

  const auto original = map.tryGetCellSnapshot(idx_a, 31.0, 1.0);
  const auto reloaded = restored.tryGetCellSnapshot(idx_a, 31.0, 1.0);
  ASSERT_TRUE(original.has_value());
  ASSERT_TRUE(reloaded.has_value());
  EXPECT_EQ(original->components.size(), reloaded->components.size());
  EXPECT_NEAR(original->p_present, reloaded->p_present, 1.0e-9);
  EXPECT_EQ(original->flag, reloaded->flag);

  std::filesystem::remove(temp_file);
}

TEST(FlowMap, PerPassageOneVotePerVisit) {
  auto cfg = defaultConfig();
  cfg.visit_gap_s = 2.0;
  cfg.visit_max_s = 10.0;
  cfg.nudft_min_obs = 2;
  FlowMap map(cfg);
  const auto idx = makeIndex(3, 3, 3);

  // One track crossing the cell: 10 frames at 10 Hz -> ONE weights sample,
  // and only once the visit closes.
  for (int i = 0; i < 10; ++i) {
    map.addObservation(idx, 0.4, 1.2, static_cast<double>(i) * 0.1, 0.1, /*track_id=*/7);
  }
  EXPECT_EQ(map.predict(idx, 1.0, 1.0).flag, CellFlag::COLD_START);  // visit open
  map.flushVisits();
  EXPECT_EQ(map.predict(idx, 1.0, 1.0).flag, CellFlag::COLD_START);  // 1 < min_obs

  // A second track's crossing supplies the second sample -> model-based.
  for (int i = 0; i < 5; ++i) {
    map.addObservation(idx, 0.5, 1.0, 5.0 + i * 0.1, 0.1, /*track_id=*/8);
  }
  map.flushVisits();
  EXPECT_EQ(map.predict(idx, 6.0, 1.0).flag, CellFlag::MODEL_BASED);
}

TEST(FlowMap, PerPassageVisitClosesOnGapAndCellChange) {
  auto cfg = defaultConfig();
  cfg.visit_gap_s = 1.0;
  cfg.nudft_min_obs = 2;
  FlowMap map(cfg);
  const auto a = makeIndex(0, 0, 0);
  const auto b = makeIndex(1, 0, 0);

  map.addObservation(a, 0.4, 1.2, 0.0, 0.1, 7);
  map.addObservation(a, 0.4, 1.2, 0.1, 0.1, 7);
  // Same track reappears past the gap: the first visit closes (sample #1
  // for cell a), a new one opens.
  map.addObservation(a, 0.4, 1.2, 5.0, 0.1, 7);
  // Track moves to cell b: the second visit closes (sample #2 for cell a).
  map.addObservation(b, 0.4, 1.2, 5.1, 0.1, 7);
  EXPECT_EQ(map.predict(a, 6.0, 1.0).flag, CellFlag::MODEL_BASED);  // 2 samples
  EXPECT_EQ(map.predict(b, 6.0, 1.0).flag, CellFlag::COLD_START);   // visit open
}

TEST(FlowMap, PerPassageAnonymousFallsBackPerDetection) {
  auto cfg = defaultConfig();
  cfg.nudft_min_obs = 2;
  FlowMap map(cfg);
  const auto idx = makeIndex(2, 2, 2);
  map.addObservation(idx, 0.4, 1.2, 0.0, 0.1);  // no id
  map.addObservation(idx, 0.4, 1.2, 0.1, 0.1);  // no id
  EXPECT_EQ(map.predict(idx, 1.0, 1.0).flag, CellFlag::MODEL_BASED);
}

}  // namespace kairos
