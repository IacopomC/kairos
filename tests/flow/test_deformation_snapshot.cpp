/**
 * @file test_deformation_snapshot.cpp
 * @brief Tests for FlowTemporalModule::applyDeformationSnapshot: a
 * caller-supplied stamped control-point warp must re-key flow cells exactly
 * as the backend-driven alignment would, including per-epoch corrections
 * when cells created at different times carry different drifts.
 */
#include <gtest/gtest.h>
#include <kairos/temporal/flow_temporal_module.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>
#include <vector>

namespace kairos {
using namespace hydra;

namespace {

constexpr float kVoxel = 0.4f;

FlowTemporalModule::Config moduleConfig() {
  FlowTemporalModule::Config cfg;
  cfg.publish_metrics = false;
  cfg.publish_voxel_markers = false;
  cfg.flow.fixed_mu_rho = 1.1;
  cfg.flow.fixed_sigma_theta = 0.4;
  cfg.flow.fixed_sigma_rho = 0.3;
  cfg.flow.nudft_min_obs = 5;
  cfg.flow.candidate_periods = {3600.0f};
  return cfg;
}

/// World-frame center of the voxel holding (x, y, 0).
Eigen::Vector3d cellCenter(double x, double y) {
  const auto ix = static_cast<int>(std::floor(x / kVoxel));
  const auto iy = static_cast<int>(std::floor(y / kVoxel));
  return {(ix + 0.5) * kVoxel, (iy + 0.5) * kVoxel, 0.5 * kVoxel};
}

/// Feed one detection batch at `positions` (time t_s), all cells visible.
void observeAt(FlowTemporalModule& mod,
               double t_s,
               const std::vector<Eigen::Vector3d>& positions) {
  std::vector<double> theta(positions.size(), 0.0);
  std::vector<double> rho(positions.size(), 1.0);
  ASSERT_TRUE(mod.pushDetections(t_s, positions, theta, rho));
  mod.flush();
}

std::set<std::tuple<int, int, int>> cellKeys(const FlowTemporalModule& mod) {
  std::set<std::tuple<int, int, int>> keys;
  for (const auto& c : mod.flowCells()) {
    keys.insert({static_cast<int>(c.index.x()), static_cast<int>(c.index.y()),
                 static_cast<int>(c.index.z())});
  }
  return keys;
}

std::tuple<int, int, int> keyOf(const Eigen::Vector3d& p) {
  return {static_cast<int>(std::floor(p.x() / kVoxel)),
          static_cast<int>(std::floor(p.y() / kVoxel)),
          static_cast<int>(std::floor(p.z() / kVoxel))};
}

/// Trajectory control points around `center` at time `t_s`, displaced by
/// `drift` on the before side and corrected (no displacement) after.
void addControlRing(std::vector<Eigen::Vector3d>& before,
                    std::vector<double>& stamps,
                    std::vector<Eigen::Vector3d>& after,
                    const Eigen::Vector3d& center,
                    double t_s,
                    const Eigen::Vector3d& drift) {
  for (int i = 0; i < 6; ++i) {
    const double a = i * M_PI / 3.0;
    const Eigen::Vector3d truth =
        center + Eigen::Vector3d(2.0 * std::cos(a), 2.0 * std::sin(a), 0.0);
    before.push_back(truth + drift);
    stamps.push_back(t_s + 0.1 * i);
    after.push_back(truth);
  }
}

}  // namespace

TEST(DeformationSnapshot, UniformDriftReKeysAllCells) {
  FlowTemporalModule mod(moduleConfig());
  mod.start();

  // The world as believed: every registered position carries drift d.
  const Eigen::Vector3d d(0.8, 0.0, 0.0);  // two voxels in +x
  std::vector<Eigen::Vector3d> truth = {cellCenter(1.0, 1.0), cellCenter(3.0, 1.0),
                                        cellCenter(1.0, 3.0), cellCenter(3.0, 3.0)};
  std::vector<Eigen::Vector3d> drifted;
  for (const auto& p : truth) drifted.push_back(p + d);

  std::vector<Eigen::Vector3d> visible = drifted;
  mod.setVisibleVoxels(kVoxel, visible);
  observeAt(mod, 1000.0, drifted);
  ASSERT_EQ(cellKeys(mod).size(), truth.size());
  for (const auto& p : drifted) {
    EXPECT_TRUE(cellKeys(mod).count(keyOf(p)));
  }

  std::vector<Eigen::Vector3d> before, after;
  std::vector<double> stamps;
  addControlRing(before, stamps, after, cellCenter(2.0, 2.0) + d, 1000.0, d);
  mod.applyDeformationSnapshot(1010.0, before, stamps, after);

  const auto keys = cellKeys(mod);
  EXPECT_EQ(keys.size(), truth.size());
  for (const auto& p : truth) {
    EXPECT_TRUE(keys.count(keyOf(p))) << "cell not re-keyed to truth";
  }
  for (const auto& p : drifted) {
    EXPECT_FALSE(keys.count(keyOf(p))) << "drifted key survived alignment";
  }
  const auto events = mod.lcEventsCopy();
  ASSERT_EQ(events.size(), 1u);
  // LcEvent carries the last detection timestamp, not the snapshot time.
  EXPECT_NEAR(events[0].t_seconds, 1000.0, 1e-6);
  EXPECT_GT(events[0].n_remap_pairs, 0u);
}

TEST(DeformationSnapshot, PerEpochDriftCorrectedByStampedControlPoints) {
  auto cfg = moduleConfig();
  cfg.deformation_tolerance_s = 10.0;  // window control points per epoch
  FlowTemporalModule mod(cfg);
  mod.start();

  // Epoch A (t=1000): cells near the origin drifted by dA. Epoch B (t=2000):
  // cells 30 m away drifted by dB. A full-history snapshot must correct each
  // epoch's cells by its own drift.
  const Eigen::Vector3d dA(0.8, 0.0, 0.0);
  const Eigen::Vector3d dB(0.0, -1.2, 0.0);
  std::vector<Eigen::Vector3d> truth_a = {cellCenter(1.0, 1.0), cellCenter(2.0, 1.4)};
  std::vector<Eigen::Vector3d> truth_b = {cellCenter(30.0, 1.0), cellCenter(31.0, 1.4)};
  std::vector<Eigen::Vector3d> drifted_a, drifted_b;
  for (const auto& p : truth_a) drifted_a.push_back(p + dA);
  for (const auto& p : truth_b) drifted_b.push_back(p + dB);

  std::vector<Eigen::Vector3d> visible = drifted_a;
  visible.insert(visible.end(), drifted_b.begin(), drifted_b.end());
  mod.setVisibleVoxels(kVoxel, visible);
  observeAt(mod, 1000.0, drifted_a);
  observeAt(mod, 2000.0, drifted_b);
  ASSERT_EQ(cellKeys(mod).size(), truth_a.size() + truth_b.size());

  std::vector<Eigen::Vector3d> before, after;
  std::vector<double> stamps;
  addControlRing(before, stamps, after, cellCenter(1.5, 1.2) + dA, 1000.0, dA);
  addControlRing(before, stamps, after, cellCenter(30.5, 1.2) + dB, 2000.0, dB);
  mod.applyDeformationSnapshot(2010.0, before, stamps, after);

  const auto keys = cellKeys(mod);
  for (const auto& p : truth_a) {
    EXPECT_TRUE(keys.count(keyOf(p))) << "epoch-A cell not corrected by dA";
  }
  for (const auto& p : truth_b) {
    EXPECT_TRUE(keys.count(keyOf(p))) << "epoch-B cell not corrected by dB";
  }
  for (const auto& p : drifted_a) {
    EXPECT_FALSE(keys.count(keyOf(p)));
  }
  for (const auto& p : drifted_b) {
    EXPECT_FALSE(keys.count(keyOf(p)));
  }
}

TEST(DeformationSnapshot, UnsortedStampsAreHandled) {
  FlowTemporalModule mod(moduleConfig());
  mod.start();

  const Eigen::Vector3d d(0.8, 0.0, 0.0);
  const auto truth = cellCenter(1.0, 1.0);
  const Eigen::Vector3d drifted = truth + d;
  mod.setVisibleVoxels(kVoxel, {drifted});
  observeAt(mod, 1000.0, {drifted});

  std::vector<Eigen::Vector3d> before, after;
  std::vector<double> stamps;
  addControlRing(before, stamps, after, cellCenter(1.0, 1.0) + d, 1000.0, d);
  // Shuffle so the caller-order invariant is exercised.
  std::reverse(before.begin(), before.end());
  std::reverse(stamps.begin(), stamps.end());
  std::reverse(after.begin(), after.end());
  mod.applyDeformationSnapshot(1010.0, before, stamps, after);

  EXPECT_TRUE(cellKeys(mod).count(keyOf(truth)));
  EXPECT_FALSE(cellKeys(mod).count(keyOf(drifted)));
}

}  // namespace kairos
