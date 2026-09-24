/**
 * @file test_heading_rotation.cpp
 * @brief Tests for the rotational half of a loop-closure correction:
 * FlowCellState::rotateHeading re-expresses a cell's directional state in the
 * corrected frame, and a control-point warp carrying yaw applies it to every
 * cell it moves.
 */
#include <gtest/gtest.h>
#include <kairos/flow/cell_state.h>
#include <kairos/flow/swgmm.h>
#include <kairos/temporal/flow_temporal_module.h>

#include <cmath>
#include <memory>
#include <vector>

namespace kairos {
using namespace hydra;

namespace {

constexpr int kSlots = 8;
constexpr double kSlotWidth = 2.0 * M_PI / kSlots;
constexpr float kVoxel = 0.4f;

FlowMapConfig fixedConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.candidate_periods = {3600.0f};
  return cfg;
}

std::shared_ptr<const std::vector<double>> testOmegas(const FlowMapConfig& cfg) {
  std::vector<double> omegas;
  for (const auto period : cfg.candidate_periods) {
    if (period > 0.0f) {
      omegas.push_back(2.0 * M_PI / static_cast<double>(period));
    }
  }
  return std::make_shared<const std::vector<double>>(std::move(omegas));
}

/// One detection without a track id, fed the way FlowMap feeds it: the cell
/// update, then one weights sample from the detection's responsibilities.
void observe(FlowCellState& cell, double theta, double rho, double t_seconds,
             const FlowMapConfig& cfg) {
  std::vector<double> r;
  cell.update(theta, rho, t_seconds, 1.0, cfg, &r);
  if (!r.empty()) {
    cell.feedWeightsSample(t_seconds, r, cfg);
  }
}

/// A cell fed the same crossings at every heading offset by @p heading_offset,
/// so two cells built with offsets a whole slot apart hold the same state up
/// to a cyclic shift.
FlowCellState cellObserving(const FlowMapConfig& cfg, double heading_offset) {
  FlowCellState cell(std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));
  // A directional profile with structure, so a rotation is detectable: most
  // crossings along one heading, a few along its neighbour, none elsewhere.
  const std::vector<double> headings = {0.0, 0.0, 0.0, 0.0, kSlotWidth,
                                        kSlotWidth, 2.0 * kSlotWidth};
  for (size_t i = 0; i < headings.size(); ++i) {
    observe(cell, headings[i] + heading_offset, 1.0 + 0.1 * static_cast<double>(i),
            100.0 * static_cast<double>(i), cfg);
  }
  return cell;
}

std::vector<double> slotMeans(const FlowCellState& cell) {
  std::vector<double> out;
  out.reserve(cell.nudft_models.size());
  for (const auto& model : cell.nudft_models) {
    out.push_back(model.gamma0);
  }
  return out;
}

Eigen::Vector3d cellCenter(double x, double y) {
  const auto ix = static_cast<int>(std::floor(x / kVoxel));
  const auto iy = static_cast<int>(std::floor(y / kVoxel));
  return {(ix + 0.5) * kVoxel, (iy + 0.5) * kVoxel, 0.5 * kVoxel};
}

}  // namespace

TEST(HeadingRotation, WholeSlotMatchesObservingTheRotatedHeadings) {
  const auto cfg = fixedConfig();
  auto rotated_state = cellObserving(cfg, 0.0);
  const auto observed_rotated = cellObserving(cfg, 2.0 * kSlotWidth);

  rotated_state.rotateHeading(2.0 * kSlotWidth);

  const auto from_rotation = slotMeans(rotated_state);
  const auto from_observation = slotMeans(observed_rotated);
  ASSERT_EQ(from_rotation.size(), from_observation.size());
  for (size_t k = 0; k < from_rotation.size(); ++k) {
    // A whole-slot rotation is a permutation of the per-slot predictors, so it
    // reproduces the state built from rotated observations exactly.
    EXPECT_NEAR(from_rotation[k], from_observation[k], 1.0e-12) << "slot " << k;
  }
}

TEST(HeadingRotation, WholeSlotAlsoPermutesTheHarmonicsAndSpeeds) {
  const auto cfg = fixedConfig();
  auto rotated_state = cellObserving(cfg, 0.0);
  const auto before = cellObserving(cfg, 0.0);

  rotated_state.rotateHeading(kSlotWidth);

  for (int k = 0; k < kSlots; ++k) {
    const auto source = static_cast<size_t>((k - 1 + kSlots) % kSlots);
    const auto& a = rotated_state.nudft_models[static_cast<size_t>(k)];
    const auto& b = before.nudft_models[source];
    ASSERT_EQ(a.gammas.size(), b.gammas.size());
    for (size_t f = 0; f < a.gammas.size(); ++f) {
      EXPECT_NEAR(std::abs(a.gammas[f] - b.gammas[f]), 0.0, 1.0e-12);
    }
    EXPECT_NEAR(a.m2, b.m2, 1.0e-12);
    EXPECT_NEAR(rotated_state.nudft_mu_rho_models[static_cast<size_t>(k)].gamma0,
                before.nudft_mu_rho_models[source].gamma0, 1.0e-12);
  }
}

TEST(HeadingRotation, FractionalRotationInterpolatesAdjacentSlots) {
  const auto cfg = fixedConfig();
  auto rotated_state = cellObserving(cfg, 0.0);
  const auto before = slotMeans(cellObserving(cfg, 0.0));

  // Half a slot: each rotated slot is the midpoint of the two it straddles.
  rotated_state.rotateHeading(0.5 * kSlotWidth);

  const auto after = slotMeans(rotated_state);
  for (int k = 0; k < kSlots; ++k) {
    const auto lo = static_cast<size_t>((k - 1 + kSlots) % kSlots);
    const auto expected = 0.5 * (before[lo] + before[static_cast<size_t>(k)]);
    EXPECT_NEAR(after[static_cast<size_t>(k)], expected, 1.0e-12) << "slot " << k;
  }
}

TEST(HeadingRotation, RotationConservesTotalWeight) {
  const auto cfg = fixedConfig();
  auto cell = cellObserving(cfg, 0.0);
  double before_sum = 0.0;
  for (const auto value : slotMeans(cell)) before_sum += value;

  cell.rotateHeading(0.3);  // not a whole number of slots

  double after_sum = 0.0;
  for (const auto value : slotMeans(cell)) after_sum += value;
  EXPECT_NEAR(after_sum, before_sum, 1.0e-12);
}

TEST(HeadingRotation, FullTurnAndZeroAreNoOps) {
  const auto cfg = fixedConfig();
  const auto reference = slotMeans(cellObserving(cfg, 0.0));

  auto zero = cellObserving(cfg, 0.0);
  zero.rotateHeading(0.0);
  auto full = cellObserving(cfg, 0.0);
  full.rotateHeading(2.0 * M_PI);

  for (int k = 0; k < kSlots; ++k) {
    const auto slot = static_cast<size_t>(k);
    EXPECT_NEAR(slotMeans(zero)[slot], reference[slot], 1.0e-12);
    EXPECT_NEAR(slotMeans(full)[slot], reference[slot], 1.0e-12);
  }
}

TEST(HeadingRotation, NegativeRotationIsTheInverseOfPositive) {
  const auto cfg = fixedConfig();
  const auto reference = slotMeans(cellObserving(cfg, 0.0));
  auto cell = cellObserving(cfg, 0.0);

  // Whole slots, where the resample is a permutation and inverts exactly.
  cell.rotateHeading(3.0 * kSlotWidth);
  cell.rotateHeading(-3.0 * kSlotWidth);

  const auto after = slotMeans(cell);
  for (int k = 0; k < kSlots; ++k) {
    EXPECT_NEAR(after[static_cast<size_t>(k)], reference[static_cast<size_t>(k)],
                1.0e-12)
        << "slot " << k;
  }
}

namespace {

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

/// Index of the slot carrying the most predicted weight.
int dominantSlot(const FlowPrediction& prediction) {
  int best = 0;
  for (size_t k = 1; k < prediction.pi.size(); ++k) {
    if (prediction.pi[k] > prediction.pi[static_cast<size_t>(best)]) {
      best = static_cast<int>(k);
    }
  }
  return best;
}

}  // namespace

TEST(HeadingRotation, YawCarryingSnapshotRotatesTheStoredHeadings) {
  FlowTemporalModule mod(moduleConfig());
  mod.start();

  // One cell, all its crossings heading along +x (slot 0). The platform's yaw
  // estimate is off by +90 degrees, so the correction turns the frame back.
  const auto position = cellCenter(1.0, 1.0);
  mod.setVisibleVoxels(kVoxel, {position});
  for (int i = 0; i < 10; ++i) {
    const std::vector<Eigen::Vector3d> batch = {position};
    ASSERT_TRUE(mod.pushDetections(100.0 * i, batch, {0.0}, {1.0}));
    mod.flush();
  }

  const auto index = mod.flowMap().cellObservationCounts().front().index;
  ASSERT_EQ(dominantSlot(mod.flowMap().predict(index, 1000.0, 1.0)), 0);

  // Control points with no translation and a quarter-turn of yaw: the cells
  // stay where they are and only the directional state turns.
  std::vector<Eigen::Vector3d> before, after;
  std::vector<double> stamps, yaw;
  for (int i = 0; i < 6; ++i) {
    const double a = i * M_PI / 3.0;
    const Eigen::Vector3d p = position + Eigen::Vector3d(2.0 * std::cos(a),
                                                         2.0 * std::sin(a), 0.0);
    before.push_back(p);
    after.push_back(p);
    stamps.push_back(0.1 * i);
    yaw.push_back(M_PI / 2.0);
  }
  mod.applyDeformationSnapshot(1010.0, before, stamps, after, yaw);

  const auto counts = mod.flowMap().cellObservationCounts();
  ASSERT_FALSE(counts.empty());
  const auto rotated = mod.flowMap().predict(counts.front().index, 1000.0, 1.0);
  // A quarter turn at eight slots is two slots.
  EXPECT_EQ(dominantSlot(rotated), 2);
}

TEST(HeadingRotation, RepeatedCorrectionsRotateOnce) {
  FlowTemporalModule mod(moduleConfig());
  mod.start();

  // A cell alive across many corrections must end up rotated by its own epoch's
  // error, not by that error once per correction. The positional side is
  // recomputed from each cell's frozen original index for the same reason.
  const auto position = cellCenter(1.0, 1.0);
  mod.setVisibleVoxels(kVoxel, {position});
  for (int i = 0; i < 10; ++i) {
    const std::vector<Eigen::Vector3d> batch = {position};
    ASSERT_TRUE(mod.pushDetections(100.0 * i, batch, {0.0}, {1.0}));
    mod.flush();
  }

  std::vector<Eigen::Vector3d> before, after;
  std::vector<double> stamps, yaw;
  for (int i = 0; i < 6; ++i) {
    const double a = i * M_PI / 3.0;
    const Eigen::Vector3d p = position + Eigen::Vector3d(2.0 * std::cos(a),
                                                         2.0 * std::sin(a), 0.0);
    before.push_back(p);
    after.push_back(p);
    stamps.push_back(0.1 * i);
    // One slot, not a quarter turn: five quarter turns would land back on the
    // same slot and the test would pass even while compounding.
    yaw.push_back(kSlotWidth);
  }

  mod.applyDeformationSnapshot(1010.0, before, stamps, after, yaw);
  const auto after_first = mod.flowMap().cellObservationCounts().front().index;
  const auto slot_after_first =
      dominantSlot(mod.flowMap().predict(after_first, 1000.0, 1.0));

  // The same correction, delivered four more times.
  for (int repeat = 0; repeat < 4; ++repeat) {
    mod.applyDeformationSnapshot(1020.0 + repeat, before, stamps, after, yaw);
  }
  const auto counts = mod.flowMap().cellObservationCounts();
  ASSERT_FALSE(counts.empty());
  EXPECT_EQ(dominantSlot(mod.flowMap().predict(counts.front().index, 1000.0, 1.0)),
            slot_after_first)
      << "repeated corrections compounded the rotation: five applications of "
         "one slot would land five slots away";
}

TEST(HeadingRotation, LaterCorrectionAppliesOnlyTheAddedRotation) {
  FlowTemporalModule mod(moduleConfig());
  mod.start();

  const auto position = cellCenter(1.0, 1.0);
  mod.setVisibleVoxels(kVoxel, {position});
  for (int i = 0; i < 10; ++i) {
    const std::vector<Eigen::Vector3d> batch = {position};
    ASSERT_TRUE(mod.pushDetections(100.0 * i, batch, {0.0}, {1.0}));
    mod.flush();
  }

  auto ring = [&](double yaw_rad, std::vector<Eigen::Vector3d>& before,
                  std::vector<Eigen::Vector3d>& after, std::vector<double>& stamps,
                  std::vector<double>& yaw) {
    for (int i = 0; i < 6; ++i) {
      const double a = i * M_PI / 3.0;
      const Eigen::Vector3d p = position + Eigen::Vector3d(2.0 * std::cos(a),
                                                           2.0 * std::sin(a), 0.0);
      before.push_back(p);
      after.push_back(p);
      stamps.push_back(0.1 * i);
      yaw.push_back(yaw_rad);
    }
  };

  // A quarter turn, then a correction asking for half a turn in total: the
  // second pass must add the remaining quarter, landing on four slots, not six.
  std::vector<Eigen::Vector3d> b1, a1, b2, a2;
  std::vector<double> s1, y1, s2, y2;
  ring(M_PI / 2.0, b1, a1, s1, y1);
  ring(M_PI, b2, a2, s2, y2);
  mod.applyDeformationSnapshot(1010.0, b1, s1, a1, y1);
  mod.applyDeformationSnapshot(1020.0, b2, s2, a2, y2);

  const auto counts = mod.flowMap().cellObservationCounts();
  ASSERT_FALSE(counts.empty());
  // Half a turn at eight slots is four slots from the original heading 0.
  EXPECT_EQ(dominantSlot(mod.flowMap().predict(counts.front().index, 1000.0, 1.0)), 4);
}

TEST(HeadingRotation, SnapshotWithoutYawLeavesHeadingsUntouched) {
  FlowTemporalModule mod(moduleConfig());
  mod.start();

  const auto position = cellCenter(1.0, 1.0);
  mod.setVisibleVoxels(kVoxel, {position});
  for (int i = 0; i < 10; ++i) {
    const std::vector<Eigen::Vector3d> batch = {position};
    ASSERT_TRUE(mod.pushDetections(100.0 * i, batch, {0.0}, {1.0}));
    mod.flush();
  }

  std::vector<Eigen::Vector3d> before, after;
  std::vector<double> stamps;
  for (int i = 0; i < 6; ++i) {
    const double a = i * M_PI / 3.0;
    const Eigen::Vector3d p = position + Eigen::Vector3d(2.0 * std::cos(a),
                                                         2.0 * std::sin(a), 0.0);
    before.push_back(p);
    after.push_back(p);
    stamps.push_back(0.1 * i);
  }
  mod.applyDeformationSnapshot(1010.0, before, stamps, after);

  const auto counts = mod.flowMap().cellObservationCounts();
  ASSERT_FALSE(counts.empty());
  EXPECT_EQ(dominantSlot(mod.flowMap().predict(counts.front().index, 1000.0, 1.0)), 0);
}

}  // namespace kairos
