#include <gtest/gtest.h>
#include <kairos/flow/cell_state.h>
#include <kairos/flow/swgmm.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

FlowMapConfig sharingConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 5;
  cfg.candidate_periods = {3600.0f};
  cfg.sharing_lent_crossings = 3.0;
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

std::unique_ptr<FlowCellState> makeCell(const FlowMapConfig& cfg) {
  return std::make_unique<FlowCellState>(
      std::make_unique<FixedComponents>(cfg), cfg, testOmegas(cfg));
}

// A one-hot crossing in slot k: the share a detection exactly on slot k's mean
// heading contributes, so the expected weights stay exact.
std::vector<double> oneHot(int K, int k) {
  std::vector<double> r(static_cast<size_t>(K), 0.0);
  r[static_cast<size_t>(k)] = 1.0;
  return r;
}

void seed(FlowCellState& cell, int k, int n, const FlowMapConfig& cfg) {
  const int K = static_cast<int>(cell.nudft_models.size());
  for (int i = 0; i < n; ++i) {
    cell.feedWeightsSample(static_cast<double>(i), oneHot(K, k), cfg);
  }
}

// Pool a set of lender cells the way FlowMap does: count-weighted per slot,
// skipping any that has no crossing of its own.
NeighbourhoodEvidence pool(const std::vector<const FlowCellState*>& lenders) {
  NeighbourhoodEvidence out;
  for (const auto* lender : lenders) {
    if (!lender || lender->n_crossings <= 0 || lender->nudft_models.empty()) {
      continue;
    }
    const size_t K = lender->nudft_models.size();
    if (out.weighted_sum.empty()) {
      out.weighted_sum.assign(K, 0.0);
    }
    const double weight = static_cast<double>(lender->n_crossings);
    for (size_t k = 0; k < K; ++k) {
      out.weighted_sum[k] += weight * std::max(0.0, lender->nudft_models[k].gamma0);
    }
    out.crossings += weight;
  }
  return out;
}

double slot(const FlowCellState& cell, int k, const FlowMapConfig& cfg,
            const NeighbourhoodEvidence* nb) {
  return cell.sharedMean(k, cfg, nb);
}

}  // namespace

// The neighbourhood holds a share m/(C+m) of the reported weight, so the value
// reported is the cell's own mean shrunk toward the neighbourhood estimate by
// exactly the evidence lent.
TEST(EvidenceSharing, ReportsShrunkMean) {
  const auto cfg = sharingConfig();
  auto cell = makeCell(cfg);
  auto lender = makeCell(cfg);
  seed(*cell, 0, 1, cfg);    // all of its own mass in slot 0
  seed(*lender, 4, 8, cfg);  // lender's mass in the opposing slot
  const auto nb = pool({lender.get()});

  const double m = cfg.sharing_lent_crossings;
  const double own_share = 1.0 / (1.0 + m);
  EXPECT_NEAR(slot(*cell, 0, cfg, &nb), own_share, 1e-12);
  EXPECT_NEAR(slot(*cell, 4, cfg, &nb), m / (1.0 + m), 1e-12);
}

// The lent quantity is fixed while the cell's own evidence grows, so the
// neighbourhood's share falls to zero rather than settling at a fixed fraction.
TEST(EvidenceSharing, ShareVanishesWithOwnEvidence) {
  const auto cfg = sharingConfig();
  auto lender = makeCell(cfg);
  seed(*lender, 4, 500, cfg);
  const auto nb = pool({lender.get()});
  const double m = cfg.sharing_lent_crossings;

  double previous_share = 1.0;
  for (const int own : {1, 10, 50, 400}) {
    auto cell = makeCell(cfg);
    seed(*cell, 0, own, cfg);
    // Slot 4 carries none of the cell's own mass, so what it reports there is
    // the neighbourhood's share of the estimate.
    const double share = slot(*cell, 4, cfg, &nb);
    EXPECT_NEAR(share, m / (static_cast<double>(own) + m), 1e-12);
    EXPECT_LT(share, previous_share) << "share did not fall at C=" << own;
    previous_share = share;
  }
  EXPECT_LT(previous_share, 0.01) << "share has not vanished by 400 crossings";
}

// Reading the weights must not alter them: the stored state stays the cell's own
// running mean, which is what keeps it order-invariant.
TEST(EvidenceSharing, LeavesStoredStateUntouched) {
  const auto cfg = sharingConfig();
  auto cell = makeCell(cfg);
  auto lender = makeCell(cfg);
  seed(*cell, 0, 2, cfg);
  seed(*lender, 4, 50, cfg);
  const auto nb = pool({lender.get()});

  std::vector<double> before;
  for (const auto& model : cell->nudft_models) {
    before.push_back(model.gamma0);
  }
  for (int i = 0; i < 20; ++i) {
    for (int k = 0; k < static_cast<int>(cell->nudft_models.size()); ++k) {
      (void)slot(*cell, k, cfg, &nb);
    }
    (void)cell->predict(100.0 + i, 1.0, cfg, &nb);
  }
  for (size_t k = 0; k < before.size(); ++k) {
    EXPECT_NEAR(cell->nudft_models[k].gamma0, before[k], 1e-15) << "slot " << k;
  }
}

// A lender contributes in proportion to its own crossings, so a busy neighbour
// outweighs a barely-seen one rather than counting equally.
TEST(EvidenceSharing, LenderWeightFollowsCrossings) {
  const auto cfg = sharingConfig();
  auto cell = makeCell(cfg);
  auto busy = makeCell(cfg);
  auto sparse = makeCell(cfg);
  seed(*cell, 0, 1, cfg);
  seed(*busy, 2, 400, cfg);
  seed(*sparse, 6, 1, cfg);
  const auto nb = pool({busy.get(), sparse.get()});

  EXPECT_NEAR(slot(*cell, 2, cfg, &nb) / slot(*cell, 6, cfg, &nb), 400.0, 1e-6);
}

// A cell allocated but never crossed carries an all-zero weight vector, which is
// an absence of evidence: it must not enter the pool.
TEST(EvidenceSharing, UncrossedLenderIsIgnored) {
  const auto cfg = sharingConfig();
  auto lender = makeCell(cfg);
  auto empty = makeCell(cfg);
  seed(*lender, 4, 8, cfg);
  ASSERT_EQ(empty->n_crossings, 0);

  const auto with_empty = pool({lender.get(), empty.get()});
  const auto without = pool({lender.get()});
  EXPECT_EQ(with_empty.crossings, without.crossings);
  for (size_t k = 0; k < without.weighted_sum.size(); ++k) {
    EXPECT_NEAR(with_empty.weighted_sum[k], without.weighted_sum[k], 1e-15);
  }
}

// With no neighbourhood to draw on, or with no crossings lent, a cell reports
// its own mean unchanged.
TEST(EvidenceSharing, FallsBackToOwnMean) {
  auto cfg = sharingConfig();
  auto cell = makeCell(cfg);
  seed(*cell, 0, 3, cfg);
  const double own = cell->nudft_models[0].gamma0;

  const NeighbourhoodEvidence empty_pool;
  EXPECT_NEAR(slot(*cell, 0, cfg, &empty_pool), own, 1e-15);
  EXPECT_NEAR(slot(*cell, 0, cfg, nullptr), own, 1e-15);

  auto lender = makeCell(cfg);
  seed(*lender, 4, 100, cfg);
  const auto nb = pool({lender.get()});
  cfg.sharing_lent_crossings = 0.0;
  EXPECT_NEAR(slot(*cell, 0, cfg, &nb), own, 1e-15);
}

// A cell with no crossings of its own reports the neighbourhood estimate, which
// is the case the mechanism exists for.
TEST(EvidenceSharing, UnobservedCellReportsNeighbourhood) {
  const auto cfg = sharingConfig();
  auto cell = makeCell(cfg);
  auto lender = makeCell(cfg);
  seed(*lender, 4, 20, cfg);
  const auto nb = pool({lender.get()});
  ASSERT_EQ(cell->n_crossings, 0);

  EXPECT_NEAR(slot(*cell, 4, cfg, &nb), 1.0, 1e-12);
  EXPECT_NEAR(slot(*cell, 0, cfg, &nb), 0.0, 1e-12);
}

// The reported weights remain a distribution over the slots: shrinkage is a
// convex combination of two vectors that each sum to one.
TEST(EvidenceSharing, ReportedWeightsSumToOne) {
  const auto cfg = sharingConfig();
  auto a = makeCell(cfg);
  auto b = makeCell(cfg);
  auto cell = makeCell(cfg);
  seed(*a, 3, 17, cfg);
  seed(*b, 5, 4, cfg);
  const auto nb = pool({a.get(), b.get()});

  const int K = static_cast<int>(cell->nudft_models.size());
  for (int i = 0; i < 40; ++i) {
    cell->feedWeightsSample(200.0 + i, oneHot(K, i % K), cfg);
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
      const double v = slot(*cell, k, cfg, &nb);
      EXPECT_GE(v, 0.0);
      sum += v;
    }
    ASSERT_NEAR(sum, 1.0, 1e-9) << "after crossing " << i;
  }
}

// Sharing substitutes the mean term of the forecast and leaves the harmonics as
// fitted, so the periodic structure the coefficients carry is preserved.
TEST(EvidenceSharing, HarmonicsUnaffected) {
  const auto cfg = sharingConfig();
  auto cell = makeCell(cfg);
  auto lender = makeCell(cfg);
  seed(*lender, 4, 50, cfg);
  const auto nb = pool({lender.get()});

  // A slot whose share swings through the day, so its harmonic terms are
  // non-zero and the two reconstructions differ by the mean alone.
  const int K = static_cast<int>(cell->nudft_models.size());
  for (int i = 0; i < 40; ++i) {
    const double t = 300.0 * i;
    cell->feedWeightsSample(t, oneHot(K, (i % 4 == 0) ? 2 : 0), cfg);
  }
  const auto& model = cell->nudft_models[2];
  ASSERT_GT(model.n, cfg.nudft_min_obs);

  const double t_query = 12345.0;
  const double own_mean = model.gamma0;
  const double shared = slot(*cell, 2, cfg, &nb);
  ASSERT_NE(shared, own_mean);

  const double own_forecast = model.predictGated(t_query, cfg.nudft_order);
  const double shared_forecast =
      model.predictGatedWithMean(t_query, cfg.nudft_order, shared);
  // The harmonic contribution is identical; only the mean term moved.
  EXPECT_NEAR(shared_forecast - own_forecast, shared - own_mean, 1e-12);
}

}  // namespace kairos
