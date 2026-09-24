#include "kairos/flow/flow_map.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <unordered_set>

#include <config_utilities/validation.h>
#include <glog/logging.h>

namespace kairos {
using namespace hydra;  // base-Hydra types visible

namespace {

constexpr int kFlowMapCheckpointVersion = 1;

// The helpers in this anonymous namespace are a frequency-set builder plus the
// JSON (de)serialization primitives for indices, NUDFT vectors, and mixture
// components, used by the checkpoint save/load methods further down.

bool setError(std::string* error, const std::string& message) {
  if (error) {
    *error = message;
  }

  return false;
}

// Convert candidate periods (seconds) into angular frequencies omega = 2*pi/p,
// dropping any non-positive period. This single set is shared by every predictor.
std::vector<double> makeFrequencySet(const std::vector<float>& periods) {
  std::vector<double> omegas;
  omegas.reserve(periods.size());
  for (const auto period : periods) {
    const double p = static_cast<double>(period);
    if (p <= 0.0) {
      continue;
    }

    omegas.push_back(2.0 * M_PI / p);
  }

  return omegas;
}

nlohmann::json indexToJson(const GlobalIndex& index) {
  return nlohmann::json::array({index.x(), index.y(), index.z()});
}

bool indexFromJson(const nlohmann::json& record,
                   GlobalIndex& index,
                   std::string* error) {
  if (!record.is_array() || record.size() != 3 || !record[0].is_number_integer() ||
      !record[1].is_number_integer() || !record[2].is_number_integer()) {
    return setError(error, "invalid GlobalIndex record");
  }

  index = GlobalIndex(record[0].get<int>(), record[1].get<int>(), record[2].get<int>());
  return true;
}

nlohmann::json nudftVectorToJson(const std::vector<NUDFTModel>& models) {
  nlohmann::json result = nlohmann::json::array();
  for (const auto& model : models) {
    result.push_back(nudftToJson(model));
  }

  return result;
}

bool nudftVectorFromJson(const nlohmann::json& record,
                         std::vector<NUDFTModel>& models,
                         std::string* error) {
  if (!record.is_array()) {
    return setError(error, "invalid NUDFT model array");
  }

  models.clear();
  models.reserve(record.size());
  for (const auto& value : record) {
    NUDFTModel model;
    if (!nudftFromJson(value, model, error)) {
      return false;
    }

    models.push_back(std::move(model));
  }

  return true;
}

void configureNudftVector(std::vector<NUDFTModel>& models,
                          const std::shared_ptr<const std::vector<double>>& omegas,
                          int min_observations) {
  for (auto& model : models) {
    model.omegas_ptr = omegas;
    model.min_observations = min_observations;
  }
}

nlohmann::json swndToJson(const SWNDComponent& component) {
  return {{"mu_theta", component.mu_theta},
          {"mu_rho", component.mu_rho},
          {"sigma_theta", component.Sigma(0, 0)},
          {"sigma_rho", component.Sigma(1, 1)},
          {"sigma_cross", component.Sigma(0, 1)}};
}

bool swndFromJson(const nlohmann::json& record,
                  SWNDComponent& component,
                  std::string* error) {
  if (!record.is_object()) {
    return setError(error, "invalid SWND component record");
  }

  if (!record.contains("mu_theta") || !record.contains("mu_rho") ||
      !record.contains("sigma_theta") || !record.contains("sigma_rho") ||
      !record.contains("sigma_cross")) {
    return setError(error, "missing SWND component fields");
  }

  if (!record.at("mu_theta").is_number() || !record.at("mu_rho").is_number() ||
      !record.at("sigma_theta").is_number() || !record.at("sigma_rho").is_number() ||
      !record.at("sigma_cross").is_number()) {
    return setError(error, "non-numeric SWND component field");
  }

  component.mu_theta = record.at("mu_theta").get<double>();
  component.mu_rho = record.at("mu_rho").get<double>();
  const double sigma_theta = record.at("sigma_theta").get<double>();
  const double sigma_rho = record.at("sigma_rho").get<double>();
  const double sigma_cross = record.at("sigma_cross").get<double>();
  component.Sigma << sigma_theta, sigma_cross, sigma_cross, sigma_rho;
  precompute(component);
  return true;
}

}  // namespace

FlowMap::FlowMap(const FlowMapConfig& cfg)
    : cfg_(::config::checkValid(cfg)),
      shared_omegas_(std::make_shared<const std::vector<double>>(
          makeFrequencySet(cfg_.candidate_periods))) {}

void FlowMap::ensureCells(const GlobalIndexSet& indices) {
  const bool timing_enabled = cfg_.enable_timing;
  const auto start = std::chrono::steady_clock::now();
  size_t inserted = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& index : indices) {
      if (cells_.count(index)) {
        continue;
      }
      // Cells made by ensureCells() have no observation yet, so we don't set
      // the original-position metadata here — the first addObservation() does.
      cells_.emplace(index, FlowCellState(makeModel(), cfg_, shared_omegas_, global_speed_prior_));
      ++inserted;
    }
  }

  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    const double per_index_us =
        indices.empty() ? 0.0 : static_cast<double>(elapsed_us) / indices.size();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      timing_.ensure_cells_count++;
      timing_.ensure_cells_total_us += static_cast<uint64_t>(elapsed_us);
    }
    LOG(INFO) << "[FlowTiming] ensureCells total=" << elapsed_us
              << " us, indices=" << indices.size() << ", inserted=" << inserted
              << ", avg=" << per_index_us << " us/index";
  }
}

void FlowMap::addObservation(const GlobalIndex& index,
                             double theta,
                             double rho,
                             double t_seconds,
                             double delta_t_visible,
                             int64_t track_id) {
  const bool timing_enabled = cfg_.enable_timing;
  const auto start = std::chrono::steady_clock::now();
  bool updated = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = cells_.find(index);
    if (iter == cells_.end()) {
      auto [inserted_iter, inserted] =
          cells_.emplace(index, FlowCellState(makeModel(), cfg_, shared_omegas_, global_speed_prior_));
      iter = inserted_iter;
      (void)inserted;
    }

    // Pin the original (pre-deformation) hash + stamp on first observation.
    // Subsequent loop closures move the cell's KEY in cells_ via remapCells,
    // but original_index/creation_timestamp_ns stay frozen so the next
    // deformation feeds the deformation graph the truly-original observation
    // location and time, not a stale already-corrected position.
    auto& cell = iter->second;
    if (!cell.original_set) {
      cell.original_index = index;
      cell.creation_timestamp_ns =
          t_seconds > 0.0 ? static_cast<uint64_t>(t_seconds * 1.0e9) : uint64_t{0};
      cell.original_set = true;
    }

    std::vector<double> r;
    cell.update(theta, rho, t_seconds, delta_t_visible, cfg_, &r);

    // Per-passage weights feeding: accumulate this detection's responsibility
    // into the track's open visit; the visit is fed to its cell as ONE
    // averaged sample when the track changes cell, goes silent for
    // visit_gap_s, or exceeds visit_max_s. Detections without a track id
    // cannot be grouped into passages and feed a per-detection sample.
    if (!r.empty()) {
      if (track_id < 0) {
        // No identity to group detections into a passage: this detection is
        // itself the weights sample.
        cell.feedWeightsSample(t_seconds, r, cfg_);
      } else {
        auto vit = open_visits_.find(track_id);
        if (vit != open_visits_.end()) {
          auto& v = vit->second;
          const bool moved = (v.cell != index);
          const bool stale = (t_seconds - v.t_last) > cfg_.visit_gap_s;
          const bool capped = (t_seconds - v.t_first) > cfg_.visit_max_s;
          if (moved || stale || capped) {
            closeVisitLocked(v);
            open_visits_.erase(vit);
            vit = open_visits_.end();
          }
        }
        if (vit == open_visits_.end()) {
          OpenVisit v;
          v.cell = index;
          v.sum_r = r;
          v.n = 1.0;
          v.t_first = t_seconds;
          v.t_last = t_seconds;
          open_visits_.emplace(track_id, std::move(v));
        } else {
          auto& v = vit->second;
          for (size_t k = 0; k < r.size() && k < v.sum_r.size(); ++k) {
            v.sum_r[k] += r[k];
          }
          v.n += 1.0;
          v.t_last = t_seconds;
        }
      }
    }

    // Feed the map-wide running-mean speed prior, mirroring cell.update()'s
    // rho > 0 guard so only accepted observations count toward it.
    if (rho > 0.0) {
      global_speed_prior_->add(rho);
    }

    updated = true;
    if (timing_enabled) {
      const auto elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      timing_.add_observation_count++;
      timing_.add_observation_total_us += static_cast<uint64_t>(elapsed_us);
      timing_.add_observation_samples_us.push_back(static_cast<uint32_t>(elapsed_us));
    }
  }

  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    LOG_EVERY_N(INFO, 2000) << "[FlowTiming] addObservation last=" << elapsed_us
                            << " us, updated=" << (updated ? 1 : 0);
  }
}

FlowPrediction FlowMap::predict(const GlobalIndex& index,
                                double t_seconds,
                                double delta_t) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = cells_.find(index);
  if (iter == cells_.end()) {
    return {};
  }

  const auto neighbourhood = timedNeighbourhoodEvidenceLocked(index);
  return iter->second.predict(t_seconds, delta_t, cfg_, &neighbourhood);
}

std::vector<double> FlowMap::responsibilities(const GlobalIndex& index,
                                              double theta,
                                              double rho) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = cells_.find(index);
  if (iter == cells_.end() || !iter->second.swgmm) {
    return {};
  }

  return iter->second.swgmm->responsibilities(theta, rho);
}

void FlowMap::addPoissonExposure(const GlobalIndexSet& visible_indices, double dt_visible,
                                 double t_seconds) {
  if (dt_visible <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& index : visible_indices) {
    const auto iter = cells_.find(index);
    if (iter != cells_.end()) {
      iter->second.poisson.addExposure(dt_visible, t_seconds);
    }
  }
}

void FlowMap::closeVisitLocked(OpenVisit& visit) {
  if (visit.n <= 0.0) {
    return;
  }
  const auto iter = cells_.find(visit.cell);
  if (iter == cells_.end()) {
    return;  // cell merged or re-keyed away while the visit was open
  }
  std::vector<double> mean_r(visit.sum_r.size());
  for (size_t k = 0; k < mean_r.size(); ++k) {
    mean_r[k] = visit.sum_r[k] / visit.n;
  }
  iter->second.feedWeightsSample(0.5 * (visit.t_first + visit.t_last), mean_r, cfg_);
}

void FlowMap::flushVisits() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [id, visit] : open_visits_) {
    closeVisitLocked(visit);
  }
  open_visits_.clear();
}

std::vector<FlowMap::PresenceGridEntry> FlowMap::presenceGrid(
    const GlobalIndexSet& indices, double t_seconds,
    const std::vector<double>& horizons_s) const {
  std::vector<PresenceGridEntry> out;
  out.reserve(indices.size());
  std::lock_guard<std::mutex> lock(mutex_);
  const int order = cfg_.nudft_order;
  const int min_obs = cfg_.nudft_min_obs;
  for (const auto& index : indices) {
    const auto iter = cells_.find(index);
    if (iter == cells_.end()) {
      continue;  // never-observed voxel: no presence model to query
    }
    const auto& p = iter->second.poisson;
    const double speed = iter->second.cellSpeedMean();  // -1 if no observations
    const double dwell = (speed > 0.0) ? cfg_.voxel_size_m / speed : -1.0;
    PresenceGridEntry e;
    e.index = index;
    e.p_present.reserve(horizons_s.size());
    for (const double h : horizons_s) {
      e.p_present.push_back(p.predictPresence(t_seconds, h, dwell, order, min_obs));
    }
    out.push_back(e);
  }
  return out;
}

std::vector<FlowMap::FlowGridEntry> FlowMap::flowGrid(const GlobalIndexSet& indices,
                                                      double t_seconds) const {
  std::vector<FlowGridEntry> out;
  out.reserve(indices.size());
  std::lock_guard<std::mutex> lock(mutex_);
  const int order = cfg_.nudft_order;
  const int min_obs = cfg_.nudft_min_obs;
  // Per-slot forecast weights at the gated spectral order. Mirrors queryPi's
  // fallback: a cell whose weight predictors are not yet valid reports its
  // mean-term mix.
  const auto forecast_pi = [this](const FlowCellState& cell, double t, int ord,
                                  const NeighbourhoodEvidence& nb) {
    const int K = cell.swgmm ? cell.swgmm->numComponents() : 1;
    std::vector<double> pi(static_cast<size_t>(K), 1.0 / static_cast<double>(K));
    if (static_cast<int>(cell.nudft_models.size()) != K) {
      return pi;
    }
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
      const auto& m = cell.nudft_models[static_cast<size_t>(k)];
      // The shared (evidence-sharing) mean stands in for gamma0.
      const double mean = cell.sharedMean(k, cfg_, &nb);
      const double v = (m.n > 0 || nb.usable())
                           ? std::max(0.0, cell.nudft_valid
                                               ? m.predictGatedWithMean(t, ord, mean)
                                               : mean)
                           : 0.0;
      pi[static_cast<size_t>(k)] = v;
      sum += v;
    }
    if (sum <= 0.0) {
      std::fill(pi.begin(), pi.end(), 1.0 / static_cast<double>(K));
      return pi;
    }
    for (auto& v : pi) {
      v /= sum;
    }
    return pi;
  };
  for (const auto& index : indices) {
    const auto iter = cells_.find(index);
    if (iter == cells_.end()) {
      continue;  // never-observed voxel: nothing to forecast
    }
    const auto& cell = iter->second;
    FlowGridEntry e;
    e.index = index;
    e.lambda = cell.poisson.predictLambda(t_seconds, order, min_obs);
    const auto neighbourhood = neighbourhoodEvidenceLocked(index);
    e.pi = forecast_pi(cell, t_seconds, order, neighbourhood);
    out.push_back(e);
  }
  return out;
}

std::optional<FlowCellSnapshot> FlowMap::tryGetCellSnapshot(const GlobalIndex& index,
                                                            double t_seconds,
                                                            double delta_t) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = cells_.find(index);
  if (iter == cells_.end()) {
    return std::nullopt;
  }

  const auto neighbourhood = timedNeighbourhoodEvidenceLocked(index);
  return iter->second.snapshot(t_seconds, delta_t, cfg_, &neighbourhood);
}

std::vector<GlobalIndex> FlowMap::cellIndices() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<GlobalIndex> indices;
  indices.reserve(cells_.size());
  for (const auto& [index, _] : cells_) {
    (void)_;
    indices.push_back(index);
  }
  return indices;
}

std::vector<FlowMap::CellCount> FlowMap::cellObservationCounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<CellCount> out;
  out.reserve(cells_.size());
  for (const auto& [index, state] : cells_) {
    out.push_back({index, state.total_observations});
  }
  return out;
}

void FlowMap::setSharingTopology(SharingTopology topology) {
  std::lock_guard<std::mutex> lock(mutex_);
  sharing_topology_ = std::move(topology);
}

bool FlowMap::hasSharingTopology() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !sharing_topology_.place_of_voxel.empty();
}

NeighbourhoodEvidence FlowMap::neighbourhoodEvidenceLocked(const GlobalIndex& index) const {
  // Caller must already hold mutex_; this does not acquire it.
  NeighbourhoodEvidence out;
  if (cfg_.sharing_lent_crossings <= 0.0) {
    return out;
  }
  const auto place_iter = sharing_topology_.place_of_voxel.find(index);
  if (place_iter == sharing_topology_.place_of_voxel.end()) {
    return out;  // the place graph does not cover this cell yet
  }

  // Pool one place's support set, skipping the cell itself: a voxel lends to its
  // neighbours, not to itself, and a voxel with no crossing carries no evidence.
  const auto pool_place = [&](spark_dsg::NodeId place) {
    const auto support = sharing_topology_.support_of_place.find(place);
    if (support == sharing_topology_.support_of_place.end()) {
      return;
    }
    for (const auto& neighbour_index : support->second) {
      if (neighbour_index == index) {
        continue;
      }
      const auto cell_iter = cells_.find(neighbour_index);
      if (cell_iter == cells_.end()) {
        continue;
      }
      const auto& neighbour = cell_iter->second;
      if (neighbour.n_crossings <= 0) {
        continue;
      }
      const size_t K = neighbour.nudft_models.size();
      if (K == 0) {
        continue;
      }
      if (out.weighted_sum.empty()) {
        out.weighted_sum.assign(K, 0.0);
      } else if (out.weighted_sum.size() != K) {
        continue;  // slot geometry must line up for the pool to be index-aligned
      }
      const double weight = static_cast<double>(neighbour.n_crossings);
      for (size_t k = 0; k < K; ++k) {
        out.weighted_sum[k] += weight * std::max(0.0, neighbour.nudft_models[k].gamma0);
      }
      out.crossings += weight;
    }
  };

  // Walk out from the cell's own place along the place edges, up to the hop
  // budget the clearance sets, pooling every place reached. At a budget of one
  // this is the place and its immediate siblings.
  const auto own_place = place_iter->second;
  int budget = 1;
  const auto hops_iter = sharing_topology_.hops_of_place.find(own_place);
  if (hops_iter != sharing_topology_.hops_of_place.end()) {
    budget = std::max(1, hops_iter->second);
  }

  std::unordered_set<spark_dsg::NodeId> visited{own_place};
  std::vector<spark_dsg::NodeId> frontier{own_place};
  pool_place(own_place);
  for (int hop = 0; hop < budget && !frontier.empty(); ++hop) {
    std::vector<spark_dsg::NodeId> next;
    for (const auto place : frontier) {
      const auto siblings = sharing_topology_.siblings_of_place.find(place);
      if (siblings == sharing_topology_.siblings_of_place.end()) {
        continue;
      }
      for (const auto sibling : siblings->second) {
        if (!visited.insert(sibling).second) {
          continue;
        }
        pool_place(sibling);
        next.push_back(sibling);
      }
    }
    frontier = std::move(next);
  }
  return out;
}

NeighbourhoodEvidence FlowMap::timedNeighbourhoodEvidenceLocked(const GlobalIndex& index) const {
  if (!cfg_.enable_timing) {
    return neighbourhoodEvidenceLocked(index);
  }
  const auto start = std::chrono::steady_clock::now();
  auto out = neighbourhoodEvidenceLocked(index);
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  timing_.sharing_count++;
  timing_.sharing_total_us += static_cast<uint64_t>(elapsed_us);
  timing_.sharing_samples_us.push_back(static_cast<uint32_t>(elapsed_us));
  return out;
}

std::vector<CellRemapInput> FlowMap::cellRemapInputs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<CellRemapInput> out;
  out.reserve(cells_.size());
  for (const auto& [index, cell] : cells_) {
    CellRemapInput input;
    input.current_index = index;
    input.original_index = cell.original_index;
    input.creation_timestamp_ns = cell.creation_timestamp_ns;
    input.original_set = cell.original_set;
    out.push_back(input);
  }
  return out;
}

void FlowMap::remapCells(
    const std::vector<std::pair<GlobalIndex, GlobalIndex>>& remap_pairs) {
  remapCells(remap_pairs, {});
}

void FlowMap::remapCells(
    const std::vector<std::pair<GlobalIndex, GlobalIndex>>& remap_pairs,
    const std::vector<double>& yaw_per_pair) {
  const bool timing_enabled = cfg_.enable_timing;
  const auto start = std::chrono::steady_clock::now();
  size_t processed_cells = 0;
  size_t collision_count = 0;
  size_t identity_pairs = 0;
  size_t moved_pairs = 0;
  size_t rotated_cells = 0;
  // Diagnostics for the cross-room-merge complaint: when two cells collide
  // at the same target, what was their separation in voxel units? A handful
  // of voxels = legitimate drift correction; tens of voxels = the deformation
  // field is mapping physically-distant cells onto one target.
  int64_t max_source_sep_sq_v = 0;
  int64_t sum_source_sep_sq_v = 0;
  int64_t max_displacement_sq_v = 0;
  int64_t sum_displacement_sq_v = 0;
  size_t detail_logged = 0;
  constexpr size_t kDetailCap = 32;

  {
    std::lock_guard<std::mutex> lock(mutex_);

  // Feed every open visit before cell keys move: a visit keyed to a
  // pre-deformation index would otherwise be dropped by closeVisitLocked's
  // cells_.find after the remap.
  for (auto& [id, visit] : open_visits_) {
    closeVisitLocked(visit);
  }
  open_visits_.clear();
    if (remap_pairs.empty() || cells_.empty()) {
      return;
    }

    int64_t pre_total = 0;
    for (const auto& [_, cell] : cells_) {
      (void)_;
      pre_total += static_cast<int64_t>(cell.total_observations);
    }
    total_observations_pre_remap_ = pre_total;

    auto index_sq_distance = [](const GlobalIndex& a, const GlobalIndex& b) {
      const int64_t dx = static_cast<int64_t>(a.x()) - static_cast<int64_t>(b.x());
      const int64_t dy = static_cast<int64_t>(a.y()) - static_cast<int64_t>(b.y());
      const int64_t dz = static_cast<int64_t>(a.z()) - static_cast<int64_t>(b.z());
      return dx * dx + dy * dy + dz * dz;
    };

    // A rotating correction carries one yaw per pair; a purely translational
    // one carries none and the lookup stays empty.
    GlobalIndexMap<double> yaw_lookup;
    if (!yaw_per_pair.empty()) {
      if (yaw_per_pair.size() != remap_pairs.size()) {
        LOG(ERROR) << "[FlowMap.remap] yaw_per_pair size " << yaw_per_pair.size()
                   << " does not match remap_pairs " << remap_pairs.size()
                   << "; applying the translation only.";
      } else {
        yaw_lookup.reserve(remap_pairs.size());
        for (size_t i = 0; i < remap_pairs.size(); ++i) {
          if (std::abs(yaw_per_pair[i]) > 0.0) {
            yaw_lookup[remap_pairs[i].first] = yaw_per_pair[i];
          }
        }
      }
    }

    GlobalIndexMap<GlobalIndex> remap_lookup;
    remap_lookup.reserve(remap_pairs.size());
    for (const auto& [old_index, new_index] : remap_pairs) {
      remap_lookup[old_index] = new_index;
      const int64_t dsq = index_sq_distance(new_index, old_index);
      if (dsq == 0) {
        ++identity_pairs;
      } else {
        ++moved_pairs;
      }
      sum_displacement_sq_v += dsq;
      if (dsq > max_displacement_sq_v) {
        max_displacement_sq_v = dsq;
      }
    }

    GlobalIndexMap<FlowCellState> remapped;
    GlobalIndexMap<GlobalIndex> remapped_origin;
    remapped.reserve(cells_.size());
    remapped_origin.reserve(cells_.size());

    const double now_s = std::chrono::duration<double>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();

    for (auto& [old_index, cell] : cells_) {
      ++processed_cells;
      const auto remap_iter = remap_lookup.find(old_index);
      const GlobalIndex target_index =
          remap_iter == remap_lookup.end() ? old_index : remap_iter->second;

      // Rotate before the move, so a cell that collides with another at the
      // target is merged with both sides already in the corrected frame. The
      // requested angle is measured against the frame the cell's observations
      // were accumulated in, the same reference the positional correction uses,
      // so only the part not yet carried is applied.
      if (!yaw_lookup.empty()) {
        const auto yaw_iter = yaw_lookup.find(old_index);
        if (yaw_iter != yaw_lookup.end()) {
          const double target = yaw_iter->second;
          const double delta = std::atan2(
              std::sin(target - cell.heading_rotation_applied),
              std::cos(target - cell.heading_rotation_applied));
          if (std::abs(delta) > 1.0e-9) {
            cell.rotateHeading(delta);
            cell.heading_rotation_applied = target;
            ++rotated_cells;
          }
        }
      }

      auto target_iter = remapped.find(target_index);
      if (target_iter == remapped.end()) {
        remapped.emplace(target_index, std::move(cell));
        remapped_origin.emplace(target_index, old_index);
        continue;
      }

      // Statistical merge: preserve sufficient statistics from both cells
      // instead of the winner-take-all overwrite that used to live here.
      const auto origin_iter = remapped_origin.find(target_index);
      const GlobalIndex source_a =
          origin_iter == remapped_origin.end() ? target_index : origin_iter->second;
      MergeEvent event;
      event.source_a = source_a;
      event.source_b = old_index;
      event.target = target_index;
      event.n_a = target_iter->second.total_observations;
      event.n_b = cell.total_observations;
      event.timestamp = now_s;

      // Capture pre-merge cell flags so the detail log can show which side
      // contributed mature state vs which side was a fresh single-obs cell.
      const bool nudft_valid_a = target_iter->second.nudft_valid;
      const bool nudft_valid_b = cell.nudft_valid;

      const int64_t sep_sq = index_sq_distance(source_a, old_index);
      sum_source_sep_sq_v += sep_sq;
      if (sep_sq > max_source_sep_sq_v) {
        max_source_sep_sq_v = sep_sq;
      }

      const int64_t da_sq = index_sq_distance(source_a, target_index);
      const int64_t db_sq = index_sq_distance(old_index, target_index);

      if (detail_logged < kDetailCap) {
        ++detail_logged;
        LOG(INFO) << "[FlowMerge.detail] target=(" << target_index.x() << ","
                  << target_index.y() << "," << target_index.z()
                  << ") A=(" << source_a.x() << "," << source_a.y() << ","
                  << source_a.z() << ")"
                  << " B=(" << old_index.x() << "," << old_index.y() << ","
                  << old_index.z() << ")"
                  << " |A-B|_v=" << std::sqrt(static_cast<double>(sep_sq))
                  << " |A-T|_v=" << std::sqrt(static_cast<double>(da_sq))
                  << " |B-T|_v=" << std::sqrt(static_cast<double>(db_sq))
                  << " n_a=" << event.n_a << " n_b=" << event.n_b
                  << " nudft=" << (nudft_valid_a ? 1 : 0) << "/"
                  << (nudft_valid_b ? 1 : 0);
      }

      target_iter->second.mergeFrom(std::move(cell), cfg_);
      merge_events_.push_back(event);
      ++collision_count;
    }

    cells_ = std::move(remapped);

    int64_t post_total = 0;
    for (const auto& [_, cell] : cells_) {
      (void)_;
      post_total += static_cast<int64_t>(cell.total_observations);
    }
    total_observations_post_remap_ = post_total;
  }

  // Always log a one-line per-remap summary so offline runs leave a trail of
  // what happened at each loop closure (no need to inspect ROS topics).
  // The conservation invariant must hold under correct statistical merging.
  const bool conserved = total_observations_pre_remap_ == total_observations_post_remap_;
  const double mean_disp_sq_v =
      remap_pairs.empty()
          ? 0.0
          : static_cast<double>(sum_displacement_sq_v) /
                static_cast<double>(remap_pairs.size());
  const double mean_sep_sq_v =
      collision_count == 0
          ? 0.0
          : static_cast<double>(sum_source_sep_sq_v) /
                static_cast<double>(collision_count);
  LOG(INFO) << "[FlowMerge] remap_pairs=" << remap_pairs.size()
            << " identity=" << identity_pairs << " moved=" << moved_pairs
            << " cells_in=" << processed_cells << " cells_out=" << cells_.size()
            << " rotated=" << rotated_cells
            << " collisions=" << collision_count
            << " obs_pre=" << total_observations_pre_remap_
            << " obs_post=" << total_observations_post_remap_
            << " conserved=" << (conserved ? "true" : "false")
            << " disp_v: max=" << std::sqrt(static_cast<double>(max_displacement_sq_v))
            << " rms=" << std::sqrt(mean_disp_sq_v)
            << " src_sep_v: max=" << std::sqrt(static_cast<double>(max_source_sep_sq_v))
            << " rms=" << std::sqrt(mean_sep_sq_v);
  if (!conserved) {
    LOG(WARNING) << "[FlowMerge] observation conservation invariant violated: "
                 << "delta=" << (total_observations_post_remap_ - total_observations_pre_remap_);
  }
  // Heuristic flag: if two source cells more than 10 voxels apart get merged,
  // the deformation field is collapsing physically-distant locations together.
  // 10 voxels ≈ 1 m at the default 0.1 m voxel size — well above legitimate
  // drift correction at the timescales we're closing loops on.
  if (max_source_sep_sq_v > 100) {
    LOG(WARNING) << "[FlowMerge] cross-cell source separation suspicious: "
                 << "max=" << std::sqrt(static_cast<double>(max_source_sep_sq_v))
                 << " voxels (collisions=" << collision_count
                 << "). Two cells this far apart should not deform onto the same target.";
  }

  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    const double per_cell_us =
        processed_cells == 0 ? 0.0 : static_cast<double>(elapsed_us) / processed_cells;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      timing_.remap_cells_count++;
      timing_.remap_cells_total_us += static_cast<uint64_t>(elapsed_us);
      timing_.remap_cells_samples_us.push_back(static_cast<uint32_t>(elapsed_us));
    }
    LOG(INFO) << "[FlowTiming] remapCells total=" << elapsed_us
              << " us, remap_pairs=" << remap_pairs.size() << ", cells=" << processed_cells
              << ", avg=" << per_cell_us << " us/cell";
  }
}

std::vector<MergeEvent> FlowMap::drainMergeEvents() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<MergeEvent> out;
  out.swap(merge_events_);
  return out;
}

int64_t FlowMap::totalObservationsPreRemap() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_observations_pre_remap_;
}

int64_t FlowMap::totalObservationsPostRemap() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_observations_post_remap_;
}

void FlowMap::runMaintenance(double t_seconds) {
  const bool timing_enabled = cfg_.enable_timing;
  const auto start = std::chrono::steady_clock::now();
  size_t cell_count = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cell_count = cells_.size();
    // Close visits whose track has gone silent (left the cell or the field of
    // view); without this sweep a departed track's last visit would stay open
    // until checkpoint time.
    for (auto it = open_visits_.begin(); it != open_visits_.end();) {
      if ((t_seconds - it->second.t_last) > cfg_.visit_gap_s) {
        closeVisitLocked(it->second);
        it = open_visits_.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (timing_enabled) {
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    const double per_cell_us =
        cell_count == 0 ? 0.0 : static_cast<double>(elapsed_us) / cell_count;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      timing_.run_maintenance_count++;
      timing_.run_maintenance_total_us += static_cast<uint64_t>(elapsed_us);
    }
    LOG(INFO) << "[FlowTiming] runMaintenance total=" << elapsed_us
              << " us, cells=" << cell_count << ", avg=" << per_cell_us << " us/cell";
  }
}

FlowMap::TimingStats FlowMap::timingStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return timing_;
}

std::vector<FlowMap::StabilityRecord> FlowMap::stabilityRecords() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<StabilityRecord> out;
  out.reserve(cells_.size());
  for (const auto& [_, cell] : cells_) {
    (void)_;
    if (cell.total_observations <= 0) {
      continue;
    }
    out.push_back({cell.total_observations, cell.n_obs_to_stable});
  }
  return out;
}

std::vector<FlowCellFlagSample> FlowMap::cellFlagSamples(double t_seconds,
                                                         double delta_t,
                                                         int stride,
                                                         int max_samples) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<FlowCellFlagSample> out;
  out.reserve(cells_.size());

  const int safe_stride = std::max(1, stride);
  int counter = 0;
  for (const auto& [index, cell] : cells_) {
    ++counter;
    if (counter % safe_stride != 0) {
      continue;
    }

    FlowCellFlagSample sample;
    sample.index = index;
    const auto neighbourhood = neighbourhoodEvidenceLocked(index);
    sample.flag = cell.predict(t_seconds, delta_t, cfg_, &neighbourhood).flag;
    out.push_back(sample);

    if (max_samples > 0 && static_cast<int>(out.size()) >= max_samples) {
      break;
    }
  }

  return out;
}

std::vector<FlowDebugSample> FlowMap::debugSamples(double t_seconds,
                                                   double delta_t,
                                                   int stride,
                                                   int max_samples) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<FlowDebugSample> out;
  out.reserve(cells_.size());

  const int safe_stride = std::max(1, stride);
  int counter = 0;
  for (const auto& [index, cell] : cells_) {
    ++counter;
    if (counter % safe_stride != 0) {
      continue;
    }

    const auto neighbourhood = neighbourhoodEvidenceLocked(index);
    const auto snapshot = cell.snapshot(t_seconds, delta_t, cfg_, &neighbourhood);
    if (snapshot.components.empty()) {
      continue;
    }

    const auto best_iter = std::max_element(snapshot.components.begin(),
                                            snapshot.components.end(),
                                            [](const auto& lhs, const auto& rhs) {
      return lhs.weight < rhs.weight;
    });

    FlowDebugSample sample;
    sample.index = index;
    sample.theta = best_iter->theta;
    sample.rho = best_iter->rho;
    sample.p_present = snapshot.p_present;
    sample.confidence = snapshot.confidence;
    sample.flag = snapshot.flag;
    out.push_back(sample);

    if (max_samples > 0 && static_cast<int>(out.size()) >= max_samples) {
      break;
    }
  }

  return out;
}

std::array<size_t, 3> FlowMap::countFlags(double t_seconds, double delta_t) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::array<size_t, 3> counts = {0, 0, 0};
  for (const auto& [index, cell] : cells_) {
    const auto neighbourhood = neighbourhoodEvidenceLocked(index);
    const auto prediction = cell.predict(t_seconds, delta_t, cfg_, &neighbourhood);
    const auto idx = static_cast<size_t>(prediction.flag);
    if (idx < counts.size()) {
      ++counts[idx];
    }
  }

  return counts;
}

nlohmann::json FlowMap::toJson() const {
  std::lock_guard<std::mutex> lock(mutex_);

  nlohmann::json record;
  record["schema"] = "hydra_flow_map_checkpoint";
  record["version"] = kFlowMapCheckpointVersion;
  record["config"] = {{"nudft_min_obs", cfg_.nudft_min_obs},
                      {"candidate_periods", cfg_.candidate_periods}};
  // Map-wide running mean speed (the cold-start speed prior). Serialized so it
  // resumes exactly across scenes.
  record["global_speed_prior"] = {{"sum", global_speed_prior_->sum},
                                  {"count", global_speed_prior_->count}};
  record["cells"] = nlohmann::json::array();

  for (const auto& [index, cell] : cells_) {
    nlohmann::json cell_record;
    cell_record["index"] = indexToJson(index);

    nlohmann::json components = nlohmann::json::array();
    for (const auto& component : cell.swgmm->exportComponents()) {
      components.push_back(swndToJson(component));
    }
    cell_record["components"] = components;

    cell_record["nudft_models"] = nudftVectorToJson(cell.nudft_models);
    cell_record["nudft_mu_rho_models"] = nudftVectorToJson(cell.nudft_mu_rho_models);
    cell_record["nudft_valid"] = cell.nudft_valid;
    cell_record["nudft_fit_timestamp"] = cell.nudft_fit_timestamp;
    cell_record["t_previous_observation"] = cell.t_previous_observation;

    cell_record["poisson"] = {{"alpha", cell.poisson.alpha},
                              {"beta", cell.poisson.beta},
                              {"n_frames", cell.poisson.n_frames},
                              {"nudft_valid", cell.poisson.nudft_valid},
                              {"nudft", nudftToJson(cell.poisson.nudft)},
                              {"win_t0", cell.poisson.win_t0},
                              {"win_last", cell.poisson.win_last},
                              {"win_events", cell.poisson.win_events},
                              {"win_visible", cell.poisson.win_visible}};

    cell_record["total_observations"] = cell.total_observations;
    cell_record["n_crossings"] = cell.n_crossings;
    // Original-position metadata for stamped deformation. Stored even if
    // unset (original_set=false on cells from older checkpoints).
    cell_record["original_index"] = indexToJson(cell.original_index);
    cell_record["creation_timestamp_ns"] = cell.creation_timestamp_ns;
    cell_record["original_set"] = cell.original_set;
    // Rotation already carried, so a resumed run does not re-rotate state the
    // pre-checkpoint corrections had already turned.
    cell_record["heading_rotation_applied"] = cell.heading_rotation_applied;

    record["cells"].push_back(cell_record);
  }

  return record;
}

bool FlowMap::fromJson(const nlohmann::json& record, std::string* error) {
  if (!record.is_object()) {
    return setError(error, "flow checkpoint root must be a JSON object");
  }

  if (!record.contains("version") || !record.at("version").is_number_integer()) {
    return setError(error, "flow checkpoint missing integer version field");
  }

  const int version = record.at("version").get<int>();
  if (version != kFlowMapCheckpointVersion) {
    std::ostringstream stream;
    stream << "unsupported flow checkpoint version " << version;
    return setError(error, stream.str());
  }

  if (!record.contains("cells") || !record.at("cells").is_array()) {
    return setError(error, "flow checkpoint missing cell array");
  }

  // Reuse the FlowMap's owned shared_omegas_ rather than building a fresh
  // allocation per fromJson() call. All loaded cells share pointer-identity
  // with cells later created via addObservation/ensureCells.
  const auto& shared_omegas = shared_omegas_;

  GlobalIndexMap<FlowCellState> loaded_cells;
  loaded_cells.reserve(record.at("cells").size());

  for (const auto& cell_record : record.at("cells")) {
    if (!cell_record.is_object()) {
      return setError(error, "invalid cell entry in flow checkpoint");
    }

    if (!cell_record.contains("index")) {
      return setError(error, "flow checkpoint cell missing index");
    }

    GlobalIndex index;
    if (!indexFromJson(cell_record.at("index"), index, error)) {
      return false;
    }

    FlowCellState cell(makeModel(), cfg_, shared_omegas_, global_speed_prior_);

    // Default the original to the cell's own KEY so even checkpoints written
    // before this field existed deform from a self-consistent position. This
    // is degraded vs a true creation-time index, but never worse than the
    // pre-fix behaviour.
    cell.original_index = index;

    if (cell_record.contains("nudft_models") &&
        !nudftVectorFromJson(cell_record.at("nudft_models"), cell.nudft_models, error)) {
      return false;
    }

    if (cell_record.contains("nudft_mu_rho_models") &&
        !nudftVectorFromJson(cell_record.at("nudft_mu_rho_models"),
                            cell.nudft_mu_rho_models,
                            error)) {
      return false;
    }

    cell.nudft_valid = cell_record.value("nudft_valid", cell.nudft_valid);
    cell.nudft_fit_timestamp =
        cell_record.value("nudft_fit_timestamp", cell.nudft_fit_timestamp);
    cell.t_previous_observation =
        cell_record.value("t_previous_observation", cell.t_previous_observation);

    if (cell_record.contains("poisson")) {
      const auto& poisson_json = cell_record.at("poisson");
      if (!poisson_json.is_object()) {
        return setError(error, "invalid poisson record in flow checkpoint");
      }

      cell.poisson.alpha = poisson_json.value("alpha", cell.poisson.alpha);
      cell.poisson.n_frames = poisson_json.value("n_frames", cell.poisson.n_frames);
      cell.poisson.beta = poisson_json.value("beta", cell.poisson.beta);
      cell.poisson.nudft_valid =
          poisson_json.value("nudft_valid", cell.poisson.nudft_valid);
      cell.poisson.win_t0 = poisson_json.value("win_t0", cell.poisson.win_t0);
      cell.poisson.win_last = poisson_json.value("win_last", cell.poisson.win_last);
      cell.poisson.win_events =
          poisson_json.value("win_events", cell.poisson.win_events);
      cell.poisson.win_visible =
          poisson_json.value("win_visible", cell.poisson.win_visible);
      if (poisson_json.contains("nudft") &&
          !nudftFromJson(poisson_json.at("nudft"), cell.poisson.nudft, error)) {
        return false;
      }
    }

    cell.total_observations =
        cell_record.value("total_observations", cell.total_observations);
    // Checkpoints written before the crossing counter existed fall back to a
    // weight predictor's sample count, which is the same quantity when no merge
    // has refused a channel.
    const int n_crossings_default =
        cell.nudft_models.empty() ? cell.n_crossings : cell.nudft_models.front().n;
    cell.n_crossings = cell_record.value("n_crossings", n_crossings_default);

    if (cell_record.contains("original_index")) {
      GlobalIndex orig;
      if (!indexFromJson(cell_record.at("original_index"), orig, error)) {
        return false;
      }
      cell.original_index = orig;
    }
    cell.creation_timestamp_ns =
        cell_record.value("creation_timestamp_ns", cell.creation_timestamp_ns);
    cell.original_set = cell_record.value("original_set", cell.original_set);
    cell.heading_rotation_applied = cell_record.value(
        "heading_rotation_applied", cell.heading_rotation_applied);

    configureNudftVector(cell.nudft_models, shared_omegas, cfg_.nudft_min_obs);
    configureNudftVector(cell.nudft_mu_rho_models, shared_omegas, cfg_.nudft_min_obs);
    cell.poisson.nudft.omegas_ptr = shared_omegas;
    cell.poisson.nudft.min_observations = cfg_.nudft_min_obs;
    // Configured, non-serialized: restore the rate window and the order gate.
    cell.poisson.rate_window_s = cfg_.presence_rate_window_s;
    cell.poisson.nudft.gate_enabled = true;
    for (auto& m : cell.nudft_models) {
      m.gate_enabled = true;
    }

    loaded_cells.emplace(index, std::move(cell));
  }

  std::lock_guard<std::mutex> lock(mutex_);
  open_visits_.clear();  // visits are transient; a fresh load starts clean
  cells_ = std::move(loaded_cells);

  // Restore the map-wide running mean speed (data-driven cold-start prior).
  // Prefer the serialized value; for checkpoints written before it existed, fall
  // back to reconstructing it from the per-slot speed models (gamma0 is a slot's
  // mean observed speed, gamma0 * n its sum). All cells share this accumulator.
  if (record.contains("global_speed_prior") &&
      record.at("global_speed_prior").is_object()) {
    global_speed_prior_->sum = record.at("global_speed_prior").value("sum", 0.0);
    global_speed_prior_->count =
        record.at("global_speed_prior").value("count", static_cast<long long>(0));
  } else {
    global_speed_prior_->sum = 0.0;
    global_speed_prior_->count = 0;
    for (const auto& [idx, cell] : cells_) {
      (void)idx;
      for (const auto& m : cell.nudft_mu_rho_models) {
        if (m.n > 0) {
          global_speed_prior_->sum += m.gamma0 * static_cast<double>(m.n);
          global_speed_prior_->count += m.n;
        }
      }
    }
  }
  return true;
}

bool FlowMap::saveToFile(const std::string& filepath) const {
  std::ofstream output(filepath);
  if (!output.good()) {
    return false;
  }

  output << toJson().dump(2);
  return output.good();
}

bool FlowMap::loadFromFile(const std::string& filepath, std::string* error) {
  std::ifstream input(filepath);
  if (!input.good()) {
    return setError(error, "failed to open checkpoint file: " + filepath);
  }

  nlohmann::json record;
  try {
    input >> record;
  } catch (const std::exception& e) {
    return setError(error, std::string("failed to parse checkpoint JSON: ") + e.what());
  }

  return fromJson(record, error);
}

size_t FlowMap::numCells() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cells_.size();
}

FlowMap::StateStats FlowMap::stateStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  StateStats out;
  out.cells = cells_.size();
  for (const auto& [_, cell] : cells_) {
    (void)_;
    out.total_components += cell.swgmm ? static_cast<size_t>(cell.swgmm->numComponents()) : 0;
    if (cell.nudft_valid) {
      ++out.cells_with_valid_nudft;
    }
    for (const auto& m : cell.nudft_models) {
      out.total_pi_nudft_obs += static_cast<size_t>(m.n);
    }
  }
  return out;
}

std::unique_ptr<SWGMMBase> FlowMap::makeModel() const {
  return std::make_unique<FixedComponents>(cfg_);
}

}  // namespace kairos
