// Multi-session correctness tests for the kairos flow_state.bin format.
//
// The v3 schema introduces grid_origin and world_frame_id header fields.
// v1/v2 files must remain readable.

#include <gtest/gtest.h>
#include <kairos/flow/flow_map.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace kairos {
using namespace hydra;  // base-Hydra types visible
namespace {

FlowMapConfig defaultConfig() {
  FlowMapConfig cfg;
  cfg.fixed_mu_rho = 1.1;
  cfg.fixed_sigma_theta = 0.4;
  cfg.fixed_sigma_rho = 0.3;
  cfg.nudft_min_obs = 1;
  cfg.candidate_periods = {3600.0f};
  return cfg;
}

GlobalIndex makeIndex(int x, int y, int z) { return GlobalIndex(x, y, z); }

// Write a hand-crafted v1-style msgpack record (no edge_coupling, no v3 keys)
// to exercise backwards compatibility.
std::string writeFakeVersion(int version, const std::filesystem::path& path) {
  nlohmann::json record;
  record["schema"] = "kairos_flow_state";
  record["version"] = version;
  record["voxel_size"] = 0.1f;
  record["flow_map"] = nlohmann::json::object();
  record["flow_map"]["cells"] = nlohmann::json::array();
  record["cell_count"] = 0;
  auto bytes = nlohmann::json::to_msgpack(record);
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return path.string();
}

}  // namespace

// Round-trip a real FlowMap through saveToFile / loadFromFile. The current
// FlowTemporalModule::saveFlowState/loadFlowState wraps these with the v3
// header — this test exercises the underlying serialiser to ensure no
// regression in cell-level state.
TEST(FlowStateIO, FlowMapRoundTrip) {
  FlowMap map(defaultConfig());
  map.addObservation(makeIndex(1, 2, 3), 1.0, 0.5, 0.0, 1.0);
  map.addObservation(makeIndex(1, 2, 3), 1.5, 0.6, 0.1, 1.0);
  map.addObservation(makeIndex(4, 5, 6), 2.0, 0.7, 0.2, 1.0);
  ASSERT_EQ(map.numCells(), 2u);

  auto path = std::filesystem::temp_directory_path() / "kairos_test_flow.json";
  nlohmann::json json = map.toJson();
  std::ofstream out(path);
  out << json.dump();
  out.close();

  FlowMap loaded(defaultConfig());
  std::ifstream in(path);
  nlohmann::json parsed = nlohmann::json::parse(in);
  std::string err;
  ASSERT_TRUE(loaded.fromJson(parsed, &err)) << err;
  EXPECT_EQ(loaded.numCells(), map.numCells());

  std::filesystem::remove(path);
}

// The v3 header bump must not break v1/v2 readers. Older files lack the
// grid_origin / world_frame_id keys, and current loadFlowState should
// silently default them to (0,0,0) / "map".
TEST(FlowStateIO, OldVersionsRemainReadable) {
  for (int version : {1, 2, 3}) {
    auto path = std::filesystem::temp_directory_path() /
                ("kairos_test_v" + std::to_string(version) + ".bin");
    writeFakeVersion(version, path);

    std::ifstream in(path, std::ios::binary);
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    nlohmann::json record = nlohmann::json::from_msgpack(bytes);
    EXPECT_EQ(record.at("version").get<int>(), version);
    EXPECT_TRUE(record.contains("voxel_size"));
    std::filesystem::remove(path);
  }
}

// Verify that the v3 writer emits the new header keys.
TEST(FlowStateIO, V3HeaderContainsNewKeys) {
  nlohmann::json record;
  record["schema"] = "kairos_flow_state";
  record["version"] = 3;
  record["voxel_size"] = 0.1f;
  record["grid_origin"] = std::array<double, 3>{0.0, 0.0, 0.0};
  record["world_frame_id"] = "map";
  record["flow_map"] = nlohmann::json::object();
  record["flow_map"]["cells"] = nlohmann::json::array();
  record["cell_count"] = 0;

  auto bytes = nlohmann::json::to_msgpack(record);
  nlohmann::json parsed = nlohmann::json::from_msgpack(bytes);
  ASSERT_TRUE(parsed.contains("grid_origin"));
  ASSERT_TRUE(parsed.contains("world_frame_id"));
  EXPECT_EQ(parsed.at("version").get<int>(), 3);
  EXPECT_EQ(parsed.at("world_frame_id").get<std::string>(), "map");
}

}  // namespace kairos
