#pragma once

#include <config_utilities/factory.h>
#include <hydra/active_window/active_window_module.h>
#include <hydra/backend/update_functions.h>

#include <string>

namespace kairos {

/**
 * @brief Active-window sink that feeds map updates to a flow module.
 *
 * Add it to the active window's `sinks`; the module is found by name.
 */
struct FlowMapSink : public hydra::ActiveWindowModule::Sink {
  struct Config {
    //! Name of the flow module to feed
    std::string module = "flow_temporal";
  } const config;

  explicit FlowMapSink(const Config& config);

  void call(uint64_t timestamp_ns,
            const hydra::VolumetricMap& map,
            const hydra::ActiveWindowOutput& output) const override;

  std::string printInfo() const override;

};

void declare_config(FlowMapSink::Config& config);

/**
 * @brief Backend update functor that runs a flow module's backend update.
 *
 * The flow module writes its per-place summaries into the backend graph here, so
 * functors listed after this one (e.g., archetypes, rooms) see them in the same
 * update.
 */
struct FlowUpdateFunctor : public hydra::UpdateFunctor {
  struct Config {
    //! Name of the flow module to update
    std::string module = "flow_temporal";
  } const config;

  explicit FlowUpdateFunctor(const Config& config);

  void call(const spark_dsg::DynamicSceneGraph& unmerged,
            hydra::SharedDsgInfo& dsg,
            const hydra::UpdateInfo::ConstPtr& info) const override;

};

void declare_config(FlowUpdateFunctor::Config& config);

}  // namespace kairos
