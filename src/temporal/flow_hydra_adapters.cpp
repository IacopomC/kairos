#include "kairos/temporal/flow_hydra_adapters.h"

#include <config_utilities/factory.h>

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/shared_dsg_info.h>

#include "kairos/temporal/flow_module_registry.h"
#include "kairos/temporal/flow_temporal_module.h"

namespace kairos {

namespace {

const auto sink_registration =
    config::RegistrationWithConfig<hydra::ActiveWindowModule::Sink,
                                   FlowMapSink,
                                   FlowMapSink::Config>("FlowMapSink");

const auto functor_registration =
    config::RegistrationWithConfig<hydra::UpdateFunctor,
                                   FlowUpdateFunctor,
                                   FlowUpdateFunctor::Config>("FlowUpdateFunctor");

}  // namespace

void declare_config(FlowMapSink::Config& config) {
  using namespace config;
  name("FlowMapSink::Config");
  field(config.module, "module");
}

FlowMapSink::FlowMapSink(const Config& config) : config(config::checkValid(config)) {}

void FlowMapSink::call(uint64_t timestamp_ns,
                       const hydra::VolumetricMap& map,
                       const hydra::ActiveWindowOutput& output) const {
  auto module = FlowModuleRegistry::get(config.module);
  if (module) {
    module->handleMapUpdate(timestamp_ns, map, output);
  }
}

std::string FlowMapSink::printInfo() const { return config::toString(config); }

void declare_config(FlowUpdateFunctor::Config& config) {
  using namespace config;
  name("FlowUpdateFunctor::Config");
  field(config.module, "module");
}

FlowUpdateFunctor::FlowUpdateFunctor(const Config& config)
    : config(config::checkValid(config)) {}

void FlowUpdateFunctor::call(const spark_dsg::DynamicSceneGraph&,
                             hydra::SharedDsgInfo& dsg,
                             const hydra::UpdateInfo::ConstPtr& info) const {
  auto module = FlowModuleRegistry::get(config.module);
  if (!module) {
    return;
  }

  if (!info->current_deformation_graph) {
    LOG_FIRST_N(WARNING, 1) << "[FlowUpdateFunctor] backend update without a "
                               "deformation graph, skipping flow update";
    return;
  }

  module->handleBackendUpdate(info->timestamp_ns, *dsg.graph, *info->current_deformation_graph);
}

}  // namespace kairos
