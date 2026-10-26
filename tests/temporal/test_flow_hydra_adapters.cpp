#include <config_utilities/virtual_config.h>
#include <gtest/gtest.h>
#include <hydra/common/module.h>
#include <kairos/temporal/flow_hydra_adapters.h>
#include <kairos/temporal/flow_module_registry.h>
#include <kairos/temporal/flow_temporal_module.h>

namespace kairos {

TEST(FlowModuleRegistry, ConfiguredModuleIsFoundByName) {
  FlowTemporalModule::Config config;
  config.name = "test_flow";

  {
    // created the way a pipeline creates configured modules
    const auto module = config::VirtualConfig<hydra::Module>(config).create();
    ASSERT_TRUE(module);
    EXPECT_EQ(FlowModuleRegistry::get("test_flow"), module.get());
    EXPECT_EQ(FlowModuleRegistry::get("other"), nullptr);
  }

  // a destroyed module is no longer found
  EXPECT_EQ(FlowModuleRegistry::get("test_flow"), nullptr);
}

TEST(FlowModuleRegistry, AdaptersAreRegistered) {
  // the sink and functor can be created by name from config
  const auto sink =
      config::VirtualConfig<hydra::ActiveWindowModule::Sink, true>(FlowMapSink::Config{})
          .create();
  EXPECT_TRUE(sink);

  const auto functor =
      config::VirtualConfig<hydra::UpdateFunctor, true>(FlowUpdateFunctor::Config{})
          .create();
  EXPECT_TRUE(functor);

  // with no module running, the functor does nothing
  hydra::SharedDsgInfo dsg({{{spark_dsg::DsgLayers::PLACES, 3}}});
  auto info = std::make_shared<hydra::UpdateInfo>();
  functor->call(*dsg.graph, dsg, info);
}

}  // namespace kairos
