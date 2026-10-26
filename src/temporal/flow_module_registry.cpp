#include "kairos/temporal/flow_module_registry.h"

#include <map>
#include <mutex>

namespace kairos {

namespace {

std::mutex& registryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, FlowTemporalModule*>& registry() {
  static std::map<std::string, FlowTemporalModule*> modules;
  return modules;
}

}  // namespace

void FlowModuleRegistry::add(const std::string& name, FlowTemporalModule* module) {
  std::lock_guard<std::mutex> lock(registryMutex());
  registry()[name] = module;
}

void FlowModuleRegistry::remove(const std::string& name,
                                const FlowTemporalModule* module) {
  std::lock_guard<std::mutex> lock(registryMutex());
  auto iter = registry().find(name);
  if (iter != registry().end() && iter->second == module) {
    registry().erase(iter);
  }
}

FlowTemporalModule* FlowModuleRegistry::get(const std::string& name) {
  std::lock_guard<std::mutex> lock(registryMutex());
  auto iter = registry().find(name);
  return iter == registry().end() ? nullptr : iter->second;
}

}  // namespace kairos
