#pragma once

#include <string>

namespace kairos {

class FlowTemporalModule;

/**
 * @brief Running flow modules by name.
 *
 * Hydra creates the flow module, its map sink and its backend functor from config,
 * each on its own; the sink and functor find the module here. A module registers
 * itself on construction and removes itself on destruction. Lookups return the
 * module running under a name, or nullptr if none is.
 */
struct FlowModuleRegistry {
  static void add(const std::string& name, FlowTemporalModule* module);
  //! Removes the entry only if it still points to the given module
  static void remove(const std::string& name, const FlowTemporalModule* module);
  static FlowTemporalModule* get(const std::string& name);
};

}  // namespace kairos
