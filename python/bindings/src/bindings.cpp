/* -----------------------------------------------------------------------------
 * KAIROS python bindings entry point (_kairos_bindings).
 *
 * The pipeline, sensors and logging come from Hydra's own bindings
 * (hydra_python); this module adds the flow-module handle and helpers. Loading it
 * also registers the KAIROS modules, sinks and functors with Hydra's factories,
 * since they live in libkairos.
 * -------------------------------------------------------------------------- */
#include <pybind11/pybind11.h>

#include "kairos/bindings/bow_producer.h"
#include "kairos/bindings/flow_module.h"

namespace py = pybind11;

PYBIND11_MODULE(_kairos_bindings, m) {
  py::module_::import("spark_dsg");
  ::kairos::python::flow_module::addBindings(m);
  ::kairos::python::bow_producer::addBindings(m);
}
