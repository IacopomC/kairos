/* -----------------------------------------------------------------------------
 * KAIROS python bindings entry point (_kairos_bindings).
 *
 * Self-contained: registers Hydra's generic helper types (glog, image, sensor
 * input/sensors) and the base HydraPipeline, then KAIROS's KairosPipeline.
 * `kairos run` therefore depends only on this module, not on _hydra_bindings.
 * -------------------------------------------------------------------------- */
#include <pybind11/pybind11.h>

#include <Eigen/Geometry>
#include <sstream>

#include "hydra/bindings/glog_utilities.h"
#include "hydra/bindings/python_image.h"
#include "hydra/bindings/python_pipeline.h"
#include "hydra/bindings/python_sensor_input.h"
#include "hydra/bindings/python_sensors.h"
#include "kairos/bindings/kairos_pipeline.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_kairos_bindings, m) {
  py::module_::import("spark_dsg");
  py::options options;

  // Generic Hydra pybind infrastructure (compiled into this module from the
  // hydra-flow source tree).
  ::hydra::python::glog_utilities::addBindings(m);
  ::hydra::python::python_image::addBindings(m);
  ::hydra::python::python_sensor_input::addBindings(m);
  ::hydra::python::python_sensors::addBindings(m);
  ::hydra::python::python_pipeline::addBindings(m);  // base "HydraPipeline"

  // KAIROS contribution.
  ::kairos::python::kairos_pipeline::addBindings(m);  // "KairosPipeline"

  py::class_<Eigen::Quaterniond>(m, "Quaterniond")
      .def(py::init([]() { return Eigen::Quaterniond::Identity(); }))
      .def(py::init([](double w, double x, double y, double z) {
             return Eigen::Quaterniond(w, x, y, z);
           }),
           "w"_a,
           "x"_a,
           "y"_a,
           "z"_a)
      .def("__repr__", [](const Eigen::Quaterniond& q) {
        std::stringstream ss;
        ss << "Quaterniond<w=" << q.w() << ", x=" << q.x() << ", y=" << q.y()
           << ", z=" << q.z() << ">";
        return ss.str();
      });
}
