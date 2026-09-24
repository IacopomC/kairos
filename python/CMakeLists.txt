cmake_minimum_required(VERSION 3.16)
project(kairos_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer the pip-installed pybind11 (modern; matches the API the bindings use)
# over any old system apt package.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_rc)
if(_pybind11_rc EQUAL 0)
  find_package(pybind11 REQUIRED PATHS "${_pybind11_cmake_dir}" NO_DEFAULT_PATH)
else()
  find_package(pybind11 REQUIRED)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# When built standalone (via setup.py) find the catkin-installed libs; when
# added from kairos/CMakeLists the targets already exist.
if(PROJECT_IS_TOP_LEVEL)
  find_package(hydra REQUIRED)
  find_package(kairos REQUIRED)
endif()

# Hydra's generic pybind helpers + base pipeline live in the hydra-flow source
# tree (they are Hydra's code and stay there). KAIROS compiles them into its
# OWN self-contained bindings module so `kairos run` never imports
# _hydra_bindings. Override HYDRA_FLOW_DIR if the layout differs.
set(HYDRA_FLOW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hydra-kairos"
    CACHE PATH "Path to the hydra-flow source tree")
set(_hb_src "${HYDRA_FLOW_DIR}/python/bindings/src")
set(_hb_inc "${HYDRA_FLOW_DIR}/python/bindings/include")
if(NOT EXISTS "${_hb_src}/python_pipeline.cpp")
  message(FATAL_ERROR
    "Could not find Hydra binding sources at ${_hb_src}. "
    "Set -DHYDRA_FLOW_DIR=/path/to/hydra-flow.")
endif()

pybind11_add_module(_kairos_bindings
  bindings/src/bindings.cpp
  bindings/src/kairos_pipeline.cpp
  ${_hb_src}/python_pipeline.cpp
  ${_hb_src}/glog_utilities.cpp
  ${_hb_src}/python_image.cpp
  ${_hb_src}/python_sensor_input.cpp
  ${_hb_src}/python_sensors.cpp
  ${_hb_src}/color_parser.cpp
  ${_hb_src}/depth_parser.cpp
  ${_hb_src}/label_parser.cpp
)
target_include_directories(_kairos_bindings PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/include
  ${_hb_inc}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(_kairos_bindings PRIVATE
  kairos::kairos
  hydra::hydra
  ${OpenCV_LIBRARIES}
)
