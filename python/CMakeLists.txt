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

# When built standalone (via setup.py) find the installed libs; when added from
# kairos/CMakeLists the targets already exist.
if(PROJECT_IS_TOP_LEVEL)
  find_package(hydra REQUIRED)
  find_package(kairos REQUIRED)
endif()

# DBoW3 powers the offline bag-of-words producer (bindings/src/bow_producer.cpp).
include(FetchContent)
FetchContent_Declare(
  dbow3
  GIT_REPOSITORY https://github.com/rmsalinas/DBow3.git
  GIT_TAG 1cc587bbfc08c0adb0ea2d00d81ad6cd4103ce39
)
# Skip DBoW3's bundled demo/utility executables.
set(BUILD_UTILS OFF CACHE BOOL "" FORCE)
# DBoW3's CMakeLists declares cmake_minimum_required(VERSION 2.6), which
# CMake 3.27+ refuses. CMAKE_POLICY_VERSION_MINIMUM tells CMake to apply
# the policies of 3.5 even when the included project asked for 2.6.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
# Build DBoW3 static, linked into _kairos_bindings, so the module carries no
# separate shared library to ship. CMP0077 lets this BUILD_SHARED_LIBS override
# DBoW3's own option() default.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(_kairos_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(dbow3)
set(BUILD_SHARED_LIBS ${_kairos_shared_libs})
unset(_kairos_shared_libs)
if(TARGET DBoW3)
  target_compile_options(DBoW3 PRIVATE -w)  # silence third-party warnings
  set_target_properties(DBoW3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ORB lives in opencv_features2d
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d)
find_package(pose_graph_tools REQUIRED)

pybind11_add_module(_kairos_bindings
  bindings/src/bindings.cpp
  bindings/src/bow_producer.cpp
  bindings/src/flow_module.cpp
)
target_include_directories(_kairos_bindings PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/include
  ${dbow3_SOURCE_DIR}/src  # DBoW3.h umbrella header
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(_kairos_bindings PRIVATE
  kairos::kairos
  hydra::hydra
  pose_graph_tools::pose_graph_tools
  DBoW3
  ${OpenCV_LIBRARIES}
)
