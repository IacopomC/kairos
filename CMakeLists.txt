cmake_minimum_required(VERSION 3.16)
project(kairos VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libs" ON)
option(KAIROS_ENABLE_TESTS "Build KAIROS unit tests" OFF)

# KAIROS builds on top of Hydra (as a library) but is its own package and
# namespace (kairos::). It never modifies Hydra's source.
find_package(hydra REQUIRED)
find_package(config_utilities REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(kimera_pgmo REQUIRED)
find_package(spark_dsg REQUIRED)
find_package(spatial_hash REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(kairos
  src/flow/cell_state.cpp
  src/flow/config.cpp
  src/flow/edge_coupling_map.cpp
  src/flow/flow_map.cpp
  src/flow/nudft.cpp
  src/flow/poisson_model.cpp
  src/flow/swgmm.cpp
  src/flow/swnd.cpp
  src/temporal/flow_temporal_module.cpp
)
add_library(kairos::kairos ALIAS kairos)

target_include_directories(kairos PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(kairos PRIVATE ${PCL_INCLUDE_DIRS})

target_link_libraries(kairos
  PUBLIC
    hydra::hydra
    config_utilities::config_utilities
    Eigen3::Eigen
    kimera_pgmo::kimera_pgmo
    spark_dsg::spark_dsg
    spatial_hash::spatial_hash
    nlohmann_json::nlohmann_json
  PRIVATE
    ${PCL_LIBRARIES}
)
set_property(TARGET kairos PROPERTY POSITION_INDEPENDENT_CODE ON)

if(KAIROS_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- install + export so downstream (kairos_ros, python bindings) can
# ---- find_package(kairos) ----
install(TARGETS kairos
  EXPORT kairos-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kairos-targets
  FILE kairosTargets.cmake
  NAMESPACE kairos::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kairos
)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/kairosConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/kairosConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kairos
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/kairosConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/kairosConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/kairosConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kairos
)

# install config + docs alongside the package (referenced by the python CLI)
install(DIRECTORY config/ DESTINATION ${CMAKE_INSTALL_DATADIR}/kairos/config
        OPTIONAL)
