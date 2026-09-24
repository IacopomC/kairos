# KAIROS unit tests. Configure with -DKAIROS_ENABLE_TESTS=ON, then `ctest`.
# Every flow/test_*.cpp is globbed into one test_kairos executable so a new
# test file is picked up automatically (no manual registration step to forget).
find_package(GTest REQUIRED)
include(GoogleTest)

file(GLOB KAIROS_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/flow/test_*.cpp")

add_executable(test_kairos ${KAIROS_TEST_SOURCES})
target_link_libraries(test_kairos PRIVATE kairos GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_kairos)
