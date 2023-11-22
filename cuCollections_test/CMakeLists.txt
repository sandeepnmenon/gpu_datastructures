cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(static_map_test LANGUAGES CXX CUDA)
include(cmake/CPM.cmake)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

add_executable(static_map_test static_map_test.cpp)
target_link_libraries(static_map_test cuco)