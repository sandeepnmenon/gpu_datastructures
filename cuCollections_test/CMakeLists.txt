cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(static_map_test LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES "60") # or higher, according to your GPU

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
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

add_executable(host_bulk_example host_bulk_example.cu)
target_link_libraries(host_bulk_example cuco)
