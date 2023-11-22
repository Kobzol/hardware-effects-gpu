cmake_minimum_required(VERSION 3.0)
project(hardware-effects-gpu)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")

set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
enable_language(CUDA)

add_subdirectory(bank-conflicts)
add_subdirectory(memory-coalescing)
add_subdirectory(shared-memory-limit)
