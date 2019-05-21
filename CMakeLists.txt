cmake_minimum_required(VERSION 3.0)
project(hardware-effects-gpu)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

add_subdirectory(bank-conflicts)
add_subdirectory(memory-coalescing)
