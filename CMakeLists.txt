cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(hardware-effects-gpu LANGUAGES CUDA CXX)

enable_language(CUDA)

add_subdirectory(bank-conflicts)
add_subdirectory(memory-coalescing)
add_subdirectory(shared-memory-limit)
