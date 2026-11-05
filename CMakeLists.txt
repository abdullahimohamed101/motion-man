# MotionMan native engine (spec C§4, plan native-g0-g1 step 1). The Python app does not use CMake.
cmake_minimum_required(VERSION 3.24)
project(motionman_native VERSION 0.1.0 LANGUAGES CXX)

option(MOTIONMAN_CUDA "Build the CUDA/TensorRT targets (NVIDIA PC only; fails if CUDA or TensorRT is missing)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
add_subdirectory(native)
