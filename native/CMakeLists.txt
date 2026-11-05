# Portable core: headers and CPU-testable code, no CUDA headers (spec C§3, C§4).
add_library(motionman_core INTERFACE)
target_include_directories(motionman_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(motionman_core INTERFACE cxx_std_20)

if(MOTIONMAN_CUDA)
  # NVIDIA-only targets. A missing toolkit or TensorRT is a configure error, never a silent CPU build (C§3).
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ENV TENSORRT_ROOT PATH_SUFFIXES include)
  find_library(TENSORRT_NVINFER nvinfer HINTS ENV TENSORRT_ROOT PATH_SUFFIXES lib lib64)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR "MOTIONMAN_CUDA=ON but TensorRT was not found (NvInfer.h / libnvinfer). "
                        "Install it or set TENSORRT_ROOT; see docs/environment.md.")
  endif()
  message(STATUS "CUDA ${CUDAToolkit_VERSION}, TensorRT headers in ${TENSORRT_INCLUDE_DIR}")
endif()

add_subdirectory(tests)
