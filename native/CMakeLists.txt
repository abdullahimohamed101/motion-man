# Portable core: headers and CPU-testable code, no CUDA headers (spec C§3, C§4).
add_library(motionman_core INTERFACE)
target_include_directories(motionman_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(motionman_core INTERFACE cxx_std_20)

# Portable CPU code (no OpenCV, no CUDA): preprocessing shared by the engine and its tests.
add_library(motionman_cpu STATIC src/preprocess.cpp src/json.cpp src/sha256.cpp src/track/lapjv.cpp
                                 src/track/kalman.cpp src/track/bytetrack.cpp src/records.cpp)
target_link_libraries(motionman_cpu PUBLIC motionman_core)
target_compile_options(motionman_cpu PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
# The tracker reproduces the reference's float32 box arithmetic (ADR-006): no fused multiply-add contraction.
set_source_files_properties(src/track/lapjv.cpp src/track/kalman.cpp src/track/bytetrack.cpp PROPERTIES COMPILE_OPTIONS
                            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)

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

# OpenCV decodes images and video on the PC; the portable core never needs it.
find_package(OpenCV QUIET COMPONENTS core imgcodecs)
if(MOTIONMAN_CUDA)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "MOTIONMAN_CUDA=ON but OpenCV (core, imgcodecs) was not found; see docs/environment.md.")
  endif()
  # GPU detector (TensorRT + CUDA RAII) and the motionman-native executable (spec C§3, C§7).
  add_library(motionman_gpu STATIC src/gpu/trt_detector.cpp)
  target_include_directories(motionman_gpu PUBLIC src ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(motionman_gpu PUBLIC motionman_cpu CUDA::cudart CUDA::nvml ${TENSORRT_NVINFER})
  target_compile_options(motionman_gpu PRIVATE -Wall -Wextra -Werror)
  add_executable(motionman-native src/app/main.cpp)
  target_link_libraries(motionman-native PRIVATE motionman_gpu ${OpenCV_LIBS})
  target_include_directories(motionman-native PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_options(motionman-native PRIVATE -Wall -Wextra -Werror)
endif()

add_subdirectory(tests)
