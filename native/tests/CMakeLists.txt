# Portable unit tests: no GPU, network or weights (INV-8, plan D4: no test framework download).
add_executable(test_types unit/test_types.cpp)
target_link_libraries(test_types PRIVATE motionman_core)
target_compile_options(test_types PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME types COMMAND test_types)
set_tests_properties(types PROPERTIES LABELS unit)

add_executable(test_source_time unit/test_source_time.cpp)
target_link_libraries(test_source_time PRIVATE motionman_core)
target_compile_options(test_source_time PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME source_time COMMAND test_source_time ${PROJECT_SOURCE_DIR}/tests/fixtures/source_time.json)
set_tests_properties(source_time PROPERTIES LABELS unit)

add_executable(test_preprocess unit/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE motionman_cpu)
target_compile_options(test_preprocess PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME preprocess COMMAND test_preprocess)
set_tests_properties(preprocess PROPERTIES LABELS unit)

# Golden-tensor check against the Python reference: needs OpenCV (PNG decode) and the PC's cached tensors.
find_package(OpenCV QUIET COMPONENTS core imgcodecs)
set(MOTIONMAN_CACHE "$ENV{HOME}/.cache/motionman" CACHE PATH "PC cache with golden PNGs and model artifacts")
if(OpenCV_FOUND)
  add_executable(test_preprocess_golden golden/test_preprocess_golden.cpp)
  target_link_libraries(test_preprocess_golden PRIVATE motionman_cpu ${OpenCV_LIBS})
  target_include_directories(test_preprocess_golden PRIVATE ${OpenCV_INCLUDE_DIRS})
  add_test(NAME preprocess_golden
           COMMAND test_preprocess_golden ${MOTIONMAN_CACHE}/golden
                   ${MOTIONMAN_CACHE}/models/rtdetr_r18_640/golden_tensors)
  set_tests_properties(preprocess_golden PROPERTIES LABELS golden)
endif()

if(MOTIONMAN_CUDA)
  # Real GPU tests: label gpu, run only on the NVIDIA PC (INV-8 keeps them out of pytest and CI).
  add_executable(test_cuda_smoke gpu/cuda_smoke.cu)
  target_link_libraries(test_cuda_smoke PRIVATE motionman_core CUDA::cudart)
  add_test(NAME cuda_smoke COMMAND test_cuda_smoke)
  set_tests_properties(cuda_smoke PROPERTIES LABELS gpu)
endif()
