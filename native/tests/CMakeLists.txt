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

add_executable(test_json_sha256 unit/test_json_sha256.cpp)
target_link_libraries(test_json_sha256 PRIVATE motionman_cpu)
target_compile_options(test_json_sha256 PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME json_sha256 COMMAND test_json_sha256 ${PROJECT_SOURCE_DIR}/configs/native/rtdetr_r18_640.json)
set_tests_properties(json_sha256 PROPERTIES LABELS unit)

add_executable(test_lapjv unit/test_lapjv.cpp)
target_link_libraries(test_lapjv PRIVATE motionman_cpu)
target_compile_options(test_lapjv PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME lapjv COMMAND test_lapjv ${PROJECT_SOURCE_DIR}/tests/fixtures/lapjv.json)
set_tests_properties(lapjv PROPERTIES LABELS unit)

add_executable(test_tracker_replay unit/test_tracker_replay.cpp)
target_link_libraries(test_tracker_replay PRIVATE motionman_cpu)
target_compile_options(test_tracker_replay PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME tracker_replay COMMAND test_tracker_replay ${PROJECT_SOURCE_DIR}/tests/fixtures/tracker)
set_tests_properties(tracker_replay PROPERTIES LABELS unit)

add_executable(test_native_records unit/test_native_records.cpp)
target_link_libraries(test_native_records PRIVATE motionman_cpu)
target_compile_options(test_native_records PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Werror>)
add_test(NAME native_records COMMAND test_native_records ${PROJECT_SOURCE_DIR}/tests/fixtures/native_records.json)
set_tests_properties(native_records PROPERTIES LABELS unit)

# Golden-tensor check against the Python reference: needs OpenCV (PNG decode, found in native/) and the PC's cache.
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

  # G1 parity (NG-10): C++ preprocessing + TensorRT FP32 on the golden images vs the PyTorch reference R0.
  set(MOTIONMAN_PYTHON "$ENV{HOME}/.cache/motionman/venv/bin/python" CACHE FILEPATH "Export venv with parity.py deps")
  set(mm_profile ${PROJECT_SOURCE_DIR}/configs/native/rtdetr_r18_640.json)
  set(mm_model ${MOTIONMAN_CACHE}/models/rtdetr_r18_640)
  add_test(NAME trt_infer COMMAND motionman-native infer --profile ${mm_profile} --model-dir ${mm_model}
                                  --images ${MOTIONMAN_CACHE}/golden --out ${mm_model}/trt_fp32.json)
  set_tests_properties(trt_infer PROPERTIES LABELS gpu FIXTURES_SETUP trt_fp32)
  add_test(NAME trt_parity COMMAND ${MOTIONMAN_PYTHON} ${PROJECT_SOURCE_DIR}/tools/model_export/parity.py rtdetr_r18_640
                                   --ref r0 --cand trt_fp32 --gate fp32)
  set_tests_properties(trt_parity PROPERTIES LABELS gpu FIXTURES_REQUIRED trt_fp32)
  add_test(NAME trt_refuse_tampered
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/gpu/refuse_tampered.sh $<TARGET_FILE:motionman-native>
                   ${MOTIONMAN_PYTHON} ${mm_profile} ${mm_model} ${MOTIONMAN_CACHE}/golden)
  set_tests_properties(trt_refuse_tampered PROPERTIES LABELS gpu)
endif()
