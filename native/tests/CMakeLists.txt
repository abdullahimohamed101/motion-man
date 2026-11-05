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
