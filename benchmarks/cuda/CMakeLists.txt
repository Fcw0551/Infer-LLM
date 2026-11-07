cmake_minimum_required(VERSION 3.18)
project(InferLLMCUDABench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(INFERLLM_WITH_CUDA ON CACHE BOOL "CUDA benchmark" FORCE)
set(INFERLLM_WITH_METAL OFF CACHE BOOL "CUDA benchmark" FORCE)
get_filename_component(INFERLLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
execute_process(COMMAND git -C "${INFERLLM_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE BENCH_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE BENCH_GIT_STATUS)
if(NOT BENCH_GIT_STATUS EQUAL 0)
    set(BENCH_GIT_COMMIT "unknown")
endif()
# 复用引擎目标；不编译 src/main.cpp，不修改项目根构建文件。
add_subdirectory("${INFERLLM_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/engine" EXCLUDE_FROM_ALL)
find_package(Threads REQUIRED)
add_executable(inferllm-bench bench.cpp)
target_link_libraries(inferllm-bench PRIVATE inferllm Threads::Threads)
# 当前 Linux/WSL CPU-only 链接需要显式带上 pthread。
target_compile_options(inferllm PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-pthread>")
target_link_options(inferllm-bench PRIVATE -pthread)
target_compile_options(inferllm-bench PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(inferllm-bench PRIVATE
    BENCH_BUILD_TYPE="$<CONFIG>"
    BENCH_GIT_COMMIT="${BENCH_GIT_COMMIT}"
    BENCH_DEFAULT_MODEL="${INFERLLM_ROOT}/models/Qwen3-0.6B-F32.gguf")
enable_testing()
add_test(NAME bench_help COMMAND inferllm-bench --help)
add_test(NAME bench_bad_threads COMMAND inferllm-bench --threads -1)
add_test(NAME bench_bad_mode COMMAND inferllm-bench --mode invalid)
add_test(NAME bench_missing_value COMMAND inferllm-bench --model)
add_test(NAME bench_bad_profile COMMAND inferllm-bench --profile unknown)
add_test(NAME bench_profile_conflict COMMAND inferllm-bench --profile p32 -p 32)
set_tests_properties(bench_bad_threads bench_bad_mode bench_missing_value bench_bad_profile bench_profile_conflict PROPERTIES WILL_FAIL TRUE)
