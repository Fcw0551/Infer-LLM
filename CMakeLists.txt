cmake_minimum_required(VERSION 3.16)
project(InferLLM LANGUAGES CXX)

# 设置变量
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 核心库（src 下的所有 .cpp，除了 main.cpp）
add_library(inferllm STATIC
    src/gguf_parser.cpp
    src/util.cpp
    src/model.cpp
)

target_include_directories(inferllm PUBLIC include)
target_compile_options(inferllm PRIVATE -Wall -Wextra)

# 主程序
add_executable(inferllm_main src/main.cpp)
target_link_libraries(inferllm_main PRIVATE inferllm)

# # 测试程序
# add_executable(test_mmap test/test_mmap.cpp)
# target_link_libraries(test_mmap PRIVATE inferllm)
