cmake_minimum_required(VERSION 3.16)
project(InferLLM LANGUAGES CXX)

# 设置变量
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 核心库（src 下的所有 .cpp，除了 main.cpp）
add_library(inferllm STATIC
    src/models/Qwen3.cpp
    src/gguf_parser.cpp
    src/graph.cpp
    src/log.cpp
    src/model.cpp
    src/op_factory.cpp
    src/util.cpp
)

target_include_directories(inferllm PUBLIC include)
target_compile_options(inferllm PRIVATE -Wall -Wextra)

# dot 图的输出目录：和 include/ src/ 同级的 models_dot/（绝对路径，所以在哪个目录跑都一样）
target_compile_definitions(inferllm PRIVATE
    INFERLLM_MODELS_DOT_DIR="${CMAKE_SOURCE_DIR}/models_dot")

# 主程序
add_executable(inferllm_main src/main.cpp)
target_link_libraries(inferllm_main PRIVATE inferllm)

# 测试程序
add_executable(test_qwen3 test/test_qwen3_build_graph.cpp)
target_link_libraries(test_qwen3 PRIVATE inferllm)
