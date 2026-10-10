cmake_minimum_required(VERSION 3.18)
project(InferLLM LANGUAGES CXX)

# 设置变量
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 启用不同后端设备
option(INFERLLM_WITH_CUDA "Build with CUDA backend" OFF)
option(INFERLLM_WITH_METAL "Build with Metal backend" OFF)


# 核心库（src 下的所有 .cpp，除了 main.cpp）
add_library(inferllm STATIC
    src/backends/cpu_backend.cpp
    src/models/Qwen3.cpp
    src/backend.cpp
    src/gguf_parser.cpp
    src/graph.cpp
    src/log.cpp
    src/model.cpp
    src/op_factory.cpp
    src/util.cpp
    
)

target_include_directories(inferllm PUBLIC include)
target_compile_options(inferllm PRIVATE -Wall -Wextra)


if(INFERLLM_WITH_CUDA)
    # 查找 CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)

    # 把 CUDA 的源文件加入编译
    target_sources(inferllm PRIVATE
        src/backends/cuda_backend.cpp
        # src/backends/cuda_kernels.cu    # 将来写 kernel 时加这里
    )

    # 链接 CUDA Runtime
    target_link_libraries(inferllm PRIVATE CUDA::cudart)

    # 定义宏，让 #ifdef INFERLLM_WITH_CUDA 生效
    target_compile_definitions(inferllm PRIVATE INFERLLM_WITH_CUDA)

    #让 .cu 文件找到 include 目录
    target_include_directories(inferllm PRIVATE include)
endif()



# dot 图的输出目录：和 include/ src/ 同级的 models_dot/（绝对路径，所以在哪个目录跑都一样）
target_compile_definitions(inferllm PRIVATE
    INFERLLM_MODELS_DOT_DIR="${CMAKE_SOURCE_DIR}/models_dot")


# 主程序
add_executable(inferllm_main src/main.cpp)
target_link_libraries(inferllm_main PRIVATE inferllm)

# 测试程序
add_executable(test_qwen3 test/test_qwen3_build_graph.cpp)
target_link_libraries(test_qwen3 PRIVATE inferllm)
