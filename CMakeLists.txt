cmake_minimum_required(VERSION 3.18)
project(InferLLM LANGUAGES CXX)

# 设置变量
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 启用不同后端设备
option(INFERLLM_WITH_CUDA "Build with CUDA backend" ON)
option(INFERLLM_WITH_METAL "Build with Metal backend" OFF)


# 核心库（src 下的所有 .cpp，除了 main.cpp）
add_library(inferllm STATIC
    src/backends/cpu_backend.cpp
    src/backends/cuda_backend.cpp
    src/kvcache/KVStorages/naiveKVStorage.cpp
    src/kvcache/KVStrategys/contiguousKVStrategy.cpp
    src/kvcache/kvcacheManager.cpp
    src/kvcache/kvstorage.cpp
    src/kvcache/kvstrategy.cpp
    src/models/Qwen3.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/tokenizer_gguf.cpp
    src/tokenizer/tokenizer_pre.cpp
    src/tokenizer/tokenizer_bpe.cpp
    src/activationManager.cpp
    src/backend.cpp
    src/gguf_parser.cpp
    src/graph.cpp
    src/log.cpp
    src/model.cpp
    src/op_factory.cpp
    src/sampler.cpp
    src/scheduler.cpp
    src/tensor.cpp
    src/util.cpp
    
)

target_include_directories(inferllm PUBLIC include)
target_compile_options(inferllm PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
)

find_path(PCRE2_INCLUDE_DIR pcre2.h REQUIRED)
find_library(PCRE2_LIBRARY NAMES pcre2-8 REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc)
target_include_directories(inferllm PRIVATE ${PCRE2_INCLUDE_DIR})
target_link_libraries(inferllm PRIVATE ${PCRE2_LIBRARY} ICU::uc)


if(INFERLLM_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set_target_properties(inferllm PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 89
    )

    target_sources(inferllm PRIVATE
        src/kernels/cuda/kernel_f32.cu
    )

    target_link_libraries(inferllm PUBLIC CUDA::cudart)
    target_compile_definitions(inferllm PRIVATE INFERLLM_WITH_CUDA)
endif()


# dot 图的输出目录：和 include/ src/ 同级的 models_dot/（绝对路径，所以在哪个目录跑都一样）
target_compile_definitions(inferllm PRIVATE
    INFERLLM_MODELS_DOT_DIR="${CMAKE_SOURCE_DIR}/models_dot")


# 主程序
add_executable(inferllm_main src/main.cpp)
target_link_libraries(inferllm_main PRIVATE inferllm)

# 测试程序
# add_executable(test_qwen3 test/test_qwen3_build_graph.cpp)
# target_link_libraries(test_qwen3 PRIVATE inferllm)
