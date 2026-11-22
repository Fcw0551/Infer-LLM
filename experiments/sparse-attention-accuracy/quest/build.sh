#!/usr/bin/env bash
set -euo pipefail
quest_dir=experiments/sparse-attention-accuracy/quest
quest_build=${1:-/tmp/infer-quest-build}
mkdir -p "$quest_build"
cmake --build benchmarks/cuda/build --target inferllm -j 6
/usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_89 -ccbin=/usr/bin/g++-13 -O3 \
  -DINFERLLM_WITH_CUDA -DINFERLLM_QUEST_EXPERIMENT \
  -c src/kernels/cuda/kernel_f32.cu -o "$quest_build/kernel.o"
/usr/bin/g++-13 -std=c++23 -O3 -pthread -Iinclude -I/usr/local/cuda/include \
  '-DBENCH_DEFAULT_MODEL=""' '-DBENCH_GIT_COMMIT=""' '-DBENCH_BUILD_TYPE="Release"' \
  "$quest_dir/run.cpp" "$quest_build/kernel.o" benchmarks/cuda/build/engine/libinferllm.a \
  -L"$HOME/miniconda3/lib" -L/usr/local/cuda/lib64 \
  -Wl,-rpath,"$HOME/miniconda3/lib":/usr/local/cuda/lib64 -lpcre2-8 -licuuc -lcudart \
  -o "$quest_build/run"
