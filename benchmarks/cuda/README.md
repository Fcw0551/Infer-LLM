# Infer-LLM CUDA benchmark

独立 C++ 入口沿用 CPU benchmark 的计时与 CSV 格式。使用 `cuda:0` 优先、CPU embedding，全部 KV 常驻 GPU；输入与 θ 表由 CPU 持有，调度器负责跨设备复制。引擎代码不因测试而修改。

## 构建与快速检查

```bash
cmake -S benchmarks/cuda -B benchmarks/cuda/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_PREFIX_PATH="$HOME/miniconda3"
cmake --build benchmarks/cuda/build --target inferllm-bench -j 4
ctest --test-dir benchmarks/cuda/build --output-on-failure
./benchmarks/cuda/build/inferllm-bench --mode pg -p 4 -n 2 -t 8 -r 1 --warmup 0
# 同一入口的 CPU 对照
./benchmarks/cuda/build/inferllm-bench --backend cpu --mode pg -p 4 -n 2 -t 8 -r 1 --warmup 0
```

依赖在系统目录时可省略 CMAKE_PREFIX_PATH。当前引擎 CMake 的 CUDA 架构为 89，适配本次 RTX 4060 Laptop；其他显卡应相应调整。需可访问 GPU 的 Linux/WSL 环境。

## 四组正式测试

```bash
python3 benchmarks/cuda/run_matrix.py --output docs/performance/cuda-contiguous-kv-20261107
```

使用新的输出目录；脚本拒绝覆盖已有测量 CSV。四组 P=32/64/128/256，N=128；每组独立进程串行执行，8 个 CPU worker，预热 1 次、正式 3 次。合成输入 `(100+i)%vocab_size`，贪心采样，遇 EOS 仍继续。每轮 KV 序列独立，计算量为 P+127。

`execute` 包含 Scheduler 的调度、分配、传输、计算及 `cudaDeviceSynchronize` 等待；发生执行错误也先等待再退出。TTFT、TPOT、request、wall、RSS 等字段沿用 [CPU benchmark 口径](../cpu/README.md)。不是纯 kernel 性能，也不是完全 GPU 推理。

脚本检查 CPU/GPU 空闲门槛，保留每秒 `/proc` 和 `nvidia-smi` 采样、源码指纹、模型 SHA256、构建及设备信息。GPU 显存是整张卡观测值，包含 Windows 显示和其他进程；采样不能捕获所有瞬时峰值。RSS 只统计主机进程，不能替代显存。温度、功耗和利用率不代表受控频率或已排除全部宿主负载。

运行失败或资源压力超阈值时停止，保留失败数据；不将不完整数据作为成功性能结果。输出 checksum 只检查生成序列重复性，不是逐算子数值正确性证明。

## 绘图

```bash
python3 docs/performance/cuda-contiguous-kv-20261107/plot_results.py
```

绘图需要 matplotlib，输出 SVG、PNG 和 `summary.json`。脚本校验每组三次完整数据、相同 checksum、P+127 计算量及计时公式。
