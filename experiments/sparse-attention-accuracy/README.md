# LongBench 稀疏 attention 精度试跑

[官方 MultiFieldQA-en 数据](https://huggingface.co/datasets/zai-org/LongBench/tree/bb67bab2b7402e46377f156d830d7510748bbc33/multifieldqa_en)固定在 `bb67bab2b7402e46377f156d830d7510748bbc33`。实测文件 SHA-256 为 `1c225242a15871ef857c6e815c74338ae54f51e4789ea3f991a6f06d922fb4e7`。使用种子 20261120 打乱，按顺序选择前六条实际 Qwen tokenizer 编码长度在 1024–4096 的完整 prompt。完整输入为[官方题面](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json)加 Qwen3 非思考 chat 包装。

使用本地 Qwen3-0.6B F32 GGUF、RTX 4060 Laptop GPU 与连续 KV。权重 SHA-256 为 `bb38dab890935b8dbd2cb57e68669f39c34ff54dad07ec0dcd80049071bbcf80`。三组都用 512 token 分块做 dense prefill，首个答案 token 也来自 dense prefill。后续使用贪心解码，最多生成 24 token，遇 EOS 停止。

| 模式 | 后续 decode 每个 query 可以访问的 token |
|---|---|
| dense | 所有因果可见位置 |
| recent512 | 最近 512 个位置，含当前位置 |
| sink4_recent508 | 最早四个位置与最近 508 个位置的并集 |

评分复现[官方英文 QA F1](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py)，对多参考答案取最大值。逐题预测和分数见 `results/per_sample.json`，均分见 `results/summary.md`。这是六题试跑，得分受样本构成和输出长度上限影响。mask 保留全量 KV 和当前 CUDA 分数缓冲，本次只测质量。

## 复现

先按 [CUDA benchmark 构建说明](../../benchmarks/cuda/README.md) 配置构建目录，然后构建引擎并编译独立入口。下面是本机路径。

```bash
cmake --build benchmarks/cuda/build --target inferllm -j 6
g++-13 -std=c++23 -O3 -pthread -Iinclude -I/usr/local/cuda/include \
  '-DBENCH_DEFAULT_MODEL=""' '-DBENCH_GIT_COMMIT=""' '-DBENCH_BUILD_TYPE="Release"' \
  experiments/sparse-attention-accuracy/run.cpp benchmarks/cuda/build/engine/libinferllm.a \
  -L"$HOME/miniconda3/lib" -L/usr/local/cuda/lib64 \
  -Wl,-rpath,"$HOME/miniconda3/lib":/usr/local/cuda/lib64 -lpcre2-8 -licuuc -lcudart \
  -o /tmp/infer-longbench-run
```

下载上述 Parquet 和官方 `dataset2prompt.json`。使用可导入 PyArrow 的 Python 执行准备步骤。

```bash
experiment=experiments/sparse-attention-accuracy
python3 "$experiment/longbench.py" prepare dataset.parquet dataset2prompt.json /tmp/longbench-cases
mkdir /tmp/longbench-results
INFERLLM_THREADS=8 /tmp/infer-longbench-run models/Qwen3-0.6B-F32.gguf \
  /tmp/longbench-results /tmp/longbench-cases/*.txt
python3 "$experiment/longbench.py" score /tmp/longbench-cases/references.json /tmp/longbench-results
```

CUDA mask 的小样本数值检查可单独运行。用零 Q/K 让 softmax 均匀，验证输出等于可见 V 的均值，覆盖因果边界、全覆盖窗口、窗口与前缀重叠、两条 query 以及 GQA。

```bash
g++-13 -std=c++23 -DINFERLLM_WITH_CUDA -Iinclude -I/usr/local/cuda/include \
  experiments/sparse-attention-accuracy/check_mask.cpp benchmarks/cuda/build/engine/libinferllm.a \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -o /tmp/infer-longbench-mask
/tmp/infer-longbench-mask
```
