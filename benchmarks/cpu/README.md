# Infer-LLM CPU C++ benchmark

独立入口 `bench.cpp` 直接链接 `inferllm`，调用模型、Graph、Scheduler 和 Sampler。
使用 `std::chrono::steady_clock` 在进程内计时，不启动 main、不解析日志、不依赖 Python。
模型和 CPU 线程池仅加载一次，GraphContext/RoPE 表跨轮保留；每一步仍新建图；Scheduler 跨步复用激活缓冲区。每轮请求创建独立 KV 序列，结束后释放。

## 编译

在项目根目录运行：

```bash
cmake -S benchmarks/cpu -B benchmarks/cpu/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/miniconda3"
cmake --build benchmarks/cpu/build --target inferllm-bench -j 4
```

PCRE2/ICU 安装在系统目录时可以省略 `CMAKE_PREFIX_PATH`。这些依赖来自现有引擎的
构建要求，benchmark 本身不做分词。需要 C++23 编译器，当前支持 Linux/WSL。
此独立构建强制关闭 CUDA/Metal，并补充 pthread 链接，不修改根 CMakeLists.txt
和原来的 `build/`。后续修改源码后，重新执行第二条增量编译命令即可。

## 运行

```bash
# 默认：Qwen3-0.6B-F32，自动线程数，16 个输入 token，8 次生成，预热 1 次、正式 3 次
./benchmarks/cpu/build/inferllm-bench

# 只测固定长度输入处理
./benchmarks/cpu/build/inferllm-bench --mode pp -p 32 -t 4 -r 3

# 只测连续生成；首次处理 prompt，随后每步使用 KV Cache 处理一个新 token
./benchmarks/cpu/build/inferllm-bench --mode pg -p 16 -n 32 -t 4 -r 3

# 指定模型，并保存每轮数据；相同 CSV 路径会覆盖旧文件
./benchmarks/cpu/build/inferllm-bench \
  -m models/Qwen3-0.6B-F32.gguf --mode all -p 16 -n 8 -t 4 -r 3 \
  --csv benchmarks/cpu/results/baseline.csv

# 快速检查
./benchmarks/cpu/build/inferllm-bench --mode all -p 4 -n 2 -r 1 --warmup 0
./benchmarks/cpu/build/inferllm-bench --help
```

不传 `-t`（或传 `-t 0`）时，由 CPUBackend 自动检测线程数，覆盖继承的
`INFERLLM_THREADS` 设置。当前后端策略是 `hardware_concurrency()` 返回值的一半，
向下取整、至少 1 个线程；这是启发式选择，不是自动寻找最优性能线程数。
控制台与 CSV 都记录后端实际使用的数量。显式传 `-t 4` 等正整数仍可手动指定。

仅接受当前已经验证的 F32 权重。输入采用固定合成 token：`(100+i) % vocab_size`，
不使用 main 中的 prompt 或聊天模板，不对生成文字的语义做评估。

`pp`：每轮执行一张包含 P 个输入 token 的图，不读取 logits、不采样。
`pg`：每轮从同一份 P token 输入开始，贪心采样固定 N 次。首次 Prefill 处理 P 个 token，
随后 N-1 次 Decode 各处理一个 token，复用历史 KV。CSV 的 `evaluated_tokens` 必须等于 `P+N-1`。
**这是固定工作量基准，遇到 EOS 也继续执行**，确保各轮任务长度一致；不是聊天生成流程。
`--mode all` 按 pp、pg 顺序执行，两种模式分别预热。`--warmup 0` 可禁用预热。

## 计时范围

| 字段 | 范围 |
|---|---|
| setup | 后端、GGUF、超参数、权重加载和放置；不计入结果 |
| prepare | 查询历史长度、预留 KV、拷贝输入、准备绝对 position、更新输入 Tensor |
| build | build_graph（含取得各层 KV 访问）、拓扑排序 |
| execute | Scheduler::execute：图切分、激活分配、同步 CPU 计算 |
| sample | 读取最后位置的 logits、贪心采样；pp 为 0 |
| cleanup | 归还各层 KV 访问、提交新增 KV、销毁本步图；激活由 Scheduler 持有 |
| wall | 本轮从创建 KV 序列到最后一步清理并释放序列，含循环管理开销 |

执行阶段计时直接包围 `Scheduler::execute()`。由于该接口将分配和计算封装在一起，
**execute 不是纯算子时间**，也没有通过修改引擎来拆分它。
当前 CPU compute 同步返回，计时有效；未来测异步 GPU 必须在计时边界同步。

- pp 整轮吞吐量：`P / wall`；另报告调度执行阶段输入吞吐量 `P / execute`。
- pg 整轮吞吐量：`N / wall`。
- pg 首 token 耗时：从本轮开始至第一次采样完成，不含模型加载/分词。
- pg 后续生成速度：`(N-1) / (最后一次采样时刻-第一次采样时刻)`；N=1 不报告此项。
- 多轮报告每轮吞吐量的算术均值和样本标准差；单轮只报告测量值。

每轮 phases 是所有步骤的累计值。CSV 记录秒、token/s 和生成序列校验值；控制台时间用 ms。
预热不计入汇总，CSV 只保存正式轮次，写完一轮即刷新。发生错误返回非零退出码。
引擎 cout 调试日志被丢弃，减少终端 I/O 干扰；日志字符串构造本身仍可能有少量成本。

当前 pg 标为 `pg-kv`。Prefill 仍计算所有输入位置的 logits，Decode 每步只计算一个位置。
TTFT 包含本轮 KV 序列创建与首次计算；模型加载不计入。
历史无 KV 报告使用旧的逐节点激活释放方式，不能把两版差异全部归因于 KV Cache。
比较版本时固定模型文件、F32 精度、P/N、线程数、构建类型和预热次数，尽量避免其他重负载。

## 检查工具

```bash
ctest --test-dir benchmarks/cpu/build --output-on-failure
```

以上检查帮助信息和非法参数；真实模型测试使用前面的快速检查命令。
构建产物和 CSV 结果分别在 `build/`、`results/`，均已忽略版本跟踪。

## 固定四组测试与报告指标

| `--profile` | 场景 | 输入 token | 输出 token |
|---|---|---:|---:|
| p32 | 输入 32 | 32 | 128 |
| p64 | 输入 64 | 64 | 128 |
| p128 | 输入 128 | 128 | 128 |
| p256 | 输入 256 | 256 | 128 |

profile 自动选择 `pg`，不能与 `-p/-n` 同时使用。每组单独启动进程、串行执行，
避免不同组争抢 CPU/内存，并让进程峰值 RSS 对应该组。以下是单次、不预热的快速容量测试命令（不用于新报告的稳态均值）：

```bash
mkdir -p benchmarks/cpu/results/matrix
for profile in p32 p64 p128 p256; do
  ./benchmarks/cpu/build/inferllm-bench --profile "$profile" \
    -t 0 -r 1 --warmup 0 \
    --csv "benchmarks/cpu/results/matrix/$profile.csv" \
    > "benchmarks/cpu/results/matrix/$profile.log" 2>&1
  status=$?
  printf '%s exit=%s\n' "$profile" "$status"
done
```

本次每组正式运行一次、不预热，不代表稳态均值。需要重复性统计时使用 `-r 3 --warmup 1`；
不能将这两种设置混为同一测试条件。模型不在默认路径时增加 `-m /path/to/model.gguf`。

令 `t0` 为生成循环开始，`t1` 为第一次采样完成，`tN` 为最后一次采样完成：

| 报告指标 | 计算方式 / 范围 | CSV 字段 |
|---|---|---|
| TTFT | `(t1-t0) × 1000`，单位 ms | `ttft_s`（保存秒） |
| 平均 Decode ms/token / TPOT | `(tN-t1) × 1000 / (N-1)` | `tpot_ms` |
| Decode tokens/s | `(N-1) / (tN-t1)` | `decode_tokens_per_second` |
| 请求总耗时 | `tN-t0`，单位秒；不含最后一步清理 | `request_s` |
| 峰值内存 | `getrusage(RUSAGE_SELF).ru_maxrss / 1024`，单位 MiB | `peak_rss_mib` |

TTFT 是本地引擎指标，不包括网络、排队、分词、聊天模板和模型加载。未预热时仍包含
首次访问 mmap 权重的缺页成本。RSS 包含该进程加载、预热、正式运行的高水位及驻留的
文件映射页，不等于虚拟地址空间、激活分配总量或整个系统内存。N=1 时没有后续 token，
TPOT 与 Decode tokens/s 留空。多次重复时 RSS 是累计高水位，不能解读为各轮独立峰值。

失败返回非零退出码，日志保留原因和失败时峰值 RSS；CSV 只写完整成功的轮次。
失败组的完整请求指标应标为 N/A，失败前 RSS 不代表该组成功执行所需内存。
控制台 `Engine git` 来自 CMake 配置时的 HEAD，更新提交后应重新运行 CMake 配置命令。

生成模式在第一个 token 和每 16 个 token 后向 stderr 输出进度，便于检查长测试；少量日志开销计入整轮时间。失败日志同时记录已完成步数与失败时输入长度。


## KV Cache 四组稳态测试

```bash
python3 benchmarks/cpu/run_matrix.py --output docs/performance/cpu-contiguous-kv-20261104
```

每组固定 8 个 worker，独立进程串行运行，预热 1 轮、正式 3 轮；每轮均重新创建序列，
固定输出 128 个 token。脚本在每组前检查空闲状态，每秒保存 `/proc` 内存、Swap、PSI
和 CPU 计数。低于 512 MiB 可用内存会终止本组；观察到交换、内存 full stall、
低于 1 GiB 可用内存或平均外部 CPU 占用超过全机容量的 5% 时停止矩阵，不把结果直接当作无干扰数据。
`p*-system.csv` 与 `system-summary.json` 保留监测依据，`source-hashes.json` 用于确认测试期间源码未变。
这些检查不能观测所有宿主机活动或控制 CPU 频率。
