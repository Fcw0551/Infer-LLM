# Quest 16-token 块预算实验

使用 LongBench 原版 MultiFieldQA-en 和本地 Qwen3-0.6B F32 GGUF。源数据、提示词和评分规则见[上一级实验](../README.md)。完整题面以 Qwen tokenizer 的实际长度排序，将 150 条分为三等份。预先排除之前试跑或测时的题，用固定种子 20261122 抽出不重叠的 10 条调参题和 5 条验证题。短、中、长的数量分别为 3/4/3 和 1/2/2。入选题号和原始数据 ID 写入 `results/metadata/splits.json`。

每题对照共用同一份 dense 文档 KV。每条策略从同一个文档末尾继续，重新计算问题和答案。只在实验入口回退连续 KV 的有效长度，之前策略写入的尾部由下一策略覆盖。正文与绝对位置保持不变。问题、提示后缀和答案逐 token 处理，贪心生成最多 24 token。Quest 对每个 query head 单独打分并保留当前块。

- 调参对同样十题跑 dense 和 256 / 512 / 1024 / 2048 token，分别对应最多 16 / 32 / 64 / 128 块。
- 选择均分比 dense 下降不超过 2 分且没有单题下降超过 20 分的最小预算。都不通过时，仅锁定均分最高的配置作为待验证候选。均分同分时依次比较严重掉分题数和预算，选择严重掉分题数最少、预算更高的配置。
- 锁定后，另外五题各跑 dense 和锁定配置。按官方规则计算英文词级 F1，对多参考答案取最高分。原先留出集只测锁定预算。随后在同一批五题上补齐 256、1024、2048，复用 dense 与 512 的输出。完整留出曲线用于对比各档，并保留原先锁定的 512。
- 每条完成的输出先写临时文件再重命名，重跑会跳过已完成的输出。运行同一结果目录时只启动一个进程。

```bash
work=experiments/sparse-attention-accuracy/quest
bash "$work/build.sh"
python3 -u "$work/sweep.py" /tmp/infer-quest-build/run \
  models/Qwen3-0.6B-F32.gguf /tmp/infer-longbench-cases "$work/results" --full-validation
```

从[结果索引](results/README.md)进入。目录按用途组织。

| 位置 | 内容 |
|---|---|
| [figures/](results/figures/) | 一张总图，左侧为平均 F1，右侧为相对本组 Dense 的分差 |
| [scores/](results/scores/) | 逐题预测、参考答案和 F1 |
| [predictions/](results/predictions/) | 按调参集与留出集存放的 75 份原始模型输出 |
| [metadata/](results/metadata/) | 150 条长度盘点、实际抽样清单和调参阶段的预算选择 |

只重画图和更新汇总时，直接读取保存的分数。

```bash
python3 "$work/sweep.py" report "$work/results"
```

原型在全量 KV 上估计块分数并屏蔽未选位置。这个实验评估答题质量。

## 32 token/块对照

[32-token 块结果与对照图](results-block32/README.md)。使用 16-token 实验锁定的 15 道题和四档 token 预算，沿用同一份 Dense 回答和官方英文 QA F1。只新增 60 条 Quest 预测。两种块大小均保留当前块，每题在各档之间复用 dense 文档前缀。

```bash
work=experiments/sparse-attention-accuracy/quest
bash "$work/build.sh"
python3 -u "$work/compare_blocks.py" /tmp/infer-quest-build/run \
  models/Qwen3-0.6B-F32.gguf /tmp/infer-longbench-cases \
  "$work/results" "$work/results-block32"
```

重复执行同一命令会补齐缺失的预测；模型、题面和参数应与首次运行一致。

## 64 token/块对照

[16、32、64 token/块合并报告](results-block64/README.md)。64-token 轮沿用相同的 15 道题、Dense 输出和历次四档 token 预算，对应每头最多 4、8、16、32 块。合并报告直接读取前两轮的已保存得分。

```bash
work=experiments/sparse-attention-accuracy/quest
bash "$work/build.sh"
python3 -u "$work/compare_blocks.py" /tmp/infer-quest-build/run \
  models/Qwen3-0.6B-F32.gguf /tmp/infer-longbench-cases \
  "$work/results" "$work/results-block64" --block 64 \
  --block32-results "$work/results-block32"
```
