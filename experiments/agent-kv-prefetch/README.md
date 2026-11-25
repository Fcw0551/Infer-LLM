# Agent 轨迹回放与 KV 预取

## 十题完整回放

本地 Qwen3-0.6B F32 回放同一份 Claude Opus 4.6 + mini-SWE-agent Bash Only 提交里的十道
SWE-bench Verified 题。两个配置共用题目、轨迹哈希和回放轮次，最长一题 15,011 token。
各批次目录按实例 ID 保存压缩选块轨迹、轮次记录、官方判分和 LRU 报告。

| Quest 配置 | 汇总 | 逐题数据 |
|---|---|---|
| 16–1024 | [全局 LRU 汇总](swe-replay/verified-b16-t1024/summary.md) | [逐题 CSV](swe-replay/verified-b16-t1024/comparison.csv) |
| 64–2048 | [全局 LRU 汇总](swe-replay/verified-b64-t2048/summary.md) | [逐题 CSV](swe-replay/verified-b64-t2048/comparison.csv) |

## 早期 Astropy 单题回放

这两组记录覆盖前 23 轮，截止于 8,192 token 上限。

- [64–2048 选块预测](swe-replay/opus-astropy-12907-b64-t2048/prediction.md) 比较近期 token、跨层和工具边界规则。
- [64–2048 全局 LRU](swe-replay/opus-astropy-12907-b64-t2048/lru_global_unlimited.md) 与 [16–1024 全局 LRU](swe-replay/opus-astropy-12907-b16-t1024/lru_global_unlimited.md) 比较无预取、上一 token、上一层、仅工具边界及组合策略。各目录保留分层 CSV、轮次记录和压缩选块轨迹。
- [原始公开轨迹](https://swe-bench-submissions.s3.amazonaws.com/bash-only/20260217_mini-v2.0.0_claude-4-6-opus/trajs/astropy__astropy-12907/astropy__astropy-12907.traj.json) · [官方判分报告](https://swe-bench-submissions.s3.amazonaws.com/bash-only/20260217_mini-v2.0.0_claude-4-6-opus/logs/astropy__astropy-12907/report.json)

从仓库根目录运行以下命令可用现有轨迹复算。模拟按整层 KV 块计数，预取假设即时到达。

```bash
replay=experiments/agent-kv-prefetch/swe-replay
run="$replay/opus-astropy-12907-b64-t2048"
python3 "$replay/predict.py" "$run"
python3 "$replay/simulate_global.py" "$run"
for batch in verified-b16-t1024 verified-b64-t2048; do
  python3 "$replay/compare_batch.py" "$replay/$batch"
done
```

重新生成轨迹使用 `swe-replay/worker.cpp` 和 `swe-replay/replay.py`。
按照 [Quest 构建脚本](../sparse-attention-accuracy/quest/build.sh) 编译，用回放 worker 替换最终链接命令中的
`quest/run.cpp`，再运行 `replay.py --help` 查看输入和参数。
