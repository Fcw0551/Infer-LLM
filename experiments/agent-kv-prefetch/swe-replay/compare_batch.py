"""Compare tool-boundary prefetch with previous-token prefetch per task."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

STRATEGIES = ("无预取", "上一 token", "上一 token + 工具边界")
MIB = 2**20


def load_case(directory, expected, config):
    rounds = [json.loads(line) for line in (directory / "rounds.jsonl").read_text().splitlines()]
    metadata, *calls = rounds
    actual_config = metadata["block_size"], metadata["budget_per_head"]
    if actual_config != config:
        raise ValueError(f"unexpected Quest configuration: {directory.name}")
    report = json.loads((directory / "report.json").read_text())
    if not calls or any(call["status"] != "ok" for call in calls):
        raise ValueError(f"incomplete replay: {directory.name}")
    if report[metadata["instance"]]["resolved"] is not True:
        raise ValueError(f"unresolved instance: {directory.name}")
    totals = {}
    for line in (directory / "lru_global_unlimited.md").read_text().splitlines():
        fields = [field.strip() for field in line.strip("|").split("|")]
        if len(fields) in (8, 9) and fields[0].isdigit() and fields[1] in STRATEGIES:
            demand = float(fields[2].replace(",", ""))
            prefetch = float(fields[4].replace(",", ""))
            totals[int(fields[0]), fields[1]] = demand, demand + prefetch
    boundaries = dict.fromkeys(totals, 0)
    with (directory / "lru_global_unlimited_by_layer.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = int(row["global_capacity_bytes"]) // MIB, row["strategy"]
            if row["phase"] == "boundary" and key in boundaries:
                boundaries[key] += int(row["demand_bytes"]) / MIB
    final = calls[-1]
    context = final["start"] + final["input_tokens"] + final["reply_tokens"]
    actual = len(calls), context, metadata["sha256"]
    if actual != expected:
        raise ValueError(f"replay does not match manifest: {directory.name}")
    return metadata["instance"], len(calls), context, totals, boundaries


def percentages(old, new):
    return f"{(new - old) / old:+.1%}" if old else "—"


def summarize(batch):
    _, block, budget = batch.name.rsplit("-", 2)
    config = int(block.removeprefix("b")), int(budget.removeprefix("t"))
    with (batch / "cases.csv").open() as stream:
        cases = [load_case(batch / row["instance"], (int(row["expected_rounds"]),
                 int(row["expected_tokens"]), row["sha256"]), config) for row in csv.DictReader(stream)]
    if len(cases) != 10:
        raise ValueError(f"expected 10 cases, found {len(cases)}")
    print("# 10 条 SWE-bench 轨迹的工具边界预取对照\n")
    print("所有题来自 Claude Opus 4.6 的同一份已判定通过的 Bash Only 提交。")
    print(f"使用本地 Qwen3 回放，Quest {config[0]}–{config[1]}，全局 LRU，无单层预取上限。"
          "预取假设即时到达，表中变化量为累计模拟读流量。\n")
    print("| 共享容量 | 上一 token 相对无预取按需 | 上一 token 相对无预取总读取 | "
          "加工具边界后按需 | 仅上一 token 边界按需 MiB | 组合策略边界按需 MiB | "
          "加工具边界后边界按需变化 | 加工具边界后总读取 | 边界改善题数 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    per_case = []
    for capacity in (448, 896, 1344):
        base, token, combined = [(capacity, strategy) for strategy in STRATEGIES]
        demand_base = demand_token = demand_combined = 0
        total_base = total_token = total_combined = 0
        boundary_token = boundary_combined = 0
        wins = 0
        changes = []
        for instance, calls, context, totals, boundaries in cases:
            base_demand, base_total = totals[base]
            token_demand, token_total = totals[token]
            comb_demand, comb_total = totals[combined]
            old_boundary, new_boundary = boundaries[token], boundaries[combined]
            demand_base += base_demand
            demand_token += token_demand
            demand_combined += comb_demand
            total_base += base_total
            total_token += token_total
            total_combined += comb_total
            boundary_token += old_boundary
            boundary_combined += new_boundary
            wins += new_boundary < old_boundary
            changes.append(old_boundary - new_boundary)
            per_case.append((instance, calls, context, capacity, token_demand, comb_demand,
                             old_boundary, new_boundary, token_total, comb_total))
        print(f"| {capacity} MiB | {percentages(demand_base, demand_token)} | "
              f"{percentages(total_base, total_token)} | {percentages(demand_token, demand_combined)} | "
              f"{boundary_token:,.1f} | {boundary_combined:,.1f} | "
              f"{percentages(boundary_token, boundary_combined)} | {percentages(total_token, total_combined)} | "
              f"{wins}/10（中位数少读 {median(changes):,.1f} MiB） |")

    print("\n百分比按各题字节数先求和再计算。加工具边界后的变化以仅上一 token 为基线。")
    with (batch / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["instance", "rounds", "context_tokens", "capacity_mib",
                         "token_demand_mib", "combined_demand_mib",
                         "token_boundary_demand_mib", "combined_boundary_demand_mib",
                         "token_total_mib", "combined_total_mib"])
        writer.writerows(per_case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    summarize(parser.parse_args().batch)
