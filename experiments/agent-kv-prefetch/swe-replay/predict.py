"""Compare equal-budget predictors on layer-wide Quest block demand."""
from collections import defaultdict, deque
import gzip
from itertools import groupby
import json
from pathlib import Path
import sys

WINDOWS = (1, 2, 4, 8)
BUDGETS = (16, 32, 64)


def rank(history, limit):
    scores = defaultdict(int)
    for weight, blocks in enumerate(history, 1):
        for block in blocks:
            if block < limit:
                scores[block] += weight
    return sorted(scores, key=lambda block: (-scores[block], -block))


def read_positions(directory):
    with gzip.open(directory / "blocks.jsonl.gz", "rt") as trace:
        def position_key(row):
            return row["call"], row["phase"], row["position"]

        offsets = defaultdict(int)
        for (call, phase, position), rows in groupby(map(json.loads, trace), position_key):
            layers = [set() for _ in range(28)]
            offset = offsets[call, phase]
            count = 0
            for index, row in enumerate(rows):
                layer, head = divmod(index, 16)
                assert (row["layer"], row["query_head"]) == (offset + layer, head)
                layers[layer].update(set(row["blocks"]) - {row["current_block"]})
                count += 1
            assert count == 28 * 16
            offsets[call, phase] += 28
            yield call, phase, position, layers


def analyze(directory):
    with (directory / "rounds.jsonl").open() as stream:
        block_size = json.loads(next(stream))["block_size"]
    totals = defaultdict(lambda: [0, 0, 0, 0])
    missed = defaultdict(lambda: [0, 0])
    history = deque(maxlen=max(WINDOWS))
    tool_history = []
    previous = None

    def score(cohort, demand, limit, predictors):
        demand = {block for block in demand if block < limit}
        recent = range(limit - 1, -1, -1)
        for budget in BUDGETS:
            count = min(budget, limit)
            for name, candidates in predictors.items():
                selected = list(candidates[:count])
                if len(selected) < count:
                    present = set(selected)
                    selected.extend([block for block in recent if block not in present][:count - len(selected)])
                add(cohort, budget, name, len(demand.intersection(selected)), count, len(demand))
            add(cohort, budget, "均匀随机期望", count * len(demand) / limit if limit else 0,
                count, len(demand))
            add(cohort, budget, "最近写入", len(demand.intersection(recent[:count])), count, len(demand))

    def add(cohort, budget, name, hits, selected, demand):
        row = totals[cohort, budget, name]
        row[0] += hits
        row[1] += selected
        row[2] += demand
        row[3] += 1

    for call, phase, position, layers in read_positions(directory):
        if phase == "boundary":
            tool_history = list(history) if previous and previous[0] == call - 1 else []
            tool_limit = (previous[1] + 1) // block_size if tool_history else 0
            boundary = position
            history.clear()
        if history and previous != (call, position - 1):
            history.clear()
        for layer, demand in enumerate(layers):
            if len(history) == max(WINDOWS):
                limit = (position - 1) // block_size
                old_sets = [step[layer] for step in history]
                score("同层近期 token", demand, limit,
                      {f"最近 {n} token": rank(old_sets[-n:], limit) for n in WINDOWS})
                missing = {block for block in demand - old_sets[-1] if block < limit}
                for n in WINDOWS[1:]:
                    counts = missed[n]
                    counts[0] += len(missing.intersection(set().union(*old_sets[-n:-1])))
                    counts[1] += len(missing)
            if layer >= 4:
                limit = position // block_size
                score("同 token 跨层", demand, limit,
                      {f"L-{n}": rank([layers[layer - n]], limit) for n in (1, 2, 4)})
            offset = position - boundary
            if len(tool_history) == max(WINDOWS) and offset in (0, 1, 2, 4, 8):
                old_sets = [step[layer] for step in tool_history]
                score(f"工具边界 +{offset}", demand, tool_limit,
                      {f"工具前最近 {n} token": rank(old_sets[-n:], tool_limit) for n in WINDOWS})
        history.append(layers)
        previous = call, position

    print("# Quest 选块预测")
    print()
    print("以整层各头的选块并集作为需求，当前块排除。各策略使用相同的 16、32、64 块预算，"
          "可选块不足时按实际数量计。近期窗口按从旧到新的 1 到 N 权重累计访问频次，"
          "同分时优先较新的块位置，候选不足时按最近写入补齐。")
    print()
    print("同层比较只保留同轮已有连续 8 个历史观测的位置，目标仅计预测时已完整的旧块。"
          "跨层比较统一以第 4 层及之后为目标（层号从 0 开始）。"
          "工具边界 +0 是追加工具结果后的最后一个输入位置，其余偏移按回复 token 位置计，"
          "需求仅计工具调用前已经完整的旧块。")
    print()
    print("| 场景 | 预算（块 / 层） | 策略 | 样本（位置 × 层） | 精确率 | 覆盖率 |")
    print("|---|---:|---|---:|---:|---:|")
    for (cohort, budget, name), (hits, selected, demand, samples) in totals.items():
        print(f"| {cohort} | {budget} | {name} | {samples} | {hits / selected:.2%} | {hits / demand:.2%} |")
    print("\n上一 token 漏掉的需求中，在窗口内更早位置出现过的比例（未限制候选数）\n")
    for n, (recovered, missing) in missed.items():
        print(f"- 最近 {n} token 窗口 {recovered / missing:.2%}（{recovered:,} / {missing:,}）")


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
