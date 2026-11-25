"""Summarize Quest selections in a replay run."""
import gzip
from itertools import groupby
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
with (directory / "rounds.jsonl").open() as rounds:
    config = json.loads(next(rounds))
block_size = config["block_size"]
block_budget = config["budget_per_head"] // block_size
raw = directory / "blocks.jsonl"
stream = raw.open() if raw.exists() else gzip.open(directory / "blocks.jsonl.gz", "rt")

# Each trace phase starts its layer counter at zero. Decode increments it
# across tokens, so recover the layer after validating the full sequence.
counts = {}
def accumulate(name, hits, total):
    count = counts.setdefault(name, [0, 0])
    count[0] += hits
    count[1] += total

def key(row):
    return row["call"], row["phase"], row["position"]

previous = None
previous_sets = []
previous_decode = None
phase_key = None
layer_offset = 0
positions = 0
with stream:
    for (call, phase, position), rows in groupby(map(json.loads, stream), key):
        if phase_key != (call, phase):
            phase_key = (call, phase)
            layer_offset = 0
        rows = list(rows)
        assert len(rows) == 28 * 16
        history = position // block_size
        layers = [set() for _ in range(28)]
        groups = [set() for _ in range(28 * 8)]
        for index, row in enumerate(rows):
            layer, head = divmod(index, 16)
            assert (row["layer"], row["query_head"]) == (layer_offset + layer, head)
            selected = set(row["blocks"])
            assert history in selected and len(selected) == len(row["blocks"]) <= block_budget
            selected.remove(history)
            assert all(0 <= block < history for block in selected)
            layers[layer].update(selected)
            groups[layer * 8 + row["kv_head"]].update(selected)
            accumulate("单 query head 历史块覆盖率", len(selected), history)
        layer_offset += 28
        for group in groups:
            accumulate("同 KV head 并集覆盖率", len(group), history)
        for layer in layers:
            accumulate("整层并集覆盖率", len(layer), history)
        if previous == (call, position - 1):
            for old, current in zip(previous_sets, layers, strict=True):
                accumulate("上一 token 选块覆盖率", len(old & current), len(current))
        for prior, current in zip(layers, layers[1:]):
            accumulate("上一层选块覆盖率", len(prior & current), len(current))
        if phase == "boundary" and previous_decode and previous_decode[0] == call - 1:
            _, old_position, old_layers = previous_decode
            for prior, current in zip(old_layers, layers, strict=True):
                old_demand = {block for block in current if block < (old_position + 1) // block_size}
                accumulate("工具边界旧块覆盖率", len(prior & old_demand), len(old_demand))
        if phase == "decode":
            previous_decode = (call, position, layers)
        previous, previous_sets = (call, position), layers
        positions += 1

print(f"逐 token 记录 {positions} 个位置，每位置 28 层、16 个 query head。")
for label, (hits, total) in counts.items():
    print(f"- {label} {hits / total:.1%}（{hits:,} / {total:,}）")
