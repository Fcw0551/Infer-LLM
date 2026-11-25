"""Replay layer-wide historical KV demand with a shared GPU LRU cache."""
import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

from predict import rank, read_positions


def simulate(directory, prefetch_cap_mib):
    with (directory / "rounds.jsonl").open() as stream:
        block_size = json.loads(next(stream))["block_size"]
    block_bytes = block_size * 8 * 128 * 2 * 4  # KV heads, head dimension, K/V, F32
    max_prefetch_blocks = (None if prefetch_cap_mib is None
                           else prefetch_cap_mib * 2**20 // block_bytes)
    # Use the same token capacities across block sizes.
    settings = [(capacity, window) for capacity in (2048, 4096, 6144)
                for window in ("无预取", "上一 token", "上一层", "仅工具边界", "上一 token + 工具边界")]
    caches = {key: OrderedDict() for key in settings}
    totals = defaultdict(lambda: [0, 0, 0])
    layer_reads = defaultdict(int)
    previous_layers = []
    previous = None

    for call, phase, position, layers in read_positions(directory):
        tool_boundary = phase == "boundary" and previous and previous[0] == call - 1
        for layer, demand in enumerate(layers):
            token_candidates = []
            if previous == (call, position - 1):
                token_candidates = rank([previous_layers[layer]], (position - 1) // block_size)
            tool_candidates = (rank([previous_layers[layer]], (previous[1] + 1) // block_size)
                               if tool_boundary else [])
            ranked = {
                "无预取": [],
                "上一 token": token_candidates,
                "仅工具边界": tool_candidates,
                "上一 token + 工具边界": tool_candidates if tool_boundary else token_candidates,
                "上一层": rank([layers[layer - 1]], position // block_size) if layer else [],
            }
            for capacity, window in settings:
                cache = caches[capacity, window]
                capacity_blocks = capacity * 28 // block_size
                demand_reads, prefetched, wasted = totals[capacity, window, phase]
                if ranked[window]:
                    # Reserve an equal byte allowance for reads that miss residency.
                    candidates = [block for block in ranked[window] if (layer, block) not in cache]
                    for block in candidates[:max_prefetch_blocks]:
                        cache[layer, block] = True
                        prefetched += 1
                        if len(cache) > capacity_blocks:
                            wasted += cache.popitem(last=False)[1]

                # Count simultaneous demand against the post-prefetch snapshot.
                demand_keys = [(layer, block) for block in sorted(demand)]
                misses = sum(key not in cache for key in demand_keys)
                demand_reads += misses
                layer_reads[capacity, window, phase, layer] += misses
                for key in demand_keys:
                    cache[key] = False
                    cache.move_to_end(key)
                while len(cache) > capacity_blocks:
                    wasted += cache.popitem(last=False)[1]
                totals[capacity, window, phase] = [demand_reads, prefetched, wasted]
        previous_layers = layers
        previous = (call, position)

    print(f"# {directory.name} · 离线 LRU 模拟\n")
    cap = ("预取全部未驻留候选。" if prefetch_cap_mib is None
           else f"预取上限为每层每位置 {prefetch_cap_mib} MiB。")
    print(f"整层历史块，F32，28 层共享全局 LRU。{cap}"
          "上一 token 仅限同轮相邻位置。上一层仅限同 token 的前一层，块位置映射到当前层。\n"
          "工具边界用上一轮最后记录的回复 token，只预取工具执行前的完整历史块。\n"
          "同一步需求批量检查，需求块按块编号更新 LRU。未使用预取包括驱逐和运行结束仍未命中。"
          "模型只模拟已记录位置的历史块读取，省略 dense prefill、当前块和实际搬运时延。\n")
    print("| 共享容量（MiB） | 策略 | 按需 MiB | 少读 MiB | 预取 MiB | 未用预取 MiB | 总读取变化 MiB | 边界按需 MiB | 边界按需少读 MiB |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    mib = block_bytes / 2**20
    for capacity, window in settings:
        boundary = totals[capacity, window, "boundary"]
        decode = totals[capacity, window, "decode"]
        reads, prefetched, wasted = map(sum, zip(boundary, decode))
        wasted += sum(caches[capacity, window].values())
        baseline_boundary = totals[capacity, "无预取", "boundary"][0]
        saved = baseline_boundary + totals[capacity, "无预取", "decode"][0] - reads
        print(f"| {capacity * 28 * block_bytes // block_size // 2**20} | {window} | {reads*mib:,.1f} | {saved*mib:,.1f} | "
              f"{prefetched*mib:,.1f} | {wasted*mib:,.1f} | {(prefetched-saved)*mib:+,.1f} | "
              f"{boundary[0]*mib:,.1f} | {(baseline_boundary-boundary[0])*mib:,.1f} |")

    suffix = "unlimited" if prefetch_cap_mib is None else f"cap{prefetch_cap_mib}mib"
    with (directory / f"lru_global_{suffix}_by_layer.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["global_capacity_bytes", "strategy", "phase", "layer",
                         "demand_bytes", "saved_demand_bytes"])
        for (capacity, window, phase, layer), reads in sorted(layer_reads.items()):
            baseline = layer_reads[capacity, "无预取", phase, layer]
            writer.writerow([capacity * 28 * block_bytes // block_size, window, phase, layer, reads * block_bytes,
                             (baseline - reads) * block_bytes])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefetch-cap-mib", type=int, help="Optional per-layer per-position cap")
    args = parser.parse_args()
    if args.prefetch_cap_mib is not None and args.prefetch_cap_mib < 0:
        parser.error("prefetch cap must be nonnegative")
    simulate(args.directory, args.prefetch_cap_mib)
