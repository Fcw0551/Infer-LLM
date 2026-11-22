#!/usr/bin/env python3
"""Compare Quest block sizes on the same paired LongBench samples."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from sweep import BUDGETS, evaluate, metrics


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def chart(path, summaries):
    blocks = sorted({row['block'] for row in summaries} - {0})
    colors = {16: '#1765a6', 32: '#a64720', 64: '#397738'}
    width, height = 1100, 500
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        '<title>Quest 块大小对照</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="system-ui,sans-serif" font-size="15" fill="#172432">',
    ]

    def text(x, y, value, attrs=''):
        svg.append(f'<text x="{x}" y="{y}" {attrs}>{value}</text>')

    text(32, 40, 'Quest 块大小与 LongBench 答题得分', 'font-size="25" font-weight="700"')
    text(32, 67, '同一批题 · 同一 Dense 基线 · 相同的每头 token 预算')
    for i, block in enumerate(blocks):
        color = colors[block]
        x = 34 + i * 180
        svg.append(f'<path d="M{x},92h24" stroke="{color}" stroke-width="3"/>')
        text(x + 32, 97, f'{block} token/块')
    text(34 + len(blocks) * 180 + 14, 97, '灰色虚线表示各组 Dense')
    ymax = max(10, min(100, 10 * math.ceil((max(row['f1'] for row in summaries) + 4) / 10)))
    for panel, (group, label) in enumerate((('tuning', '调参集（10 题）'), ('validation', '留出集（5 题）'))):
        left, top, w, h = 86 + 535 * panel, 157, 400, 243
        y = lambda score: top + h * (1 - score / ymax)
        xs = dict(zip(BUDGETS, (left + 28 + i * (w - 56) / 3 for i in range(4))))
        rows = [r for r in summaries if r['split'] == group]
        dense = next(r['f1'] for r in rows if r['block'] == 0)
        text(left, 139, f'{label} · 平均 F1', 'font-weight="650"')
        for tick in range(0, ymax + 1, 10):
            yy = y(tick)
            svg.append(f'<path d="M{left},{yy:.1f}h{w}" stroke="#e1e6eb"/>')
            text(left - 10, yy + 5, str(tick), 'text-anchor="end"')
        yy = y(dense)
        svg.append(f'<path d="M{left},{yy:.1f}h{w}" stroke="#687684" stroke-dasharray="6 5"/>')
        text(left + w + 7, yy + 4, f'{dense:.2f}', 'font-size="12"')
        for b, x in xs.items():
            text(x, 426, f'{b:,}', 'text-anchor="middle"')
        text(left + w / 2, 463, '每个 query head 的 token 预算', 'text-anchor="middle"')
        for block in blocks:
            color = colors[block]
            points = [r for r in rows if r['block'] == block]
            coords = ' '.join(f'{xs[r["budget"]]:.1f},{y(r["f1"]):.1f}' for r in points)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.8"/>')
            for row in points:
                b, score = row['budget'], row['f1']
                xx, yy = xs[b], y(score)
                tooltip = f'<title>{block} token/块，预算 {b:,}，F1 {score:.2f}</title>'
                if block == 16:
                    svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{color}" stroke="white">{tooltip}</circle>')
                elif block == 32:
                    svg.append(f'<rect x="{xx-4:.1f}" y="{yy-4:.1f}" width="8" height="8" fill="{color}" stroke="white">{tooltip}</rect>')
                else:
                    svg.append(f'<path d="M{xx:.1f},{yy-6:.1f} l6,6 l-6,6 l-6,-6 Z" fill="{color}" stroke="white">{tooltip}</path>')
                if len(blocks) == 2:
                    peer = next(r['f1'] for r in rows if r['budget'] == b and r['block'] not in (0, block))
                    above = score > peer or (score == peer and block == blocks[-1])
                    text(xx, yy + (-11 if above else 18), f'{score:.2f}',
                         f'text-anchor="middle" fill="{color}" font-size="12" stroke="white" stroke-width="3" paint-order="stroke"')
    svg.append('</g></svg>')
    path.write_text('\n'.join(svg) + '\n')


def build_report(root, results):
    blocks = sorted(results)
    summaries = []
    header = '| 集合 | 预算 | Dense F1 | ' + ' | '.join(f'{b}-token 块 F1' for b in blocks) + ' |'
    lines = ['# Quest 块大小对照', '', 'LongBench MultiFieldQA-en · Qwen3-0.6B F32',
             '', '[对照曲线](figures/block_comparison.svg) · [逐题分数](scores/per_sample.json) · '
             '[汇总 CSV](scores/summary.csv) · [原始预测](predictions/)', '', header,
             '|---|---:|---:|' + '---:|' * len(blocks)]
    for group, label in [('tuning', '调参集（10）'), ('validation', '留出集（5）')]:
        def identity(rows):
            return [(r['sample'], r['source_id'], r['prompt_tokens'], r['predictions']['dense'])
                    for r in rows]

        before = results[16][group]
        for block in blocks:
            if identity(before) != identity(results[block][group]):
                raise ValueError(f'dataset or Dense baseline differs for {group}, block {block}')
        dense = statistics.mean(r['f1']['dense'] for r in before)
        summaries.append(dict(split=group, block=0, budget=0, f1=dense, delta=0, worse=0, large_drop=0))
        for b in BUDGETS:
            budget_rows = [dict(split=group, block=block, **metrics(results[block][group], b))
                           for block in blocks]
            summaries.extend(budget_rows)
            scores = ' | '.join(f'{r["f1"]:.2f}' for r in budget_rows)
            lines.append(f'| {label} | {b:,} | {dense:.2f} | {scores} |')
    lines += ['', '各块大小使用同一批题及同一份 Dense 输出。']
    with (root / 'scores/summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    chart(root / 'figures/block_comparison.svg', summaries)
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'model', 'cases', 'baseline', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--block', type=int, choices=(32, 64), default=32)
    parser.add_argument('--block32-results', type=Path, help='Include completed 32-token scores in a 64-token report')
    args = parser.parse_args()
    if args.block32_results and args.block != 64:
        parser.error('--block32-results requires --block 64')
    root = args.output
    for folder in ('predictions/tuning', 'predictions/validation', 'scores', 'figures', 'metadata'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    groups = load(args.baseline / 'metadata/splits.json')
    previous = {16: load(args.baseline / 'scores/per_sample.json')}
    if args.block32_results:
        previous[32] = load(args.block32_results / 'scores/per_sample.json')
    config = dict(block_tokens=args.block, token_budgets=list(BUDGETS))
    config_path = root / 'metadata/config.json'
    if config_path.exists() and load(config_path) != config:
        raise ValueError('output directory contains a different configuration')
    save(config_path, config)
    save(root / 'metadata/splits.json', groups)
    env = dict(os.environ, INFERLLM_THREADS='8')
    env.pop('QUEST_VALIDATION', None)
    env.pop('QUEST_MAX_PROMPT', None)
    new = {}
    with (root / 'metadata/run.log').open('a', buffering=1) as log:
        for group, rows in groups.items():
            directory = root / 'predictions' / group
            for row in rows:
                name = f'{row["sample"]}.dense'
                data = (args.baseline / 'predictions' / group / name).read_bytes()
                target = directory / name
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f'baseline differs: {target}')
                target.write_bytes(data)
            inputs = [args.cases / f'{row["sample"]}.txt' for row in rows]
            print(f'{args.block}-token blocks: {group} ({len(inputs)} samples)', flush=True)
            command = [args.runner, args.model, directory, 'sweep', str(args.block), '2048', *inputs]
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as proc:
                for line in proc.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                if proc.wait():
                    raise subprocess.CalledProcessError(proc.returncode, command)
            new[group] = evaluate(rows, directory, BUDGETS)
            save(root / f'scores/{group}.json', new[group])
    save(root / 'scores/per_sample.json', new)
    build_report(root, {**previous, args.block: new})


if __name__ == '__main__':
    main()
