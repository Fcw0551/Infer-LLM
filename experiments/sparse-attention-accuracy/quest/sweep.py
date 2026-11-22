#!/usr/bin/env python3
"""Run the stratified, paired Quest budget sweep and summarize QA F1."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import runpy
import statistics
import subprocess
import sys

BUDGETS = (256, 512, 1024, 2048)
SEED = 20261122
PILOT = {'003', '004', '006', '007', '014', '017', '057', '070', '092'}


def choose(inventory, refs):
    with inventory.open() as stream:
        ordered = sorted(csv.DictReader(stream),
                         key=lambda row: (int(row['prompt_tokens']), row['sample']))
    assert len(ordered) == len(refs) == 150
    rng = random.Random(SEED)
    groups = {'tuning': [], 'validation': []}
    for index, (tune, validate) in enumerate(zip((3, 4, 3), (1, 2, 2))):
        pool = [r for r in ordered[index * 50:(index + 1) * 50] if r['sample'] not in PILOT]
        rng.shuffle(pool)
        for name, selected in [('tuning', pool[:tune]), ('validation', pool[tune:tune+validate])]:
            for row in selected:
                groups[name].append(dict(**row, stratum=('short', 'medium', 'long')[index],
                                         **refs[row['sample']]))
    assert len({row['sample'] for rows in groups.values() for row in rows}) == 15
    return groups


def evaluate(rows, directory, budgets):
    f1 = runpy.run_path(str(Path(__file__).resolve().parent.parent / 'longbench.py'))['f1']
    result = []
    for row in rows:
        modes = ['dense', *(f'b{b}' for b in budgets)]
        answers = {m: (directory / f'{row["sample"]}.{m}').read_text() for m in modes}
        scores = {m: 100 * max(f1(answer, ref) for ref in row['answers'])
                  for m, answer in answers.items()}
        result.append(dict(**row, predictions=answers, f1=scores))
    return result


def metrics(rows, budget):
    mode = f'b{budget}'
    scores = [row['f1'][mode] for row in rows]
    deltas = [score - row['f1']['dense'] for score, row in zip(scores, rows)]
    return dict(budget=budget, f1=statistics.mean(scores), delta=statistics.mean(deltas),
                worse=sum(d < -1e-9 for d in deltas), large_drop=sum(d < -20 for d in deltas))


def chart(path, tune, validation):
    series = [(tune, '#1765a6', '调参集', 'circle'),
              (validation, '#a64720', '留出集', 'square')]
    series = [item for item in series if item[0]]
    width, height = 1120, 520
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        '<title>Quest 预算与 LongBench 答题得分</title>',
        '<desc>左图为平均 F1，右图为相对各自 Dense 基线的分差。</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g font-family="system-ui, sans-serif" fill="#172432" font-size="15">',
    ]

    def label(x, y, text, extra=''):
        svg.append(f'<text x="{x}" y="{y}" {extra}>{text}</text>')

    label(36, 39, 'Quest 预算与 LongBench 答题得分', 'font-size="26" font-weight="700"')
    label(36, 67, 'MultiFieldQA-en · Qwen3-0.6B F32 · 固定 16 token/块')
    for idx, (rows, color, name, marker) in enumerate(series):
        x = 38 + 255 * idx
        svg.append(f'<path d="M{x},{94}h24" stroke="{color}" stroke-width="3"/>')
        label(x + 32, 99, f'{name}（{len(rows)} 题）')
    label(548, 99, '虚线表示 Dense 基线', 'fill="#485867"')

    values = {
        name: [metrics(rows, b) for b in BUDGETS if f'b{b}' in rows[0]['f1']]
        for rows, _, name, _ in series
    }
    dense = {name: statistics.mean(row['f1']['dense'] for row in rows)
             for rows, _, name, _ in series}
    absolute_max = max([*dense.values(), *(v['f1'] for group in values.values() for v in group)])
    deltas = [v['delta'] for group in values.values() for v in group]
    for panel, field in enumerate(('f1', 'delta')):
        left, top, w, h = 90 + panel * 550, 153, 400, 250
        if panel == 0:
            lo, hi, step = 0, max(10, 10 * math.ceil((absolute_max + 3) / 10)), 10
        else:
            lo = min(-5, 5 * math.floor((min(deltas) - 2) / 5))
            hi = max(5, 5 * math.ceil((max(deltas) + 2) / 5))
            step = 5
        y = lambda v: top + h * (hi - v) / (hi - lo)
        xs = {b: left + 22 + i * (w - 44) / 3 for i, b in enumerate(BUDGETS)}
        label(left, 138, '平均 QA F1（满分 100）' if panel == 0 else '相对本组 Dense 的 F1 分差',
              'font-weight="650"')
        for tick in range(lo, hi + 1, step):
            yy = y(tick)
            svg.append(f'<path d="M{left},{yy:.1f}h{w}" stroke="#e1e6eb"/>')
            label(left - 10, yy + 5, str(tick), 'text-anchor="end" fill="#485867"')
        for budget, x in xs.items():
            label(x, 425, f'{budget:,}', 'text-anchor="middle"')
            label(x, 447, f'{budget // 16} 块', 'text-anchor="middle" font-size="13" fill="#485867"')
        label(left + w / 2, 481, '每个 query head 的 token 预算', 'text-anchor="middle"')
        if panel == 1:
            svg.append(f'<path d="M{left},{y(0):.1f}h{w}" stroke="#687684" stroke-dasharray="6 5"/>')
        for _, color, name, marker in series:
            if panel == 0:
                yy = y(dense[name])
                svg.append(f'<path d="M{left},{yy:.1f}h{w}" stroke="{color}" stroke-opacity=".65" stroke-dasharray="6 5"/>')
                label(left + w + 7, yy + 4, f'{dense[name]:.2f}', f'fill="{color}" font-size="12"')
            points = [(xs[v['budget']], y(v[field]), v[field]) for v in values[name]]
            coords = ' '.join(f'{x:.1f},{yy:.1f}' for x, yy, _ in points)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.8"/>')
            for x, yy, score in points:
                if marker == 'circle':
                    svg.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="4.5" fill="{color}" stroke="#fff"/>')
                else:
                    svg.append(f'<rect x="{x-4:.1f}" y="{yy-4:.1f}" width="8" height="8" fill="{color}" stroke="#fff"/>')
                # Label the higher point above and the lower point below.
                budget = next(b for b, xx in xs.items() if xx == x)
                other = [v[field] for key, group in values.items() if key != name
                         for v in group if v['budget'] == budget]
                above = not other or score > other[0] or (score == other[0] and marker == 'square')
                offset = -11 if above else 18
                value = f'{score:.2f}' if panel == 0 else f'{score:+.2f}'
                label(x, yy + offset, value,
                      f'text-anchor="middle" fill="{color}" font-size="12" stroke="#fff" stroke-width="3" paint-order="stroke"')
    svg.append('</g></svg>')
    path.write_text('\n'.join(svg) + '\n')

def report(root, tune, validation, selection):
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    chart(figures / 'budget_f1.svg', tune, validation)
    data = [('调参集', tune), ('留出集', validation)]
    lines = ['# Quest 预算扫描结果', '',
             'Qwen3-0.6B F32 · LongBench MultiFieldQA-en · 固定 16 token/块。', '',
             '[平均 F1 与相对 Dense 分差图](figures/budget_f1.svg) · '
             '[逐题答案和分数](scores/per_sample.json) · [原始预测](predictions/) · '
             '[抽样与选参记录](metadata/)', '',
             '| 集合 | 预算 | 平均 F1 | 相对 Dense | 掉分题 | 掉分超过 20 的题 |',
             '|---|---:|---:|---:|---:|---:|']
    for group, rows in data:
        dense = statistics.mean(row['f1']['dense'] for row in rows)
        label = f'{group}（{len(rows)} 题）'
        lines.append(f'| {label} | Dense | {dense:.2f} | 0.00 | 0 | 0 |')
        for budget in BUDGETS:
            if f'b{budget}' not in rows[0]['f1']:
                continue
            value = metrics(rows, budget)
            lines.append(f'| {label} | {budget:,} | {value["f1"]:.2f} | {value["delta"]:+.2f} | '
                         f'{value["worse"]} | {value["large_drop"]} |')
    lines += ['', f'调参阶段锁定 **{selection["selected"]:,} token**。'
              f'调参门槛{"通过" if selection["passed"] else "未通过"}。'
              '门槛为平均 F1 降幅不超过 2 分且无单题降幅超过 20 分。']
    if all(f'b{b}' in validation[0]['f1'] for b in BUDGETS):
        lines += ['', '留出集四档使用同一批五题。'
                  '查看留出集后比较各档属于补充分析，原调参选择仍记录在 metadata 中。']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        redraw = argparse.ArgumentParser(description='Redraw from saved scores without running inference')
        redraw.add_argument('results', type=Path)
        root = redraw.parse_args(sys.argv[2:]).results
        rows = json.loads((root / 'scores/per_sample.json').read_text())
        selection = json.loads((root / 'metadata/selection.json').read_text())
        report(root, rows['tuning'], rows['validation'], selection)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runner', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('--full-validation', action='store_true',
                        help='evaluate every budget on the same five held-out questions')
    args = parser.parse_args()
    root = args.results
    for folder in ('metadata', 'predictions', 'scores'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    refs = json.loads((args.cases / 'references.json').read_text())
    inventory = root / 'metadata/prompt_inventory.csv'
    with inventory.open('w') as stream:
        subprocess.run([args.runner, args.model, 'inventory', *sorted(args.cases.glob('*.txt'))],
                       stdout=stream, check=True)
    groups = choose(inventory, refs)
    manifest = json.dumps(groups, ensure_ascii=False, indent=2) + '\n'
    manifest_path = root / 'metadata/splits.json'
    if manifest_path.exists() and manifest_path.read_text() != manifest:
        raise ValueError('existing split differs from computed split')
    manifest_path.write_text(manifest)

    def run(group, budget):
        directory = root / 'predictions' / group
        directory.mkdir(exist_ok=True)
        env = dict(os.environ, INFERLLM_THREADS='8')
        env.pop('QUEST_VALIDATION', None)
        locked_only = group == 'validation' and not args.full_validation
        if locked_only:
            env['QUEST_VALIDATION'] = '1'
        prompts = [args.cases / f'{row["sample"]}.txt' for row in groups[group]]
        subprocess.run([args.runner, args.model, directory, 'sweep', '16', str(budget), *prompts],
                       env=env, check=True)
        return evaluate(groups[group], directory, [budget] if locked_only else BUDGETS)

    print('Tuning split:', [row['sample'] for row in groups['tuning']], flush=True)
    print('Validation split:', [row['sample'] for row in groups['validation']], flush=True)
    tune = run('tuning', 2048)
    options = [metrics(tune, b) for b in BUDGETS]
    acceptable = [o for o in options if o['delta'] >= -2 and o['large_drop'] == 0]
    selected = acceptable[0] if acceptable else max(options, key=lambda o: (o['delta'], -o['large_drop'], o['budget']))
    budget = selected['budget']
    selection = dict(selected=budget, passed=bool(acceptable), tuning=options)
    (root / 'metadata/selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(f'Locked {budget} tokens; tuning gate passed: {bool(acceptable)}', flush=True)
    validation = run('validation', budget)
    (root / 'scores/per_sample.json').write_text(json.dumps(dict(tuning=tune, validation=validation),
                                                     ensure_ascii=False, indent=2) + '\n')
    report(root, tune, validation, selection)


if __name__ == '__main__':
    main()
