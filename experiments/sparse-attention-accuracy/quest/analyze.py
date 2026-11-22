#!/usr/bin/env python3
"""Summarize observed attention mass and score paired LongBench answers."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import runpy
import statistics


def read_curves(paths):
    groups = defaultdict(list)
    cases = defaultdict(dict)
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                phase, method = row['phase'], row['method']
                layer, head, length, block, budget, read = (
                    int(row[field]) for field in ('layer', 'head', 'length', 'block', 'budget', 'read'))
                mass = float(row['mass'])
                assert 0 < read <= min(length, budget * (2 if method == 'union' else 1))
                assert math.isfinite(mass) and 0 <= mass <= 1.0001
                key = (phase, method, block, budget)
                groups[key].append((read, min(mass, 1)))
                identity = (path.stem, phase, layer, head, length, block, budget)
                if method in ('oracle', 'quest'):
                    assert method not in cases[identity]
                    cases[identity][method] = mass
                if method != 'union' and budget >= length + block - 1:
                    assert abs(mass - 1) < 1e-4
    assert cases
    for pair in cases.values():
        assert pair['oracle'] + 1e-5 >= pair['quest']
    rows = []
    for (phase, method, block, budget), vals in sorted(groups.items()):
        reads, masses = zip(*vals)
        rows.append(dict(phase=phase, method=method, block=block, budget=budget,
                         median_read=statistics.median(reads), count=len(vals),
                         median_mass=statistics.median(masses),
                         p10_mass=sorted(masses)[int(.1 * (len(masses) - 1))]))
    return rows


def plot(rows, path):
    colors = {1: '#303030', 16: '#1468a0', 32: '#af4300', 64: '#4d7627'}
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="730" viewBox="0 0 1060 730">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<g font-family="sans-serif" font-size="13" fill="#252525">']

    def label(x, y, value, extra=''):
        elements.append(f'<text x="{x}" y="{y}" {extra}>{value}</text>')

    label(530, 32, 'Retained dense attention mass vs. selected tokens per query head',
          'text-anchor="middle" font-size="20"')
    for ix, (phase, metric) in enumerate((p, m) for m in ('median_mass', 'p10_mass')
                                         for p in ('question', 'answer')):
        x, y = 65 + (ix % 2) * 510, 79 + (ix // 2) * 290
        w, h = 435, 216
        label(x, y - 12, f'{phase.title()} · {"median" if metric == "median_mass" else "10th percentile"}')
        for value in (0, .25, .5, .75, 1):
            yy = y + h * (1 - value)
            elements.append(f'<path d="M{x},{yy}h{w}" stroke="#ddd"/>')
            label(x - 8, yy + 4, f'{value:.0%}', 'text-anchor="end"')
        for tokens in (16, 64, 256, 1024, 4096):
            xx = x + w * (math.log2(tokens) - 4) / 8
            label(xx, y + h + 20, str(tokens), 'text-anchor="middle"')
        for method, block in [('token', 1)] + [(m, b) for b in (16, 32, 64) for m in ('oracle', 'quest')]:
            selected = [r for r in rows if (r['phase'],r['method'],r['block']) == (phase,method,block)]
            points = ' '.join(f'{x+w*(math.log2(r["median_read"])-4)/8:.2f},{y+h*(1-r[metric]):.2f}'
                              for r in selected)
            dash = ' stroke-dasharray="6 4"' if method == 'oracle' else ''
            elements.append(f'<polyline points="{points}" fill="none" stroke="{colors[block]}" stroke-width="2"{dash}/>')
    label(65, 645, 'Black: Dense Top-token. Dashed: Dense Top-block. Solid colors: Quest Top-block.')
    for x, size in ((65,16),(215,32),(365,64)):
        label(x, 673, f'Block {size} tokens', f'fill="{colors[size]}"')
    label(65, 701, 'Both block methods reserve the current block. X uses the median actual selected token count.')
    elements.append('</g></svg>')
    path.write_text('\n'.join(elements))


def quality(ref_path, result):
    f1 = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'longbench.py'))['f1']
    refs = json.loads(ref_path.read_text())
    rows = []
    for file in sorted(result.glob('*.quest')):
        sample = file.stem
        ref = refs[sample]
        answers = {mode: (result / f'{sample}.{mode}').read_text() for mode in ('dense', 'quest', 'quest_half')}
        scores = {mode: max(f1(answer, target) for target in ref['answers']) * 100
                  for mode, answer in answers.items()}
        rows.append(dict(sample=sample, **ref, predictions=answers, f1=scores))
    assert rows and len(rows) == len(list(result.glob('*.dense')))
    (result / 'quality.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    curves = sub.add_parser('curves')
    curves.add_argument('directory', type=Path)
    qa = sub.add_parser('quality')
    qa.add_argument('references', type=Path)
    qa.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.mode == 'quality':
        for row in quality(args.references, args.directory):
            print(row['sample'], row['f1'])
    else:
        directory = args.directory
        rows = read_curves(sorted(directory.glob('[0-9]*.csv')))
        with (directory / 'coverage_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, directory / 'coverage.svg')
        for row in rows:
            if row['method'] in ('quest', 'union') and row['budget'] in (256, 512, 1024):
                print(row)
