#!/usr/bin/env python3
"""Prepare and score a paired MultiFieldQA-en pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import string


MODES = ('dense', 'recent512', 'sink4_recent508')


def normalize(text):
    text = ''.join(char for char in text.lower() if char not in string.punctuation)
    return re.sub(r'\b(a|an|the)\b', ' ', text).split()


def f1(answer, reference):
    predicted, expected = normalize(answer), normalize(reference)
    shared = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * shared / (len(predicted) + len(expected)) if shared else 0.0


def prepare(dataset, templates, output):
    import pyarrow.parquet as parquet
    rows = parquet.read_table(dataset).to_pylist()
    random.Random(20261120).shuffle(rows)
    template = json.loads(templates.read_text())['multifieldqa_en']
    output.mkdir()
    references = {}
    for index, row in enumerate(rows):
        name = f'{index:03d}'
        prompt = template.format(**row)
        (output / f'{name}.txt').write_text(
            f'<|im_start|>user\n{prompt}<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n', encoding='utf-8')
        references[name] = dict(source_id=row['_id'], question=row['input'], answers=row['answers'])
    (output / 'references.json').write_text(json.dumps(references, ensure_ascii=False, indent=2))


def score(references_path, output):
    references = json.loads(references_path.read_text())
    items = []
    for name, ref in references.items():
        files = [output / f'{name}.{mode}' for mode in MODES]
        if not any(file.exists() for file in files):
            continue
        if not all(file.exists() for file in files):
            raise ValueError(f'incomplete sample {name}')
        answers = {mode: file.read_text() for mode, file in zip(MODES, files)}
        scores = {mode: max(f1(answer, target) for target in ref['answers'])
                  for mode, answer in answers.items()}
        items.append(dict(sample=name, **ref, predictions=answers, f1=scores))
    if not items:
        raise ValueError('no completed paired outputs')
    (output / 'per_sample.json').write_text(json.dumps(items, ensure_ascii=False, indent=2))
    means = {mode: 100 * sum(item['f1'][mode] for item in items) / len(items) for mode in MODES}
    lines = ['# LongBench MultiFieldQA-en pilot', '', f'Paired samples: {len(items)}', '',
             '| Mode | QA F1 (0–100) | Δ vs dense | Worse / tied / better |',
             '|---|---:|---:|---:|']
    for mode, mean in means.items():
        diffs = [item['f1'][mode] - item['f1']['dense'] for item in items]
        changes = f'{sum(d < 0 for d in diffs)} / {sum(d == 0 for d in diffs)} / {sum(d > 0 for d in diffs)}'
        lines.append(f'| {mode} | {mean:.2f} | {mean-means["dense"]:+.2f} | {changes} |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, paths in [('prepare', ('dataset', 'templates', 'output')),
                           ('score', ('references', 'output'))]:
        subparser = commands.add_parser(command)
        for path in paths:
            subparser.add_argument(path, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.dataset, args.templates, args.output)
    else:
        score(args.references, args.output)
