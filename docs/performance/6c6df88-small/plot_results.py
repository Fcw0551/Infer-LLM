"""Render the archived C++ benchmark CSVs; no inference is run here.
Usage: MPLCONFIGDIR=/tmp/inferllm-mpl python plot_results.py
Requires matplotlib. Outputs SVG and PNG next to this file.
"""
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
X = [32, 64, 128, 256]
rows = []
for p in X:
    with (ROOT / f'p{p}.csv').open() as f:
        entries = list(csv.DictReader(f))
    assert len(entries) == 1
    r = entries[0]
    assert int(r['prompt_tokens']) == p and int(r['generated_tokens']) == 128
    assert math.isclose(float(r['request_s']), float(r['ttft_s']) + 127 * float(r['tpot_ms']) / 1000, abs_tol=1e-5)
    assert math.isclose(float(r['decode_tokens_per_second']) * float(r['tpot_ms']), 1000, abs_tol=1e-5)
    rows.append(r)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'svg.fonttype': 'path',
                     'figure.facecolor': 'white', 'axes.facecolor': '#fafbfd'})
FOOT = 'Qwen3-0.6B F32 | CPU, 8 workers + main | no KV cache | output: 128 tokens\nEngine: 6c6df88 | 1 run per point, no warmup | 2026-10-28'

def setup(ax):
    ax.set_xticks(X)
    ax.set_xlim(16, 276)
    ax.set_xlabel('Initial input length (tokens)')
    ax.grid(axis='both', color='#dce2ea', linewidth=.7)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(ROOT / f'{name}.{ext}', dpi=180, facecolor='white')
    plt.close(fig)

def latency(name, title, key, scale, label, color):
    y = [float(r[key]) * scale for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=.12, right=.97, top=.84, bottom=.23)
    setup(ax)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=17, pad=20)
    ax.set_ylabel(label)
    ax.set_ylim(0, max(y)*1.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.plot(X, y, color=color, marker='o', markersize=8, linewidth=2.3)
    for x, v in zip(X, y):
        ax.annotate(f'{v:,.1f}', (x, v), xytext=(0, 12), textcoords='offset points',
                    ha='center', color=color, fontweight='bold', fontsize=10)
    fig.text(.12, .055, FOOT, fontsize=9, color='#536174', linespacing=1.6)
    save(fig, name)

latency('ttft', 'Time to first token (TTFT)', 'ttft_s', 1000, 'TTFT (ms)', '#2563eb')
latency('tpot', 'Average time per output token (TPOT)', 'tpot_ms', 1, 'TPOT (ms/token, excluding first token)', '#b45309')

fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(left=.12, right=.88, top=.78, bottom=.23)
setup(ax)
ax.set_yscale('log')
ax.set_ylim(.05, 60000)
ax.set_yticks([.1, 1, 10, 100, 1000, 10000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
ax.set_ylabel('Mean phase time (ms/step) - LOG SCALE')
ax.minorticks_off()
right = ax.twinx()
right.set_ylim(0, 6500)
right.set_ylabel('Peak process RSS (MiB) - LINEAR SCALE', color='#be185d')
right.tick_params(axis='y', colors='#be185d')
right.spines['right'].set_color('#be185d')
series = [('build_s', 'Build / topology [left]', '#2563eb', 'o', -16),
          ('sample_s', 'Logits / sampling [left]', '#15803d', 's', 10),
          ('cleanup_s', 'Cleanup [left]', '#a16207', '^', -16),
          ('execute_s', 'Scheduler execution [left]', '#7c3aed', 'D', 10)]
handles = []
for key, label, color, marker, offset in series:
    y = [float(r[key])*1000/int(r['generated_tokens']) for r in rows]
    line, = ax.plot(X, y, label=label, color=color, marker=marker, markersize=7, linewidth=2)
    handles.append(line)
    for x, v in zip(X, y):
        ax.annotate(f'{v:,.2f}' if v >= 1 else f'{v:.3f}', (x, v),
                    xytext=(0, offset), textcoords='offset points', ha='center',
                    color=color, fontsize=9, fontweight='bold')
y = [float(r['peak_rss_mib']) for r in rows]
line, = right.plot(X, y, label='Peak RSS [right]', color='#be185d', marker='X',
                   markersize=8, linewidth=2, linestyle='--')
handles.append(line)
for x, v in zip(X, y):
    right.annotate(f'{v:,.1f}', (x, v), xytext=(0, 12), textcoords='offset points',
                   ha='center', color='#be185d', fontsize=9, fontweight='bold')
fig.suptitle('Phase timings and peak memory', x=.12, y=.96, ha='left', fontweight='bold', fontsize=18)
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .91), ncol=3,
           frameon=False, fontsize=10)
fig.text(.12, .12, 'Phase times = totals / 128 steps (including first step). Peak RSS is not averaged.\nTwo independent y-axes: curve heights across time and memory are not comparable.', fontsize=9, color='#536174', linespacing=1.6)
fig.text(.12, .035, FOOT, fontsize=9, color='#536174', linespacing=1.6)
save(fig, 'resources')
print('Validated 4 CSV rows; generated ttft, tpot, resources (SVG + PNG).')
