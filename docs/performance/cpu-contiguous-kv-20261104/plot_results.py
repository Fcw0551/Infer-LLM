"""Plot the archived four-profile KV-cache measurements; does not run inference."""
import csv
import json
import math
from pathlib import Path
import statistics as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
X = [32, 64, 128, 256]
rows = {}
summary = {}
for p in X:
    with (ROOT / f'p{p}.csv').open() as f:
        runs = list(csv.DictReader(f))
    assert len(runs) == 3, (p, len(runs))
    assert len({r['token_checksum'] for r in runs}) == 1, 'Repeated token outputs differ'
    for r in runs:
        assert r['test'] == 'pg-kv'
        assert int(r['threads']) == 8
        assert int(r['prompt_tokens']) == p and int(r['generated_tokens']) == 128
        assert int(r['evaluated_tokens']) == p + 127
        assert math.isclose(float(r['request_s']), float(r['ttft_s'])+127*float(r['tpot_ms'])/1000, abs_tol=1e-6)
        assert math.isclose(float(r['decode_tokens_per_second'])*float(r['tpot_ms']),1000,abs_tol=1e-6)
    rows[p] = runs
    metrics = {}
    for key in ('ttft_s','tpot_ms','decode_tokens_per_second','request_s','build_s','execute_s','sample_s','cleanup_s','wall_s'):
        values = [float(r[key]) for r in runs]
        metrics[key] = {'mean': stats.mean(values), 'sd': stats.stdev(values)}
    metrics['peak_rss_mib'] = max(float(r['peak_rss_mib']) for r in runs)
    metrics['token_checksum'] = runs[0]['token_checksum']
    summary[p] = metrics
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                    'axes.spines.top':False,'svg.fonttype':'path',
                    'figure.facecolor':'white','axes.facecolor':'#fafbfd'})
FOOT = 'KV cache | 128 output tokens | 1 warmup + 3 measured requests per point\nPoints: mean; error bars: sample SD. Lines connect measured input lengths only.'

def setup(ax):
    ax.set_xticks(X)
    ax.set_xlim(16,276)
    ax.set_xlabel('Initial input length (tokens)')
    ax.grid(axis='both',color='#dce2ea',linewidth=.7)
    ax.set_axisbelow(True)

def save(fig,name):
    for ext in ('svg','png'):
        fig.savefig(ROOT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

def latency(name,title,key,scale,label,color):
    ys = [summary[p][key]['mean']*scale for p in X]
    sd = [summary[p][key]['sd']*scale for p in X]
    fig,ax=plt.subplots(figsize=(10,6))
    fig.subplots_adjust(left=.12,right=.97,top=.84,bottom=.23)
    setup(ax)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=17,pad=20)
    ax.set_ylabel(label)
    ax.set_ylim(0,max(y+s for y,s in zip(ys,sd))*1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
    ax.errorbar(X,ys,yerr=sd,color=color,marker='o',markersize=7,linewidth=2.3,capsize=5)
    for x,y,s in zip(X,ys,sd):
        ax.annotate(f'{y:,.2f}',(x,y+s),xytext=(0,12),textcoords='offset points',ha='center',color=color,fontweight='bold',fontsize=10)
    fig.text(.12,.055,FOOT,fontsize=9,color='#536174',linespacing=1.6)
    save(fig,name)

latency('ttft','Time to first token (TTFT)','ttft_s',1000,'TTFT (ms)','#2563eb')
latency('tpot','Average decode time per token (TPOT)','tpot_ms',1,'TPOT (ms/token, excluding first token)','#b45309')

fig,ax=plt.subplots(figsize=(12,8))
fig.subplots_adjust(left=.12,right=.87,top=.76,bottom=.23)
setup(ax)
ax.set_yscale('log')
ax.set_ylabel('Mean phase time (ms/step) - LOG SCALE')
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
ax.minorticks_off()
lines=[]
for label,key,color,marker in [('Build','build_s','#2563eb','o'),('Read logits / sample','sample_s','#16a34a','s'),('Cleanup','cleanup_s','#a16207','^'),('Schedule / execute','execute_s','#9333ea','D')]:
    y=[summary[p][key]['mean']*1000/128 for p in X]
    line,=ax.plot(X,y,color=color,marker=marker,linewidth=2,label=label)
    lines.append(line)
    for x,v in zip(X,y):
        ax.annotate(f'{v:.3f}',(x,v),xytext=(0,-16 if key == 'build_s' else 9),textcoords='offset points',ha='center',fontsize=9,color=color)
ax2=ax.twinx()
rss=[summary[p]['peak_rss_mib'] for p in X]
line,=ax2.plot(X,rss,color='#db2777',marker='x',linestyle='--',linewidth=2,label='Peak RSS')
lines.append(line)
ax2.set_ylabel('Peak RSS (MiB)',color='#db2777')
ax2.set_ylim(0,max(rss)*1.25)
for x,v in zip(X,rss):
    ax2.annotate(f'{v:.1f}',(x,v),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9,color='#db2777')
ax.set_title('Phase costs and peak resident memory',loc='left',fontweight='bold',fontsize=17,pad=58)
fig.legend(lines,[x.get_label() for x in lines],loc='upper center',bbox_to_anchor=(.50,.87),ncol=3,frameon=False)
fig.text(.12,.055,'Phase costs: cumulative time / 128 steps, averaged over 3 requests.\nRSS: process high-water mark including warmup. Left and right axes are independent.',fontsize=9,color='#536174',linespacing=1.6)
save(fig,'resources')
print(json.dumps(summary,indent=2))
