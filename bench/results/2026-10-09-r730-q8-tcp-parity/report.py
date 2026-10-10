"""Audit saved token IDs and reproduce the summary and chart. Requires matplotlib."""
import json
import hashlib
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
labels = ['matched-local', 'matched-loopback']
sizes = [128, 1024, 8192]
prompts = json.loads((ROOT / 'prompts.json').read_text())
rows = {}
for label in labels:
    data = json.loads((ROOT / (label + '.json')).read_text())
    assert data[-1]['kind'] == 'complete'
    requests = [x for x in data if x['kind'] == 'request']
    assert len(requests) == 9
    rows[label] = {(x['tokens'], x['repeat']): x for x in requests}
    assert set(rows[label]) == {(n, r) for n in sizes for r in range(3)}
    for n in sizes:
        first = rows[label][n, 0]
        for r in range(3):
            x = rows[label][n, r]
            assert x['reused_tokens'] == 0 and len(x['output_ids']) == x['output_tokens'] == 128
            assert x['input_sha256'] == first['input_sha256']
            assert len(prompts[str(n)]) == n
            assert x['input_sha256'] == hashlib.sha256(json.dumps(prompts[str(n)]).encode()).hexdigest()
            assert x['output_ids'] == first['output_ids']
            assert abs(x['decode_tps'] - 127 / (x['total_s'] - x['ttft_s'])) < 1e-8
for key, local in rows[labels[0]].items():
    remote = rows[labels[1]][key]
    assert local['input_sha256'] == remote['input_sha256']
    assert local['output_ids'] == remote['output_ids']

summary = []
for n in sizes:
    for label in labels:
        row = {'input_tokens': n, 'mode': label, 'n': 3}
        for metric in ['decode_tps', 'ttft_s', 'total_s']:
            values = [rows[label][n, r][metric] for r in range(3)]
            row[metric] = {'median': median(values), 'min': min(values), 'max': max(values)}
        summary.append(row)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={'width_ratios': [1.12, 1]})
fig.patch.set_facecolor('#f8fafc')
colors = ['#176b87', '#d96a2b']
names = ['In-process split', 'TCP loopback stages']
for ax, metric, title, ylabel in zip(axes, ['decode_tps', 'ttft_s'],
                                    ['Decode throughput', 'Time to first token'],
                                    ['Output tokens / second · higher is better', 'Seconds · lower is better']):
    for j, label in enumerate(labels):
        vals = [[rows[label][n, r][metric] for r in range(3)] for n in sizes]
        meds = [median(v) for v in vals]
        xs = [i + (j - .5) * .36 for i in range(3)]
        ax.bar(xs, meds, width=.32, color=colors[j], label=names[j], zorder=3)
        for x, m, v in zip(xs, meds, vals):
            ax.plot([x, x], [min(v), max(v)], color='#162433', linewidth=1.2, zorder=4)
            ax.text(x, m + (.2 if metric == 'decode_tps' else 1.1), f'{m:.2f}',
                    ha='center', fontsize=10, weight='bold', color='#162433')
    ax.set_xticks(range(3), ['128', '1,024', '8,192'])
    ax.set_xlabel('Input tokens')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', weight='bold', pad=15)
    ax.set_ylim(0, 13.2 if metric == 'decode_tps' else 76)
    ax.grid(axis='y', color='#e2e8f0', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
fig.suptitle('Three Tesla P4s: exact tokens over TCP, with lower decode throughput',
             x=.065, y=.99, ha='left', fontsize=17, weight='bold', color='#152f45')
fig.text(.065, .905, 'Q8_0 · FP16 KV · same GPUs / layers / expert placement · 9 of 9 output sequences identical',
         color='#23634c', fontsize=11)
handles, leglabels = axes[0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
fig.text(.065, .055, 'Median and min–max of 3 runs per length; 128 generated tokens. Decode excludes TTFT; model loading excluded.', fontsize=9, color='#475569')
fig.text(.065, .02, 'TCP is localhost on one R730, not a LAN speedup test. No MTP, prefix reuse or pipeline windows. Measured 2026-10-09.', fontsize=9, color='#475569')
fig.subplots_adjust(top=.8, bottom=.24, left=.075, right=.97, wspace=.3)
fig.savefig(ROOT / 'comparison.png', dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print('PASS: 9 cross-mode sequences and 12 repeat comparisons; all 128 tokens identical.')
for n in sizes:
    a = median(rows[labels[0]][n, r]['decode_tps'] for r in range(3))
    b = median(rows[labels[1]][n, r]['decode_tps'] for r in range(3))
    print(f'{n}: local {a:.3f}, TCP {b:.3f} tok/s; TCP {100*(1-b/a):.1f}% lower')
