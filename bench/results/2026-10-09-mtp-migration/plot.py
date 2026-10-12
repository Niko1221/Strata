"""Regenerate figures from the published measurements, without rounded inputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'cuda-256k-to-1m-results.json').read_text())

def row(kind, label=None):
    matches = [r for r in data if r['kind'] == kind and (label is None or r.get('label') == label)]
    assert len(matches) == 1, (kind, label)
    return matches[0]

def duration(record):
    # Migration has a client wall clock; other slots expose native milliseconds.
    return record['wall_s'] if 'wall_s' in record else record['ms'] / 1000

fresh = [row('generate', 'A-prefix')['ttft_s']]
suffix = [0.0]
restore = [0.0]
for n in (524288, 1048576):
    fresh.append(row('generate', f'A-extend-{n}-prefix')['ttft_s'])
    suffix.append(row('generate', f'B-extend-{n}-prefix')['ttft_s'])
    restore.append(duration(row('extension-restore', f'B-extend-{n}')))
migration = duration(row('B-migration'))
save = duration(row('save-source'))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(13.5, 8.4), facecolor='#f5f7fb')
ax.set_facecolor('#f5f7fb')
colors = {'fresh':'#315c9b', 'convert':'#16998a', 'suffix':'#9ad9ce'}
ys = np.arange(3) * 2.5
for i, y in enumerate(ys):
    ax.barh(y-.42, fresh[i], height=.64, color=colors['fresh'])
    ax.text(fresh[i]+2, y-.42, f'{fresh[i]:.1f}s', va='center', fontweight='bold')
    ax.barh(y+.42, migration, height=.64, color=colors['convert'])
    ax.barh(y+.42, suffix[i], left=migration, height=.64, color=colors['suffix'])
    total = migration + suffix[i]
    ax.text(total+2, y+.42, f'{total:.1f}s' + (' *' if i else ''), va='center', fontweight='bold')
ax.set_yticks(ys, ['256K\nconverted prefix', '512K\n256K prefix + new suffix', '1M\n256K prefix + new suffix'])
ax.invert_yaxis()
ax.set_xlim(0, max(fresh)*1.17)
ax.set_xlabel('Seconds (lower is better)')
ax.xaxis.grid(True, alpha=.16)
ax.set_axisbelow(True)
fig.suptitle('Reuse the old 256K. Prefill the new suffix.', x=.04, y=.96, ha='left', fontsize=22, fontweight='bold', color='#172235')
fig.text(.04,.903,'Experimental RoPE → YaRN 4× conversion with MTP 4  |  RTX PRO 6000 · ISTA IQ3_XXS · FP16 KV',fontsize=11)
fig.legend(handles=[Patch(color=colors['fresh'],label='Fresh YaRN prefill'),
                   Patch(color=colors['convert'],label='First conversion + restore'),
                   Patch(color=colors['suffix'],label='New-suffix prefill')],loc='upper left',bbox_to_anchor=(.21,.875),ncol=3,frameon=False)
fig.text(.04,.175, 'All 12 retrieval replies recovered all three markers. History remains approximate.',fontweight='bold',fontsize=12)
fig.text(.04,.132, 'Against fresh YaRN, 32 forced tokens per size: top-token agreement 32/32 · 31/32 · 31/32.\n'
                   'Mean KL: 0.01282 · 0.00613 · 0.00508. Exact-replay controls matched on CUDA.',fontsize=10.5)
fig.text(.04,.063, f'n=1 per condition; warm filesystem; startup and original ordinary prefill excluded. Add {save:.2f}s if source SAVE is needed.\n'
                    '* 512K/1M conversion + suffix totals sum separate measurements, not timed end-to-end requests.\n'
                    'First conversion includes model fingerprinting. These synthetic cases do not establish broad task quality.',fontsize=9,color='#475569')
fig.subplots_adjust(left=.22,right=.96,top=.82,bottom=.27)
fig.savefig(ROOT/'overview.png',dpi=160)
fig.savefig(ROOT/'overview.svg')
svg = ROOT/'overview.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n', encoding='utf-8')
print(json.dumps({'fresh_s':fresh, 'conversion_restore_s':migration, 'suffix_s':suffix, 'source_save_s':save}))
