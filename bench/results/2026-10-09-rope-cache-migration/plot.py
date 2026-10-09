"""Regenerate the figure from recorded observations (requires matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
cases = json.loads((root / 'measurements.json').read_text())['cases']
fresh, migrated, kl, agreement = [], [], [], []
for case in cases:
    rows = case['rows']
    fresh.append(next(r['total_s'] for r in rows if r.get('label') == 'A-prefix'))
    migrated.append(next(r['wall_s'] for r in rows if r['kind'] == 'B-migration'))
    quality = next(r for r in rows if r['kind'] == 'quality' and r['reference'] == 'A' and r['candidate'] == 'B')
    kl.append(quality['mean_kl'])
    agreement.append(quality['top_agreement'])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={'width_ratios': [1.3, 1]})
fig.patch.set_facecolor('#f8fafc')
fig.suptitle('Saved-cache migration: less replay, approximate state', fontsize=18, fontweight='bold', x=.05, ha='left')
fig.text(.05, .88, 'ISTA IQ3_XXS | FP16 KV | table-based RoPE | one run per condition', color='#526173')
y = np.arange(len(cases))
ax = axes[0]
ax.barh(y-.18, fresh, height=.32, color='#334e78', label='Fresh YaRN prefix + first token')
ax.barh(y+.18, migrated, height=.32, color='#148c86', label='Convert + restore existing cache')
for i, (a, b) in enumerate(zip(fresh, migrated)):
    ax.text(a+.5, i-.18, f'{a:.2f}s', va='center', fontsize=9)
    ax.text(b+.5, i+.18, f'{b:.2f}s', va='center', fontsize=9)
ax.set_yticks(y, [c['label'] for c in cases]); ax.invert_yaxis()
ax.set_xlim(0, max(fresh+migrated)*1.2); ax.set_xlabel('Seconds (lower is better)')
ax.set_title('First-use cost, including model fingerprint', loc='left', fontsize=11)
ax.legend(loc='lower right', fontsize=8, frameon=False)
ax.xaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
ax = axes[1]
ax.barh(y, kl, height=.48, color='#c17f28')
for i, (k, a) in enumerate(zip(kl, agreement)):
    ax.text(k+.0003, i, f'{k:.4f} KL\n{a:.1%} top-token agreement', va='center', fontsize=9)
ax.set_yticks(y, [c['label'] for c in cases]); ax.invert_yaxis()
ax.set_xlim(0, max(kl)*1.8); ax.set_xlabel('Mean KL divergence (0 = identical distributions)')
ax.set_title('Converted vs fresh YaRN: 32 forced positions', loc='left', fontsize=11)
ax.xaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
fig.text(.05, .105, 'All paths retrieved the three codes correctly. These are repeated-filler probes, not broad quality evaluations.', fontsize=10)
fig.text(.05, .055, 'Engine startup and source SAVE excluded. Migration includes first-use fingerprint, read, conversion, write and restore.\n'
         'Source snapshots contain 4,095 / 262,143 tokens; the prompt boundary is replayed under YaRN before continuation.', fontsize=9, color='#526173')
fig.subplots_adjust(left=.10, right=.98, top=.78, bottom=.24, wspace=.38)
fig.savefig(root/'overview.png', dpi=180, facecolor=fig.get_facecolor())
fig.savefig(root/'overview.svg', facecolor=fig.get_facecolor())

svg = root / "overview.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
