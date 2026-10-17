"""Render measured MTP latency trajectories; never extrapolate the 64K arm."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
summary = json.loads((DATA / 'summary.json').read_text())
results = json.loads((DATA / 'results.json').read_text())
by_prompt = {s['prompt']: s for s in summary}
cases = {(c['input_target'], c['mtp']): c for c in results['cases']}
OFF, ON = '#475569', '#087e8b'
INK, MUTED, GRID = '#142638', '#586b7d', '#e4eaf0'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'text.color': INK, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': GRID, 'svg.fonttype': 'none',
    'savefig.facecolor': '#f8fafc',
})
fig = plt.figure(figsize=(14.5, 10.7), facecolor='#f8fafc')
grid = fig.add_gridspec(2, 2, left=.075, right=.955, top=.80, bottom=.30,
                        hspace=.30, wspace=.18, height_ratios=[1.15, 1])
fig.text(.075, .962, 'Faster decode has an upfront cost', fontsize=25, weight='bold', va='top')
fig.text(.075, .914, 'MTP completion-time crossover on an RTX 3070', fontsize=17, va='top')
fig.text(.075, .88, '8 GB VRAM  /  Ryzen 7 5800X  /  64 GB RAM  /  Flash-Next GSQ-RCO IQ3_S  /  INT8 KV',
         fontsize=12, color=MUTED, va='top')
fig.legend(handles=[Line2D([0], [0], color=OFF, lw=3, label='MTP off'),
                    Line2D([0], [0], color=ON, lw=3, label='MTP on')],
           loc='upper right', bbox_to_anchor=(.96, .927), ncol=2,
           frameon=False, fontsize=13, handlelength=1.5)

for column, prompt in enumerate([4096, 16384]):
    s = by_prompt[prompt]
    off = np.array(json.loads((DATA / f'{prompt}-off-arrivals.json').read_text()))
    on = np.array(json.loads((DATA / f'{prompt}-on-arrivals.json').read_text()))
    n = min(len(off), len(on))
    assert n == 8192
    x = np.arange(1, n + 1)
    gain = off[:n] - on[:n]
    # This is the final sign change, not a fitted intersection.
    nonwins = np.flatnonzero(gain <= 0)
    crossover = int(nonwins[-1] + 2)
    assert crossover == s['mtp_faster_after']
    assert (gain[crossover-1:] > 0).all()
    top, bottom = fig.add_subplot(grid[0, column]), fig.add_subplot(grid[1, column])
    for ax in [top, bottom]:
        ax.set_facecolor('white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=0, pad=9)
        ax.grid(axis='y', color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 8192)
        ax.set_xticks([0, 2048, 4096, 6144, 8192], ['0', '2,048', '4,096', '6,144', '8,192'])
    top.set_title(f'{prompt:,} input tokens', loc='left', fontsize=18, weight='bold', pad=16)
    top.set_ylim(0, 275)
    top.set_yticks([0, 60, 120, 180, 240])
    top.set_ylabel('Time to output token (seconds)')
    top.plot(x, off, color=OFF, lw=2.4)
    top.plot(x, on, color=ON, lw=2.4)
    top.text(.025, .94, f'Prefill  {s["off_prefill_s"]:.2f} → {s["on_prefill_s"]:.2f} s',
             transform=top.transAxes, color=MUTED, fontsize=12, va='top')
    top.scatter([crossover], [off[crossover-1]], color=INK, s=30, zorder=4)
    for values, color in [(off, OFF), (on, ON)]:
        top.annotate(f'{values[-1]:.1f} s', (8192, values[-1]),
                     xytext=(-10, 9), textcoords='offset points', ha='right',
                     color=color, weight='bold', fontsize=13)
    bottom.set_ylim(-18, 86)
    bottom.set_yticks([-15, 0, 20, 40, 60, 80])
    bottom.set_ylabel('Seconds saved by MTP on')
    bottom.set_xlabel('Generated output tokens', labelpad=12)
    bottom.axhline(0, color=INK, lw=1)
    bottom.fill_between(x, gain, 0, where=gain >= 0, color=ON, alpha=.10, linewidth=0)
    bottom.fill_between(x, gain, 0, where=gain < 0, color='#b45309', alpha=.12, linewidth=0)
    bottom.plot(x, gain, color=ON, lw=2.4)
    bottom.scatter([crossover], [gain[crossover-1]], color=INK, s=30, zorder=4)
    bottom.annotate(f'Catch-up ≈{crossover:,} tokens', (crossover, 0),
                    xytext=(2300 if prompt == 4096 else 2700, -12),
                    textcoords='data', fontsize=12, color=INK,
                    arrowprops={'arrowstyle': '-', 'color': MUTED, 'lw': 1})
    bottom.annotate(f'{gain[-1]:.1f} s saved', (8192, gain[-1]),
                    xytext=(-10, 9), textcoords='offset points', ha='right',
                    color=ON, weight='bold', fontsize=13)
    bottom.text(.025, .94, 'Above zero: MTP on finishes sooner', transform=bottom.transAxes,
                color=MUTED, fontsize=11, va='top')

fig.text(.075, .20, '64K input: long-output crossover unverified', fontsize=16, weight='bold', color='#9a4c12')
fig.text(.075, .165, 'At output token 512: off 147.2 s · on 175.7 s. The off arm ended at 759 tokens after repeated imports.',
         fontsize=12, color=MUTED)
fig.text(.075, .11, 'One measured run per condition · warm engines · no prompt reuse · output sequences differ',
         fontsize=12, color=MUTED)
fig.text(.075, .075, 'The curves are recorded token arrivals, not constant-TPS projections. No timing beyond the shared output range.',
         fontsize=11, color=MUTED)
fig.text(.075, .04, 'Strata 61b3fb5d · measured October 11–12, 2026 · comparison changes MTP and --spec 2 → 4',
         fontsize=11, color=MUTED)
fig.savefig(HERE / 'mtp-crossover.png', dpi=180)
fig.savefig(HERE / 'mtp-crossover.svg')
svg_path = HERE / 'mtp-crossover.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n',
                    encoding='utf-8', newline='\n')
plt.close(fig)
print('Rendered mtp-crossover.png and mtp-crossover.svg')
