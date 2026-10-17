"""Render latency and relative speedup together, using measured summary files."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', type=Path, help='disk-restore directory with short/long summary.json')
    args = ap.parse_args()
    rows = []
    for group in ('short', 'long'):
        rows += json.loads((args.results / group / 'summary.json').read_text())
    values = {(r['mode'], r['prefix_tokens']): r['ttft_s_p50'] for r in rows if r['build'] == 'pr1489'}
    targets = sorted({n for _, n in values})
    colors = {'cold': '#CF612B', 'pinned': '#087C78', 'disk_restore': '#386FB4'}
    labels = {'cold': 'Read entire prompt', 'pinned': 'Reuse live prefix', 'disk_restore': 'Restore file + request'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': '#213C50', 'axes.labelcolor': '#213C50',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#B4C4D0'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 10), sharey=True,
                                      gridspec_kw={'width_ratios': [1.12, 1]})
    fig.patch.set_facecolor('#F7FAFC')
    for ax in (left, right):
        ax.set_facecolor('#F7FAFC')
        ax.set_xscale('log')
        ax.set_ylim(10.7, -.7)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.grid(axis='x', which='major', color='#CAD6DE', alpha=.6, lw=.7)
        ax.set_axisbelow(True)
        for i in range(len(targets)):
            if i % 2 == 0:
                ax.axhspan(i-.48, i+.48, color='#EAF0F5', alpha=.6, zorder=0)
        ax.axhline(7.5, color='#94A9B8', lw=1, linestyle=(0, (4, 4)))
    left.set_yticks(range(len(targets)), [f'{n//1024}K' for n in targets])
    left.set_ylabel('Unchanged prefix', labelpad=12)
    left.set_xlim(.16, 40)
    left.set_xticks([.2, .5, 1, 2, 5, 10, 25], ['0.2', '0.5', '1', '2', '5', '10', '25'])
    left.set_xlabel('Seconds to first text token · logarithmic scale\nLess time is better', labelpad=15)
    right.set_xlim(.58, 75)
    right.set_xticks([1, 2, 5, 10, 20, 50], ['1×', '2×', '5×', '10×', '20×', '50×'])
    right.set_xlabel('Full-prefill time ÷ cached-path time · logarithmic scale\nMore speedup is better', labelpad=15)
    right.axvspan(.58, 1, color='#F5DACC', alpha=.55)
    right.axvline(1, color='#6D7C86', lw=1.2)
    right.text(1.04, -.49, 'BREAK-EVEN', fontsize=8, color='#697B88')
    left.set_title('Actual waiting time', loc='left', weight='bold', size=15, pad=18)
    right.set_title('Benefit relative to reading the whole prompt', loc='left', weight='bold', size=14, pad=18)
    for i, n in enumerate(targets):
        for mode, offset in [('cold', -.26), ('pinned', 0), ('disk_restore', .26)]:
            x = values[mode, n]
            left.scatter(x, i+offset, s=31, color=colors[mode], zorder=4)
            left.annotate(f'{x:.3f}', (x, i+offset), xytext=(7, -3), textcoords='offset points',
                          size=8.5, color=colors[mode])
        cold = values['cold', n]
        for mode, offset in [('pinned', -.16), ('disk_restore', .16)]:
            ratio = cold / values[mode, n]
            right.plot([1, ratio], [i+offset]*2, color=colors[mode],
                       lw=3.5 if mode == 'disk_restore' else 1.2,
                       alpha=1 if mode == 'disk_restore' else .4)
            right.scatter(ratio, i+offset, s=48 if mode == 'disk_restore' else 24, color=colors[mode], zorder=4)
            right.annotate(f'{ratio:.2f}×', (ratio, i+offset), xytext=(7, -11 if ratio < 1 else -3), textcoords='offset points',
                           size=9, color=colors[mode])
    handles = [Line2D([0], [0], color=colors[m], marker='o', lw=0, label=labels[m])
               for m in ('disk_restore', 'cold', 'pinned')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055, .902), ncol=3,
               frameon=False, fontsize=11, handletextpad=.5, columnspacing=2)
    fig.suptitle('Disk restore: 11.6× faster first token at 128K', x=.058, y=.981,
                 ha='left', fontsize=24, weight='bold')
    fig.text(.059, .942, '22.611 s full prefill → 1.956 s restore + first token  •  Single 128K example; warm OS page cache', size=11)
    fig.text(.059, .916, 'Integration #1489  •  ISTA IQ2_XS  •  RTX PRO 6000 Blackwell  •  FP16 KV  •  128-token replies', size=10)
    fig.text(.06, .087, '1K–8K: three-sample medians. 32K–128K: one example each. Dashed divider also marks 32K → 256K capacity.', size=9)
    fig.text(.06, .063, 'Disk series includes RESTORE + request, with warm OS page cache. Initial SAVE / prefill and model loading are excluded.', size=9)
    fig.text(.06, .037, 'Main: live-prefix results are similar; session SAVE failed at all 11 sizes in this non-MTP configuration. No disk latency is inferred.',
             size=9, color='#963724')
    fig.subplots_adjust(left=.075, right=.96, top=.827, bottom=.185, wspace=.18)
    for ext in ('png', 'svg'):
        path = args.results / ('overview.' + ext)
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    main()
