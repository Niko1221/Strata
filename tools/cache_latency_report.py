"""Summarize raw cache_latency.py observations and render publication charts.

Requires matplotlib for charts. Quantiles use the linear (Hyndman-Fan type 7)
empirical estimator. Never pool warm-ups, alternate requests, builds or modes.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

PERCENTILES = (50, 80, 90, 95, 99)
MODES = ['cold', 'pinned', 'disk_restore', 'ram_switch', 'disk_switch']
LABELS = {'cold': 'Cache disabled', 'pinned': 'Pinned prefix',
          'disk_restore': 'Disk restore + request', 'ram_switch': 'Switch / RAM', 'disk_switch': 'Switch / disk'}
COLORS = {'cold': '#D66A36', 'pinned': '#137B80', 'disk_restore': '#376EB2',
          'ram_switch': '#6D5CB5', 'disk_switch': '#B24777'}


def quantile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered)-1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi]-ordered[lo]) * (rank-lo)


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        if row['phase'] == 'measured':
            groups[(row['build'], row['mode'], row['prefix_tokens'])].append(row)
    result = []
    for (build, mode, prefix), group in sorted(groups.items()):
        good = [r for r in group if not r.get('error') and r['ttft_s'] is not None]
        fixed = [r for r in good if r['usage']['output_tokens'] == 128]
        hits = [r for r in good if r['usage']['input_tokens_details']['cached_tokens'] >= prefix]
        attempted = sum(r.get('attempted', r.get('restore', {}).get('status', 200) == 200) for r in group)
        item = dict(build=build, mode=mode, prefix_tokens=prefix, scheduled=len(group),
                    attempted=attempted, n=len(good), blocked=len(group)-attempted,
                    errors=len(group)-len(good), replies_128=len(fixed), full_prefix_hits=len(hits),
                    cache_hit_rate=len(hits)/len(good) if good else None,
                    input_tokens_min=min((r['usage']['input_tokens'] for r in good), default=None),
                    input_tokens_max=max((r['usage']['input_tokens'] for r in good), default=None),
                    cached_tokens_median=statistics.median(r['usage']['input_tokens_details']['cached_tokens'] for r in good) if good else None)
        for metric, subset in [('ttft_s', good), ('total_s', good), ('total_128_s', fixed)]:
            values = [r['total_s' if metric == 'total_128_s' else metric] for r in subset]
            for p in PERCENTILES:
                item[f'{metric}_p{p}'] = quantile(values, p)
        result.append(item)
    return result


def charts(summary, out, subtitle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#253D50', 'text.color': '#253D50',
                         'axes.edgecolor': '#BBC8CF', 'savefig.facecolor': '#F8FAFC'})
    builds = sorted({r['build'] for r in summary})
    targets = sorted({r['prefix_tokens'] for r in summary})
    counts = [r['n'] for r in summary if r['n'] > 0]
    minimum = min(counts, default=0)
    single = bool(counts) and all(n == 1 for n in counts)
    sample_label = str(minimum) if minimum == max(counts, default=0) else f'{minimum}–{max(counts, default=0)}'
    note = (f'n = {sample_label} per successful cell. '
            'Warm-ups excluded; cache misses retained. '
            + ('PRELIMINARY: too few samples for tail estimates.' if minimum < 100
               else 'p99 is exploratory (200 samples gives about two upper-tail observations).'))
    for metric, title, filename in [('ttft_s', 'Time to first token', 'ttft'),
                                     ('total_128_s', 'Time to complete 128 output tokens', 'completion')]:
        fig, axes = plt.subplots(1, len(builds), figsize=(14, 6), squeeze=False, sharey=True)
        fig.patch.set_facecolor('#F8FAFC')
        for ax, build in zip(axes[0], builds):
            ax.set_facecolor('#F8FAFC')
            for mode in MODES:
                points = sorted((r for r in summary if r['build'] == build and r['mode'] == mode
                                 and r[metric+'_p50'] is not None), key=lambda r: r['prefix_tokens'])
                if not points:
                    continue
                x = [r['prefix_tokens']/1024 for r in points]
                med = [r[metric+'_p50'] for r in points]
                ax.plot(x, med, color=COLORS[mode], marker='o', lw=2.4, label=LABELS[mode])
                if minimum >= 100:
                    high = [r[metric+'_p95'] for r in points]
                    ax.fill_between(x, med, high, color=COLORS[mode], alpha=.12)
                    ax.plot(x, high, color=COLORS[mode], lw=1, linestyle='--')
            ax.set_title('Strata main' if build == 'main' else 'Integration #1489', loc='left', weight='bold', pad=15)
            ax.set_xlabel('Pinned prefix length (Ki tokens)')
            ax.set_xticks([t/1024 for t in targets])
            ax.grid(axis='y', alpha=.18)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=9)
            failed = [r for r in summary if r['build'] == build and r['n'] == 0]
            if failed:
                modes = ', '.join(LABELS[m] for m in dict.fromkeys(r['mode'] for r in failed))
                ax.text(.02, .72, modes + ': failed; no latency plotted', transform=ax.transAxes,
                        size=9, color='#963724', wrap=True)
        axes[0][0].set_ylabel('Seconds · lower is better')
        fig.suptitle(title, x=.06, y=.99, ha='left', size=23, weight='bold')
        fig.text(.06, .905, subtitle, size=10)
        fig.text(.06, .035, note, size=9)
        fig.text(.06, .067, ('Points: one observed request per cell' if single else 'Solid: p50')
                 + ('   •   Dashed / shaded extent: p95' if minimum >= 100 else '   •   Tail curves withheld until ≥100 samples/cell'), size=9)
        fig.subplots_adjust(top=.80, bottom=.19, left=.065, right=.97, wspace=.12)
        for ext in ('png', 'svg'):
            path = out / f'{filename}.{ext}'
            fig.savefig(path, dpi=180)
            if ext == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    if minimum < 100:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharey=True)
    fig.patch.set_facecolor('#F8FAFC')
    for ax, target in zip(axes.flat, targets):
        ax.set_facecolor('#F8FAFC')
        for row in summary:
            if row['prefix_tokens'] != target:
                continue
            ax.plot(range(5), [row[f'ttft_s_p{p}'] for p in PERCENTILES],
                    color=COLORS[row['mode']], linestyle='-' if row['build'] == 'main' else '--',
                    marker='o' if row['build'] == 'main' else 's', markersize=3, lw=1.5)
        ax.set_title(f'{target//1024}K prefix', loc='left', weight='bold')
        ax.set_xticks(range(5), [f'p{p}' for p in PERCENTILES])
        ax.grid(axis='y', alpha=.18)
    # Set shared limits after every panel is populated. Setting a lower limit
    # on the first panel disables autoscaling and clips later, slower panels.
    upper = max(r[f'ttft_s_p{p}'] for r in summary for p in PERCENTILES
                if r[f'ttft_s_p{p}'] is not None)
    axes.flat[0].set_ylim(0, upper * 1.08)
    present = {r['mode'] for r in summary}
    handles = [Line2D([0], [0], color=COLORS[m], lw=3, label=LABELS[m]) for m in MODES if m in present]
    handles += [Line2D([0], [0], color='#253D50', label='main', marker='o'),
                Line2D([0], [0], color='#253D50', linestyle='--', label='#1489', marker='s')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, .065))
    fig.suptitle('Measured TTFT percentiles, from median to tail', x=.055, y=.985, ha='left', size=23, weight='bold')
    fig.text(.055, .93, subtitle + '  •  TTFT in seconds, lower is better', size=10)
    fig.text(.055, .025, note, size=9)
    fig.subplots_adjust(top=.87, bottom=.17, hspace=.36, wspace=.15)
    for ext in ('png', 'svg'):
        path = out / f'percentiles.{ext}'
        fig.savefig(path, dpi=180)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--subtitle', default='Serial local HTTP • fixed model and hardware • FP16 KV • 128-token replies')
    args = ap.parse_args()
    rows = []
    # Campaign manifests select completed runs so interrupted/retried jobs cannot double-count trials.
    manifest = args.input / 'campaign.json'
    if manifest.exists():
        jobs = json.loads(manifest.read_text())['results']
        files = [args.input / Path(job['output']).name / 'requests.jsonl' for job in jobs if job['exit_code'] == 0]
    else:
        files = sorted(args.input.rglob('requests.jsonl'))
    seen = set()
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['phase'] == 'measured':
                key = (row['build'], row['mode'], row['prefix_tokens'], row['trial'])
                if key in seen:
                    raise ValueError(f'Duplicate measured trial: {key}')
                seen.add(key)
            rows.append(row)
    summary = summarize(rows)
    if not summary:
        raise ValueError('No completed measurements')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    indexed = {(r['build'], r['mode'], r['prefix_tokens'], r['trial']): r
               for r in rows if r['phase'] == 'measured'}
    comparisons = []
    builds = sorted({r['build'] for r in summary})
    for build in builds:
        for mode in MODES[1:]:
            pairs = []
            for key, candidate in indexed.items():
                if key[:2] != (build, mode):
                    continue
                baseline = indexed.get((build, 'cold', key[2], key[3]))
                if baseline:
                    pairs.append((baseline, candidate))
            if pairs:
                comparisons.append(dict(build=build, comparison='cold vs '+mode, pairs=len(pairs),
                    identical_requests=sum(a['request_sha256'] == b['request_sha256'] for a, b in pairs),
                    identical_output_text=sum(a['output_sha256'] == b['output_sha256'] for a, b in pairs),
                    pairs_with_errors=sum(bool(a['error'] or b['error']) for a, b in pairs)))
    (args.output / 'comparisons.json').write_text(json.dumps(comparisons, indent=2))
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    charts(summary, args.output, args.subtitle)


if __name__ == '__main__':
    main()
