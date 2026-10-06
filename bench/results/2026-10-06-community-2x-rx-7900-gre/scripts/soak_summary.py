#!/usr/bin/env python3
"""Rebuild the soak's summary and its engine-log slice after soak.py's teardown.

soak.py writes data/soak-mem.csv (the samples), data/soak.json (one entry per
prompt, updated as it goes) and data/soak-engine.log (its slice of the engine
log) only at the very end, after it has stopped the sampler. sample_mem.py
notices its stop file at the top of its loop, i.e. up to one --interval later,
so with --interval 60 soak.py's 30 s wait raises TimeoutExpired and the process
exits without writing the summary or the slice. The samples and the per-prompt
records are complete; this script turns them into the summary soak.py would have
written, and cuts the engine-log slice out of the session log.

    soak_summary.py [--log PATH] [--data DIR] [--dry-run]

Writes data/soak-summary.json and data/soak-engine.log.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parents[1]
DATA = REPORT / 'data'
ENGINE_LOG = ROOT / 'strata-coder-iq1_m.log'
STARTED = re.compile(r'^\[strata\] \S+ \S+ engine started: .*$', re.M)
FAULTS = re.compile(r'no progress|\bnan\b|crash|never rang|doorbell|abort|'
                    r'segmentation fault|\bkilled\b|out of memory', re.I)


def first_peak(values, scale, unit):
    if not values:
        return None
    return {'first': round(values[0] / scale, 3), 'peak': round(max(values) / scale, 3),
            'unit': unit}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, default=ENGINE_LOG, help='the session engine log')
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    d = args.data

    # --- the engine log slice: the last start-up that used the soak's config ---
    text = args.log.read_text(encoding='utf-8', errors='replace')
    starts = [m.start() for m in STARTED.finditer(text)]
    soak_at = None
    for i, at in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[at:end]
        if '--resident-budget-gib' in block:
            soak_at = at                    # the last one is the soak's own server
    segment = ''
    if soak_at is not None:
        i = starts.index(soak_at)
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        segment = text[soak_at:end]

    # --- the samples ---
    with (d / 'soak-mem.csv').open() as fh:
        rows = list(csv.DictReader(fh))
    vram_cols = [c for c in (rows[0].keys() if rows else []) if c.endswith('_vram_used')]

    def col(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, '', '-1')]

    # --- the prompts soak.py recorded as it ran ---
    interim = json.loads((d / 'soak.json').read_text())
    requests = interim.get('requests', [])
    faults = [ln for ln in segment.splitlines() if FAULTS.search(ln)]
    summary = {
        'built_by': 'scripts/soak_summary.py (soak.py\'s teardown raised '
                    'TimeoutExpired waiting for sample_mem.py to notice its stop file)',
        'started': interim.get('started'), 'minutes': interim.get('minutes'),
        'interval_s': interim.get('interval_s'), 'config': interim.get('config'),
        'samples': len(rows),
        'sample_span_s': round(float(rows[-1]['unix_s']) - float(rows[0]['unix_s']), 1) if rows else None,
        'prompts': len(requests),
        'prompt_errors': [r for r in requests if 'error' in r],
        'finish_reasons': {f: sum(1 for r in requests if r.get('finish') == f)
                           for f in sorted({r.get('finish') for r in requests if r.get('finish')})},
        'wall_s_min_median_max': ([round(min(w), 2), round(sorted(w)[len(w) // 2], 2),
                                   round(max(w), 2)]
                                  if (w := [r['wall_s'] for r in requests if 'wall_s' in r])
                                  else None),
        'survived': not any('error' in r for r in requests) and not faults,
        'engine_rss_mib': first_peak(col('engine_rss_mib'), 1, 'MiB'),
        'ram_used_mib': first_peak(col('ram_used_mib'), 1, 'MiB'),
        'vram_used': {c.replace('_vram_used', ''): first_peak(col(c), 2**30, 'GiB')
                      for c in vram_cols},
        'fault_lines': faults,
    }
    if not args.dry_run:
        (d / 'soak-engine.log').write_text(segment)
        (d / 'soak-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'fault_lines'}, indent=2))
    if faults:
        print('fault lines:', *faults, sep='\n  ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
