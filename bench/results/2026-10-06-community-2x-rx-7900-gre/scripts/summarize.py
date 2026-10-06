#!/usr/bin/env python3
"""Print every number the README needs, with median and range, from data/.

Reads what the other scripts wrote and prints markdown rows:
  * data/interleave500.json  - the 500-token interleaved pipeline A/B
  * data/ab-<label>-A.json / -B.json - the bench_prefill arms
  * data/soak.json and data/soak-mem.csv - the 30-minute soak
Nothing is invented: a row is printed only if its file is there.

Usage: summarize.py [--data DIR]
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'


def mr(values):
    """median (min..max), or the reason there is none"""
    v = [x for x in values if x is not None]
    if not v:
        return 'not measured'
    if len(v) == 1:
        return f'{v[0]:.1f}'
    return f'{statistics.median(v):.1f} ({min(v):.1f}..{max(v):.1f})'


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def interleave(data):
    rs = load(data / 'interleave500.json')
    if rs is None:
        print('\n## interleave500: MISSING\n')
        return
    print('\n## interleave500 (500 greedy tokens per request, one restart per '
          'request)\n')
    print('| arm | runs | prompt tok/s median (range) | decode tok/s median '
          '(range) | TTFT s median (range) | wall s median (range) | '
          'finish |')
    print('| --- | ---: | --- | --- | --- | --- | --- |')
    arms = {}
    for arm in ('A', 'B'):
        sel = [r for r in rs if r['arm'] == arm and r.get('metrics')]
        arms[arm] = sel
        m = [r['metrics'] for r in sel]
        print(f'| {arm} | {len(sel)} | {mr([x["prefill_tps"] for x in m])} | '
              f'{mr([x["decode_tps"] for x in m])} | '
              f'{mr([r.get("ttft_s") for r in sel])} | '
              f'{mr([r.get("wall_s") for r in sel])} | '
              f'{sorted({r.get("finish_reason") for r in sel})} |')
    pairs = sorted({r['pair'] for r in rs})
    deltas = []
    for p in pairs:
        a = next((r for r in rs if r['pair'] == p and r['arm'] == 'A'
                  and r.get('metrics')), None)
        b = next((r for r in rs if r['pair'] == p and r['arm'] == 'B'
                  and r.get('metrics')), None)
        if a and b:
            da, db = a['metrics']['decode_tps'], b['metrics']['decode_tps']
            deltas.append(100 * (db - da) / da)
            print(f'- pair {p}: A {da:.1f} -> B {db:.1f} tok/s '
                  f'({100 * (db - da) / da:+.1f}%), prompt '
                  f'{a["metrics"]["prefill_tps"]:.0f} -> '
                  f'{b["metrics"]["prefill_tps"]:.0f} tok/s')
    if deltas:
        print(f'- B vs A decode change: median {statistics.median(deltas):+.1f}% '
              f'({min(deltas):+.1f}%..{max(deltas):+.1f}%) over {len(deltas)} pairs')
    errors = [r for r in rs if r.get('error')]
    if errors:
        print(f'- FAILED requests: {len(errors)}: '
              + '; '.join(f'pair {r["pair"]}{r["arm"]}: {r["error"]}'
                          for r in errors))
    fields = [('prompt_tokens', 'prompt tokens'), ('reused', 'reused'),
              ('fresh', 'fresh'), ('generated', 'generated')]
    for key, name in fields:
        vals = [r['metrics'][key] for r in rs if r.get('metrics')]
        if vals:
            print(f'- {name}: {min(vals):.0f}..{max(vals):.0f} '
                  f'(median {statistics.median(vals):.0f})')


def arms(data):
    print('\n## short arms (bench_prefill.py: 128 tokens, warmup + '
          '4 fresh + 4 follow-up per arm)\n')
    labels = sorted({p.name[3:-7] for p in data.glob('ab-*-A.json')})
    print('| arm | side | kind | runs | prompt tok/s | decode tok/s | '
          'prompt read ms | decode ms |')
    print('| --- | --- | --- | ---: | --- | --- | --- | --- |')
    for lab in labels:
        for side in ('A', 'B'):
            rs = load(data / f'ab-{lab}-{side}.json')
            if not rs:
                print(f'| {lab} | {side} | - | 0 | not measured | | | |')
                continue
            for kind in ('fresh', 'followup'):
                sel = [r for r in rs if r['kind'] == kind]
                m = [r['metrics'] for r in sel]
                if not m:
                    continue
                print(f'| {lab} | {side} | {kind} | {len(m)} | '
                      f'{mr([x["prefill_tps"] for x in m])} | '
                      f'{mr([x["decode_tps"] for x in m])} | '
                      f'{mr([x["prefill_ms"] for x in m])} | '
                      f'{mr([x["decode_ms"] for x in m])} |')


def soak(data):
    # soak-summary.json is what soak_summary.py rebuilt from the samples and the
    # per-prompt records after soak.py's teardown raised; soak.json is the raw
    # per-prompt record soak.py wrote before that.
    s = load(data / 'soak-summary.json') or load(data / 'soak.json')
    if s is None:
        print('\n## soak: MISSING\n')
        return
    print('\n## soak\n')
    for k in ('built_by', 'minutes', 'interval_s', 'samples', 'sample_span_s',
              'prompts', 'survived', 'finish_reasons', 'wall_s_min_median_max',
              'engine_rss_mib', 'ram_used_mib'):
        if k in s and s[k] is not None:
            v = s[k]
            print(f'- {k}: {json.dumps(v) if isinstance(v, dict) else v}')
    if s.get('vram_used') is not None:
        print(f'- vram_used: {json.dumps(s["vram_used"])}')
    pe = s.get('prompt_errors')
    print(f'- prompt_errors: {len(pe) if isinstance(pe, list) else pe}')
    fl = s.get('fault_lines') or []
    print(f'- fault_lines: {len(fl)}')
    for ln in fl[:20]:
        print(f'    {ln}')
    csv_path = data / 'soak-mem.csv'
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
        if rows:
            print(f'- first sample: {rows[0]}')
            print(f'- last sample : {rows[-1]}')


def batch_concurrent(data):
    print('\n## batch concurrency\n')
    for n in (2, 4):
        c = load(data / f'batch-concurrent-batch{n}.json')
        if c is None:
            print(f'- batch{n}: MISSING')
            continue
        print(f'- batch{n}: slots={c["batch"]} elapsed {c["elapsed_s"]} s, '
              f'hung_slots={c["hung_slots"]}, aggregate_decode_tps='
              f'{c["aggregate_decode_tps"]}')
        for r in c['requests']:
            print(f'    slot {r["slot"]}: ttft {r["ttft_s"]} s, wall '
                  f'{r["wall_s"]} s, finish {r["finish"]}, error {r["error"]}')
        for d in c['server_done']:
            print(f'    done: {d["tokens"]} tokens in {d["wall_s"]} s '
                  f'({d["decode_tps"]} tok/s) finish {d["finish"]}, '
                  f'cancelled {d["cancelled"]}')
        print(f'    engine lines kept: {c["engine_log_lines_found"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA)
    args = p.parse_args()
    interleave(args.data)
    arms(args.data)
    soak(args.data)
    batch_concurrent(args.data)


if __name__ == '__main__':
    main()
