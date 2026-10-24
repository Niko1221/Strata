#!/usr/bin/env python3
"""Real-model regression: QFUSE single-token commits equal teacher-forced reads.

Uses existing local weights. Compares complete consumed target state, not just
the winning token. The pre-fix verifier skips GDN commits for T=1 with QFUSE.
"""
import argparse
import json
import os
from pathlib import Path
import re

from dflash_sampling import Engine, replace, validate_residency


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--expert-cache', type=int, default=2000)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    args = list(cfg['args'])
    for key in ('--dflash', '--dflash-block', '--dflash-window', '--dflash-head', '--dflash-vocab', '--mtp'):
        replace(args, key, None)
    for key, value in {'--expert-cache':str(a.expert_cache), '--adapt-every':'0', '--adapt-swaps':'0',
                       '--pcie-frac':'0', '--short-read':'4096', '--max-context':'4096', '--kv-resident':'4096',
                       '--spec':'2', '--spec-min-p':'0', '--suffix-draft':'0', '--lookup-chain':'0',
                       '--prompt-cache':'16', '--prompt-cache-every':'16', '--conversation-cache-mib':'0'}.items():
        replace(args, key, value)
    args.append('--spec-split')
    env = dict(os.environ, STRATA_QFUSE='1', STRATA_ONE_TOKEN_COMMIT='1', STRATA_STATE_HASH='1',
               STRATA_PREFILL_CPU_SHARE='0', STRATA_IQ_MT_MIN='1')
    ids = [50,3815,2923,539,2923,13,6558,281,2007,8,283,220,18,87,61,17,471,220,22,87,478,220,17,13]
    log = out / 'engine.txt'
    engine = Engine(a.engine.resolve(), args, Path(cfg.get('cwd', Path.cwd())), env, log)
    records = {}
    def request(label, prompt, count):
        r = engine.request(prompt, 'temperature=0 seed=7 pin=16', count)
        records[label] = r
        assert 'error' not in r and len(r['tokens']) == count, r
        return r['tokens']
    try:
        validate_residency(engine.info, a.expert_cache)
        first = request('decode', ids, 32)
        live = request('live', ids + first + [13], 1)
        request('reset', [50,3815,220,17,10,17,28], 1)
        forced = request('teacher_forced', ids + first[:-1], 1)
        continued = request('teacher_continuation', ids + first + [13], 1)
    finally:
        engine.close()
        (out / 'results.json').write_text(json.dumps(dict(info=engine.info, records=records), indent=2))
    states = [dict(re.findall(r'(\w+)=([0-9a-f,-]+)', line))
              for line in log.read_text().splitlines() if 'STATE_HASH L=' in line]
    assert len(states) == 5, 'missing consumed-state fingerprints'
    keys = ('L', 'gdn', 'ple', 'tail', 'pooled', 'kv', 'dead', 'pooled_full', 'ple_prev')
    assert all(all(k in state for k in keys) for state in states), 'incomplete state fingerprints'
    for x, y in ((0,3), (1,4)):
        differences = {k:[states[x].get(k),states[y].get(k)] for k in keys
                       if states[x].get(k) != states[y].get(k)}
        assert not differences, differences
    assert forced == first[-1:] and live == continued, records
    print('QFUSE single-token commits: full target-state and output parity PASS')


if __name__ == '__main__':
    main()
