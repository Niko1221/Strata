"""Check target parity at each arm's actual expert placement, after throughput runs."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from dflash_sampling import Engine, replace, validate_residency
from dflash_logit_explain import records, explain

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--arms', default='bf16,Q4_0,Q8_0,Q5_0')
o = p.parse_args()
prompts = json.loads((OUT / 'prompts.json').read_text())
ar = {r['task']: r['tokens'] for r in json.loads((OUT / 'ar.json').read_text())['rows']}
report = {}

def first(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)

for arm in o.arms.split(','):
    data = json.loads((OUT / f'{arm}.json').read_text())
    command = json.loads((OUT / f'{arm}-command.json').read_text())
    exe = Path(command['command'][0])
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == command['sha256']
    for task, ids in prompts.items():
        key = f'{arm}-{task}'
        rows = [r for r in data['rows'] if r['task'] == task]
        expected = rows[0]['tokens']
        assert all(r['tokens'] == expected for r in rows), (key, 'nonrepeatable output')
        controls = [c for c in data.get('quality_controls', []) if c['task'] == task]
        if expected == ar[task]:
            report[key] = dict(status='PASS', kind='exact versus maximum-cache target-only', tokens=len(expected))
            continue
        if controls and controls[0]['tokens'] == expected:
            report[key] = dict(status='PASS', kind='exact versus same-placement deterministic target-only',
                               tokens=len(expected), auto_ar_first_difference=first(ar[task], expected))
            continue
        index = first(controls[0]['tokens'] if controls else ar[task], expected)
        for attempt in range(2):
            folder = OUT / 'quality' / key / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            logits = folder / 'emitted-logits.bin'
            logits.unlink(missing_ok=True)
            args = command['command'][1:-1].copy()
            slots = int(data['info']['expert_slots'])
            replace(args, '--expert-cache', str(slots))
            env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
            env.update(command['env'])
            env.update(STRATA_DF_TARGET_PROBE=str(len(ids) - 1 + index),
                       STRATA_DF_TARGET_LOGITS=str(logits))
            (folder / 'command.json').write_text(json.dumps(dict(command=[str(exe), *args, '--serve'],
                env={k: v for k, v in env.items() if k.startswith('STRATA_')}, sha256=command['sha256']), indent=2))
            print('START', key, 'index', index, 'slots', slots, flush=True)
            e = Engine(exe, args, ROOT, env, folder / 'engine.log')
            try:
                validate_residency(e.info, slots)
                greedy = e.request(ids, 'temperature=0 seed=1234 dflash_block=3', len(expected))
                target = e.request(ids, 'temperature=1 top_k=1 top_p=1 min_p=0 seed=1234', len(expected))
                assert 'error' not in greedy and 'error' not in target, (greedy, target)
                assert target['offered'] == target['accepted'] == 0, target
                assert greedy['tokens'] == expected, (key, 'failed to reproduce benchmark output')
                assert len(target['tokens']) == len(expected)
                result = dict(info=e.info, greedy=greedy, deterministic_target=target)
            finally:
                e.close()
            (folder / 'requests.json').write_text(json.dumps(result, indent=2))
            difference = first(greedy['tokens'], target['tokens'])
            if difference is None:
                report[key] = dict(status='PASS', kind='exact versus same-placement deterministic target-only',
                                   tokens=len(expected), fixture=str(folder.relative_to(OUT)))
                break
            if difference != index:
                index = difference
                continue
            samples = list(records(logits))
            assert len(samples) == 2, (key, len(samples))
            (hg, g), (ht, t) = samples
            assert hg['index'] == ht['index'] == index, (hg, ht)
            assert greedy['tokens'][:index] == target['tokens'][:index]
            assert hg['pick'] == greedy['tokens'][index] and ht['pick'] == target['tokens'][index]
            metrics = explain(t, g, ht['pick'], hg['pick'])
            metrics.update(candidate_header=hg, target_header=ht)
            (folder / 'near-tie.json').write_text(json.dumps(metrics, indent=2))
            assert metrics['documented_near_tie'], (key, metrics)
            report[key] = dict(status='PASS', kind='documented first-divergence top-two near-tie',
                               first_difference=index, target_gap=metrics['target_pair_gap'],
                               candidate_gap=metrics['candidate_pair_gap'],
                               fixture=str(folder.relative_to(OUT)))
            break
        else:
            raise AssertionError((key, 'could not locate stable first divergence'))
        (OUT / 'quality-summary.json').write_text(json.dumps(report, indent=2))
        print('PASS', key, report[key]['kind'], flush=True)
(OUT / 'quality-summary.json').write_text(json.dumps(report, indent=2))
print('PASS: all checked requests match a target control or have a measured top-two near-tie.', flush=True)
