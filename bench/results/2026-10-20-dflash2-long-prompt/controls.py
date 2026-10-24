# Local native-protocol probe; run from the repository root. No downloads.
import json, os, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine, replace, validate_residency
out = Path(os.environ.get('STRATA_LONG_RESULTS', root.parent / 'Strata-data/dflash2-long-prompt'))
out.mkdir(parents=True, exist_ok=True)
src = root / 'bench/results/2026-10-19-dflash2-pr-performance'
prior = json.loads((out / 'confirm-dflash2.json').read_text())
c = json.loads((out / 'confirm-dflash2-command.json').read_text())
args = c['command'][1:-1]
slots = int(prior['info']['expert_slots'])
replace(args, '--expert-cache', str(slots))
replace(args, '--spec-min-p', '0')
replace(args, '--vram-reserve-mib', '480')
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env'])
(out / 'controls-command.json').write_text(json.dumps({'command': [str(root / 'build-dflash2/strata'), *args, '--serve'], 'env': c['env'], 'expected_slots': slots}, indent=2))
e = Engine(root / 'build-dflash2/strata', args, root, env, out / 'controls.log')
result = {'info': dict(e.info), 'rows': []}
try:
    assert e.info['spec_min_p'] == '0.00', e.info
    validate_residency(e.info, slots, prior['info'])
    prompts = json.loads((src / 'prompts.json').read_text())
    for task in ['code', 'long']:
        for mode, sampling in [('explicit-cli-zero', 'temperature=0 seed=1234'), ('target-only', 'temperature=1 top_k=1 top_p=1 min_p=0 seed=1234')]:
            print('START', task, mode, flush=True)
            r = e.request(prompts[task], sampling, 96)
            assert 'error' not in r, r
            r.update(task=task, mode=mode)
            r['matches_confirmed_rows'] = all((r['tokens'] == x['tokens'] for x in prior['rows'] if x['task'] == task and x['repeat'] >= 0))
            result['rows'].append(r)
            (out / 'controls.json').write_text(json.dumps(result, indent=2))
            print('DONE', task, mode, r['matches_confirmed_rows'], flush=True)
finally:
    e.close()
