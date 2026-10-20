# Run from the repository root with existing local model artifacts.
import json, os, sys, hashlib
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine
out = Path(os.environ.get('STRATA_DRAFT_RESULTS', root.parent / 'Strata-data/dflash2-draft-profile'))
out.mkdir(parents=True, exist_ok=True)
tag = sys.argv[1]
c = json.loads((root / ('bench/results/2026-10-19-dflash2-pr-performance/' + ('mtp-default-command.json' if tag == 'mtp' else 'Q4_0-command.json'))).read_text())
p = json.loads((root / 'bench/results/2026-10-19-dflash2-pr-performance/prompts.json').read_text())
exe = out / 'strata-before' if tag == 'before' else root / 'build-dflash2/strata'
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env'])
env['STRATA_DECODE_TIMING'] = '1'
sampling = 'temperature=0 seed=1234 spec_min_p=0.5' + ('' if tag == 'mtp' else ' dflash_block=3')
e = Engine(exe, c['command'][1:-1], root, env, out / f'{tag}.log')
rows = []
try:
    for task in ['code', 'long']:
        r = e.request(p[task], sampling, 96)
        rows.append(dict(task=task, warm=True, **r))
        print(tag, 'warm', task, r.get('decode_ms'), flush=True)
    for rep in range(3):
        for task in ['code', 'long'] if rep % 2 == 0 else ['long', 'code']:
            r = e.request(p[task], sampling, 96)
            rows.append(dict(task=task, rep=rep, **r))
            print(tag, rep, task, r.get('decode_ms'), flush=True)
            (out / f'{tag}.json').write_text(json.dumps(dict(info=e.info, command=[str(exe), *c['command'][1:]], env={k: v for k, v in env.items() if k.startswith('STRATA_')}, sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), rows=rows), indent=2))
    if tag == 'after':
        for task in ['code', 'long']:
            r = e.request(p[task], 'temperature=1 top_k=1 top_p=1 min_p=0 seed=1234', 96)
            rows.append(dict(task=task, control=True, **r))
            print(tag, 'target-control', task, r.get('decode_ms'), flush=True)
            (out / f'{tag}.json').write_text(json.dumps(dict(info=e.info, command=[str(exe), *c['command'][1:]], env={k: v for k, v in env.items() if k.startswith('STRATA_')}, sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), rows=rows), indent=2))
finally:
    e.close()
