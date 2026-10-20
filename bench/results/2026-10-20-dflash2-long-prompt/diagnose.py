# Local native-protocol probe; run from the repository root. No downloads.
import json, os, sys, time
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine
out = Path(os.environ.get('STRATA_LONG_RESULTS', root.parent / 'Strata-data/dflash2-long-prompt'))
out.mkdir(parents=True, exist_ok=True)
p = root / 'bench/results/2026-10-19-dflash2-pr-performance'
c = json.loads((p / 'Q4_0-command.json').read_text())
prompts = json.loads((p / 'prompts.json').read_text())
env = os.environ.copy()
env.update(c['env'])
env['STRATA_DECODE_TIMING'] = '1'
e = Engine(out / 'strata-before', c['command'][1:-1], root, env, out / 'diagnose.log')
rows = []
try:
    for task, k, floor in [('long', 3, 0), ('long', 3, 0.5), ('long', 2, 0), ('code', 3, 0)]:
        print('START', task, k, floor, flush=True)
        r = e.request(prompts[task], f'temperature=0 seed=1234 dflash_block={k} spec_min_p={floor}', 96)
        r.update(task=task, block=k, floor=floor)
        rows.append(r)
        (out / 'diagnose.json').write_text(json.dumps(dict(info=e.info, rows=rows), indent=2))
        print('DONE', task, k, floor, {x: r.get(x) for x in ['decode_ms', 'accepted', 'offered', 'error']}, flush=True)
finally:
    e.close()
