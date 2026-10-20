# Run from the repository root with existing local model artifacts.
import json, os, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine
out = Path(os.environ.get('STRATA_DRAFT_RESULTS', root.parent / 'Strata-data/dflash2-draft-profile'))
out.mkdir(parents=True, exist_ok=True)
c = json.loads((root / 'bench/results/2026-10-19-dflash2-pr-performance/Q4_0-command.json').read_text())
p = json.loads((root / 'bench/results/2026-10-19-dflash2-pr-performance/prompts.json').read_text())
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env'])
env.update(STRATA_DF_EVENTS='1', STRATA_DECODE_TIMING='1')
tag = sys.argv[1] if len(sys.argv) > 1 else 'before'
exe = out / 'strata-before' if tag == 'before' else root / 'build-dflash2/strata'
e = Engine(exe, c['command'][1:-1], root, env, out / f'profile-{tag}.log')
try:
    r = e.request(p['long'], 'temperature=0 seed=1234 dflash_block=3 spec_min_p=0.5', 96)
    (out / f'profile-{tag}.json').write_text(json.dumps(dict(info=e.info, result=r), indent=2))
    print({k: r.get(k) for k in ['decode_ms', 'accepted', 'offered']}, flush=True)
finally:
    e.close()
