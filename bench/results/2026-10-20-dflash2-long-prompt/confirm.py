# Local native-protocol probe; run from the repository root. No downloads.
import argparse, hashlib, json, os, sys, time
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine, replace
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['dflash2', 'mtp'], required=True)
o = p.parse_args()
out = Path(os.environ.get('STRATA_LONG_RESULTS', root.parent / 'Strata-data/dflash2-long-prompt'))
out.mkdir(parents=True, exist_ok=True)
src = root / 'bench/results/2026-10-19-dflash2-pr-performance'
c = json.loads((src / ('Q4_0-command.json' if o.mode == 'dflash2' else 'mtp-default-command.json')).read_text())
args = c['command'][1:-1]
if o.mode == 'dflash2':
    replace(args, '--spec-min-p', None)
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env'])
env['STRATA_DECODE_TIMING'] = '1'
exe = root / 'build-dflash2/strata'
tag = 'confirm-' + o.mode
command = {'command': [str(exe), *args, '--serve'], 'env': {k: v for k, v in env.items() if k.startswith('STRATA_')}, 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}
(out / f'{tag}-command.json').write_text(json.dumps(command, indent=2))
prompts = json.loads((src / 'prompts.json').read_text())
rows = []
warms = []
e = Engine(exe, args, root, env, out / f'{tag}.log')
info = dict(e.info)
try:
    if o.mode == 'dflash2':
        assert info['spec_min_p'] == '0.50', info

    def request(task, floor, rep, k=3):
        sample = 'temperature=0 seed=1234'
        if o.mode == 'dflash2':
            sample += f' dflash_block={k}'
        if floor is not None:
            sample += f' spec_min_p={floor}'
        print('START', o.mode, task, floor, rep, k, flush=True)
        t = time.monotonic()
        r = e.request(prompts[task], sample, 96)
        assert 'error' not in r and len(r['tokens']) == 96, r
        r.update(task=task, floor=floor, repeat=rep, block=k if o.mode == 'dflash2' else None, wall_ms=(time.monotonic() - t) * 1000, tok_s=96000 / r['decode_ms'])
        print('DONE', r['task'], floor, rep, k, r['tok_s'], r['accepted'], r['offered'], flush=True)
        return r

    def save():
        (out / f'{tag}.json').write_text(json.dumps({'info': info, 'warmups': warms, 'rows': rows}, indent=2))
    for task in ['code', 'long']:
        warms.append(request(task, None, -1))
        save()
    for rep in range(3):
        for floor in ([0, None] if rep % 2 == 0 else [None, 0]) if o.mode == 'dflash2' else [None]:
            for task in ['code', 'long']:
                rows.append(request(task, floor, rep))
                save()
    if o.mode == 'dflash2':
        for k, floor in [(3, 0.7), (2, 0.5), (7, 0.5)]:
            rows.append(request('long', floor, -2, k))
            save()
finally:
    e.close()
