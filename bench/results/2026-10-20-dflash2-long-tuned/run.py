"""Run from the repo root, using the existing local model artifacts (no downloads)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine, replace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['mtp-sweep', 'dflash2-sweep', 'dflash2-grid', 'mtp', 'dflash2', 'dflash2-il'])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--block', type=int, default=7)
p.add_argument('--floor', type=float, default=0.8)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--expert-cache', type=int)
p.add_argument('--reserve', type=int)
p.add_argument('--q8-head-il', action='store_true')
p.add_argument('--only-selected', action='store_true')
p.add_argument('--prompt-cache', type=int, default=0)
p.add_argument('--pin-prompt', action='store_true')
p.add_argument('--grid-blocks', default='3,4,5,6,7')
p.add_argument('--grid-floors', default='.7,.75,.8,.85,.9')
p.add_argument('--draft-quant', choices=['Q4_0','Q5_0','Q8_0','bf16'], default='Q4_0')
o = p.parse_args()
o.output.mkdir(parents=True, exist_ok=True)
source = root / 'bench/results/2026-10-19-dflash2-pr-performance'
mtp = o.mode.startswith('mtp')
c = json.loads((source / ('mtp-default-command.json' if mtp else 'Q4_0-command.json')).read_text())
prompts = json.loads((source / 'prompts.json').read_text())
args = c['command'][1:-1]
replace(args, '--prompt-cache', str(o.prompt_cache))
replace(args, '--spec-min-p', str(o.floor))
if o.expert_cache is not None:
    replace(args, '--expert-cache', str(o.expert_cache))
if o.reserve is not None:
    replace(args, '--vram-reserve-mib', str(o.reserve))
if not mtp:
    replace(args, '--dflash-block', str(o.block))
    draft = Path(args[args.index('--dflash') + 1])
    replace(args, '--dflash', str(draft.with_name(f'dflash2-{o.draft_quant}.gguf')))
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env'])
env['STRATA_DECODE_TIMING'] = '1'
if o.mode == 'dflash2-il' or o.q8_head_il:
    env['STRATA_DFLASH_HEAD_IL'] = '1'
    env['STRATA_MMVQ_IL_Q8_HEAD'] = '1'
exe = root / 'build-dflash2/strata'
tag = o.mode
if any((o.output / f'{tag}{suffix}').exists() for suffix in ['.json', '.log', '-command.json']):
    p.error(f'{tag} already exists; choose a fresh output directory')
result = dict(command=[str(exe), *args, '--serve'],
              env={k: v for k, v in env.items() if k.startswith('STRATA_')},
              sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), rows=[])
(o.output / f'{tag}-command.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))

def save():
    (o.output / f'{tag}.json').write_text(json.dumps(result, indent=2))

e = Engine(exe, args, root, env, o.output / f'{tag}.log')
result['info'] = dict(e.info)
try:
    if o.expert_cache is not None:
        assert int(e.info['expert_slots']) == o.expert_cache, e.info
    def run(task, floor, rep, control=False, block=None):
        sampling = 'temperature=1 top_k=1 top_p=1 min_p=0 seed=1234' if control else f'temperature=0 seed=1234 spec_min_p={floor}'
        if not mtp:
            sampling += f' dflash_block={o.block if block is None else block}'
        if o.pin_prompt:
            sampling += f' pin={len(prompts[task])-2}'
        start = time.monotonic()
        r = e.request(prompts[task], sampling, 96)
        assert 'error' not in r and len(r['tokens']) == 96, r
        r.update(task=task, floor=floor, rep=rep, control=control, sampling=sampling,
                 tok_s=96000/r['decode_ms'], wall_ms=1000*(time.monotonic()-start))
        result['rows'].append(r)
        save()
        print(tag, task, floor, rep, 'control' if control else '', r['tok_s'], r['accepted'], r['offered'], flush=True)

    if o.mode == 'dflash2-grid':
        run('long', o.floor, -1, block=o.block)
        for k in map(int, o.grid_blocks.split(',')):
            for f in map(float, o.grid_floors.split(',')):
                run('long', f, -2, block=k)
        run('long', o.floor, -1, block=o.block)
    elif o.mode == 'dflash2-sweep':
        for k, f in [(3,.5),(3,.7),(3,.8),(3,.9),(2,.7),(4,.7),(5,.7),(4,.8),(5,.8),(7,.8),(3,.6),(4,.6),(5,.6)]:
            run('long', f, -2, block=k)
    elif o.mode == 'mtp-sweep':
        for f in [0.5, 0.7, 0.8, 0.9]:
            run('long', f, -2)
    else:
        for task in ['code', 'long']:
            run(task, o.floor, -1)
        for rep in range(o.repeats):
            for task in ['code', 'long'] if rep % 2 == 0 else ['long', 'code']:
                for floor in ([0.5, o.floor] if rep % 2 == 0 else [o.floor, 0.5]) if o.floor != 0.5 and not o.only_selected else [o.floor]:
                    run(task, floor, rep, block=3 if floor == 0.5 and not mtp and o.floor != 0.5 and not o.only_selected else None)
        if not mtp:
            for task in ['code', 'long']:
                run(task, None, -3, control=True)
finally:
    e.close()
