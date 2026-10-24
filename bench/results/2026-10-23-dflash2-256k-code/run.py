"""Run from the repo root, using the existing local model artifacts (no downloads)."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from dflash_sampling import Engine, replace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['mtp-sweep', 'dflash2-sweep', 'dflash2-grid', 'mtp', 'dflash2', 'dflash2-il', 'dflash2-pcie', 'mtp-pcie'])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--min-free-mib', type=int, default=512)
p.add_argument('--target-only', action='store_true', help='mtp mode: remove drafter and use a one-row verifier')
p.add_argument('--spec-split', action='store_true', help='Use the existing opt-in two-group verifier')
p.add_argument('--reference', type=Path, help='JSON with target-only controls for MTP validation')
p.add_argument('--pcie-mode', choices=['auto','dma','kernel','direct'])
p.add_argument('--prompts', type=Path, help='JSON mapping of case names to token ID arrays')
p.add_argument('--engine', type=Path, default=root / 'build-dflash2/strata')
p.add_argument('--env', action='append', default=[], metavar='STRATA_NAME=VALUE')
p.add_argument('--count', type=int, default=96)
p.add_argument('--spec', type=int)
p.add_argument('--mtp-max-t', type=int)
p.add_argument('--context', type=int)
p.add_argument('--resident', type=int)
p.add_argument('--dflash-window', type=int)
p.add_argument('--block', type=int, default=7)
p.add_argument('--floor', type=float, default=0.8)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--expert-profile', type=Path)
p.add_argument('--expert-cache', type=int)
p.add_argument('--reserve', type=int)
p.add_argument('--q8-head-il', action='store_true')
p.add_argument('--pcie', type=float, default=0)
p.add_argument('--pool-tasks', type=int)
p.add_argument('--pool-workers', type=int)
p.add_argument('--adapt-every', type=int, default=0)
p.add_argument('--adapt-swaps', type=int, default=0)
p.add_argument('--tasks', default='code,long')
p.add_argument('--prefill', type=int, default=256)
p.add_argument('--short-read', type=int, default=4096)
p.add_argument('--only-selected', action='store_true')
p.add_argument('--prompt-cache', type=int, default=0)
p.add_argument('--pin-prompt', action='store_true')
p.add_argument('--grid-blocks', default='3,4,5,6,7')
p.add_argument('--grid-floors', default='.7,.75,.8,.85,.9')
p.add_argument('--draft-path', type=Path, help='Separate experimental drafter artifact; originals stay intact')
p.add_argument('--draft-quant', choices=['Q4_0','Q5_0','Q8_0','bf16'], default='Q4_0')
o = p.parse_args()
o.output.mkdir(parents=True, exist_ok=True)
source = root / 'bench/results/2026-10-19-dflash2-pr-performance'
mtp = o.mode.startswith('mtp')
c = json.loads((source / ('mtp-default-command.json' if mtp else 'Q4_0-command.json')).read_text())
prompt_path = o.prompts or source / 'prompts.json'
prompts = json.loads(prompt_path.read_text())
args = c['command'][1:-1]
if o.mtp_max_t is not None:
    replace(args, '--mtp-max-t', str(o.mtp_max_t))
if o.spec_split:
    args.append('--spec-split')
if o.target_only and not mtp:
    p.error('--target-only requires mtp mode')
if o.pcie_mode:
    replace(args, '--pcie-mode', o.pcie_mode)
for flag, value in [('--spec', o.spec), ('--max-context', o.context), ('--kv-resident', o.resident)]:
    if value is not None:
        replace(args, flag, str(value))
replace(args, '--prompt-cache', str(o.prompt_cache))
replace(args, '--prefill', str(o.prefill))
replace(args, '--short-read', str(o.short_read))
replace(args, '--pcie-frac', str(o.pcie))
replace(args, '--adapt-every', str(o.adapt_every))
replace(args, '--adapt-swaps', str(o.adapt_swaps))
if o.pool_tasks is not None:
    replace(args, '--pool-tasks', str(o.pool_tasks))
if o.pool_workers is not None:
    replace(args, '--pool-workers', str(o.pool_workers))
replace(args, '--spec-min-p', str(o.floor))
if o.expert_profile:
    replace(args, '--expert-profile', str(o.expert_profile.resolve()))
if o.expert_cache is not None:
    replace(args, '--expert-cache', str(o.expert_cache))
if o.reserve is not None:
    replace(args, '--vram-reserve-mib', str(o.reserve))
if not mtp:
    replace(args, '--dflash-block', str(o.block))
    if o.dflash_window is not None:
        replace(args, '--dflash-window', str(o.dflash_window))
    draft = Path(args[args.index('--dflash') + 1])
    replace(args, '--dflash', str(o.draft_path.resolve() if o.draft_path else draft.with_name(f'dflash2-{o.draft_quant}.gguf')))
if o.target_only:
    replace(args, '--mtp', None)
    # Native packs require startup capacity >=2; the server uses T=1 without a drafter.
    replace(args, '--spec', str(o.spec if o.spec is not None else 2))
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(c['env'])
env['STRATA_DECODE_TIMING'] = '1'
if mtp or o.mode in ('dflash2-il', 'dflash2-grid', 'dflash2-pcie') or o.q8_head_il:
    env['STRATA_DFLASH_HEAD_IL'] = '1'
    env['STRATA_MMVQ_IL_Q8_HEAD'] = '1'
for item in o.env:
    key, sep, value = item.partition('=')
    if not sep or not key.startswith('STRATA_'):
        p.error('--env requires STRATA_NAME=VALUE')
    env[key] = value
if o.count < 1:
    p.error('--count must be positive')
exe = o.engine.resolve()
tag = o.mode
if any((o.output / f'{tag}{suffix}').exists() for suffix in ['.json', '.log', '-command.json']):
    p.error(f'{tag} already exists; choose a fresh output directory')
result = dict(command=[str(exe), *args, '--serve'],
              env={k: v for k, v in env.items() if k.startswith('STRATA_')},
              sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
              prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
              expert_profile_sha256=hashlib.sha256(Path(args[args.index('--expert-profile')+1]).read_bytes()).hexdigest(),
              count=o.count, rows=[])
(o.output / f'{tag}-command.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))

def save():
    (o.output / f'{tag}.json').write_text(json.dumps(result, indent=2))

e = Engine(exe, args, root, env, o.output / f'{tag}.log')
result['info'] = dict(e.info)
try:
    startup_log = (o.output / f'{tag}.log').read_text()
    free_samples = re.findall(r'strata serve: (\d+) MiB of VRAM free with everything loaded', startup_log)
    assert free_samples, 'missing startup VRAM headroom measurement'
    result['startup_free_mib'] = int(free_samples[-1])
    save()
    assert result['startup_free_mib'] >= o.min_free_mib, f"insufficient headroom: {result['startup_free_mib']} MiB"
    if o.expert_cache is not None:
        assert int(e.info['expert_slots']) == o.expert_cache, e.info
    def run(task, floor, rep, control=False, block=None, pcie=None):
        control = control or o.target_only
        sampling = 'temperature=1 top_k=1 top_p=1 min_p=0 seed=1234' if control else f'temperature=0 seed=1234 spec_min_p={floor}'
        if control and mtp:
            sampling += ' spec_min_p=1'
        if not mtp:
            sampling += f' dflash_block={o.block if block is None else block}'
        if pcie is not None:
            sampling += f' pcie_frac={pcie}'
        if o.pin_prompt:
            sampling += f' pin={len(prompts[task])-2}'
        start = time.monotonic()
        r = e.request(prompts[task], sampling, o.count)
        assert 'error' not in r and len(r['tokens']) == o.count, r
        if control:
            assert r['offered'] == 0, 'control offered proposals; use a separate target-only process'
        r.update(task=task, floor=floor, rep=rep, control=control, sampling=sampling,
                 tok_s=1000*o.count/r['decode_ms'], wall_ms=1000*(time.monotonic()-start))
        result['rows'].append(r)
        save()
        print(tag, task, floor, rep, 'control' if control else '', r['tok_s'], r['accepted'], r['offered'], flush=True)

    if o.mode in ('dflash2-pcie','mtp-pcie'):
        run('long', o.floor, -1, pcie=0)
        for share in [0,.1,.2,.3,.4,.5,.75,1,0]:
            run('long', o.floor, -2, pcie=share)
            sample = result['rows'][-1]
            if not mtp:
                run('long', None, -3, control=True, pcie=share)
                sample['matches_control'] = sample['tokens'] == result['rows'][-1]['tokens']
                save()
    elif o.mode == 'dflash2-grid':
        run('long', o.floor, -1, block=o.block)
        for k in map(int, o.grid_blocks.split(',')):
            for f in map(float, o.grid_floors.split(',')):
                run('long', f, -2, block=k)
        run('long', o.floor, -1, block=o.block)
        run('long', None, -3, control=True)
        reference_tokens = result['rows'][-1]['tokens']
        for row in result['rows']:
            row['matches_control'] = row['tokens'] == reference_tokens
        save()
    elif o.mode == 'dflash2-sweep':
        for k, f in [(3,.5),(3,.7),(3,.8),(3,.9),(2,.7),(4,.7),(5,.7),(4,.8),(5,.8),(7,.8),(3,.6),(4,.6),(5,.6)]:
            run('long', f, -2, block=k)
    elif o.mode == 'mtp-sweep':
        for f in [0.5, 0.7, 0.8, 0.9]:
            run('long', f, -2)
    else:
        for task in o.tasks.split(','):
            run(task, o.floor, -1)
        for rep in range(o.repeats):
            for task in o.tasks.split(',') if rep % 2 == 0 else list(reversed(o.tasks.split(','))):
                for floor in ([0.5, o.floor] if rep % 2 == 0 else [o.floor, 0.5]) if o.floor != 0.5 and not o.only_selected else [o.floor]:
                    run(task, floor, rep, block=3 if floor == 0.5 and not mtp and o.floor != 0.5 and not o.only_selected else None)
        if not mtp:
            for task in o.tasks.split(','):
                run(task, None, -3, control=True)
                control = result['rows'][-1]['tokens']
                measured = [r for r in result['rows'] if r['task'] == task and r['rep'] >= 0]
                assert all(r['tokens'] == control for r in measured), f'{task}: target-only token mismatch'
            result['validation'] = 'same-process target-only controls passed'
        elif o.reference:
            reference = json.loads(o.reference.read_text())
            for key in ['expert_profile_sha256','prompt_sha256','count']:
                assert reference[key] == result[key], f'reference differs: {key}'
            for key in ['context','kv','kv_resident','expert_slots','expert_slots_primary','pcie_frac','pool_workers']:
                assert reference['info'][key] == result['info'][key], f'reference setting differs: {key}'
            for task in o.tasks.split(','):
                controls = [r for r in reference['rows'] if r['task'] == task and r['control'] and r['offered'] == 0]
                assert controls, f'{task}: missing genuine target-only control'
                expected = controls[-1]['tokens']
                assert all(r['tokens'] == expected for r in result['rows'] if r['task'] == task and r['rep'] >= 0), f'{task}: target-only mismatch'
            result['validation'] = 'external target-only controls passed'
        else:
            result['validation'] = 'target-only run' if o.target_only else 'not run; supply --reference'
        save()
    if o.reference:
        reference = json.loads(o.reference.read_text())
        for key in ['expert_profile_sha256','prompt_sha256','count']:
            assert reference[key] == result[key], f'common reference differs: {key}'
        for key in ['context','kv','kv_resident','expert_slots','expert_slots_primary','pcie_frac','pool_workers','spec']:
            assert reference['info'][key] == result['info'][key], f'common reference setting differs: {key}'
        for flag in ['--prefill','--short-read','--prompt-cache']:
            value = lambda command: command[command.index(flag)+1]
            assert value(reference['command']) == value(result['command']), f'prompt preparation differs: {flag}'
        assert ('--spec-split' in reference['command']) == ('--spec-split' in result['command']), 'prompt split differs'
        for task in o.tasks.split(','):
            controls = [r for r in reference['rows'] if r['task'] == task and r['control'] and r['offered'] == 0]
            assert controls and all(r['tokens'] == controls[0]['tokens'] for r in controls), 'unstable common control'
            assert all(r['tokens'] == controls[0]['tokens'] for r in result['rows'] if r['task'] == task), \
                f'{task}: cross-arm common target-only mismatch'
        result['validation'] = 'common target-only controls passed; matching prompt preparation and capacity'
        save()
finally:
    e.close()
