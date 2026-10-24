"""Fill available VRAM: original/quantized DFlash2 versus MTP, local models only."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from dflash_sampling import Engine, replace, validate_residency

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm', choices=['ar', 'mtp', 'bf16', 'Q8_0', 'Q5_0', 'Q4_0'], required=True)
p.add_argument('--tag')
p.add_argument('--repeat', type=int, default=3)
p.add_argument('--count', type=int, default=96)
p.add_argument('--blocks', default='3', help='DFlash proposal counts, excluding the anchor')
p.add_argument('--expert-cache', default='auto')
p.add_argument('--reserve-mib', type=int, default=512)
p.add_argument('--spec-min-p', type=float, default=0.0)
p.add_argument('--tasks', default='code,math,prose,long')
p.add_argument('--probe-position', type=int, help='absolute target-logit fixture position; diagnostic runs only')
o = p.parse_args()
tag = o.tag or o.arm
blocks = [int(b) for b in o.blocks.split(',')] if o.arm not in ('ar', 'mtp') else [None]
assert all(b is None or 1 <= b <= 7 for b in blocks)
cfg = json.loads((OUT / 'target-config.json').read_text())
args = cfg['args']
for key in ('--mtp', '--dflash', '--dflash-block', '--dflash-window', '--dflash-head', '--dflash-vocab'):
    replace(args, key, None)
for key, value in {'--expert-cache': o.expert_cache, '--vram-reserve-mib': str(o.reserve_mib),
                   '--pcie-frac': '0', '--prefill': '256', '--adapt-every': '0',
                   '--adapt-swaps': '0', '--spec': '8', '--spec-min-p': str(o.spec_min_p),
                   '--suffix-draft': '0', '--lookup-chain': '0', '--prompt-cache': '0',
                   '--short-read': '4096', '--max-context': '8192', '--kv-resident': '8192'}.items():
    replace(args, key, value)
if o.arm == 'mtp':
    replace(args, '--mtp', str(ROOT.parent / 'Strata-data/mtp/rt'))
    replace(args, '--spec', '4')
elif o.arm != 'ar':
    replace(args, '--dflash', str(ROOT.parent / f'Strata-data/dflash2-official/dflash2-{o.arm}.gguf'))
    replace(args, '--dflash-block', str(blocks[0]))
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
env.update(STRATA_PREFILL_CPU_SHARE='0', STRATA_IQ_MT_MIN='1', STRATA_SPEC_PROB='0',
           STRATA_SPEC_COUPLED='0', STRATA_SPEC_GUMBEL='0', STRATA_DFLASH_HEAD_IL='0',
           STRATA_DFLASH_GRAPH_BUCKET='0', STRATA_DFLASH2_TILED_TOPK='1',
           STRATA_DFLASH2_SERIAL_SELECT='0', STRATA_DFLASH2_ANCHOR_LOGITS='0')
if o.probe_position is not None:
    logits = OUT / f'{tag}-logits.bin'
    logits.unlink(missing_ok=True)
    env.update(STRATA_DF_TARGET_PROBE=str(o.probe_position), STRATA_DF_TARGET_LOGITS=str(logits))
prompts = json.loads((OUT / 'prompts.json').read_text())
prompts = {k: prompts[k] for k in o.tasks.split(',')}
exe = ROOT / 'build-dflash2/strata'
command = dict(command=[str(exe), *args, '--serve'], sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               env={k: v for k, v in env.items() if k.startswith('STRATA_')},
               arm=o.arm, blocks=blocks, repeat=o.repeat, count=o.count,
               expert_cache=o.expert_cache, reserve_mib=o.reserve_mib, spec_min_p=o.spec_min_p,
               prompt_lengths={k: len(v) for k, v in prompts.items()})
(OUT / f'{tag}-command.json').write_text(json.dumps(command, indent=2))
e = Engine(exe, args, ROOT, env, OUT / f'{tag}.log')
result = dict(info=e.info, rows=[], warmups=[], vram=[], quality_controls=[])

def vram():
    result['vram'].append(dict(
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                                     '--format=csv,noheader,nounits'], text=True).strip(),
        processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                           '--format=csv,noheader,nounits'], text=True).strip(),
        engine_pid=e.p.pid))

def save():
    (OUT / f'{tag}.json').write_text(json.dumps(result, indent=2))

def request(task, block, repeat, force_target=False):
    # Positive temperature selects DFlash's target-only fallback. A single
    # surviving top-k candidate makes the sampling step deterministic.
    sampling = ('temperature=1 top_k=1 top_p=1 min_p=0 seed=1234'
                if force_target else 'temperature=0 seed=1234')
    if block is not None:
        sampling += f' dflash_block={block}'
    start = time.monotonic()
    r = e.request(prompts[task], sampling, o.count)
    assert 'error' not in r and len(r['tokens']) == o.count, r
    r.update(task=task, block=block, repeat=repeat, wall_ms=1000 * (time.monotonic() - start),
             tok_s=o.count * 1000 / r['decode_ms'])
    return r

try:
    assert int(e.info['expert_slots']) > 0, e.info
    if o.expert_cache != 'auto':
        validate_residency(e.info, int(o.expert_cache))
    vram()
    for block in blocks:
        for task in prompts:
            r = request(task, block, -1)
            result['warmups'].append(r)
            save()
            print(tag, 'warm', task, block, round(r['tok_s'], 3), flush=True)
    vram()
    for repeat in range(o.repeat):
        # Reverse block order on alternating repeats to limit order bias in a sweep.
        for block in (blocks if repeat % 2 == 0 else blocks[::-1]):
            for task in prompts:
                r = request(task, block, repeat)
                result['rows'].append(r)
                save()
                print(tag, repeat, task, block, round(r['tok_s'], 3), r['accepted'], r['offered'], flush=True)
    vram()
    save()
    if o.arm not in ('ar', 'mtp'):
        reference = ({r['task']: r['tokens'] for r in json.loads((OUT / 'ar.json').read_text())['rows']}
                     if (OUT / 'ar.json').exists() else {})
        for task in prompts:
            rows = [r for r in result['rows'] if r['task'] == task]
            if task not in reference or any(r['tokens'] != reference[task] for r in rows):
                control = request(task, None, -2, force_target=True)
                assert control['offered'] == control['accepted'] == 0, control
                control['mode'] = 'target-only, deterministic top_k=1'
                control['matches_speculative'] = all(r['tokens'] == control['tokens'] for r in rows)
                control['matches_auto_ar'] = control['tokens'] == reference[task] if task in reference else None
                result['quality_controls'].append(control)
                save()
                print(tag, 'QUALITY', task, 'same-placement target match', control['matches_speculative'],
                      'auto-AR match', control['matches_auto_ar'], flush=True)
finally:
    e.close()
