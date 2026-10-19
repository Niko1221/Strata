"""Local DFlash2 VRAM audit; sequential engines, no downloads or config edits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from dflash_sampling import Engine, replace
base = json.loads((ROOT / 'bench/results/2026-10-19-dflash2-quant-mtp/Q8_0-command.json').read_text())
exe = ROOT / 'build-dflash2/strata'
results = []

def memory():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip()

for quant, elastic in [('bf16', False), ('Q8_0', False), ('Q4_0', False), ('Q8_0', True)]:
    tag = quant + ('-elastic' if elastic else '-eager')
    args = list(base['command'][1:])
    args.remove('--serve')
    for flag, value in {'--max-context': '131072', '--kv-resident': '32768', '--dflash-window': '0',
                        '--vram-reserve-mib': '700', '--expert-cache': 'auto',
                        '--dflash': str(ROOT.parent / f'Strata-data/dflash2-official/dflash2-{quant}.gguf')}.items():
        replace(args, flag, value)
    env = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
    env.update(base['env'])
    env['STRATA_DFLASH_KV_GROW'] = '1' if elastic else '0'
    row = dict(tag=tag, command=[str(exe), *args, '--serve'],
               executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               env={k:v for k,v in env.items() if k.startswith('STRATA_')}, before=memory())
    results.append(row)
    print('START', tag, flush=True)
    engine = None
    start = time.monotonic()
    try:
        engine = Engine(exe, args, ROOT, env, OUT / f'{tag}.log')
        row.update(info=engine.info, startup_seconds=time.monotonic()-start, ready=memory())
        print('READY', tag, row['ready'], row['info'], flush=True)
        prompt = [50, 3815, 2923, 539, 2923, 13, 6558, 281, 2007, 8, 283, 220, 18, 87, 61, 17, 471, 220, 22, 87, 478, 220, 17, 13]
        if elastic:
            prompt = (prompt * 352)[:8448]
        row['prompt_tokens'] = len(prompt)
        row['request'] = engine.request(prompt, 'temperature=0 seed=1234', 8)
        row['after_request'] = memory()
        print('REQUEST', tag, row['request'], row['after_request'], flush=True)
    except Exception as exc:
        row['error'] = repr(exc)
        print('ERROR', tag, repr(exc), flush=True)
    finally:
        if engine is not None:
            engine.close()
        row['after_close'] = memory()
        (OUT / 'results.json').write_text(json.dumps(results, indent=2))
