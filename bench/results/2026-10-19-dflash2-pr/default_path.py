"""Compare unchanged AR/MTP paths with main, extension OFF and extension ON."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from dflash_sampling import Engine, replace, validate_residency

p = argparse.ArgumentParser(description=__doc__)
for name in ('baseline', 'off', 'on', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
fixtures = ROOT / 'bench/results/2026-10-19-dflash2-quant-mtp'
base = json.loads((fixtures / 'target-config.json').read_text())['args']
prompts = json.loads((fixtures / 'prompts.json').read_text())
prompts = {name: prompts[name] for name in ('code', 'math', 'long')}
env = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
env.update(STRATA_PREFILL_CPU_SHARE='0', STRATA_SPEC_COUPLED='0',
           STRATA_SPEC_GUMBEL='0', STRATA_IQ_MT_MIN='1')
results = {}
for build in ('baseline', 'off', 'on'):
    exe = getattr(a, build).resolve()
    for mode in ('ar', 'mtp'):
        tag = build + '-' + mode
        args = list(base)
        for key in ('--mtp', '--dflash', '--dflash-block', '--dflash-window', '--dflash-head', '--dflash-vocab'):
            replace(args, key, None)
        for key, value in {'--expert-cache':'2000', '--max-context':'4096', '--kv-resident':'4096',
                           '--prefill':'256', '--vram-reserve-mib':'700', '--spec': '4' if mode=='mtp' else '2',
                           '--spec-min-p':'0.5', '--adapt-every':'0', '--adapt-swaps':'0', '--pcie-frac':'0',
                           '--suffix-draft':'0', '--lookup-chain':'0', '--prompt-cache':'0',
                           '--conversation-cache-mib':'0'}.items():
            replace(args, key, value)
        if mode == 'mtp': replace(args, '--mtp', str(ROOT.parent/'Strata-data/mtp/rt'))
        row = dict(command=[str(exe),*args,'--serve'], executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                   env={k:v for k,v in env.items() if k.startswith('STRATA_')}, requests={})
        results[tag] = row
        print('START',tag,flush=True)
        e = Engine(exe,args,ROOT,env,a.output/f'{tag}.log')
        try:
            row['info'] = e.info
            validate_residency(e.info,2000,results.get('baseline-'+mode,{}).get('info'))
            for name,ids in prompts.items():
                for sampling in ('temperature=0 seed=7','temperature=0.8 top_k=40 top_p=0.95 min_p=0.05 seed=7'):
                    key = name + ':' + sampling
                    r = e.request(ids,sampling,32)
                    row['requests'][key] = r
                    assert 'error' not in r and 0 < len(r['tokens']) <= 32,r
                    if build != 'baseline':
                        assert r['tokens'] == results['baseline-'+mode]['requests'][key]['tokens'], (tag,key)
            print('PASS',tag,sum(len(r['tokens']) for r in row['requests'].values()),'token IDs',flush=True)
        finally:
            e.close()
            (a.output/'default-path.json').write_text(json.dumps(results,indent=2))
