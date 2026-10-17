"""Opt-in reproduction of the report's frozen synthetic workload; never starts a model by default."""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path
from catalog import ROOT, digest, save, validate_config

PROFILES = ('grow-pipeline2', 'grow-pipeline2-dma1-bounce')

def fixtures():
    cases = json.loads(gzip.decompress((ROOT / 'synthetic-cases.json.gz').read_bytes()))
    assert sorted(c['expected_prompt_tokens'] for c in cases.values()) == [8192, 65536, 131072, 261000]
    for c in cases.values():
        assert digest(json.dumps(c['messages'], ensure_ascii=False, sort_keys=True).encode()) == c['messages_sha256']
        assert c['required_output_tokens'] == c['max_tokens'] == 1024
    return cases

def config(args, profile, folder):
    for value in (args.strata_src, args.engine, args.pack, args.native, args.ple_gguf, args.expert_profile, args.mtp):
        if not Path(value).exists(): raise FileNotFoundError(value)
    source = str(Path(args.strata_src).resolve())
    native_args = ['--pack', str(Path(args.pack).resolve()), '--native', str(Path(args.native).resolve()),
        '--ple-gguf', str(Path(args.ple_gguf).resolve()), '--expert-profile', str(Path(args.expert_profile).resolve()),
        '--expert-cache', 'auto', '--spec', '4', '--spec-min-p', '0.5', '--mtp', str(Path(args.mtp).resolve()),
        '--prefill', '4096', '--trim-stage-weights', '--max-context', '262144', '--kv', 'int8',
        '--kv-grow', '--pipeline-windows', '2']
    cfg = dict(exe=str(Path(args.engine).resolve()), cwd=source, server_root=source, args=native_args,
        tokenizer=str(Path(args.pack).resolve() / 'tokenizer'), model_name='strata-experiment',
        log=str(folder / (profile + '.log')), lib_dirs=[str(Path(p).resolve()) for p in args.lib_dir],
        port=18080, gpu=[0, 1], gpus_asked=True, layer_split='32', host='127.0.0.1',
        api_monitor=False, expected_engine_version=args.engine_version, parallel=1, local_no_api_key=False,
        env={'STRATA_DMA_BATCH': '0', 'STRATA_DMA_BOUNCE': '0', 'STRATA_KV_GROW_HOLD': '0'})
    if profile == PROFILES[1]: cfg['env'].update(STRATA_DMA_BATCH='1', STRATA_DMA_BOUNCE='1')
    validate_config(cfg)
    return cfg

def run(args):
    from harness import modules, preflight, visit
    cases = fixtures()
    psutil, _ = modules()
    hardware = preflight(psutil)  # refuses an existing Strata, busy GPU or insufficient RAM
    folder = ROOT / 'results' / time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    cfgs = {p: config(args, p, folder) for p in PROFILES}
    schedule = [dict(pair='report', profile=p, arm=p, round=n) for n in range(1, args.rounds + 1)
                for p in (PROFILES if n % 2 else PROFILES[::-1])]
    lock = ROOT / 'run.lock'
    with lock.open('x', encoding='utf-8') as f: f.write(str(__import__('os').getpid()))
    save(folder / 'plan.json', dict(schedule=schedule, hardware=hardware, output_tokens=1024,
        warmup='8192 actual input/128 actual output, thenfirst diagnostic+steady perlength',
        engine_sha256=digest(Path(args.engine).read_bytes()),
        scripts={p.name:digest(p.read_bytes()) for p in ROOT.glob('*.py')},
        compressed_fixture_sha256=digest((ROOT/'synthetic-cases.json.gz').read_bytes())))
    try:
        for item in schedule:
            if (ROOT/'STOP_REQUESTED').exists(): raise RuntimeError('Operator requested stop')
            preflight(psutil)
            print(item['profile'], 'round', item['round'], flush=True)
            result = visit(item, cfgs[item['profile']], cases, folder, 'cold', 'repeat10-steady', False)
            if not result['valid']: raise RuntimeError(str(result['errors']))
            assert len(result['groups']) == 6
            assert all(r['validation']['passed'] for g in result['groups'] for r in g['rows'])
        save(folder / 'COMPLETE.json', dict(visits=len(schedule), rounds=args.rounds))
    finally:
        lock.unlink(missing_ok=True)
    print('Completed:', folder)
    print('Raw local logs/configs contain machine paths: do not publish them without sanitizing.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--check-fixtures', action='store_true')
    p.add_argument('--rounds', type=int, choices=range(1,11), default=10)
    for name in ('strata-src','engine','pack','native','ple-gguf','expert-profile','mtp'):
        p.add_argument('--'+name)
    p.add_argument('--engine-version', default='0.1.40')
    p.add_argument('--lib-dir', action='append', default=[])
    args = p.parse_args()
    if args.run:
        if not all(getattr(args,n) for n in ('strata_src','engine','pack','native','ple_gguf','expert_profile','mtp')):
            p.error('All source/model/engine paths are required for --run')
        run(args)
    elif args.check_fixtures:
        print('Synthetic fixture message hashes and token targets verified:', list(fixtures()))
    else: print('No model or GPU work without --run; --check-fixtures is CPU-only.')
