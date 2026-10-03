#!/usr/bin/env python3
"""Matched Q8/FP16/YaRN 1M ownership comparison, including failures and fallback.

A short allocation probe verifies that the requested ownership path activates.
The actual comparison uses separate fresh engines, fixed expert-cache capacity,
the earlier near-1M coding prompt and its 4096-token budget/natural-stop policy.
No profiler is enabled. File-tier experts are allowed and explicitly recorded.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import sys
import time

H = Path.home()
R = Path(__file__).resolve().parents[1]
ENGINE = H / 'src/strata-q8-exchange-rotation-1a50d913'
SOURCE = '1a50d913bf910a1f63fbc1a0788a7083e3ca5f8c'
ENGINE_SHA = 'd14ed6b69a1814ce4b5c08932a47d6921a55fa0aa8dea50427ccf0782d1ad997'
PY = H / 'src/Strata/.venv/bin/python'
D = H / 'fleet-downloads/rtxpro-q8-rotation-1m-20261003'
PRIOR = H / 'fleet-downloads/rtxpro-q8-model-dram-20261003-r2/status.json'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def option(args, key, value):
    if key in args:
        args[args.index(key) + 1] = str(value)
    else:
        args.extend([key, str(value)])


def worker(out):
    sys.path.insert(0, str(ENGINE / 'tools'))
    from configure_rtxpro import configure
    out.mkdir(parents=True, exist_ok=False)
    if hashlib.sha256((ENGINE / 'build/strata').read_bytes()).hexdigest() != ENGINE_SHA:
        raise RuntimeError('Frozen benchmark engine changed')
    state = {'started': time.time(), 'source': SOURCE, 'engine_sha256': ENGINE_SHA,
        'weights': 'Q8_0', 'kv': 'fp16', 'rope': 'YaRN factor 4, original context 262144',
        'allocated_context': 1048576, 'expert_cache_slots': 10874,
        'resident_expert_budget_gib': 56, 'ple': 'RAM', 'mode': 'MTP, suffix drafts off',
        'input_tokens': 1044472, 'output_budget': 4096, 'protocol_reserve': 8,
        'stop_policy': 'Natural EOS allowed, same prompt and maximum output for both arms.',
        'earlier_result': {'engine_source': 'e359f44851e86f25162bf7fe2f70c2e387e0672b',
                           'output_tps': 41.3645, 'same_binary_control': False},
        'records': [], 'comparisons': [],
        'scope': 'One unprofiled paired comparison; report either sign of gain, fallback, failure and output divergence.'}
    target = out / 'matrix.json'
    save(target, state)

    def run(label, rotate, input_tokens, output_tokens):
        state['current'] = label
        save(target, state)
        cfg = configure({'weights': 'Q8_0', 'kv': 'fp16', 'load_projection': False}, ENGINE, H, 1048576)
        cfg['env'].update(STRATA_PLE_PREFAULT_THREADS='8', STRATA_ADAPT_NOWAIT='0',
                          STRATA_EXCHANGE_ROTATE=str(rotate), STRATA_FLEET_PROFILE='0')
        option(cfg['args'], '--expert-cache', 10874)
        cfg['args'] += ['--rope-scaling', 'yarn', '--rope-scale', '4', '--yarn-orig-ctx', '262144']
        config = out / (label + '-config.json')
        save(config, cfg)
        command = [PY, ENGINE / 'tools/bench_mtp_modes.py', '--config', config, '--output', out / label,
            '--input-tokens', str(input_tokens), '--output-tokens', str(output_tokens), '--mode', 'on',
            '--workload', 'long', '--repetitions', '1', '--cases', 'coding', '--source-commit', SOURCE,
            '--verify-window', '8', '--mtp-window', '4', '--suffix-draft', '0']
        record = {'label': label, 'rotation_requested': bool(rotate), 'started': time.time(),
                  'input_tokens': input_tokens, 'output_budget': output_tokens}
        state['records'].append(record)
        save(target, state)
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, timeout=4800)
        record.update(exit=result.returncode, finished=time.time())
        engine_log_path = out / label / 'engine-mtp-on.log'
        engine_log = engine_log_path.read_text() if engine_log_path.exists() else ''
        record['rotation_active'] = 'exchange buffer rotation enabled' in engine_log
        record['placement'] = [line for line in engine_log.splitlines() if any(s in line for s in
            ('RAM budget', 'cache complement ready', 'resident RAM mode:', 'exchange buffer rotation',
             'exchange rotation unavailable', 'host memcpy bytes avoided', 'expert cache auto',
             'resident RAM:', 'PLE table locked', 'RoPE', 'rope', 'decode timing'))]
        result_path = out / label / 'result.json'
        if result_path.exists():
            data = json.loads(result_path.read_text())
            measured = data['runs'][0]
            case = measured['cases'][0]
            record.update(prompt_sha256=data['prompt_sha256'], engine_info=measured['engine_info'],
                          startup_seconds=measured['startup_seconds'], case=case)
            if measured['engine_info']['context'] != 1048576 or measured['engine_info']['kv'] != 'fp16':
                raise RuntimeError('Incorrect context or KV configuration')
            if case['input_tokens'] != input_tokens or not case['output_tokens']:
                raise RuntimeError('Missing expected input or generated output')
        save(target, state)
        return record

    try:
        probe = run('allocation-probe-rotate1', 1, 8192, 128)
        if probe['exit']:
            raise RuntimeError('Short 1M allocation probe failed; no full prompt attempted')
        state['probe_rotation_active'] = probe['rotation_active']
        save(target, state)
        # Even fallback is reported. Both full requests use identical placement.
        control = run('1m-copy', 0, 1044472, 4096)
        if control['exit']:
            raise RuntimeError('1M copy control failed; candidate not attempted')
        candidate = run('1m-rotate', 1, 1044472, 4096)
        if candidate['exit']:
            raise RuntimeError('1M ownership candidate failed')
        if control['prompt_sha256'] != candidate['prompt_sha256']:
            raise RuntimeError('Prompts differ')
        a, b = control['case'], candidate['case']
        aa, bb = a['token_ids'], b['token_ids']
        first = next((i for i, (x, y) in enumerate(zip(aa, bb)) if x != y),
                     min(len(aa), len(bb)) if len(aa) != len(bb) else None)
        state['comparisons'].append({'control': control['label'], 'candidate': candidate['label'],
            'rotation_active': candidate['rotation_active'], 'first_token_difference': first,
            'control_output_tokens': a['output_tokens'], 'candidate_output_tokens': b['output_tokens'],
            'control_tps': a['decode_tps'], 'rotation_tps': b['decode_tps'],
            'decode_gain_pct': 100 * (b['decode_tps'] / a['decode_tps'] - 1),
            'control_effective_tps': a['effective_output_tps'], 'rotation_effective_tps': b['effective_output_tps'],
            'effective_gain_pct': 100 * (b['effective_output_tps'] / a['effective_output_tps'] - 1),
            'control_prefill_seconds': a['timings']['prompt_ms'] / 1000,
            'rotation_prefill_seconds': b['timings']['prompt_ms'] / 1000,
            'matched_output_claim': first is None and candidate['rotation_active']})
        state['completed'] = True
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        state['finished'] = time.time()
        save(target, state)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
        return
    sys.path.insert(0, str(H / 'fleet-downloads'))
    import run_q4_trace_1c0c2bb as base
    base.wh.CUTOFF = time.time() + 24 * 3600
    runner = base.CgroupRun(D, (R / 'source-commit.txt').read_text().strip())
    runner.s.update(authorization='User explicitly requested ownership off/on at 1M and GitHub publication regardless of outcome',
                    shutdown_afterwards=False, current='waiting for queued Q8 counters',
                    dependency=str(PRIOR), operational_guard_not_user_deadline=True)
    runner.save()
    lock = (H / 'fleet-downloads/.rtxpro-bandwidth.lock').open('a')
    try:
        while not PRIOR.exists() or not json.loads(PRIOR.read_text()).get('finished'):
            runner.check_time()
            time.sleep(5)
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                runner.check_time()
                time.sleep(5)
        step = runner.gpu('rotation-1m', [PY, Path(__file__), '--worker', D / '{attempt}'], {}, timeout=14400)
        runner.s['matrix'] = str(D / step['label'] / 'matrix.json')
        runner.save()
    except BaseException as error:
        runner.finish(error)
        raise
    else:
        runner.finish()


if __name__ == '__main__':
    main()
