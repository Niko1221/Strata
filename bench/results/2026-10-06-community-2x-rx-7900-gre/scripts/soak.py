#!/usr/bin/env python3
"""30-minute soak: keep one server up, loop fresh prompts, sample RAM/VRAM.

Starts the server from the given config, then for `--minutes` sends one greedy
prompt after another (a fresh conversation every time) while sample_mem.py
writes a CSV every `--interval` seconds: whole-machine RAM, each card's VRAM
from amdgpu's sysfs counters, and the engine process's RSS.

Writes a JSON summary next to the CSV: whether the soak survived, RSS and VRAM
at the first sample and at the peak, the number of prompts and errors, and any
line in the engine log's soak segment that looks like a fault.

Usage:
  soak.py --config CFG --model NAME --engine-log LOG --out-dir DIR \
          --minutes 30 --interval 60
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PY = ROOT / '.venv/bin/python'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def health(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=5) as r:
            return json.load(r).get('loaded')
    except (urllib.error.URLError, OSError):
        return False


def prompt_for(i):
    topics = [
        'a Python function that returns the n-th Fibonacci number',
        'the difference between a list and a tuple in Python',
        'how a hash map handles collisions, with a small example',
        'a short explanation of how SQL indexes speed up a query',
        'a Python context manager that times a block of code',
        'what backpressure means in a streaming pipeline',
        'a compact explanation of big-O for binary search',
        'how to make a function idempotent, with an example',
        'the difference between concurrency and parallelism',
        'a Python snippet that merges two sorted lists',
        'why floating point addition is not associative',
        'a short description of how DNS resolution works',
        'how to version a database schema safely',
        'a Python decorator that retries a flaky call',
        'what a monotonic clock is for, and why wall clocks jump',
    ]
    t = topics[i % len(topics)]
    return (f'QUESTION {i + 1}: Explain {t}. Give the reasoning step by step, '
            f'include runnable code where it helps, and end with two pitfalls '
            f'that catch people out. Number your points and be thorough; this '
            f'is question {i + 1} of a long series, so do not abbreviate.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--engine-log', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--minutes', type=float, default=30)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--max-tokens', type=int, default=128)
    args = p.parse_args()

    url = args.url.rstrip('/')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    engine_log = Path(args.engine_log)
    csv_path = out / 'soak-mem.csv'
    json_path = out / 'soak.json'
    server_out = out / 'soak-server.out'
    stop_file = out / 'soak-sampler.stop'
    if stop_file.exists():
        stop_file.unlink()

    log_offset = engine_log.stat().st_size if engine_log.exists() else 0
    started = time.time()

    server = subprocess.Popen(
        [str(PY), str(ROOT / 'serve/server.py'), '--engine', 'strata',
         '--config', args.config, '--port', '8080'],
        cwd=ROOT, stdout=server_out.open('ab'), stderr=subprocess.STDOUT,
        start_new_session=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f'server exited with {server.returncode} before '
                             f'ready; see {server_out}')
        if health(url):
            break
        time.sleep(2)
    else:
        raise SystemExit(f'server never reported loaded; see {server_out}')
    log('server ready; starting the sampler and the prompt loop')

    sampler = subprocess.Popen(
        [str(PY), str(Path(__file__).resolve().parent / 'sample_mem.py'),
         '--out', str(csv_path), '--seconds', str(args.minutes * 60 + 120),
         '--interval', str(args.interval), '--engine-rss',
         '--stop-file', str(stop_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    requests = []
    end = time.monotonic() + args.minutes * 60
    i = 0
    while time.monotonic() < end:
        i += 1
        body = {'model': args.model,
                'messages': [{'role': 'user', 'content': prompt_for(i)}],
                'max_tokens': args.max_tokens, 'temperature': 0,
                'top_k': 1, 'top_p': 1, 'min_p': 0, 'seed': 42,
                'reasoning_effort': 'none'}
        req = urllib.request.Request(url + '/v1/chat/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
            choice = data['choices'][0]
            requests.append({'n': i, 'wall_s': round(time.monotonic() - t0, 2),
                             'finish': choice.get('finish_reason'),
                             'usage': data.get('usage'),
                             'head': (choice['message'].get('content') or '')[:80]})
            log(f'prompt {i}: {requests[-1]["wall_s"]} s, '
                f'{requests[-1]["finish"]}')
        except Exception as exc:                        # noqa: BLE001
            requests.append({'n': i, 'wall_s': round(time.monotonic() - t0, 2),
                             'error': f'{type(exc).__name__}: {exc}'})
            log(f'prompt {i}: ERROR {requests[-1]["error"]}')
        json_path.write_text(json.dumps(
            {'started': started, 'minutes': args.minutes,
             'interval_s': args.interval, 'config': args.config,
             'requests': requests}, indent=2) + '\n')
        if time.monotonic() < end:
            time.sleep(5)

    log(f'{i} prompts sent; stopping the sampler and the server')
    stop_file.write_text('stop\n')
    # sample_mem.py sleeps a whole --interval between checks, so give it one full
    # interval plus a margin; this soak's first run raised TimeoutExpired here and
    # skipped the summary (scripts/soak_summary.py rebuilds it from the CSV).
    try:
        sampler.wait(timeout=args.interval + 60)
    except subprocess.TimeoutExpired:
        sampler.kill()
        sampler.wait(timeout=10)
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + 30
    while server.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if server.poll() is None:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    subprocess.run(['pkill', '-f', 'engine/strata --serve'], check=False)

    # the engine log segment this soak wrote
    segment = ''
    if engine_log.exists():
        with engine_log.open('rb') as fh:
            fh.seek(log_offset)
            segment = fh.read().decode('utf-8', 'replace')
        (out / 'soak-engine.log').write_text(segment)

    samples = []
    if csv_path.exists():
        with csv_path.open() as fh:
            samples = list(csv.DictReader(fh))
    vram_cols = [c for c in (samples[0].keys() if samples else []) if c.endswith('_vram_used')]

    def col(name):
        return [float(r[name]) for r in samples if r.get(name) not in (None, '', '-1')]

    def peak_stat(name):
        vals = col(name)
        if not vals:
            return None
        first, peak = vals[0], max(vals)
        return {'first': round(first / 2**30, 3) if 'vram' in name else round(first / 2**20, 1),
                'peak': round(peak / 2**30, 3) if 'vram' in name else round(peak / 2**20, 1),
                'unit': 'GiB' if 'vram' in name else 'MiB'}

    fault_re = re.compile(r'no progress|\bnan\b|crash|never rang|doorbell|abort|'
                          r'segmentation fault|\bkilled\b|out of memory', re.I)
    faults = [ln for ln in segment.splitlines() if fault_re.search(ln)]
    summary = {
        'started': started, 'ended': time.time(),
        'minutes_asked': args.minutes, 'interval_s': args.interval,
        'config': args.config, 'samples': len(samples),
        'prompts': len(requests),
        'prompt_errors': [r for r in requests if 'error' in r],
        'survived': not any('error' in r for r in requests) and not faults,
        'server_returncode': server.returncode,
        'engine_rss_mib': peak_stat('engine_rss_mib'),
        'ram_used_mib': peak_stat('ram_used_mib'),
        'vram_used': {c.replace('_vram_used', ''): peak_stat(c) for c in vram_cols},
        'fault_lines': faults,
    }
    json_path.write_text(json.dumps(summary, indent=2) + '\n')
    log('soak summary: ' + json.dumps({k: v for k, v in summary.items()
                                       if k != 'fault_lines'}))


if __name__ == '__main__':
    main()
