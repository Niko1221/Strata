#!/usr/bin/env python3
"""Interleaved A/B of 500-token greedy generations, one server restart per arm.

The two arms are the same server config with one engine argument changed (the
caller builds both configs). Requests alternate A, B, A, B, ... for `--pairs`
pairs, and the server is stopped and started before every request, so no arm
runs on a cache the other arm warmed.

Every request is a fresh conversation at temperature 0 with a fixed seed. The
decode speed is parsed from the engine's own completion line in the engine log
(the same line bench_prefill.py parses), never from wall time; wall time and a
streamed time-to-first-token are recorded next to it.

Usage:
  interleave500.py --config-a CFG --config-b CFG --pairs 10 \
      --model NAME --engine-log LOG --output JSON --server-out FILE
"""
import argparse
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
LINE = re.compile(
    r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), '
    r'(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps',
          'generated', 'decode_ms', 'decode_tps')

# Five prompts of a similar shape but a different length (about 1.8K, 2.4K, 3.0K,
# 3.6K and 4.2K prompt tokens, counted by the engine at run time); pair i uses
# prompt i % 5, so both arms of a pair see exactly the same text.
QUESTIONS = [
    'Review this source and describe its behavior precisely in a paragraph. Then '
    'explain the most relevant boundary case and the smallest useful regression '
    'test for it, covering integer equality, zero, negative input and unexpected '
    'types.',

    'Summarize what this module does in prose, then list three ways it could be '
    'made faster without changing its results, with the cost of each and a short '
    'sketch of the change.',

    'Write a code review of this file: naming, error handling, edge cases and '
    'tests. Number your findings by severity and give the smallest patch that '
    'would fix the worst one.',

    'Explain this code to a new maintainer: the data flow, what state survives '
    'between calls, and what would break first if the input grew by 100x. End '
    'with two questions you would ask the original author.',

    'Turn this into a short design note: what problem it solves, the invariants '
    'it keeps, how it fails, and what you would monitor in production. Be '
    'specific about the numbers you would watch.',
]


def make_prompt(i):
    """The prompt for pair i % 5: a code listing of n rules plus a question."""
    n = 60 + 20 * (i % 5)
    code = '\n'.join(
        f'export function rule{j}(x) {{ return x === {j} ? x + {j + 1} '
        f': x - {j}; }}' for j in range(n))
    return f'CASE {i}: {QUESTIONS[i % 5]}\n\n{code}'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def request_json(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Server:
    def __init__(self, config, out_path, url, log_path):
        self.config = str(config)
        self.out_path = Path(out_path)
        self.url = url
        self.log_path = Path(log_path)
        self.proc = None

    def start(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        out = self.out_path.open('ab')
        self.proc = subprocess.Popen(
            [str(PY), str(ROOT / 'serve/server.py'), '--engine', 'strata',
             '--config', self.config, '--port', '8080'],
            cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'server exited with {self.proc.returncode}; '
                                   f'see {self.out_path}')
            try:
                with urllib.request.urlopen(self.url + '/health', timeout=5) as r:
                    if json.load(r).get('loaded'):
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError(f'server never reported loaded; see {self.out_path}')

    def stop(self):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 30
        while self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait(timeout=10)
        self.proc = None
        subprocess.run(['pkill', '-f', 'engine/strata --serve'], check=False)
        time.sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-a', required=True)
    p.add_argument('--config-b', required=True)
    p.add_argument('--pairs', type=int, default=10)
    p.add_argument('--model', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--engine-log', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--server-out', required=True)
    p.add_argument('--max-tokens', type=int, default=500)
    args = p.parse_args()

    url = args.url.rstrip('/')
    engine_log = Path(args.engine_log)
    results = []
    configs = {'A': args.config_a, 'B': args.config_b}

    def serve(arm, pair, prompt_id):
        srv = Server(configs[arm], args.server_out, url, engine_log)
        log(f'pair {pair + 1}/{args.pairs} arm {arm}: starting the server')
        with open(args.server_out, 'ab') as fh:
            fh.write(f'\n===== pair {pair + 1} arm {arm} '
                     f'{time.strftime("%Y-%m-%d %H:%M:%S")} =====\n'.encode())
        srv.start()
        offset = engine_log.stat().st_size
        body = {
            'model': args.model,
            'messages': [{'role': 'user', 'content': make_prompt(prompt_id)}],
            'max_tokens': args.max_tokens,
            'temperature': 0, 'top_k': 1, 'top_p': 1, 'min_p': 0, 'seed': 42,
            'reasoning_effort': 'none',
            'stream': True,
        }
        rec = {'pair': pair + 1, 'arm': arm, 'prompt_id': prompt_id,
               'config': configs[arm]}
        text = []
        try:
            req = urllib.request.Request(
                url + '/v1/chat/completions', data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            ttft = None
            usage = None
            finish = None
            with urllib.request.urlopen(req, timeout=900) as resp:
                for raw in resp:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta') or {}
                        got = delta.get('content') or delta.get('reasoning_content')
                        if got:
                            if ttft is None:
                                ttft = time.monotonic() - start
                            if delta.get('content'):
                                text.append(delta['content'])
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
            wall = time.monotonic() - start
        except Exception as exc:                       # noqa: BLE001
            rec['error'] = f'{type(exc).__name__}: {exc}'
            log(f'pair {pair + 1} arm {arm}: REQUEST FAILED {rec["error"]}')
            results.append(rec)
            Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
            srv.stop()
            return

        metrics = None
        raw_line = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with engine_log.open('rb') as fh:
                fh.seek(offset)
                blob = fh.read().decode('utf-8', 'replace')
            matches = [l for l in blob.splitlines() if LINE.search(l)]
            if matches:
                raw_line = matches[0]
                metrics = dict(zip(FIELDS,
                                   map(float, LINE.search(raw_line).groups())))
                break
            time.sleep(0.05)
        rec.update({'wall_s': round(wall, 3),
                    'ttft_s': round(ttft, 3) if ttft is not None else None,
                    'metrics': metrics, 'engine_line': raw_line,
                    'finish_reason': finish, 'usage': usage,
                    'chars': sum(len(t) for t in text),
                    'text_head': ''.join(text)[:120]})
        if metrics is None:
            rec['error'] = 'no engine timing line found in the log'
        log(f'pair {pair + 1} arm {arm}: '
            + (f"decode {metrics['decode_tps']:.1f} tok/s, "
               f"prompt {metrics['prefill_tps']:.1f} tok/s, "
               f"ttft {rec['ttft_s']} s"
               if metrics else 'NO ENGINE LINE'))
        results.append(rec)
        Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
        srv.stop()

    for pair in range(args.pairs):
        prompt_id = pair % len(QUESTIONS)
        for arm in ('A', 'B'):          # A first in every pair, B right after
            serve(arm, pair, prompt_id)

    log(f'done: {len(results)} requests -> {args.output}')


if __name__ == '__main__':
    main()
