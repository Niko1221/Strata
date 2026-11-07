#!/usr/bin/env python3
"""Long-context and functional legs of the Strata release battery.

Runs against an otherwise idle Strata server and reads the engine's OWN timing line
(the same regex the prefill bench uses), so the numbers are the engine's, not a client guess.

Legs, identical for every arm so two releases are comparable:
  code       short request, a real answer must come back (functional smoke)
  long       one fresh prompt of ~27.8K tokens, 1 token out -> prefill tok/s at that size
  needle     ~7.5K-token document with a code buried mid-way -> must quote it exactly
  overflow   a prompt past the config's 131072-token window -> must be REFUSED cleanly

Usage:
  strata-longprobe.py --url http://127.0.0.1:8095 --model <id> \
      --engine-log <tree>/strata-coder-iq1_m-sycl-tuned.log --label <arm> --out <json>
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PATTERN = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), '
                     r'(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps', 'generated', 'decode_ms', 'decode_tps')
LINE = 'export function rule{i}(x) {{ return x === {i} ? x + {ip1} : x - {i}; }}'


def rules(n, needle=None, at=0):
    out = []
    for i in range(n):
        if needle and i == at:
            out.append(needle)
        out.append(LINE.format(i=i, ip1=i + 1))
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8095')
    p.add_argument('--model', required=True)
    p.add_argument('--engine-log', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    base = args.url.rstrip('/') + '/v1/chat/completions'
    results = []

    def call(messages, max_tokens, timeout=900):
        body = dict(model=args.model, messages=messages, max_tokens=max_tokens, temperature=0,
                    top_k=1, top_p=1, min_p=0, seed=42, reasoning_effort='none')
        req = urllib.request.Request(base, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        offset = args.engine_log.stat().st_size
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            return dict(http=e.code, body=e.read().decode(errors='replace')[:400],
                        wall_s=round(time.monotonic() - t0, 2))
        wall = round(time.monotonic() - t0, 2)
        deadline = time.monotonic() + 2
        metrics = None
        while True:
            with args.engine_log.open('rb') as log:
                log.seek(offset)
                m = PATTERN.findall(log.read().decode(errors='replace'))
            if m or time.monotonic() >= deadline:
                if m:
                    metrics = dict(zip(FIELDS, map(float, m[0])))
                break
            time.sleep(0.02)
        choice = data['choices'][0]
        return dict(http=200, wall_s=wall, metrics=metrics, usage=data.get('usage'),
                    finish=choice.get('finish_reason'),
                    content=(choice['message'].get('content') or '')[:400])

    def record(leg, payload):
        row = dict(label=args.label, leg=leg, **payload)
        results.append(row)
        args.out.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row)[:600], flush=True)

    # 1. functional smoke: a code answer, not just a 200
    record('code', call([{'role': 'user', 'content':
            'Write a Python function that reverses a singly linked list in place. Code only, no prose.'}], 200))

    # 2. the previous release's long-prompt leg: ~27.8K tokens read in one shot
    record('long', call([{'role': 'user', 'content':
            'Read this source and state in one sentence what every function does.\n' + rules(927)}], 1))

    # 3. needle inside a ~7.5K window
    needle = '// REMEMBER THIS CODE: ZQ-7731-KX'
    record('needle', call([{'role': 'user', 'content':
            'A code is hidden in the source below. Reply with only the code.\n' + rules(250, needle, 125)}], 32))

    # 4. past the window: must be refused cleanly, never truncated
    record('overflow', call([{'role': 'user', 'content':
            'Summarize.\n' + rules(4600)}], 1))

    print('done', args.label)


if __name__ == '__main__':
    main()
