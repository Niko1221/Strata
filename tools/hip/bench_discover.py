#!/usr/bin/env python3
"""Serial, token-counted streaming probes for an idle, dedicated Strata server.

Run with the model's strata_tokenizer extension installed. Restart the engine
between configuration arms. This script never changes context or starts services.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate

TIMING = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), (\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps', 'generated', 'decode_ms', 'decode_tps')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--engine-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--sizes', default='1024,4096,8192')
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--max-tokens', type=int, default=32)
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--followups', action='store_true', help='probe cached continuation after each fresh request')
    a = p.parse_args()
    if a.repetitions < 1 or a.max_tokens < 1:
        p.error('repetitions and output room must be positive')
    sizes = [int(x) for x in a.sizes.split(',')]
    if any(n <= 64 or n + a.max_tokens + 8 > 131072 for n in sizes):
        p.error('each prompt must leave output room plus 8 tokens in 128K')
    import strata_tokenizer as ST
    vocab = json.loads((a.tokenizer / 'vocab.json').read_text())
    tokens = [''] * (max(vocab.values()) + 1)
    for text, i in vocab.items():
        tokens[i] = text
    tok = ST.Tokenizer(tokens, (a.tokenizer / 'merges.txt').read_text().split('\n'),
                       json.loads((a.tokenizer / 'token_type.json').read_text()))
    template = ChatTemplate(a.tokenizer / 'chat_template.jinja')
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('STRATA_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
    base = a.url.rstrip('/')
    with urllib.request.urlopen(urllib.request.Request(base + '/health', headers=headers)) as r:
        health = json.load(r)
    if health['max_context'] != 131072:
        raise RuntimeError('server does not retain exactly 128K context')
    results = {'label': a.label, 'health': health, 'samples': []}

    def probe(target, trial, warmup=False):
        # Distinct early content prevents reuse of prior recurrent checkpoints.
        prefix = f'TRIAL {trial} SIZE {target}. Read the notes and reply READY.\n'
        def prompt(count):
            messages = [{'role': 'user', 'content': prefix + ' alpha' * count}]
            ids = tok.encode(template.render(messages, enable_thinking=False), parse_special=True)
            return messages, ids
        lo, hi = 0, target
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(prompt(mid)[1]) <= target:
                lo = mid
            else:
                hi = mid - 1
        messages, ids = prompt(lo)
        reply = request(messages, ids, target, trial, warmup)
        if a.followups and not warmup:
            messages += [{'role': 'assistant', 'content': reply},
                         {'role': 'user', 'content': 'Reply READY again.'}]
            ids = tok.encode(template.render(messages, enable_thinking=False), parse_special=True)
            if len(ids) + a.max_tokens + 8 > 131072:
                raise RuntimeError('continuation leaves insufficient context; choose a smaller fresh target')
            request(messages, ids, target, trial, fresh=False)

    def request(messages, ids, target, trial, warmup=False, fresh=True):
        offset = a.engine_log.stat().st_size
        req = urllib.request.Request(base + '/v1/chat/completions', headers=headers,
            data=json.dumps(dict(model=a.model, messages=messages, max_tokens=a.max_tokens,
                stream=True, stream_options={'include_usage': True}, temperature=0,
                top_k=1, top_p=1, min_p=0, seed=42, reasoning_effort='none')).encode())
        start = time.monotonic()
        first = None
        chunks, usage, finish = [], None, None
        with urllib.request.urlopen(req, timeout=a.timeout) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                data = json.loads(line[6:])
                if data.get('usage'):
                    usage = data['usage']
                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    text = delta.get('content') or delta.get('reasoning_content') or ''
                    if text:
                        if first is None:
                            first = time.monotonic() - start
                        chunks.append(text)
                    finish = choice.get('finish_reason') or finish
        wall = time.monotonic() - start
        deadline = time.monotonic() + 2
        while True:
            with a.engine_log.open('rb') as log:
                log.seek(offset)
                matches = TIMING.findall(log.read().decode(errors='replace'))
            if matches or time.monotonic() >= deadline:
                break
            time.sleep(.02)
        if len(matches) != 1:
            raise RuntimeError('expected one completed timing line from an idle server')
        metrics = dict(zip(FIELDS, map(float, matches[0])))
        if not fresh and metrics['reused'] <= 0:
            raise RuntimeError('cached continuation did not reuse prompt state')
        if metrics['prompt_tokens'] != len(ids) or (fresh and metrics['reused'] != 0):
            raise RuntimeError('token count mismatch or fresh prompt reused state')
        sample = dict(target=target, trial=trial, warmup=warmup, kind='fresh' if fresh else 'followup', actual_tokens=len(ids),
            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), ttft_s=first,
            wall_s=wall, metrics=metrics, usage=usage, finish_reason=finish, text=''.join(chunks))
        results['samples'].append(sample)
        a.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(sample), flush=True)
        return sample['text']

    probe(128, 0, True)
    for target in sizes:
        for trial in range(1, a.repetitions + 1):
            probe(target, trial)


if __name__ == '__main__':
    main()
