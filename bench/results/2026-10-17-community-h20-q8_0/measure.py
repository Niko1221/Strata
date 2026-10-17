#!/usr/bin/env python3
"""Replay synthetic fixtures against an already running server; never starts or reconfigures it.

request_one is copied verbatim from the archived measurement client. The CLI and
fixture wrapper were added for publication and have only been checked offline.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
BACKENDS = {}


def fixtures():
    rows = json.loads((HERE / 'prompts.json').read_text())['prompts']
    result = {}
    for row in rows:
        text = ''.join(p['text'] if 'text' in p else p['repeat'] * p['count'] for p in row['segments'])
        assert len(text.encode()) == row['bytes']
        assert hashlib.sha256(text.encode()).hexdigest() == row['sha256']
        result[row['id']] = text
    return result


def request_one(backend, content, limit, barrier):
    info = BACKENDS[backend]
    body = {'model': info['model'], 'messages': [{'role': 'user', 'content': content}],
        'max_tokens': limit, 'temperature': 0, 'top_p': 1, 'seed': 42,
        'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_effort': 'none',
        'stream': True, 'stream_options': {'include_usage': True}}
    request = urllib.request.Request(info['base'] + '/v1/chat/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', **info['headers']})
    barrier.wait()
    begin = time.perf_counter()
    first = None
    usage = timings = finish = None
    chunks, errors = [], []
    reasoning_chars = 0
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                value = line[5:].strip()
                if value == '[DONE]':
                    break
                obj = json.loads(value)
                if obj.get('error'):
                    errors.append(str(obj['error'])[:400])
                usage = obj.get('usage') or usage
                timings = obj.get('timings') or timings
                for choice in obj.get('choices', []):
                    delta = choice.get('delta') or {}
                    text = delta.get('content') or ''
                    reasoning_chars += len(delta.get('reasoning_content') or '')
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        chunks.append(text)
                    finish = choice.get('finish_reason') or finish
    except urllib.error.HTTPError as exc:
        errors.append('HTTP ' + str(exc.code) + ': ' + exc.read(400).decode(errors='replace'))
    except Exception as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc)[:300])
    end = time.perf_counter()
    generated = (usage or {}).get('completion_tokens', 0)
    decode = (generated - 1) / (end - first) if first and generated > 1 else None
    return dict(start=begin, end=end, ttft_s=first-begin if first else None, total_s=end-begin,
        client_decode_tps=decode, completion_tokens=generated, prompt_tokens=(usage or {}).get('prompt_tokens'),
        cached_tokens=((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens'),
        usage=usage, engine_timings=timings, finish_reason=finish, output=''.join(chunks),
        reasoning_chars=reasoning_chars, errors=errors,
        ok=not errors and bool(chunks) and generated == limit and reasoning_chars == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    write = sub.add_parser('prompt')
    write.add_argument('id')
    run = sub.add_parser('request')
    run.add_argument('--url', required=True)
    run.add_argument('--model', required=True)
    run.add_argument('--ids', nargs='+', required=True)
    run.add_argument('--max-tokens', type=int, required=True)
    run.add_argument('--api-key-env', default='STRATA_BENCH_API_KEY')
    args = parser.parse_args()
    prompts = fixtures()
    if args.command == 'verify':
        print(json.dumps({'verified': len(prompts), 'prompt_ids': list(prompts)}, indent=2))
        return
    if args.command == 'prompt':
        print(prompts[args.id], end='')
        return
    key = os.environ.get(args.api_key_env)
    BACKENDS['target'] = {'base': args.url.rstrip('/'), 'model': args.model,
                          'headers': {'Authorization': 'Bearer ' + key} if key else {}}
    contents = [prompts[name] for name in args.ids]
    barrier = threading.Barrier(len(contents))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(contents)) as pool:
        rows = list(pool.map(lambda content: request_one('target', content, args.max_tokens, barrier), contents))
    wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
    print(json.dumps({'requests': rows, 'wall_s': wall,
                      'aggregate_tps': sum(r['completion_tokens'] for r in rows) / wall}, indent=2))


if __name__ == '__main__':
    main()
