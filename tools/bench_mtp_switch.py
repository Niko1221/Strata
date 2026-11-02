#!/usr/bin/env python3
"""Measure one-GPU requests or a dual-GPU service's 1/2/1 request sequence.

Uses an already-running OpenAI-compatible server. Does not change configuration.
Stream timing cannot prove which engine path ran: retain the matching engine log.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:58159')
    parser.add_argument('--model', required=True)
    parser.add_argument('--build-id', required=True, help='Exact tested commit/binary hash')
    parser.add_argument('--topology', choices=['single-gpu', 'dual-gpu'], required=True)
    parser.add_argument('--scenario', choices=['solo', 'switch'], default='solo')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--short-tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--reasoning-effort', choices=['none', 'low', 'medium', 'high'], default='none')
    parser.add_argument('--api-key-env', default='STRATA_BENCH_API_KEY')
    args = parser.parse_args()
    if min(args.tokens, args.short_tokens, args.repeats) <= 0 or args.timeout <= 0:
        parser.error('token counts, repeats and timeout must be positive')
    if args.scenario == 'switch' and args.short_tokens >= args.tokens:
        parser.error('switch needs --short-tokens below --tokens')
    if args.output.exists():
        parser.error('output already exists; preserve previous records')
    prompt = args.prompt_file.read_text(encoding='utf-8')
    started = time.perf_counter()
    now = lambda: time.perf_counter() - started
    rows = []
    phases = []

    def request(label, tokens, first=None, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=args.timeout)
        row = dict(label=label, requested_tokens=tokens, started_s=now(), delta_times_s=[], usage=None,
                   finish_reason=None, done=False, error=None)
        digests = {k: hashlib.sha256() for k in ['content', 'reasoning_content']}
        body = dict(model=args.model, messages=[dict(role='user', content=prompt)], max_tokens=tokens,
                    temperature=0, seed=42, stream=True, stream_options=dict(include_usage=True),
                    reasoning_effort=args.reasoning_effort)
        if args.reasoning_effort == 'none':
            body['chat_template_kwargs'] = dict(enable_thinking=False)
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get(args.api_key_env)
        if key:
            headers['Authorization'] = 'Bearer ' + key
        try:
            req = urllib.request.Request(args.endpoint.rstrip('/') + '/v1/chat/completions',
                                         data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                for line in response:
                    if now() - row['started_s'] > args.timeout:
                        raise TimeoutError('Total request deadline exceeded')
                    text = line.decode('utf-8').strip()
                    if not text.startswith('data:'):
                        continue
                    payload = text[5:].strip()
                    if payload == '[DONE]':
                        row['done'] = True
                        break
                    event = json.loads(payload)
                    if event.get('error'):
                        # Keep a failure without copying a server error that may contain private paths.
                        raise RuntimeError('Server returned an SSE error')
                    if event.get('usage') is not None:
                        row['usage'] = event['usage']
                    for choice in event.get('choices', []):
                        if choice.get('finish_reason') is not None:
                            row['finish_reason'] = choice['finish_reason']
                        delta = choice.get('delta') or {}
                        pieces = {k: delta[k] for k in ['content', 'reasoning_content'] if delta.get(k)}
                        if pieces:
                            for name, piece in pieces.items():
                                digests[name].update(piece.encode())
                            row['delta_times_s'].append(now())
                            if first is not None:
                                first.set()
            if not row['done'] or row['finish_reason'] in [None, 'error', 'cancel', 'cancelled']:
                raise RuntimeError('Incomplete or cancelled stream')
        except Exception as error:
            row['error'] = type(error).__name__
        row['ended_s'] = now()
        row['text_sha256'] = {k: h.hexdigest() for k, h in digests.items()}
        row['ttft_s'] = (row['delta_times_s'][0] - row['started_s']) if row['delta_times_s'] else None
        row['wall_s'] = row['ended_s'] - row['started_s']
        count = (row['usage'] or {}).get('completion_tokens')
        row['e2e_output_tps'] = count / row['wall_s'] if isinstance(count, int) and count >= 0 else None
        return row

    def record(name, items, extra=None):
        rows.extend(items)
        wall = max(r['ended_s'] for r in items) - min(r['started_s'] for r in items)
        counts = [(r['usage'] or {}).get('completion_tokens') for r in items]
        valid = all(isinstance(n, int) and n >= 0 for n in counts) and all(r['error'] is None for r in items)
        phases.append(dict(name=name, requests=[r['label'] for r in items], wall_s=wall,
                           aggregate_e2e_output_tps=sum(counts) / wall if valid else None, **(extra or {})))
        save()

    def save():
        report = dict(build_id=args.build_id, topology=args.topology, scenario=args.scenario,
                      model=args.model, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      prompt_bytes=len(prompt.encode()), tokens=args.tokens, short_tokens=args.short_tokens,
                      repeats=args.repeats, temperature=0, seed=42, reasoning_effort=args.reasoning_effort,
                      cache_policy='server state retained',
                      timing='HTTP end-to-end; usage token counts only; no estimated decode rates',
                      phases=phases, requests=rows)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    for n in range(args.repeats):
        record('solo-before', [request(f'solo-before-{n}', args.tokens)])
    if args.scenario == 'switch':
        with ThreadPoolExecutor(max_workers=2) as pool:
            for n in range(args.repeats):
                barrier = threading.Barrier(2)
                futures = [pool.submit(request, f'pair-{n}-{i}', args.tokens, None, barrier) for i in range(2)]
                items = [f.result() for f in futures]
                overlap = min(r['ended_s'] for r in items) > max(r['started_s'] for r in items)
                record('simultaneous', items, dict(http_intervals_overlap=overlap))
            event = threading.Event()
            long = pool.submit(request, 'transition-long', args.tokens, event)
            deadline = time.perf_counter() + args.timeout
            while not event.wait(0.05):
                if long.done() or time.perf_counter() > deadline:
                    break
            short = request('transition-short', args.short_tokens) if event.is_set() else None
            primary = long.result()
            if short is not None:
                overlap = short['started_s'] < primary['ended_s']
                resumed = sum(t > short['ended_s'] for t in primary['delta_times_s'])
                record('staggered', [primary, short], dict(http_intervals_overlap=overlap,
                       primary_deltas_after_secondary=resumed,
                       observed_1_2_1=overlap and resumed > 0 and primary['error'] is None and short['error'] is None))
            else:
                record('staggered', [primary], dict(observed_1_2_1=False))
        for n in range(args.repeats):
            record('solo-after', [request(f'solo-after-{n}', args.tokens)])
    failed = any(r['error'] for r in rows) or any(
        p.get('observed_1_2_1') is False or p.get('http_intervals_overlap') is False for p in phases)
    print(json.dumps(dict(output=str(args.output), requests=len(rows), failed=failed)))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
