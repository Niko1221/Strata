#!/usr/bin/env python3
"""N concurrent 500-token generations against one `--batch N` server.

The short arms (scripts/ab-run.sh) send one request at a time, so on their own
`--batch 2` only shows that the engine still starts and serves. This sends N
requests at once (the batch slots exist for exactly that) and reports

  * each slot's decode tok/s from the server's own per-request "done" line -
    the engine's completion line is one per prompt chunk under --batch, so it
    cannot be summed,
  * the aggregate over the N slots,
  * time to first token and wall time per slot,
  * whether any slot failed to finish - the "--batch slot waits for a doorbell
    that never rings" failure shows up here as a timeout.

Usage:
  batch_concurrent.py --config CFG --model NAME --engine-log LOG \
      --out FILE --n 2 [--max-tokens 500]
"""
import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from interleave500 import FIELDS, LINE, Server, log, make_prompt


def one(url, model, slot, prompt_id, max_tokens, results, lock):
    """Stream one completion; the record is appended under `lock`."""
    body = {'model': model,
            'messages': [{'role': 'user', 'content': make_prompt(prompt_id)}],
            'max_tokens': max_tokens, 'temperature': 0,
            'top_k': 1, 'top_p': 1, 'min_p': 0, 'seed': 42,
            'reasoning_effort': 'none', 'stream': True}
    rec = {'slot': slot, 'prompt_id': prompt_id}
    t0 = time.monotonic()
    ttft, chars, finish, err = None, [], None, None
    try:
        req = urllib.request.Request(
            url + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta') or {}
                    got = delta.get('content') or delta.get('reasoning_content')
                    if got:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        if delta.get('content'):
                            chars.append(delta['content'])
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
    except (urllib.error.URLError, OSError, ValueError) as exc:
        err = f'{type(exc).__name__}: {exc}'
    rec.update({'ttft_s': round(ttft, 3) if ttft else None,
                'wall_s': round(time.monotonic() - t0, 2),
                'finish': finish, 'error': err,
                'text': ''.join(chars)[:200]})
    with lock:
        results.append(rec)
        log(f'slot {slot}: {rec["wall_s"]} s, ttft {rec["ttft_s"]}, '
            f'finish={rec["finish"]}' + (f', error {err}' if err else ''))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--engine-log', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--max-tokens', type=int, default=500)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--timeout', type=float, default=300,
                   help='seconds a slot may run before it counts as hung')
    args = p.parse_args()

    url = args.url.rstrip('/')
    engine_log = Path(args.engine_log)
    server_out = Path(args.out).with_suffix('.server.out')
    # Server.start() appends, and both N-runs reuse the same file name: start from
    # an empty one so the "[strata] done:" lines below belong to this run only.
    server_out.parent.mkdir(parents=True, exist_ok=True)
    server_out.write_bytes(b'')
    server = Server(args.config, server_out, url, engine_log)
    offset = engine_log.stat().st_size if engine_log.exists() else 0
    results, lock = [], threading.Lock()

    log(f'starting the server for --batch {args.n}')
    server.start()
    log('server ready; sending %d concurrent requests' % args.n)
    threads = [threading.Thread(target=one,
                                args=(url, args.model, k, k, args.max_tokens,
                                      results, lock))
               for k in range(args.n)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.timeout)
    hung = [k for k, t in enumerate(threads) if t.is_alive()]
    server.stop()

    lines = []
    if engine_log.exists():
        with engine_log.open('rb') as fh:
            fh.seek(offset)
            segment = fh.read().decode('utf-8', 'replace')
        Path(args.out).with_suffix('.engine.log').write_text(segment)
        for m in LINE.finditer(segment):
            lines.append(dict(zip(FIELDS, m.groups())))

    # The engine's completion line is not one-per-request under --batch: the prompt
    # path logs a line per chunk ("1 checkpoints") and slots hand their context to
    # each other, so summing them would count the same request several times. The
    # server prints one "[strata] done: N tokens in T s (X tok/s) (finish, cancel=...)"
    # line per finished request, which is unambiguous.
    done = [{'tokens': int(t), 'wall_s': float(s), 'decode_tps': float(d),
             'finish': f, 'cancelled': c}
            for t, s, d, f, c in re.findall(
                r'\[strata\] done: (\d+) tokens in ([\d.]+) s \(([0-9.]+) tok/s\) '
                r'\((\w+), cancel=(\w+)\)',
                server_out.read_text(encoding='utf-8', errors='replace'))] \
        if server_out.exists() else []
    agg = round(sum(r['decode_tps'] for r in done), 1) if done else None
    out = {
        'batch': args.n, 'config': args.config, 'max_tokens': args.max_tokens,
        'requests': sorted(results, key=lambda r: r['slot']),
        'hung_slots': hung, 'elapsed_s': round(time.monotonic() - started, 1),
        'server_done': done, 'aggregate_decode_tps': agg,
        'engine_lines': lines, 'engine_log_lines_found': len(lines),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    log(f'aggregate decode {agg} tok/s over {len(done)} finished request(s); '
        f'{len(lines)} engine line(s); hung slots: {hung}; wrote {args.out}')


if __name__ == '__main__':
    main()
