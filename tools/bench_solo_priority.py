"""Compare an unchanged server and an opt-in solo_priority server with the same fixture.

STRATA_API_KEY is read from the environment; it is never written to the result.
Run once with each config. The tool neither changes configs nor executes model tools.

python tools/bench_solo_priority.py --fixture fixture.json --out baseline.json
python tools/bench_solo_priority.py --fixture fixture.json --out priority.json

Fixture: {"messages": [...], "max_tokens": 3072}. Use identical input, a warm cache,
and at least three repetitions per arm. Cold and warm reads must be reported separately.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def request(base, body):
    started = time.monotonic()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    wire = {"temperature": 0, "seed": 117, "reasoning_effort": "none", **body,
            "stream": True, "stream_options": {"include_usage": True}}
    content, reasoning, tools, first, usage, timings, done = [], [], {}, None, None, None, False
    with urlopen(Request(base.rstrip('/')+'/v1/chat/completions', data=json.dumps(wire).encode(),
                         headers=headers), timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            data = line[5:].strip()
            if data == b'[DONE]': done = True; break
            event = json.loads(data)
            if event.get('error'): raise RuntimeError(str(event['error']))
            usage = event.get('usage') or usage
            timings = event.get('timings') or timings
            for choice in event.get('choices', []):
                d = choice.get('delta', {})
                # A role/tool-name header is not a model token or the first usable answer.
                if first is None and (d.get('content') or d.get('reasoning_content') or
                        any(t.get('function', {}).get('arguments') for t in d.get('tool_calls', []))):
                    first = time.monotonic()
                content.append(d.get('content') or '')
                reasoning.append(d.get('reasoning_content') or '')
                for call in d.get('tool_calls', []):
                    item = tools.setdefault(call['index'], {'name':'', 'arguments':''})
                    for key in item: item[key] += call.get('function', {}).get(key) or ''
    ended = time.monotonic()
    if not done: raise RuntimeError('Stream ended without [DONE]')
    return {'start':started, 'end':ended, 'wall_s':ended-started,
            'ttft_s':first-started if first is not None else None,
            'usage':usage, 'timings':timings, 'content':''.join(content),
            'reasoning_content':''.join(reasoning), 'tool_calls':list(tools.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8080')
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--delay', type=float, default=7)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    if a.out.exists(): p.error('output already exists; choose a new path')
    fixture = json.loads(a.fixture.read_text(encoding='utf-8'))
    short = {'messages':[{'role':'user','content':'Reply with only this JSON exactly: {"check":"COPPER-731","value":42,"missing":null}'}],
             'max_tokens':128, 'priority':1}
    rows = []
    archive = fixture.pop('archive', False)
    fixture.pop('sha256', None)
    expected = fixture.pop('expected', None)
    task = fixture.pop('task', None)
    warm = None
    if archive:
        warm = request(a.base, {**fixture, 'max_tokens':1024})
        if json.loads(warm['content']) != expected:
            raise RuntimeError('Warm audit did not match the expected facts')
        fixture['messages'] += [{'role':'assistant','content':warm['content']}]
    for index in range(a.rounds):
        job = dict(fixture)
        if archive:
            text=(f'Performance round{index+1}. Return JSON with exactly two keys in this order: '
                  '"audit": the complete answer object to the archive TASK (all20 fields); '
                  '"code": a JSON string containing a substantive Python module implementing a transactional diving-game inventory, '
                  'idempotent purchases, validated saves, oxygen transitions, replay, and at least100 unit test functions. '
                  'Include exact integer validation, atomic failures, cap boundaries and unknown-license handling. '
                  'Produce the full implementation and many tests, at least6000 words; do not abbreviate or finish early. '
                  'The archive facts remain authoritative. Do not call tools.')
            job['messages'] = fixture['messages'] + [{'role':'user','content':text}]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            long = pool.submit(request, a.base, job)
            time.sleep(a.delay)
            small = pool.submit(request, a.base, short)
            results = [long.result(), small.result()]
        wall = max(x['end'] for x in results) - min(x['start'] for x in results)
        tokens = sum((x['usage'] or {}).get('completion_tokens', 0) for x in results)
        if archive:
            # The intentionally capped code string may not close, but its audit precedes it.
            start = results[0]['content'].find('{', results[0]['content'].find('"audit"'))
            audit, _ = json.JSONDecoder().raw_decode(results[0]['content'][start:])
            results[0]['audit_correct'] = audit == expected
        rows.append({'clients':results, 'pair_wall_s':wall, 'aggregate_tok_s':tokens/wall})
        a.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'pair_wall_s':wall, 'short_ttft_s':results[1]['ttft_s'],
                          'aggregate_tok_s':tokens/wall}), flush=True)


if __name__ == '__main__': main()
