#!/usr/bin/env python3
"""Decode speed on three short questions against a running Strata server (OpenAI-compatible API).

  python3 decode_probe.py [--port 18085] [--minps 0.5] [--reps 2] [--text-out replies.jsonl]

Greedy, no thinking, streamed. Each question is sent once as a warm-up, then --reps times per spec_min_p in --minps
(sent per request as strata_tune.spec_min_p). tok/s = (completion_tokens - 1) / (last - first content token time).
Output: "dec minp=<p> rep=<r> <label>: gen <n> tok <t/s> t/s" and "dec-mean minp=<p>: en <> ja <> code <> mean <>"
(each question: the median of its reps). --text-out appends every reply (warm-up rep 0 included) as one JSON line.
"""
import argparse
import json
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=18085)
ap.add_argument('--minps', default='0.5')
ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--text-out', default=None)
a = ap.parse_args()
BASE = f'http://127.0.0.1:{a.port}'
QS = [('en', 'Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, '
              'fast recovery). Be thorough.', 400),
      ('ja', 'Pythonのジェネレーターとイテレーターの違いを、具体的なコード例を交えて日本語で詳しく説明してください。', 600),
      ('code', 'Write a Python function that parses an ISO 8601 duration string like "P3DT4H5M6S" into total '
               'seconds, with tests. Output only code.', 400)]


def gen(content, max_tokens, minp):
    body = {'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'temperature': 0,
            'stream': True, 'stream_options': {'include_usage': True},
            'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False}}
    if minp is not None:
        body['strata_tune'] = {'spec_min_p': minp}
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    tf = tl = None; usage = {}; n = 0; parts = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            ev = json.loads(data)
            usage = ev.get('usage') or usage
            for c in ev.get('choices') or []:
                if (c.get('delta') or {}).get('content'):
                    now = time.time(); tf = tf or now; tl = now; n += 1
                    parts.append(c['delta']['content'])
    ct = usage.get('completion_tokens') or n
    return ct, ((ct - 1) / (tl - tf) if tf and tl and tl > tf else 0.0), ''.join(parts)


def save(label, minp, rep, text):
    if a.text_out:
        with open(a.text_out, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'label': label, 'minp': minp, 'rep': rep, 'text': text}, ensure_ascii=False) + '\n')


for k, q, mt in QS:   # warm-up: the window graphs are captured on first use
    save(k, None, 0, gen(q, mt, None)[2])
for p in [float(x) for x in a.minps.split(',') if x]:
    res = {k: [] for k, _, _ in QS}
    for r in range(1, a.reps + 1):
        for k, q, mt in QS:
            ct, tps, text = gen(q, mt, p)
            save(k, p, r, text)
            res[k].append(tps)
            print(f'dec minp={p} rep={r} {k}: gen {ct} tok {tps:.1f} t/s', flush=True)
    med = {k: statistics.median(v) for k, v in res.items()}
    print(f"dec-mean minp={p}: en {med['en']:.1f} ja {med['ja']:.1f} code {med['code']:.1f} "
          f"mean {statistics.mean(med.values()):.1f} min {min(med.values()):.1f}", flush=True)
