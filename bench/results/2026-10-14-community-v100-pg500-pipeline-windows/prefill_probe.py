#!/usr/bin/env python3
"""Prompt (prefill) speed and long-context recall against a running Strata server (OpenAI-compatible API).

  python3 prefill_probe.py --src-dir <a llama.cpp checkout> [--port 18085] [--sizes 32000,64000,128000,32001,64001] [--decode 512]

Each size n sends about 3.0 * n characters of C/C++ source (src/*.cpp, ggml/src/*.c*, ggml/src/ggml-cuda/*.cu of the
checkout, in name order) with a passphrase note in the middle and a question at the end; a fresh first line per request keeps
the server from reusing an earlier prompt. Our text came from a llama.cpp checkout at 7fe450e with some files of our own
build changed, so a plain checkout gives slightly different text and token counts (prompt.csv has the counts we got).
Greedy, no thinking, streamed. prompt tok/s = prompt_tokens / time to the first content token (includes HTTP and the
first generated token). PASS = the answer contains the passphrase.
Output, one line per request: "pf-<n>: prompt <tok> tok <t/s> t/s | wall <s>s | PASS|FAIL", then
"decode: gen <tok> tok <t/s> t/s" for one --decode-token answer to a short question.
"""
import argparse
import glob
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--src-dir', required=True)
ap.add_argument('--port', type=int, default=18085)
ap.add_argument('--sizes', default='32000,64000,128000,32001,64001')
ap.add_argument('--salt', default=str(time.time_ns()))
ap.add_argument('--decode', type=int, default=512, help='tokens for the short answer at the end (0 skips it)')
a = ap.parse_args()
BASE = f'http://127.0.0.1:{a.port}'


def stream(content, max_tokens):
    body = {'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'temperature': 0,
            'stream': True, 'stream_options': {'include_usage': True},
            'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time(); t_first = t_last = None; parts = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
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
                piece = (c.get('delta') or {}).get('content') or ''
                if piece:
                    now = time.time(); t_first = t_first or now; t_last = now; parts.append(piece)
    return ''.join(parts), usage, t0, t_first, t_last


d = a.src_dir
src = ''.join(open(f, encoding='utf-8', errors='replace').read() for f in sorted(
    glob.glob(os.path.join(d, 'src/*.cpp')) + glob.glob(os.path.join(d, 'ggml/src/*.c*'))
    + glob.glob(os.path.join(d, 'ggml/src/ggml-cuda/*.cu'))))
NEEDLE = 'IMPORTANT NOTE FOR THE READER: the maintenance passphrase is "BLUE-HERON-4721".'
Q = ('\n\nQuestion: somewhere in the code above there is a note with a maintenance passphrase. '
     'What is the passphrase? Then name one C++ class that appears in the code. Answer briefly.')
for n in [int(x) for x in a.sizes.split(',') if x]:
    body = src[:int(n * 3.0)]   # the server has no /tokenize; about 3.0 characters a token for this source
    if len(body) < int(n * 3.0):
        print(f'pf-{n}: source too short ({len(src)} chars)'); continue
    mid = len(body) // 2
    body = f'// probe {a.salt} size {n}\n' + body[:mid] + '\n// ' + NEEDLE + '\n' + body[mid:]
    try:
        text, usage, t0, tf, tl = stream(body + Q, 40)
    except Exception as e:
        print(f'pf-{n}: ERROR {e}'); continue
    pt = usage.get('prompt_tokens') or 0
    ttft = (tf or time.time()) - t0
    ok = 'PASS' if 'BLUE-HERON-4721' in text else 'FAIL'
    print(f'pf-{n}: prompt {pt} tok {pt / ttft if ttft > 0 else 0:.1f} t/s | wall {time.time() - t0:.1f}s | {ok}', flush=True)
if a.decode:
    text, usage, t0, tf, tl = stream(f'({a.salt}) Write a detailed technical explanation of how a B-tree handles insertions '
                                     'and node splits, with a worked example.', a.decode)
    ct = usage.get('completion_tokens') or 0
    dec = (tl - tf) if tf and tl and tl > tf else 0
    print(f'decode: gen {ct} tok {(ct - 1) / dec if dec > 0 else 0:.1f} t/s | ttft {(tf or t0) - t0:.2f}s', flush=True)
