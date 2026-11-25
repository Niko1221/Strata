#!/usr/bin/env python3
"""c32c_chat.py PORT [CLIENTS=32] [MAXTOK=24] [OFFSET=0] : like c32_chat.py, but client i asks about index i+OFFSET, so one client of a failed 32-client run
can be repeated alone ("What is A plus B?", A = 10 + k, B = 3 + k%7 (easy sums: the three-digit ones gave one deterministic model slip, 198+354 -> 542, also alone), k = i + OFFSET). Prints each wrong client, wall time, tokens per second.
Exit 0 only when every client answered right. Loopback only. Stdlib only."""
import json
import sys
import threading
import time
import urllib.request

port = int(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 24
off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
base = f"http://127.0.0.1:{port}"
with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]
res = [None] * n


def one(i):
    k = i + off
    a, b = 10 + k, 3 + (k % 7)
    body = {"model": model, "messages": [{"role": "user", "content": f"What is {a} plus {b}? Answer with just the number."}], "stream": False,
            "temperature": 0, "max_tokens": maxtok, "reasoning_effort": "none"}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        text = d["choices"][0]["message"].get("content") or ""
        res[i] = (str(a + b) in text, time.time() - t, d.get("usage", {}).get("completion_tokens", 0), text.strip()[:40], a + b)
    except Exception as e:
        res[i] = (False, time.time() - t, 0, repr(e)[:80], a + b)


t0 = time.time()
ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
[t.start() for t in ts]
[t.join() for t in ts]
wall = time.time() - t0
good = sum(1 for r in res if r and r[0])
toks = sum(r[2] for r in res if r)
for i, r in enumerate(res):
    if not r[0]:
        print(f"client {i + off} WRONG/FAILED after {r[1]:.1f}s: got {r[3]!r}, expected {r[4]}")
lat = sorted(r[1] for r in res)
print(f"c{n}@{off}: {good}/{n} right, wall {wall:.1f}s, {toks} completion tokens = {toks / wall:.1f} tok/s together, latency median {lat[len(lat) // 2]:.1f}s max {lat[-1]:.1f}s")
sys.exit(0 if good == n else 1)
