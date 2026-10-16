#!/usr/bin/env python3
# burst.py N - N parallel long requests (7K prompt, 60 tokens out) fired at once; aggregate tok/s over wall.
import json, os, random, sys, threading, time, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PORT = os.environ.get("BENCH_PORT", "18099")
WORDS = ("the engine reads a prompt and writes an answer while the cache keeps the experts that were used most often "
         "in the last rounds so a second request with a different first line has to be read again from the start").split()
results = []
def one(i):
    nonce = "%016x" % random.getrandbits(64)
    prompt = f"Run {nonce}. Continue this text with a long plain description of a harbour town, no lists.\n\n" + " ".join(random.Random(nonce).choice(WORDS) for _ in range(7000))
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60, "temperature": 0, "reasoning_effort": "none", "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); urllib.request.urlopen(req, timeout=1800).read(); results.append((i, time.time() - t))
ts = [threading.Thread(target=one, args=(i,)) for i in range(N)]
t0 = time.time()
for t in ts: t.start()
for t in ts: t.join()
wall = time.time() - t0
for i, w in sorted(results): print(f"  burst[{i}] wall {w:.1f}s")
gen = N * 60
print(f"== BURST x{N}: wall {wall:.1f}s | aggregate decode {gen/wall:.1f} tok/s (each request also read 7K prompt)")
