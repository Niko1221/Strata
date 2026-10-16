#!/usr/bin/env python3
# longbench.py LABEL [RUNS32] [RUNS128] [RUNS250] - long-context prompts straight to the engine (BENCH_PORT).
# fresh prompts (random nonce), temperature 0, 60 tokens out; speeds from the engine's own log lines.
import json, os, random, re, statistics, sys, time, urllib.request
LABEL = sys.argv[1]
RUNS = {"32k": int(sys.argv[2]) if len(sys.argv) > 2 else 2,
        "128k": int(sys.argv[3]) if len(sys.argv) > 3 else 2,
        "250k": int(sys.argv[4]) if len(sys.argv) > 4 else 1}
PORT, LOG = os.environ.get("BENCH_PORT", "18099"), os.environ.get("BENCH_LOG", "~/Strata/ab-dense-f32/logs/lg.log")
WORDS = ("the engine reads a prompt and writes an answer while the cache keeps the experts that were used most often "
         "in the last rounds so a second request with a different first line has to be read again from the start").split()
PAT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
res = {}
for case, words in (("32k", 32400), ("128k", 130300), ("250k", 248200)):
    for i in range(RUNS[case]):
        nonce = "%016x" % random.getrandbits(64)
        prompt = f"Run {nonce}. Continue this text with a long plain description of a harbour town, no lists.\n\n" + \
                 " ".join(random.Random(nonce).choice(WORDS) for _ in range(words))
        before = sum(1 for _ in open(LOG, errors="replace"))
        body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60, "temperature": 0, "reasoning_effort": "none", "stream": False}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t = time.time()
        try:
            urllib.request.urlopen(req, timeout=5400).read()
        except Exception as e:
            print(f"{LABEL} {case} run {i}: FAIL {e}", flush=True); continue
        wall = time.time() - t; time.sleep(1.5)
        lines = open(LOG, errors="replace").read().split("\n")[before:]
        m = [x for x in (PAT.search(l) for l in lines) if x]
        if not m: print(f"{LABEL} {case} run {i}: no log line wall {wall:.0f}s", flush=True); continue
        g = m[-1].groups()
        res.setdefault(case, []).append((float(g[4]), float(g[7]), int(g[0]), wall))
        print(f"{LABEL} {case} run {i}: prompt {g[0]} tok ({g[2]} read) at {g[4]} tok/s | {g[5]} generated at {g[7]} tok/s | wall {wall:.0f}s", flush=True)
for case, v in res.items():
    print(f"== {LABEL} {case}: prompt median {statistics.median(x[0] for x in v):.0f} tok/s | decode median {statistics.median(x[1] for x in v):.1f} tok/s | n={len(v)}")
