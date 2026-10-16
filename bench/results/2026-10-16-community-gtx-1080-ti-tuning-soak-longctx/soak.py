#!/usr/bin/env python3
# soak.py MINUTES - back-to-back requests (short/long alternating) for MINUTES, drift report every 10 min.
import json, os, random, re, statistics, sys, time, urllib.request
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 60
PORT, LOG = "18099", "~/Strata/ab-dense-f32/logs/soak.log"
WORDS = ("the engine reads a prompt and writes an answer while the cache keeps the experts that were used most often "
         "in the last rounds so a second request with a different first line has to be read again from the start").split()
PAT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
rows = []
t0 = time.time(); n = 0
while (time.time() - t0) < MIN * 60:
    long = n % 2 == 1
    toks, mt = (7000, 60) if long else (520, 200)
    nonce = "%016x" % random.getrandbits(64)
    prompt = f"Run {nonce}. Continue this text with a long plain description of a harbour town, no lists.\n\n" + " ".join(random.Random(nonce).choice(WORDS) for _ in range(toks))
    before = sum(1 for _ in open(LOG, errors="replace"))
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
            "temperature": 0, "reasoning_effort": "none", "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        tw = time.time()
        urllib.request.urlopen(req, timeout=1800).read()
        wall = time.time() - tw
    except Exception as e:
        print(f"[{(time.time()-t0)/60:5.1f}m] n={n} {('long' if long else 'short')} FAIL {e}", flush=True); n += 1; continue
    time.sleep(1.0)
    lines = open(LOG, errors="replace").read().split("\n")[before:]
    m = [x for x in (PAT.search(l) for l in lines) if x]
    if not m: print(f"[{(time.time()-t0)/60:5.1f}m] n={n} no log line", flush=True); n += 1; continue
    g = m[-1].groups()
    rows.append({"n": n, "case": "long" if long else "short", "prompt_tps": float(g[4]), "decode_tps": float(g[7]),
                 "read": int(g[2]), "gen": int(g[5]), "wall": wall, "t_min": (time.time() - t0) / 60})
    print(f"[{(time.time()-t0)/60:5.1f}m] n={n} {'long' if long else 'short'}: prompt {g[4]} tok/s | decode {g[7]} tok/s | wall {wall:.1f}s", flush=True)
    n += 1
print("\n== drift (first third vs last third, per case)")
for case in ("short", "long"):
    v = [r for r in rows if r["case"] == case]
    if len(v) >= 6:
        k = len(v) // 3
        f, l = v[:k], v[-k:]
        fm, lm = statistics.median(x["prompt_tps"] for x in f), statistics.median(x["prompt_tps"] for x in l)
        fd, ld = statistics.median(x["decode_tps"] for x in f), statistics.median(x["decode_tps"] for x in l)
        print(f"== {case}: prompt {fm:.1f} -> {lm:.1f} tok/s ({(lm/fm-1)*100:+.1f}%), decode {fd:.1f} -> {ld:.1f} tok/s ({(ld/fd-1)*100:+.1f}%), n={len(v)}")
json.dump(rows, open("~/Strata/ab-dense-f32/logs/soak-results.json", "w"), indent=1)
print("total requests:", len(rows), "failures:", n - len(rows))
