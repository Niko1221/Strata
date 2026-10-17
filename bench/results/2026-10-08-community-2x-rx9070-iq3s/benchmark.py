#!/usr/bin/env python3
"""Community benchmark for a running Strata server (OpenAI-compatible API).

Nonce-cold prefill sweep + warm prefix-reuse pair + streaming TTFT, with a memory
telemetry sampler. Writes results.json, summary.json, ttft.json, telemetry.jsonl.

Usage:
    STRATA_API_KEY=... python3 benchmark.py --url http://127.0.0.1:11634 --out .
      (or put the key in ./api-key.txt)

Fields in results.json (one object per request):
    target        requested filler size (tokens, approximate)
    actual        server-reported prompt_tokens
    reused        timings.cache_n (prefix-cache reused)
    read          timings.prompt_n (freshly read)
    prefill_tps   timings.prompt_per_second
    decode_tps    timings.predicted_per_second
    generated     timings.predicted_n
    draft         timings.draft_n / draft_n_accepted
    wall_s        client wall time
    finish        finish_reason
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

BLOCK = (
    "def process_records(records, cache=None):\n"
    "    # normalize and deduplicate the incoming batch\n"
    "    seen = set()\n"
    "    out = []\n"
    "    for rec in records:\n"
    "        key = (rec.get('id'), rec.get('version'))\n"
    "        if key in seen:\n"
    "            continue\n"
    "        seen.add(key)\n"
    "        out.append(normalize(rec))\n"
    "    return out\n"
)

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:11634")
ap.add_argument("--out", default=".")
ap.add_argument("--max-tokens", type=int, default=260)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--sizes", default="4096,32768,128000")
ap.add_argument("--big", type=int, default=244000, help="single extra run near the context limit (0 to skip)")
args = ap.parse_args()

URL = args.url.rstrip("/") + "/v1/chat/completions"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("STRATA_API_KEY", "")
if not KEY and (OUT / "api-key.txt").exists():
    KEY = (OUT / "api-key.txt").read_text().strip()
if not KEY and Path("api-key.txt").exists():
    KEY = Path("api-key.txt").read_text().strip()


def body_for(target: int, max_tokens: int) -> dict:
    nonce = f"[nonce-{int.from_bytes(os.urandom(8), 'big'):x}]\n"
    text = ("Review the following code and explain in detail, in about 250 words, what it does, "
            "how it works, and any edge cases. Do not stop early.\n\n" + nonce)
    while len(text) / 4 < target:
        text += BLOCK
    return {"model": "strata", "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none"}


def post(body: dict, stream: bool = False, timeout: float = 2400):
    if stream:
        body = dict(body, stream=True)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    if not stream:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return d, time.time() - t0
    # streaming: time to first non-empty content delta (TTFT)
    t0 = time.time()
    ttft, out = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except ValueError:
                continue
            delta = (j.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece and ttft is None:
                ttft = time.time() - t0
            out.append(piece)
    return {"content": "".join(out), "ttft": ttft}, time.time() - t0


def engine_pid():
    try:
        return int(subprocess.check_output(["pgrep", "-f", "[e]ngine/strata"]).split()[0])
    except Exception:
        return None


def mem_sample():
    pid = engine_pid()
    s = {"t": time.time()}
    if pid:
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith(("Rss:", "Locked:")):
                    s["engine_" + line.split(":")[0].lower()] = int(line.split()[1])
        except OSError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                s["mem_available_kb"] = int(line.split()[1])
    except OSError:
        pass
    for card in ("card0", "card1"):
        p = Path(f"/sys/class/drm/{card}/device/mem_info_vram_used")
        if p.exists():
            try:
                s[f"vram_{card}_b"] = int(p.read_text())
            except OSError:
                pass
    return s


stop = threading.Event()
telemetry = OUT / "telemetry.jsonl"


def sampler():
    with telemetry.open("w") as f:
        while not stop.is_set():
            f.write(json.dumps(mem_sample()) + "\n")
            f.flush()
            time.sleep(1)


th = threading.Thread(target=sampler, daemon=True)
th.start()

runs = []
targets = [int(x) for x in args.sizes.split(",") if x]

print("== cold prefill sweep (unique nonce, no reuse) ==", flush=True)
for target in targets:
    for i in range(args.reps):
        d, wall = post(body_for(target, args.max_tokens))
        u, t = d.get("usage", {}), d.get("timings", {})
        run = {"kind": "cold", "target": target, "run": i + 1, "actual": u.get("prompt_tokens"),
               "reused": t.get("cache_n"), "read": t.get("prompt_n"),
               "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
               "generated": t.get("predicted_n"), "draft_n": t.get("draft_n"),
               "draft_accepted": t.get("draft_n_accepted"), "wall_s": round(wall, 2),
               "finish": d["choices"][0].get("finish_reason")}
        runs.append(run)
        (OUT / "results.json").write_text(json.dumps(runs, indent=1))
        print(f"  {target:>7} #{i+1}: prompt={run['actual']} reused={run['reused']} "
              f"read={run['read']} prefill={run['prefill_tps']:.1f} decode={run['decode_tps']:.1f} "
              f"gen={run['generated']} wall={wall:.1f}s", flush=True)

if args.big:
    print("== single near-context-limit run ==", flush=True)
    d, wall = post(body_for(args.big, args.max_tokens))
    u, t = d.get("usage", {}), d.get("timings", {})
    run = {"kind": "cold-big", "target": args.big, "run": 1, "actual": u.get("prompt_tokens"),
           "reused": t.get("cache_n"), "read": t.get("prompt_n"),
           "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
           "generated": t.get("predicted_n"), "draft_n": t.get("draft_n"),
           "draft_accepted": t.get("draft_n_accepted"), "wall_s": round(wall, 2),
           "finish": d["choices"][0].get("finish_reason")}
    runs.append(run)
    print(f"  {args.big:>7}: prompt={run['actual']} prefill={run['prefill_tps']:.1f} "
          f"decode={run['decode_tps']:.1f} wall={wall:.1f}s", flush=True)

print("== warm prefix reuse (same ~8K prompt twice) ==", flush=True)
warm_body = body_for(8000, args.max_tokens)          # fixed, no nonce change between the two
for i in range(2):
    d, wall = post(warm_body)
    u, t = d.get("usage", {}), d.get("timings", {})
    run = {"kind": "warm", "target": 8000, "run": i + 1, "actual": u.get("prompt_tokens"),
           "reused": t.get("cache_n"), "read": t.get("prompt_n"),
           "prefill_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
           "generated": t.get("predicted_n"), "wall_s": round(wall, 2),
           "finish": d["choices"][0].get("finish_reason")}
    runs.append(run)
    print(f"  warm #{i+1}: reused={run['reused']} read={run['read']} wall={wall:.2f}s", flush=True)

print("== streaming TTFT (short prompt) ==", flush=True)
ttft_runs = []
for i in range(3):
    d, wall = post(body_for(200, 128), stream=True)
    got = d["ttft"]
    ttft_runs.append({"run": i + 1, "ttft_s": got, "wall_s": round(wall, 2),
                      "chars": len(d["content"])})
    print(f"  ttft #{i+1}: {got if got is None else round(got, 3)}s  wall={wall:.2f}s", flush=True)

stop.set()
th.join(timeout=3)

(OUT / "results.json").write_text(json.dumps(runs, indent=1))
(OUT / "ttft.json").write_text(json.dumps(ttft_runs, indent=1))


def agg(rs):
    xs = [r["prefill_tps"] for r in rs if r.get("prefill_tps") is not None]
    ys = [r["decode_tps"] for r in rs if r.get("decode_tps") is not None]
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None
    return {"n": len(rs), "actual": rs[0].get("actual"), "reused": rs[0].get("reused"),
            "generated": rs[0].get("generated"),
            "prefill_median": med(xs), "prefill_range": [min(xs), max(xs)] if xs else None,
            "decode_median": med(ys), "decode_range": [min(ys), max(ys)] if ys else None}


summary = {"summary": [agg([r for r in runs if r["kind"] == "cold" and r["target"] == t]) for t in targets],
           "big": agg([r for r in runs if r["kind"] == "cold-big"]) if args.big else None,
           "warm": [r for r in runs if r["kind"] == "warm"],
           "ttft": ttft_runs}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("\nwrote results.json, summary.json, ttft.json, telemetry.jsonl ->", OUT, flush=True)
