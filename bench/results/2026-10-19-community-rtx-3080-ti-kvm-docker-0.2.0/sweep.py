#!/usr/bin/env python3
"""sweep.py - prompt and decode speed of a running Strata server at several prompt sizes.

For each size a prompt is cut from text files (by default this repository's docs/*.md and README*.md, in sorted path
order), sized with /v1/messages/count_tokens, and sent through /v1/chat/completions with thinking off, greedy
sampling, a fixed seed and a fixed output cap. Every prompt starts with a fresh nonce, so nothing is reused from the
conversation cache: each run reads the whole prompt. Rates are the engine's own (the response's `timings` block);
time to first token and total time are measured at the client.

    STRATA_API_KEY=... python sweep.py --label stock --sizes 4k,32k,128k --runs 3 --out stock.jsonl docs/*.md README*.md

One JSON line per run, then one `summary` line per size with the median and range. Fields: prompt_tokens and
reused_tokens (tokens), generated_tokens (tokens), prompt_tps and decode_tps (tokens/s, engine clock), ttft_s and
total_s (seconds, client clock), finish (the API's finish_reason).
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("corpus", nargs="+", help="text files the prompts are cut from")
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--label", required=True)
ap.add_argument("--sizes", default="4k,32k,128k")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()
KEY = os.environ.get("STRATA_API_KEY", "")
TEXT = "\n\n".join(pathlib.Path(p).read_text(errors="replace") for p in sorted(a.corpus))
SHORT = "Explain in three sentences why a UPS should trigger a staggered shutdown."
ASK = "In about 200 words, summarise the documents above."


def post(path, body):
    head = {"Content-Type": "application/json"}
    if KEY:
        head["Authorization"] = "Bearer " + KEY
    return urllib.request.urlopen(urllib.request.Request(a.url + path, json.dumps(body).encode(), head), timeout=3600)


def tokens(text):
    with post("/v1/messages/count_tokens", {"model": "strata", "messages": [{"role": "user", "content": text}]}) as r:
        return json.load(r)["input_tokens"]


def prompt_for(target):
    if target == 0:
        return f"[run {uuid.uuid4()}] {SHORT}"
    need = int(target * len(TEXT[:200000]) / tokens(TEXT[:200000]))
    for _ in range(4):
        if need > len(TEXT):
            sys.exit(f"the corpus holds {len(TEXT)} characters, {need} are needed for {target} tokens")
        got = tokens(TEXT[:need])
        if abs(got - target) <= target * 0.01:
            break
        need = int(need * target / got)
    return f"[run {uuid.uuid4()}]\n{TEXT[:need]}\n\n{ASK}"


def run(prompt):
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": "none", "temperature": 0, "seed": 1,
            "max_tokens": a.max_tokens}
    out, t0 = {}, time.monotonic()
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            for k in ("usage", "timings", "error"):
                if d.get(k):
                    out[k] = d[k]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and "ttft_s" not in out:
                    out["ttft_s"] = round(time.monotonic() - t0, 2)
                if c.get("finish_reason"):
                    out["finish"] = c["finish_reason"]
    t, u = out.get("timings") or {}, out.get("usage") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "reused_tokens": t.get("cache_n"),
            "generated_tokens": t.get("predicted_n"), "prompt_tps": t.get("prompt_per_second"),
            "decode_tps": t.get("predicted_per_second"), "ttft_s": out.get("ttft_s"),
            "total_s": round(time.monotonic() - t0, 2), "finish": out.get("finish"), "error": out.get("error")}


def spread(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return None if not v else {"median": statistics.median(v), "min": min(v), "max": max(v)}


def size(s):
    s = s.strip().lower()
    return 0 if s in ("short", "0") else int(float(s[:-1]) * (1024 if s.endswith("k") else 1)) if s.endswith("k") else int(s)


with open(a.out, "a") as f:
    for s in a.sizes.split(","):
        rows = []
        for i in range(a.runs):
            row = {"label": a.label, "size": s.strip(), "run": i + 1, **run(prompt_for(size(s)))}
            rows.append(row)
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")
        summ = {"label": a.label, "size": s.strip(), "summary": True, "runs": len(rows),
                "prompt_tokens": rows[-1]["prompt_tokens"], "generated_tokens": spread(rows, "generated_tokens"),
                "prompt_tps": spread(rows, "prompt_tps"), "decode_tps": spread(rows, "decode_tps"),
                "ttft_s": spread(rows, "ttft_s")}
        print(json.dumps(summ), flush=True)
        f.write(json.dumps(summ) + "\n")
        f.flush()
