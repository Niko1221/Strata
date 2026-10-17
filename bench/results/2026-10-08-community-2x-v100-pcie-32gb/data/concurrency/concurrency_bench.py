#!/usr/bin/env python3
"""Concurrent-request benchmark for a running Strata server.

MODELED ON docs/BATCHING.md's measurement: C requests sent at once, an ~800-word
essay prompt each, 256 tokens per answer, greedy, thinking off; median of 3 rounds.
Usage: concurrency_bench.py <url> <clients> [rounds]
"""
import json
import sys
import time
import threading
import statistics
import requests

URL = sys.argv[1]
CLIENTS = int(sys.argv[2])
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

TOPICS = [
    "the history of lighthouses", "how vaccines work", "the water cycle", "bird migration",
    "the printing press", "photosynthesis", "the silk road", "volcanoes", "tides", "bees",
    "the industrial revolution", "antarctica", "chess", "coffee", "the moon landing",
    "coral reefs", "railways", "paper", "glass", "the deep sea", "wind", "rice", "maps",
    "clocks", "bridges",
]

PROMPT = (
    "Write an essay of about 800 words on {topic}. Cover its history, how it works, and "
    "why it matters today. Structure it with an introduction, three body sections and a "
    "conclusion. Topic {nonce}: {topic}."
)

def one_request(idx: int, out: dict):
    topic = TOPICS[(idx + 7) % len(TOPICS)]
    prompt = PROMPT.format(topic=topic, nonce=idx)
    body = {
        "model": "strata",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": True,
    }
    t0 = time.time()
    ttft = None
    chunks = 0
    try:
        with requests.post(URL + "/v1/chat/completions", json=body, stream=True,
                           timeout=1800) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    try:
                        d = json.loads(payload)
                    except ValueError:
                        continue
                    delta = (d.get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content"):
                        chunks += 1
                        if ttft is None:
                            ttft = time.time() - t0
        out[idx] = {"ttft": ttft, "elapsed": time.time() - t0, "tokens": chunks}
    except Exception as e:  # noqa: BLE001
        out[idx] = {"error": str(e)[:120]}

print(f"clients={CLIENTS} rounds={ROUNDS} url={URL}")
rounds = []
for rnd in range(ROUNDS):
    out = {}
    threads = [threading.Thread(target=one_request, args=(rnd * CLIENTS + i, out))
               for i in range(CLIENTS)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    errs = [v for v in out.values() if "error" in v]
    if errs:
        print("  errors:", errs[:2])
    good = [v for v in out.values() if "error" not in v]
    if not good:
        continue
    per = [v["tokens"] / v["elapsed"] for v in good]
    total = sum(v["tokens"] for v in good) / wall
    rounds.append((statistics.median(per), total,
                   statistics.median([v["ttft"] for v in good]),
                   statistics.median([v["tokens"] for v in good])))
    print(f"  round {rnd+1}: per-request {statistics.median(per):5.1f} tok/s  "
          f"total {total:5.1f} tok/s  TTFT-med {statistics.median([v['ttft'] for v in good]):5.2f}s  "
          f"tokens-med {statistics.median([v['tokens'] for v in good]):.0f}")

if rounds:
    m = lambda i: statistics.median([r[i] for r in rounds])  # noqa: E731
    print(f"MEDIAN over {len(rounds)} rounds: per-request {m(0):.1f} tok/s  "
          f"total {m(1):.1f} tok/s  TTFT {m(2):.2f}s  tokens {m(3):.0f}")
