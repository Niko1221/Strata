#!/usr/bin/env python3
"""Two concurrent requests that share a long system prompt: their decode routing overlap is what a batched MoE would save
on a workload of one agent template (run with DS4_TRACE on and analyse the file with route_overlap.py)."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

KEY = open("/data/works/Strata/api-key.txt").read().strip()
MODEL = "deepseek-v4.1-flash"
URL = "http://127.0.0.1:8099/v1/chat/completions"
SYSTEM = ("You are a careful assistant that answers briefly and precisely. "
          "You know Python, CUDA and distributed systems. ") * 40
Q = ["What is the capital of France? Answer in one short sentence.",
     "Write a haiku about rain."]


def ask(i, out):
    body = {"model": MODEL, "stream": True, "temperature": 0, "max_tokens": 40,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": Q[i]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    n = 0
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
    out[i] = {"tokens": n, "wall": round(time.time() - t0, 2)}


def main() -> int:
    out = {}
    ts = [threading.Thread(target=ask, args=(i, out)) for i in range(2)]
    for t in ts:
        t.start()
        time.sleep(0.15)
    [t.join() for t in ts]
    print("shared-prefix pair:", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
