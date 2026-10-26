#!/usr/bin/env python3
"""Soak the batched server: rounds of concurrent requests, each streamed to its end, all answered.

    python3 batch_soak.py [rounds] [clients]      # against the test server on 8099

Checks that every request finishes with a non-empty answer and that the server stays healthy; prints one line a round and
a summary, exit 0 when nothing failed.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

KEY = open("/data/works/Strata/api-key.txt").read().strip()
MODEL = "deepseek-v4.1-flash"
URL = "http://127.0.0.1:8099/v1/chat/completions"
PROMPTS = [
    "Name three uses of a tensor core, one line each.",
    "Explain in two sentences why batching helps throughput.",
    "What is the capital of France? Answer in one short sentence.",
    "List four colours and nothing else.",
    "In one sentence, what does a KV cache hold?",
    "Write a haiku about rain.",
]
fails = []


def one(i, prompt, out):
    body = {"model": MODEL, "stream": True, "temperature": 0, "max_tokens": 40 + 20 * (i % 3),
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    text, events = [], 0
    try:
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
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    events += 1
                    text.append(piece)
        out[i] = {"ok": bool(text), "tokens": events, "wall": round(time.time() - t0, 2), "text": "".join(text)[:40]}
    except Exception as e:              # noqa: BLE001 - reported in the summary
        out[i] = {"ok": False, "error": repr(e)}


def health() -> str:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=5) as r:
            return str(r.status)
    except Exception as e:              # noqa: BLE001
        return repr(e)


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    clients = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t_all = time.time()
    for rnd in range(rounds):
        out = {}
        ts = [threading.Thread(target=one, args=(k, PROMPTS[(rnd + k) % len(PROMPTS)], out)) for k in range(clients)]
        t0 = time.time()
        [t.start() for t in ts]
        [t.join() for t in ts]
        bad = [k for k in out if not out[k].get("ok")]
        fails.extend("%d/%d: %s" % (rnd, k, out[k].get("error", "empty answer")) for k in bad)
        print("round %d: %d clients, %.2f s, tokens %s, failed %d" % (
            rnd, clients, time.time() - t0, [out[k].get("tokens") for k in sorted(out)], len(bad)), flush=True)
    print("health after: %s | total %.1f s | requests %d | failed %d" % (
        health(), time.time() - t_all, rounds * clients, len(fails)), flush=True)
    if fails:
        print("FAILURES: " + " ; ".join(fails[:6]))
        return 1
    print("batch_soak: all requests answered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
