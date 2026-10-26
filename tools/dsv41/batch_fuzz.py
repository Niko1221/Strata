#!/usr/bin/env python3
"""Fuzz the batched server: concurrent requests that finish, requests that are abandoned mid-stream, one-token answers,
and a canary after every round so a slot that nobody released shows up at once.

    python3 batch_fuzz.py [rounds]      # against the test server on 8099

Exit 0 when every round kept the server answering and every request that was not abandoned got an answer.
"""
from __future__ import annotations

import json
import random
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
    "Say yes.",
    "Count from one to five.",
]
rng = random.Random(7)


def one(i, prompt, max_tokens, abandon_after, out):
    body = {"model": MODEL, "stream": True, "temperature": 0,
            "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    got, t0 = 0, time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
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
                    got += 1
                    if abandon_after and got >= abandon_after:
                        r.close()                 # the client goes away mid-answer
                        out[i] = {"state": "abandoned", "tokens": got, "wall": round(time.time() - t0, 2)}
                        return
        out[i] = {"state": "done", "tokens": got, "wall": round(time.time() - t0, 2)}
    except Exception as e:                      # noqa: BLE001 - reported by the caller
        out[i] = {"state": "error", "error": repr(e), "tokens": got, "wall": round(time.time() - t0, 2)}


def canary(timeout=120):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 8,
            "messages": [{"role": "user", "content": "Say hi."}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        return time.time() - t0, bool(d["choices"][0]["message"])
    except Exception as e:                      # noqa: BLE001
        return None, repr(e)


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    fails = []
    for rnd in range(rounds):
        n = rng.randint(2, 4)
        jobs = []
        out = {}
        ts = []
        for k in range(n):
            prompt = rng.choice(PROMPTS)
            max_tokens = rng.choice([1, 8, 24, 48, 80])
            abandon = rng.choice([0, 0, 0, 3, 6]) if max_tokens > 6 else 0
            jobs.append((prompt, max_tokens, abandon))
            t = threading.Thread(target=one, args=(k, prompt, max_tokens, abandon, out))
            ts.append(t)
        t0 = time.time()
        for t in ts:
            t.start()
            time.sleep(rng.uniform(0.0, 0.3))
        for t in ts:
            t.join()
        wall = round(time.time() - t0, 2)
        bad = [k for k in out if out[k]["state"] == "error" or (out[k]["state"] == "done" and out[k]["tokens"] == 0)]
        fails.extend("round %d client %d: %s" % (rnd, k, out[k]) for k in bad)
        ct, can = canary()
        ok_canary = ct is not None and ct < 120
        if not ok_canary:
            fails.append("round %d: canary failed (%s)" % (rnd, can))
        print("round %2d: %d clients, %.2f s, states %s, canary %s" % (
            rnd, n, wall, [out[k]["state"] for k in sorted(out)],
            ("%.2f s" % ct) if ok_canary else "FAILED"), flush=True)
        if not ok_canary:
            break
    if fails:
        print("FAILURES:")
        for f in fails[:10]:
            print("  " + f)
        return 1
    print("batch_fuzz: every round answered, canary green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
