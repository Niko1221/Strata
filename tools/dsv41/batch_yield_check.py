#!/usr/bin/env python3
"""A long prompt gives way to a short one (#656, BYIELD): how long the short client waits for its first token.

    python3 batch_yield_check.py            # against the test server on 8099

Client A sends a ~5k-token prompt (its read takes seconds), client B a short one 1.2 s later.  Without BYIELD the engine
reads A whole before B's request is looked at; with it the read stops at a chunk boundary, B is served, and A's read
goes on.  Prints one line: a_first, a_wall, b_first, b_wall.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

KEY = open("/data/works/Strata/api-key.txt").read().strip()
MODEL = "deepseek-v4.1-flash"
LONG = "The quick brown fox jumps over the lazy dog. " * 400
SHORT = "Say hello in one word."
out = {}


def ask(name, prompt, max_tokens=24):
    body = {"model": MODEL, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    first, first_any, text = None, None, []
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
            think = delta.get("reasoning_content") or ""
            piece = delta.get("content") or ""
            if think or piece:
                if first_any is None:
                    first_any = round(time.time() - t0, 2)
            if piece:
                if first is None:
                    first = round(time.time() - t0, 2)
                text.append(piece)
    out[name] = {"first": first, "first_any": first_any, "wall": round(time.time() - t0, 2),
                 "text": "".join(text)[:60]}


def main():
    a = threading.Thread(target=ask, args=("a", LONG))
    a.start()
    time.sleep(1.2)
    b = threading.Thread(target=ask, args=("b", SHORT))
    b.start()
    a.join()
    b.join()
    print("yield check: a first-token %s / wall %s | b first-token %s / wall %s | b text %r" % (
        out["a"]["first_any"], out["a"]["wall"], out["b"]["first_any"], out["b"]["wall"], out["b"]["text"]),
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())