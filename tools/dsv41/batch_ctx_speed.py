#!/usr/bin/env python3
"""Decode speed against the context length, on the batched server: how much of a token's time is attention (which grows
with the context) and how much is the rest (the MoE, which does not) - the number that says what M2's varlen attention
could win at chat-sized contexts.

    python3 batch_ctx_speed.py            # against the test server on 8099

For each prompt size (tokens) it sends one greedy request, streams it, and reports the prompt's read time, the decode's
tokens/s from the first token to the last, and the whole wall time.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

KEY = open("/data/works/Strata/api-key.txt").read().strip()
MODEL = "deepseek-v4.1-flash"
URL = "http://127.0.0.1:8099/v1/chat/completions"
FILLER = ("The quick brown fox jumps over the lazy dog. A tensor core multiplies small matrices in one instruction. "
          "Attention reads a window of keys and values for every token it decodes. ")
SIZES = [200, 4000, 12000, 24000]


def make_prompt(tokens):
    # (~1.4 tokens a word for this text: enough to land near the size)
    words = max(8, int(tokens / 1.4))
    text = (FILLER * (words // 30 + 1))[: words * 6]
    return text + "\n\nIn one short sentence, what is the capital of France?"


def run(tokens):
    body = {"model": MODEL, "stream": True, "temperature": 0, "max_tokens": 48,
            "messages": [{"role": "user", "content": make_prompt(tokens)}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    first, last, n = None, None, 0
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if prompt_tokens is None and d.get("usage"):
                prompt_tokens = d["usage"].get("prompt_tokens")
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if first is None:
                    first = now
                last = now
                n += 1
    wall = time.time() - t0
    decode_s = (last - first) if (first and last and last > first) else 0.0
    tps = (n - 1) / decode_s if decode_s > 0 else 0.0
    print("prompt %6d tokens (asked %5d): first token %.2f s, %d tokens, decode %.2f tok/s, wall %.2f s" % (
        prompt_tokens or -1, tokens, (first - t0) if first else 0.0, n, tps, wall), flush=True)
    return {"prompt_tokens": prompt_tokens, "first_s": (first - t0) if first else 0.0, "tokens": n, "tok_s": tps,
            "wall": wall}


def main() -> int:
    out = []
    for t in SIZES:
        try:
            out.append({"asked": t, **run(t)})
        except Exception as e:              # noqa: BLE001
            print("prompt %d failed: %r" % (t, e), flush=True)
            out.append({"asked": t, "error": repr(e)})
    json.dump(out, open("/data/works/batch_ctx_speed.json", "w"))
    ok = [r for r in out if "tok_s" in r and r.get("tokens", 0) > 4]
    if len(ok) >= 2:
        print("\nslowest / fastest decode: %.1f%% - the attention's share of a token's time is what grows with the "
              "context" % (100.0 * (1.0 - ok[0]["tok_s"] / ok[-1]["tok_s"]) if ok[-1]["tok_s"] > 0 else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
