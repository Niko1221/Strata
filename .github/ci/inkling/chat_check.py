#!/usr/bin/env python3
"""chat0221.py PORT : three non-streaming chats on a local Strata server, the ANSWER TEXT is checked (not only the status code).

1. greedy (temperature 0): the capital of Australia must contain "Canberra";
2. sampled (temperature 0.8, top_p 0.95): 17 * 23 must contain "391" (the sampler path is the one that uses double precision on the
   Alchemist cards: an engine without the FP64 emulation environment dies on the first request that samples);
3. a longer sampled reply (about 150 words) that must be non-empty and stop on its own or at the limit.
Prints the replies; exit 0 only when 1 and 2 hold. Loopback only; no key is read or printed."""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
base = f"http://127.0.0.1:{port}"
with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]


def chat(prompt, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, **extra}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    ch = data["choices"][0]
    return (ch["message"].get("content") or ""), ch.get("finish_reason"), time.time() - t, data.get("usage", {})


ok = True
text, fin, dt, use = chat("What is the capital of Australia? Answer in one short sentence.", temperature=0, max_tokens=300, reasoning_effort="none")
good = "canberra" in text.lower()
print(f"greedy   {dt:5.1f}s finish={fin} usage={use} reply={text!r}  {'OK' if good else 'WRONG'}")
ok &= good
text, fin, dt, use = chat("What is 17 times 23? Answer with just the number.", temperature=0.8, top_p=0.95, max_tokens=300, reasoning_effort="none")
good = "391" in text
print(f"sampled  {dt:5.1f}s finish={fin} usage={use} reply={text!r}  {'OK' if good else 'WRONG'}")
ok &= good
text, fin, dt, use = chat("Write about 150 words on why the sea is salty.", temperature=0.8, top_p=0.95, max_tokens=400, reasoning_effort="none")
good = len(text.split()) > 40
print(f"long     {dt:5.1f}s finish={fin} usage={use} words={len(text.split())} head={text[:100]!r}  {'OK' if good else 'SHORT'}")
print("ANSWER-TEXT-OK" if ok else "ANSWER-TEXT-WRONG")
sys.exit(0 if ok else 1)
