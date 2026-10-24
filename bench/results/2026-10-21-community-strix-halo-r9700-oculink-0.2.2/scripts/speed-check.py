#!/usr/bin/env python3
"""speed-check.py URL KEY N [N ...] [--tokens T]  - tokens per second per stream with N requests decoding at once,
on text that can be checked: each request counts upward from its own start number and is cut at T tokens (default
500).  A stream counts as right when its numbers are consecutive.  Thinking off, the model card's sampling."""
import json
import re
import sys
import threading
import time
import urllib.request

a = [x for i, x in enumerate(sys.argv[1:], 1) if not x.startswith("--") and sys.argv[i - 1] != "--tokens"]
URL, KEY, NS = a[0].rstrip("/"), a[1], [int(x) for x in a[2:]]
T = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 500
OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def one(i, out):
    start = 1000 * (i + 1) + 1
    body = {"model": "qwen3.8-flash-next", "max_tokens": T, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"Count upward from {start} to {start + 900}, the numbers "
                                                     "separated by single spaces, nothing else."}],
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0, first, text, toks = time.time(), None, [], None
    try:
        for line in OPEN.open(req, timeout=900):
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                toks = d["usage"].get("completion_tokens")
            for ch in d.get("choices") or []:
                c = (ch.get("delta") or {}).get("content")
                if c:
                    first = first or time.time()
                    text.append(c)
        t1 = time.time()
        toks = toks or len(text)        # no usage in the stream: one delta is about one token
        nums = [int(x) for x in re.findall(r"\d+", "".join(text))]
        ok = len(nums) > 20 and nums[0] == start and all(b == x + 1 for x, b in zip(nums[:-2], nums[1:-1]))
        out[i] = (ok, toks, (toks - 1) / (t1 - first) if toks and first and t1 > first else 0.0, first - t0 if first else -1)
    except Exception as e:
        out[i] = (False, None, 0.0, -1, repr(e))


for n in NS:
    out = [None] * n
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    tps = [o[2] for o in out]
    print(f"{n} at once: text right {sum(bool(o[0]) for o in out)}/{n}; per stream {min(tps):.1f}-{max(tps):.1f} tok/s "
          f"(sum {sum(tps):.1f}); first token after {min(o[3] for o in out):.1f}-{max(o[3] for o in out):.1f} s; "
          f"{sum(o[1] or 0 for o in out)} tokens in {wall:.1f} s" + "".join(f"  ERR {o[4]}" for o in out if len(o) > 4), flush=True)
