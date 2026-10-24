#!/usr/bin/env python3
"""shortp.py CFG [TAG] - short prompts against the server on 127.0.0.1:8080 (key from CFG, never printed): three of
about 570 tokens (the 10-19 test), one of about 2K and one of about 4K, each new text (no prefix reuse), 50 tokens out,
temperature 0.  Prints the prompt tokens, the time to the full answer and the server's prompt speed when it says."""
import json, os, sys, time, urllib.request

cfg, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "s")
tag += "-" + os.urandom(3).hex()   # every invocation new text: a second pass must not reuse the first one's prefix
k = json.load(open(cfg))["api_key"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + k}
src = open("/usr/lib/python3.14/argparse.py", errors="replace").read()
jobs = [("570", 0, 2300), ("570", 1, 2300), ("570", 2, 2300), ("2K", 3, 8000), ("4K", 4, 16000)]
for name, i, n in jobs:
    off = (i * 9000) % max(1, len(src) - n)
    body = {"model": "qwen3.8-flash-next", "max_tokens": 50, "temperature": 0,
            "messages": [{"role": "user", "content": "Note %s-%d.\n%s\nWhat does this code do? One sentence." % (tag, i, src[off:off + n])}]}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                                                json.dumps(body).encode(), H), timeout=900))
    dt = time.time() - t0
    t = r.get("timings") or {}
    pps = t.get("prompt_per_second")
    print("short %-4s %5d prompt tokens, %5.1f s to the answer%s" % (
        name, r["usage"]["prompt_tokens"], dt, (", prompt %.0f tok/s" % pps) if pps else ""), flush=True)
