#!/usr/bin/env python3
"""longctx_chat.py PORT TOKENS [NEEDLES] : a needle in a long prompt. About TOKENS tokens of numbered filler lines with NEEDLES (default 3) secret codes
hidden at 25% / 50% / 75% depth; asks for them back (greedy, no reasoning). Exit 0 only when every code is in the answer. Prints usage, time and the reply.
Loopback only. Stdlib only."""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
tokens = int(sys.argv[2])
needles = int(sys.argv[3]) if len(sys.argv) > 3 else 3
base = f"http://127.0.0.1:{port}"
with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
    model = json.load(r)["data"][0]["id"]

per_line = 18  # about 18 tokens (measured: 4096 asked = 5312 real at 14) for one "Log entry ..." line
n_lines = max(10, tokens // per_line)
codes = [f"{(7919 * (i + 3)) % 9000 + 1000}-{(104729 * (i + 5)) % 9000 + 1000}" for i in range(needles)]
at = {int(n_lines * (i + 1) / (needles + 1)): codes[i] for i in range(needles)}
lines = []
for i in range(n_lines):
    if i in at:
        lines.append(f"Log entry {i}: the vault access code number {list(at).index(i) + 1} is {at[i]}.")
    else:
        lines.append(f"Log entry {i}: the river level was normal and the weather stayed calm today.")
q = " ".join(f"{k + 1}" for k in range(needles))
prompt = "\n".join(lines) + f"\n\nQuestion: list the vault access codes number {q} from the log, in order, one per line, nothing else."
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0, "max_tokens": 120, "reasoning_effort": "none"}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t = time.time()
try:
    with urllib.request.urlopen(req, timeout=7000) as r:
        data = json.load(r)
except Exception as e:
    print(f"needle{tokens}: REQUEST FAILED {e!r} after {time.time() - t:.0f}s")
    sys.exit(2)
ch = data["choices"][0]
text = ch["message"].get("content") or ""
ok = all(c in text for c in codes)
print(f"needle{tokens} {time.time() - t:.0f}s usage={data.get('usage')} finish={ch.get('finish_reason')} codes={codes} reply={text[:160]!r} {'OK' if ok else 'WRONG'}")
sys.exit(0 if ok else 1)
