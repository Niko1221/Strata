"""prefill-curve.py - incremental prompt-read time by size, the coder's most common prompt shape.

The coder's turns append a tool result of a few hundred to a few thousand tokens to a conversation the server
already holds; their TTFT is that increment's read. This grows ONE conversation over the agent-turns.py context
(cached after the first request) by tool results of SIZES tokens (approximate, 4 chars per token), REPEAT rounds,
max_tokens 1, thinking off, and prints the median TTFT per size. Used to tune STRATA_PREFILL_HELP / _FRAC, whose
share depends on the read's size (sycl/src/prefill/prefill.cpp split_help_frac).
Usage: python3 -I prefill-curve.py [base_url]   (STRATA_KEY, SPEED_NONCE as in agent-turns.py)
"""
import json
import os
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
KEY = os.environ.get("STRATA_KEY", "")
AUTH = {**({"Authorization": f"Bearer {KEY}", "x-api-key": KEY} if KEY else {})}
REPO = os.path.expanduser("~/home_config")
NONCE = os.environ.get("SPEED_NONCE", str(time.time()))
SIZES = [300, 600, 1000, 1500, 2000, 2500, 3000, 4000]
REPEAT = 2   # 2 x ~15K tokens on the ~66K context stays under --max-context 131072
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402
SYSTEM = import_module("agent-turns").SYSTEM


def source():
    """Endless real text for tool results: repo Python files, in a fixed order."""
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(REPO, "tools")) for f in fs if f.endswith(".py"))
    text = "".join(open(p, encoding="utf-8", errors="replace").read() for p in paths)
    return text * (1 + 400_000 // max(1, len(text)))


def ttft(messages):
    body = {"model": "x", "stream": True, "max_tokens": 1, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **AUTH})
    # max_tokens 1: the whole response is the read plus one token. (The first chunk, an empty role delta, comes
    # back before the read, so time to the end, not to the first chunk.)
    t0 = time.time()
    word = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for c in json.loads(line[5:]).get("choices") or []:
                word.append((c.get("delta") or {}).get("content") or "")
    return time.time() - t0, "".join(word)


if __name__ == "__main__":
    src, pos = source(), 0
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[run {NONCE}] Read the files I send; answer each with one word."}]
    ttft(msgs + [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Ready?"}])  # warm the context
    msgs.append({"role": "assistant", "content": "OK"})
    res = {s: [] for s in SIZES}
    for _ in range(REPEAT):
        for s in SIZES:
            chunk = src[pos:pos + 4 * s]
            pos += 4 * s
            msgs.append({"role": "user", "content": "Tool result:\n" + chunk})
            t, word = ttft(msgs)
            res[s].append(t)
            msgs.append({"role": "assistant", "content": word or "OK"})
    for s in SIZES:
        print(f"  prefill {s}: median read+1 {statistics.median(res[s]):.3f}s runs "
              + ",".join(f"{t:.3f}" for t in res[s]), flush=True)
