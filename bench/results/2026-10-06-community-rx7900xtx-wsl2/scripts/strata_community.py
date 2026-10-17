#!/usr/bin/env python3
"""strata_community.py PORT OUTDIR [N=3] [OUTTOK=256]: the measurements asked by Strata's docs/COMMUNITY_BENCHMARKS.md, against a running Strata server.
Per iteration (N of them, run in this order):
  short      a short fixed prompt (a coding question), cache cold (a random nonce opens the message so no prefix is reused)
  long       a long prompt made of Strata's own public docs (DOCS_DIR, truncated to ~LONG_CHARS characters) + a question, cache cold (nonce)
  followup   the SAME conversation continued by a second short question: the long prefix is reused from the conversation cache
Each request is streamed (temperature 0, fixed output cap, ignore nothing): time to first token, total latency, prompt / cached / completion tokens from the usage block,
prefill tok/s = (prompt - cached) / ttft, decode tok/s = completion / (total - ttft). One JSON line per request in OUTDIR/runs.jsonl and a CSV with the same columns."""
import csv, json, os, random, sys, time
import requests

port, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUTTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 256
DOCS = os.environ.get("DOCS_DIR", os.path.expanduser("~/ref/Strata/docs"))
LONG_CHARS = int(os.environ.get("LONG_CHARS", "30000"))
os.makedirs(out, exist_ok=True)


def docs_text():
    t = ""
    for f in sorted(os.listdir(DOCS)):
        if f.endswith(".md"):
            t += open(os.path.join(DOCS, f), encoding="utf-8", errors="ignore").read() + "\n\n"
        if len(t) >= LONG_CHARS:
            break
    return t[:LONG_CHARS]


def chat(msgs):
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", stream=True, timeout=3600,
                      json={"model": "x", "messages": msgs, "max_tokens": OUTTOK, "temperature": 0, "stream": True,
                            "stream_options": {"include_usage": True}})
    first = None; usage = None; text = ""
    for line in r.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            usage = j["usage"]
        for c in j.get("choices", []):
            d = c.get("delta", {})
            piece = d.get("content") or d.get("reasoning_content")
            if piece:
                if first is None:
                    first = time.time()
                text += piece
    t1 = time.time()
    return t0, first, t1, usage or {}, text


def record(kind, it, t0, first, t1, u, rows):
    pt = u.get("prompt_tokens", 0); ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0); co = u.get("completion_tokens", 0)
    ttft = (first - t0) if first else float("nan")
    row = dict(kind=kind, iteration=it, prompt_tokens=pt, cached_tokens=ct, completion_tokens=co, ttft_s=round(ttft, 2), total_s=round(t1 - t0, 2),
               prefill_tok_s=round((pt - ct) / ttft, 1) if first and ttft > 0 and pt - ct > 0 else None,
               decode_tok_s=round(co / (t1 - first), 2) if first and t1 > first else None)
    rows.append(row)
    open(os.path.join(out, "runs.jsonl"), "a").write(json.dumps(row) + "\n")
    print(row, flush=True)


docs = docs_text()
rows = []
for it in range(1, N + 1):
    nonce = "Session %08x. " % random.getrandbits(32)
    short = [{"role": "user", "content": nonce + "Write a Python function that merges overlapping intervals, with a short explanation."}]
    t0, f, t1, u, _ = chat(short); record("short", it, t0, f, t1, u, rows)
    long_msgs = [{"role": "user", "content": nonce + "Here is documentation text:\n\n" + docs + "\n\nQuestion: list the main configuration flags mentioned above and what each does."}]
    t0, f, t1, u, ans = chat(long_msgs); record("long", it, t0, f, t1, u, rows)
    follow = long_msgs + [{"role": "assistant", "content": ans}, {"role": "user", "content": "Which of those flags matter for AMD GPUs?"}]
    t0, f, t1, u, _ = chat(follow); record("followup", it, t0, f, t1, u, rows)
with open(os.path.join(out, "runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
