#!/usr/bin/env python3
"""strata_ctx_sweep.py PORT OUTDIR CTX [N=2] [OUTTOK=256]: prompt-length sweep (k = 1024) against a running Strata server started with -c CTX.
Lengths 16k 32k 64k 96k 128k 192k 228k 256k; each prompt holds N_k - 800 tokens (room for the question, the template and the output cap) of Strata's own source tree
(*.md, *.py, *.cpp of ~/ref/Strata-0.1.40, fixed order, so the prefix of every length is the same text) behind a random nonce (cache cold), then a question.
Characters per token are calibrated on the first (16k) request; a request that overflows the context is retried once 8% shorter. Every request streamed, temperature 0:
ttft, total, prompt / completion tokens from the usage block, prefill tok/s = prompt/ttft, decode tok/s = completion/(total - ttft). One JSON line per request in OUTDIR/sweep.jsonl."""
import json, os, random, subprocess, sys, time
import requests

port, out, CTX = sys.argv[1], sys.argv[2], int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 2
OUTTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 256
SRC = os.environ.get("SRC_DIR", os.path.expanduser("~/ref/Strata-0.1.40"))
os.makedirs(out, exist_ok=True)
K = 1024
LENGTHS = [16, 32, 64, 96, 128, 192, 228, 256]


def corpus():
    parts = []
    for ext in (".md", ".py", ".cpp"):
        for root, dirs, files in os.walk(SRC):
            dirs[:] = sorted(d for d in dirs if d not in (".git", ".venv", "build-hip", "node_modules"))
            for f in sorted(files):
                if f.endswith(ext):
                    p = os.path.join(root, f)
                    parts.append("\n\n=== %s ===\n" % os.path.relpath(p, SRC) + open(p, encoding="utf-8", errors="ignore").read())
    return "".join(parts)


def chat(prompt):
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", stream=True, timeout=7200,
                      json={"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": OUTTOK, "temperature": 0,
                            "stream": True, "stream_options": {"include_usage": True}})
    if r.status_code != 200:
        return t0, None, time.time(), {}, "HTTP %d %s" % (r.status_code, r.text[:200])
    first = None; usage = None; err = ""
    for line in r.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("error"):
            err = str(j["error"])[:200]
        if j.get("usage"):
            usage = j["usage"]
        for c in j.get("choices", []):
            d = c.get("delta", {})
            if (d.get("content") or d.get("reasoning_content")) and first is None:
                first = time.time()
    return t0, first, time.time(), usage or {}, err


text = corpus()
print("corpus chars", len(text), flush=True)
pts = [(0, 0)]      # (chars, prompt_tokens) of the successful requests so far, ascending: the corpus is one fixed prefix, deeper parts tokenize denser, so extrapolate from the last two points
chars_for = {}      # length_k -> chars that reached the target in iteration 1, reused (same text length) in the next iterations
for it in range(1, N + 1):
    for kk in LENGTHS:
        target = int((kk * K - 800) * 0.985)
        scale = 1.0
        for attempt in (1, 2):
            if kk in chars_for:
                chars = int(chars_for[kk] * scale)
            else:
                if len(pts) >= 3:
                    (c1, t1), (c2, t2) = pts[-2], pts[-1]
                    slope = (c2 - c1) / max(t2 - t1, 1)
                else:
                    slope = 3.0
                chars = int((pts[-1][0] + (target - pts[-1][1]) * slope) * scale)
            if chars > len(text):
                print("corpus too short for", kk, "k", flush=True); break
            prompt = "Session %08x. Source files follow.\n" % random.getrandbits(32) + text[:chars] + "\n\nQuestion: summarize in five bullet points what the code above does."
            t0, f, t1, u, err = chat(prompt)
            pt = u.get("prompt_tokens", 0); co = u.get("completion_tokens", 0)
            row = dict(length_k=kk, iteration=it, attempt=attempt, chars=chars, prompt_tokens=pt, completion_tokens=co,
                       ttft_s=round(f - t0, 2) if f else None, total_s=round(t1 - t0, 2),
                       prefill_tok_s=round(pt / (f - t0), 1) if f and pt else None,
                       decode_tok_s=round(co / (t1 - f), 2) if f and t1 > f and co else None, error=err)
            open(os.path.join(out, "sweep.jsonl"), "a").write(json.dumps(row) + "\n")
            print(row, flush=True)
            if err and attempt == 1:
                scale = 0.9; continue
            if pt and not err:
                chars_for.setdefault(kk, chars)
                if it == 1:
                    pts.append((chars, pt))
            break
