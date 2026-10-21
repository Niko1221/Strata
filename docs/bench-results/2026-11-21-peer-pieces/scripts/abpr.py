#!/usr/bin/env python3
"""A/B run: fixed prompts (the repository docs shuffled with a seed per round), streamed, one JSON line per prompt.
usage: abpr.py ARM ROUND OUT.jsonl"""
import json, sys, time, random, glob, urllib.request
arm, rnd, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
URL = "http://localhost:8081/v1/chat/completions"
DOCS = [open(p, errors="ignore").read() for p in sorted(glob.glob("<strata>/docs/*.md"))]
def prompt(chars, seed):
    r = random.Random(seed); d = DOCS[:]; r.shuffle(d)
    return (f"Dokument-ID {seed}\n\n" + "\n\n".join(d))[:chars] + "\n\nWas ist der Kern des Textes?"
def run(text, max_tokens):
    body = json.dumps({"model": "qwen3.8-flash-next", "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": text}]}).encode()
    t0 = time.time(); t1 = None; usage = None
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if t1 is None and ch and ch[0].get("delta", {}).get("content"): t1 = time.time()
    return t1 - t0, usage["prompt_tokens"]
run("Hallo", 4)   # warm-up
for chars in (32000, 128000):
    t, n = run(prompt(chars, 4000 + rnd), 8)
    row = dict(arm=arm, round=rnd, prompt_chars=chars, prompt_tokens=n, ttft_s=round(t, 2), prefill_tps=round(n / t))
    print(json.dumps(row), flush=True); open(out, "a").write(json.dumps(row) + "\n")
