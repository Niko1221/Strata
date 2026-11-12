#!/usr/bin/env python3
"""Teacher-forced probe (STRATA_LOGPOS): five requests of falling size, so the positions restart between requests and the cases can be told apart."""
import json, random, glob, urllib.request
URL = "http://localhost:8081/v1/chat/completions"
DOCS = [open(p, errors="ignore").read() for p in sorted(glob.glob("<strata>/docs/*.md"))]
CODE = [open(p, errors="ignore").read() for p in sorted(glob.glob("<strata>/src/**/*.cpp", recursive=True))[:200]]
REQS = [("docs", 34000), ("docs", 24000), ("code", 16000), ("docs", 8000), ("code", 3000)]
def prompt(kind, n_tok, seed):
    r = random.Random(seed); d = (DOCS if kind == "docs" else CODE)[:]; r.shuffle(d)
    return (f"Dokument-ID {seed}\n\n" + "\n\n".join(d))[: n_tok * 4] + "\n\nWas ist der Kern des Textes?"
def ask(msgs):
    body = json.dumps({"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 1,
                       "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800) as f:
        return json.load(f)["usage"]["prompt_tokens"]
for i, (kind, n) in enumerate(REQS):
    long_text = prompt(kind, n, 9100 + i).rsplit("\n\nWas ist", 1)[0]
    r = random.Random(500 + i); pool = (DOCS if kind == "docs" else CODE)[:]; r.shuffle(pool)
    ext = "\n\n".join(pool)[: 1600]
    t = ask([{"role": "user", "content": long_text}, {"role": "assistant", "content": "Verstanden."},
             {"role": "user", "content": ext}])
    print(kind, n, t, "tok", flush=True)
