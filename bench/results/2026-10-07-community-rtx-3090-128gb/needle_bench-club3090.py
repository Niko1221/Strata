#!/usr/bin/env python3
"""Needle-in-a-haystack recall against a llama.cpp (club-3090) server.

Same haystack, code words, seed, lengths and depths as tools/needle_bench.py, but
the model is the llama.cpp alias (default qwen3.8-27b) and thinking is disabled
server-side (INSTRUCT=1), so no chat_template_kwargs is sent. The context limit is
read from /props.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"),
           ("third_party/llama.cpp/docs", "*.md")]
WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern",
         "meadow", "cobalt", "juniper", "tundra", "velvet", "ember"]
CHARS_PER_TOKEN = 3.2


def haystack(n_chars: int) -> str:
    files = []
    for d, pat in SOURCES:
        files += sorted((ROOT / d).rglob(pat))
    parts, total = [], 0
    while total < n_chars:
        for f in files:
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"\n\n=== {f.relative_to(ROOT).as_posix()} ===\n{t}")
            total += len(parts[-1])
            if total >= n_chars:
                break
        if not files:
            raise SystemExit("no text files found to build the haystack from")
    return "".join(parts)[:n_chars]


def ask(url: str, model: str, prompt: str, timeout: float) -> tuple[str, int, float]:
    body = {"model": model, "max_tokens": 40, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return (out["choices"][0]["message"].get("content") or ""), out["usage"]["prompt_tokens"], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lengths", default="32k", help="comma-separated, e.g. 1k,32k,128k,262k")
    ap.add_argument("--depths", default="10,50,90", help="where the code word sits, percent of the text")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", help="also write the results as JSON")
    a = ap.parse_args()
    try:
        props = json.loads(urllib.request.urlopen(a.url.rstrip("/") + "/props", timeout=10).read())
        ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx") or 0)
    except (OSError, ValueError):
        ctx = 0
    rnd = random.Random(7)
    rows, found = [], 0
    for L in a.lengths.split(","):
        tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        tokens = int(tokens * 0.98)
        if ctx and tokens + 200 > ctx:
            print(f"{L:>5}: skipped (the server's context is {ctx})")
            continue
        text = haystack(int(tokens * CHARS_PER_TOKEN))
        for d in (int(x) for x in a.depths.split(",")):
            word = f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}-{rnd.randint(100, 999)}"
            cut = int(len(text) * d / 100)
            cut = text.rfind("\n", 0, cut) + 1 or cut
            needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
            prompt = (text[:cut] + needle + text[cut:] +
                      "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
            try:
                answer, n, secs = ask(a.url, a.model, prompt, a.timeout)
            except (OSError, urllib.error.HTTPError) as e:
                print(f"{L:>5} depth {d:>3}%: request failed ({e})")
                rows.append({"length": L, "depth": d, "error": str(e)})
                continue
            ok = word in answer
            found += ok
            print(f"{L:>5} depth {d:>3}%: {'FOUND' if ok else 'missed'}  ({n:,} prompt tokens, {secs:.0f} s)"
                  + ("" if ok else f"  answer: {answer.strip()[:80]!r}"), flush=True)
            rows.append({"length": L, "depth": d, "prompt_tokens": n, "seconds": round(secs, 1), "found": ok,
                         "word": word, "answer": answer.strip()[:200]})
    done = [r for r in rows if "found" in r]
    print(f"{found} of {len(done)} found")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0 if done and found == len(done) else 1


if __name__ == "__main__":
    sys.exit(main())
