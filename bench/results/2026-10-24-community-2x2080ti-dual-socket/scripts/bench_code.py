#!/usr/bin/env python3
"""Code-workload benchmark against whichever Strata server runs on :8080.

  python3 bench_code.py <label> [engine-log-path]

Sends a fixed set of coding prompts (300 tokens each) plus one long prompt, then reads
the engine's own log for decode tok/s, cache-hit and fresh-prefill speed. Works against
both the old (0.1.43.1) and the new (0.2.x) engines; the log path differs per install.
"""
import json, os, pathlib, re, statistics as st, sys, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
def _model():
    try:
        with urllib.request.urlopen(URL.replace("/chat/completions", "/models"), timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return "qwen3.8-flash-next-unsloth-ud-iq4_xs"
MODEL = _model()
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
LOG = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else \
    pathlib.Path(os.environ.get("STRATA_LOG", "engine.log"))

PROMPTS = [
    "Write a Python class implementing a thread-safe LRU cache with TTL expiry. Include type hints, a docstring and one usage example.",
    "Implement a Python function that merges two sorted lists of dicts by a key, handling duplicates. State the time complexity.",
    "Write a bash script that finds files older than N days under a directory tree and deletes them, with a dry-run flag and confirmation prompt.",
    "Implement quicksort in Rust with generics and a comparator, then explain the pivot choice and its worst case.",
    "Write a Python async function that fetches many URLs concurrently with a semaphore, per-request timeout and bounded retries.",
    "Implement a small JSON-schema validator in Python for a useful subset of draft-07, with clear error messages.",
]
FILLER = ("The quarterly logistics review covers warehouse throughput, inbound scheduling "
          "and dock allocation for every regional hub. ")
LONG = FILLER * 640 + "\n\nList the three most frequent topics mentioned above."

RE_GEN = re.compile(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
RE_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)%")
RE_PRE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")
RE_LEN = re.compile(r"\(length,")


def send(prompt, max_tokens=300):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.6}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        json.load(r)
    return time.perf_counter() - t0


def main():
    mark = len(LOG.read_text(errors="replace"))
    walls = [send(p) for p in PROMPTS]
    tail = LOG.read_text(errors="replace")[mark:]
    gens = [float(x[2]) for x in RE_GEN.findall(tail)]
    hits = [float(x) for x in RE_HIT.findall(tail)]

    # long prompt: fresh (a new nonce keeps it out of the prompt cache)
    mark = len(LOG.read_text(errors="replace"))
    w_long = send(LONG + f"\n\n<!-- nonce {time.time_ns()} -->", max_tokens=40)
    ltail = LOG.read_text(errors="replace")[mark:]
    pres = [(int(a), int(b), int(c), int(d), float(e)) for a, b, c, d, e in RE_PRE.findall(ltail)]
    pre = max(pres, key=lambda x: x[0]) if pres else None

    r = {"label": LABEL,
         "decode_median": st.median(gens) if gens else None,
         "decode_all": gens,
         "hit_median": st.median(hits) if hits else None,
         "truncated": len(RE_LEN.findall(tail)),
         "wall_median": st.median(walls),
         "long_prompt": {"tokens": pre[0], "read": pre[2], "ms": pre[3], "tps": pre[4]} if pre else None,
         "long_wall": w_long}
    print(f"{LABEL}: decode中位 {r['decode_median'] or 0:5.1f} tok/s (n={len(gens)})  "
          f"命中中位 {r['hit_median'] or 0:5.1f}%  截断 {r['truncated']}  "
          f"代码wall中位 {r['wall_median']:.1f}s", flush=True)
    if pre:
        print(f"  长 prompt {pre[0]} tok: 读入 {pre[3]/1000:.1f}s @ {pre[4]:.0f} tok/s", flush=True)
    pathlib.Path(f"bench_{LABEL}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
