#!/usr/bin/env python3
"""llm-bench.py BASE_URL LABEL [KTOK ...] - prefill and decode of any OpenAI-style server at long contexts (default
64K, 128K, 200K), two runs each, with the prompts of hal-bench.py (the same text cut at the same places: the
tokens-per-character figure Halogen counted, kept in results/per-char.txt), temperature 0, 600 tokens out, the
server's default thinking.  For test servers on their own port (nobody else sends them requests): no idle wait,
5 s between requests.  KEYFILE=path: a JSON file whose "api_key" is sent as the bearer token (never printed).
Figures from the client: first-token time (prefill, with the request's overhead) and decode = (tokens - 1) / (last
token - first token).  Writes ~/evals/results/llm-bench-LABEL.jsonl."""
import glob, json, os, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
LABEL = sys.argv[2]
SIZES = [int(k) for k in sys.argv[3:]] or [64, 128, 200]
REPS = 2
OUT = os.path.expanduser("~/evals/results/llm-bench-%s.jsonl" % LABEL)
PER_CHAR = float(open(os.path.expanduser("~/evals/results/per-char.txt")).read())
QUESTION = ("\n\nAbove are several modules of the Python standard library. In about 300 words, explain what the last "
            "module above is for and how its main functions work.")
TAG = os.environ.get("BENCH_TAG", "a")   # another tag = other prompts: a server with a conversation cache that
                                         # already read the "a" prompts would reuse them (first token in 0.3 s)
HDR = {"Content-Type": "application/json"}
if os.environ.get("KEYFILE"):
    HDR["Authorization"] = "Bearer " + json.load(open(os.environ["KEYFILE"]))["api_key"]


def corpus():
    files = sorted(glob.glob("/usr/lib/python3.*/*.py"))
    return "".join("\n\n# ===== %s =====\n%s" % (os.path.basename(f), open(f, errors="replace").read()) for f in files)


TEXT = corpus()


def prompt(tag, ktok):
    return "Run %s.\n" % tag + TEXT[:int(ktok * 1000 / PER_CHAR)] + QUESTION


def get(path, timeout=10):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, headers=HDR), timeout=timeout))


def ask(model, text, max_tokens=600):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    t0 = time.time(); first = last = None; usage = {}; chunks = 0; extra = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for k in ("timings", "metrics", "x_strata", "x_gufo"):
                if d.get(k):
                    extra[k] = d[k]
            if any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                   or (c.get("delta") or {}).get("reasoning") for c in d.get("choices") or []):
                last = time.time(); chunks += 1
                if first is None:
                    first = last
    t1 = time.time()
    n = usage.get("completion_tokens") or chunks
    dec = (n - 1) / (last - first) if first and last and last > first and n > 1 else None
    return {"ttft_s": round((first or t1) - t0, 2), "wall_s": round(t1 - t0, 2), "decode_s": round((last or t1) - (first or t1), 2),
            "decode_tps": round(dec, 2) if dec else None, "chunks": chunks, "usage": usage, "extra": extra}


def main():
    model = get("/v1/models")["data"][0]["id"]
    print("%s: model %s, label %s, tokens per character %.4f" % (BASE, model, LABEL, PER_CHAR), flush=True)
    w = ask(model, prompt("warmup", 16), 200)
    print("warm-up 16K: first token %.1f s" % w["ttft_s"], flush=True)
    with open(OUT, "a") as f:
        for ktok in SIZES:
            for rep in range(1, REPS + 1):
                time.sleep(5)
                tag = "%dk-%d-%s" % (ktok, rep, TAG)
                r = ask(model, prompt(tag, ktok))
                r.update(label=LABEL, model=model, ktok=ktok, rep=rep, tag=tag, t_end=time.time())
                f.write(json.dumps(r) + "\n"); f.flush()
                u = r["usage"]; pt = u.get("prompt_tokens") or 0
                print("%-14s %3dK run %d: %7d tokens, first token %6.1f s (%5.0f tok/s), decode %5.1f tok/s over %d tokens"
                      % (LABEL, ktok, rep, pt, r["ttft_s"], pt / r["ttft_s"] if r["ttft_s"] else 0, r["decode_tps"] or 0,
                         u.get("completion_tokens") or r["chunks"]), flush=True)
    print("BENCH-DONE", flush=True)


main()
