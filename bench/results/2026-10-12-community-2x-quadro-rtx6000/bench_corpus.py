#!/usr/bin/env python3
"""Fixed inference corpus for the Strata NUMA comparison.

Sends deterministic prompts (same bytes every run) to the OpenAI-compatible
endpoint and records, per request:
  - API-side timing (TTFT, total, tokens from usage)
  - the engine-log lines produced during that request (byte slice), and the
    metrics parsed from them (prefill tok/s, decode tok/s, drafts).

Usage:
  bench_corpus.py --out <dir> --server-out <path> [--no-64k]

Everything lives under <dir>: results.json, per-request engine/server log slices.
"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
ENGINE_LOG = Path("strata-unsloth-ud-q4_k_xl.log")

WORDS = (
    "cache expert router tensor quant kernel matrix layer token prompt decode "
    "prefill memory socket numa channel bandwidth latency throughput vector "
    "shard index profile resident stream pinned budget context window pipeline "
    "scheduler worker thread affinity block page fault reclaim eviction atomic "
    "sparse dense projection attention key value norm embedding vocab logits "
    "gradient batch slot ring buffer checkpoint draft verify speculative"
).split()

SERVE_RE = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms "
    r"\(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), "
    r"drafts accepted (\d+) of (\d+), (\d+) checkpoints"
)
DONE_RE = re.compile(r"\[strata\] done: (\d+) tokens in (\d+) s \(([\d.]+) tok/s\)")


def gen_filler(n_tokens, seed):
    rng = random.Random(seed)
    # ~4 chars/token
    n_words = max(1, int(n_tokens * 4 / 6))
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def prompt_for(kind, idx):
    seed = {"short": 1000, "medium": 2000, "long": 3000, "xlong": 4000}[kind] + idx
    if kind == "short":
        tasks = [
            "Write a Python LRU cache class with get and put, code only.",
            "Explain in three sentences what a NUMA node is.",
            "Write a bash one-liner to find the ten largest files under /var.",
            "Give a short C function that reverses a singly linked list.",
            "Explain the difference between latency and throughput briefly.",
            "Write a SQL query selecting the top 5 customers by revenue.",
            "Describe how a CPU cache miss is handled, in four sentences.",
            "Write a Python function to parse an ISO-8601 date without libraries.",
            "Summarize what happens during a PCIe memory transaction.",
            "Write a regex matching IPv4 addresses, then explain it briefly.",
            "Give pseudocode for merge sort.",
            "Explain why mmap plus mlock is used for pinned inference buffers.",
        ]
        return tasks[idx % len(tasks)]
    body = gen_filler({"medium": 8000, "long": 32000, "xlong": 64000}[kind], seed)
    tail = ("\n\nUsing only the text above, state in one sentence what the most "
            "frequent topic appears to be, then give the capital of France.")
    return body + tail


def http_post(path, payload, stream=True, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def wait_health(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            d = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read())
            if d.get("loaded"):
                return d
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit("server not ready")


def stream_request(prompt, max_tokens):
    payload = {
        "model": "strata",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft = None
    completion = 0
    prompt_tokens = 0
    with http_post("/v1/chat/completions", payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except Exception:
                continue
            if chunk.get("usage"):
                completion = chunk["usage"].get("completion_tokens", completion)
                prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.time() - t0
    total = time.time() - t0
    return (ttft or total), total, completion, prompt_tokens


def read_slice(path, start, end):
    if not path or not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(max(0, end - start)).decode("utf-8", "ignore")


def size_of(path):
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def run_one(d, kind, idx, max_tokens, server_out, scored=True):
    prompt = prompt_for(kind, idx)
    E0, S0 = size_of(ENGINE_LOG), size_of(server_out)
    t0 = time.time()
    ttft, total, comp, ptok = stream_request(prompt, max_tokens)
    t1 = time.time()
    E1, S1 = size_of(ENGINE_LOG), size_of(server_out)
    elog = read_slice(ENGINE_LOG, E0, E1)
    slog = read_slice(server_out, S0, S1)
    eng = None
    ms = SERVE_RE.findall(elog)
    if ms:
        (p_tok, reused, read, read_ms, pre_tok_s, gen, gen_ms, dec_tok_s, da, db, ck) = ms[-1]
        eng = {
            "prompt_tok": int(p_tok), "reused": int(reused), "read": int(read),
            "read_ms": int(read_ms), "prefill_tok_s": float(pre_tok_s),
            "generated": int(gen), "gen_ms": int(gen_ms), "decode_tok_s": float(dec_tok_s),
            "drafts_accepted": int(da), "drafts_total": int(db), "checkpoints": int(ck),
        }
    done = None
    dm = DONE_RE.findall(slog)
    if dm:
        done = {"tokens": int(dm[-1][0]), "seconds": int(dm[-1][1]), "tok_s": float(dm[-1][2])}
    rec = {
        "kind": kind, "idx": idx, "scored": scored,
        "target_context": {"medium": 8000, "long": 32000, "xlong": 64000}.get(kind),
        "wall_start": t0, "wall_end": t1,
        "api_ttft_s": round(ttft, 3), "api_total_s": round(total, 3),
        "api_prompt_tokens": ptok, "api_completion_tokens": comp,
        "api_decode_tok_s": round(comp / (total - ttft), 2) if total > ttft and comp else None,
        "engine": eng, "server_done": done,
    }
    if scored:
        tag = f"{kind}{idx:02d}"
        (d / f"elog-{tag}.txt").write_text(elog)
        (d / f"slog-{tag}.txt").write_text(slog)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-64k", action="store_true")
    args = ap.parse_args()

    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    health = wait_health()
    print("ready:", json.dumps({k: health.get(k) for k in ("model", "max_context", "images", "loaded")}))

    results = {"health": health, "requests": []}

    # warmup (not scored): exercise both short and a medium context
    print("warmup ...")
    for i in range(3):
        run_one(d, "short", i, args.max_tokens, args.server_out, scored=False)
    run_one(d, "medium", 0, args.max_tokens, args.server_out, scored=False)

    plan = [("short", i) for i in range(12)] + [("medium", i) for i in range(3)]
    plan += [("long", i) for i in range(2)]
    if not args.no_64k:
        plan += [("xlong", 0)]

    for kind, idx in plan:
        rec = run_one(d, kind, idx, args.max_tokens, args.server_out, scored=True)
        results["requests"].append(rec)
        eng = rec["engine"] or {}
        print(f"  {kind}{idx:02d}: in={rec['api_prompt_tokens']} out={rec['api_completion_tokens']} "
              f"ttft={rec['api_ttft_s']:.2f}s "
              f"prefill={eng.get('prefill_tok_s','?')} decode={eng.get('decode_tok_s','?')} tok/s")

    (d / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {d/'results.json'}")


if __name__ == "__main__":
    main()
