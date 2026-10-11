#!/usr/bin/env python3
"""Measure Strata prefill / decode speed at a given context tier.

Both numbers reported here are the engine's own, read back out of its log:

    strata serve: prompt N tokens = R reused + P read in T ms (X tok/s), G generated in U ms (Y tok/s)
    strata serve: decode expert cache hit rate: H% (a hits / b lookups)

X is the prefill (prompt) rate, Y is the decode rate.  TTFT is the engine's own prompt time plus
the request overhead; the engine prints the prompt time, so TTFT is taken as that.

Every run uses a unique nonce at the very start of the prompt, so the server reports "0 reused" and
no run rides on the prefix cache of the one before it.  That is what "fresh run" means below.

    python measure.py --tier 4k --runs 3 --out results-4k.json

Standard library only.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

HOST = "http://127.0.0.1:8080"
MODEL = "qwen3.8-flash-next-iq3_xxs"
DEFAULT_LOG = r"D:\Strata\Strata-main\strata-iq3_xxs.log"

# target prompt tokens -> how many repetitions of FILLER approximate it
FILLER = "The quick brown fox jumps over the lazy dog. "
TASK = ("\n\nTask: count from 1 to 200, one number per line. Do not explain, do not add headers.\n")
CHARS_PER_TOKEN = 4.5                      # measured on this model with this filler

PROMPT_RE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)% \(([\d.]+) \w+ / ([\d.]+) \w+\)")
KV_RE = re.compile(r"KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
DRAFT_RE = re.compile(r"drafts accepted (\d+) of (\d+)")


def build_prompt(nonce, target_tokens):
    chars = int(target_tokens * CHARS_PER_TOKEN)
    reps = max(1, chars // len(FILLER))
    return nonce + "\n" + (FILLER * reps) + TASK


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def tail_since(path, offset):
    """The new bytes of the log since `offset`, decoded permissively."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []


def one_run(tier, run_idx, target_tokens, log_path, timeout):
    nonce = "BENCH-%s-run%d-%d" % (tier, run_idx, int(time.time() * 1000) % 1000000)
    prompt = build_prompt(nonce, target_tokens)
    body = json.dumps({"model": MODEL, "max_tokens": 256, "stream": False,
                       "messages": [{"role": "user", "content": prompt}]}).encode()

    offset = log_size(log_path)
    started = time.time()
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    wall = time.time() - started

    time.sleep(1.5)                        # let the engine finish writing its log lines
    lines = tail_since(log_path, offset)

    rec = {"tier": tier, "run": run_idx, "nonce": nonce, "wall_s": round(wall, 2),
           "prompt_tokens": payload["usage"]["prompt_tokens"],
           "completion_tokens": payload["usage"]["completion_tokens"],
           "cached_tokens": payload["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
           "finish_reason": payload["choices"][0].get("finish_reason")}

    for ln in lines:
        m = PROMPT_RE.search(ln)
        if m:
            rec.update({"read_tokens": int(m.group(3)), "reused_tokens": int(m.group(2)),
                        "prompt_ms": int(m.group(4)), "prefill_tok_s": float(m.group(5)),
                        "engine_gen_tokens": int(m.group(6)), "decode_ms": int(m.group(7)),
                        "decode_tok_s": float(m.group(8))})
        m = HIT_RE.search(ln)
        if m:
            rec["expert_cache_hit_pct"] = float(m.group(1))
        m = KV_RE.search(ln)
        if m:
            rec["kv_vram_hit_pct"] = float(m.group(1))
            rec["kv_ram_mib"] = float(m.group(3))
        m = DRAFT_RE.search(ln)
        if m:
            rec["drafts_accepted"] = int(m.group(1))
            rec["drafts_total"] = int(m.group(2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="label for this sweep, e.g. 4k / 32k / 128k")
    ap.add_argument("--tokens", type=int, help="target prompt tokens (default: parsed from --tier)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    target = a.tokens
    if target is None:
        m = re.match(r"^(\d+)\s*k$", a.tier.lower())
        target = int(m.group(1)) * 1024 if m else int(a.tier)

    print("tier %s, target %d prompt tokens, %d fresh runs" % (a.tier, target, a.runs))
    runs = []
    for i in range(1, a.runs + 1):
        try:
            rec = one_run(a.tier, i, target, a.log, a.timeout)
        except Exception as exc:
            print("  run %d FAILED: %r" % (i, exc))
            runs.append({"tier": a.tier, "run": i, "error": repr(exc)})
            continue
        runs.append(rec)
        print("  run %d: prompt %s tok @ %s tok/s | decode %s tok/s | hit %s%% | wall %ss | reused %s"
              % (i, rec.get("read_tokens"), rec.get("prefill_tok_s"), rec.get("decode_tok_s"),
                 rec.get("expert_cache_hit_pct"), rec.get("wall_s"), rec.get("reused_tokens")))

    ok = [r for r in runs if "decode_tok_s" in r]
    summary = {}
    if ok:
        def med(key):
            v = sorted(r[key] for r in ok if key in r)
            return v[len(v) // 2] if v else None
        summary = {"runs_ok": len(ok), "median_prefill_tok_s": med("prefill_tok_s"),
                   "median_decode_tok_s": med("decode_tok_s"),
                   "median_hit_pct": med("expert_cache_hit_pct"),
                   "min_decode_tok_s": min(r["decode_tok_s"] for r in ok),
                   "max_decode_tok_s": max(r["decode_tok_s"] for r in ok),
                   "prompt_tokens": ok[0]["prompt_tokens"]}
        print("  MEDIAN: prefill %.1f tok/s, decode %.1f tok/s (%.1f-%.1f), hit %.1f%%, %d prompt tokens"
              % (summary["median_prefill_tok_s"], summary["median_decode_tok_s"],
                 summary["min_decode_tok_s"], summary["max_decode_tok_s"],
                 summary["median_hit_pct"] or 0, summary["prompt_tokens"]))

    out = {"tier": a.tier, "target_tokens": target, "runs": runs, "summary": summary}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s" % a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
