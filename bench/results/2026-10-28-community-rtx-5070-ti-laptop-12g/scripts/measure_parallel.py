#!/usr/bin/env python3
"""Send N requests to a running Strata server at the same moment and read the engine's own numbers back.

Same method as measure.py: every request starts with a unique nonce, so the server reports
"0 reused" and no request rides on the prefix cache of another.  The engine's log lines are read
back after every round:

    strata serve: prompt N tokens = R reused + P read in T ms (X tok/s), G generated in U ms (Y tok/s)
    strata serve: decode expert cache hit rate: H% (a hits / b lookups)

With --concurrent 2 the requests are released together by a threading.Barrier so they really
overlap.  The client-side number per request is its own wall time; the engine's lines are the
server-side ones and are reported in log order.

    python measure_parallel.py --concurrent 2 --tokens 4096 --runs 3 --out par2-4k.json
    python measure_parallel.py --concurrent 1 --tokens 4096 --runs 3 --out solo-4k.json

Standard library only.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

HOST = "http://127.0.0.1:8080"  # overridden by --host
MODEL = "qwen3.8-flash-next-iq3_xxs"
DEFAULT_LOG = r"D:\Strata\Strata-main\strata-iq3_xxs.log"

FILLER = "The quick brown fox jumps over the lazy dog. "
TASK = ("\n\nTask: count from 1 to 200, one number per line. Do not explain, do not add headers.\n")
CHARS_PER_TOKEN = 4.5

PROMPT_RE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)% \(([\d.]+) \w+ / ([\d.]+) \w+\)")
KV_RE = re.compile(r"KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
DRAFT_RE = re.compile(r"drafts accepted (\d+) of (\d+)")
SLOT_RE = re.compile(r"\bslot\b", re.I)


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
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []


TUNE = None  # 每个请求带过去的引擎调参，main() 里由 --tune 填


def one_request(idx, tier, target_tokens, timeout, barrier, out):
    """One blocking chat completion.  Waits on the barrier so every thread fires together."""
    nonce = "PAR-%s-r%d-%d-%d" % (tier, idx, int(time.time() * 1000) % 1000000, os.getpid() % 997)
    prompt = build_prompt(nonce, target_tokens)
    req_body = {"model": MODEL, "max_tokens": 256, "stream": False,
                "messages": [{"role": "user", "content": prompt}]}
    if TUNE:
        req_body["strata_tune"] = TUNE
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        barrier.wait(timeout=60)
        started = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        wall = time.time() - started
        out[idx] = {"idx": idx, "nonce": nonce, "wall_s": round(wall, 2),
                    "prompt_tokens": payload["usage"]["prompt_tokens"],
                    "completion_tokens": payload["usage"]["completion_tokens"],
                    "cached_tokens": payload["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    "finish_reason": payload["choices"][0].get("finish_reason")}
    except Exception as exc:
        out[idx] = {"idx": idx, "error": repr(exc)}


def parse_lines(lines):
    """Every engine line of interest, in log order."""
    recs = []
    for ln in lines:
        m = PROMPT_RE.search(ln)
        if m:
            recs.append({"kind": "prompt", "line": ln.strip(),
                         "read_tokens": int(m.group(3)), "reused_tokens": int(m.group(2)),
                         "prompt_ms": int(m.group(4)), "prefill_tok_s": float(m.group(5)),
                         "gen_tokens": int(m.group(6)), "decode_ms": int(m.group(7)),
                         "decode_tok_s": float(m.group(8))})
            continue
        m = HIT_RE.search(ln)
        if m:
            recs.append({"kind": "hit", "line": ln.strip(), "expert_cache_hit_pct": float(m.group(1))})
            continue
        m = KV_RE.search(ln)
        if m:
            recs.append({"kind": "kv", "line": ln.strip(), "kv_vram_hit_pct": float(m.group(1)),
                         "kv_ram_mib": float(m.group(3))})
            continue
        m = DRAFT_RE.search(ln)
        if m:
            recs.append({"kind": "draft", "line": ln.strip(),
                         "accepted": int(m.group(1)), "total": int(m.group(2))})
    return recs


def one_round(tier, run_idx, target_tokens, log_path, timeout, concurrent):
    offset = log_size(log_path)
    got = {}
    barrier = threading.Barrier(concurrent)
    threads = [threading.Thread(target=one_request,
                                args=(i, tier, target_tokens, timeout, barrier, got))
               for i in range(concurrent)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    round_wall = time.time() - t0

    time.sleep(2.0)                        # let the engine flush its log lines
    recs = parse_lines(tail_since(log_path, offset))

    cli = [got[i] for i in range(concurrent)]
    ok = [c for c in cli if "wall_s" in c]
    round_rec = {"tier": tier, "run": run_idx, "concurrent": concurrent,
                 "round_wall_s": round(round_wall, 2), "clients": cli}

    if ok:
        gen = sum(c["completion_tokens"] for c in ok)
        slowest = max(c["wall_s"] for c in ok)
        per = [round(c["completion_tokens"] / c["wall_s"], 1) for c in ok]
        round_rec.update({
            "total_generated": gen,
            "slowest_wall_s": slowest,
            "aggregate_tok_s": round(gen / slowest, 1),
            "per_request_end_to_end_tok_s": per,
            "per_request_median_e2e": sorted(per)[len(per) // 2],
        })

    prompts = [r for r in recs if r["kind"] == "prompt"]
    hits = [r for r in recs if r["kind"] == "hit"]
    round_rec["engine_prompt_lines"] = prompts
    round_rec["engine_hit_lines"] = hits
    if prompts:
        dec = sorted(p["decode_tok_s"] for p in prompts)
        pre = sorted(p["prefill_tok_s"] for p in prompts)
        round_rec["engine_decode_tok_s"] = dec
        round_rec["engine_median_decode_tok_s"] = dec[len(dec) // 2]
        round_rec["engine_median_prefill_tok_s"] = pre[len(pre) // 2]
        round_rec["engine_min_decode_tok_s"] = dec[0]
    if hits:
        round_rec["expert_cache_hit_pct"] = hits[-1]["expert_cache_hit_pct"]
    return round_rec


def main():
    global HOST, TUNE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--tier", required=True, help="label for this sweep, e.g. 4k")
    ap.add_argument("--tokens", type=int, required=True, help="target prompt tokens")
    ap.add_argument("--concurrent", type=int, default=2, help="requests fired together (1 = one at a time)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tune", default=None, help='每个请求带过去的引擎调参 JSON，例 {"pcie_frac":0.20}')
    a = ap.parse_args()
    HOST = a.host
    TUNE = json.loads(a.tune) if a.tune else None

    print("tier %s, %d prompt tokens, %d concurrent, %d rounds"
          % (a.tier, a.tokens, a.concurrent, a.runs))
    rounds = []
    for i in range(1, a.runs + 1):
        r = one_round(a.tier, i, a.tokens, a.log, a.timeout, a.concurrent)
        rounds.append(r)
        print("  round %d: round_wall %ss | aggregate %s tok/s | per-request e2e %s | "
              "engine decode %s | hit %s%%"
              % (i, r.get("round_wall_s"), r.get("aggregate_tok_s"),
                 r.get("per_request_end_to_end_tok_s"), r.get("engine_decode_tok_s"),
                 r.get("expert_cache_hit_pct")))
        errs = [c for c in r["clients"] if "error" in c]
        for e in errs:
            print("     client %d FAILED: %s" % (e["idx"], e["error"]))

    good = [r for r in rounds if r.get("engine_decode_tok_s")]
    summary = {}
    if good:
        def med(key):
            v = sorted(r[key] for r in good if r.get(key) is not None)
            return v[len(v) // 2] if v else None
        summary = {"rounds_ok": len(good),
                   "concurrent": a.concurrent,
                   "median_aggregate_tok_s": med("aggregate_tok_s"),
                   "median_per_request_e2e_tok_s": med("per_request_median_e2e"),
                   "median_engine_decode_tok_s": med("engine_median_decode_tok_s"),
                   "median_engine_prefill_tok_s": med("engine_median_prefill_tok_s"),
                   "median_hit_pct": med("expert_cache_hit_pct"),
                   "prompt_tokens": good[0]["engine_prompt_lines"][0]["read_tokens"] if good[0]["engine_prompt_lines"] else None}
        print("  MEDIAN: aggregate %.1f tok/s | per-request e2e %.1f tok/s | engine decode %.1f tok/s | hit %.1f%%"
              % (summary["median_aggregate_tok_s"], summary["median_per_request_e2e_tok_s"],
                 summary["median_engine_decode_tok_s"], summary["median_hit_pct"] or 0))

    out = {"tier": a.tier, "target_tokens": a.tokens, "concurrent": a.concurrent,
           "rounds": rounds, "summary": summary}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s" % a.out)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
