#!/usr/bin/env python3
"""Strata tuning benchmark: decode (greedy + sampled), cold prompt reads at 3 sizes, warm follow-up turns.

    python3 tune_bench.py --url http://127.0.0.1:8080/v1 --model qwen3.8-flash-next-iq3_s --name NAME [--runs 3]
        [--out results-NAME.json] [--quick]
    python3 tune_bench.py --selftest          # offline checks of the stream validation, no server needed

Writes --out (default results/NAME.json) and prints a one-line summary. Every cold prompt starts with a fresh nonce so
no prefix cache or conversation cache can serve it. Warm turns replay a conversation (exact assistant text) and add
~3K new tokens. The long document is read from longctx.txt.gz next to this script, only for the cold and warm tests.

Output format 2 (this version): every request is kept under "requests" with its status, finish reason, token counts,
cached prompt tokens, client timings and structured output hashes (reasoning and answer hashed separately, plus the
completion token count, the finish reason and one combined hash; see bench_stream.py). Failed requests (error event,
bad or missing finish_reason, missing usage, premature end) are listed under "failures" and left out of every median
and summary; a summary value whose inputs all failed is null. Format 1 (the first version) kept only
sha1(answer text + completion token count)[:10] per run, ignored error events and fell back to chunk counts.
Exit status: 0 all requests ok, 1 some failed (the JSON is still written), 2 the warm-up request failed or a usage
error (--runs must be positive; nothing is sent).
"""
import argparse
import functools
import gzip
import os
import statistics
import sys
import time
import uuid
import json

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from bench_stream import positive_int, public, stream  # noqa: E402
from decode_bench import PROMPTS  # noqa: E402

CPT = 3.6   # chars per token, rough; the server's usage gives the real count
DECODE_MAX_TOKENS = 1024


@functools.lru_cache(maxsize=1)
def longctx():
    """The long-context fixture: Strata's docs/*.md concatenated (see README.md). Read lazily, straight from the .gz."""
    with gzip.open(os.path.join(HERE, "longctx.txt.gz"), "rt", encoding="utf-8") as f:
        return f.read()


def med(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), nd) if xs else None


def geo(xs):
    return round(statistics.geometric_mean(xs), 1) if xs and all(x is not None and x > 0 for x in xs) else None


class Bench:
    def __init__(self, url, model):
        self.url, self.model = url, model
        self.requests, self.failures = [], []

    def ask(self, label, messages, temp, max_tokens, seed=None, need_decode=False):
        r = stream(self.url, self.model, messages, temp, max_tokens, seed=seed)
        ok = r["status"] == "ok" and (r["tps"] is not None or not need_decode)
        if r["status"] == "ok" and not ok:
            r = dict(r, status="no_decode_span", error="fewer than two streamed pieces: no decode rate")
        self.requests.append(public(r, label=label))
        if not ok:
            self.failures.append({"label": label, "status": r["status"], "error": r["error"]})
            print(f"  FAILED {label}: {r['status']}: {r['error']}", flush=True)
        r["ok"] = ok
        return r


def run_bench(a):
    b = Bench(a.url, a.model)
    out = {"script": "tune_bench.py", "format": 2, "name": a.name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model": a.model, "params": {"runs": a.runs, "quick": a.quick, "decode_max_tokens": DECODE_MAX_TOKENS,
                                        "sampled": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                                    "seeds": [1000 + i for i in range(a.runs)]}},
           "decode": {}, "cold": {}, "warm": {}}

    w = b.ask("warm-up", [{"role": "user", "content": "Say OK."}], 0, 8)
    if not w["ok"]:
        return finish(a, out, b, warmup_failed=True)
    for temp in (0.0, 1.0):
        for kind, p in PROMPTS.items():
            key = f"{kind}@{temp}"
            rs = [b.ask(f"{key} run {i + 1}", [{"role": "user", "content": p}], temp, DECODE_MAX_TOKENS,
                        seed=(1000 + i) if temp else None, need_decode=True) for i in range(a.runs)]
            ok = [r for r in rs if r["ok"]]
            out["decode"][key] = {
                "tps": med([r["tps"] for r in ok]), "all": [round(r["tps"], 1) for r in ok],
                "n_ok": len(ok), "n_failed": len(rs) - len(ok),
                "completion_tokens": [r["completion_tokens"] for r in rs],
                "finish_reason": [r["finish_reason"] for r in rs],
                "hash_combined": [(r["hashes"] or {}).get("combined") if r["ok"] else None for r in rs],
                "hash_answer": [(r["hashes"] or {}).get("answer") if r["ok"] else None for r in rs],
                "hash_reasoning": [(r["hashes"] or {}).get("reasoning") if r["ok"] else None for r in rs],
            }
            print(f"  decode {kind:9s} temp {temp}: {out['decode'][key]['all']}"
                  + (f"  ({len(rs) - len(ok)} failed)" if len(ok) < len(rs) else ""), flush=True)
    if not a.quick:
        doc = longctx()
        for k in (4, 16, 64):
            msg = f"[{uuid.uuid4()}]\n" + doc[: int(k * 1000 * CPT)] + "\n\nIn one sentence: what is this document about?"
            r = b.ask(f"cold {k}k", [{"role": "user", "content": msg}], 0, 16)
            if not r["ok"] or not r["ttft"] or not r["prompt_tokens"]:
                out["cold"][f"{k}k"] = {"status": r["status"], "error": r["error"]}
                continue
            pt = r["prompt_tokens"]
            out["cold"][f"{k}k"] = {"status": "ok", "prompt_tokens": pt, "cached_tokens": r["cached_tokens"],
                                   "ttft": round(r["ttft"], 2), "pp_tps": round(pt / r["ttft"], 0),
                                   "engine_pp_tps": r["engine_pp_tps"], "answer": r["text"][:120]}
            print(f"  cold {k}k: {pt} tokens, ttft {r['ttft']:.2f} s, {pt / r['ttft']:.0f} tok/s", flush=True)
        # warm: 32K-token conversation, then 3 follow-ups that each add ~3K tokens of "tool output"
        base = doc[: int(32000 * CPT)]
        msgs = [{"role": "user", "content": f"[{uuid.uuid4()}]\n" + base + "\n\nList three topics this covers, one line each."}]
        r = b.ask("warm base (32K)", msgs, 0, 128)
        warm, broken = [], None
        if not r["ok"]:
            broken = "warm base (32K)"
        else:
            msgs.append({"role": "assistant", "content": r["text"]})
            for i in range(3):
                chunk = doc[int((40000 + i * 3000) * CPT): int((43000 + i * 3000) * CPT)]
                msgs.append({"role": "user", "content": "Tool output:\n" + chunk + "\n\nOne sentence: what does this add?"})
                r = b.ask(f"warm turn {i + 1}", msgs, 0, 128, need_decode=True)
                if not r["ok"]:
                    broken = f"warm turn {i + 1}"     # the conversation cannot continue without the answer
                    break
                msgs.append({"role": "assistant", "content": r["text"]})
                warm.append(r)
                print(f"  warm turn {i + 1}: prompt {r['prompt_tokens']} tokens (reused {r['cached_tokens']}), "
                      f"ttft {r['ttft']:.2f} s, decode {r['tps']:.0f} tok/s", flush=True)
        out["warm"] = {"ttft": med([x["ttft"] for x in warm], 2), "tps": med([x["tps"] for x in warm]),
                       "turns_ok": len(warm), "aborted_at": broken,
                       "prompt_tokens": warm[-1]["prompt_tokens"] if warm else None,
                       "turns": [{"ttft": round(x["ttft"], 2), "tps": round(x["tps"], 1), "prompt_tokens": x["prompt_tokens"],
                                  "cached_tokens": x["cached_tokens"]} for x in warm]}
    return finish(a, out, b)


def finish(a, out, b, warmup_failed=False):
    g = [v["tps"] for k, v in out["decode"].items() if k.endswith("@0.0")]
    s = [v["tps"] for k, v in out["decode"].items() if k.endswith("@1.0")]
    out["summary"] = {"greedy_geo": geo(g), "sampled_geo": geo(s)}   # null unless every prompt has a successful run
    if out["cold"]:
        out["summary"].update({f"pp_{k}": v.get("pp_tps") for k, v in out["cold"].items()})
    if out["warm"]:
        out["summary"].update({"warm_ttft": out["warm"]["ttft"], "warm_tps": out["warm"]["tps"]})
    out["summary"]["requests_ok"] = len(b.requests) - len(b.failures)
    out["summary"]["requests_failed"] = len(b.failures)
    out["requests"], out["failures"] = b.requests, b.failures
    path = a.out or os.path.join("results", a.name + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("SUMMARY", a.name, json.dumps(out["summary"]), flush=True)
    if b.failures:
        print(f"{len(b.failures)} request(s) FAILED; they are excluded from the summary. See 'failures' in {path}",
              flush=True)
    return 2 if warmup_failed else (1 if b.failures else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Strata tuning benchmark (see the module docstring).")
    ap.add_argument("--url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", help="model id the server answers to")
    ap.add_argument("--name", help="run name (default output: results/NAME.json)")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--runs", type=positive_int, default=2)
    ap.add_argument("--quick", action="store_true", help="decode only (no long-context fixture needed)")
    ap.add_argument("--selftest", action="store_true", help="run the offline tests in test_bench_stream.py and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        import test_bench_stream
        return test_bench_stream.main()
    missing = [f"--{k}" for k in ("url", "model", "name") if not getattr(a, k)]
    if missing:
        ap.error("required: " + ", ".join(missing))
    return run_bench(a)


if __name__ == "__main__":
    sys.exit(main())
