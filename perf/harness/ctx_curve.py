#!/usr/bin/env python3
"""Decode rate vs context length, cold prefix (no prompt-cache reuse).

usage: ctx_curve.py --sizes 16000,32000,64000,128000,198000 [--max-tokens 192]
Env STRATA_KEY is required.
"""
import argparse, json, os, secrets, statistics, sys, time, urllib.request

FILLER = (
    "The decode path of a mixture-of-experts transformer touches only the routed experts for each token. "
    "Effective bandwidth is therefore the product of the expert read size and the number of lookups per forward pass, "
    "divided by the wall clock of that pass. Grouping lookups by expert id turns scattered single-block reads into "
    "contiguous multi-block reads, which is what raises the achievable bandwidth on a GPU with a narrow compute budget. "
    "Speculative decoding changes the economics in the other direction: every accepted draft token is a token produced "
    "by a forward pass that was already paid for, so throughput scales with tokens per forward, not with bandwidth alone. "
    "A verification window of width w reads the union of the experts of its w candidates, so the marginal cost of an "
    "extra position is the extra experts that position introduces, not a full pass over the active set. "
    "The GDN recurrence carries a state that must be updated once per accepted token in order, while attention over the "
    "kv cache is a function of the sequence position and grows with context. "
)


def build_prompt(target_tokens, factor):
    # a unique head defeats the prefix cache: reuse needs an identical token prefix
    head = " ".join(secrets.token_hex(8) for _ in range(48)) + "\n"
    out, i, need = [head], 0, int(target_tokens * factor)
    while sum(len(x) for x in out) < need:
        out.append("Section %d. %s" % (i, FILLER))
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16000,32000,64000,128000,198000")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    key = os.environ["STRATA_KEY"]

    factor = 4.1
    for target in [int(x) for x in args.sizes.split(",")]:
        for rep in range(args.reps):
            prompt = build_prompt(target, factor) + "\n\nContinue with more technical detail."
            payload = {"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": args.max_tokens, "temperature": 0,
                       "stream": False, "cache_prompt": False}
            req = urllib.request.Request(
                "http://127.0.0.1:%d/v1/chat/completions" % args.port,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=2400) as r:
                    out = json.loads(r.read().decode())
            except Exception as e:
                print(json.dumps({"target": target, "rep": rep, "error": str(e)[:200],
                                  "wall_s": round(time.time() - t0, 1)}), flush=True)
                continue
            wall = time.time() - t0
            t = out.get("timings") or {}
            usage = (out.get("usage") or {})
            gen = t.get("predicted_n") or 0
            acc = t.get("draft_n_accepted")
            fwd = gen - acc if (acc is not None and gen) else None
            got = t.get("prompt_n") or usage.get("prompt_tokens") or 0
            row = {
                "target": target, "rep": rep,
                "prompt_n": got,
                "chars": len(prompt),
                "prompt_ms": t.get("prompt_ms"),
                "prompt_tps": t.get("prompt_per_second"),
                "gen": gen,
                "decode_ms": t.get("predicted_ms"),
                "tok_s": t.get("predicted_per_second"),
                "ms_tok": t.get("predicted_per_token_ms"),
                "draft_n": t.get("draft_n"), "accepted": acc,
                "acc_pct": round(100.0 * acc / t["draft_n"], 1) if (acc is not None and t.get("draft_n")) else None,
                "forwards": fwd,
                "tok_per_fwd": round(gen / fwd, 3) if fwd else None,
                "ms_per_fwd": round(t["predicted_ms"] / fwd, 2) if fwd else None,
                "wall_s": round(wall, 1),
            }
            print(json.dumps(row), flush=True)
            if got > 2000:
                factor = max(2.0, min(8.0, factor * target / got))
    print("== factor used %.3f" % factor, file=sys.stderr)


if __name__ == "__main__":
    main()
