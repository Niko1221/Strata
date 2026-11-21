#!/usr/bin/env python3
"""Teacher-forcing parity gate: native engine vs llama.cpp bridge, position by position.

GREEDY decoding can diverge on a single near-tie and then cascade into a totally
different sequence (see compare_greedy.py). That says nothing about whether the
two FORWARD passes agree. This script removes decoding from the picture: it
feeds the *same* token sequence to both engines with the EVAL verb, which emits
top-K logprobs after every position without sampling, and compares the two
forward traces directly over --tokens positions.

The reference sequence is the bridge's own greedy output for one prompt, so the
positions cover realistic states (attention, GDN, streamed experts).

    python3 engine/vulkan/compare_forward.py --tokens 200

Exit status: 0 when top-1 logprobs agree within tolerance at every position,
1 otherwise.
"""
import argparse
import os
import pathlib
import subprocess
import sys

from compare_greedy import (
    DEFAULT_PROMPTS, LLAMA_LIB, MODEL, encode_prompts, engine_cmd,
)

HERE = pathlib.Path(__file__).resolve().parent


def _run(which, verb_line, a, env=None):
    cmd = engine_cmd(which, a.model, a.ctx, a.threads, a.lp)
    e = os.environ.copy()
    if which == "bridge":
        e["LD_LIBRARY_PATH"] = str(LLAMA_LIB) + os.pathsep + e.get("LD_LIBRARY_PATH", "")
    if env:
        e.update(env)
    proc = subprocess.run(cmd, input=verb_line + "\nQUIT\n", text=True,
                          capture_output=True, env=e, timeout=a.timeout)
    if proc.returncode != 0:
        print(f"[{which}] exited {proc.returncode}: {proc.stderr[-2000:]}", file=sys.stderr)
    return proc.stdout


def parse_lp(out):
    """Collect top-K logprob dicts from LP lines up to DONE."""
    lps = []
    for line in out.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "LP":
            pairs = {}
            for item in f[2:]:
                tok, lp = item.split(":", 1)
                pairs[int(tok)] = float(lp)
            lps.append(pairs)
        elif f[0] == "DONE":
            break
    return lps


def parse_tokens(out):
    toks = []
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == "T":
            toks.append(int(f[1]))
        elif f and f[0] == "DONE":
            break
    return toks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--prompt", default=DEFAULT_PROMPTS[0])
    ap.add_argument("--lp", type=int, default=5)
    ap.add_argument("--lp-tol", type=float, default=0.05)
    ap.add_argument("--lp-rtol", type=float, default=0.02)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    a.model = pathlib.Path(a.model).resolve()
    if not a.model.is_file():
        print(f"model not found: {a.model}", file=sys.stderr)
        return 2

    prompt_ids = encode_prompts([a.prompt])[0]
    print(f"reference: bridge greedy, {a.tokens} tokens for {a.prompt!r}", flush=True)
    ref = parse_tokens(_run("bridge", f"GEN {a.tokens} temperature=0 {','.join(map(str, prompt_ids))}", a))
    if len(ref) < a.tokens:
        print(f"WARNING: reference stopped early at {len(ref)} tokens (EOS)")
    seq = prompt_ids + ref
    print(f"evaluating {len(seq)} positions on both engines (top-{a.lp} logprobs)", flush=True)

    native = parse_lp(_run("native", f"EVAL {','.join(map(str, seq))}", a))
    bridge = parse_lp(_run("bridge", f"EVAL {','.join(map(str, seq))}", a))
    if len(native) != len(seq) or len(bridge) != len(seq):
        print(f"FAIL: expected {len(seq)} LP lines, native={len(native)} bridge={len(bridge)}")
        return 1

    worst, worst_at = 0.0, None
    agree = 0
    for pos, (n, b) in enumerate(zip(native, bridge)):
        nmax, bmax = max(n.values()), max(b.values())
        if max(n, key=n.get) == max(b, key=b.get):
            agree += 1
        d = abs(nmax - bmax)
        allowed = a.lp_tol + a.lp_rtol * max(abs(nmax), abs(bmax))
        if d - allowed > worst:
            worst, worst_at = d - allowed, (pos, nmax, bmax, allowed)
    pct = 100.0 * agree / len(seq)
    status = "PASS" if worst <= 0 else "FAIL"
    print(f"{status}: {len(seq)} positions, top-1 token agrees {agree}/{len(seq)} ({pct:.1f}%); "
          f"worst top-logprob excess {worst:+.4f}"
          + (f" at pos={worst_at[0]} native={worst_at[1]:.4f} bridge={worst_at[2]:.4f} "
             f"(allowed {worst_at[3]:.4f})" if worst_at else ""))
    return 0 if worst <= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
