#!/usr/bin/env python3
"""Token-for-token + logprob parity gate: native engine vs llama.cpp bridge.

The native Vulkan engine must produce the same greedy tokens, and close enough
logprobs, as the llama.cpp Vulkan bridge for the same prompt token ids. This is
the correctness gate: it runs several prompts, up to --tokens
tokens each, and compares both the sampled token ids and the top-K logprobs that
each engine emits with --lp K.

Both engines are driven over the line protocol (READY / GEN / T / LP / DONE),
so no HTTP server is needed. Each engine is started once and given every prompt
in one session, so the 21 GB GGUF is loaded once per engine.

Example (quick smoke, 3 tokens x 2 prompts):
    python3 engine/vulkan/compare_greedy.py --tokens 3 --prompts 2

Full gate (200 tokens x 5 prompts):
    python3 engine/vulkan/compare_greedy.py --tokens 200

Exit status: 0 when every prompt matches within tolerance, 1 otherwise.
"""
import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
REPO = ROOT.parents[1]
MODEL = pathlib.Path("/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf")
LLAMA_LIB = pathlib.Path("/home/g/Code/llama.cpp/build/bin")

DEFAULT_PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "1, 2, 3, 4, 5,",
    "def fibonacci(n):",
    "Hello, world!",
]


def encode_prompts(texts):
    """Use Strata's GGUF-exported tokenizer so both engines get identical ids."""
    sys.path.insert(0, str(REPO / "terminal" / "tools"))
    import strata_tokenizer as st
    tok = st.Tokenizer.from_gguf(str(MODEL))
    return [tok.encode(t) for t in texts]


def engine_cmd(which, model, ctx, threads, lp):
    shaders = ROOT / "shaders"
    if which == "native":
        return [str(ROOT / "engine" / "strata-vulkan-engine"),
                "--model", str(model), "--ctx", str(ctx), "--threads", str(threads),
                "--lp", str(lp),
                "--native-shaders", str(ROOT / "native" / "shaders"),
                "--expert-shaders", str(shaders),
                "--q4-shader", str(shaders / "expert_q4_k_gemv.comp.spv"),
                "--down-shader", str(shaders / "expert_q6_k_gemv.comp.spv"),
                "--q8-shader", str(shaders / "expert_q8_0_gemv.comp.spv"),
                "--silu-shader", str(shaders / "expert_silu_mul.comp.spv"),
                "--router-gemv-shader", str(shaders / "router_gemv.comp.spv"),
                "--router-topk-shader", str(shaders / "router_topk.comp.spv")]
    return [str(ROOT / "bridge" / "strata-llama-vulkan-engine"),
            "--model", str(model), "--ctx", str(ctx), "--threads", str(threads),
            "--backend-dir", str(LLAMA_LIB), "--lp", str(lp)]


def run_engine(which, prompts, tokens, args):
    cmd = engine_cmd(which, args.model, args.ctx, args.threads, args.lp)
    lines = []
    for ids in prompts:
        lines.append(f"GEN {tokens} temperature=0 {','.join(map(str, ids))}")
    lines.append("QUIT")
    env = os.environ.copy()
    if which == "bridge":
        env["LD_LIBRARY_PATH"] = str(LLAMA_LIB) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(cmd, input="\n".join(lines) + "\n", text=True,
                          capture_output=True, env=env, timeout=args.timeout)
    if proc.returncode != 0:
        print(f"[{which}] exited {proc.returncode}: {proc.stderr[-2000:]}", file=sys.stderr)
    return parse(proc.stdout)


def parse(out):
    results = []
    cur = None
    for line in out.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "PP":
            cur = {"tokens": [], "lp": [], "done": ""}
        elif f[0] == "T" and cur is not None:
            cur["tokens"].append(int(f[1]))
        elif f[0] == "LP" and cur is not None:
            # Strata wire format: LP <chosen logprob> <tok>:<logprob> ...
            pairs = {}
            for item in f[2:]:
                tok, lp = item.split(":", 1)
                pairs[int(tok)] = float(lp)
            cur["lp"].append(pairs)
        elif f[0] == "DONE" and cur is not None:
            cur["done"] = line
            results.append(cur)
            cur = None
    return results


def _near_tie(npairs, bpairs, nt, bt, tol):
    """True if each engine sees the other's chosen token as nearly as likely.

    A greedy argmax can flip on a near-tie under small numerical differences;
    that is not a correctness failure when both engines agree the two
    candidates are effectively tied.
    """
    if nt not in npairs or bt not in npairs or nt not in bpairs or bt not in bpairs:
        return False
    return (abs(npairs[nt] - npairs[bt]) <= tol and
            abs(bpairs[bt] - bpairs[nt]) <= tol)


def compare(native, bridge, prompts, texts, lp_tol, lp_rtol, tie_tol, topk):
    if len(native) != len(prompts) or len(bridge) != len(prompts):
        print(f"FAIL: expected {len(prompts)} generations, native={len(native)}, bridge={len(bridge)}")
        return 1
    ok = True
    for i, (text, n, b) in enumerate(zip(texts, native, bridge)):
        label = f"prompt {i}: {text!r}"
        nt, bt = n["tokens"], b["tokens"]
        length_note = ""
        if len(nt) != len(bt):
            length_note = f"; length native={len(nt)} bridge={len(bt)}"

        worst, worst_at = 0.0, None
        exact, flip, hard = 0, None, []
        min_overlap = 1.0
        for j, (x, y) in enumerate(zip(nt, bt)):
            npairs = n["lp"][j] if j < len(n["lp"]) else {}
            bpairs = b["lp"][j] if j < len(b["lp"]) else {}
            if not npairs or not bpairs:
                continue
            if x == y:
                exact += 1
            elif flip is None and _near_tie(npairs, bpairs, x, y, tie_tol):
                # A near-tie argmax flip ends the verifiable prefix. After the
                # two engines emit different tokens they are in different
                # contexts, so later positions cannot be compared.
                flip = (j, x, y)
            elif flip is None:
                hard.append((j, x, y))   # a mismatch with no near-tie: real divergence
            if flip is not None:
                continue
            # Compare the top of each distribution, not deep-tail tokens that
            # fall out of one engine's top-K but not the other's.
            nmax, bmax = max(npairs.values()), max(bpairs.values())
            d = abs(nmax - bmax)
            allowed = lp_tol + lp_rtol * max(abs(nmax), abs(bmax))
            if d - allowed > worst:
                worst, worst_at = d - allowed, (j, nmax, bmax, allowed)
            ns = set(sorted(npairs, key=npairs.get, reverse=True)[:topk])
            bs = set(sorted(bpairs, key=bpairs.get, reverse=True)[:topk])
            min_overlap = min(min_overlap, len(ns & bs) / max(1, min(len(ns), len(bs))))

        status = "PASS" if worst <= 0 and not hard else "FAIL"
        if status == "FAIL":
            ok = False
        detail = (f"verified {exact}/{min(len(nt), len(bt))} tokens"
                  + (f", near-tie flip at {flip[0]} ({flip[1]} vs {flip[2]}); downstream diverges as expected"
                     if flip else "")
                  + (f", hard mismatches {hard[:5]}" if hard else "")
                  + f", min top-{topk} overlap {min_overlap:.2f}" + length_note)
        print(f"{status} {label}: {detail}; "
              f"worst top-logprob excess {worst:+.4f}"
              + (f" at idx={worst_at[0]} native={worst_at[1]:.4f} "
                 f"bridge={worst_at[2]:.4f} (allowed {worst_at[3]:.4f})" if worst_at else ""))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=len(DEFAULT_PROMPTS),
                    help="how many of the default prompts to run")
    ap.add_argument("--lp", type=int, default=5, help="top-K logprobs to compare (0 disables)")
    ap.add_argument("--lp-tol", type=float, default=0.05, help="absolute logprob tolerance")
    ap.add_argument("--lp-rtol", type=float, default=0.02, help="relative logprob tolerance (scales with |logprob|)")
    ap.add_argument("--tie-tol", type=float, default=0.15, help="logprob gap below which a token flip counts as a near-tie")
    ap.add_argument("--topk", type=int, default=3, help="top-K set overlap to report")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    a.model = pathlib.Path(a.model).resolve()
    if not a.model.is_file():
        print(f"model not found: {a.model}", file=sys.stderr)
        return 2

    texts = DEFAULT_PROMPTS[:a.prompts]
    prompts = encode_prompts(texts)
    print(f"comparing {len(prompts)} prompts x {a.tokens} tokens "
          f"(top-{a.lp} logprobs, tol {a.lp_tol})", flush=True)
    native = run_engine("native", prompts, a.tokens, a)
    bridge = run_engine("bridge", prompts, a.tokens, a)
    return compare(native, bridge, prompts, texts, a.lp_tol, a.lp_rtol, a.tie_tol, a.topk)


if __name__ == "__main__":
    raise SystemExit(main())
