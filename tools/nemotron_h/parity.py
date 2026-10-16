"""tools/nemotron_h/parity.py - strata-nh against llama.cpp on Nemotron-H MoE (Nemotron 3.5 Super VL 120B-A12B).

Per prompt: llama.cpp (nh_llama_dump, -cmoe) tokenizes it and answers greedily; then on that one sequence (prompt +
llama.cpp's answer, teacher-forced) both engines dump every step's logits, and the report is KL(llama || strata) and
the top-1 agreement per step.  --layers also dumps the prompt's 88 sublayer outputs from both and prints each
sublayer's relative error (|x_strata - x_llama| / |x_llama| over the prompt's positions), which says WHERE a
divergence starts.  llama.cpp's own noise floor (its CPU backend against its CUDA one) is --floor: slow (an 83 GB
CPU forward), so off by default.

    python tools/nemotron_h/parity.py --dump BIN --nh BIN --model GGUF --out DIR [--n 48] [--layers] [--floor]
        [--prompts tools/nemotron_h/parity_prompts.json] [--nh-args "--expert-slots 64"]
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

V = 131072
EOS = 11                                        # <|im_end|>
N_EMBD = 4096


def ids_of(text: str, key: str) -> list[int]:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no {key}: in the output")


def run(cmd: list[str], log: Path) -> str:
    print("  $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed (rc {r.returncode}), see {log}")
    return r.stdout


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def chat(text: str) -> str:
    """The model's chat template for one user turn, thinking off (the GGUF template's <think></think> prefill)."""
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def layer_report(ref: Path, got: Path, n_tok: int, n_layer: int) -> None:
    want = n_tok * n_layer * N_EMBD               # the prompt's rows (the forced steps' follow)
    A = np.fromfile(ref, dtype=np.float32, count=want)
    B = np.fromfile(got, dtype=np.float32, count=want)
    if A.size != want or B.size != want:
        print(f"  layer dumps: short files {A.size} / {B.size} (want {want})")
        return
    A = A.reshape(n_tok, n_layer, N_EMBD)
    B = B.reshape(n_tok, n_layer, N_EMBD)
    print("  sublayer  rel.err(mean over positions)  max")
    for l in range(n_layer):
        a, b = A[:, l], B[:, l]
        keep = np.abs(a).sum(axis=1) > 0           # llama.cpp keeps only the output rows of its last sublayer
        if not keep.any():
            continue
        e = np.linalg.norm(a[keep] - b[keep], axis=1) / np.maximum(np.linalg.norm(a[keep], axis=1), 1e-12)
        print(f"  {l:3d}       {e.mean():.5f}                       {e.max():.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="nh_llama_dump (built against llama.cpp with Nemotron-H support)")
    ap.add_argument("--nh", required=True, help="strata-nh")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(Path(__file__).with_name("parity_prompts.json")))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--nh-args", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    K: dict[str, list] = {"strata": [], "cpu": []}
    T1: dict[str, list] = {"strata": [], "cpu": []}
    for i, text in enumerate(prompts):
        tag = f"p{i}"
        print(f"{tag}: {text[:60]!r}", flush=True)
        o = run([a.dump, "--model", a.model, "--prompt", chat(text), "--n", str(a.n)], out / f"{tag}.llama.log")
        prompt, gen = ids_of(o, "prompt"), ids_of(o, "gen")
        toks, force = ",".join(map(str, prompt)), ",".join(map(str, gen[:-1]))
        f = {k: out / f"{tag}.{k}.full" for k in ("gpu", "cpu", "strata")}
        if not f["gpu"].exists():
            run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--dump-full-logits", str(f["gpu"])] +
                (["--dump-layers", str(out / f"{tag}.llama.layers")] if a.layers else []), out / f"{tag}.gpu.log")
        if a.floor and not f["cpu"].exists():
            run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--ngl", "0", "--dump-full-logits",
                 str(f["cpu"])], out / f"{tag}.cpu.log")
        so = run([a.nh, "--model", a.model, "--tokens", toks, "--force", force, "--dump-full-logits", str(f["strata"])] +
                 (["--dump-layers", str(out / f"{tag}.strata.layers")] if a.layers else []) + shlex.split(a.nh_args),
                 out / f"{tag}.strata.log")
        sgen = ids_of(so, "gen")
        P = log_softmax(np.fromfile(f["gpu"], dtype=np.float32).reshape(-1, V))
        line = f"  {len(P)} steps; strata's own first picks {'match' if sgen[:1] == gen[:1] else 'DIFFER'}"
        for k in ("strata", "cpu"):
            if not f[k].exists():
                continue
            Q = log_softmax(np.fromfile(f[k], dtype=np.float32).reshape(-1, V))
            n = min(len(P), len(Q))
            d = (np.exp(P[:n]) * (P[:n] - Q[:n])).sum(axis=1)
            t1 = P[:n].argmax(axis=1) == Q[:n].argmax(axis=1)
            K[k].extend(d.tolist())
            T1[k].extend(t1.tolist())
            line += f"; {k}: KL {d.mean():.5f} (max {d.max():.4f}) top-1 {t1.mean():.3f}"
        print(line, flush=True)
        if a.layers:
            layer_report(out / f"{tag}.llama.layers", out / f"{tag}.strata.layers", len(prompt), 88)
    print("-- summary")
    for k in ("strata", "cpu"):
        if K[k]:
            print(f"{k:6s} vs llama.cpp CUDA: mean KL {np.mean(K[k]):.5f} (median {np.median(K[k]):.6f}, p99 "
                  f"{np.percentile(K[k], 99):.4f}), top-1 {np.mean(T1[k]):.4f} over {len(K[k])} steps")
    if K["cpu"]:
        print(f"ratio strata / floor: {np.mean(K['strata']) / max(np.mean(K['cpu']), 1e-12):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
