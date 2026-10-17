#!/usr/bin/env python3
"""The DFlash feature-capture gate (docs/DFLASH.md): run the engine twice over a shared
deterministic prompt and compare the contracted HC residuals the two kernel paths capture for the
boundary position.

    python tools/dflash_taps.py --engine ./build/strata --pack PACK --native SHARD1 --ple-gguf SHARD2 \
        [--expert-profile FILE] [--prompt-tokens 400] [--prefill 256] [--spec 2]

Run A (N tokens) anchors its first verify window at position N-1: the window path's taps for the
prompt's last token.  Run B (N+1 tokens, same prefix) batches that same position through the prompt
path.  Both captures must agree per tap (within bf16-vs-f32 kernel tolerance) - a wrong layer, the
wrong side of the HC contraction, the wrong width, a token offset or a tap-ordering bug each leaves
a distinct per-tap signature here.  Needs a GPU and the target model; nothing else.

The reference-forward comparison against the PixelML exporter's taps is a separate harness (the
exporter needs the vLLM stack); this gate is Strata-internal by design.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

MAGIC = 0x31504644
BOUNDARIES = [4, 16, 24, 36, 44]   # the trained taps [3, 15, 23, 35, 43] + 1


def records(path: str):
    data = open(path, "rb").read()
    at = 0
    while at < len(data):
        magic, source, pos0, T, n_taps, n_embd, flags = struct.unpack_from("<7I", data, at)
        if magic != MAGIC:
            raise SystemExit(f"bad magic {magic:#x} at byte {at}")
        at += 28
        rows = np.frombuffer(data, "<f4", n_taps * T * n_embd, at).reshape(n_taps, T, n_embd)
        at += rows.nbytes
        yield {"source": source, "pos0": pos0, "T": T, "taps": rows, "bf16": bool(flags)}


def run_engine(args, n_tokens, out) -> None:
    rng = np.random.default_rng(7)
    tokens = ",".join(str(int(x)) for x in rng.integers(1000, 240000, size=n_tokens))
    env = dict(os.environ, STRATA_DFLASH_TAPS=out)
    cmd = [args.engine, "--pack", args.pack, "--native", args.native]
    if args.ple_gguf:
        cmd += ["--ple-gguf", args.ple_gguf]
    if args.expert_profile:
        cmd += ["--expert-profile", args.expert_profile]
    cmd += ["--prefill", str(args.prefill), "--spec", str(args.spec), "--max-new", "2", "--tokens", tokens]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit(f"engine run (n={n_tokens}) failed; see the log above")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./build/strata")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--ple-gguf", default=None)
    ap.add_argument("--expert-profile", default=None)
    ap.add_argument("--prompt-tokens", type=int, default=400)
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--spec", type=int, default=2)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        fa, fb = os.path.join(d, "a.bin"), os.path.join(d, "b.bin")
        n = args.prompt_tokens
        print(f"run A: {n} tokens (the window anchors at {n - 1}) ...")
        run_engine(args, n, fa)
        print(f"run B: {n + 1} tokens (the prompt path batches position {n - 1}) ...")
        run_engine(args, n + 1, fb)

        wa = [r for r in records(fa) if r["source"] == 1]
        pb = [r for r in records(fb) if r["source"] == 0]
        if not wa or not pb:
            raise SystemExit("fixture records missing a source")
        w = wa[0]                      # the first window: row 0 = position n-1
        p = next(r for r in pb if r["pos0"] <= n - 1 < r["pos0"] + r["T"])
        off = (n - 1) - p["pos0"]
        n_taps = w["taps"].shape[0]
        if p["taps"].shape[0] != n_taps:
            raise SystemExit("tap count differs between the runs")

        print(f"position {n - 1}: window row 0 (f32 path) vs prompt chunk at {p['pos0']} row {off} (bf16 path)")
        worst, ok = 0.0, True
        for t in range(n_taps):
            a = w["taps"][t, 0].astype(np.float64)
            b = p["taps"][t, off].astype(np.float64)
            cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
            err = np.abs(a - b)
            rel = float(err.mean() / max(np.abs(a).mean(), 1e-30))
            bnd = BOUNDARIES[t] if t < len(BOUNDARIES) else "?"
            print(f"  tap {t} (boundary {bnd}): cos {cos:.6f}  mean|diff| {err.mean():.3e}  "
                  f"max|diff| {err.max():.3e}  rel mean {rel:.2e}  |a| {np.linalg.norm(a):.1f}")
            worst = max(worst, 1.0 - cos)
            # The two kernel paths differ by real arithmetic (bf16 batched GEMMs on the prompt side,
            # multi-row AVX2 expert kernels in the window), measured at cos >= 0.996 on the IQ3_XXS
            # target; a wiring bug (wrong layer / half / side / width / offset / ordering) lands at
            # cos <= 0.7.  The gate sits in that gap.
            if cos < 0.99 or rel > 0.12:
                ok = False
        print("TAP-GATE " + ("PASS" if ok else "FAIL") + f"  (worst 1-cos {worst:.2e})")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
