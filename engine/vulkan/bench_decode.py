#!/usr/bin/env python3
"""Decode/prompt speed of the two Vulkan engines on the same prompt.

Runs the native engine and/or the llama.cpp bridge over the line protocol and
reads the engine's own `PP` and `DONE` timings, so the numbers are measured
inside the engine, not around Python. One process per engine, one prompt, one
run per engine by default; use --repeat for A/B pairs and report the median.

    python3 engine/vulkan/bench_decode.py --tokens 32
    python3 engine/vulkan/bench_decode.py --engines native,bridge --repeat 3

Both engines load the same 21 GB GGUF; a fresh process pays that cost each time.
Run with --lp 0 (the default) so the test-only logprob lines do not add work.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_greedy import MODEL, encode_prompts, engine_cmd, LLAMA_LIB  # noqa: E402


def run_once(which: str, ids: list[int], args) -> dict:
    cmd = engine_cmd(which, args.model, args.ctx, args.threads, 0)
    lines = [f"GEN {min(args.tokens, 8)} temperature=0 {','.join(map(str, ids))}"
             for _ in range(max(0, getattr(args, "warmup", 0)))]
    lines += [f"GEN {args.tokens} temperature=0 {','.join(map(str, ids))}", "QUIT"]
    line = "\n".join(lines) + "\n"
    env = os.environ.copy()
    if which == "bridge":
        env["LD_LIBRARY_PATH"] = str(LLAMA_LIB) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(cmd, input=line, text=True, capture_output=True, env=env,
                          timeout=args.timeout)
    out = {"engine": which, "returncode": proc.returncode}
    for text in proc.stdout.splitlines():
        fields = text.split()
        if fields and fields[0] == "PP" and len(fields) >= 4:
            out["prompt_tokens"] = int(fields[1])
            out["prompt_ms"] = float(fields[3])
        elif fields and fields[0] == "DONE" and len(fields) >= 5:
            out["generated"] = int(fields[1])
            out["decode_ms"] = float(fields[4])
            out["finish"] = fields[5] if len(fields) > 5 else ""
    if proc.returncode != 0:
        out["error"] = proc.stderr[-500:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--prompt", default="What is 2+2?")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--engines", default="native", help="comma-separated: native,bridge")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1, help="GEN requests before the measured one, in the same process (warms the expert cache)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", help="write the full measurements here as JSON")
    a = ap.parse_args()
    ids = encode_prompts([a.prompt])[0]
    print(f"prompt: {a.prompt!r} -> {len(ids)} tokens; requested {a.tokens} new tokens")
    print(f"{'engine':>8} {'prompt tok/s':>13} {'decode tok/s':>13} {'tokens':>7} {'finish':>9}")
    all_runs = []
    for which in (x.strip() for x in a.engines.split(",") if x.strip()):
        runs = []
        for _ in range(max(1, a.repeat)):
            r = run_once(which, ids, a)
            all_runs.append(r)
            if "error" in r or "decode_ms" not in r:
                print(f"{which:>8}  ERROR: {r.get('error', 'no DONE line')[:120]}")
                continue
            runs.append(r)
        if not runs:
            continue
        p_tps = [r["prompt_tokens"] / (r["prompt_ms"] / 1000) for r in runs if r["prompt_ms"] > 0]
        d_tps = [r["generated"] / (r["decode_ms"] / 1000) for r in runs if r["decode_ms"] > 0 and r["generated"]]
        p_med = statistics.median(p_tps) if p_tps else 0
        d_med = statistics.median(d_tps) if d_tps else 0
        print(f"{which:>8} {p_med:>13.1f} {d_med:>13.1f} {runs[-1]['generated']:>7} {runs[-1].get('finish',''):>9}")
    if a.out:
        Path(a.out).write_text(json.dumps({"prompt": a.prompt, "prompt_ids": ids,
                                           "tokens": a.tokens, "runs": all_runs}, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0 if all("error" not in r for r in all_runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
