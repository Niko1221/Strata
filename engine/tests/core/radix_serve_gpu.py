"""Compare native Q36 or HY4 greedy decoding across a resident-session eviction.

Usage: python engine/tests/core/radix_serve_gpu.py q36 BUILD/strata-q36 MODEL.gguf -- --expert-vram 6 --kv fp16
       python engine/tests/core/radix_serve_gpu.py hy4 BUILD/strata-hy4 MODEL.strata -- --engram ENGRAM.strata
"""
import argparse
import json
import random
import subprocess
import sys
import time


def generate(proc, prompt):
    start = time.perf_counter()
    proc.stdin.write(f"GEN 8 temperature=0 seed=41 {','.join(map(str, prompt))}\n")
    proc.stdin.flush()
    tokens = []
    first_ms = None
    for line in proc.stdout:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "ERR":
            raise RuntimeError(line.strip())
        if fields[0] == "T":
            if first_ms is None:
                first_ms = round((time.perf_counter() - start) * 1000, 2)
            tokens.append(int(fields[1]))
        if fields[0] == "DONE":
            if not tokens or fields[5] in ("nonfinite", "cancel"):
                raise RuntimeError(f"Incomplete generation: {line}")
            return dict(tokens=tokens, reused=int(fields[8]), ttft_ms=first_ms)
    raise RuntimeError("Engine exited during generation")


def run(args, extra, cache_mib):
    context_flag = "--max-context" if args.family == "q36" else "--ctx"
    command = [args.engine, "--serve", "--model", args.model, context_flag, str(args.prefix + 128),
               "--radix-cache-mib", str(cache_mib)]
    if args.family == "q36":
        command.extend(["--batch", "1"])
    command.extend(extra)
    rng = random.Random(4147)
    prefix = rng.choices(range(32, 256), k=args.prefix)
    # Short unrelated turns discard resident state without filling the host cache.
    prompts = [prefix + [41] * 24, [83] * 11, [87] * 11, [93] * 11,
               prefix + [53] * 24, prefix + [41] * 24]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        try:
            for line in proc.stdout:
                if line.startswith("READY"):
                    break
                if line.startswith("ERR"):
                    raise RuntimeError(line.strip())
            else:
                raise RuntimeError(f"Engine exited before READY: {proc.wait()}")
            return [generate(proc, prompt) for prompt in prompts]
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    sep = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("q36", "hy4"))
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("--prefix", type=int, default=512)
    parser.add_argument("--cache-mib", type=int, default=1024)
    args = parser.parse_args(sys.argv[1:sep])
    if args.prefix < 512 or args.prefix % 512:
        parser.error("--prefix must be a positive multiple of 512")
    disabled, enabled = (run(args, sys.argv[sep + 1:], budget) for budget in (0, args.cache_mib))
    print(json.dumps(dict(disabled=disabled, enabled=enabled), indent=2))
    if [row["tokens"] for row in enabled] != [row["tokens"] for row in disabled]:
        raise SystemExit("Generated token IDs differ")
    if disabled[-2]["reused"] or enabled[-2]["reused"] != args.prefix:
        raise SystemExit("Expected a portable prefix hit after resident-session eviction")
