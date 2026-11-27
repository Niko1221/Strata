#!/usr/bin/env python3
"""Decode benchmark for the Strata server on the host.

Runs N identical decode-heavy requests (fixed prompt, fixed max_tokens,
temperature 0) and prints per-run tok/s plus the median. The API key is
resolved silently (env var -> ~/.hermes/.env -> podman exec) and never printed.

    python3 bench_decode.py            # 5 runs, 256 tokens
    python3 bench_decode.py --runs 7 --max-tokens 128
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("STRATA_URL", "http://localhost:8090/v1/chat/completions")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq3_xxs")
PROMPT = ("Write a detailed, vivid description of a lighthouse at dawn, "
          "covering the sea, the light, the rocks and the keeper's routine.")


def resolve_key() -> str:
    key = os.environ.get("STRATA_API_KEY", "").strip()
    if not key:
        env = Path.home() / ".hermes/.env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("STRATA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not key and shutil.which("podman"):
        try:
            key = subprocess.run(
                ["podman", "exec", "strata", "cat", "/data/state/api-key"],
                capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            pass
    return key


def one_run(key: str, max_tokens: int) -> tuple[float, int, int]:
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    dt = time.perf_counter() - t0
    usage = data.get("usage", {})
    return dt, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    key = resolve_key()
    if not key:
        sys.exit("no API key found (set STRATA_API_KEY, add it to ~/.hermes/.env, or run this on the host)")
    results = []
    for i in range(a.runs):
        try:
            dt, out_tok, in_tok = one_run(key, a.max_tokens)
        except Exception as e:
            print(f"run {i + 1}: FAILED {e}")
            continue
        tps = out_tok / dt if dt else 0
        results.append(tps)
        print(f"run {i + 1}: {out_tok} tok in {dt:6.2f}s  = {tps:6.1f} tok/s  (prompt {in_tok})")
    if results:
        print(f"median: {statistics.median(results):.1f} tok/s over {len(results)} runs "
              f"({a.max_tokens}-token outputs, temperature 0)")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
