#!/usr/bin/env python3
"""Real request sizes, from the engine's own per-request lines.

Why this exists: the evaluation protocol used to pin "full context" at 128K, which is the model's context
ceiling - not what anyone actually sends. The rungs must come from measured usage, so this reads the one
place the numbers already exist and turns them into percentiles.

Where the lines come from (no content, only counts):

    # raw engine log (`--log <file>`); the line the engine writes when a request finishes
    prompt 12288 tokens = 4096 reused + 8192 read in 1810.2 ms (4522.1 tok/s), 96 generated in 761.0 ms (126.1 tok/s)

    # serve, with STRATA_REQUEST_LINES=1 (serve/server.py:690 restates the same numbers on stdout)
    [strata] request prompt 12288 cached 4096 output 96 prompt_read 1810 ms total 2571 ms prefill 4522 tok/s decode 126 tok/s

`prompt` is the whole context the request needed (cached prefix + newly read), so that is the number a
training step would have to hold. `reused` is only the part the conversation cache already had.

Usage:
    python usage.py engine.log serve.out --json usage.json --device "4090-48g"
    python usage.py --min-samples 200 engine.log      # exits 3 when there is not enough to be a percentile

Exit: 0 = percentiles written, 3 = too few samples (the caller must not invent a number), 2 = no lines found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Both shapes of the same event. Kept tolerant of the "cancelled mid-prompt" form (`+ 12288 of 98179 read`).
_ENGINE = re.compile(
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms")
_ECHO = re.compile(
    r"\[strata\] request prompt (?P<prompt>\d+) cached (?P<reused>\d+) output (?P<out>\d+)")

# Buckets chosen to line up with the KV ladder (fp16 <=32K, int8 <=128K, k8v4 <=256K), not with powers of two
# for their own sake: a bucket boundary that is also a KV switch is the boundary someone has to act on.
BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def scan(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _ENGINE.search(line) or _ECHO.search(line)
                if not m:
                    continue
                d = m.groupdict()
                rows.append({"prompt": int(d["prompt"]), "reused": int(d["reused"]),
                             "output": int(d.get("out") or 0), "src": path})
    return rows


def pct(values: list[int], q: float) -> int:
    """Linear interpolation on the sorted sample; q in [0, 1]."""
    if not values:
        return 0
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return int(round(s[lo] + (s[hi] - s[lo]) * (pos - lo)))


def histogram(values: list[int]) -> list[dict]:
    out, lo = [], 0
    for hi in BUCKETS:
        n = sum(1 for v in values if lo <= v < hi)
        out.append({"from": lo, "to": hi, "n": n})
        lo = hi
    out.append({"from": lo, "to": None, "n": sum(1 for v in values if v >= lo)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="engine log files or serve stdout captures")
    ap.add_argument("--device", default="", help="tag this distribution with the machine it came from")
    ap.add_argument("--json", default="", help="write the summary here")
    ap.add_argument("--min-samples", type=int, default=200,
                    help="below this, no percentiles are reported (a percentile of 12 requests is noise)")
    a = ap.parse_args()

    rows = scan(a.logs)
    if not rows:
        print("no request lines found. Engine log needs --log <file>; serve needs STRATA_REQUEST_LINES=1.",
              file=sys.stderr)
        return 2

    prompts = [r["prompt"] for r in rows]
    summary = {
        "device": a.device,
        "requests": len(rows),
        "sources": sorted({r["src"] for r in rows}),
        "prompt_tokens": {
            "p50": pct(prompts, 0.50), "p90": pct(prompts, 0.90), "p99": pct(prompts, 0.99),
            "max": max(prompts), "mean": int(sum(prompts) / len(prompts)),
        },
        "reused_share": round(sum(r["reused"] for r in rows) / max(1, sum(prompts)), 3),
        "histogram": histogram(prompts),
        "sufficient": len(rows) >= a.min_samples,
    }

    print(f"device      : {a.device or '(untagged)'}")
    print(f"requests    : {len(rows)}  (need >= {a.min_samples} for percentiles)")
    t = summary["prompt_tokens"]
    print(f"prompt tok  : p50 {t['p50']}  p90 {t['p90']}  p99 {t['p99']}  max {t['max']}  mean {t['mean']}")
    print(f"prefix reuse: {summary['reused_share'] * 100:.1f}% of prompt tokens came from the conversation cache")
    print("histogram   :")
    for b in summary["histogram"]:
        label = f"[{b['from']}, {b['to']})" if b["to"] else f"[{b['from']}, inf)"
        share = b["n"] / len(rows)
        print(f"  {label:>18}  {b['n']:6d}  {share * 100:5.1f}%  {'#' * int(share * 40)}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"written     : {a.json}")

    if not summary["sufficient"]:
        print(f"\nNOT ENOUGH SAMPLES ({len(rows)} < {a.min_samples}): do not use these as rungs. "
              f"Keep collecting; the model ceiling is not a substitute.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
