#!/usr/bin/env python3
"""Routing-trace corpus / prefetch gate.

Reads traces written by the native engine's ``--trace FILE`` option (one line
per layer per token: ``layer id0 id1 ... id7``) and measures the two overlaps a
prefetch heuristic could exploit:

* **consecutive layers** inside one token — what ``prefetch_route(l+1,
  ids_from_l)`` relies on;
* **same layer across tokens** — what a "prefetch next token from this token's
  route" heuristic would rely on.

A checked-in corpus from the B390 is in ``traces/``. Running the tool is
informational by default; ``--fail-under 0.25`` turns it into a gate.

    python3 engine/vulkan/check_prefetch.py
    python3 engine/vulkan/check_prefetch.py --trace /tmp/opencode/route.trace --json
    python3 engine/vulkan/check_prefetch.py --fail-under 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TRACES = sorted((HERE / "traces").glob("*.trace"))


def parse_trace(text: str) -> list[list[tuple[int, list[int]]]]:
    """Split a trace into tokens. A layer-0 line starts a new token."""
    tokens: list[list[tuple[int, list[int]]]] = []
    current: list[tuple[int, list[int]]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            layer, ids = int(parts[0]), [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if layer == 0 and current:
            tokens.append(current)
            current = []
        current.append((layer, ids))
    if current:
        tokens.append(current)
    return tokens


def _overlap(a: list[int], b: list[int]) -> float:
    """Fraction of b's experts that were also in a (the cache-hit fraction)."""
    return len(set(a) & set(b)) / max(1, len(b))


def analyze(texts: list[str]) -> dict:
    tokens = [token for text in texts for token in parse_trace(text)]
    consecutive: list[float] = []
    same_layer: list[float] = []
    for token in tokens:
        for (l0, prev), (l1, cur) in zip(token, token[1:]):
            if l1 == l0 + 1:
                consecutive.append(_overlap(prev, cur))
    for first, second in zip(tokens, tokens[1:]):
        for (l0, ids0), (l1, ids1) in zip(first, second):
            if l0 == l1:
                same_layer.append(_overlap(ids0, ids1))
    layers = sorted({layer for token in tokens for layer, _ in token})
    return {
        "traces": len(texts), "tokens": len(tokens), "layers_per_token": len(layers),
        "consecutive_pairs": len(consecutive), "same_layer_pairs": len(same_layer),
        "mean_consecutive_overlap": sum(consecutive) / len(consecutive) if consecutive else None,
        "min_consecutive_overlap": min(consecutive) if consecutive else None,
        "max_consecutive_overlap": max(consecutive) if consecutive else None,
        "mean_same_layer_overlap": sum(same_layer) / len(same_layer) if same_layer else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", nargs="*", default=None,
                    help="trace files (default: every *.trace in engine/vulkan/traces)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="exit 1 when mean consecutive-layer overlap is below this fraction")
    a = ap.parse_args()
    paths = [Path(p) for p in a.trace] if a.trace else DEFAULT_TRACES
    if not paths:
        print("no trace data: pass --trace FILE or put a .trace in engine/vulkan/traces")
        return 1
    texts = []
    for path in paths:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            print(f"cannot read {path}: {exc}")
            return 1
    report = analyze(texts)
    report["files"] = [str(p) for p in paths]
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"files={len(paths)} tokens={report['tokens']} layers/token={report['layers_per_token']} "
              f"pairs={report['consecutive_pairs']}")
        for label, key in (("consecutive layers", "mean_consecutive_overlap"),
                           ("same layer, next token", "mean_same_layer_overlap")):
            value = report[key]
            print(f"{label:<24} {value:.1%}" if value is not None else f"{label:<24} n/a")
    if a.fail_under is not None and report["mean_consecutive_overlap"] is not None:
        if report["mean_consecutive_overlap"] < a.fail_under:
            if not a.json:
                print(f"FAIL: one-layer prefetch overlap {report['mean_consecutive_overlap']:.1%} "
                      f"< {a.fail_under:.1%}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
