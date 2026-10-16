"""Compare JSON logits without executing or importing any native runtime.

python -m tools.muse.compare --reference reference.json --actual native.json
Elementwise acceptance: abs(actual-ref) <= atol + rtol*abs(ref).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_ATOL = 2e-5
DEFAULT_RTOL = 2e-5


def _read(value: dict, label: str) -> tuple[list[int], np.ndarray]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected object")
    tokens = value.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError(f"{label}: invalid tokens")
    rows = value.get("logits")
    if not isinstance(rows, list) or len(rows) != len(tokens):
        raise ValueError(f"{label}: expected one logits row per forced token")
    if any(not isinstance(row, list) or not row or
           any(type(x) not in (int, float) for x in row) for row in rows):
        raise ValueError(f"{label}: expected numeric logits rows")
    try:
        logits = np.asarray(rows, dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: invalid logits shape/number") from exc
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError(f"{label}: logits must be rectangular and finite")
    return tokens, logits


def compare(reference: dict, actual: dict, *, atol: float = DEFAULT_ATOL,
            rtol: float = DEFAULT_RTOL) -> str:
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    tokens, expected = _read(reference, "reference")
    actual_tokens, received = _read(actual, "actual")
    if tokens != actual_tokens:
        raise ValueError("forced token mismatch")
    if expected.shape != received.shape:
        raise ValueError(f"logits shape mismatch: expected {expected.shape}, got {received.shape}")
    error = np.abs(received - expected)
    accepted = error <= atol + rtol * np.abs(expected)
    if not accepted.all():
        row, col = np.argwhere(~accepted)[0]
        raise ValueError(f"logits mismatch at [{row},{col}]: expected {expected[row,col]:.9g}, "
                         f"got {received[row,col]:.9g}; max_abs={error.max():.9g}")
    return f"PASS {expected.shape[0]} x {expected.shape[1]} logits; max_abs={error.max():.9g}; atol={atol:g}, rtol={rtol:g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--atol", default=DEFAULT_ATOL, type=float)
    parser.add_argument("--rtol", default=DEFAULT_RTOL, type=float)
    args = parser.parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        actual = json.loads(args.actual.read_text(encoding="utf-8"))
        print(compare(reference, actual, atol=args.atol, rtol=args.rtol))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
