"""tools/memt_compare.py - compare two `memt-memory-v1` snapshots.

P6 asks whether the engine's FP32 memory state matches memt_sglang's after the
same commits.  Both sides write the same format, so the comparison is a file
comparison: per-layer maximum absolute and relative difference, cosine
similarity, and whether the tensors are bit-identical.

    python tools/memt_compare.py --a ref.safetensors --identity-a ID_A \\
                                 --b engine.safetensors --identity-b ID_B [--json]

Exit status 0 when every layer is present in both, the shapes agree, and the
per-layer cosine is at least `--min-cosine` (default 0.999); 1 otherwise; 2 for a
file/identity error.  `--json` prints the report instead of a table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import memt_snapshot as S  # noqa: E402


def _flatten(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def compare_layer(ref: np.ndarray, got: np.ndarray) -> dict:
    """One layer's metrics.  `ref`/`got` are [1, H, D, D] FP32.

    `rel_l2` is ||ref - got|| / ||ref||: cosine is scale-invariant, so it alone
    would call a state that is uniformly twice as large a match.
    """
    if tuple(ref.shape) != tuple(got.shape):
        return {"same_shape": False, "ref_shape": list(ref.shape), "got_shape": list(got.shape)}
    a, b = _flatten(ref), _flatten(got)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), 1e-12)
    cosine = float(np.dot(a, b) / (math.sqrt(float(np.dot(a, a))) * math.sqrt(float(np.dot(b, b))) + 1e-30))
    return {
        "same_shape": True,
        "bit_identical": bool(np.array_equal(a, b)),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float((diff / scale).max()) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)) if diff.size else 0.0,
        "cosine": cosine,
        "nonfinite": int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum()),
    }


def compare(ref: Mapping[int, np.ndarray], got: Mapping[int, np.ndarray], *,
            min_cosine: float = 0.999, max_rel_l2: float = 0.25) -> dict:
    """The whole comparison: the layer sets must match, then every layer's metrics.

    A layer passes at `cosine >= min_cosine` **and** `rel_l2 <= max_rel_l2` (and
    finite, and the same shape).  The two bounds are a *gross-error ceiling*, not
    an agreed numerical tolerance: the reference runs BF16, this engine Q8_0, so
    the real tolerance is agreed from the numbers a GPU run reports.
    """
    want, have = set(ref), set(got)
    report = {
        "layers": {},
        "missing_in_engine": sorted(want - have),
        "extra_in_engine": sorted(have - want),
        "min_cosine": float(min_cosine),
        "max_rel_l2": float(max_rel_l2),
        "ok": False,
    }
    for layer in sorted(want & have):
        report["layers"][str(layer)] = compare_layer(ref[layer], got[layer])
    report["ok"] = (
        not report["missing_in_engine"]
        and not report["extra_in_engine"]
        and bool(report["layers"])
        and all(item["same_shape"]
                and item.get("cosine", 0.0) >= min_cosine
                and item.get("rel_l2", math.inf) <= max_rel_l2
                and not item.get("nonfinite", 0)
                for item in report["layers"].values())
    )
    return report


def load_pair(path_a, identity_a, path_b, identity_b) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    a = S.load(Path(path_a), identity_a)
    b = S.load(Path(path_b), identity_b)
    if set(a) != set(b):
        raise S.SnapshotError("layer sets differ: %r vs %r" % (sorted(a), sorted(b)))
    for layer in a:
        if a[layer].shape != b[layer].shape:
            raise S.SnapshotError("memory.%d shapes differ: %r vs %r" % (layer, a[layer].shape, b[layer].shape))
    return a, b


def format_table(report: dict) -> str:
    lines = ["%6s %10s %10s %12s %12s %10s %6s"
             % ("layer", "cosine", "rel_l2", "max_abs", "max_rel", "identical", "shape")]
    for layer, item in report["layers"].items():
        if not item["same_shape"]:
            lines.append("%6s %10s %10s %12s %12s %10s %6s" % (layer, "-", "-", "-", "-", "-", "DIFF"))
            continue
        passes = (item["cosine"] >= report["min_cosine"] and item["rel_l2"] <= report["max_rel_l2"])
        lines.append("%6s %10.8f %10.3e %12.3e %12.3e %10s %6s" % (
            layer, item["cosine"], item["rel_l2"], item["max_abs"], item["max_rel"],
            "yes" if item["bit_identical"] else "no", "ok" if passes else "FAIL"))
    if report["missing_in_engine"]:
        lines.append("missing in the engine snapshot: %r" % report["missing_in_engine"])
    if report["extra_in_engine"]:
        lines.append("extra in the engine snapshot: %r" % report["extra_in_engine"])
    lines.append("result: %s (min cosine %.6f, max rel_l2 %.3f)"
                 % ("PASS" if report["ok"] else "FAIL", report["min_cosine"], report["max_rel_l2"]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the reference snapshot (memt_sglang)")
    ap.add_argument("--identity-a", required=True)
    ap.add_argument("--b", required=True, help="the engine snapshot")
    ap.add_argument("--identity-b", required=True)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--max-rel-l2", type=float, default=0.25,
                    help="gross-error ceiling on ||a-b||/||a|| (cosine alone cannot see a scaled state)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        ref, got = load_pair(args.a, args.identity_a, args.b, args.identity_b)
    except (OSError, S.SnapshotError) as exc:
        print("memt_compare: %s" % exc, file=sys.stderr)
        return 2
    report = compare(ref, got, min_cosine=args.min_cosine, max_rel_l2=args.max_rel_l2)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_table(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
