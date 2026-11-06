"""Compare genuine device logits/residual dumps with the independent oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def log_softmax(values):
    shifted = values.astype(np.float64) - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compare_logits(reference, actual, tolerance=0.02):
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 248320:
        raise ValueError("expected equally shaped float32 [steps,248320] logits")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite logits")
    expected = log_softmax(reference)
    measured = log_softmax(actual)
    rows = []
    for step, (gold, output) in enumerate(zip(expected, measured)):
        top = np.argsort(gold)[-5:][::-1]
        actual_top = np.argsort(output)[-5:][::-1]
        divergence = float((np.exp(gold) * (gold - output)).sum())
        error = float(abs(output[top] - gold[top]).max())
        rows.append({
            "step": step, "reference_top5": top.tolist(), "device_top5": actual_top.tolist(),
            "top5_same_order": bool(np.array_equal(top, actual_top)),
            "top5_max_logprob_error": error, "kl_reference_device": divergence,
            "pass": bool(np.array_equal(top, actual_top) and error <= tolerance),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--logprob-tolerance", type=float, default=0.02)
    parser.add_argument("--reference-layers", type=Path)
    parser.add_argument("--actual-layers", type=Path)
    parser.add_argument("--layer-relative-tolerance", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = np.fromfile(args.reference, dtype=np.float32).reshape(-1, 248320)
    actual = np.fromfile(args.actual, dtype=np.float32).reshape(-1, 248320)
    rows = compare_logits(reference, actual, args.logprob_tolerance)
    layers = []
    if args.reference_layers or args.actual_layers:
        if not args.reference_layers or not args.actual_layers:
            parser.error("both layer dump arguments are required")
        reference_layers = np.fromfile(args.reference_layers, dtype=np.float32).reshape(-1, 48, 4, 2560)
        for position in range(len(reference_layers)):
            for layer in range(48):
                target = args.actual_layers / f"position-{position:05d}-layer-{layer:02d}.npy"
                expected = reference_layers[position, layer].astype(np.float64).ravel()
                measured = np.load(target).astype(np.float64).ravel()
                relative = float(np.linalg.norm(expected - measured) / max(np.linalg.norm(expected), 1e-30))
                layers.append({"position": position, "layer": layer, "relative_l2": relative,
                               "pass": bool(relative <= args.layer_relative_tolerance)})
    result = {"pass": all(row["pass"] for row in rows + layers),
              "logits": rows, "layers": layers,
              "logprob_tolerance": args.logprob_tolerance,
              "layer_relative_tolerance": args.layer_relative_tolerance}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
