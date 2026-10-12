"""Describe retained fresh-forward differences; never sets gate tolerances."""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_logits(path):
    raw = path.read_bytes()
    assert len(raw) == 248320 * 4, (path, len(raw))
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    assert all(math.isfinite(x) for x in values), path
    return values


def softmax(logits):
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    total = math.fsum(weights)
    return [w / total for w in weights], maximum + math.log(total)


def quantile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(reference, candidate):
    size = len(reference)
    delta = [b - a for a, b in zip(reference, candidate)]
    mean = math.fsum(delta) / size
    absolute = sorted(abs(x) for x in delta)
    p, log_zp = softmax(reference)
    q, log_zq = softmax(candidate)
    reference_order = sorted(range(size), key=reference.__getitem__, reverse=True)
    candidate_order = sorted(range(size), key=candidate.__getitem__, reverse=True)
    max_index = max(range(size), key=lambda i: abs(delta[i]))
    reference_top = reference_order[0]
    candidate_top = candidate_order[0]
    return {
        "compared_logits": size,
        "exact_equal_logits": sum(x == 0 for x in delta),
        "max_absolute_error": absolute[-1],
        "mean_signed_error": mean,
        "rms_error": math.sqrt(math.fsum(x * x for x in delta) / size),
        "absolute_error_quantiles": {str(f): quantile(absolute, f) for f in (0.5, 0.9, 0.99, 0.999)},
        "after_removing_uniform_mean_shift": {
            "max_absolute_error": max(abs(x - mean) for x in delta),
            "rms_error": math.sqrt(math.fsum((x - mean) ** 2 for x in delta) / size),
        },
        "reference_range": [min(reference), max(reference)],
        "candidate_range": [min(candidate), max(candidate)],
        "top1_equal": reference_top == candidate_top,
        "reference_top1": {
            "token_id": reference_top,
            "reference_probability": p[reference_top],
            "candidate_probability": q[reference_top],
            "reference_top1_top2_logit_gap": reference[reference_top] - reference[reference_order[1]],
        },
        "candidate_top1": {
            "token_id": candidate_top,
            "reference_probability": p[candidate_top],
            "candidate_probability": q[candidate_top],
            "candidate_top1_top2_logit_gap": candidate[candidate_top] - candidate[candidate_order[1]],
        },
        "top_k_intersection_count": {str(k): len(set(reference_order[:k]) & set(candidate_order[:k])) for k in (5, 10, 50)},
        "max_error_token": {
            "token_id": max_index,
            "reference_logit": reference[max_index],
            "candidate_logit": candidate[max_index],
            "reference_probability": p[max_index],
            "candidate_probability": q[max_index],
            "reference_rank": reference_order.index(max_index) + 1,
            "candidate_rank": candidate_order.index(max_index) + 1,
        },
        "total_variation_distance": 0.5 * math.fsum(abs(a - b) for a, b in zip(p, q)),
        "max_absolute_probability_change": max(abs(a - b) for a, b in zip(p, q)),
        "kl_reference_to_candidate_nats": math.fsum(a * ((x - log_zp) - (y - log_zq)) for a, x, y in zip(p, reference, candidate)),
        "kl_candidate_to_reference_nats": math.fsum(b * ((y - log_zq) - (x - log_zp)) for b, x, y in zip(q, reference, candidate)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    progress_path = args.run / "progress.json"
    progress = json.loads(progress_path.read_text())
    commands = {row["label"]: row for row in progress["commands"]}
    result = {
        "status": "DIAGNOSTIC_ONLY",
        "timestamp": time.time(),
        "script_sha256": sha256(Path(__file__)),
        "source_progress": str(progress_path),
        "source_progress_sha256": sha256(progress_path),
        "oracle_sha256": progress["oracle_sha256"],
        "oracle_source_revision": progress["oracle_source_revision"],
        "hardware": progress["hardware"],
        "weight_format": progress["weight_format"],
        "kv_format": progress["kv_format"],
        "model_files": progress["model_files"],
        "probability_definition": "softmax of raw logits at temperature 1, without grammar, penalties or truncation",
        "arithmetic": "Python float64, stable max-subtracted exponentials, math.fsum; little-endian float32 inputs",
        "limitations": [
            "Fresh forwards at three synthetic histories; no branch checkpoint or restore in these comparisons.",
            "No tolerance is set or modified; these diagnostics are not acceptance tests.",
            "Uniform-shift-adjusted error and probability metrics supplement, never replace, the raw-logit contract.",
        ],
        "comparisons": [],
    }
    for length in (1, 31, 127):
        ref_path = args.run / f"n{length}-reference.f32"
        reference = read_logits(ref_path)
        for chunk in (1, 31, 128):
            path = args.run / f"n{length}-chunk{chunk}.f32"
            for vector in (ref_path, path):
                assert sha256(vector) == commands[vector.stem]["logits_sha256"], vector
            ref_ids = [int(x) for x in ref_path.with_suffix(".tokens").read_text().replace(",", " ").split()]
            candidate_ids = [int(x) for x in path.with_suffix(".tokens").read_text().replace(",", " ").split()]
            assert ref_ids == candidate_ids and len(ref_ids) == length
            row = {"length": length, "reference_chunk": 128, "candidate_chunk": chunk,
                   "reference_sha256": sha256(ref_path), "candidate_sha256": sha256(path),
                   "reference_command": commands[ref_path.stem]["command"],
                   "candidate_command": commands[path.stem]["command"],
                   **compare(reference, read_logits(path))}
            result["comparisons"].append(row)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "comparisons": [{k: row[k] for k in (
                          "length", "candidate_chunk", "max_absolute_error", "rms_error", "top1_equal",
                          "reference_top1", "candidate_top1", "total_variation_distance",
                          "kl_reference_to_candidate_nats", "max_error_token")}
                          for row in result["comparisons"] if row["max_absolute_error"]]}, indent=2))


if __name__ == "__main__":
    main()
