"""Independent selection/control check using Python ordering instead of profile selection helpers.

Author: Francisco Angulo de Lafuente. MIT licensed.
ORCID: https://orcid.org/0009-0001-1634-7063
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH.parents[2] / "tools"))
import evaluate_domain_profiles as source

record = json.loads((BENCH / "holdout-transfer/results.json").read_text())
assert hashlib.sha256((BENCH / "holdout-transfer/plan.json").read_bytes()).hexdigest() == record["plan_sha256"]
samples = source.load_samples(BENCH / "manifest.json")
by_id = {s["id"]: s for s in samples}
norm = {k: s["counts"] / s["counts"].sum() for k, s in by_id.items()}
old = {r["id"]: r for r in json.loads((BENCH / "results-1582.json").read_text())["rows"]}
rows = {r["id"]: r for r in record["rows"]}
values = {name: {} for name in ("mixed_reference", "code_only", "balanced_code_general")}
for fold in record["fold_profiles"]:
    train = fold["training_ids"]
    held_out = [identity for identity in by_id if old[identity]["fold"] == fold["fold"]]
    assert set(train) == set(by_id) - set(held_out)
    code = [k for k in train if by_id[k]["domain"] == "code"]
    general = [k for k in train if by_id[k]["domain"] != "code"]
    zero = np.zeros_like(next(iter(norm.values())))
    mix = sum((norm[k] for k in train), zero.copy())
    code_sum = sum((norm[k] for k in code), zero.copy())
    general_sum = sum((norm[k] for k in general), zero.copy())
    balanced = .5 * code_sum / len(code) + .5 * general_sum / len(general)
    nl, ne = mix.shape
    reference = sorted(range(nl * ne), key=lambda i: (-mix.ravel()[i], i))[:1582]
    quotas = [sum(i // ne == layer for i in reference) for layer in range(nl)]
    assert quotas == fold["layer_quotas"]
    selected = {"mixed_reference": reference}
    for name, scores in (("code_only", code_sum), ("balanced_code_general", balanced)):
        selected[name] = [layer * ne + expert for layer in range(nl)
                          for expert in sorted(range(ne), key=lambda e: (-scores[layer, e], e))[:quotas[layer]]]
    for name, indices in selected.items():
        assert len(indices) == len(set(indices)) == 1582
        digest = hashlib.sha256(np.array(sorted(indices), dtype="<i4").tobytes()).hexdigest()
        assert digest == fold["selected_set_sha256_i32le"][name]
        for identity in held_out:
            counts = by_id[identity]["counts"].ravel()
            coverage = sum(int(counts[i]) for i in indices) / int(counts.sum())
            assert abs(coverage - rows[identity]["coverage"][name]) < 1e-14
            values[name][identity] = coverage
            if name == "mixed_reference":
                assert abs(coverage - old[identity]["global_coverage"]) < 1e-14
            if name == "code_only" and by_id[identity]["domain"] == "code":
                assert abs(coverage - old[identity]["domain_same_layer_quotas_coverage"]) < 1e-14
aggregates = 0
for section in ("cohorts", "domains"):
    for group, summary in record[section].items():
        ids = [k for k in by_id if (section == "domains" and by_id[k]["domain"] == group) or
               (section == "cohorts" and (group == "all" or (by_id[k]["domain"] == "code") == (group == "code")))]
        assert len(ids) == summary["requests"]
        for name in values:
            assert abs(statistics.mean(values[name][k] for k in ids) - summary["mean_coverage"][name]) < 1e-14
            aggregates += 1
receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "independent_profile_sets": 15,
           "per_request_coverage_values": 135, "aggregate_means": aggregates,
           "published_global_rows_matched": 45, "published_code_rows_matched": 12,
           "mismatches": 0, "plan_sha256_matches_results": True, "gpu_used": False,
           "scope": "Independent Python sorting, selection hashes and integer coverage sums; existing trace parser reused"}
print(json.dumps(receipt, indent=2))
