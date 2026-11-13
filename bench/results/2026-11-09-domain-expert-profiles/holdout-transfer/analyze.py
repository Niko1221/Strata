"""Compare fixed profiles across code and non-code holdouts, on logged routing only.

Author: Francisco Angulo de Lafuente. MIT licensed.
ORCID: https://orcid.org/0009-0001-1634-7063
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH.parents[2] / "tools"))
import evaluate_domain_profiles as ev


def analyze():
    plan_bytes = (HERE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    for name, digest in plan["inputs_sha256"].items():
        text_bytes = (BENCH / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(text_bytes).hexdigest() == digest, name
    samples = ev.load_samples(BENCH / "manifest.json")
    assert Counter(s["domain"] for s in samples) == plan["expected_domains"]
    reference = json.loads((BENCH / "results-1582.json").read_text())
    slots = plan["slots"]
    assert reference["slots"] == slots and reference["folds"] == plan["folds"]
    assert reference["seed"] == plan["seed"]
    assert len(reference["rows"]) == len(samples) == 45
    by_id = {s["id"]: i for i, s in enumerate(samples)}
    assert len(by_id) == len(samples)
    assert {r["id"] for r in reference["rows"]} == set(by_id)
    dist = np.array([s["counts"] / s["counts"].sum() for s in samples])
    shape = dist.shape[1:]
    profiles, rows = [], []
    for fold in range(plan["folds"]):
        held_out = [r for r in reference["rows"] if r["fold"] == fold]
        assert held_out
        training_ids = held_out[0]["training_ids"]
        assert all(r["training_ids"] == training_ids for r in held_out)
        assert set(training_ids) == set(by_id) - {r["id"] for r in held_out}
        assert training_ids == sorted(training_ids)
        train = [by_id[identity] for identity in training_ids]
        code = [i for i in train if samples[i]["domain"] == "code"]
        non_code = [i for i in train if samples[i]["domain"] != "code"]
        assert code and non_code
        mixed_score = dist[train].sum(axis=0)
        mixed = ev.ranked(mixed_score)[:slots]
        quotas = np.bincount(mixed // shape[1], minlength=shape[0])
        selection = {
            "mixed_reference": mixed,
            "code_only": ev.quota_profile(dist[code].sum(axis=0), quotas),
            "balanced_code_general": ev.quota_profile(
                0.5 * dist[code].mean(axis=0) + 0.5 * dist[non_code].mean(axis=0), quotas),
        }
        for indices in selection.values():
            assert len(indices) == len(set(indices.tolist())) == slots
            assert np.array_equal(np.bincount(indices // shape[1], minlength=shape[0]), quotas)
        profiles.append({"fold": fold, "training_ids": training_ids,
                         "code_training_ids": [samples[i]["id"] for i in code],
                         "non_code_training_ids": [samples[i]["id"] for i in non_code],
                         "layer_quotas": quotas.tolist(),
                         "selected_set_sha256_i32le": {
                             name: hashlib.sha256(np.sort(indices).astype("<i4").tobytes()).hexdigest()
                             for name, indices in selection.items()}})
        for old in held_out:
            sample = samples[by_id[old["id"]]]
            counts = sample["counts"].ravel()
            coverage = {name: float(counts[indices].sum() / counts.sum())
                        for name, indices in selection.items()}
            assert abs(coverage["mixed_reference"] - old["global_coverage"]) < 1e-14
            if sample["domain"] == "code":
                assert abs(coverage["code_only"] - old["domain_same_layer_quotas_coverage"]) < 1e-14
            rows.append({"id": sample["id"], "domain": sample["domain"], "fold": fold,
                         "coverage": coverage})
    rows.sort(key=lambda r: r["id"])

    def summarize(selected):
        means = {name: float(np.mean([r["coverage"][name] for r in selected]))
                 for name in plan["profiles"]}
        return {"requests": len(selected), "mean_coverage": means,
                "gain_over_mixed_pp": {name: 100 * (value - means["mixed_reference"])
                                       for name, value in means.items()}}

    cohorts = {"all": summarize(rows),
               "code": summarize([r for r in rows if r["domain"] == "code"]),
               "non_code": summarize([r for r in rows if r["domain"] != "code"])}
    domains = {domain: summarize([r for r in rows if r["domain"] == domain])
               for domain in sorted(plan["expected_domains"])}
    return {"kind": "exploratory_cross_domain_static_routing_coverage",
            "author": "Francisco Angulo de Lafuente", "orcid": "https://orcid.org/0009-0001-1634-7063",
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "slots": slots, "folds": plan["folds"], "seed": plan["seed"],
            "weighting": "one held-out request, one vote; non-code retains its observed domain mixture",
            "cohorts": cohorts, "domains": domains, "fold_profiles": profiles, "rows": rows,
            "zero_tolerance_static_diagnostic": {
                name: {"both_cohorts_nonregressing": all(
                    cohorts[group]["gain_over_mixed_pp"][name] >= 0 for group in ("code", "non_code")),
                       "regressing_domains": [group for group in domains
                                              if domains[group]["gain_over_mixed_pp"][name] < 0]}
                for name in plan["profiles"] if name != "mixed_reference"},
            "checks": {"published_global_rows_matched": 45, "published_code_rows_matched": 12,
                       "training_excludes_entire_held_out_fold": True,
                       "matched_slots_and_layer_quotas": True},
            "limits": plan["limits"]}


if __name__ == "__main__":
    result = analyze()
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("cohorts", "domains", "zero_tolerance_static_diagnostic", "checks")}, indent=2))
