"""Summarize completed real-weight runs without hiding numerical failures."""
import argparse
import json
from pathlib import Path


def summarize(reports):
    assert reports and all(d.get("execution_complete") for d in reports)
    assert all(len(d["cases"]) == 7 and len(d["expert_sweep"]) == 512 for d in reports)
    assert all(d["weights"] == reports[0]["weights"] for d in reports)
    assert all(d["source_sha256"] == reports[0]["source_sha256"] for d in reports)
    assert all(d["binary_sha256"] == reports[0]["binary_sha256"] for d in reports)
    for d in reports:
        assert [x["expert"] for x in d["expert_sweep"]] == list(range(512))
    groups = [[d[k] for d in reports] for k in ("cases", "expert_sweep")]
    stable = []
    for group in groups:
        stable.append(all(len({rows[i]["output_sha256"] for rows in group}) == 1
                          and len({rows[i]["input_sha256"] for rows in group}) == 1
                          for i in range(len(group[0]))))
    layers = [c for d in reports for c in d["cases"]]
    experts = [c for d in reports for c in d["expert_sweep"]]
    quant = [c for d in reports for c in d["quantization_checks"]]
    projections = [c for d in reports for c in d["projection_checks"]]
    return {
        "fresh_processes": len(reports), "layer": 0,
        "weight_repository": reports[0]["weights"]["repository"],
        "weight_revision": reports[0]["weights"]["revision"],
        "independent_origin_hashes_verified": reports[0]["weights"]["independent_origin_hashes_verified"],
        "complete_layer_cases": len(layers),
        "complete_layer_cases_passed": sum(c["passed"] for c in layers),
        "complete_layer_output_hashes_stable": stable[0],
        "forced_expert_cases": len(experts),
        "forced_expert_cases_passed": sum(c["passed"] for c in experts),
        "forced_expert_output_hashes_stable": stable[1],
        "failed_experts_per_process": [[c["expert"] for c in d["expert_sweep"] if not c["passed"]] for d in reports],
        "projection_checks": len(projections), "projection_failures": sum(not c["passed"] for c in projections),
        "max_projection_error_over_bound": max(c["max_error_over_bound"] for c in projections),
        "quantization_checks": len(quant), "quantization_failures": sum(not c["passed"] for c in quant),
        "max_quantization_scale_ulps": max(c["max_scale_ulps"] for c in quant),
        "all_resident_weights_unchanged": all(d["weights_unchanged"] for d in reports),
        "max_layer_output_normalized_rms_error": max(c["output"]["normalized_rms_error"] for c in layers),
        "max_layer_output_max_normalized_error": max(c["output"]["max_normalized_error"] for c in layers),
        "max_expert_normalized_rms_error": max(c["normalized_rms_error"] for c in experts),
        "max_expert_max_normalized_error": max(c["max_normalized_error"] for c in experts),
        "torch_npu_peak_allocated_bytes": max(d["npu_peak_allocated_bytes"] for d in reports),
        "torch_npu_peak_reserved_bytes": max(d["npu_peak_reserved_bytes"] for d in reports),
        "all_original_numerical_budgets_passed": all(d["passed"] for d in reports),
        "budgets": reports[0]["budgets"],
        "binary_sha256": reports[0]["binary_sha256"],
        "source_sha256": reports[0]["source_sha256"],
        "scope": reports[0]["scope"], "input_scope": reports[0]["input_scope"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = summarize([json.loads(p.read_text()) for p in args.reports])
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
