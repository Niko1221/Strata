"""Synthetic records only: these results are not hardware measurements."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.perf131k import statistics as stats


def fixture(stage="full"):
    manifest = {"schema": 2, "data_kind": "synthetic", "stage": stage,
                "analysis_sha256": stats.analysis_sha256(),
                "capacity": 131072, "seed": 17, "bootstrap_draws": 200,
                "permutation_draws": 200,
                "arms": {arm: {"source_sha256": letter * 64, "binary_sha256": letter * 64}
                         for arm, letter in (("baseline", "a"), ("candidate", "b"))},
                "shared_identity_sha256": "c" * 64,
                "families": {f"f{i}": {"fold": i} for i in range(5)},
                "endpoints": {"e2e_s": "lower", "decode_tps": "higher"}, "pairs": []}
    for i in range(5):
        for variant, repeats in (("short", 1), ("long", 3)):
            for repeat in range(repeats):
                manifest["pairs"].append({"family": f"f{i}", "variant": variant, "repeat": repeat,
                    "order": "AB" if len(manifest["pairs"]) % 2 == 0 else "BA",
                    "token_ids_sha256": "d" * 64,
                    "input_tokens": 126000 if stage == "full" else 4096,
                    "output_reserve": 1024 if stage == "full" else 256})
    frozen_sha = stats.digest(manifest)
    rows = []
    for pair in manifest["pairs"]:
        for arm in ("baseline", "candidate"):
            # Unequal repetition counts must not change equal variant/family weighting.
            value = 10 if pair["variant"] == "short" else 30
            rows.append({**pair, "arm": arm, "status": "PASS", "manifest_sha256": frozen_sha,
                "identity": manifest["arms"][arm], "shared_identity_sha256": "c" * 64,
                "capacity": 131072, "data_kind": "synthetic", "raw_artifact_sha256": "e" * 64,
                "independent_receipt_sha256": "f" * 64, "generated_steps": 512,
                "processed_input_tokens": 126000, "kv_progress_tokens": 126000,
                "recurrence_progress_tokens": 126000,
                "metrics": {"e2e_s": value - (2 if arm == "candidate" else 0),
                            "decode_tps": 5 + (1 if arm == "candidate" else 0)}})
    return manifest, rows, frozen_sha


class StatisticsTests(unittest.TestCase):
    def test_equal_family_variant_weight_and_synthetic_label(self):
        report = stats.analyze(*fixture())
        self.assertEqual(report["status"], "DESIGN_NOT_RUN")
        self.assertEqual(report["performance_verdict"], "NOT_EVALUATED")
        self.assertEqual((report["n_family"], report["n_variant"], report["n_repeat_pairs"]), (5, 10, 20))
        metric = report["endpoints"]["e2e_s"]
        self.assertEqual(metric["baseline_family_median"], 20)
        self.assertEqual(metric["paired_median_difference_candidate_minus_baseline"], -2)
        self.assertEqual(metric["family_bootstrap_95_ci"], [-2, -2])
        self.assertEqual(metric["draws"], 32)
        self.assertEqual(metric["p_value"], 0.0625)

    def test_missing_duplicate_failed_unknown_runs_block_all_endpoints(self):
        for action in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(status="FAILED"),
                       lambda rows: rows[0].update(family="unfrozen")):
            manifest, rows, frozen = fixture()
            action(rows)
            result = stats.analyze(manifest, rows, frozen)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertNotIn("endpoints", result)

    def test_full_receipt_count_progress_identity_and_metrics_gate(self):
        mutations = [{"generated_steps": 511}, {"independent_receipt_sha256": None},
                     {"processed_input_tokens": 125999}, {"kv_progress_tokens": 0},
                     {"recurrence_progress_tokens": 0}, {"capacity": 32768},
                     {"shared_identity_sha256": "0" * 64}, {"manifest_sha256": "0" * 64},
                     {"input_tokens": 125999}, {"output_reserve": 256},
                     {"metrics": {"e2e_s": float("nan"), "decode_tps": 5}},
                     {"metrics": {"e2e_s": 10}}, {"data_kind": "measured"}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                manifest, rows, frozen = fixture()
                rows[0].update(mutation)
                self.assertEqual(stats.analyze(manifest, rows, frozen)["status"], "BLOCKED")

    def test_frozen_map_cannot_change_after_sha_and_runs_binding(self):
        manifest, rows, frozen = fixture()
        manifest["families"]["f0"]["fold"] = 4
        self.assertEqual(stats.analyze(manifest, rows, frozen)["status"], "BLOCKED")

    def test_invalid_manifest_blocks_even_when_rehashed(self):
        for mutate in (lambda m: m.update(capacity=32768),
                       lambda m: m["pairs"][0].update(input_tokens=125999),
                       lambda m: m["pairs"][0].update(output_reserve=256),
                       lambda m: m["families"]["f0"].update(fold=4),
                       lambda m: m["arms"]["baseline"].update(source_sha256="UNRESOLVED"),
                       lambda m: m.update(bootstrap_draws=0)):
            manifest, rows, _ = fixture()
            mutate(manifest)
            frozen = stats.digest(manifest)
            for row in rows:
                row["manifest_sha256"] = frozen
            self.assertEqual(stats.analyze(manifest, rows, frozen)["status"], "BLOCKED")

    def test_pilot_reserve_stays_256_and_is_not_full_qualification(self):
        report = stats.analyze(*fixture("pilot"))
        self.assertEqual(report["stage"], "pilot")
        self.assertEqual(report["performance_verdict"], "NOT_EVALUATED")

    def test_holm_known_values(self):
        self.assertEqual(stats.holm({"a": .01, "b": .04, "c": .03}),
                         {"a": .03, "c": .06, "b": .06})

    def test_exact_and_seeded_monte_carlo(self):
        self.assertEqual(stats.sign_flip([1, 2, 3], 7, 200)["p_value"], .25)
        first = stats.sign_flip(list(range(1, 18)), 7, 200)
        self.assertEqual(first, stats.sign_flip(list(range(1, 18)), 7, 200))
        self.assertEqual(first["method"], "monte_carlo_add_one")
        self.assertGreater(first["p_value"], 0)

    def test_permutation_p_value_does_not_depend_on_metric_unit(self):
        for scale in (1e-15, 1e-9, 1.0, 1e9, 1e15):
            with self.subTest(scale=scale):
                self.assertEqual(stats.sign_flip([scale, 2 * scale, 3 * scale], 7, 200)["p_value"], .25)

    def test_analysis_source_must_match_precollection_freeze(self):
        manifest, rows, _ = fixture()
        manifest["analysis_sha256"] = "0" * 64
        frozen = stats.digest(manifest)
        for row in rows:
            row["manifest_sha256"] = frozen
        self.assertEqual(stats.analyze(manifest, rows, frozen)["status"], "BLOCKED")
        manifest["analysis_sha256"] = stats.analysis_sha256()
        manifest["schema"] = 1
        frozen = stats.digest(manifest)
        for row in rows:
            row["manifest_sha256"] = frozen
        self.assertEqual(stats.analyze(manifest, rows, frozen)["status"], "BLOCKED")

    def test_fixed_evaluation_folds_report_family_effects_without_extra_n(self):
        manifest, rows, frozen = fixture()
        for row in rows:
            if row["family"] == "f0" and row["arm"] == "candidate":
                row["metrics"]["e2e_s"] += 1
        report = stats.analyze(manifest, rows, frozen)
        self.assertEqual(report["n_family"], 5)
        folds = report["endpoints"]["e2e_s"]["evaluation_fold_summaries"]
        self.assertEqual(folds["0"]["families"], ["f0"])
        self.assertEqual(folds["0"]["paired_median_difference_candidate_minus_baseline"], -1)
        self.assertEqual(folds["1"]["paired_median_difference_candidate_minus_baseline"], -2)
        self.assertEqual(report["performance_verdict"], "NOT_EVALUATED")

    def test_order_and_resampling_reproducible(self):
        manifest, rows, frozen = fixture()
        self.assertEqual(stats.analyze(manifest, rows, frozen),
                         stats.analyze(manifest, list(reversed(rows)), frozen))

    def test_cli_missing_inputs_returns_blocked_json_exit_two(self):
        command = [sys.executable, "-B", "-m", "tools.perf131k.statistics",
                   "__absent_manifest__.json", "__absent_runs__.json", "--frozen-sha256", "0" * 64]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "BLOCKED")

    def test_cli_valid_synthetic_uses_canonical_digest_not_raw_file(self):
        manifest, rows, frozen = fixture()
        with tempfile.TemporaryDirectory(prefix="statistics-test-", dir=Path(__file__).resolve().parents[2]) as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (root / "runs.json").write_text(json.dumps(rows), encoding="utf-8")
            command = [sys.executable, "-B", "-m", "tools.perf131k.statistics",
                       str(root / "manifest.json"), str(root / "runs.json"), "--frozen-sha256", frozen]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "DESIGN_NOT_RUN")
            self.assertEqual(report["performance_verdict"], "NOT_EVALUATED")
            import hashlib
            command[-1] = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("endpoints", json.loads(result.stdout))

    def test_protocol_keeps_real_execution_not_run(self):
        protocol = json.loads((Path(__file__).resolve().parents[2] /
                               "docs/perf131k-statistical-protocol.json").read_text())
        self.assertEqual(protocol["status"], "DESIGN_NOT_RUN")
        gate = protocol["full_qualification_gate"]
        self.assertEqual((gate["history_tokens"], gate["full_test_reserve"],
                          gate["actual_generated_steps_min"]), (126000, 1024, 512))
        self.assertEqual((protocol["output_reserve"], gate["pilot_output_reserve"]), (256, 256))

    def test_malformed_input_and_derived_overflow_fail_closed(self):
        for manifest, rows in ((None, []), ({}, None), ({"schema": 1}, [None])):
            self.assertEqual(stats.analyze(manifest, rows, "0" * 64)["status"], "BLOCKED")
        manifest, rows, frozen = fixture()
        for row in rows:
            row["metrics"]["e2e_s"] = 1e308 if row["arm"] == "candidate" else 1e-308
        report = stats.analyze(manifest, rows, frozen)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertNotIn("endpoints", report)


if __name__ == "__main__":
    unittest.main()
