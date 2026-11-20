"""CPU-only preparation/consumer tests. Never start a model or GPU process."""
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import uuid
import shutil
import os
import importlib.util
import sys
from tools.perf131k import native_pilot as native
if os.environ.get("P01_NATIVE_UNDER_TEST"):
    spec = importlib.util.spec_from_file_location("native_under_test", os.environ["P01_NATIVE_UNDER_TEST"])
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    native.ROOT = Path(__file__).resolve().parents[2]  # Keep mutant input location equivalent to the real checkout.


class NativePilotTests(TestCase):
    def fake_cell(self, *, raw=None, stale_tokenizer=False, trace="OFF", job_commit_gib=None,
                  resources=None, observed=None, memory_diagnostics=False, blob_diagnostics=False, job_result=None,
                  baseline_env=None, use_pc_memory=False, resource_sequence=None, frozen_pc_budget_bytes=None):
        base = Path("H:/Strata/worktrees/p01-a-20261025/_scratch/perf131k/b2-fixtures") / uuid.uuid4().hex
        base.mkdir(parents=True)
        shutil.copytree(native.BASE / "prepared", base / "prepared")
        path = base / "prepared/binding.json"
        if stale_tokenizer or baseline_env:
            binding = json.loads(path.read_text())
            if stale_tokenizer:
                binding["tokenizer"]["vocab.json"] = "0" * 64
            if baseline_env:
                binding["baseline_env"].update(baseline_env)
            path.write_text(json.dumps(binding))
        def fake_runtime(command, root, env, output, budget, sample, *, stdin_path, **diagnostics):
            if observed is not None:
                observed.update(budget=vars(budget), launch=json.loads((root / "launch.json").read_text()),
                                diagnostics=diagnostics)
            output.write_text(raw or "READY 131072 stop\nT 1\nDONE 1 4096 100 200 stop 0 0 0\n"
                              "T 1\nDONE 1 4096 10 200 stop 0 0 4096\n")
            return {"status": "FINISHED", "exit_code": 0, "termination_confirmed": True,
                    "job_active_after": 0, "owned_root_pid": 123, **(job_result or {})}
        class Tok:
            def decode(self, ids):
                return native.SENTINEL
        if resources is None:
            resources = {"available_ram": 20 * 1024**3, "commit_headroom": 30 * 1024**3, "H_free": 30 * 1024**3}
        with patch.object(native, "BASE", base), patch.object(native.windows_job, "run_owned", side_effect=fake_runtime) as spawn, \
             patch.object(native, "system_resources", return_value=resources, side_effect=resource_sequence), \
             patch.object(native, "physical_ram_bytes", return_value=32 * 1024**3), \
             patch.object(native, "tokenizer", return_value=Tok()):
            options = {} if job_commit_gib is None else {"job_commit_gib": job_commit_gib}
            if memory_diagnostics:
                options["memory_diagnostics"] = True
            if blob_diagnostics:
                options["blob_diagnostics"] = True
            if use_pc_memory:
                options["use_pc_memory"] = True
            if frozen_pc_budget_bytes is not None:
                options["frozen_pc_budget_bytes"] = frozen_pc_budget_bytes
            code = native.run_cell(4096, trace, binding_sha=native.sha(path), **options)
        result = json.loads((base / f"native-4096-{trace.lower()}/result.json").read_text())
        return code, result, spawn.call_count

    def test_ten_GiB_exact_admission_reaches_owned_consumer(self):
        observed = {}
        limits = {"available_ram": 12 * 1024**3, "commit_headroom": 16 * 1024**3, "H_free": 8 * 1024**3}
        code, result, calls = self.fake_cell(job_commit_gib=10, resources=limits, observed=observed)
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(observed["budget"]["job_commit_bytes"], 10 * 1024**3)
        self.assertEqual(observed["launch"]["budget"], observed["budget"])
        self.assertEqual(result["requested_job_commit_gib"], 10)

    def test_ten_GiB_each_insufficient_resource_prevents_child(self):
        for key, minimum in (("available_ram", 12), ("commit_headroom", 16), ("H_free", 8)):
            with self.subTest(resource=key):
                limits = {"available_ram": 12 * 1024**3, "commit_headroom": 16 * 1024**3, "H_free": 8 * 1024**3}
                limits[key] = minimum * 1024**3 - 1
                code, result, calls = self.fake_cell(job_commit_gib=10, resources=limits)
                self.assertEqual((code, calls), (77, 0))
                self.assertEqual(result["status"], "PREFLIGHT_BLOCKED")

    def test_blob_diagnostics_reach_child_and_cannot_be_performance_comparison(self):
        observed = {}
        code, result, calls = self.fake_cell(observed=observed, blob_diagnostics=True)
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(observed["launch"]["env"]["STRATA_PERF131K_BLOB_DIAG"], "1")
        self.assertTrue(observed["launch"]["source_blob_diagnostics"])
        self.assertTrue(result["source_blob_diagnostics"])
        self.assertTrue(result["measurement_valid"])
        self.assertFalse(result["comparison_valid"])

    def test_sixteen_GiB_explicit_arm_preserves_headroom_and_no_auto_increase(self):
        observed = {}
        limits = {"available_ram": 18 * 1024**3, "commit_headroom": 22 * 1024**3, "H_free": 8 * 1024**3}
        code, result, calls = self.fake_cell(job_commit_gib=16, resources=limits, observed=observed,
                                             blob_diagnostics=True)
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(observed["budget"]["job_commit_bytes"], 16 * 1024**3)
        for key in ("available_ram", "commit_headroom", "H_free"):
            low = {**limits, key: limits[key] - 1}
            code, result, calls = self.fake_cell(job_commit_gib=16, resources=low)
            self.assertEqual((code, calls), (77, 0))
        self.assertEqual(native.make_budget().job_commit_bytes, 8 * 1024**3)

    def test_windows_case_insensitive_blob_flag_cannot_escape_comparison_exclusion(self):
        observed = {}
        code, result, calls = self.fake_cell(observed=observed,
            baseline_env={"strata_perf131k_blob_diag": "1"})
        self.assertEqual((code, calls), (0, 1))
        self.assertTrue(result["source_blob_diagnostics"])
        self.assertFalse(result["comparison_valid"])
        self.assertNotIn("strata_perf131k_blob_diag", observed["launch"]["env"])
        code, result, calls = self.fake_cell(baseline_env={
            "STRATA_PERF131K_BLOB_DIAG": "0", "strata_perf131k_blob_diag": "1"})
        self.assertEqual(calls, 0)
        self.assertEqual(result["status"], "INPUT_BINDING_FAILED")

    def test_pc_memory_cap_freezes_and_rechecks_before_any_child(self):
        gib = 1024**3
        planning = {"available_ram": 20 * gib, "commit_headroom": 48 * gib, "H_free": 30 * gib}
        observed = {}
        code, result, calls = self.fake_cell(use_pc_memory=True, blob_diagnostics=True,
            memory_diagnostics=True, resources=planning, observed=observed)
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(result["frozen_job_commit_bytes"], 30 * gib)
        self.assertEqual(observed["budget"]["min_available_ram"], 2 * gib)
        self.assertFalse(result["comparison_valid"])
        for key, floor in (("available_ram", 8 * gib), ("commit_headroom", 36 * gib), ("H_free", 8 * gib)):
            code, result, calls = self.fake_cell(use_pc_memory=True, resource_sequence=[
                planning, {**planning, key: floor - 1}])
            self.assertEqual((code, calls), (77, 0))
            self.assertEqual(result["frozen_job_commit_bytes"], 30 * gib)

    def test_pc_memory_rounds_down_and_respects_available_commit(self):
        gib = 1024**3
        resources = {"available_ram": 20 * gib, "commit_headroom": 25 * gib + 12345, "H_free": 10 * gib}
        budget = native.make_pc_budget(resources, 32 * gib)
        self.assertEqual(budget.job_commit_bytes, 19 * gib)
        self.assertEqual(native.make_budget().job_commit_bytes, 8 * gib)
        with self.assertRaises(ValueError):
            native.make_pc_budget(resources, True)
        with self.assertRaises(ValueError):
            native.make_pc_budget({**resources, "commit_headroom": 5 * gib}, 32 * gib)

    def test_predeclared_pc_cap_supports_normal_trials_without_per_arm_resizing(self):
        gib = 1024**3
        resources = {"available_ram": 20 * gib, "commit_headroom": 48 * gib, "H_free": 30 * gib}
        code, result, calls = self.fake_cell(frozen_pc_budget_bytes=30 * gib, resources=resources)
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(result["frozen_job_commit_bytes"], 30 * gib)
        self.assertTrue(result["comparison_valid"])
        code, result, calls = self.fake_cell(frozen_pc_budget_bytes=30 * gib,
            resources={**resources, "commit_headroom": 36 * gib - 1})
        self.assertEqual((code, calls), (77, 0))
        self.assertEqual(result["frozen_job_commit_bytes"], 30 * gib)
        code, result, calls = self.fake_cell(frozen_pc_budget_bytes=30 * gib, resources=resources,
                                             memory_diagnostics=True)
        self.assertFalse(result["comparison_valid"])
        for cap in (True, -1, 30 * gib + 1, 31 * gib):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                native.make_frozen_pc_budget(cap, 32 * gib)

    def test_budget_override_is_bounded_and_default_remains_eight(self):
        observed = {}
        code, _, calls = self.fake_cell(observed=observed)
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(observed["budget"]["job_commit_bytes"], 8 * 1024**3)
        for value in (True, 10.0, 0, 9, 11, 64, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native.make_budget(value)

    def test_malformed_DONE_retains_known_cleanup(self):
        code, result, _ = self.fake_cell(raw="READY 131072 stop\nDONE 1\n")
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], "ANALYSIS_FAILED")
        self.assertFalse(result["measurement_valid"])
        self.assertTrue(result["termination_confirmed"])
        self.assertEqual(result["job_active_after"], 0)
        self.assertEqual(result["owned_root_pid"], 123)

    def test_stale_tokenizer_prevents_child_creation(self):
        code, result, calls = self.fake_cell(stale_tokenizer=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, 0)
        self.assertEqual(result["status"], "INPUT_BINDING_FAILED")

    def test_missing_ON_trace_invalidates_comparison(self):
        code, result, _ = self.fake_cell(trace="ON")
        self.assertNotEqual(code, 0)
        self.assertFalse(result["comparison_valid"])
        self.assertEqual(result["status"], "TRACE_INVALID")
        self.assertIsNone(result["requests"][0]["engine_reported_tokens_per_second"])

    def test_incomplete_trace_does_not_invalidate_native_request_totals(self):
        from tools.perf131k import capture_trace
        with patch.object(capture_trace, "analyze", return_value={"trace_status": "INCOMPLETE",
                          "counts": {"dropped": 1}, "record_count": 4096, "source": "host_only"}):
            code, result, _ = self.fake_cell(trace="ON")
        self.assertNotEqual(code, 0)
        self.assertTrue(result["measurement_valid"])
        self.assertFalse(result["comparison_valid"])
        self.assertEqual(len(result["requests"]), 2)

    def test_post_run_identity_failure_keeps_cleanup_and_rejects_measurement(self):
        original = native.verify_inputs
        calls = 0
        def verify(binding):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected post-run identity read failure")
            return original(binding)
        with patch.object(native, "verify_inputs", side_effect=verify):
            code, result, _ = self.fake_cell()
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], "INPUT_BINDING_FAILED")
        self.assertFalse(result["measurement_valid"])
        self.assertEqual(result["job_active_after"], 0)
        self.assertTrue(result["termination_confirmed"])

    def test_real_tokenizer_fixtures_and_fake_runtime_connection(self):
        base = Path("H:/Strata/worktrees/p01-a-20261025/_scratch/perf131k/b2-fixtures") / uuid.uuid4().hex
        base.mkdir(parents=True)
        (base / "CPU_UNIT_FIXTURE").write_text("All runtime counters in this directory are scripted CPU fixtures.")
        frozen = json.loads((native.BASE / "prepared/binding.json").read_text())
        inventory = base / "_docs/perf131k"
        inventory.mkdir(parents=True)
        (inventory / "model-inventory.json").write_text(json.dumps({"saved_Q2_configuration": {
            "args": frozen["native_args"], "env": frozen["baseline_env"]}}), encoding="utf-8")
        with patch.object(native, "BASE", base), patch.object(native, "ROOT", base):
            native.prepare()  # Reads tokenizer and verification exe only; no model tensor reads.
            binding = json.loads((base / "prepared/binding.json").read_text())
            for count in (4096, 32768):
                ids = json.loads((base / f"prepared/tokens-{count}.json").read_text())
                self.assertEqual(len(ids), count)
                self.assertEqual(binding["capacity"], 131072)
            def fake_runtime(command, root, env, raw, budget, sample, *, stdin_path):
                # StrataEngine passes flags directly to this native executable;
                # "generate" is the Python CLI's subcommand, not a native argv.
                self.assertEqual(command[0], str(native.BINARY))
                self.assertTrue(command[1].startswith("--"))
                self.assertNotIn("generate", command[1:])
                self.assertIn("--serve", command)
                self.assertIn("131072", command)
                self.assertEqual(stdin_path.read_text().count("GEN 256 "), 2)
                raw.write_text("READY 131072 stop\nT 1\nDONE 1 4096 100 200 stop 0 0 0\n"
                               "T 1\nDONE 1 4096 10 200 stop 0 0 4096\n")
                return {"status": "FINISHED", "exit_code": 0, "termination_confirmed": True,
                        "job_active_after": 0}
            resources = {"available_ram": 20 * 1024**3, "commit_headroom": 30 * 1024**3, "H_free": 30 * 1024**3}
            class Tok:
                def decode(self, ids):
                    return native.SENTINEL
            with patch.object(native.windows_job, "run_owned", side_effect=fake_runtime), \
                 patch.object(native, "system_resources", return_value=resources), \
                 patch.object(native, "tokenizer", return_value=Tok()):
                self.assertEqual(native.run_cell(4096, "OFF", binding_sha=native.sha(base / "prepared/binding.json")), 0)
            result = json.loads((base / "native-4096-off/result.json").read_text())
            self.assertTrue(result["measurement_valid"])
            self.assertTrue(result["requests"][0]["recall_probe_pass"])
            self.assertEqual(result["P17_PPL_full_quality"], "NOT_RUN")
