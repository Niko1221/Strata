"""Synthetic receipt tests, not hardware performance evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.analyze_gdn_native_profile import compare


class NativeProfileAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.baseline = Path(self.directory.name) / "baseline"
        self.profile = Path(self.directory.name) / "profile"
        binding = dict(binary_sha256="fixture", capacity=4096, prompt_tokens=[1, 2],
                       max_new_per_request=2, requests=1, budget={}, tokenizer_hashes={},
                       profile_sha256="fixture", model_files=[], command=["fixture", "--pcie-frac", "0.18"],
                       instrumented_env={"STRATA_ROUTE_TAIL_SKIP": "0"})
        receipt = dict(exit_code=0, termination_confirmed=True, job_active_after=0, observed_dropped_bytes=0)
        request = dict(count=2, tokens=[3, 4], decode_ms=10.0, prompt_ms=20.0)
        for path in (self.baseline, self.profile):
            path.mkdir()
            self.write(path, "binding.json", binding)
            self.write(path, "result.json", receipt)
            self.write(path, "summary.json", dict(pass_=True, requests=[request], timing_lines=[]))
            summary = self.read(path, "summary.json")
            summary["pass"] = summary.pop("pass_")
            self.write(path, "summary.json", summary)
        binding["instrumented_env"].update(STRATA_VERIFY_PROFILE="1", STRATA_DECODE_TIMING="1")
        self.write(self.profile, "binding.json", binding)
        summary = self.read(self.profile, "summary.json")
        summary["timing_lines"] = [
            "strata decode timing: 2 windows, avg T 1.00, 1.00 tokens/window, 5.00 ms/window",
            "strata decode GPU stages (ms/window): GDN layers: q8+qkv/q-idx gemv 0.50 rec 0.25 waitCPU 1.00"
            " | QSA layers: attention 0.25 waitCPU 1.00 | total 3.00 ms/window over 2 windows",
        ]
        self.write(self.profile, "summary.json", summary)

    @staticmethod
    def write(path, name, value):
        (path / name).write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def read(path, name):
        return json.loads((path / name).read_text(encoding="utf-8"))

    def test_matching_completed_fixture_parses_units_and_named_stages(self):
        result = compare(self.baseline, self.profile)
        self.assertTrue(result["emitted_token_ids_equal"])
        self.assertEqual(result["requests"][0]["baseline_ms_per_emitted_token"], 5.0)
        self.assertEqual(result["profiles"][0]["gpu_instrumented_stages_ms_per_window"]["GDN"]["rec"], 0.25)

    def test_failed_run_cannot_become_a_timing_result(self):
        receipt = self.read(self.baseline, "result.json")
        receipt["exit_code"] = 125
        self.write(self.baseline, "result.json", receipt)
        with self.assertRaisesRegex(ValueError, "Incomplete or failed"):
            compare(self.baseline, self.profile)

    def test_changed_runtime_argument_is_rejected(self):
        binding = self.read(self.profile, "binding.json")
        binding["command"][-1] = "0.5"
        self.write(self.profile, "binding.json", binding)
        with self.assertRaisesRegex(ValueError, "Unmatched engine arguments"):
            compare(self.baseline, self.profile)

    def test_prompt_windows_contaminating_gpu_average_are_detected(self):
        summary = self.read(self.profile, "summary.json")
        summary["timing_lines"][1] = summary["timing_lines"][1].replace("over 2 windows", "over 27 windows")
        self.write(self.profile, "summary.json", summary)
        self.assertFalse(compare(self.baseline, self.profile)["gpu_host_window_counts_match"])

    def test_changed_output_tokens_are_reported_without_claiming_parity(self):
        summary = self.read(self.profile, "summary.json")
        summary["requests"][0]["tokens"] = [3, 5]
        self.write(self.profile, "summary.json", summary)
        self.assertFalse(compare(self.baseline, self.profile)["emitted_token_ids_equal"])


if __name__ == "__main__":
    unittest.main()