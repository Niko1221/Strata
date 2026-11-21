"""The Vulkan research benchmark helpers stay honest about what they measure."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
VULKAN = Path(__file__).resolve().parents[3] / "engine" / "vulkan"


def load(name: str):
    path = VULKAN / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchDecode(unittest.TestCase):
    def setUp(self):
        self.bench = load("bench_decode.py")

    def test_reads_engine_timings_and_computes_rates(self):
        stdout = ("INFO engine=strata\nREADY 4096 stop\n"
                  "PP 7 7 1400.0 5.0\n"
                  "T 271\nT 248068\nT 198\nT 760\n"
                  "DONE 4 7 1400.0 800.0 length\n")
        args = SimpleNamespace(model="m", ctx=4096, threads=8, tokens=4, timeout=10)
        completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
        with mock.patch.object(self.bench.subprocess, "run", return_value=completed):
            run = self.bench.run_once("native", [1, 2, 3], args)
        self.assertEqual(run["prompt_tokens"], 7)
        self.assertEqual(run["prompt_ms"], 1400.0)
        self.assertEqual(run["generated"], 4)
        self.assertEqual(run["decode_ms"], 800.0)
        self.assertEqual(run["finish"], "length")
        self.assertEqual(4 / 0.8, run["generated"] / (run["decode_ms"] / 1000))

    def test_failed_engine_is_reported(self):
        completed = subprocess.CompletedProcess([], 2, stdout="", stderr="no such device")
        args = SimpleNamespace(model="m", ctx=4096, threads=8, tokens=4, timeout=10)
        with mock.patch.object(self.bench.subprocess, "run", return_value=completed):
            run = self.bench.run_once("native", [1], args)
        self.assertEqual(run["returncode"], 2)
        self.assertIn("no such device", run["error"])


class BenchNvme(unittest.TestCase):
    def setUp(self):
        self.bench = load("bench_nvme.py")

    def test_parses_fio_json_with_latency(self):
        payload = {"jobs": [{"read": {"iops": 1234.5, "bw": 2048 * 1024,
                                      "clat_ns": {"mean": 500000,
                                                  "percentile": {"50.000000": 400000, "99.000000": 900000}},
                                      "io_bytes": 4096}}], "error": 0}
        completed = subprocess.CompletedProcess([], 0,
                                                stdout="note: queue depth capped\n" + json.dumps(payload), stderr="")
        with mock.patch.object(self.bench.subprocess, "run", return_value=completed):
            row = self.bench.run_one("fio", Path("/tmp/x"), 4096, 8, 1, True, 0)
        self.assertEqual(row["iops"], 1234.5)
        self.assertEqual(row["bw_mib_s"], 2048.0)
        self.assertEqual(row["p50_us"], 400.0)
        self.assertEqual(row["p99_us"], 900.0)

    def test_fio_error_is_captured(self):
        completed = subprocess.CompletedProcess([], 1, stdout="", stderr="permission denied")
        with mock.patch.object(self.bench.subprocess, "run", return_value=completed):
            row = self.bench.run_one("fio", Path("/tmp/x"), 4096, 1, 1, True, 0)
        self.assertIn("permission denied", row["error"])


if __name__ == "__main__":
    unittest.main()
