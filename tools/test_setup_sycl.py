"""Tests for setup.py --backend sycl, the experimental Intel Arc engine (sycl/, PR #423): it warns, stops with a clear
message on Windows, and on Linux hands the run to sycl/setup_intel.py without its --backend flag. The CUDA / HIP
paths never see it. No GPU, no network.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(argv, win):
    out = io.StringIO()
    with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
            contextlib.redirect_stdout(out):
        try:
            rc = setup.sycl_setup(argv)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue(), call


class SyclBackend(unittest.TestCase):
    def test_windows_stops_with_the_docs_pointer(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("EXPERIMENTAL", out)
        self.assertIn("docs/INTEL_ARC.md", out)
        call.assert_not_called()

    def test_linux_hands_over_to_setup_intel_without_the_backend_flag(self):
        rc, out, call = run(["--model", "IQ2_XS", "--backend", "sycl", "--port", "8085"], win=False)
        self.assertEqual(rc, 0)
        self.assertIn("EXPERIMENTAL", out)
        self.assertIn("built from source", out)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], ["--model", "IQ2_XS", "--port", "8085"])

    def test_backend_equals_form_is_dropped_too(self):
        _, _, call = run(["--backend=sycl", "--check"], win=False)
        self.assertEqual(call.call_args[0][0][2:], ["--check"])

    def test_the_parser_accepts_sycl(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'"--backend", choices=\["cuda", "hip", "sycl"\]')

    def test_setup_intel_finds_the_setup_functions_it_replaces(self):
        """sycl/setup_intel.py swaps these steps; setup.py must still have them (it stops with a message otherwise)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        names = re.search(r'for name in \(([^)]*)\):', src).group(1)
        for name in re.findall(r'"(\w+)"', names):
            self.assertTrue(callable(getattr(setup, name, None)), name)


class CMakeOption(unittest.TestCase):
    def test_sycl_is_off_by_default_and_exclusive(self):
        src = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
        self.assertIn('option(STRATA_ENABLE_SYCL "EXPERIMENTAL', src)
        self.assertRegex(src, r'option\(STRATA_ENABLE_SYCL "[^"]*" OFF\)')
        self.assertIn("add_subdirectory(sycl)", src)


if __name__ == "__main__":
    unittest.main()
