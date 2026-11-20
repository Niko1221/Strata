"""Differential and native-helper contract tests for `strata-sysinfo` (#5970)."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from strata_control import native_helpers, sysinfo


_RELEASE_DIR = Path(__file__).resolve().parents[2] / "crates" / "strata-sysinfo" / "target" / "release"
_EXE = _RELEASE_DIR / ("strata-sysinfo.exe" if os.name == "nt" else "strata-sysinfo")


class StrataSysinfoDifferentialTest(unittest.TestCase):
    def setUp(self):
        if not _EXE.is_file():
            self.skipTest(f"release binary not built at {_EXE}")

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            self.assertIsNone(native_helpers.program("sysinfo"))

    def test_bad_cli_exits_2_with_empty_stdout(self):
        done = subprocess.run([str(_EXE)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")

    def test_differential_against_python(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
            py = (
                sysinfo.system(),
                sysinfo.machine(),
                sysinfo.release(),
                sysinfo.version(),
                sysinfo.processor(),
                sysinfo.node(),
                sysinfo.platform_string(),
            )
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "sysinfo", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            rs = (
                sysinfo.system(),
                sysinfo.machine(),
                sysinfo.release(),
                sysinfo.version(),
                sysinfo.processor(),
                sysinfo.node(),
                sysinfo.platform_string(),
            )
        self.assertEqual(rs, py)
