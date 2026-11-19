"""Differential and native-helper contract tests for `strata-procs` (#5973)."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from strata_control import native_helpers, procs


_RELEASE_DIR = Path(__file__).resolve().parents[2] / "crates" / "strata-procs" / "target" / "release"
_EXE = _RELEASE_DIR / ("strata-procs.exe" if os.name == "nt" else "strata-procs")


class StrataProcsDifferentialTest(unittest.TestCase):
    def setUp(self):
        if not _EXE.is_file():
            self.skipTest(f"release binary not built at {_EXE}")

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            self.assertIsNone(native_helpers.program("procs"))

    def test_bad_cli_exits_2_with_empty_stdout(self):
        done = subprocess.run([str(_EXE), "start-token", str(os.getpid())], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")

    def test_current_process_differential(self):
        pid = os.getpid()
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
            py_ppid = procs.parent_of(pid)
            py_exe = procs.exe_path(pid)
            py_cmd = procs.command_line(pid)
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "procs", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            rs_tok = procs.start_token(pid)
            rs_ppid = procs.parent_of(pid)
            rs_exe = procs.exe_path(pid)
            rs_cmd = procs.command_line(pid)
        self.assertIsInstance(rs_tok, str)
        self.assertTrue(len(rs_tok) > 0)
        if py_ppid is not None:
            self.assertEqual(rs_ppid, py_ppid)
        if py_exe is not None and rs_exe is not None:
            self.assertEqual(os.path.normcase(rs_exe), os.path.normcase(py_exe))
        if py_cmd is not None:
            self.assertEqual(rs_cmd, py_cmd)
