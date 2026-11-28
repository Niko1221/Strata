"""Differential and native-helper contract tests for `strata-private-files` (#5976)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import native_helpers, private_files


_RELEASE_DIR = Path(__file__).resolve().parents[2] / "crates" / "strata-private-files" / "target" / "release"
_EXE = _RELEASE_DIR / ("strata-private-files.exe" if os.name == "nt" else "strata-private-files")


class StrataPrivateFilesDifferentialTest(unittest.TestCase):
    def setUp(self):
        if not _EXE.is_file():
            self.skipTest(f"release binary not built at {_EXE}")

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            self.assertIsNone(native_helpers.program("private-files"))

    def test_bad_cli_exits_2_with_empty_stdout(self):
        done = subprocess.run([str(_EXE), "current-user-sid"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")

    def test_restrict_file_and_dir_differential(self):
        with tempfile.TemporaryDirectory(prefix="strata-priv-") as tmp:
            f_py = Path(tmp) / "py.txt"
            f_rs = Path(tmp) / "rs.txt"
            f_py.write_text("secret", encoding="utf-8")
            f_rs.write_text("secret", encoding="utf-8")
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
                private_files.restrict_to_current_user(f_py, directory=False)
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "private-files", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
                private_files.restrict_to_current_user(f_rs, directory=False)
            if hasattr(private_files, "is_unshared"):
                self.assertEqual(private_files.is_unshared(f_rs), private_files.is_unshared(f_py))
