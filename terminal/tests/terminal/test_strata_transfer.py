"""Differential and native-helper contract tests for `strata-transfer` (#5974)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import native_helpers, transfer


_RELEASE_DIR = Path(__file__).resolve().parents[2] / "crates" / "strata-transfer" / "target" / "release"
_EXE = _RELEASE_DIR / ("strata-transfer.exe" if os.name == "nt" else "strata-transfer")


class StrataTransferDifferentialTest(unittest.TestCase):
    def setUp(self):
        if not _EXE.is_file():
            self.skipTest(f"release binary not built at {_EXE}")

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            self.assertIsNone(native_helpers.program("transfer"))

    def test_bad_cli_exits_2_with_empty_stdout(self):
        done = subprocess.run([str(_EXE), "sha256-file", "missing.bin"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")

    def test_sha256_file_differential(self):
        with tempfile.TemporaryDirectory(prefix="strata-xfer-") as tmp:
            sample = Path(tmp) / "sample.bin"
            sample.write_bytes(b"strata-transfer-payload-\x00\xff" * 1024)
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
                py = transfer.sha256_file(sample)
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "transfer", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
                rs = transfer.sha256_file(sample)
            self.assertEqual(rs, py)
