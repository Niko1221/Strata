"""Differential and native-helper contract tests for `strata-feed` (#5975)."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from strata_control import feed, native_helpers
from strata_control.feed import FeedError


_RELEASE_DIR = Path(__file__).resolve().parents[2] / "crates" / "strata-feed" / "target" / "release"
_EXE = _RELEASE_DIR / ("strata-feed.exe" if os.name == "nt" else "strata-feed")


class StrataFeedDifferentialTest(unittest.TestCase):
    def setUp(self):
        if not _EXE.is_file():
            self.skipTest(f"release binary not built at {_EXE}")

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            self.assertIsNone(native_helpers.program("feed"))

    def test_bad_cli_exits_2_with_empty_stdout(self):
        done = subprocess.run([str(_EXE), "validate"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")

    def test_valid_and_rejected_differential(self):
        good = b'{"schema_version":1,"sequence":9007199254740993,"channel":"stable","generated_at":"2026-11-14T00:00:00Z"}'
        bad = b'{"schema_version":2,"sequence":42,"channel":"stable","generated_at":"2026-11-14T00:00:00Z"}'
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
            py_feed = feed.parse(good, source="https://example.test/feed.json", key_id="k1")
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "feed", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            rs_feed = feed.parse(good, source="https://example.test/feed.json", key_id="k1")
            with self.assertRaises(FeedError):
                feed.parse(bad, source="https://example.test/feed.json", key_id="k1")
        self.assertEqual(rs_feed.sequence, py_feed.sequence)
        self.assertEqual(rs_feed.channel, py_feed.channel)
