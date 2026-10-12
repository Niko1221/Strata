"""Guard against silently running setup instead of calibration when incompatible CLI flags are combined."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class CalibrationCliTests(unittest.TestCase):
    def test_conflicts_fail_before_setup_has_side_effects(self):
        for flags in (("--model", "IQ3_XXS"), ("--family", "qwen"), ("--setup",), ("--check",)):
            with self.subTest(flags=flags):
                stderr = io.StringIO()
                with mock.patch.object(sys, "argv", ["setup.py", "--calibrate", "--yes", *flags]), \
                     mock.patch.object(setup, "data_folder", side_effect=AssertionError("setup must not start")), \
                     contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        setup.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--calibrate cannot be combined", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
