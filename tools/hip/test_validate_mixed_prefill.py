#!/usr/bin/env python3
"""Unit tests for helpers in validate_mixed_prefill.py."""
import importlib.util
import pathlib
import sys
import types
import unittest

# The helper under test does not need tokenizer dependencies; use a stub so this remains a stdlib-only test.
sys.modules.setdefault("strata_tokenizer", types.ModuleType("strata_tokenizer"))


SCRIPT = pathlib.Path(__file__).with_name("validate_mixed_prefill.py")
spec = importlib.util.spec_from_file_location("validate_mixed_prefill", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WithoutVisionArgsTests(unittest.TestCase):
    def test_removes_flag_and_value(self):
        self.assertEqual(runner.without_vision_args(["--mtp", "--vision", "cpu", "--spec", "4"]),
                         ["--mtp", "--spec", "4"])

    def test_removes_flag_without_consuming_next_option(self):
        self.assertEqual(runner.without_vision_args(["--vision", "--mtp", "--vision=cpu"]), ["--mtp"])

    def test_keeps_unrelated_arguments(self):
        args = ["--model", "local/model", "--max-context", "90000"]
        self.assertEqual(runner.without_vision_args(args), args)


if __name__ == "__main__":
    unittest.main()
