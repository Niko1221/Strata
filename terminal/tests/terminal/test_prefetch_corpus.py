"""The routing-trace corpus validator reports both overlaps the engine could prefetch."""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

VULKAN = Path(__file__).resolve().parents[3] / "engine" / "vulkan"


def load():
    spec = importlib.util.spec_from_file_location("check_prefetch", VULKAN / "check_prefetch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrefetchCorpus(unittest.TestCase):
    def setUp(self):
        self.tool = load()

    def test_parse_trace_starts_a_new_token_at_layer_zero(self):
        tokens = self.tool.parse_trace("0 1 2\n1 3 4\n0 5 6\n1 7 8\n")
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0], [(0, [1, 2]), (1, [3, 4])])
        self.assertEqual(tokens[1], [(0, [5, 6]), (1, [7, 8])])

    def test_both_overlaps_are_computed(self):
        # token 0: layer 0 -> {1,2}, layer 1 -> {1,2} (100% consecutive)
        # token 1: layer 0 -> {1,2}, layer 1 -> {9,8} (0% consecutive,
        #          50% same-layer across tokens for layer 1: {1,2} vs {9,8}? 0;
        #          layer 0 is {1,2} vs {1,2} -> 100%)
        text = "0 1 2\n1 1 2\n0 1 2\n1 9 8\n"
        r = self.tool.analyze([text])
        self.assertEqual(r["tokens"], 2)
        self.assertAlmostEqual(r["mean_consecutive_overlap"], 0.5)
        self.assertAlmostEqual(r["mean_same_layer_overlap"], 0.5)

    def test_check_in_traces_stays_silent_on_success(self):
        with mock.patch.object(sys, "argv", ["check_prefetch"]):
            out = io.StringIO()
            with redirect_stdout(out):
                rc = self.tool.main()
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("consecutive layers", out.getvalue())

    def test_fail_under_turns_it_into_a_gate(self):
        with mock.patch.object(sys, "argv", ["check_prefetch", "--fail-under", "0.9"]), \
             redirect_stdout(io.StringIO()):
            rc = self.tool.main()
        self.assertEqual(rc, 1)

    def test_missing_trace_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "none.trace"
            with mock.patch.object(sys, "argv", ["check_prefetch", "--trace", str(missing)]), \
                 redirect_stdout(io.StringIO()) as out:
                rc = self.tool.main()
        self.assertEqual(rc, 1)
        self.assertIn("cannot read", out.getvalue())


if __name__ == "__main__":
    unittest.main()
