"""Inkling-2-Mini drafting default (0221-DSPARK): the drafter is on only where its verify window gives the same answers as plain decoding,
a person's explicit choice always wins, and an engine that lists spec_opts=1 is told to draft with one linear chain (--draft-beam 1)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import inkling_family  # noqa: E402
from serve import inkling_kind  # noqa: E402


class DraftingDefault(unittest.TestCase):
    def test_every_backend_starts_as_auto_and_is_off_until_a_probe_says_on(self):
        for backend in (None, "cuda", "hip", "sycl", "metal"):
            self.assertEqual(inkling_family.default_drafting(backend), "auto", backend)
        self.assertFalse(hasattr(inkling_family, "EXACT_DRAFTING_BACKENDS"))      # no bit-exact gate any more (owner, 2026-11-23)
        for cfg in ({}, {"backend": "cuda"}, {"backend": "hip"}, {"drafting": "auto"}):
            self.assertFalse(inkling_kind.wants_drafter(cfg), cfg)                 # no valid probe yet: off (tests/terminal/test_inkling_probe.py)
        args = inkling_kind.engine_args({"arch": "inkling2_loop", "args": ["--native", "m.strata"]}, announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "off")

    def test_an_explicit_choice_wins_both_ways(self):
        self.assertTrue(inkling_kind.wants_drafter({"drafting": "dspark"}))
        self.assertTrue(inkling_kind.wants_drafter({"drafting": "dspark", "backend": "hip"}))
        for val in ("off", "none", "false", "0", False):
            self.assertFalse(inkling_kind.wants_drafter({"drafting": val, "backend": "cuda"}), val)

    def test_the_window_shape_goes_to_an_engine_that_takes_it(self):
        cfg = {"arch": "inkling2_loop", "drafting": "dspark", "args": ["--native", "m.strata"]}
        self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "dspark", "--draft-beam", "1", "--draft-adapt"])
        # an older engine (no spec_opts) is not given a flag it would refuse
        self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "dspark"}), ["--draft", "dspark"])
        # the run config can ask for the beam tree
        cfg4 = dict(cfg, draft_beam=4)
        self.assertEqual(inkling_kind.drafter_args(cfg4, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "dspark", "--draft-beam", "4", "--draft-adapt"])
        # no drafter in the file: nothing
        self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "none", "spec_opts": "1"}), [])
        # drafting off still emits --draft off (#6670)
        self.assertEqual(inkling_kind.drafter_args(dict(cfg, drafting="off"), {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "off"])


if __name__ == "__main__":
    unittest.main()
