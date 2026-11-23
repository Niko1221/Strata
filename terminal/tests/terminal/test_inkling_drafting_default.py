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
    def setUp(self):
        self._saved = inkling_family.EXACT_DRAFTING_BACKENDS

    def tearDown(self):
        inkling_family.EXACT_DRAFTING_BACKENDS = self._saved

    def test_off_until_a_backend_is_proven_exact(self):
        self.assertEqual(inkling_family.EXACT_DRAFTING_BACKENDS, ())
        for backend in (None, "cuda", "hip", "sycl", "metal"):
            self.assertEqual(inkling_family.default_drafting(backend), "off", backend)
        self.assertFalse(inkling_kind.wants_drafter({}))
        self.assertFalse(inkling_kind.wants_drafter({"backend": "cuda"}))
        args = inkling_kind.engine_args({"arch": "inkling2_loop", "args": ["--native", "m.strata"]}, announce=False)
        self.assertEqual(args[args.index("--draft") + 1], "off")

    def test_a_proven_backend_lets_a_measured_probe_decide(self):
        inkling_family.EXACT_DRAFTING_BACKENDS = ("cuda",)
        self.assertEqual(inkling_family.default_drafting("cuda"), "auto")
        self.assertEqual(inkling_family.default_drafting(None), "auto")
        self.assertEqual(inkling_family.default_drafting("hip"), "off")
        # "auto" is on only with a probe that says so (tests/terminal/test_inkling_probe.py); without one it is off
        self.assertFalse(inkling_kind.wants_drafter({"backend": "cuda"}))
        self.assertFalse(inkling_kind.wants_drafter({"backend": "hip"}))

    def test_an_explicit_choice_wins_both_ways(self):
        self.assertTrue(inkling_kind.wants_drafter({"drafting": "dspark"}))
        self.assertTrue(inkling_kind.wants_drafter({"drafting": "dspark", "backend": "hip"}))
        inkling_family.EXACT_DRAFTING_BACKENDS = ("cuda",)
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
