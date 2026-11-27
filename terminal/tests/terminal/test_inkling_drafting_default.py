"""Inkling-2-Mini drafting default (0221-DSPARK): the drafter is on only where its verify window gives the same answers as plain decoding,
a person's explicit choice always wins, and an engine that lists spec_opts=1 is told to draft with one linear chain (--draft-beam 1)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        # 0.2.23: drafting is DFlash 2 and nothing else, on when dflash2.safetensors is beside the model (a 1-byte file stands in for it under a patched pin)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(inkling_family, "DRAFTER_FILES", {**inkling_family.DRAFTER_FILES, inkling_family.DFLASH2_FILE: (1, "0" * 64)}), \
                mock.patch.dict("os.environ", {"STRATA_INK_DFLASH2": ""}):
            (Path(td) / inkling_family.DFLASH2_FILE).write_bytes(b"x")
            cfg = {"arch": "inkling2_loop", "drafting": "dflash2", "args": ["--native", str(Path(td) / "m.strata")]}
            self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "dflash2", "--draft-beam", "1", "--draft-adapt"])
            # an older engine (no spec_opts) is not given a flag it would refuse
            self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "dspark"}), ["--draft", "dflash2"])
            # the run config can ask for the beam tree
            cfg4 = dict(cfg, draft_beam=4)
            self.assertEqual(inkling_kind.drafter_args(cfg4, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "dflash2", "--draft-beam", "4", "--draft-adapt"])
            # the model file carries no DSpark (a GGUF): the DFlash 2 file is the drafter, the flag is the same
            self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "none", "spec_opts": "1"}), ["--draft", "dflash2", "--draft-beam", "1", "--draft-adapt"])
            # drafting off still emits --draft off (#6670)
            self.assertEqual(inkling_kind.drafter_args(dict(cfg, drafting="off"), {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "off"])
            # the old word means "on" too
            self.assertEqual(inkling_kind.drafter_args(dict(cfg, drafting="dspark"), None), ["--draft", "dflash2"])
        # without the file there is no drafter, whatever the model file carries
        cfg = {"arch": "inkling2_loop", "drafting": "dflash2", "args": ["--native", "m.strata"]}
        self.assertEqual(inkling_kind.drafter_args(cfg, {"drafter": "dspark", "spec_opts": "1"}), ["--draft", "off"])


if __name__ == "__main__":
    unittest.main()
