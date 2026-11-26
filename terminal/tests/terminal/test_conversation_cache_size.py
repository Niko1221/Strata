"""#7044: the conversation cache grows with the PC's RAM instead of stopping at the engine's 8 GiB.

Control passes the engine's own `--conversation-cache-mib N` (no engine change) only when N is above 8192, from the PC's RAM and the model's
size; every other start is left exactly as it was: a small PC, a model that fills the RAM, an explicit setting, parking turned off, an engine
program without the option, an unreadable RAM figure."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import conversation_cache_size as C  # noqa: E402

GIB = 1024


class Budget(unittest.TestCase):
    def test_a_big_pc_gets_a_share_of_its_ram(self):
        # 256 GiB, 240 free, a 100 GiB model: 40 % of (240 - 100 - 4 - 2.5) = 53.4 GiB; a quarter of the RAM is 64 GiB: the first
        mib = C.budget_mib(256 * GIB, 240 * GIB, 100 * GIB)
        self.assertEqual(mib, 54528)
        self.assertEqual(mib % 256, 0)
        self.assertGreater(mib, C.ENGINE_AUTO_CAP_MIB)

    def test_a_quarter_of_the_ram_is_the_most(self):
        # 128 GiB, 120 free, a 20 GiB model: 40 % of the room is 37.4 GiB, a quarter of the RAM is 32 GiB
        self.assertEqual(C.budget_mib(128 * GIB, 120 * GIB, 20 * GIB), 32 * GIB)

    def test_the_ceiling(self):
        self.assertEqual(C.budget_mib(2048 * GIB, 2000 * GIB, 100 * GIB), C.CEILING_MIB)

    def test_never_below_the_engines_eight_gib(self):
        # the cases where the number would not beat the engine's own cap are left to the engine: no flag
        self.assertIsNone(C.budget_mib(32 * GIB, 28 * GIB, 0))            # a quarter of 32 GiB is 8 GiB: not above the cap
        self.assertIsNone(C.budget_mib(64 * GIB, 55 * GIB, 40 * GIB))     # the model fills the RAM: 40 % of the 8.5 GiB left is 3.4
        self.assertIsNone(C.budget_mib(16 * GIB, 12 * GIB, 4 * GIB))
        self.assertIsNone(C.budget_mib(64 * GIB, 30 * GIB, 60 * GIB))     # more model than free RAM
        self.assertIsNone(C.budget_mib(0, 0, 0))
        self.assertIsNone(C.budget_mib(256 * GIB, 0, 0))

    def test_a_bigger_floor_leaves_less(self):
        self.assertLess(C.budget_mib(256 * GIB, 240 * GIB, 100 * GIB, 20 * GIB), C.budget_mib(256 * GIB, 240 * GIB, 100 * GIB))

    def test_growth_follows_the_ram(self):
        last = 0
        for total in (48, 64, 96, 128, 192, 256, 384, 512):
            mib = C.budget_mib(total * GIB, total * GIB * 0.9, 30 * GIB) or 0
            self.assertGreaterEqual(mib, last, total)
            last = mib
        self.assertGreater(last, 64 * GIB)


class Args(unittest.TestCase):
    BIG = (256.0 * GIB, 240.0 * GIB)

    def with_model(self, size_mib=0):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        f = Path(d.name) / "model.gguf"
        with open(f, "wb") as h:
            h.truncate(int(size_mib * (1 << 20)))
        return str(f)

    def test_the_flag_is_added_on_a_big_pc(self):
        said = []
        args = ["--native", self.with_model(1), "--max-context", "262144"]
        out = C.args_with_budget(args, ram=self.BIG, say=said.append)
        self.assertEqual(out[:len(args)], args)
        self.assertEqual(out[-2], "--conversation-cache-mib")
        self.assertGreater(int(out[-1]), 8192)
        self.assertEqual(len(said), 1)
        self.assertIn(out[-1], said[0])
        self.assertEqual(args, ["--native", args[1], "--max-context", "262144"])        # the caller's list is not changed

    def test_the_model_size_is_taken_off_the_free_ram(self):
        args = ["--native", self.with_model(1)]
        small = int(C.args_with_budget(args, ram=self.BIG)[-1])
        with mock.patch.object(C, "model_mib", return_value=100.0 * GIB):        # (no 100 GiB file is made for a test)
            big = int(C.args_with_budget(args, ram=self.BIG)[-1])
        self.assertGreater(small, big)
        sized = Path(args[1])
        self.assertAlmostEqual(C.model_mib(args), sized.stat().st_size / (1 << 20))

    def test_a_small_pc_is_left_alone(self):
        args = ["--native", self.with_model(1)]
        self.assertIs(C.args_with_budget(args, ram=(16.0 * GIB, 12.0 * GIB)), args)
        self.assertIs(C.args_with_budget(args, ram=(0.0, 0.0)), args)

    def test_what_the_person_set_is_never_touched(self):
        for extra in (["--conversation-cache-mib", "0"], ["--conversation-cache-mib", "24576"], ["--prompt-cache", "0"],
                      ["--conversation-cache-slots", "0"]):
            args = ["--native", self.with_model(1), *extra]
            self.assertIs(C.args_with_budget(args, ram=self.BIG), args, extra)
        args = ["--native", self.with_model(1), "--prompt-cache", "6", "--conversation-cache-slots", "2"]
        self.assertEqual(C.args_with_budget(args, ram=self.BIG)[-2], "--conversation-cache-mib")      # (other values do not stop it)

    def test_the_switch_in_the_environment(self):
        args = ["--native", self.with_model(1)]
        with mock.patch.dict("os.environ", {"STRATA_NO_AUTO_CONVERSATION_CACHE": "1"}):
            self.assertIs(C.args_with_budget(args, ram=self.BIG), args)

    def test_the_engines_own_floor_is_used(self):
        base = ["--native", self.with_model(1)]
        low = int(C.args_with_budget(base, ram=(64.0 * GIB, 60.0 * GIB))[-1])
        high = int(C.args_with_budget(base + ["--conversation-cache-min-free-mib", "20000"], ram=(64.0 * GIB, 60.0 * GIB))[-1]) \
            if C.budget_mib(64.0 * GIB, 60.0 * GIB, 1, 20000) else None
        self.assertGreater(low, 8192)
        self.assertTrue(high is None or high < low)

    def test_a_missing_model_file_counts_as_nothing(self):
        self.assertEqual(C.model_mib(["--native", "/no/such/file.gguf"]), 0.0)
        self.assertEqual(C.model_mib(["--max-context", "4096"]), 0.0)


if __name__ == "__main__":
    unittest.main()
