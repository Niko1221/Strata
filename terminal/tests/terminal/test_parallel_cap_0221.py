"""0221 APPTUI: `strata launch --parallel N` takes up to 64 (the GRID lane's 64-client server target) and says plainly when a family or a card cannot hold N.

Limits, from the engines: strata-inkling runs up to 64 slots (kMaxParallelSlots); the Strata engine's batch window holds 16 rows in the CUDA and HIP builds and
8 in the others (a lone card cannot run more); strata-hy4 answers one request. The default stays what it was (fit to the card).
Nothing is started."""

from __future__ import annotations

import argparse
import unittest

from strata_cli.launch import cmd
from strata_cli.launch import settings as S


def levels(s, **kw):
    return [(n.level, n.text) for n in S.check_parallel(s, **kw)]


class ParseAndHelp(unittest.TestCase):
    def test_the_flag_takes_1_to_64(self):
        self.assertEqual(cmd._parallel("64"), 64)
        self.assertEqual(cmd._parallel("1"), 1)
        for bad in ("0", "65", "x", "-3"):
            with self.assertRaises(argparse.ArgumentTypeError):
                cmd._parallel(bad)

    def test_the_help_does_not_say_8(self):
        parser = argparse.ArgumentParser()
        sp = parser.add_subparsers()
        cmd.add_parsers(sp)
        text = sp.choices["launch"].format_help()
        self.assertIn("1 to 64", text)
        self.assertNotIn("1 to 8,", text)


class PerFamily(unittest.TestCase):
    def test_inkling_64_on_a_big_card_is_fine(self):
        s = S.LaunchSettings(family="inkling-2-mini", model="SQ-20", context=32768, parallel=64)
        self.assertEqual(levels(s, vram_gb=32.0, ram_gb=64.0, vendor="nvidia"), [])

    def test_inkling_64_on_a_12_gb_card_warns_that_the_experts_shrink(self):
        s = S.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=32768, parallel=64)
        got = levels(s, vram_gb=12.0, ram_gb=64.0, vendor="nvidia")
        self.assertEqual([lv for lv, _ in got], ["warn"])
        self.assertIn("64 requests at once at 32K leave room for", got[0][1])
        self.assertIn("one request:", got[0][1])

    def test_inkling_64_on_an_8_gb_card_is_refused_before_the_start(self):
        s = S.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=32768, parallel=64)
        got = levels(s, vram_gb=8.0, ram_gb=32.0, vendor="nvidia")
        self.assertEqual([lv for lv, _ in got], ["error"])
        self.assertIn("the engine cannot start", got[0][1])
        # fewer slots on the same card are only a warning
        s.parallel = 4
        self.assertEqual([lv for lv, _ in levels(s, vram_gb=8.0, ram_gb=32.0, vendor="nvidia")], ["warn"])

    def test_one_request_is_never_a_notice(self):
        for fam in ("inkling-2-mini", "qwen", "hy4-flash"):
            s = S.LaunchSettings(family=fam, model="SQ-10" if fam == "inkling-2-mini" else "IQ3_S", parallel=1)
            self.assertEqual(levels(s, vram_gb=8.0, ram_gb=16.0, vendor="nvidia"), [], fam)

    def test_qwen_keeps_what_its_batch_window_holds_and_says_so(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_S", parallel=64)
        for vendor, cap in (("nvidia", 16), ("amd", 16), ("intel", 8), (None, 8)):
            got = levels(s, vram_gb=32.0, vendor=vendor)
            self.assertEqual([lv for lv, _ in got], ["warn"], vendor)
            self.assertIn(f"it keeps {cap}", got[0][1])
        self.assertEqual(S.max_parallel("qwen", "nvidia"), 16)
        self.assertEqual(S.max_parallel("qwen", "intel"), 8)
        self.assertEqual(S.max_parallel("inkling-2-mini"), 64)
        self.assertEqual(S.max_parallel("hy4-flash"), 1)

    def test_qwen_up_to_the_window_is_the_old_card_check(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_S", parallel=12)
        got = levels(s, vram_gb=12.0, vendor="nvidia")
        self.assertEqual([lv for lv, _ in got], ["warn"])
        self.assertIn("usually holds", got[0][1])

    def test_hy4_says_it_answers_one_at_a_time(self):
        s = S.LaunchSettings(family="hy4-flash", model="SQ-64", parallel=8)
        got = levels(s, vram_gb=32.0, vendor="nvidia")
        self.assertEqual([lv for lv, _ in got], ["warn"])
        self.assertIn("one request at a time", got[0][1])

    def test_more_than_64_is_an_error_whatever_the_family(self):
        s = S.LaunchSettings(family="inkling-2-mini", model="SQ-10", parallel=65)
        self.assertEqual([lv for lv, _ in levels(s, vram_gb=32.0, ram_gb=64.0)], ["error"])


class SetupGetsTheSlots(unittest.TestCase):
    def test_setup_args_carry_parallel_for_inkling_only(self):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home():
            ink = Runner(S.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=32768, parallel=64), catalog=catalog(HW_12)).setup_args()
            self.assertEqual(ink[ink.index("--parallel") + 1], "64")
            one = Runner(S.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=32768, parallel=1), catalog=catalog(HW_12)).setup_args()
            self.assertNotIn("--parallel", one)
            qwen = Runner(S.LaunchSettings(family="qwen", model="IQ3_S", context=32768, parallel=8), catalog=catalog(HW_12)).setup_args()
            self.assertNotIn("--parallel", qwen)

    def test_the_planner_makes_room_for_the_slots(self):
        from strata_control import inkling_fit as F
        one = F.place("SQ-10", 12.0, 64.0, False, 32768, False, parallel=1)
        many = F.place("SQ-10", 12.0, 64.0, False, 32768, False, parallel=64)
        self.assertLess(many["vram_gb"], one["vram_gb"])


if __name__ == "__main__":
    unittest.main()
