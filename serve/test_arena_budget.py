"""serve/test_arena_budget.py - "arena_budget": the expert arena's RAM cap as a share of this PC's memory, not a fixed
number.  The share is converted at every start from the RAM the machine actually has, clamped at both ends
("min_gib", "headroom_gib"), and passed to the engine as --resident-budget-gib (which implies --mmap-experts).  An
absent key, "auto", a bad value or a hand-written --resident-budget-gib leave the engine's arguments untouched.

    python -m unittest serve.test_arena_budget
"""
from __future__ import annotations

import io
import unittest

from serve import arena_budget
from serve.arena_budget import Plan, args_for, plan, settings


class Off(unittest.TestCase):
    def test_no_key_no_cap(self):
        for cfg in ({}, {"arena_budget": None}, {"arena_budget": "auto"}, {"arena_budget": " AUTO "},
                    {"arena_budget": False}):
            self.assertIsNone(plan(cfg, 64), cfg)
            self.assertEqual(args_for(cfg, ["--pack", "x"], total_gib=64, out=io.StringIO()), ["--pack", "x"], cfg)

    def test_nothing_is_printed_when_off(self):
        out = io.StringIO()
        arena_budget.reset()
        args_for({}, ["--pack", "x"], total_gib=64, out=out)
        self.assertEqual(out.getvalue(), "")

    def test_ram_unreadable_is_not_a_guess(self):
        self.assertIsNone(plan({"arena_budget": 0.45}, 0))
        self.assertIsNone(plan({"arena_budget": 0.45}, -1))
        self.assertEqual(args_for({"arena_budget": 0.45}, ["--pack", "x"], total_gib=0, out=io.StringIO()),
                         ["--pack", "x"])


class Share(unittest.TestCase):
    """The same key on different hardware: the GiB follow the machine, which is the whole point of a share."""

    def test_the_share_follows_the_machine(self):
        for total, want in ((16, 7), (32, 14), (64, 28), (128, 57), (256, 115)):
            p = plan({"arena_budget": 0.45}, total)
            self.assertEqual(p.gib, want, f"{total} GiB -> {p.gib}, wanted {want}")

    def test_the_shorthand_and_the_object_agree(self):
        a = plan({"arena_budget": 0.45}, 64)
        b = plan({"arena_budget": {"share": 0.45}}, 64)
        self.assertEqual(a, b)
        self.assertEqual(a.share, arena_budget.SHARE_DEFAULT)

    def test_true_is_the_default_share(self):
        p = plan({"arena_budget": True}, 64)
        self.assertEqual((p.share, p.min_gib, p.headroom_gib), (arena_budget.SHARE_DEFAULT,
                                                               arena_budget.MIN_GIB_DEFAULT,
                                                               arena_budget.HEADROOM_GIB_DEFAULT))

    def test_never_above_the_cap(self):
        # 0.45 x 63.7 = 28.665 GiB: a rounding up would be over the share the user asked for
        self.assertEqual(plan({"arena_budget": 0.45}, 63.7).gib, 28)

    def test_the_arguments(self):
        # --resident-budget-gib alone: the engine sets mmap_experts / resident_cpu_experts / resident_pin itself
        self.assertEqual(plan({"arena_budget": 0.45}, 64).args, ["--resident-budget-gib", "28"])
        got = args_for({"arena_budget": 0.45}, ["--pack", "x", "--kv", "int8"], total_gib=64, out=io.StringIO())
        self.assertEqual(got, ["--pack", "x", "--kv", "int8", "--resident-budget-gib", "28"])
        self.assertNotIn("--mmap-experts", got)          # redundant, and refused beside some pack layouts

    def test_the_caller_s_list_is_not_changed(self):
        args = ["--pack", "x"]
        args_for({"arena_budget": 0.45}, args, total_gib=64, out=io.StringIO())
        self.assertEqual(args, ["--pack", "x"])


class Clamp(unittest.TestCase):
    def test_headroom_holds_it_down(self):
        p = plan({"arena_budget": {"share": 0.5, "headroom_gib": 40}}, 64)      # 32 wanted, 24 allowed
        self.assertEqual((p.gib, p.clamped), (24, "ceiling"))

    def test_the_floor_raises_it(self):
        p = plan({"arena_budget": {"share": 0.1, "min_gib": 12}}, 64)           # 6.4 wanted, 12 is the floor
        self.assertEqual((p.gib, p.clamped), (12, "floor"))

    def test_the_floor_never_beats_the_headroom(self):
        # min_gib 100 on a 64 GB PC would leave less than nothing: the ceiling wins and the floor is not applied
        p = plan({"arena_budget": {"min_gib": 100}}, 64)
        self.assertEqual((p.gib, p.clamped), (28, "share"))

    def test_a_pc_too_small_for_a_cap_gets_none(self):
        self.assertIsNone(plan({"arena_budget": 0.5}, 8))                       # 8 - 8 = 0 GiB of ceiling
        self.assertIsNone(plan({"arena_budget": 0.1}, 8.5))                     # 0.5 GiB: under one whole GiB

    def test_how_the_scenarios_move_it(self):
        # a PC used for modelling keeps more for the other program, one used only for the model keeps less
        busy = plan({"arena_budget": {"share": 0.35, "headroom_gib": 24}}, 64)
        alone = plan({"arena_budget": {"share": 0.75, "headroom_gib": 8}}, 64)
        self.assertEqual(busy.gib, 22)
        self.assertEqual(alone.gib, 48)


class Refused(unittest.TestCase):
    def test_out_of_range(self):
        for v in (0.05, 0.9, 0.95, -1, 2, 100):
            with self.assertRaises(ValueError, msg=v):
                settings(v)

    def test_wrong_type(self):
        for v in ("0.45", [0.45], {"share": "0.45"}, {"share": 0.45, "extra": 1}, {"min_gib": -1}):
            with self.assertRaises(ValueError, msg=v):
                settings(v)

    def test_a_bad_value_is_ignored_not_fatal(self):
        out = io.StringIO()
        arena_budget.reset()
        args = args_for({"arena_budget": 0.99}, ["--pack", "x"], total_gib=64, out=out)
        self.assertEqual(args, ["--pack", "x"])                  # the engine sizes the arena itself, as before
        self.assertIn("arena_budget", out.getvalue())
        self.assertIn("0.1-0.85", out.getvalue())


class Explicit(unittest.TestCase):
    def test_a_number_written_by_hand_wins(self):
        args = ["--pack", "x", "--resident-budget-gib", "40"]
        self.assertEqual(args_for({"arena_budget": 0.45}, args, total_gib=64, out=io.StringIO()), args)
        self.assertNotIn("28", args_for({"arena_budget": 0.45}, args, total_gib=64, out=io.StringIO()))


class Said(unittest.TestCase):
    def test_the_line_names_the_share_the_ram_and_the_result(self):
        line = plan({"arena_budget": 0.45}, 63.7).describe()
        for part in ("0.45", "63.7", "28", "--resident-budget-gib", "35.7"):
            self.assertIn(part, line)

    def test_said_once_per_start(self):
        out = io.StringIO()
        arena_budget.reset()
        cfg = {"arena_budget": 0.45}
        args_for(cfg, ["--pack", "x"], total_gib=64, out=out)    # engine_args runs twice at a start
        args_for(cfg, ["--pack", "x"], total_gib=64, out=out)
        self.assertEqual(out.getvalue().count("arena_budget:"), 1)


class Wiring(unittest.TestCase):
    """The server builds the engine's arguments through engine_args: the cap arrives there, and nothing else moves."""

    def test_engine_args(self):
        from serve.server import engine_args
        cfg = {"args": ["--pack", "data/packs/q2_0", "--kv", "int8"], "arena_budget": 0.45}
        got = engine_args(cfg)
        want = plan({"arena_budget": 0.45}, None).gib          # this PC's own RAM, read the same way
        self.assertEqual(got, ["--pack", "data/packs/q2_0", "--kv", "int8", "--resident-budget-gib", str(want)])

    def test_engine_args_without_the_key_is_byte_identical(self):
        from serve.server import engine_args
        cfg = {"args": ["--pack", "data/packs/q2_0", "--kv", "int8"]}
        self.assertEqual(engine_args(cfg), cfg["args"])


if __name__ == "__main__":
    unittest.main()
