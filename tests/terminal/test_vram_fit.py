"""strata_control/vram_fit.py: the fit math with mock card sizes, and its use in setup (sycl/setup_intel.py) and strata launch."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import vram_fit as V  # noqa: E402


class Measured(unittest.TestCase):
    """The estimate against the 12 starts on the Arc A750 (8 GB): engine slots / MiB short, from /data/022-arcgyro/out/fa_*.txt."""

    def slack(self, ctx, reserve, prefill):
        return V.slack_mib("Gyro-M", 8.0, ctx, "int8", reserve, prefill, "intel")

    def test_the_session_matches_the_engines_three_numbers(self):
        for ctx, mib in ((8192, 234), (16384, 347), (32768, 573)):
            self.assertAlmostEqual(V.session_mib(ctx), mib, delta=8)

    def test_the_default_does_not_start_and_the_estimate_says_how_short(self):
        self.assertAlmostEqual(self.slack(32768, 300, "auto"), -747, delta=40)     # engine: about 748 MiB more
        self.assertAlmostEqual(self.slack(16384, 300, "auto"), -497, delta=40)     # 497
        self.assertAlmostEqual(self.slack(8192, 100, "auto"), -171, delta=40)      # 171

    def test_the_known_good_start_has_room_for_its_slots(self):
        self.assertAlmostEqual(self.slack(8192, 100, 1024), 412 * V.SLOT_MIB, delta=60)    # 412 slots
        self.assertAlmostEqual(self.slack(32768, 100, 1024), 107 * V.SLOT_MIB, delta=60)   # 107 slots
        self.assertAlmostEqual(self.slack(8192, 100, 512), 683 * V.SLOT_MIB, delta=60)     # 683 slots
        self.assertAlmostEqual(self.slack(8192, 100, 256), 820 * V.SLOT_MIB, delta=60)     # 820 slots

    def test_gyro_s_default_fits_as_it_did(self):
        self.assertTrue(V.fits("Gyro-S", 8.0, 32768, "int8", 300, "auto", "intel"))        # 410 slots measured


class Plan(unittest.TestCase):
    def test_gyro_m_on_8gb_is_lowered_until_it_fits_and_says_so(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "auto", "intel", title="Gyro-M")
        self.assertTrue(f.changed and f.fits)
        self.assertEqual((f.ctx, f.reserve_mib, f.prefill), (16384, 100, "1024"))
        self.assertTrue(f.line.startswith("Gyro-M on this 8 GB card: context set to 16K"), f.line)
        self.assertTrue(f.line.endswith("so it fits"))
        self.assertGreaterEqual(f.slack_mib, V.needed_mib())

    def test_a_chosen_context_is_kept_and_the_rest_is_lowered(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "auto", "intel", ctx_fixed=True)
        self.assertEqual(f.ctx, 32768)
        self.assertFalse(f.fits)                           # 32K cannot fit even with 100 / 512: said, not refused
        self.assertIn("still tight", f.line)

    def test_a_chosen_reserve_is_kept(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 500, "auto", "intel", reserve_fixed=True)
        self.assertEqual(f.reserve_mib, 500)
        self.assertLess(f.ctx, 32768)

    def test_a_chosen_chunk_is_kept(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "2048", "intel", prefill_fixed=True)
        self.assertIsNone(f.prefill)

    def test_the_smallest_start_is_returned_when_nothing_fits_never_a_refusal(self):
        f = V.plan("Gyro-M", 6.0, 32768, "int8", 300, "auto", "intel")
        self.assertEqual(f.ctx, 8192)
        self.assertFalse(f.fits)
        self.assertTrue(f.line)

    def test_only_the_ladder_is_used(self):
        for gb in (7.0, 7.5, 8.0, 8.5, 9.0, 10.0):
            f = V.plan("Gyro-M", gb, 32768, "int8", 300, "auto", "intel")
            self.assertIn(f.ctx, (32768, 16384, 8192))

    def test_a_config_that_fits_comes_back_unchanged(self):
        for model, gb, ctx in (("Gyro-M", 12.0, 32768), ("Gyro-M", 16.0, 65536), ("Gyro-S", 8.0, 32768), ("Gyro-S", 12.0, 131072),
                               ("Qwen3.8-Flash-Next-IQ3_XXS", 8.0, 32768), ("Qwen3.8-Flash-Next-IQ3_XXS", 12.0, 131072),
                               ("Qwen3.8-Flash-Next-IQ3_XXS", 16.0, 262144), ("Qwen3.8-Flash-Next-IQ2_XS", 24.0, 262144)):
            f = V.plan(model, gb, ctx, "int8", 300, "auto", "nvidia")
            self.assertEqual((f.ctx, f.reserve_mib, f.prefill, f.changed, f.line), (ctx, 300, None, False, ""), (model, gb, ctx))

    def test_unknown_card_size_and_tiny_cards_are_left_alone(self):
        for gb in (0, None, 4.0):
            f = V.plan("Gyro-M", gb, 32768, "int8", 300, "auto", "intel")
            self.assertFalse(f.changed)

    def test_fit_context_is_the_ladder_step(self):
        self.assertEqual(V.fit_context("Gyro-M", 8.0, 32768), 16384)
        self.assertEqual(V.fit_context("Gyro-M", 12.0, 32768), 32768)
        self.assertEqual(V.fit_context("Gyro-M", 8.0, 8192), 8192)

    def test_generic_model_is_found_short_only_when_far_short(self):
        self.assertFalse(V.plan("Qwen3.8-Flash-Next-IQ3_XXS", 8.0, 32768, "int8", 300, "auto", "intel").changed)
        self.assertTrue(V.plan("Qwen3.8-Flash-Next-IQ3_XXS", 6.0, 131072, "int8", 300, "auto", "intel").changed)


class SetupIntel(unittest.TestCase):
    @staticmethod
    def mod():
        spec = importlib.util.spec_from_file_location("setup_intel_fit", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, native="Gyro-M-TQ2_0.gguf", extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / native),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", "--vram-reserve-mib", "300", *extra]}

    def val(self, c, flag):
        return c["args"][c["args"].index(flag) + 1]

    def run_it(self, m, cfg, vram, drv="i915", **kw):
        notes = []
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, vram, drv, "56a1", fit_notes=notes, **kw)
        return c, notes

    def test_gyro_m_on_the_a750_gets_a_config_that_starts_and_one_line(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=True)
        self.assertEqual(self.val(c, "--max-context"), "16384")
        self.assertEqual(self.val(c, "--vram-reserve-mib"), "100")
        self.assertEqual(self.val(c, "--prefill"), "1024")
        self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
        self.assertEqual(len(notes), 1)
        self.assertIn("Gyro-M on this 8 GB card: context set to 16K", notes[0])

    def test_a_reserve_the_user_asked_for_is_kept(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=False)
        self.assertEqual(self.val(c, "--vram-reserve-mib"), "300")
        self.assertIn(self.val(c, "--max-context"), ("16384", "8192"))

    def test_a_context_the_user_asked_for_is_kept(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=True, ctx_asked=True)
        self.assertEqual(self.val(c, "--max-context"), "32768")

    def test_configs_that_fit_are_byte_identical(self):
        m = self.mod()
        for vram, drv in ((8.0, "i915"), (12.0, "xe"), (16.0, "i915"), (24.0, "xe"), (32.0, "xe")):
            for native in ("a.gguf", "Gyro-S-TQ1_0.gguf"):
                off, _ = self.run_it(m, self.cfg(m, native), vram, drv, fit=False)
                on, notes = self.run_it(m, self.cfg(m, native), vram, drv, auto_reserve=True)
                self.assertEqual(off, on, (vram, drv, native))
                self.assertEqual(notes, [])


FN = "qwen3.8-flash-next-iq3_xxs"


class SlotSessions(unittest.TestCase):
    """Request slots ("Concurrent requests" 2 or more, the engine's --batch N): each slot is a session of its own, out of the expert cache's room.
    The 8 GB Arc A750 figures are the engine's: --batch 2 at 8192 tokens with the MTP layer and a 1024 MiB reserve left 2519 MiB free and the cache needed 99 MiB
    more (0 slots, a refused start; a SIGSEGV before ab7baa5b1)."""

    def test_the_slot_session_is_the_main_sessions_size(self):
        self.assertAlmostEqual(V.session_mib(8192), 234, delta=8)       # the engine: "2 slot sessions (0.23 GiB each)"

    def test_kv_streaming_caps_what_a_session_keeps_in_vram(self):
        full = V.session_mib(131072)
        streamed = V.session_mib(131072, "int8", 32768)
        self.assertAlmostEqual(full, 1929, delta=15)
        self.assertAlmostEqual(streamed, 621, delta=30)                  # the 32K pool (573 MiB with its fixed part) plus what scales with the whole context
        self.assertEqual(V.session_mib(8192, "int8", 32768), V.session_mib(8192))     # a resident window larger than the context changes nothing
        self.assertAlmostEqual(V.session_mib(131072, "int8", 4096), V.session_mib(131072, "int8", V.KV_RESIDENT_MIN), delta=0.01)   # the engine raises a small window to 20480

    def test_without_slots_nothing_changes(self):
        a = V.slack_mib(FN, 8.0, 8192, "int8", 700, "auto", "intel")
        self.assertEqual(a, V.slack_mib(FN, 8.0, 8192, "int8", 700, "auto", "intel", 0))
        self.assertEqual(a, V.slack_mib(FN, 8.0, 8192, "int8", 700, "auto", "intel", 1))
        self.assertEqual(V.needed_mib(), V.needed_mib(FN, 1))

    def test_the_engines_own_minimum_is_the_threshold(self):
        self.assertAlmostEqual(V.needed_mib(FN, 2, "intel"), V.SLOT_RUN_MIN * V.SLOT_SLOTTED_MIB + V.MARGIN_MIB, delta=0.01)    # one slot and the margin, not 128 slots
        self.assertGreater(V.needed_mib(FN, 0), 128 * 1.2)                                                                          # a single request keeps its old rule

    def test_an_8gb_arc_with_two_requests_and_the_refused_reserve_gets_a_context_that_fits(self):
        # 8192 with a 1024 MiB reserve is the start the engine refused (the estimate says 137 MiB short; the engine 99)
        self.assertLess(V.slack_mib(FN, 8.0, 8192, "int8", 1024, "auto", "intel", 2), 0)
        f = V.plan(FN, 8.0, 8192, "int8", 1024, "auto", "intel", title=FN, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertTrue(f.changed and f.fits)
        self.assertEqual((f.ctx, f.slots, f.reserve_mib), (2048, 2, 1024))
        self.assertGreaterEqual(f.slack_mib, V.needed_mib(FN, 2, "intel"))
        self.assertIn("context set to 2K", f.line)
        self.assertTrue(f.line.endswith("so 2 requests at once fit"), f.line)

    def test_an_8gb_arc_whose_start_the_engine_runs_keeps_its_context(self):
        # a 700 MiB reserve at 8192 leaves the cache about 84 slots: the engine runs it, so it is not lowered
        f = V.plan(FN, 8.0, 8192, "int8", 700, "auto", "intel", title=FN, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertFalse(f.changed)
        self.assertEqual((f.ctx, f.slots), (8192, 2))

    def test_not_even_the_smallest_context_fits_then_one_request_at_a_time(self):
        f = V.plan(FN, 8.0, 8192, "int8", 1500, "auto", "intel", title=FN, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertEqual(f.slots, 1)
        self.assertTrue(f.changed)
        self.assertIn("one request at a time", f.line)

    def test_a_12gb_card_keeps_128k_with_streamed_kv(self):
        # setup writes --kv-resident 32768 at 128K: each of the 3 sessions keeps about 620 MiB, not 1.9 GB
        for vendor in ("intel", "other"):
            f = V.plan(FN, 12.0, 131072, "int8", 700, "auto", vendor, title=FN, reserve_fixed=True, prefill_fixed=True, slots=2, kv_resident=32768)
            self.assertFalse(f.changed, vendor)
            self.assertEqual((f.ctx, f.slots), (131072, 2))
        self.assertGreater(V.slack_mib(FN, 12.0, 131072, "int8", 700, "auto", "intel", 2, 32768), 2000)
        # charging each slot the whole 128K would have lowered it (the mistake of 2a1042877)
        self.assertLess(V.slack_mib(FN, 12.0, 131072, "int8", 700, "auto", "intel", 2, 0), 0)

    def test_a_card_with_no_measured_figure_is_left_to_the_engine(self):
        f = V.plan(FN, 8.0, 8192, "int8", 1024, "auto", "other", title=FN, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertFalse(f.changed)                # CUDA / HIP: no measured base yet, so no claim; the engine sizes its cache and says what is short
        self.assertEqual(f.ctx, 8192)

    def test_a_24gb_cuda_card_is_unchanged(self):
        f = V.plan(FN, 24.0, 131072, "int8", 700, "auto", "other", title=FN, reserve_fixed=True, prefill_fixed=True, slots=4, kv_resident=32768)
        self.assertFalse(f.changed)
        self.assertEqual((f.ctx, f.slots, f.line), (131072, 4, ""))

    def test_a_chosen_context_is_kept(self):
        f = V.plan(FN, 8.0, 8192, "int8", 1024, "auto", "intel", ctx_fixed=True, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertEqual(f.ctx, 8192)
        self.assertFalse(f.changed)            # the engine's plain refusal answers it


class FitStart(unittest.TestCase):
    """fit_start: the config's "parallel" and the engine arguments of one start (serve/server.py engine_args calls it)."""

    ARGS = ["--native", "/m/Qwen3.8-Flash-Next-IQ3_XXS-00001-of-00002.gguf", "--expert-cache", "auto", "--prefill", "auto", "--max-context", "8192",
            "--kv", "int8", "--vram-reserve-mib", "1024"]

    def cfg(self, **kw):
        c = {"model_name": FN, "parallel": 2}
        c.update(kw)
        return c

    def test_an_8gb_arc_gets_a_smaller_context_for_this_start(self):
        args, par, note = V.fit_start(self.cfg(), list(self.ARGS), 8192, "sycl")
        self.assertEqual(args[args.index("--max-context") + 1], "2048")
        self.assertIsNone(par)                                 # still 2 requests at once
        self.assertIn("2 requests at once fit", note)
        self.assertEqual(self.ARGS[self.ARGS.index("--max-context") + 1], "8192")     # the input is not touched

    def test_the_persons_own_context_wins(self):
        args, par, note = V.fit_start(self.cfg(user_args=["--max-context", "8192"]), list(self.ARGS), 8192, "sycl")
        self.assertEqual((args, par, note), (self.ARGS, None, ""))

    def test_a_12gb_card_at_128k_with_streamed_kv_is_unchanged(self):
        a = ["--native", "/m/Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768",
             "--vram-reserve-mib", "700"]
        for backend in ("cuda", "sycl"):
            self.assertEqual(V.fit_start(self.cfg(model_name="qwen3.8-flash-next-iq3_s"), list(a), 12288, backend), (a, None, ""), backend)

    def test_a_24gb_cuda_card_and_one_request_are_unchanged(self):
        big = ["--native", "/m/q.gguf", "--max-context", "131072", "--kv", "int8", "--vram-reserve-mib", "700"]
        self.assertEqual(V.fit_start(self.cfg(parallel=4), list(big), 24576, "cuda"), (big, None, ""))
        self.assertEqual(V.fit_start(self.cfg(parallel=1), list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start({"model_name": FN}, list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))

    def test_nothing_without_one_cards_memory_or_on_a_split_or_deepseek_or_metal(self):
        self.assertEqual(V.fit_start(self.cfg(), list(self.ARGS), None, "sycl"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start(self.cfg(), list(self.ARGS), 8192, "metal"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start(self.cfg(layer_split="auto"), list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start(self.cfg(arch="deepseek41"), list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))

    def test_a_start_that_cannot_fit_runs_one_at_a_time(self):
        tight = [a if a != "1024" else "1500" for a in self.ARGS]
        args, par, note = V.fit_start(self.cfg(), tight, 8192, "sycl")
        self.assertEqual(par, 1)
        self.assertIn("one request at a time", note)

    def test_the_context_is_added_when_the_args_name_none(self):
        no_ctx = list(self.ARGS)
        i = no_ctx.index("--max-context")
        del no_ctx[i:i + 2]
        args, par, note = V.fit_start(self.cfg(), no_ctx, 8192, "sycl")        # 131072 by default: lowered to what fits
        self.assertNotEqual(note, "")
        self.assertIn("--max-context", args)


if __name__ == "__main__":
    unittest.main()
