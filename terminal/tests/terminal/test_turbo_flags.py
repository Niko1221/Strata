"""The Inkling-2-Mini opt-in tail-skip presets (owner 2026-11-26): `strata launch --turbo` / `--turbo-max` write STRATA_INK_TAIL_SKIP=turbo / turbo-max.
Terminal only (the app's mirror of the speed switches is unchanged); both are OFF by default and never the default; the engine expands the names
(engine/src/inkling2/cuda/ink_tail_skip.cuh, tested by engine/tests/core/ink_tail_skip_test.cpp)."""
import unittest

from strata_control import speed_switches as ss

CUDA = [{"name": "NVIDIA GeForce RTX 5070", "vendor": "nvidia", "arch": "120", "vram_gb": 12.0}]
INK = {"arch": "inkling2_loop"}


class TurboSwitch(unittest.TestCase):
    def test_it_is_a_terminal_only_row_that_is_off_by_default(self):
        sw = ss.BY_ID["inkling_turbo"]
        self.assertTrue(sw.terminal_only)
        self.assertEqual(sw.default, "off")
        self.assertNotIn("inkling_turbo", [s.id for s in ss.app_switches()])           # the app does not show it, so native-settings.mjs needs no change
        self.assertEqual(dict(sw.env), {"STRATA_INK_TAIL_SKIP": "turbo"})
        self.assertEqual(sw.choices, (("turbo", "Turbo"), ("turbo-max", "Turbo max")))
        self.assertFalse(ss.is_on({}, sw))
        self.assertTrue(ss.is_on({"STRATA_INK_TAIL_SKIP": "turbo"}, sw))
        self.assertFalse(ss.is_on({"STRATA_INK_TAIL_SKIP": "off"}, sw))
        self.assertEqual(ss.set_switch({"A": "1"}, sw, True), {"A": "1", "STRATA_INK_TAIL_SKIP": "turbo"})
        self.assertEqual(ss.set_switch({"STRATA_INK_TAIL_SKIP": "turbo", "A": "1"}, sw, False), {"A": "1"})

    def test_it_applies_to_inkling_on_nvidia_and_amd_only(self):
        shows = lambda cfg, backend: "inkling_turbo" in [s.id for s in ss.applicable(cfg, CUDA, backend)]   # noqa: E731
        self.assertTrue(shows(INK, "cuda"))
        self.assertTrue(shows(INK, "hip"))
        for backend in ("metal", "sycl", "cpu", None):
            self.assertFalse(shows(INK, backend), backend)
        self.assertFalse(shows({}, "cuda"))                                              # a Qwen config
        self.assertFalse(shows({"arch": "qwen35moe"}, "cuda"))

    def test_the_default_start_sets_nothing_for_it(self):
        self.assertNotIn("STRATA_INK_TAIL_SKIP", ss.ENGINE_START_DEFAULTS)               # off is the engine's own default: no entry at all
        env = ss.engine_env_defaults({})
        self.assertNotIn("STRATA_INK_TAIL_SKIP", env)

    def test_the_terminal_sentence(self):
        words = ss.env_words("STRATA_INK_TAIL_SKIP")
        self.assertIn("off by default", words)
        self.assertIn("turbo-max", words)
        self.assertIn("differ slightly from exact", words)


if __name__ == "__main__":
    unittest.main()
