"""Advanced DeepSeek settings reach the engine argv without starting it."""
import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control.tune_knobs import BOOL_FLAGS


class DeepSeekOverrides(unittest.TestCase):
    def config(self, pairs, family="deepseek41"):
        base = {"exe": "strata-dsv41", "args": ["--pack", "model-pack", "--dspark", "auto", "--experts", "6"]}
        settings = S.LaunchSettings(family=family, extra=S.settings_from_pairs(pairs))
        return derive_config(base, settings, bool_flags=BOOL_FLAGS)["args"]

    def test_dspark_modes_keep_their_values(self):
        for mode in ("on", "off", "none", "auto"):
            with self.subTest(mode=mode):
                self.assertEqual(self.config([f"dspark={mode}"]),
                                 ["--pack", "model-pack", "--experts", "6", "--dspark", mode])

    def test_reserve_aliases_reach_deepseek(self):
        for key in ("vram_reserve_mib", "vram-reserve-mib", "reserve-mib"):
            with self.subTest(key=key):
                args = self.config([f"{key}=1100"])
                self.assertEqual(args[-2:], ["--reserve-mib", "1100"])
                self.assertNotIn("--vram-reserve-mib", args)

    def test_combined_overrides_replace_existing_reserve(self):
        base = {"exe": "strata-dsv41", "args": ["--dspark", "auto", "--reserve-mib", "600"]}
        settings = S.LaunchSettings(family="deepseek41", extra=S.settings_from_pairs(
            ["dspark=on", "vram_reserve_mib=1100"]))
        self.assertEqual(derive_config(base, settings, bool_flags=BOOL_FLAGS)["args"],
                         ["--dspark", "on", "--reserve-mib", "1100"])
        self.assertEqual(base["args"], ["--dspark", "auto", "--reserve-mib", "600"])

    def test_qwen_keeps_its_reserve_flag(self):
        self.assertEqual(self.config(["vram_reserve_mib=1100"], "qwen")[-2:],
                         ["--vram-reserve-mib", "1100"])

    def test_default_dspark_is_unchanged(self):
        self.assertEqual(self.config([]),
                         ["--pack", "model-pack", "--dspark", "auto", "--experts", "6"])

    def test_boolean_and_environment_settings_keep_their_behavior(self):
        args, env = S.as_argv_changes(S.settings_from_pairs(
            ["kv-grow=on", "no-pool=off", "STRATA_TEST=off"]), BOOL_FLAGS)
        self.assertEqual(args, {"--kv-grow": True, "--no-pool": None})
        self.assertEqual(env, {"STRATA_TEST": None})
