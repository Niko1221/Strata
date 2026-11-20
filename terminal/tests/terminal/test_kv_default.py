"""The KV format a run config names: the engine's own choice on gfx12 with a q4u32 build, the old format elsewhere."""

import unittest

from strata_control.kv_default import (KV_ENGINE_FLAVOR, engine_picks_kv, flavor_of, is_gfx12, kv_flag, kv_value,
                                       version_tuple)

#: this tree's engine: the Strata release it was changed from, plus the suffix of what it adds
NEW = "0.1.63-q4u32"
#: an upstream engine: the same release without the suffix, and a newer one (upstream is at 0.1.63)
OLD = "0.1.63"
UPSTREAM = "0.1.63"


class VersionTuple(unittest.TestCase):
    def test_a_string_a_tuple_and_nothing(self):
        self.assertEqual(version_tuple("0.1.59.1"), (0, 1, 59, 1))
        self.assertEqual(version_tuple((0, 1, 59, 1)), (0, 1, 59, 1))
        self.assertEqual(version_tuple(None), ())
        self.assertEqual(version_tuple("0.1.59.1-dev"), (0, 1, 59, 1))   # the numbers up to the first word

    def test_the_variant_suffix_does_not_change_the_numbers(self):
        self.assertEqual(version_tuple(NEW), (0, 1, 63))
        self.assertEqual(flavor_of(NEW), "q4u32")
        self.assertEqual(flavor_of(OLD), "")
        self.assertEqual(flavor_of(UPSTREAM), "")


class Gfx12(unittest.TestCase):
    def test_the_cards_whose_prompt_path_was_measured(self):
        self.assertTrue(is_gfx12("gfx1201"))
        self.assertTrue(is_gfx12("gfx1200"))
        self.assertTrue(is_gfx12("gfx1201:sramecc-:xnack-"))
        self.assertFalse(is_gfx12("gfx1100"))
        self.assertFalse(is_gfx12("gfx1151"))
        self.assertFalse(is_gfx12(None))


class EnginePicksKv(unittest.TestCase):
    def test_only_a_q4u32_build_on_gfx12(self):
        self.assertTrue(engine_picks_kv(KV_ENGINE_FLAVOR, hip=True, arch="gfx1201"))
        self.assertTrue(engine_picks_kv(KV_ENGINE_FLAVOR, hip=True, arch="gfx1200"))
        self.assertFalse(engine_picks_kv(KV_ENGINE_FLAVOR, hip=True, arch="gfx1100"))
        self.assertFalse(engine_picks_kv(KV_ENGINE_FLAVOR, hip=False, arch="gfx1201"))   # not an AMD card
        self.assertFalse(engine_picks_kv(KV_ENGINE_FLAVOR, hip=False, arch="sm_120"))     # not an AMD card

    def test_an_upstream_engine_never_picks(self):
        # even a newer upstream release: the suffix is the marker, not the number
        self.assertFalse(engine_picks_kv("", hip=True, arch="gfx1201"))
        self.assertFalse(engine_picks_kv("upstream", hip=True, arch="gfx1201"))


class KvFlag(unittest.TestCase):
    """What a run config writes: setup's own rule, plus the q4u32 build's choice."""

    def test_a_named_format_is_kept_on_every_engine(self):
        for flavor in (OLD, NEW, ""):
            self.assertEqual(kv_flag(KV_ENGINE_FLAVOR if flavor == NEW else "", 131072, "fp16", hip=True, arch="gfx1201"), "fp16")
            self.assertEqual(kv_flag(KV_ENGINE_FLAVOR if flavor == NEW else "", 4096, "q4u32", hip=True, arch="gfx1201"), "q4u32")

    def test_an_upstream_engine_names_8_bit_above_8k_and_16_bit_below(self):
        self.assertEqual(kv_flag("", 32768, None, hip=True, arch="gfx1201"), "int8")
        self.assertEqual(kv_flag("", 131072, None, hip=True, arch="gfx1201"), "int8")
        self.assertEqual(kv_flag("", 8192, None, hip=True, arch="gfx1201"), "fp16")     # up to 8K: what the release named

    def test_a_q4u32_build_on_another_card_names_the_format_as_before(self):
        self.assertEqual(kv_flag(KV_ENGINE_FLAVOR, 32768, None, hip=True, arch="gfx1100"), "int8")
        self.assertEqual(kv_flag(KV_ENGINE_FLAVOR, 32768, None, hip=False, arch="sm_120"), "int8")

    def test_only_gfx12_with_a_q4u32_build_is_left_to_the_engine(self):
        self.assertIsNone(kv_flag(KV_ENGINE_FLAVOR, 131072, None, hip=True, arch="gfx1201"))
        self.assertIsNone(kv_flag(KV_ENGINE_FLAVOR, 16384, None, hip=True, arch="gfx1200"))
        self.assertEqual(kv_flag(KV_ENGINE_FLAVOR, 8192, None, hip=True, arch="gfx1200"), "fp16")   # up to 8K: as before


class KvValue(unittest.TestCase):
    """The format the engine will run with: what the estimates use and what a config that always names one writes."""

    def test_what_the_engine_will_run_with(self):
        self.assertEqual(kv_value(KV_ENGINE_FLAVOR, 131072, None, hip=True, arch="gfx1201"), "q4u32")
        self.assertEqual(kv_value(KV_ENGINE_FLAVOR, 131072, None, hip=True, arch="gfx1100"), "int8")
        self.assertEqual(kv_value("", 131072, None, hip=True, arch="gfx1201"), "int8")
        self.assertEqual(kv_value(KV_ENGINE_FLAVOR, 131072, "fp16", hip=True, arch="gfx1201"), "fp16")
        self.assertEqual(kv_value("", 8192, None, hip=True, arch="gfx1201"), "fp16")
        self.assertEqual(kv_value(KV_ENGINE_FLAVOR, 8192, None, hip=True, arch="gfx1201"), "fp16")    # up to 8K: as before
        self.assertEqual(kv_value(KV_ENGINE_FLAVOR, 16384, None, hip=True, arch="gfx1201"), "q4u32")


if __name__ == "__main__":
    unittest.main()
