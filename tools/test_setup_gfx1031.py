"""Tests for setup's RX 6700 XT (gfx1031) handling: the engine carries the family's gfx1030 code, the run config runs
the card as gfx1030 so the prompt's matrix products find rocBLAS's kernels, the tuning tables are looked up under the
name the engine reports, and a setup run again refreshes setup's own override while keeping the user's speed switch.
No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_gfx1031
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class BuildArchs(unittest.TestCase):
    def test_the_6700_xt_builds_code_for_its_family_too(self):
        # rocBLAS has no gfx1031 kernels, so the engine needs the gfx1030 code objects to run the card as gfx1030
        with unittest.mock.patch.object(setup, "say", lambda *a, **k: None):
            self.assertEqual(setup.hip_build_archs({"arch": "gfx1031"}), ["gfx1030", "gfx1031"])
            self.assertEqual(setup.hip_build_archs({"archs": ["gfx1031"]}), ["gfx1030", "gfx1031"])

    def test_a_card_with_kernels_of_its_own_builds_only_its_own(self):
        # the default path stays as it was for every arch rocBLAS covers (the byte-identical rule)
        said = []
        with unittest.mock.patch.object(setup, "say", said.append):
            for arch in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1151"):
                self.assertEqual(setup.hip_build_archs({"arch": arch}), [arch])
        self.assertEqual(said, [])

    def test_a_layer_split_keeps_every_card(self):
        with unittest.mock.patch.object(setup, "say", lambda *a, **k: None):
            self.assertEqual(setup.hip_build_archs({"archs": ["gfx1031", "gfx1201"]}),
                             ["gfx1030", "gfx1031", "gfx1201"])

    def test_the_sibling_is_named_once(self):
        # two 6700 XTs: one gfx1030 added, and the engine's "already built for this PC" test sees both names
        with unittest.mock.patch.object(setup, "say", lambda *a, **k: None):
            archs = setup.hip_build_archs({"archs": ["gfx1031", "gfx1031"]})
        self.assertEqual(archs.count("gfx1030"), 1)
        self.assertTrue(set(archs) <= {"gfx1030", "gfx1031"})           # the stamp's subset test


class RunConfig(unittest.TestCase):
    def test_the_card_runs_as_the_arch_rocblas_has_kernels_for(self):
        self.assertEqual(setup.HIP_ROCBLAS_SIBLINGS["gfx1031"], "gfx1030")
        self.assertEqual(setup.HIP_ROCBLAS_AS["gfx1031"], "10.3.0")     # docs/AMD_HIP.md sets 9.0.6 for gfx906
        self.assertNotIn("gfx1030", setup.HIP_ROCBLAS_SIBLINGS)         # never chained: gfx1030 is the end of the line

    def test_the_tuning_tables_are_the_ones_the_engine_will_read(self):
        # the engine picks the table of the arch it sees, so setup looks them up under that name
        self.assertEqual(setup.hip_report_arch("gfx1031"), "gfx1030")
        for arch in ("gfx1030", "gfx1100", "gfx1201", "gfx1151"):
            self.assertEqual(setup.hip_report_arch(arch), arch)

    def test_the_override_is_setup_s_own_key(self):
        # a setup run again refreshes it; it does not linger as a user key when the card changes
        self.assertIn("HSA_OVERRIDE_GFX_VERSION", setup.SETUP_ENV)

    def test_the_users_speed_switch_survives_setup_running_again(self):
        old = {"env": {"HSA_OVERRIDE_GFX_VERSION": "9.0.6",           # setup's, from an old card
                       "STRATA_HIP_PROMPT_F16": "1",                  # the user's own (opt-in, changes the numbers)
                       "STRATA_RESIDENT_PIN": "0"}}                   # setup's
        new = {"model_name": "swift-iq3_s", "env": {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}}
        kept = setup.carry_over(old, new)
        self.assertIn("env STRATA_HIP_PROMPT_F16", kept)
        self.assertNotIn("env HSA_OVERRIDE_GFX_VERSION", kept)
        self.assertNotIn("env STRATA_RESIDENT_PIN", kept)
        self.assertEqual(new["env"], {"HSA_OVERRIDE_GFX_VERSION": "10.3.0", "STRATA_HIP_PROMPT_F16": "1"})


if __name__ == "__main__":
    unittest.main()
