"""Inkling-2-Mini fit planner: GGUF experts a backend converts at load are sized by their CONVERTED size, not their file size (#6729).

HIP, SYCL and Metal run no native kernel for most GGUF expert types: ink_gguf_transcode.h turns them into S4_K / S6_K / S8_K while loading
(an IQ1_S expert grows 1.56 -> 4.19 bits per weight, 2.7x). A plan that counts the file size leaves about 40% of the experts with no place.
CUDA keeps the experts as they are in the file. The expert element counts per build come from the GGUF headers (gguf_experts_v1.py)."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
REPO = TERMINAL.parent
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import inkling_fit as F

GGUF_BUILDS = [b for b, i in F.BUILDS.items() if i.get("format") == "gguf"]


class ConvertedSize(unittest.TestCase):
    def test_every_gguf_build_has_its_expert_elements(self):
        for b in GGUF_BUILDS:
            e = F.BUILDS[b].get("expert_elems_e9")
            self.assertTrue(e, b)
            self.assertAlmostEqual(sum(e.values()), 24.461, places=2, msg=b)       # 3072 experts x 3 x hidden x 1152
            for t in e:
                self.assertIn(t, F.GGUF_TYPE_ELEMS_BYTES, f"{b}: unknown type {t}")

    def test_cuda_and_no_backend_keep_the_file_size(self):
        for b in GGUF_BUILDS:
            for be in (None, "", "cuda", "cpu"):
                self.assertEqual(F.bodies_gb_for(b, be), F.BUILDS[b]["bodies_gb"], (b, be))

    def test_strata_builds_are_never_converted(self):
        for b in ("SQ-10", "SQ-15", "SQ-20"):
            for be in ("hip", "sycl", "metal"):
                self.assertEqual(F.bodies_gb_for(b, be), F.BUILDS[b]["bodies_gb"])

    def test_one_bit_builds_grow_on_transcoding_backends(self):
        # the IQ1 / IQ2 / Q2 builds: every expert tensor but the NVFP4 ones becomes S4_K
        for be in ("hip", "sycl", "metal"):
            s = F.bodies_gb_for("UD-IQ1_S", be)
            self.assertGreater(s / F.BUILDS["UD-IQ1_S"]["bodies_gb"], 1.8)
            self.assertLess(s / F.BUILDS["UD-IQ1_S"]["bodies_gb"], 2.1)
            self.assertGreater(F.bodies_gb_for("UD-IQ2_XXS", be) / F.BUILDS["UD-IQ2_XXS"]["bodies_gb"], 1.6)
            self.assertGreater(F.bodies_gb_for("UD-Q2_K_XL", be) / F.BUILDS["UD-Q2_K_XL"]["bodies_gb"], 1.5)

    def test_nvfp4_builds_stay_about_their_size(self):
        # UD-Q4_K_XL: NVFP4 experts stay SF4 (bit-exact), the Q8_0 ones become S8_K (a little smaller)
        for b in ("UD-IQ4_XS", "UD-Q4_K_M", "UD-Q4_K_XL"):
            r = F.bodies_gb_for(b, "hip") / F.BUILDS[b]["bodies_gb"]
            self.assertTrue(0.97 <= r <= 1.03, (b, r))

    def test_the_size_grows_as_the_bits_per_weight_fall(self):
        order = ["UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS"]
        sizes = [F.bodies_gb_for(b, "hip") for b in order]
        self.assertEqual(sizes, sorted(sizes))        # converted sizes rise with the build, never above the S4_K ceiling
        self.assertLess(max(sizes), 14.5)

    def test_plan_on_hip_places_the_converted_size(self):
        """8 GiB card, IQ1_S: HIP must plan for ~11.7 GB of experts, CUDA for the 6 GB of the file."""
        hip = F.place("UD-IQ1_S", 8.0, 64.0, False, 8192, False, dspark=False, backend="hip")
        cud = F.place("UD-IQ1_S", 8.0, 64.0, False, 8192, False, dspark=False, backend="cuda")
        want_hip = F.bodies_gb_for("UD-IQ1_S", "hip")
        self.assertAlmostEqual(hip["vram_gb"] + hip["ram_gb"] + hip["ssd_gb"], want_hip, delta=0.05)
        self.assertAlmostEqual(cud["vram_gb"] + cud["ram_gb"] + cud["ssd_gb"], F.BUILDS["UD-IQ1_S"]["bodies_gb"], delta=0.05)
        self.assertGreater(hip["ram_gb"] + hip["ssd_gb"], cud["ram_gb"] + cud["ssd_gb"])

    def test_a_converted_build_that_no_longer_fits_the_card_goes_hybrid(self):
        # 12 GiB card: UD-IQ1_S fits whole in CUDA (6 GB of experts) but not on HIP (11.7 GB + dense + K/V)
        cud = F.place("UD-IQ1_S", 12.0, 64.0, False, 4096, False, dspark=False, backend="cuda")
        hip = F.place("UD-IQ1_S", 12.0, 64.0, False, 4096, False, dspark=False, backend="hip")
        self.assertTrue(cud["full_vram"])
        self.assertFalse(hip["full_vram"])
        self.assertGreater(hip["ram_gb"], 0.0)

    def test_unified_memory_counts_the_converted_size_too(self):
        mac = F.place("UD-IQ1_S", None, 16.0, True, 8192, False, dspark=False, backend="metal")
        plain = F.place("UD-IQ1_S", None, 16.0, True, 8192, False, dspark=False, backend=None)
        self.assertGreater(mac["vram_gb"] + mac["ssd_gb"], plain["vram_gb"] + plain["ssd_gb"] + 4.0)


class RuleMatchesTheEngine(unittest.TestCase):
    """The planner's copy of the conversion rule must follow the engine's: read both C++ files and compare (skipped outside a checkout)."""

    def setUp(self):
        self.tc = REPO / "engine" / "src" / "inkling2" / "ink_gguf_transcode.h"
        self.sk = REPO / "engine" / "include" / "strata" / "quant" / "sk_codecs.hpp"
        if not (self.tc.exists() and self.sk.exists()):
            self.skipTest("engine sources are not next to the terminal")

    def test_superblock_bytes(self):
        text = self.sk.read_text(encoding="utf-8")
        for code in ("S4_K", "S6_K", "S8_K"):
            m = re.search(r"case SkCode::%s:\s*return (\d+);" % code, text)
            self.assertIsNotNone(m, code)
            self.assertEqual(int(m.group(1)), F.SK_BYTES_PER_256[code], code)

    def test_type_to_code(self):
        text = self.tc.read_text(encoding="utf-8")
        body = text[text.index("choose_sk_code_for_ggml"):]
        body = body[:body.index("\n}\n")]
        ids = {"F32": 0, "F16": 1, "Q8_0": 8, "Q8_1": 9, "Q8_K": 15, "BF16": 30, "Q5_0": 6, "Q5_1": 7, "Q5_K": 13, "Q6_K": 14}

        def code_of(tid):
            m = re.search(r"((?:case \d+:[^\n]*\n\s*)+)return SkCode::(S\d_K);", body)
            for m in re.finditer(r"((?:case \d+:[^\n]*\n\s*)+)return SkCode::(S\d_K);", body):
                if tid in [int(x) for x in re.findall(r"case (\d+):", m.group(1))]:
                    return m.group(2)
            return "S4_K"                              # the default branch

        for code, types in F.SK_CODE_OF_TYPE.items():
            for t in types:
                self.assertEqual(code_of(ids[t]), code, t)
        for tid in (2, 3, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 42):    # the <= 4 bit types of the builds
            self.assertEqual(code_of(tid), "S4_K", tid)


if __name__ == "__main__":
    unittest.main()
