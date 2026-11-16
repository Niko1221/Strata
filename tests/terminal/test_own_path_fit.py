"""Qwen3.6-35B-A3B's context against the card's memory (GitHub #5715, strata_control/own_path_fit.py): the per-token figure the engine really
allocates, the formula's inputs where setup, the start and the app read them, and what it says about the field logs it was derived from."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import setup
from strata_control import own_path_fit as F

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "src" / "qwen35" / "q36_engine.cu"
Q4 = F.inputs("qwen3.6-35b-a3b", "UD-Q4_K_XL")
Q6 = F.inputs("qwen3.6-35b-a3b", "UD-Q6_K_XL")


class PerToken(unittest.TestCase):
    def test_one_session_takes_its_fp32_cache_and_the_shared_fp16_copy(self):
        fam = setup.OWN_PATH_FAMILIES["qwen36"]
        self.assertEqual(fam["kv_bytes_per_token"], 45056)                 # a session's (a parallel slot's) K / V: unchanged
        self.assertEqual(F.BYTES_PER_TOKEN, fam["kv_bytes_per_token"] + 2048)
        self.assertEqual(F.BYTES_PER_TOKEN, 47104)
        self.assertAlmostEqual(F.cache_gib(32768), 1.4375)

    def test_the_figure_follows_the_engines_allocations(self):
        src = ENGINE.read_text(encoding="utf-8")
        # Both the attention layers and MTP use the typed layout allocator.
        self.assertEqual(len(re.findall(r"ss\.kc\.push_back\(alloc_kv\(true\)\)", src)), 2)
        self.assertEqual(len(re.findall(r"ss\.vc\.push_back\(alloc_kv\(false\)\)", src)), 2)
        self.assertIn("bool kv_k_f16 = false, kv_v_f16 = false", src)
        for side, flag in (("key", "k"), ("value", "v")):
            self.assertIn(f"{side}_layout = strata::qwen35::make_kv_layout(kv_{flag}_f16 ? strata::qwen35::Q36KvType::Fp16 : "
                          "strata::qwen35::Q36KvType::Fp32, ctx, g.n_head_kv, g.head_dim)", src)
        self.assertIn("const size_t bytes = (key ? key_layout : value_layout).allocation_bytes", src)
        self.assertIn("CK(cudaMalloc(&p, bytes))", src)
        layout = (ROOT / "include" / "strata" / "qwen35" / "kv_layout.hpp").read_text(encoding="utf-8")
        self.assertIn("l.row_stride = kv_checked_mul(dim, type == Q36KvType::Fp32 ? 4 : 2)", layout)
        self.assertIn("l.head_stride = kv_checked_mul(context, l.row_stride)", layout)
        self.assertIn("l.payload_bytes = l.allocation_bytes = kv_checked_mul(heads, l.head_stride)", layout)
        # The default FP32 cache still needs one shared FP16 copy of K and V.
        self.assertEqual(len(re.findall(r"[kv]16 = kv_[kv]_f16 \? nullptr : dalloc_t<__half>\(\(size_t\) g\.n_head_kv \* ctx \* g\.head_dim\)", src)), 2)

    def test_inputs_for_the_sizes_and_nothing_for_other_models(self):
        self.assertEqual(Q4, {"bytes_per_token": 47104, "fixed_gb": round(8.55 + 1256 / 1024 + 0.07e9 / 2 ** 30, 4)})
        self.assertEqual(F.inputs("qwen3.6-35b-a3b", "Q8_0"), Q6)
        self.assertIsNone(F.inputs("qwen3.6-35b-a3b", "UD-Q2_K"))
        self.assertIsNone(F.inputs("some-feed-model", "UD-Q4_K_XL"))
        self.assertEqual(F.from_model_name("qwen3.6-35b-a3b-ud-q4_k_xl"), Q4)
        self.assertEqual(F.from_model_name("qwen3.6-35b-a3b-q8_0"), Q6)
        self.assertIsNone(F.from_model_name("qwen3.8-flash-next-iq3_s"))
        self.assertIsNone(F.from_model_name(None))
        for name in ("qwen36",):
            self.assertEqual(setup.OWN_PATH_FAMILIES[name]["name"], F.FAMILY_NAME)
            self.assertEqual(sorted(setup.OWN_PATH_FAMILIES[name]["models"]), sorted(F.HYBRID_MIN_GIB))


class FieldLogs(unittest.TestCase):
    """The starts the figures were derived from and checked against (card memory as nvidia-smi gives it, GiB)."""

    def test_rtx_5070_12gb_5715(self):
        self.assertTrue(F.fits(Q4, 11.94, 32768))          # ran (VRAM 10.5 / 12 GB)
        self.assertFalse(F.fits(Q4, 11.94, 65536))         # "not enough VRAM left for the hybrid experts' buffers"
        self.assertFalse(F.fits(Q4, 11.94, 131072))        # the bare out-of-memory line

    def test_rtx_5060ti_16gb_3303(self):
        self.assertTrue(F.fits(Q4, 15.93, 100000))         # started, 3.93 GiB free
        self.assertFalse(F.fits(Q4, 15.93, 262144))        # out of memory

    def test_rtx_3060_12gb_q6_3687(self):
        self.assertTrue(F.fits(Q6, 12.0, 32768))           # started
        self.assertFalse(F.fits(Q6, 12.0, 131072))         # 1760 MiB needed, 0 free

    def test_rtx_3080_10gb_q8_2747(self):
        self.assertFalse(F.fits(F.inputs("qwen3.6-35b-a3b", "Q8_0"), 10.0, 32768))   # the staging pair failed


if __name__ == "__main__":
    unittest.main()
