"""Inkling-2-Mini: --kv int8 (the Qwen name, and what a profile carried over from Qwen holds) is k8v8 on CUDA and fp8 where
k8v8 is not offered, instead of being dropped (the engine then ran fp16)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_cli.launch import runner as R  # noqa: E402
from strata_cli.launch import settings as S  # noqa: E402


class TestInklingKvName(unittest.TestCase):
    def test_int8_is_k8v8_on_cuda(self):
        self.assertEqual(S.safe_inkling_kv("int8", "cuda"), ("k8v8", True))
        self.assertEqual(S.safe_inkling_kv("int8", None), ("k8v8", True))

    def test_int8_is_fp8_where_k8v8_is_not_offered(self):
        for be in ("hip", "metal", "sycl"):
            self.assertEqual(S.safe_inkling_kv("int8", be), ("fp8", True))

    def test_the_engine_names_are_unchanged(self):
        self.assertEqual(S.safe_inkling_kv("fp16", "cuda"), ("fp16", False))
        self.assertEqual(S.safe_inkling_kv("k8v4", "hip"), ("fp8", True))
        self.assertEqual(S.safe_inkling_kv("q4_0", "cuda"), (None, False))


class TestInstalledKvIsCompared(unittest.TestCase):
    def test_int8_matches_an_installed_k8v8(self):
        s = S.LaunchSettings(family="inkling-2-mini", kv="int8")
        self.assertTrue(R.desc_matches({"kv": "k8v8", "images": "off"}, s))

    def test_an_installed_fp16_is_written_again(self):
        s = S.LaunchSettings(family="inkling-2-mini", kv="int8")
        self.assertFalse(R.desc_matches({"kv": None, "images": "off"}, s))        # no --kv: the engine's fp16
        self.assertFalse(R.desc_matches({"kv": "fp16", "images": "on"}, s))        # also without a picture choice

    def test_int8_is_fp8_on_a_hip_install(self):
        s = S.LaunchSettings(family="inkling-2-mini", kv="int8")
        self.assertFalse(R.desc_matches({"kv": "k8v8", "backend": "hip", "images": "off"}, s))
        self.assertTrue(R.desc_matches({"kv": "fp8", "backend": "hip", "images": "off"}, s))

    def test_no_kv_choice_matches_any(self):
        s = S.LaunchSettings(family="inkling-2-mini")
        self.assertTrue(R.desc_matches({"kv": "fp16", "images": "off"}, s))


if __name__ == "__main__":
    unittest.main()
