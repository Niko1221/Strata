"""Differential and native-helper contract tests for `strata-fit` (#5972)."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from strata_control import gpu_fit, native_helpers, vram_fit


_RELEASE_DIR = Path(__file__).resolve().parents[2] / "crates" / "strata-fit" / "target" / "release"
_EXE = _RELEASE_DIR / ("strata-fit.exe" if os.name == "nt" else "strata-fit")


class StrataFitDifferentialTest(unittest.TestCase):
    def setUp(self):
        if not _EXE.is_file():
            self.skipTest(f"release binary not built at {_EXE}")

    def test_off_by_default_matches_python(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            self.assertIsNone(native_helpers.program("fit"))

    def test_bad_cli_exits_2_with_empty_stdout(self):
        done = subprocess.run([str(_EXE), "gpu-fit", "{}"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")

    def test_gpu_fit_differential(self):
        cases = [
            (83.6, 128.0, 24.0),
            (12.0, 64.0, 24.0),
            (50.0, None, 24.0),
            (10.0, 32.0, 0.0),
        ]
        for memory_gb, ram_gb, vram_gb in cases:
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
                py = gpu_fit.fit(memory_gb, ram_gb, vram_gb)
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "fit", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
                rs = gpu_fit.fit(memory_gb, ram_gb, vram_gb)
            self.assertEqual(rs, py, (memory_gb, ram_gb, vram_gb))

    def test_gpu_options_differential(self):
        cards = [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vendor": "nvidia", "vram_gb": 24, "usable": True},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vendor": "nvidia", "vram_gb": 24, "usable": True},
        ]
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
            py = gpu_fit.options(83.6, 128.0, cards, None)
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "fit", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
            rs = gpu_fit.options(83.6, 128.0, cards, None)
        self.assertEqual(rs, py)

    def test_vram_plan_differential(self):
        cases = [
            dict(model="Qwen3.8-Flash-Next", vram_gb=24.0, ctx=131072, kv="int8", reserve_mib=700,
                 prefill="auto", vendor="cuda", title="Flash-Next", reserve_fixed=True, prefill_fixed=True,
                 slots=0, kv_resident=32768, drafter="dflash2", kv_fixed=False),
            dict(model="Qwen3.8-Flash-Next", vram_gb=8.0, ctx=131072, kv="int8", reserve_mib=700,
                 prefill="auto", vendor="cuda", title="Flash-Next", reserve_fixed=True, prefill_fixed=True,
                 slots=2, kv_resident=0, drafter="mtp", kv_fixed=False),
        ]
        for kw in cases:
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
                py = vram_fit.plan(**kw)
            with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "fit", native_helpers.DIR_ENV: str(_RELEASE_DIR)}):
                rs = vram_fit.plan(**kw)
            self.assertEqual(rs, py, kw)
