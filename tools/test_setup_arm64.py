"""Shared ARM64 policy and Windows RTX Spark setup choices; no GPU, network, or downloads."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class Arm64Policy(unittest.TestCase):
    def test_spark_shared_memory_formula(self):
        gpu = {"name": "NVIDIA RTX Spark N1X", "vram_gb": 8.0}
        with mock.patch.object(setup, "WIN_ARM64", True):
            got = setup.nvidia_apply_uma(gpu, 120.0)
        self.assertTrue(got["uma"] and got["spark_uma"])
        self.assertEqual(got["dedicated_gb"], 8.0)
        self.assertEqual(got["shared_gb"], 96.0)
        self.assertEqual(got["vram_gb"], 104.0)
        self.assertIn("shared with RAM", setup.nvidia_mem_text(got))

    def test_other_nvidia_cards_are_not_changed(self):
        gpu = {"name": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}
        with mock.patch.object(setup, "WIN_ARM64", True):
            self.assertIs(setup.nvidia_apply_uma(gpu, 120.0), gpu)
        self.assertNotIn("uma", gpu)

    def test_arm64_host_is_detected(self):
        self.assertEqual(setup.ARM64, setup.MACHINE in ("arm64", "aarch64"))

    def test_linux_arm64_uses_native_cpu_build(self):
        with mock.patch.object(setup, "ARM64", True), mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.cpu_build_floor(False), "")
            self.assertFalse(setup.arm64_model_supported("Q2_0"))
            self.assertTrue(setup.arm64_model_supported("IQ2_XS"))

    def test_linux_spark_memory_is_left_to_cuda_runtime(self):
        gpu = {"name": "NVIDIA DGX Spark GB10", "vram_gb": 119.0}
        with mock.patch.object(setup, "WIN_ARM64", False):
            self.assertIs(setup.nvidia_apply_uma(gpu, 120.0), gpu)
        self.assertNotIn("spark_uma", gpu)

    def test_linux_arm64_check_uses_shared_policy(self):
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["128GB-1x24GB"]
        code, text, _, _ = install(ram, found, ["--check"], extra=[
            mock.patch.object(setup, "ARM64", True),
            mock.patch.object(setup, "WIN", False),
            mock.patch.object(setup, "WIN_ARM64", False),
        ])
        self.assertEqual(code, 0, text)
        self.assertIn("Linux ARM64 support is EXPERIMENTAL", text)
        self.assertIn("Q2_0     needs ~48 GB RAM: not supported on ARM64", text)
        self.assertIn("IQ2_XS   needs ~48 GB RAM: fits", text)

    def test_native_arm64_clang_is_selected(self):
        clang = ROOT / "fake-clang-cl.exe"
        with mock.patch.dict(setup.os.environ, {"STRATA_CLANG": str(clang)}), \
             mock.patch.object(setup.Path, "exists", return_value=True), \
             mock.patch.object(setup, "out", return_value="Target: aarch64-pc-windows-msvc"):
            self.assertEqual(setup.find_clang_cl(), str(clang))

    def test_arm64_visual_studio_is_not_pinned_to_one_version(self):
        seen = []
        with mock.patch.object(setup, "WIN_ARM64", True), \
             mock.patch.object(setup.Path, "exists", return_value=True), \
             mock.patch.object(setup, "out", side_effect=lambda cmd, *a, **k: seen.append(cmd) or "C:/VS"):
            self.assertEqual(setup.find_vcvars((13, 4)), Path("C:/VS/VC/Auxiliary/Build/vcvarsarm64.bat"))
        self.assertIn("[16.0,19.0)", seen[0])
        self.assertIn("Microsoft.VisualStudio.Component.VC.Tools.ARM64", seen[0])

    def test_arm64_msvc_cuda_host_compiler_is_found(self):
        with tempfile.TemporaryDirectory() as td:
            vc = Path(td) / "VC"
            vcvars = vc / "Auxiliary" / "Build" / "vcvarsarm64.bat"
            compiler = vc / "Tools" / "MSVC" / "14.51.36231" / "bin" / "Hostarm64" / "arm64" / "cl.exe"
            vcvars.parent.mkdir(parents=True)
            compiler.parent.mkdir(parents=True)
            vcvars.touch()
            compiler.touch()
            self.assertEqual(setup.find_msvc_arm64_cl(vcvars), str(compiler))


if __name__ == "__main__":
    unittest.main()
