"""CPU-only HIP architecture gate regression tests; no ROCm, GPU or downloads.

Run: python -m unittest tools.test_hip_arch_gate
"""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class HipArchGate(unittest.TestCase):
    def gate(self, arch):
        source = (ROOT / "cmake/hip_backend.cmake").read_text().split("enable_language(HIP)")[0]
        with tempfile.TemporaryDirectory() as directory:
            script = pathlib.Path(directory) / "gate.cmake"
            script.write_text('cmake_minimum_required(VERSION 3.24)\n'
                              f'set(CMAKE_HIP_ARCHITECTURES "{arch}")\n' + source)
            return subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)

    def test_gfx1151_is_explicitly_unvalidated(self):
        result = self.gate("gfx1151")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not validated", result.stderr)

    def test_gfx1151_feature_suffix(self):
        result = self.gate("gfx1151:xnack-")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_other_apu_is_not_implicitly_admitted(self):
        result = self.gate("gfx1036")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CMAKE_HIP_ARCHITECTURES", result.stderr)

    def test_existing_validated_arch_stays_validated(self):
        result = self.gate("gfx1100")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("not validated", result.stderr)

    def test_gfx1151_native_signed_dot4_guard(self):
        source = (ROOT / "include/strata/hip_compat/intrinsics.hpp").read_text()
        branch = source.split("__has_builtin(__builtin_amdgcn_sudot4)")[0]
        self.assertIn("defined(__gfx1151__)", branch)


if __name__ == "__main__":
    unittest.main()
