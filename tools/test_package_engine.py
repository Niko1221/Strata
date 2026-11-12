"""Regression tests for the Windows CUDA CPU-only vision reader package.

    python -m unittest tools.test_package_engine
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
import package_engine as PE


class PackageVisionReaders(unittest.TestCase):
    def test_default_selection_includes_the_cpu_companion(self):
        with tempfile.TemporaryDirectory(prefix="strata-package-vision-") as tmp:
            src = Path(tmp)
            for name in ("strata.exe", "strata-vision.exe", "strata-vision-cpu.exe"):
                (src / name).write_bytes(b"test")
            selected = PE.select(src, "strata.exe", tree=False, include=[])
            self.assertIn("strata-vision.exe", selected)
            self.assertIn("strata-vision-cpu.exe", selected)

    def test_cuda_arch_check_still_validates_the_cuda_reader(self):
        with tempfile.TemporaryDirectory(prefix="strata-package-vision-") as tmp:
            src = Path(tmp)
            (src / "strata-vision.exe").write_bytes(b"cuda-reader")
            (src / "strata-vision-cpu.exe").write_bytes(b"cpu-reader")
            (src / "BUILD.json").write_text(json.dumps({"vision_archs": "80"}), encoding="utf-8")
            notes = PE.check_picture_reader(src, "cuda", "windows-x86_64", ("sm_80",), variant="cuda12")
            self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
