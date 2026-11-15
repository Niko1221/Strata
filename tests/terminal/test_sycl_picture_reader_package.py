"""The Intel Arc (SYCL) engine package carries its picture readers: strata-vision (CPU) and strata-vision-sycl (Intel GPU).

Without them, strata_control/vision_reader.find() sees no reader in the engine folder and pictures are off on every Arc:
`strata launch --vision on` says "this engine has no picture reader".  This test checks the packer selection (no GPU, no
toolkit) and the packaging recipe's staging commands, so the defect cannot come back unnoticed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_engine  # noqa: E402
sys.path.insert(0, str(ROOT))
from strata_control import vision_reader as VR  # noqa: E402

SCRIPT = ROOT / "packaging" / "build_sycl_linux.sh"


class SyclPackageReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name)

    def make(self, names, vision_archs="intel-arc"):
        (self.src / "strata").write_text("x", encoding="utf-8")
        for n in names:
            (self.src / n).write_text("x", encoding="utf-8")
        (self.src / "BUILD.json").write_text(json.dumps({
            "version": "0.2.13", "archs": vision_archs, "vision": "gpu",
            "vision_cpu": "strata-vision", "vision_gpu": "strata-vision-sycl", "vision_archs": vision_archs,
        }), encoding="utf-8")

    def test_the_sycl_reader_is_taken_by_the_packer(self):
        self.make(["strata-vision", "strata-vision-sycl"])
        sel = package_engine.select(self.src, "strata", tree=False, include=["SYSTEM-LIBS.txt"])
        self.assertIn("strata-vision", sel)
        self.assertIn("strata-vision-sycl", sel)

    def test_an_intel_engine_folder_finds_both_readers(self):
        self.make(["strata-vision", "strata-vision-sycl"])
        have = VR.find(self.src, windows=False)
        self.assertIsNotNone(have["cpu"])
        self.assertIsNotNone(have["gpu"])
        r = VR.choose(self.src, "gpu", windows=False)
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name), ("gpu", "strata-vision-sycl", "strata-vision"))
        section = VR.vision_section(r, {"mmproj": "m", "model": "g"})
        self.assertTrue(section["gpu"])
        self.assertEqual(Path(section["cpu_exe"]).name, "strata-vision")

    def test_a_package_with_only_the_cpu_reader_is_still_usable(self):
        self.make(["strata-vision"])
        r = VR.choose(self.src, "gpu", windows=False)
        self.assertEqual((r.mode, r.exe.name), ("cpu", "strata-vision"))
        self.assertIn("no GPU picture reader", r.note)

    def test_the_reader_is_chmod_ed_in_the_tar(self):
        self.make(["strata-vision", "strata-vision-sycl"])
        out = Path(self.tmp.name) / "out"
        out.mkdir()
        folder = package_engine.build(self.src, "sycl", "0.2.13", "linux-x86_64", out,
                                      archs=("intel-arc",), check_reader=True, build_id="t")
        for name in ("strata-vision", "strata-vision-sycl"):
            p = folder / name
            self.assertTrue(p.is_file(), name)
            self.assertEqual(p.stat().st_mode & 0o777, 0o755, name)

    def test_the_recipe_builds_and_stages_both_readers(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for needle in ("STRATA_VISION_SYCL=ON", "strata-vision-sycl", "vision_archs", "vision_cpu",
                       "--target strata-vision", "patchelf --set-rpath", "unset SETVARS_COMPLETED"):
            self.assertIn(needle, text, needle)
        # the package's BUILD.json records the readers, so setup's vision_reader rule can pick the GPU one
        self.assertIn("vision_gpu", text)

    def test_a_sycl_package_with_only_the_cpu_reader_says_so(self):
        self.make(["strata-vision"])
        notes = package_engine.check_picture_reader(self.src, "sycl", "linux-x86_64", ("intel-arc",))
        self.assertIn("CPU reader only", notes[0])

    def test_a_sycl_package_without_any_reader_says_pictures_are_off(self):
        self.make([])
        notes = package_engine.check_picture_reader(self.src, "sycl", "linux-x86_64", ("intel-arc",))
        self.assertIn("pictures are off", notes[0])


if __name__ == "__main__":
    unittest.main()
