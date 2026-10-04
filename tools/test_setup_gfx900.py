"""gfx900 opt-in and build routing, without ROCm, downloads or a GPU."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class Gfx900(unittest.TestCase):
    def test_gate_is_explicit_and_linux_only(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(setup, "WIN", False):
            self.assertIn("STRATA_EXPERIMENTAL_GFX900=1", setup.amd_problem({"arch": "gfx900"}))
            self.assertIsNone(setup.amd_problem({"arch": "gfx1100"}))
            for value in ("0", "yes", "1"):
                os.environ["STRATA_EXPERIMENTAL_GFX900"] = value
                self.assertEqual(setup.amd_problem({"arch": "gfx900"}) is None, value == "1")
            with mock.patch.object(setup, "WIN", True):
                self.assertIsNotNone(setup.amd_problem({"arch": "gfx900"}))
            self.assertEqual(setup.amd_problem({"arch": "gfx900", "cannot_run": "runtime failed"}), "runtime failed")

    def test_rocm_pin_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(setup.rocm_version_for(["gfx900"]), "7.14.0a20260612")
            self.assertEqual(setup.rocm_version_for(["gfx1201"]), setup.ROCM_VERSION)
            self.assertTrue(setup.rocm_index("gfx900").endswith("/gfx900/"))
            os.environ["STRATA_ROCM_VERSION"] = "custom"
            self.assertEqual(setup.rocm_version_for(["gfx900"]), "custom")

    def test_build_selects_wave64_and_records_arch(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_GFX900": "1"}), \
             mock.patch.object(setup, "WIN", False):
            root = Path(d)

            def build(src, bdir, target, defs, *rest):
                self.assertEqual(bdir, root / "build-gfx900")
                self.assertIn("-DSTRATA_HIP_GFX900=ON", defs)
                self.assertIn("-DSTRATA_ENABLE_HIP=OFF", defs)
                self.assertNotIn("-DSTRATA_PREFILL_MMQ=ON", defs)
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx900", defs)
                bdir.mkdir()
                (bdir / setup.EXE).write_text("engine")
                (bdir / "strata-device").write_text("probe")

            with mock.patch.object(setup, "ROOT", root), \
                 mock.patch.object(setup, "source_hash", return_value="hash"), \
                 mock.patch.object(setup, "source_version", return_value="0.1.39"), \
                 mock.patch.object(setup, "cpu_info", return_value=("CPU", {"avx2"})), \
                 mock.patch.object(setup.shutil, "which", return_value="/usr/bin/tool"), \
                 mock.patch.object(setup, "rocm_root", return_value=(root, [str(root / "lib")])), \
                 mock.patch.object(setup, "cmake_build", side_effect=build) as compile_engine, \
                 mock.patch.object(setup, "run") as run_probe, \
                 mock.patch.object(setup, "say"):
                setup.build_engine_hip({"arch": "gfx900"}, root)
                run_probe.assert_called_once_with([str(root / "engine/strata-device"), "--selftest"])
                meta = json.loads((root / "engine/BUILD.json").read_text())
                self.assertEqual(meta["archs"], ["gfx900"])
                self.assertEqual(meta["backend"], "hip")
                setup.build_engine_hip({"arch": "gfx900"}, root)
                self.assertEqual(compile_engine.call_count, 1)
                with self.assertRaises(SystemExit):
                    setup.build_engine_hip({"arch": "gfx900", "archs": ["gfx900", "gfx1100"]}, root)
                self.assertEqual(compile_engine.call_count, 1)


if __name__ == "__main__":
    unittest.main()
