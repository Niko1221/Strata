"""HIP image encoder build and cache tests. No GPU, ROCm, or downloads required.

    python -m unittest tools.test_setup_hip
"""

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402


class HipVisionBuild(unittest.TestCase):
    def test_modes_missing_binary_and_multi_architectures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            hashes = {setup.ENGINE_SOURCES: "engine", setup.VISION_SOURCES: "vision-1"}

            def fake_build(src, bdir, target, defs, vcvars, bat_name):
                calls.append((target, defs))
                output = bdir / (setup.EXE if target == "strata" else f"bin/{setup.VEXE}")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(target.encode())

            gpu = {"arch": "gfx1201", "archs": ["gfx1201", "gfx1100"]}
            with (patch.object(setup, "ROOT", root),
                  patch.object(setup, "rocm_root", return_value=(root / "rocm", [str(root / "rocm/lib")])),
                  patch.object(setup, "source_hash", side_effect=lambda parts: hashes[parts]),
                  patch.object(setup, "source_version", return_value="test"),
                  patch.object(setup, "cmake_build", side_effect=fake_build),
                  patch.object(setup.shutil, "which", return_value="/usr/bin/c++"),
                  patch.object(setup, "say"), patch.object(setup, "ok"),
                  patch.dict(setup.os.environ, {}, clear=True)):
                setup.build_engine_hip(gpu, root / "llama", "gpu")
                self.assertEqual([target for target, _ in calls], ["strata", "strata-vision"])
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201", calls[0][1])
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201", calls[1][1])
                self.assertIn("-DSTRATA_VISION_HIP=ON", calls[1][1])
                self.assertIn("-DSTRATA_VISION_CUDA=OFF", calls[1][1])
                setup.build_engine_hip(gpu, root / "llama", "gpu")
                self.assertEqual(len(calls), 2)

                setup.build_engine_hip(gpu, root / "llama", "cpu")
                self.assertEqual([target for target, _ in calls[2:]], ["strata-vision"])
                self.assertIn("-DSTRATA_VISION_HIP=OFF", calls[-1][1])
                setup.build_engine_hip(gpu, root / "llama", "none")
                self.assertEqual(len(calls), 3)
                self.assertEqual(json.loads((root / "engine/BUILD.json").read_text())["vision"], "none")

                setup.build_engine_hip(gpu, root / "llama", "gpu")
                self.assertEqual([target for target, _ in calls[3:]], ["strata-vision"])
                (root / "engine" / setup.VEXE).unlink()
                setup.build_engine_hip(gpu, root / "llama", "gpu")
                self.assertEqual([target for target, _ in calls[4:]], ["strata-vision"])
                hashes[setup.VISION_SOURCES] = "vision-2"
                setup.build_engine_hip(gpu, root / "llama", "gpu")
                self.assertEqual([target for target, _ in calls[5:]], ["strata-vision"])

                setup.build_engine_hip({"arch": "gfx1201", "archs": ["gfx1201", "gfx1200"]},
                                       root / "llama", "gpu")
                self.assertEqual([target for target, _ in calls[6:]], ["strata", "strata-vision"])
                self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1200;gfx1201", calls[-1][1])
                meta = json.loads((root / "engine/BUILD.json").read_text())
                self.assertEqual(meta["archs"], ["gfx1200", "gfx1201"])
                self.assertEqual(meta["vision_archs"], ["gfx1200", "gfx1201"])

    def test_installed_engine_rebuilds_saved_vision_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).write_bytes(b"engine")
            gpu = {"arch": "gfx1201"}
            meta = {"backend": "hip", "src": "engine", "archs": ["gfx1201"],
                    "vision": "gpu", "vision_src": "old", "vision_archs": ["gfx1201"]}
            (eng / "BUILD.json").write_text(json.dumps(meta))
            with (patch.object(setup, "ROOT", root),
                  patch.object(setup, "source_hash", side_effect=lambda parts:
                               "engine" if parts == setup.ENGINE_SOURCES else "new"),
                  patch.object(setup, "amd_gpus", return_value=[gpu]),
                  patch.object(setup, "get_llama_cpp", return_value=root / "llama"),
                  patch.object(setup, "build_engine_hip") as build):
                setup.update_installed_engine("")
                build.assert_called_once_with({**gpu, "archs": ["gfx1201"]}, root / "llama", "gpu")
                build.reset_mock()
                meta.update(vision="cpu", vision_src="new", vision_archs=[])
                (eng / "BUILD.json").write_text(json.dumps(meta))
                setup.update_installed_engine("")  # missing CPU helper
                build.assert_called_once_with({**gpu, "archs": ["gfx1201"]}, root / "llama", "cpu")
                build.reset_mock()
                (eng / setup.VEXE).write_bytes(b"encoder")
                setup.update_installed_engine("")
                build.assert_not_called()

    def test_vision_uses_selected_hip_device_environment(self):
        from serve.server import Vision, child_env

        cfg = {"backend": "hip", "gpu": 2, "vision": {"gpu": True}}
        env = child_env(cfg)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")
        self.assertEqual(child_env({**cfg, "gpu": [2, 0]})["HIP_VISIBLE_DEVICES"], "2,0")
        with (patch("serve.server.subprocess.Popen") as popen,
              patch("serve.server.contain")):
            popen.return_value.stdout = io.StringIO("READY\n")
            vision = Vision({"exe": "strata-vision", "mmproj": "projector", "model": "model", "gpu": True},
                            env=env)
            try:
                self.assertIn("--gpu", popen.call_args.args[0])
                self.assertEqual(popen.call_args.kwargs["env"]["HIP_VISIBLE_DEVICES"], "2")
            finally:
                shutil.rmtree(vision.dir)


if __name__ == "__main__":
    unittest.main()
