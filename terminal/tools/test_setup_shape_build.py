"""#6952: a shape engine compiled with --build (strata-inkling, strata-hy4) gets code for the cards in the PC, like the Strata engine.
Its CMake step had no CMAKE_CUDA_ARCHITECTURES, so engine/cmake/backends/cuda.cmake built it for sm_120 only, and on an RTX 3090
strata-inkling launched no kernel and answered "!!!!" at ~1,000 tok/s.

    python -m unittest tools.test_setup_shape_build
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import shape_engine  # noqa: E402

NVCC = "/usr/local/cuda-13.1/bin/nvcc"


def card(index, arch, vram=24.0):
    return {"index": index, "arch": arch, "vram_gb": vram, "name": f"card{index}"}


class ShapeEngineBuild(unittest.TestCase):
    def build(self, shape, gpu, *, hip=False, cards=(), nvcc=NVCC):
        """build_shape_engine with the commands recorded instead of run; returns (the CMake configure command, the install_build_tools mock)."""
        tools = mock.Mock(return_value=(nvcc, None))
        cmds = []
        with tempfile.TemporaryDirectory() as tmp:
            src, home = Path(tmp) / "engine", Path(tmp) / "terminal"
            src.mkdir()
            home.mkdir()
            (src / "CMakeLists.txt").write_text("", encoding="utf-8")

            def fake_run(cmd, *args, **kw):
                cmds.append([str(c) for c in cmd])
                if "--build" in cmd:                   # the compile: leave the executable where the build puts it
                    exe = src / f"build-{shape}" / shape_engine.exe_name(shape)
                    exe.parent.mkdir(parents=True, exist_ok=True)
                    exe.write_text("", encoding="utf-8")

            with mock.patch.object(setup, "engine_src", return_value=src), \
                    mock.patch.object(setup, "home", return_value=home), \
                    mock.patch.object(setup, "run", side_effect=fake_run), \
                    mock.patch.object(setup, "gpus", return_value=list(cards)), \
                    mock.patch.object(setup, "install_build_tools", tools), \
                    mock.patch.object(setup.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"), \
                    mock.patch.object(setup.sys, "platform", "linux"), \
                    contextlib.redirect_stdout(io.StringIO()):
                setup.build_shape_engine(shape, gpu, hip, 13, True)
        return cmds[0], tools

    def test_inkling_is_built_for_the_card_in_the_pc(self):
        cfg, tools = self.build(shape_engine.INKLING_SHAPE, {"arch": 86, "archs": [86]}, cards=[card(0, 86)])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", cfg)
        self.assertIn(f"-DCMAKE_CUDA_COMPILER={NVCC}", cfg)
        self.assertEqual(tools.call_args.args[0]["archs"], [86])
        self.assertEqual(tools.call_args.args[0]["toolkit"], 13)

    def test_hy4_is_built_for_the_card_in_the_pc(self):
        cfg, _ = self.build(shape_engine.HY4_SHAPE, {"arch": 89, "archs": [89]}, cards=[card(0, 89)])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89", cfg)
        self.assertIn(f"-DCMAKE_CUDA_COMPILER={NVCC}", cfg)

    def test_every_usable_card_in_the_pc_gets_code(self):
        cfg, _ = self.build(shape_engine.INKLING_SHAPE, {"arch": 86, "archs": [86]}, cards=[card(0, 86), card(1, 120, 32.0)])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86;120", cfg)

    def test_windows_paths_get_forward_slashes(self):
        """#5694: CMake stops on a back-slashed CMAKE_CUDA_COMPILER, and this configure does not go through cmake_build."""
        nvcc = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
        with mock.patch.object(setup, "WIN", True):
            cfg, _ = self.build(shape_engine.INKLING_SHAPE, {"arch": 86, "archs": [86]}, cards=[card(0, 86)], nvcc=nvcc)
        self.assertIn("-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe", cfg)

    def test_hip_gets_no_cuda_options(self):
        for shape in (shape_engine.INKLING_SHAPE, shape_engine.HY4_SHAPE):
            cfg, tools = self.build(shape, {"arch": 86, "archs": [86]}, hip=True)
            self.assertFalse([d for d in cfg if d.startswith("-DCMAKE_CUDA")], shape)
            tools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
