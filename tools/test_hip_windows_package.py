"""Windows HIP package dependency tests, without a compiler or GPU.

    python -m unittest tools.test_hip_windows_package
"""
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.hip import package_windows as package  # noqa: E402


class VisionPackage(unittest.TestCase):
    def test_incomplete_encoder_options_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = ["package", "--build", str(root / "build"), "--rocm", str(root / "rocm"),
                    "--archs", "gfx1100", "--rocm-version", "test", "--out", str(root / "dist")]
            for extra in (["--vision-mode", "gpu"], ["--vision-build", str(root / "vision")],
                          ["--vision-build", str(root / "missing"), "--vision-mode", "cpu"]):
                with self.subTest(extra=extra), patch("sys.argv", args + extra), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    package.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((root / "dist").exists())

    def test_encoder_dependencies_and_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def put(name, data=b"fixture"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return path

            for program in package.PROGRAMS:
                put("build/" + program)
            put("vision/bin/strata-vision.exe")
            for dll in ("amd_comgr.dll", "amdhip64_7.dll", "encoder-only.dll", "encoder-child.dll"):
                put("rocm/bin/" + dll)
            put("rocm/bin/rocblas/library/TensileLibrary.dat")
            put("rocm/.kpack/blas_lib_gfx1100.kpack")
            put("rocm/include/hipblaslt/hipblaslt-version.h",
                b"#define HIPBLASLT_VERSION_MAJOR 1\n#define HIPBLASLT_VERSION_MINOR 5\n"
                b"#define HIPBLASLT_VERSION_PATCH 0\n")

            imports = {
                "strata.exe": ["amdhip64_7.dll"],
                "strata-vision.exe": ["encoder-only.dll"],
                "encoder-only.dll": ["encoder-child.dll"],
            }
            args = ["package", "--build", str(root / "build"), "--rocm", str(root / "rocm"),
                    "--archs", "gfx1100", "--rocm-version", "test", "--out", str(root / "dist")]
            with (patch.object(package, "imports", side_effect=lambda _, path: imports.get(path.name, [])),
                  patch.object(package, "crt_dir", return_value=root),
                  contextlib.redirect_stdout(io.StringIO())):
                for mode in ("none", "cpu", "gpu"):
                    extra = [] if mode == "none" else ["--vision-build", str(root / "vision"), "--vision-mode", mode]
                    with patch("sys.argv", args + extra):
                        self.assertEqual(package.main(), 0)
                    with zipfile.ZipFile(root / "dist" / package.ASSET) as archive:
                        names = archive.namelist()
                        meta = json.loads(archive.read("BUILD.json"))
                        self.assertEqual(meta["vision"], mode)
                        self.assertEqual(meta["vision_archs"], ["gfx1100"] if mode == "gpu" else [])
                        self.assertEqual("strata-vision.exe" in names, mode != "none")
                        for dll in ("encoder-only.dll", "encoder-child.dll"):
                            self.assertEqual("rocm/bin/" + dll in names, mode != "none")
                        self.assertIn("amdhip64_7.dll", names)


if __name__ == "__main__":
    unittest.main()
