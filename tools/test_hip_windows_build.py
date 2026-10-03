"""Windows HIP configure regressions, with a stub CMake command (no compiler, ROCm or GPU).

    python tools/test_hip_windows_build.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "executes a Windows batch configure command")
class WindowsHipConfigure(unittest.TestCase):
    def test_fresh_prompt_clears_previous_source_override(self):
        # Exercise the actual engine configure section with CMake replaced by a
        # recorder. Its saved source represents CMake's persistent cache.
        script = (ROOT / "tools/hip/build_windows.bat").read_text()
        section = script[script.index("rem ---- 3."):script.index('if "%TESTS%"=="ON" (')]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "cache.json"
            capture = root / "capture.py"
            capture.write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "p = Path(os.environ['STRATA_TEST_BUILD_RECORD'])\n"
                "cache = json.loads(p.read_text()) if p.exists() else {}\n"
                "for arg in sys.argv[1:]:\n"
                "    if arg.startswith('-DSTRATA_GGML_DIR='):\n"
                "        cache['source'] = arg.split('=', 1)[1]\n"
                "p.write_text(json.dumps(cache))\n")
            (root / "cmake.cmd").write_text(
                f'@echo off\n"{sys.executable}" "{capture}" %*\n', newline="\r\n")
            configure = root / "configure.cmd"
            configure.write_text('@echo off\nsetlocal EnableDelayedExpansion\n' + section,
                                 newline="\r\n")
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ.get("PATH", ""),
                   "SRC": str(ROOT), "BUILD_DIR": str(root / "build"), "TESTS": "OFF",
                   "STRATA_HIP_ARCHS": "gfx1100", "ROCM_F": str(root / "rocm"), "BITCODE": "bitcode",
                   "STRATA_TEST_BUILD_RECORD": str(record)}
            custom = str(root / "custom llama checkout")
            for source in (custom, None):
                if source is None:
                    env.pop("STRATA_GGML_DIR", None)
                else:
                    env["STRATA_GGML_DIR"] = source
                result = subprocess.run(["cmd", "/d", "/c", str(configure)], cwd=root,
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads(record.read_text())["source"],
                                 source.replace("\\", "/") if source else "")


if __name__ == "__main__":
    unittest.main()
