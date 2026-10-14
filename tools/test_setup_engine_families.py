"""`engine_families()`'s environment: a ready-made Windows HIP engine keeps hipblas/libhipblaslt in
rocm/bin beside it, its loader asks for them before main, and the probe is launched by setup rather
than by a user who already has that PATH (#2372). No engine, no GPU, no network.

    python -m unittest tools.test_setup_engine_families
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class FamiliesProbe(unittest.TestCase):
    def probe(self, *, lib_dirs=("rocm/bin",), stdout="qwen4exp qwen36\n", returncode=0):
        tmp = tempfile.TemporaryDirectory(prefix="families-")
        self.addCleanup(tmp.cleanup)
        eng = Path(tmp.name) / "engine"
        (eng / "rocm" / "bin").mkdir(parents=True)
        (eng / "BUILD.json").write_text(
            json.dumps({"backend": "hip", "lib_dirs": list(lib_dirs)}), encoding="utf-8")
        (eng / setup.EXE).write_text("", encoding="utf-8")
        seen: dict = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen.update(kwargs)
            return mock.Mock(stdout=stdout, returncode=returncode)

        with mock.patch("subprocess.run", fake_run):
            families = setup.engine_families(eng)
        return eng, seen, families

    def test_the_probe_gets_the_engines_library_dirs_on_path(self):
        eng, seen, families = self.probe()
        self.assertEqual(families, {"qwen4exp", "qwen36"})
        self.assertIsNotNone(seen.get("env"), "the probe ran without an environment")
        entries = seen["env"]["PATH"].split(os.pathsep)
        self.assertEqual(Path(entries[0]), eng / "rocm" / "bin")
        self.assertIn("PATH", seen["env"], "the inherited environment is kept")
        self.assertEqual(seen["cmd"], [str(eng / setup.EXE), "--families"])

    def test_an_engine_without_lib_dirs_keeps_the_inherited_path(self):
        eng, seen, families = self.probe(lib_dirs=())
        self.assertEqual(families, {"qwen4exp", "qwen36"})
        entries = seen["env"]["PATH"].split(os.pathsep)
        self.assertNotIn(str(eng / "rocm" / "bin"), entries)
        self.assertEqual(seen["env"]["PATH"], os.environ.get("PATH", ""))

    def test_unparseable_output_keeps_the_flash_next_default(self):
        _eng, _seen, families = self.probe(stdout="")
        self.assertEqual(families, {"qwen4exp"})


if __name__ == "__main__":
    unittest.main()
