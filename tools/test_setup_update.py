"""Tests for `setup.py --update` (#475, UPDATE.bat / update.sh): it refreshes what a start would - the Python packages,
the engine, each model's config and draft subset - and never starts the model.  Every outside effect is mocked: no
GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_update
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Update(unittest.TestCase):
    def config(self, d: Path, **extra) -> Path:
        rt = d / "mtp" / "rt"
        rt.mkdir(parents=True)
        p = d / "strata-iq3_s.json"
        p.write_text(json.dumps({"model_name": "Qwen IQ3_S", "exe": str(d / "engine" / setup.EXE),
                                 "args": ["--native", "x", "--mtp", str(rt)], **extra}), encoding="utf-8")
        return p

    def run_update(self, have, argv=()):
        a = mock.Mock(**{"build": False, "prebuilt": "URL", **dict(argv)})
        out = io.StringIO()
        with mock.patch.object(setup, "pip_install") as pip, \
                mock.patch.object(setup, "update_installed_engine") as eng, \
                mock.patch.object(setup, "refresh_draft_vocab") as dv, \
                mock.patch.object(setup, "start") as start, \
                mock.patch("subprocess.call") as call, \
                contextlib.redirect_stdout(out):
            rc = setup.update_install(have, a)
        return rc, pip, eng, dv, start, call, out.getvalue()

    def test_refreshes_without_starting(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d), draft_vocab="en")
            rc, pip, eng, dv, start, call, out = self.run_update([p])
        self.assertEqual(rc, 0)
        pip.assert_called_once()
        eng.assert_called_once_with("URL")
        dv.assert_called_once()
        self.assertEqual(dv.call_args[0][1], "en")                 # the model's own subset is kept
        start.assert_not_called()
        call.assert_not_called()
        self.assertIn("Strata is updated", out)

    def test_build_keeps_the_compiled_engine_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            rc, pip, eng, dv, *_ = self.run_update([p], {"build": True})
        self.assertEqual(rc, 0)
        eng.assert_not_called()
        self.assertEqual(dv.call_args[0][1], "cjk")

    def test_nothing_installed(self):
        rc, pip, eng, dv, start, call, out = self.run_update([])
        self.assertEqual(rc, 0)
        for m in (pip, eng, dv, start, call):
            m.assert_not_called()
        self.assertIn("START-HERE.bat", out)

    def test_main_update_never_starts(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            with mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                    mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                    mock.patch.object(setup, "installed_configs", return_value=[p]), \
                    mock.patch.object(setup, "update_install", return_value=0) as up, \
                    mock.patch.object(setup, "start") as start, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
        up.assert_called_once()
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
