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

    def test_a_json_that_is_no_model_config_is_skipped(self):
        """#549: a strata-*.json without "args" (not written by setup) stopped update.sh with KeyError: 'args'."""
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            other = Path(d) / "strata-notes.json"
            other.write_text(json.dumps({"note": "mine"}), encoding="utf-8")
            broken = Path(d) / "strata-cut.json"
            broken.write_text("{\"args\": [", encoding="utf-8")
            rc, pip, eng, dv, start, call, out = self.run_update([other, broken, p])
        self.assertEqual(rc, 0, out)
        self.assertIn('skipped strata-notes.json (no "exe" or "args"): it is not a Strata model config', out)
        self.assertIn("skipped strata-cut.json (not valid JSON)", out)
        dv.assert_called_once()                                     # the real model is still refreshed
        self.assertIn("Qwen IQ3_S: up to date", out)
        self.assertIn("Strata is updated", out)

    def local_engine(self, d: Path, archs=(89,)) -> Path:
        eng = Path(d) / "engine"
        eng.mkdir()
        (eng / setup.EXE).write_bytes(b"")
        (eng / "BUILD.json").write_text(json.dumps({"source": "local", "archs": list(archs), "src": "stale",
                                                  "vision": "none", "version": "0.1.38"}), encoding="utf-8")
        return eng

    def rebuild(self, cards, eng):
        with mock.patch.object(setup, "engine_dir", return_value=eng), \
                mock.patch.object(setup, "gpus", return_value=cards), \
                mock.patch.object(setup, "get_llama_cpp", return_value=None), \
                mock.patch.object(setup, "build_engine") as build, \
                contextlib.redirect_stdout(io.StringIO()):
            setup.update_installed_engine("URL", 13)
        return build

    def test_a_local_rebuild_uses_a_card_the_engine_serves(self):
        """#1485: the most-VRAM pick can be a card the model does not run on - a V100 (sm_70) beside the model's
        two 4090s entered the CUDA 13 rebuild's arch list, which CUDA 13 cannot compile."""
        cards = [{"index": 0, "name": "RTX 4090", "vram_gb": 24.0, "arch": "89"},
                 {"index": 1, "name": "Tesla V100", "vram_gb": 32.0, "arch": "70"},
                 {"index": 2, "name": "RTX 4090", "vram_gb": 24.0, "arch": "89"}]
        with tempfile.TemporaryDirectory() as d:
            build = self.rebuild(cards, self.local_engine(Path(d), (89,)))
        build.assert_called_once()
        gpu = build.call_args[0][0]
        self.assertEqual(gpu["archs"], [89])
        self.assertEqual(int(gpu["arch"]), 89)

    def test_a_local_rebuild_without_a_served_card_keeps_the_union(self):
        """No card this engine was built for is installed (the folder moved PCs): the old union stands, so the
        card that is here gets code."""
        cards = [{"index": 0, "name": "RTX 2080 Ti", "vram_gb": 11.0, "arch": "75"}]
        with tempfile.TemporaryDirectory() as d:
            build = self.rebuild(cards, self.local_engine(Path(d), (89,)))
        build.assert_called_once()
        self.assertEqual(build.call_args[0][0]["archs"], [75, 89])

    def test_installed_configs_lists_only_model_configs(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            (Path(d) / "strata-notes.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(setup, "ROOT", Path(d)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.installed_configs(), [p])

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


class SettingsLine(unittest.TestCase):
    """#564: a start prints the settings it uses (the engine options without the model's paths, and the server's
    fields), so a change made by hand to strata-<model>.json shows without reading the log."""

    CFG = {"exe": "x", "host": "0.0.0.0", "port": 8081, "api_key": "secret", "gpu": [0, 1], "fit_max_tokens": True,
           "args": ["--native", "E:\\Strata\\packs\\iq3_s", "--mtp", "/s/mtp/rt", "m.gguf", "--kv", "int8",
                    "--kv-resident", "32768", "--spec-min-p", "0.5", "--vram-reserve-mib", "2048", "--mmap-experts",
                    "--prefill", "auto"]}

    def test_summary(self):
        s = setup.settings_summary(self.CFG)
        self.assertEqual(s, "--kv int8 --kv-resident 32768 --spec-min-p 0.5 --vram-reserve-mib 2048 --mmap-experts "
                            "--prefill auto; server 0.0.0.0:8081, api key set, gpu 0,1, fit_max_tokens true")
        self.assertNotIn("secret", s)
        self.assertIn("127.0.0.1:9000", setup.settings_summary({"args": []}, 9000))

    def test_a_start_prints_it(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-iq3_s.json"
            args = [x for x in self.CFG["args"] if x != "m.gguf"]           # no model file here
            p.write_text(json.dumps({**self.CFG, "exe": str(exe), "gpu": 0, "args": args}), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(setup, "gpus", lambda: []), \
                    mock.patch.object(setup, "refresh_draft_vocab"), \
                    mock.patch.object(setup, "is_wsl", lambda: False), \
                    mock.patch.object(setup.subprocess, "call", return_value=0), \
                    contextlib.redirect_stdout(out):
                setup.start(p, None, open_browser=False, yes=True)
        text = " ".join(out.getvalue().split())
        self.assertIn("Settings (strata-iq3_s.json): --kv int8 --kv-resident 32768", text)
        self.assertIn("--vram-reserve-mib 2048", text)


if __name__ == "__main__":
    unittest.main()
