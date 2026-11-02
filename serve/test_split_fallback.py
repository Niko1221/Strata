"""serve/server.py start_engine_or_fallback: a config for the all-cards split engine starts the normal layer split, with one plain line,
when the split engine refuses or fails to come up; any other config starts exactly as before and its failures stay failures."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as S  # noqa: E402


NORMAL = os.path.abspath("/e/strata")


class FakeEngine:
    started = []
    fail_exes = set()

    def __init__(self, exe, args, cwd=None, log=None, env=None, lazy=False, **kw):
        FakeEngine.started.append((exe, list(args), cwd, dict(env or {})))
        if exe in FakeEngine.fail_exes:
            raise S.EngineDied("the engine exited before it was ready: card 0 is compute capability 8.9 and this build is for sm_86")
        self.info = {}


def split_cfg():
    return {"exe": "/e/strata-split-2", "args": ["--native", "/m/a.gguf", "--batch", "6"], "cwd": "/x", "tokenizer": "/t", "port": 8080,
            "env": {"FNX_ARENA": "0", "FNX_EXPERT_RAM_GIB": "off"},
            "split_fallback": {"exe": NORMAL, "args": ["--pack", "/p", "--native", "/m/a.gguf", "--layer-split", "auto"],
                               "env": {"STRATA_X": "1"}, "layer_split": "auto"}}


class Fallback(unittest.TestCase):
    def setUp(self):
        FakeEngine.started, FakeEngine.fail_exes = [], set()
        self.p = [mock.patch.object(S, "StrataEngine", FakeEngine), mock.patch.object(S, "effort_end_args", lambda cfg, exe, tok: None),
                  mock.patch.object(S, "configure_batch_telemetry_env", lambda env, cfg, override: None)]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def run_start(self, cfg):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine = S.start_engine_or_fallback(cfg, cfg["exe"], [], {"A": "1"}, False, None)
        return engine, out.getvalue()

    def test_a_split_engine_that_starts_is_used(self):
        engine, out = self.run_start(split_cfg())
        self.assertEqual(len(FakeEngine.started), 1)
        self.assertEqual(FakeEngine.started[0][0], "/e/strata-split-2")
        self.assertNotIn("split_fallback", engine.info)
        self.assertEqual(out, "")

    def test_a_split_engine_that_fails_is_replaced_by_the_normal_layer_split_with_one_line(self):
        FakeEngine.fail_exes = {"/e/strata-split-2"}
        engine, out = self.run_start(split_cfg())
        self.assertEqual([s[0] for s in FakeEngine.started], ["/e/strata-split-2", NORMAL])
        exe, args, cwd, env = FakeEngine.started[1]
        self.assertIn("--layer-split", args)
        self.assertNotIn("--batch", args)
        self.assertEqual(env.get("STRATA_X"), "1")
        self.assertNotIn("FNX_ARENA", env)
        self.assertTrue(engine.info.get("split_fallback"))
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("all-cards split engine could not start", lines[0])
        self.assertIn("normal layer split", lines[0])

    def test_without_a_fallback_the_failure_stays_a_failure(self):
        cfg = split_cfg()
        del cfg["split_fallback"]
        FakeEngine.fail_exes = {"/e/strata-split-2"}
        with self.assertRaises(S.EngineDied):
            self.run_start(cfg)
        self.assertEqual(len(FakeEngine.started), 1)

    def test_a_normal_config_is_untouched(self):
        cfg = {"exe": "/e/strata", "args": ["--pack", "/p"], "cwd": "/x", "tokenizer": "/t"}
        FakeEngine.fail_exes = {"/e/strata"}
        with self.assertRaises(S.EngineDied):
            self.run_start(cfg)
        self.assertEqual(len(FakeEngine.started), 1)


if __name__ == "__main__":
    unittest.main()
