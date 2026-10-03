"""serve/test_engine_control.py - the engine switch (serve/engine_control.py): a start in the background with its
progress, Stop that keeps the model unloaded (a 503, also after a restart), a cancelled start, and the HTTP guards.
No GPU: the mock engine starts like the real one (a child process writing the engine's log lines).

    python -m unittest serve.test_engine_control -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.engine_control import EngineControl, MockStartable  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, GpuBusy, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wait_for(fn, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.1)
    return False


class Control(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        tok = ByteTokenizer()
        self.engine = MockStartable(MockEngine(tok, "ok", max_context=4096), 1.0, str(self.dir / "engine.log"))
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def tearDown(self):
        self.engine.unload()
        self.tmp.cleanup()

    def control(self):
        return EngineControl(self.svc, self.dir / "ws", self.engine.log_path)

    def test_start_reports_progress_then_runs(self):
        ctl = self.control()
        self.assertEqual(ctl.status()["state"], "stopped")
        ctl.start()
        seen = set()
        self.assertTrue(wait_for(lambda: seen.add(json.dumps(ctl.status().get("progress", {}).get("phase")))
                                 or ctl.status()["state"] == "running"))
        self.assertTrue({'"experts"', '"up"'} & seen or len(seen) > 2, seen)
        st = ctl.status()
        self.assertEqual((st["state"], st["version"]), ("running", "mock"))
        self.assertGreater(st["ready_s"], 0)
        self.assertEqual(json.loads((self.dir / "ws" / "engine.json").read_text())["arena_gib"], 0.53)

    def test_stop_holds_until_start_and_outlives_a_restart(self):
        ctl = self.control()
        ctl.start()
        self.assertTrue(wait_for(lambda: ctl.status()["state"] == "running"))
        self.assertEqual(ctl.stop()["result"], "unloaded")
        self.assertFalse(self.svc.loaded())
        with self.assertRaises(GpuBusy):                 # a request does not load it again
            self.svc.load()
        again = self.control()                           # the server starting again
        again.boot()
        self.assertEqual((again.status()["state"], again.status()["held"]), ("stopped", True))
        again.start()
        self.assertTrue(wait_for(lambda: again.status()["state"] == "running"))
        self.assertTrue(again.wanted())

    def test_cancel_a_start(self):
        ctl = self.control()
        ctl.start()
        self.assertTrue(wait_for(lambda: (ctl.status().get("progress") or {}).get("phase") == "experts"))
        self.assertEqual(ctl.stop()["result"], "cancelled")
        self.assertEqual(ctl.status()["state"], "stopped")
        self.assertIsNone(ctl.status()["error"])
        self.assertFalse(self.engine.alive())
        self.assertIsNotNone(self.engine.proc.poll())     # the child is gone

    def test_last_start_size_from_the_log(self):
        (self.dir / "engine.log").write_text("strata generate: loaded 46.84 GiB at 1.92 GiB/s\n")
        self.assertEqual(self.control().expected_gib, 46.84)


class Http(unittest.TestCase):
    """GET /engine needs the key; POST /engine/start and /engine/stop also Strata's own page."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = MockStartable(MockEngine(tok, "ok", max_context=4096), 0.5)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.api_key = "k3y"
        cls.svc.engine_control = EngineControl(cls.svc, Path(cls.tmp.name) / "ws", cls.engine.log_path)
        cls.httpd = serve(cls.svc, port=0)
        cls.host = f"127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.engine.unload()
        cls.tmp.cleanup()

    def call(self, path, body=None, key="k3y", origin=None):
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = "Bearer " + key
        if origin:
            h["Origin"] = origin
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://{self.host}{path}", data=data, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_routes_and_guards(self):
        self.assertEqual(self.call("/engine", key=None)[0], 401)
        self.assertEqual(self.call("/engine/start", {}, origin="http://evil.example")[0], 403)
        code, st = self.call("/engine/start", {}, origin=f"http://{self.host}")
        self.assertEqual((code, st["state"]), (200, "starting"))
        self.assertTrue(wait_for(lambda: self.call("/engine")[1]["state"] == "running"))
        code, st = self.call("/engine/stop", {}, origin=f"http://{self.host}")
        self.assertEqual((code, st["result"]), (200, "unloaded"))
        code, err = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        self.assertEqual(code, 503)
        self.assertIn("engine is stopped", err["error"]["message"])


if __name__ == "__main__":
    unittest.main()
