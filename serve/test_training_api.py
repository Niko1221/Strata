"""POST /v1/training/*: a training session takes the engine away from serving and gives it back.

The session's engine here is a fake (a python script speaking the interactive LORA/SCORE/GEN protocol over
stdio), but everything around it is the real path: TrainBridge's spawn and READY handshake, the Service
open/close/command methods, the 409 gate on inference while a session is open, and the automatic close when
the bridge's engine dies mid-command."""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True


FAKE_ENGINE = textwrap.dedent("""
    import sys
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "EXIT":
            break
        if line.startswith("LORA "):
            print("LORAOK 7 deadbeef", flush=True)
        elif line.startswith("SCORE "):
            ids = [x for x in line.split(" ", 1)[1].split(",") if x]
            for t in range(len(ids) - 1):
                print(f"SL {t} -0.5", flush=True)
            print(f"SSUM {-0.5 * (len(ids) - 1):.6f} {len(ids) - 1} 12.0 3.0 64", flush=True)
        elif line.startswith("GEN "):
            print("T 42", flush=True)
            print("DONE", flush=True)
        else:
            print("ERR unknown command", flush=True)
""")


class TrainingApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.engine_py = tempfile.mkstemp(suffix=".py")
        with __import__("os").fdopen(fd, "w", encoding="utf-8") as f:
            f.write(FAKE_ENGINE)
        cls.argv = [sys.executable, cls.engine_py]

    @classmethod
    def tearDownClass(cls):
        Path(cls.engine_py).unlink(missing_ok=True)

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)      # LIFO: shutdown runs first, then server_close
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body, timeout=30):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            return r.status, json.loads(r.read().decode())

    def get(self, path):
        try:
            r = urllib.request.urlopen(self.base + path, timeout=10)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            return r.status, json.loads(r.read().decode())

    def test_open_command_close_and_the_gate(self):
        code, body = self.post("/v1/training/session", {"op": "open", "argv": self.argv})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "open")
        pid = body["pid"]
        self.addCleanup(self.post, "/v1/training/session", {"op": "close"})

        code, body = self.get("/v1/training/status")
        self.assertEqual((code, body["active"], body["pid"]), (200, True, pid))

        # the gate: inference and engine control fast-fail while the session owns the engine
        for path, payload in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                              ("/unload", {}), ("/load", {}), ("/v1/vram", {})):
            code, body = self.post(path, payload)
            self.assertEqual(code, 409, (path, body))
            self.assertEqual(body["error"]["type"], "training_session_active")

        # a second open refuses, the first session's engine keeps working
        code, body = self.post("/v1/training/session", {"op": "open", "argv": self.argv})
        self.assertEqual(code, 409, body)

        code, body = self.post("/v1/training/command", {"line": "LORA /tmp/x.gguf", "until": ["LORAOK"]})
        self.assertEqual(code, 200, body)
        self.assertTrue(body["lines"][-1].startswith("LORAOK"))

        code, body = self.post("/v1/training/command", {"line": "SCORE 1,2,3", "until": ["SSUM"]})
        self.assertEqual(code, 200, body)
        sl = [l for l in body["lines"] if l.startswith("SL ")]
        self.assertEqual(len(sl), 2)                       # n ids -> n-1 per-position logps
        self.assertTrue(body["lines"][-1].startswith("SSUM -1.000000"))

        # the engine said ERR: a 502, the session stays (the caller may retry)
        code, body = self.post("/v1/training/command", {"line": "BOGUS", "until": ["DONE"]})
        self.assertEqual(code, 502, body)
        code, body = self.get("/v1/training/status")
        self.assertTrue(body["active"])

        code, body = self.post("/v1/training/session", {"op": "close"})
        self.assertEqual((code, body["status"]), (200, "closed"))
        code, body = self.get("/v1/training/status")
        self.assertEqual((code, body["active"], body["pid"]), (200, False, None))

        # closing again is harmless; serving resumes: /load starts the (stub) serving engine again
        code, body = self.post("/v1/training/session", {"op": "close"})
        self.assertEqual((code, body["status"], body["was_active"]), (200, "closed", False))
        code, body = self.post("/load", {})
        self.assertEqual((code, body["status"]), (200, "loaded"))

    def test_open_rejects_bad_argv(self):
        for argv in (None, [], "strata", [""], [1, 2]):
            code, body = self.post("/v1/training/session", {"op": "open", "argv": argv})
            self.assertEqual(code, 400, (argv, body))

    def test_command_needs_a_session(self):
        code, body = self.post("/v1/training/command", {"line": "SCORE 1", "until": ["SSUM"]})
        self.assertEqual(code, 409, body)
        self.assertEqual(body["error"]["type"], "no_training_session")

    def test_engine_death_closes_the_session(self):
        code, body = self.post("/v1/training/session", {"op": "open", "argv": self.argv})
        self.assertEqual(code, 200, body)
        # EXIT makes the fake engine leave its read loop: the bridge sees EOF mid-command
        code, body = self.post("/v1/training/command", {"line": "EXIT", "until": ["DONE"]}, timeout=15)
        self.assertEqual(code, 502, body)
        code, body = self.get("/v1/training/status")
        self.assertEqual((code, body["active"]), (200, False))

    def test_engine_that_fails_to_start(self):
        code, body = self.post("/v1/training/session",
                               {"op": "open", "argv": [sys.executable, "-c", "print('ERR no model')"]})
        self.assertEqual(code, 502, body)
        self.assertIn("did not start", body["error"]["message"])
        code, body = self.get("/v1/training/status")
        self.assertFalse(body["active"])

    def test_unload_before_open_and_its_restoration(self):
        self.post("/load", {})                            # engine resident: unload() says "unloaded"
        code, body = self.post("/v1/training/session", {"op": "open", "argv": self.argv})
        self.assertEqual(code, 200, body)
        self.addCleanup(self.post, "/v1/training/session", {"op": "close"})
        self.assertFalse(self.engine.loaded)              # the serving engine stood down
        self.post("/v1/training/session", {"op": "close"})
        code, body = self.post("/load", {})
        self.assertEqual((code, body["status"]), (200, "loaded"))
        self.assertTrue(self.engine.loaded)


if __name__ == "__main__":
    unittest.main()
