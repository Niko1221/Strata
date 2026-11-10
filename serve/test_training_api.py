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
from serve.server import ByteTokenizer, Service, serve
# The lifecycle suite already owns this stub, and its generate() lets the ordinary inference API run
# against it.  Reusing it keeps one definition and lets us assert that inference survives a session.
from serve.test_lifecycle import ResidentEngine


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

        # the gate: inference and engine control fast-fail while the session owns the engine.
        # /v1/engine/* is the RESIDENT serving engine's RPC family (EngineHttp). It must stay
        # out of /v1/training/*, which is this gate's allowlist prefix (server.py:8751) - the
        # serving engine is unloaded during a session, so these two must be refused here.
        for path, payload in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                              ("/unload", {}), ("/load", {}), ("/v1/vram", {}),
                              ("/v1/engine/score", {"tokens": [1, 2]}),
                              ("/v1/engine/lora", {"path": "/tmp/x.gguf"})):
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

    def test_inference_api_works_before_and_after_a_session(self):
        """The ordinary OpenAI/Anthropic paths are untouched outside a session (CONTRIBUTING.md: preserve
        existing behavior).  A session refuses them with 409; after it closes and the serving engine is
        back, the same requests must return the same model answer - not just a 200 from /load."""
        chat = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hello"}]}
        paths = (("/v1/chat/completions", chat), ("/v1/messages", chat))

        def reply_text(body):
            """The mock's scripted reply is a thinking turn: OpenAI puts it in reasoning_content,
            Anthropic in a thinking block.  Compare the model's words, not the per-request envelope."""
            if "choices" in body:
                msg = body["choices"][0]["message"]
                return (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            return "".join(p.get("text") or p.get("thinking") or "" for p in body.get("content", []))

        self.assertEqual(self.post("/load", {})[0], 200)
        for path, payload in paths:
            code, body = self.post(path, payload)
            self.assertEqual(code, 200, (path, body))
            self.assertEqual(reply_text(body), "Hello.", (path, body))

        code, body = self.post("/v1/training/session", {"op": "open", "argv": self.argv})
        self.assertEqual(code, 200, body)
        self.addCleanup(self.post, "/v1/training/session", {"op": "close"})
        for path, payload in paths:
            code, body = self.post(path, payload)
            self.assertEqual(code, 409, (path, body))
            self.assertEqual(body["error"]["type"], "training_session_active")

        self.assertEqual(self.post("/v1/training/session", {"op": "close"})[0], 200)
        self.assertEqual(self.post("/load", {})[0], 200)     # the serving engine starts again
        for path, payload in paths:
            code, body = self.post(path, payload)
            self.assertEqual(code, 200, (path, body))
            self.assertEqual(reply_text(body), "Hello.", (path, body))

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
