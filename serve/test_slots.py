"""serve/test_slots.py - llama-server compatible /slots/0?action=save|restore over the engine's SAVE/RESTORE lines
(mock engine: no GPU, no pack).

    python -m unittest serve.test_slots -v
"""
from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeProc:
    """Stands in for the engine process: answers SAVE / RESTORE lines like `strata --serve` does."""

    def __init__(self, engine):
        self.engine = engine
        self.stdin = self
        self.sent: list[str] = []

    def write(self, s):
        self.sent.append(s)
        cmd, _, path = s.strip().partition(" ")
        if cmd == "SAVE":
            if self.engine.fail:
                self.engine.lines.put("ERR no complete session to save\n")
            else:
                Path(path).write_bytes(b"x" * 1234)
                self.engine.lines.put("SAVED 62993 1234 401.5\n")
        elif cmd == "RESTORE":
            if not Path(path).exists() or self.engine.fail:
                self.engine.lines.put("ERR session file: saved with another model (model fingerprint differs)\n")
            else:
                self.engine.lines.put("RESTORED 62993 1234 560.2\n")
        elif cmd == "DIE":
            self.engine.lines.put(None)

    def flush(self):
        pass

    def poll(self):
        return None


class FakeEngine(StrataEngine):
    def __init__(self):
        super().__init__("strata", ["--max-context", "65536"], lazy=True)
        self.max_context = 65536
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.fail = False
        self.proc = FakeProc(self)


class Slots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = FakeEngine()
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def setUp(self):
        self.engine.fail = False
        self.engine.proc.sent.clear()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_foreign_page_or_a_form_never_reaches_the_engine(self):
        for action in ("save", "restore"):
            for headers, code in (({"Content-Type": "application/json", "Origin": "http://evil.example.com"}, 403),
                                  ({"Content-Type": "text/plain"}, 415),
                                  ({"Content-Type": "application/x-www-form-urlencoded"}, 415)):
                s, b = self.post(f"/slots/0?action={action}", {"filename": "x.bin"}, headers)
                self.assertEqual(s, code, (action, headers, b))
        self.assertEqual(self.engine.proc.sent, [])
        self.assertFalse(Path(self.dir.name, "x.bin").exists())

    def test_save_writes_into_the_slot_save_path(self):
        s, b = self.post("/slots/0?action=save", {"filename": "a.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.proc.sent, [f"SAVE {Path(self.dir.name, 'a.bin')}\n"])
        self.assertEqual((b["id_slot"], b["filename"], b["n_saved"], b["n_written"]), (0, "a.bin", 62993, 1234))
        self.assertAlmostEqual(b["timings"]["save_ms"], 401.5)

    def test_restore_answers_like_llama_server(self):
        self.post("/slots/0?action=save", {"filename": "b.bin"})
        s, b = self.post("/slots/0?action=restore", {"filename": "b.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual((b["id_slot"], b["filename"], b["n_restored"], b["n_read"]), (0, "b.bin", 62993, 1234))
        self.assertAlmostEqual(b["timings"]["restore_ms"], 560.2)

    def test_refused_file_is_an_error_not_a_success(self):
        self.post("/slots/0?action=save", {"filename": "c.bin"})
        self.engine.fail = True
        s, b = self.post("/slots/0?action=restore", {"filename": "c.bin"})
        self.assertEqual(s, 400, b)
        self.assertIn("fingerprint", b["error"]["message"])

    def test_missing_file_is_not_found(self):
        s, b = self.post("/slots/0?action=restore", {"filename": "never-saved.bin"})
        self.assertEqual(s, 404, b)
        self.assertEqual(self.engine.proc.sent, [])

    def test_filename_cannot_leave_the_directory(self):
        for bad in ("../x.bin", "/etc/passwd", "a/b.bin", "", ".", "..", "x\ny.bin"):
            s, b = self.post("/slots/0?action=save", {"filename": bad})
            self.assertEqual(s, 400, (bad, b))
        self.assertEqual(self.engine.proc.sent, [])

    def test_only_slot_zero_and_known_actions(self):
        self.assertEqual(self.post("/slots/1?action=save", {"filename": "d.bin"})[0], 400)
        self.assertEqual(self.post("/slots/0?action=erase_all", {"filename": "d.bin"})[0], 400)
        self.assertEqual(self.engine.proc.sent, [])

    def test_disabled_without_a_slot_save_path(self):
        saved = self.svc.slot_save_path
        self.svc.slot_save_path = None
        try:
            s, b = self.post("/slots/0?action=save", {"filename": "e.bin"})
        finally:
            self.svc.slot_save_path = saved
        self.assertEqual(s, 501, b)

    def test_waits_for_a_running_request(self):
        """A save never interleaves with a generation: it takes the same FIFO."""
        self.svc.fifo.acquire()
        out = {}
        t = threading.Thread(target=lambda: out.update(r=self.post("/slots/0?action=save", {"filename": "f.bin"})))
        t.start()
        t.join(0.5)
        self.assertTrue(t.is_alive())
        self.assertEqual(self.engine.proc.sent, [])
        self.svc.fifo.release()
        t.join(10)
        self.assertEqual(out["r"][0], 200)


if __name__ == "__main__":
    unittest.main()
