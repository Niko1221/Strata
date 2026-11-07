"""serve/test_session_persist.py - the conversation cache's L2 (session_autosave): a finished turn is saved
under its prompt's digest, a prompt that begins with a saved conversation restores it (unless the engine
already holds it), an unrelated one restores nothing; the index LRU-trims (mock engine: no GPU, no pack).

    python -m unittest serve.test_session_persist -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.session_persist import SnapshotIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "</think>\nok"   # the reply closes its thinking (as serve/test_server.py scripts it): the answer is "ok"


class PersistMockEngine(MockEngine):
    """A MockEngine with the session-file side: SAVE reports the live conversation (the prompt plus the mock's
    reply, as the engine's SAVE does), RESTORE brings a saved one back.  Both calls are recorded, in order."""

    def __init__(self, tokenizer, script: str, max_context: int = 65536):
        super().__init__(tokenizer, script, max_context=max_context)
        self.session_calls: list[str] = []
        self.saved: dict[str, int] = {}
        self.live_tokens = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.live_tokens = len(ids) + len(self.script)   # the prompt plus the reply the engine holds (set up
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)   # front: the server closes the
        # generator at its STOP+drain, so nothing may wait for the loop's end)

    def session_file(self, action: str, path: str) -> dict:
        self.session_calls.append(f"{'SAVE' if action == 'save' else 'RESTORE'} {path}")
        if action == "save":
            Path(path).write_bytes(b"x" * 1234)
            self.saved[path] = self.live_tokens
            return {"tokens": self.live_tokens, "bytes": 1234, "ms": 401.5}
        if path in self.saved and Path(path).exists():
            self.live_tokens = self.saved[path]   # the saved conversation is live again; the GEN that follows
            return {"tokens": self.live_tokens, "bytes": 1234, "ms": 560.2}    # reads only the tokens past it
        raise RuntimeError(f"no saved session named {path}")


class SessionAutosave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = PersistMockEngine(tok, SCRIPT)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.svc.session_autosave = True
        cls.svc.snapshots = SnapshotIndex(cls.dir.name)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def setUp(self):
        self.engine.session_calls.clear()

    def chat(self, messages, max_tokens=64):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"messages": messages, "max_tokens": max_tokens}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return json.loads(e.read())

    def index(self):
        return json.loads(Path(self.dir.name, "index.json").read_text())["snapshots"]

    def test_a_finished_turn_is_saved_under_its_prompts_digest(self):
        reply = self.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop", reply)
        saves = [c for c in self.engine.session_calls if c.startswith("SAVE ")]
        self.assertEqual(len(saves), 1, self.engine.session_calls)
        path = Path(saves[0][len("SAVE "):])
        self.assertTrue(path.is_file())
        self.assertTrue(path.name.startswith("auto-"), path.name)
        idx = self.index()
        self.assertEqual([s["file"] for s in idx], [path.name])
        self.assertGreater(idx[0]["len"], 0)

    def test_a_prompt_that_continues_the_live_conversation_restores_nothing(self):
        first = self.chat([{"role": "user", "content": "hello"}])
        self.engine.session_calls.clear()
        second = self.chat([{"role": "user", "content": "hello"},
                            {"role": "assistant", "content": first["choices"][0]["message"]["content"]},
                            {"role": "user", "content": "and more"}])
        self.assertEqual(second["choices"][0]["finish_reason"], "stop", second)
        self.assertFalse([c for c in self.engine.session_calls if c.startswith("RESTORE ")],
                         self.engine.session_calls)   # the engine already holds it: its own prefix match reads nothing

    def test_a_returning_prompt_restores_its_saved_conversation(self):
        first = self.chat([{"role": "user", "content": "hello"}])
        saved = next(c for c in self.engine.session_calls if c.startswith("SAVE "))[len("SAVE "):]
        answer = first["choices"][0]["message"]["content"]
        self.chat([{"role": "user", "content": "meanwhile, another conversation"}])   # it is the live one now
        self.engine.session_calls.clear()
        second = self.chat([{"role": "user", "content": "hello"},
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content": "and more"}])
        self.assertEqual(second["choices"][0]["finish_reason"], "stop", second)
        restores = [c for c in self.engine.session_calls if c.startswith("RESTORE ")]
        self.assertEqual(restores, [f"RESTORE {saved}"], self.engine.session_calls)   # it, and only it
        saves = [c for c in self.engine.session_calls if c.startswith("SAVE ")]
        self.assertEqual(len(saves), 1, self.engine.session_calls)       # the returning turn saved its own too

    def test_an_unrelated_prompt_restores_nothing(self):
        self.chat([{"role": "user", "content": "hello"}])
        self.engine.session_calls.clear()
        reply = self.chat([{"role": "user", "content": "something else entirely"}])
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop", reply)
        self.assertFalse([c for c in self.engine.session_calls if c.startswith("RESTORE ")],
                         self.engine.session_calls)

    def test_the_index_trims_to_its_cap_and_reloads(self):
        with tempfile.TemporaryDirectory() as d:
            idx = SnapshotIndex(d, max_bytes=1500)
            for i, name in enumerate(("a", "b")):
                p = Path(d, f"auto-{name}.bin")
                p.write_bytes(b"x" * 1234)
                idx.add(10 + i, name, p.name, 1234)
            idx.save()
            self.assertEqual([s["file"] for s in idx.snapshots], ["auto-b.bin"])   # 2 x 1234 > 1500: the oldest went
            self.assertFalse(Path(d, "auto-a.bin").exists())
            self.assertTrue(Path(d, "auto-b.bin").is_file())
            back = SnapshotIndex(d, max_bytes=1500)   # reloads without the dropped file
            self.assertEqual([s["file"] for s in back.snapshots], ["auto-b.bin"])


if __name__ == "__main__":
    unittest.main()
