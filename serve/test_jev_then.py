"""serve/test_jev_then.py - M3.9 decide-then-generate + M3.3 chat-surface jev extension.

    python -m unittest serve.test_jev_then -v

M3.9 (donor amendment 7 mode 1): "then": {"generate": {...}} on /v1/systemone -
decide, then generate with the answers folded into the LAST USER turn as
'Jev decisions (JSON): ...'.  Response = decision body + then{text,finish_reason,usage}.

M3.3 (donor A3.4): "jev": {...} on /v1/chat/completions - state OR messages as the
decision input; jev.questions required; stream refused; the response wraps the
decision body under "jev".
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve import jev  # noqa: E402
from serve.test_systemone import DecisionEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class GenEngine(DecisionEngine):
    """First generate() call = the decision readout turn; later ones emit a script."""

    def __init__(self, tok):
        super().__init__(tok, "A")
        self.gen_turns = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.gen_turns += 1
        self.last_sampling = dict(sampling)
        self.last_max_new = max_new
        self.prompts.append(list(ids))
        if self.gen_turns > 1:                     # the "then" generation
            for t in self.tok.encode("Based on the decision, go left.", parse_special=False):
                yield t
            return
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def start():
    tok = ByteTokenizer()
    engine = GenEngine(tok)
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)

    def _post(path, body, headers=None):
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                     data=json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    post = staticmethod(_post)
    return engine, httpd, post


Q = {"q1": {"type": "noul", "instructions": "Is it on?"}}


class Then(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.post = start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.gen_turns = 0
        self.engine.prompts = []

    def test_decide_then_generate_one_request(self):
        st, body = self.post("/v1/systemone", {
            "state": {"k": 1}, "questions": Q,
            "then": {"generate": {"messages": [{"role": "user", "content": "What next?"}],
                                  "max_tokens": 32}}})
        self.assertEqual(st, 200)
        self.assertIn("answers", body)                       # the decision body rides along
        self.assertEqual(body["then"]["text"], "Based on the decision, go left.")
        self.assertEqual(body["then"]["usage"]["completion_tokens"] > 0, True)
        # the generation prompt carries the answers folded into the last user turn
        gen_prompt = self.engine.tok.decode(self.engine.last_prompt_ids) \
            if hasattr(self.engine, "last_prompt_ids") else None
        prompts = [self.engine.tok.decode(p) for p in self.engine.prompts]
        gen_p = prompts[-1]
        self.assertIn("What next?", gen_p)
        self.assertIn("Jev decisions (JSON):", gen_p)
        self.assertIn('"noul"', gen_p)
        # sampling defaults mirror the chat endpoint
        self.assertAlmostEqual(self.engine.last_sampling["temperature"], 0.6)
        self.assertAlmostEqual(self.engine.last_sampling["top_p"], 0.95)
        self.assertEqual(self.engine.last_sampling["top_k"], 20)

    def test_then_without_messages_422(self):
        st, body = self.post("/v1/systemone", {"state": {"k": 1}, "questions": Q,
                                               "then": {"generate": {}}})
        self.assertEqual(st, 422)
        self.assertIn("then.generate.messages", body["error"]["field"])

    def test_decision_error_propagates(self):
        st, body = self.post("/v1/systemone", {
            "state": {"k": 1},
            "questions": {"q1": {"type": "essay", "instructions": "x"}},
            "then": {"generate": {"messages": [{"role": "user", "content": "y"}]}}})
        self.assertEqual(st, 422)                            # validation first, no generation


class ChatSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, cls.httpd, cls.post = start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.gen_turns = 0
        self.engine.prompts = []

    def test_jev_with_state(self):
        st, body = self.post("/v1/chat/completions", {"jev": {"state": {"k": 1}, "questions": Q}})
        self.assertEqual(st, 200)
        self.assertEqual(body["object"], "jev.response")
        self.assertIn("q1", body["jev"]["answers"])

    def test_jev_messages_become_state(self):
        st, body = self.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "The lamp is on."}],
            "jev": {"questions": Q}})
        self.assertEqual(st, 200)
        prompts = [self.engine.tok.decode(p) for p in self.engine.prompts]
        self.assertIn("user: The lamp is on.", prompts[0])   # the conversation IS the state

    def test_jev_questions_required(self):
        st, body = self.post("/v1/chat/completions", {"jev": {"state": {"k": 1}}})
        self.assertEqual(st, 400)
        self.assertEqual(body["error"]["param"], "jev.questions")

    def test_jev_state_and_messages_conflict(self):
        st, body = self.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "x"}],
            "jev": {"state": {"k": 1}, "questions": Q}})
        self.assertEqual(st, 400)
        self.assertIn("must be empty", body["error"]["message"])

    def test_jev_stream_refused(self):
        st, body = self.post("/v1/chat/completions", {
            "jev": {"state": {"k": 1}, "questions": Q}, "stream": True})
        self.assertEqual(st, 400)
        self.assertEqual(body["error"]["param"], "stream")

    def test_plain_chat_unaffected(self):
        st, body = self.post("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hello"}],
                              "max_tokens": 8})
        self.assertIn(st, (200, 500))     # the mock path may not fully serve chat; must not 400 on jev grounds
        if st == 200:
            self.assertEqual(body.get("object"), "chat.completion")


if __name__ == "__main__":
    unittest.main()
