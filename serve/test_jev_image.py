"""serve/test_jev_image.py - M3.8: image two-call decision (mock engine + mock vision).

    python -m unittest serve.test_jev_image -v

Pins the donor contract (server.cpp:2419-2603, PLAN B V1 two-call shape):
- image_b64 present -> vision stage (summary) + decision stage, ONE wire request.
- Response: answers + image_summary + stages_used ["vision", "decision"].
- Fail-closed: empty summary -> 502, decision NOT run.
- vision_prompt override; no vision encoder -> 400; short b64 -> 422.
"""
from __future__ import annotations

import base64
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


class FakeVision:
    """Stands in for Vision: encode returns (path, n_pads); the path must exist
    (prepare() stats it for the combined-embeddings size)."""

    def __init__(self):
        self.encoded = []
        self.dir = Path("/tmp/test-jev-image-vision")

    def encode(self, source, keep=()):
        import tempfile
        self.encoded.append(source[:40])
        f = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
        f.write(b"0" * 64)
        f.close()
        return Path(f.name), 4


class SummaryEngine(DecisionEngine):
    """First generation (the vision prompt with pads) yields a scripted summary; then
    the decision rows answer as DecisionEngine.ROW."""

    SUMMARY = "A clearing with a small pond, two sheep and a red barn."

    def __init__(self, tokenizer, script):
        super().__init__(tokenizer, script)
        self.turns_seen = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.turns_seen.append(list(ids))
        if len(self.turns_seen) == 1:                     # the vision summary turn
            for t in self.tok.encode(self.SUMMARY, parse_special=False):
                yield t
            return
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def start():
    tok = ByteTokenizer()
    engine = SummaryEngine(tok, "A")
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.vision = FakeVision()
    httpd = serve(svc, port=0)

    def _post(body, headers=None):
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone",
                                     data=json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    return engine, svc, httpd, staticmethod(_post)


REQ = {"image_b64": base64.b64encode(b"x" * 128).decode(),
       "questions": {"q1": {"type": "noul", "instructions": "Is there water?"}}}


class ImageTwoCall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, cls.svc, cls.httpd, cls.post = start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # one shared engine across the class: each test starts its turn count fresh
        self.engine.turns_seen = []
        self.engine.prompts = []
        self.engine.SUMMARY = "A clearing with a small pond, two sheep and a red barn."

    def test_two_stages_one_request(self):
        st, body = self.post(REQ)
        self.assertEqual(st, 200)
        self.assertEqual(body["stages_used"], ["vision", "decision"])
        self.assertIn("pond", body["image_summary"])
        self.assertIn("q1", body["answers"])
        self.assertEqual(body["answers"]["q1"]["type"], "noul")
        # the vision encoder saw the image
        self.assertEqual(len(self.svc.vision.encoded), 1)
        # stage 2's state is the summary
        decision_prompt = self.engine.tok.decode(self.engine.turns_seen[1])
        self.assertIn("pond", decision_prompt)
        self.assertIn("Is there water?", decision_prompt)

    def test_vision_prompt_override(self):
        st, body = self.post({**REQ, "vision_prompt": "Inventory the animals:"})
        self.assertEqual(st, 200)
        # override only shapes the real vision prompt; the mock ignores it, but the
        # field must be accepted (not a 422/400)
        self.assertIn("stages_used", body)

    def test_short_b64_422(self):
        st, body = self.post({"image_b64": "tooshort", "questions": REQ["questions"]})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "image_b64")

    def test_no_vision_encoder_400(self):
        self.svc.vision = None
        try:
            st, body = self.post(REQ)
            self.assertEqual(st, 400)
            self.assertIn("vision", body["error"]["message"])
        finally:
            self.svc.vision = FakeVision()

    def test_empty_summary_fails_closed(self):
        self.engine.SUMMARY = ""
        try:
            st, body = self.post(REQ)
            self.assertEqual(st, 502)
            self.assertIn("empty summary", body["error"]["message"])
            self.assertNotIn("answers", body)
        finally:
            self.engine.SUMMARY = "A clearing with a small pond, two sheep and a red barn."

    def test_questions_validation_first(self):
        st, body = self.post({"image_b64": REQ["image_b64"],
                              "questions": {"q1": {"type": "essay", "instructions": "x"}}})
        self.assertEqual(st, 422)
        self.assertIn("type", body["error"]["field"])

    def test_direct_mode_raw_image_tokens(self):
        st, body = self.post({**REQ, "image_mode": "direct"})
        self.assertEqual(st, 200)
        self.assertEqual(body["stages_used"], ["image-direct"])
        self.assertNotIn("image_summary", body)             # no summary stage ran
        self.assertIn("q1", body["answers"])
        # ONE engine turn: image + question in the same user turn, question AFTER
        # the image pad (the SummaryEngine records turns_seen, and its first-turn
        # branch is the summary script - here that IS the readout turn)
        self.assertEqual(len(self.engine.turns_seen), 1)
        prompt = self.engine.tok.decode(self.engine.turns_seen[0])
        self.assertIn("Is there water?", prompt)
        self.assertIn("Answer:", prompt)

    def test_bad_image_mode_422(self):
        st, body = self.post({**REQ, "image_mode": "pixel-magic"})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "image_mode")


if __name__ == "__main__":
    unittest.main()


class DirectVMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, cls.svc, cls.httpd, cls.post = start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_direct_rejects_v_header(self):
        st, body = self.post(REQ, headers={"X-Jev-Mode": "V"})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"]["field"], "X-Jev-Mode")
        self.assertIn("not supported", body["error"]["message"])
