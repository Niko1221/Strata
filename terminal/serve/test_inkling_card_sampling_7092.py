"""#7092: a client that sends no sampling settings gets Inkling-2-Mini's model-card sampling, not the engine's greedy picking (which can loop).

The card's own value is generation_config.json (huggingface.co/thinkingmachines/Inkling-2-Mini): do_sample true, temperature 1.0; the README names no other sampling value, so
the default is temperature 1.0 and nothing else (the engine keeps its top_k 64 / top_p 1.0).  Every layer above the card wins: the request, the Chat settings shared with apps, the run
config's `sampling` block; temperature 0 asked for is still greedy.  Other families keep their own cards.

    python -m pytest serve/test_inkling_card_sampling_7092.py
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import MODEL_CARD_SAMPLING, ByteTokenizer, MockEngine, Service, model_card_sampling, serve  # noqa: E402
from serve.test_inkling_reasoning_budget import TEMPLATE, InkTokenizer, ThinkingEngine  # noqa: E402


class Tables(unittest.TestCase):
    def test_the_card_value_for_every_inkling_arch_name(self):
        for arch in ("inkling2_loop", "inkling2", "inkling2_loop_text"):
            for thinking in (True, False):
                self.assertEqual(model_card_sampling(arch, thinking), {"temperature": 1.0}, (arch, thinking))

    def test_the_other_cards_are_unchanged(self):
        self.assertEqual(model_card_sampling("qwen4exp", True), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(model_card_sampling("hy4flash", True), {"temperature": 0.9, "top_p": 1.0})
        self.assertEqual(sorted(MODEL_CARD_SAMPLING), ["hy4flash", "inkling2", "inkling2_loop", "inkling2_loop_text", "qwen35moe", "qwen4exp"])


class Recording(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class OverHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = Recording(tok, "ok", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.model_arch = "inkling2_loop"
        self.svc.sampling_defaults = {}
        self.svc.shared = {}
        self.addCleanup(setattr, self.svc, "model_arch", None)

    def sent(self, path="/v1/chat/completions", **body):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **body}
        self.engine.sampling = {}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        return self.engine.sampling

    def test_nothing_named_gets_the_card(self):
        s = self.sent()
        self.assertEqual(s.get("temperature"), 1.0)
        self.assertNotIn("top_p", s)
        self.assertNotIn("top_k", s)

    def test_the_anthropic_route_too(self):
        s = self.sent("/v1/messages", max_tokens=8)
        self.assertEqual(s.get("temperature"), 1.0)

    def test_the_request_wins(self):
        self.assertEqual(self.sent(temperature=0.3).get("temperature"), 0.3)
        self.assertEqual(self.sent(temperature=0).get("temperature"), 0)              # greedy asked for stays greedy
        s = self.sent(top_p=0.9)
        self.assertEqual(s.get("top_p"), 0.9)
        self.assertEqual(s.get("temperature"), 1.0)                                   # the card still fills what the request left out

    def test_the_config_block_and_the_shared_settings_win_over_the_card(self):
        self.svc.sampling_defaults = {"temperature": 0.6, "top_k": 40}
        s = self.sent()
        self.assertEqual((s.get("temperature"), s.get("top_k")), (0.6, 40))
        self.svc.shared = {"temperature": 0.2}
        self.assertEqual(self.sent().get("temperature"), 0.2)
        self.svc.sampling_defaults = {"temperature": 0}
        self.svc.shared = {}
        self.assertEqual(self.sent().get("temperature"), 0)

    def test_another_family_is_not_touched(self):
        self.svc.model_arch = "qwen4exp"
        self.assertEqual(self.sent().get("top_k"), 20)


class OnTheInklingPath(unittest.TestCase):
    """Service.run with the scripted Inkling engine: what the engine is handed."""

    def test_the_engine_gets_temperature_one_from_a_request_that_names_none(self):
        tok = InkTokenizer()
        engine = ThinkingEngine(tok, "short thought")
        seen = []
        real = engine.generate

        def generate(ids, max_new, sampling, cancel, embeddings=None):
            seen.append(dict(sampling or {}))
            return real(ids, max_new, sampling, cancel, embeddings)

        engine.generate = generate
        svc = Service(engine, tok, ChatTemplate(TEMPLATE))
        svc.model_arch = "inkling2_loop"
        self.assertTrue(svc.inkling)
        ids = tok.encode("<|message_user|><|content_text|>2+2?<|end_message|><|message_model|><|content_thinking|><|end_message|>", parse_special=True)
        list(svc.run(ids, True, [], 2000, {}, threading.Event()))
        self.assertEqual(seen[0].get("temperature"), 1.0)
        seen.clear()
        list(svc.run(ids, True, [], 2000, {"temperature": 0.0}, threading.Event()))
        self.assertEqual(seen[0].get("temperature"), 0.0)


if __name__ == "__main__":
    unittest.main()
