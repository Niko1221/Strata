"""serve/test_n_predict.py - llama.cpp's n_predict on /v1/chat/completions (#2980): 0 reads the prompt and returns
an empty completion, N > 0 is the answer limit, -1 leaves the rest of the context, and an OpenAI max_tokens wins.
No GPU, no pack: the mock engine and the byte tokenizer are enough.

    python -m unittest serve.test_n_predict -v
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
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "alpha beta gamma delta epsilon"


class NPredict(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, SCRIPT)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, stream=False):
        payload = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": stream, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    @staticmethod
    def chunks(text):
        return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]

    def test_zero_reads_the_prompt_and_generates_nothing(self):
        code, text = self.post({"n_predict": 0})
        self.assertEqual(code, 200, text)
        body = json.loads(text)
        choice = body["choices"][0]
        self.assertEqual((choice["message"]["content"], choice["finish_reason"]), ("", "length"))
        self.assertEqual(body["usage"]["completion_tokens"], 0)
        self.assertTrue(self.engine.last_prompt)                    # the prompt reached the engine
        self.assertEqual(body["usage"]["prompt_tokens"], len(self.engine.last_prompt))
        self.assertEqual(body["usage"]["total_tokens"], body["usage"]["prompt_tokens"])

    def test_zero_streams_an_empty_completion(self):
        code, text = self.post({"n_predict": 0}, stream=True)
        self.assertEqual(code, 200, text)
        cs = self.chunks(text)
        self.assertEqual(cs[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(cs[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(cs[-1]["usage"]["completion_tokens"], 0)
        self.assertFalse([c["choices"][0]["delta"].get("content") for c in cs
                          if c["choices"][0]["delta"].get("content")])
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        self.assertTrue(self.engine.last_prompt)

    def test_positive_n_predict_is_the_limit(self):
        code, text = self.post({"n_predict": 3})
        self.assertEqual(code, 200, text)
        body = json.loads(text)
        self.assertEqual(body["usage"]["completion_tokens"], 3)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_max_tokens_wins_over_n_predict(self):
        code, text = self.post({"max_tokens": 2, "n_predict": 7})
        self.assertEqual(code, 200, text)
        self.assertEqual(json.loads(text)["usage"]["completion_tokens"], 2)

    def test_negative_n_predict_leaves_the_rest_of_the_context(self):
        code, text = self.post({"n_predict": -1})
        self.assertEqual(code, 200, text)
        body = json.loads(text)
        self.assertGreater(body["usage"]["completion_tokens"], 3)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_bad_n_predict_is_400(self):
        for bad in ("0", True, 1.5, [0]):
            with self.subTest(n_predict=bad):
                self.assertEqual(self.post({"n_predict": bad})[0], 400)
        self.assertEqual(self.engine.last_prompt, [])                # nothing reached the engine


if __name__ == "__main__":
    unittest.main()
