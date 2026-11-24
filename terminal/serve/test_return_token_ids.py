"""Opt-in return_token_ids on /v1/chat/completions: python -m unittest serve.test_return_token_ids.

Covers:
  - Default path (omitted / false) leaves responses unchanged.
  - Non-boolean return_token_ids returns 400.
  - return_token_ids=True returns top-level prompt_token_ids and choices[0].token_ids.
  - Combined with logprobs=True, choices[0].logprobs.content aligns 1:1 with choices[0].token_ids
    across plain text, thinking, tool calls, and the trailing stop token.
  - Streaming carries prompt_token_ids, choices[0].token_ids, and aligned logprobs on the final chunk.
"""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import Service, serve
from serve.test_logprobs import LpEngine, ThinkTokenizer, Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ChatTemplate(ROOT / "serve" / "chat_template.jinja")


class RecordingLpEngine(LpEngine):
    def __init__(self, tok, script):
        super().__init__(tok, script)
        self.last_prompt_ids = None
        self.last_generated_ids = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt_ids = list(ids)
        self.last_generated_ids = []
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            self.last_generated_ids.append(t)
            yield t


class ReturnTokenIdsApi(unittest.TestCase):
    def _start(self, script, tok=None):
        tok = tok or Tokenizer()
        engine = RecordingLpEngine(tok, script)
        svc = Service(engine, tok, TEMPLATE)
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}", engine, tok

    def _post(self, base, body, stream=False):
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 200,
                   "chat_template_kwargs": {"enable_thinking": False}, **body}
        if stream:
            payload["stream"] = True
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        if not stream:
            return json.loads(raw)
        return [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]

    def test_default_omits_token_ids(self):
        base, _, _ = self._start("hello")
        for extra in ({}, {"return_token_ids": False}, {"return_token_ids": None}):
            out = self._post(base, {"logprobs": True, "top_logprobs": 0, **extra})
            self.assertNotIn("prompt_token_ids", out)
            self.assertNotIn("token_ids", out["choices"][0])
            # Default path still excludes the stop token from logprobs.content.
            self.assertEqual(len(out["choices"][0]["logprobs"]["content"]), len("hello"))

    def test_invalid_return_token_ids_returns_400(self):
        base, _, _ = self._start("hello")
        for bad in ("true", 1, [True], {"yes": True}):
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                            "return_token_ids": bad}).encode(),
                {"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(ctx.exception.code, 400, bad)

    def test_returns_exact_ids_without_logprobs(self):
        base, eng, _ = self._start("hello")
        out = self._post(base, {"return_token_ids": True})
        self.assertEqual(out["prompt_token_ids"], eng.last_prompt_ids)
        self.assertEqual(out["choices"][0]["token_ids"], eng.last_generated_ids)
        self.assertEqual(out["choices"][0]["message"]["content"], "hello")
        self.assertNotIn("logprobs", out["choices"][0])

    def test_text_turn_aligns_logprobs_1_to_1_including_stop(self):
        base, eng, _ = self._start("Café!")
        out = self._post(base, {"return_token_ids": True, "logprobs": True, "top_logprobs": 0})
        choice = out["choices"][0]
        self.assertEqual(out["prompt_token_ids"], eng.last_prompt_ids)
        self.assertEqual(choice["token_ids"], eng.last_generated_ids)
        lps = choice["logprobs"]["content"]
        self.assertEqual(len(lps), len(choice["token_ids"]))
        self.assertEqual(lps[0]["logprob"], -0.01)
        self.assertEqual(lps[-1]["logprob"], -(len(lps)) / 100)

    def test_tool_call_turn_aligns_logprobs_1_to_1(self):
        tool_reply = '<tool_call>\n<function=bash>\n<parameter=command>echo 42</parameter>\n</function>\n</tool_call>'
        tools = [{"type": "function", "function": {
            "name": "bash", "description": "run bash",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        }}]
        base, eng, _ = self._start(tool_reply)
        out = self._post(base, {"tools": tools, "return_token_ids": True, "logprobs": True, "top_logprobs": 0})
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(len(choice["message"]["tool_calls"]), 1)
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(out["prompt_token_ids"], eng.last_prompt_ids)
        self.assertEqual(choice["token_ids"], eng.last_generated_ids)
        lps = choice["logprobs"]["content"]
        self.assertEqual(len(lps), len(choice["token_ids"]))
        self.assertGreater(len(lps), 0)

    def test_reasoning_and_content_turn_aligns_logprobs_1_to_1(self):
        tok = ThinkTokenizer()
        base, eng, _ = self._start("step 1</think>answer", tok=tok)
        out = self._post(base, {
            "chat_template_kwargs": {"enable_thinking": True},
            "return_token_ids": True, "logprobs": True, "top_logprobs": 0,
        })
        choice = out["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "step 1")
        self.assertEqual(choice["message"]["content"], "answer")
        self.assertEqual(choice["token_ids"], eng.last_generated_ids)
        lps = choice["logprobs"]["content"]
        self.assertEqual(len(lps), len(choice["token_ids"]))
        self.assertNotIn("reasoning_content", choice["logprobs"])

    def test_streaming_includes_ids_and_aligned_logprobs_on_final_chunk(self):
        base, eng, _ = self._start("ok")
        chunks = self._post(base, {"return_token_ids": True, "logprobs": True, "top_logprobs": 0}, stream=True)
        final = chunks[-1]
        self.assertEqual(final["prompt_token_ids"], eng.last_prompt_ids)
        self.assertEqual(final["choices"][0]["token_ids"], eng.last_generated_ids)
        self.assertEqual(len(final["choices"][0]["logprobs"]["content"]), len(eng.last_generated_ids))


if __name__ == "__main__":
    unittest.main()
