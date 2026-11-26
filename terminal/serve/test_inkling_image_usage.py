"""Inkling-2-Mini picture requests: `usage` counts the picture's tokens.

The prompt carries ONE placeholder id per picture; the engine expands it to the picture's tokens (the mock engine: 256 per picture) and its DONE line
counts them.  The server told clients len(ids): the picture was left out of prompt_tokens (OpenAI, Anthropic, Responses).

    python -m pytest serve/test_inkling_image_usage.py
"""
from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve.test_inkling_serve_e2e import _MockServerHarness, _make_png_bytes, _post_json  # noqa: E402

MODEL = "inkling-2-mini-sq-20"
REPLY = "<|message_model|><|content_text|>I see a picture.<|end_message|><|content_model_end_sampling|>"
IMAGE_TOKENS = 256


def sent_ids(h, n=-1):
    return len(Path(h.record_path).read_text(encoding="utf-8").splitlines()[n].split()[-1].split(","))


class ImageUsage(unittest.TestCase):
    def run_with(self, fn):
        with tempfile.TemporaryDirectory() as td:
            with _MockServerHarness(Path(td), [REPLY] * 3, parallel=1) as h:
                return fn(h, f"http://127.0.0.1:{h.port}", base64.b64encode(_make_png_bytes()).decode("ascii"))

    def test_openai_prompt_tokens_include_the_picture(self):
        def go(h, base, b64):
            r = _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}, {"type": "text", "text": "What is this?"}]}]})
            self.assertEqual(r["usage"]["prompt_tokens"], sent_ids(h) + IMAGE_TOKENS - 1)
            self.assertEqual(r["usage"]["total_tokens"], r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"])
        self.run_with(go)

    def test_a_text_request_is_unchanged(self):
        def go(h, base, b64):
            r = _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]})
            self.assertEqual(r["usage"]["prompt_tokens"], sent_ids(h))
        self.run_with(go)

    def test_anthropic_input_tokens_include_the_picture(self):
        def go(h, base, b64):
            r = _post_json(f"{base}/v1/messages", {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}, {"type": "text", "text": "What is this?"}]}]})
            u = r["usage"]
            total = u["input_tokens"] + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
            self.assertEqual(total, sent_ids(h) + IMAGE_TOKENS - 1)
        self.run_with(go)

    def test_responses_input_tokens_include_the_picture(self):
        def go(h, base, b64):
            r = _post_json(f"{base}/v1/responses", {"model": MODEL, "max_output_tokens": 16, "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}, {"type": "input_text", "text": "What is this?"}]}]})
            self.assertEqual(r["usage"]["input_tokens"], sent_ids(h) + IMAGE_TOKENS - 1)
        self.run_with(go)


if __name__ == "__main__":
    unittest.main()
