"""Inkling-2-Mini's thinking is returned, in the field each API has for it, and the request's level reaches the template (the model's own numbers).

  chat completions   message.reasoning_content (stream: delta.reasoning_content); usage.completion_tokens_details.reasoning_tokens
  Anthropic messages a content block {"type": "thinking", "thinking": text, "signature": ""} before the text (stream: thinking_delta)
  Responses          an output item {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}
                     (stream: response.reasoning_text.delta); usage.output_tokens_details.reasoning_tokens

Levels: reasoning_effort / output_config.effort / reasoning.effort, or a number in (0, 0.99]; none, 0, enable_thinking false, thinking disabled = level 0.
The mock engine records each request's prompt, so the effort line the template wrote can be read back.

    python -m pytest serve/test_inkling_reasoning_fields.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve.test_inkling_serve_e2e import _MockServerHarness, _post_json, _post_sse  # noqa: E402

SCRIPT = ("<|message_model|><|content_thinking|>Two plus two is four.<|end_message|>"
          "<|message_model|><|content_text|>The answer is 4.<|end_message|><|content_model_end_sampling|>")
MODEL = "inkling-2-mini-sq-20"


def effort_line(h, n):
    ids = [int(x) for x in Path(h.record_path).read_text(encoding="utf-8").splitlines()[n].split()[-1].split(",")]
    text = h.tok.decode(ids)
    return float(text.split("Thinking effort level: ")[1].split("<|end_message|>")[0])      # (the mock template prints 0.0 where the real one prints 0)


class ReasoningFields(unittest.TestCase):
    def run_with(self, fn, replies=1):
        with tempfile.TemporaryDirectory() as td:
            with _MockServerHarness(Path(td), [SCRIPT] * replies, parallel=1) as h:
                return fn(h, f"http://127.0.0.1:{h.port}")

    def test_chat_completions_returns_reasoning_content(self):
        def go(h, base):
            r = _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "2+2?"}], "reasoning_effort": "medium"})
            msg = r["choices"][0]["message"]
            self.assertEqual(msg["reasoning_content"], "Two plus two is four.")
            self.assertEqual(msg["content"], "The answer is 4.")
            self.assertGreater(r["usage"]["completion_tokens_details"]["reasoning_tokens"], 0)
            self.assertEqual(effort_line(h, 0), 0.7)
            events = _post_sse(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "2+2?"}], "reasoning_effort": "high"})
            deltas = [c["delta"] for e in events for c in (e.get("choices") or [])]
            self.assertEqual("".join(d.get("reasoning_content") or "" for d in deltas), "Two plus two is four.")
            self.assertEqual("".join(d.get("content") or "" for d in deltas), "The answer is 4.")
            first_reasoning = next(i for i, d in enumerate(deltas) if d.get("reasoning_content"))
            first_content = next(i for i, d in enumerate(deltas) if d.get("content"))
            self.assertLess(first_reasoning, first_content)                      # the thinking streams first
        self.run_with(go, replies=2)

    def test_anthropic_messages_returns_a_thinking_block(self):
        def go(h, base):
            r = _post_json(f"{base}/v1/messages", {"model": MODEL, "max_tokens": 200, "messages": [{"role": "user", "content": "2+2?"}],
                                                   "thinking": {"type": "enabled", "budget_tokens": 5000}})
            blocks = r["content"]
            self.assertEqual([b["type"] for b in blocks], ["thinking", "text"])
            self.assertEqual(blocks[0]["thinking"], "Two plus two is four.")
            self.assertIn("signature", blocks[0])
            self.assertEqual(blocks[1]["text"], "The answer is 4.")
            self.assertEqual(effort_line(h, 0), 0.7)                           # budget_tokens 2K-8K -> medium
        self.run_with(go)

    def test_responses_returns_a_reasoning_item(self):
        def go(h, base):
            r = _post_json(f"{base}/v1/responses", {"model": MODEL, "input": "2+2?", "reasoning": {"effort": "low"}})
            kinds = [it["type"] for it in r["output"]]
            self.assertEqual(kinds[0], "reasoning")
            item = r["output"][0]
            self.assertEqual(item["content"], [{"type": "reasoning_text", "text": "Two plus two is four."}])
            self.assertEqual(item["summary"], [])
            msg = next(it for it in r["output"] if it["type"] == "message")
            self.assertEqual(msg["content"][0]["text"], "The answer is 4.")
            self.assertGreater(r["usage"]["output_tokens_details"]["reasoning_tokens"], 0)
            self.assertEqual(effort_line(h, 0), 0.2)
        self.run_with(go)

    def test_every_way_to_say_level_zero(self):
        asks = [{"reasoning_effort": "none"}, {"reasoning_effort": 0}, {"reasoning_effort": "off"},   # (a top-level enable_thinking is not read, as for every family: chat_template_kwargs is the place)
                {"chat_template_kwargs": {"enable_thinking": False}}, {"chat_template_kwargs": {"reasoning_effort": "none"}}]

        def go(h, base):
            for i, extra in enumerate(asks):
                _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra})
                self.assertEqual(effort_line(h, i), 0.0, extra)
        self.run_with(go, replies=len(asks))

    def test_the_words_and_numbers_reach_the_template(self):
        cases = [("minimal", 0.1), ("low", 0.2), ("medium", 0.7), ("high", 0.9), ("max", 0.99)]   # (numbers and xhigh: the real template, test_inkling_prefix_chain)

        def go(h, base):
            for i, (ask, line) in enumerate(cases):
                _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": ask})
                self.assertEqual(effort_line(h, i), line, ask)
        self.run_with(go, replies=len(cases))

    def test_a_number_is_accepted(self):
        def go(h, base):
            for ask in (0.5, 0.99, "0.3", 0, 1):
                r = _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": ask})
                self.assertEqual(r["choices"][0]["finish_reason"], "stop", ask)
        self.run_with(go, replies=5)

    def test_no_effort_means_the_default_level_and_a_bad_one_is_a_400(self):
        def go(h, base):
            _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(effort_line(h, 0), 0.2)                           # the default: low
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                _post_json(f"{base}/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "extreme"})
            self.assertEqual(ctx.exception.code, 400)
        self.run_with(go)

    def test_anthropic_thinking_disabled_is_level_zero(self):
        def go(h, base):
            _post_json(f"{base}/v1/messages", {"model": MODEL, "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}})
            self.assertEqual(effort_line(h, 0), 0.0)
        self.run_with(go)


if __name__ == "__main__":
    unittest.main()
