"""Regression tests for preserving assistant reasoning across multi-turn history."""
import unittest
from pathlib import Path

from serve.frontend import (ChatTemplate, _normalize_preserve_reasoning, anthropic_to_messages,
                             openai_to_messages, preserve_reasoning_kwargs)
from serve.responses import template_kwargs


ROOT = Path(__file__).resolve().parents[1]


class PreserveReasoning(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(ROOT / "serve/chat_template.jinja")

    def test_empty_reasoning_is_not_serialized_as_an_empty_thinking_turn(self):
        prompt = self.render(self.empty_history())
        self.assertNotIn("<think>\n\n</think>", prompt)
        self.assertIn("<|im_start|>assistant\nfirst answer", prompt)

    def empty_history(self):
        return [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer", "reasoning_content": ""},
            {"role": "user", "content": "second"},
        ]

    def render(self, history):
        return self.template.render(history, preserve_thinking=True, preserve_reasoning=True)


class PreserveReasoningKwargs(unittest.TestCase):
    """`preserve_reasoning` (llama.cpp's name), `preserve_thinking` (Qwen's) and the inverse `clear_thinking`
    reach the template through every request dialect."""

    def test_aliases_and_text_forms(self):
        for req, expected in (
            ({"chat_template_kwargs": {"preserve_reasoning": False}}, False),
            ({"chat_template_kwargs": {"preserve_thinking": "false"}}, False),
            ({"chat_template_kwargs": {"preserve_thinking": 1}}, True),
            ({"clear_thinking": True}, False),
            ({"clear_thinking": "false"}, True),
            ({"preserve_thinking": False}, False),                      # request level, not just the kwargs object
        ):
            with self.subTest(req=req):
                kw = preserve_reasoning_kwargs(req)
                self.assertIs(kw["preserve_reasoning"], expected)
                self.assertIs(kw["preserve_thinking"], expected)

    def test_explicit_template_wins_and_both_names_stay_in_step(self):
        kw = _normalize_preserve_reasoning({"preserve_reasoning": True, "preserve_thinking": False})
        self.assertIs(kw["preserve_thinking"], False)
        self.assertIs(kw["preserve_reasoning"], False)

    def test_the_openai_converter_passes_the_switch_and_enable_thinking_coexists(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
               "chat_template_kwargs": {"preserve_reasoning": False, "enable_thinking": False}}
        kwargs = openai_to_messages(req)[2]
        self.assertIs(kwargs["preserve_thinking"], False)
        self.assertIs(kwargs["enable_thinking"], False)     # enable_thinking used to reset the dict and drop the switch

    def test_the_anthropic_converter_passes_the_switch(self):
        req = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
               "chat_template_kwargs": {"preserve_thinking": False}}
        kwargs = anthropic_to_messages(req, False)[2]
        self.assertIs(kwargs["preserve_reasoning"], False)

    def test_responses_merges_shared_defaults_with_the_request_winning(self):
        kw = template_kwargs({}, {"preserve_reasoning": True})
        self.assertIs(kw["preserve_thinking"], True)
        kw = template_kwargs({"clear_thinking": True}, {"preserve_reasoning": True})
        self.assertIs(kw["preserve_reasoning"], False)


if __name__ == "__main__":
    unittest.main()
