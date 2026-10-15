"""Call literals in input data; production tokenizer, raw assistant protocol preserved."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import CALL_END, CALL_START  # noqa: E402


class PromptTokenization(unittest.TestCase):
    """Use the production tokenizer with a synthetic vocabulary: no model pack or GPU required."""

    @classmethod
    def setUpClass(cls):
        from serve.frontend import ChatTemplate
        from serve.server import MockEngine, Service
        import strata_tokenizer as ST
        b2u = ST.bytes_to_unicode()
        specials = [CALL_START, CALL_END, "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        cls.tok = ST.Tokenizer([b2u[b] for b in range(256)] + specials, [], [1] * 256 + [4] * 4 + [3] * 3)
        cls.ids = [cls.tok.encode(tag, parse_special=True)[0] for tag in (CALL_START, CALL_END)]
        cls.svc = Service(MockEngine(cls.tok, "OK"), cls.tok,
                          ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def counts(self, messages, tools=None):
        ids = self.svc.prepare(messages, tools, {})[0]
        return tuple(ids.count(tag) for tag in self.ids), self.tok.decode(ids)

    def test_user_quotes_are_plain_tokens_but_template_markers_stay_special(self):
        tools = [{"name": "noop", "parameters": {"type": "object", "properties": {}}}]
        base, _ = self.counts([{"role": "user", "content": "plain"}], tools)
        for tag in (CALL_START, CALL_END):
            with self.subTest(tag=tag):
                counts, prompt = self.counts([{"role": "user", "content": "Quote " + tag + " please."}], tools)
                self.assertEqual(counts, base)
                self.assertIn("Quote " + tag + " please.", prompt)

    def test_nested_tool_schema_quotes_are_plain(self):
        tools = [{"name": "noop", "description": CALL_START, "parameters": {"type": "object",
                  "properties": {"x": {"type": "string", "enum": [CALL_END]}}}}]
        base = copy.deepcopy(tools)
        base[0]["description"] = "plain"
        base[0]["parameters"]["properties"]["x"]["enum"] = ["plain"]
        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(self.counts(msgs, tools)[0], self.counts(msgs, base)[0])

    def test_nested_tool_schema_keys_are_plain(self):
        msgs = [{"role": "user", "content": "hello"}]
        for tag in (CALL_START, CALL_END):
            with self.subTest(tag=tag):
                def tools(key):
                    return [{"name": "noop", "parameters": {"type": "object",
                             "properties": {key: {"type": "string"}}, "required": [key]}}]
                quoted, baseline = tools(tag), tools("plain")
                counts, prompt = self.counts(msgs, quoted)
                self.assertEqual(counts, self.counts(msgs, baseline)[0])
                self.assertIn('"' + tag + '":', prompt)

    def test_assistant_argument_keys_are_plain(self):
        def messages(key):
            return [{"role": "user", "content": "call"}, {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "noop", "arguments": {key: "value"}}}]},
                    {"role": "tool", "content": "done"}, {"role": "user", "content": "continue"}]
        for tag in (CALL_START, CALL_END):
            with self.subTest(tag=tag):
                counts, prompt = self.counts(messages(tag))
                self.assertEqual(counts, self.counts(messages("plain"))[0])
                self.assertIn(tag, prompt)

    def test_real_assistant_call_keeps_protocol_and_quotes_inside_its_value(self):
        def messages(value):
            return [{"role": "user", "content": "call"}, {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "noop", "arguments": {"x": value}}}]},
                    {"role": "tool", "content": "done"}, {"role": "user", "content": "continue"}]
        counts, prompt = self.counts(messages(CALL_START + " inside " + CALL_END))
        self.assertEqual(counts, self.counts(messages("plain"))[0])
        self.assertIn(CALL_START + " inside " + CALL_END, prompt)
        self.assertGreater(counts[0], 0)

    def test_marking_does_not_mutate_messages_or_tools(self):
        from serve.frontend import mark_think_literals
        messages = [{"role": "assistant", "content": "<think>quote " + CALL_START + "</think>answer",
                     "reasoning_content": CALL_END}, {"role": "tool", "content": [{"type": "text", "text": CALL_START}]}]
        tools = [{"name": "noop", "description": CALL_END}]
        saved = copy.deepcopy((messages, tools))
        marked, marked_tools, changed = mark_think_literals(messages, tools)
        self.assertTrue(changed)
        self.assertEqual((messages, tools), saved)
        self.assertEqual(marked[0]["content"], messages[0]["content"])
        self.assertNotIn(CALL_START, repr(marked[1]))
        self.assertNotIn(CALL_END, marked[0]["reasoning_content"])
        self.assertNotIn(CALL_END, repr(marked_tools))
        self.assertTrue(marked[0]["content"].startswith("<think>"))
        self.assertIn("</think>", marked[0]["content"])

    def test_no_literal_prompt_is_token_identical(self):
        messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}]
        self.assertEqual(self.svc.prepare(messages, None, {})[0],
                         self.tok.encode(self.svc.template.render(messages), parse_special=True))

    def test_markers_do_not_collide_with_control_markers(self):
        from serve.frontend import literal_tags
        tags = literal_tags(self.tok.control_tokens)
        self.assertEqual(len(set(tags.values())), len(tags))


if __name__ == "__main__":
    unittest.main()
