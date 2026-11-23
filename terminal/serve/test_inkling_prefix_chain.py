"""#6733 / #6781 / #6789: with Inkling-2-Mini's REAL chat template (terminal/tests/data/inkling/chat_template.jinja, the model pack's own), the prompt
of one request of a conversation must be the start of the next request's prompt, with and without tools, for the shapes agent clients send
(a tool call, parallel calls, a call with text, a tool result, a user follow-up, a system message).  The engine can only reuse what the new prompt
shares with the old one from the first token.

    python -m pytest serve/test_inkling_prefix_chain.py
"""
from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import frontend, inkling_dialect, server  # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "inkling" / "chat_template.jinja"
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a command.",
                                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
]


def call(i, name, **args):
    # what a client sends back: OpenAI form, the arguments a JSON string
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def conversation():
    return [
        {"role": "system", "content": "You are a coding agent. Use the tools."},
        {"role": "user", "content": "open setup.py"},
        {"role": "assistant", "content": None, "tool_calls": [call(1, "read_file", path="setup.py")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"},
        {"role": "assistant", "content": "It prints hi. Running it.", "tool_calls": [call(2, "run", cmd="python setup.py")]},
        {"role": "tool", "tool_call_id": "call_2", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [call(3, "read_file", path="a.py"), call(4, "read_file", path="b.py")]},
        {"role": "tool", "tool_call_id": "call_3", "content": "A"},
        {"role": "tool", "tool_call_id": "call_4", "content": "B"},
        {"role": "assistant", "content": "Both files are tiny."},
        {"role": "user", "content": "thanks, now explain"},
        {"role": "assistant", "content": "setup.py prints hi."},
        {"role": "user", "content": "ok"},
    ]


class TestRealTemplatePrefixChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpl = frontend.ChatTemplate(TEMPLATE)
        cls.svc = types.SimpleNamespace(hy4=False, inkling=True, template=cls.tpl, effort_end=False)

    def render(self, messages, tools, kwargs=None):
        # the path a request takes: Service.render_prompt for an Inkling engine
        return server.Service.render_prompt(self.svc, messages, tools, dict(kwargs or {}))

    def chain(self, conv, tools, kwargs=None):
        # a request ends at a user message or at the LAST result of a block of tool results (parallel calls come back together)
        ends = [i for i, m in enumerate(conv) if m["role"] in ("user", "tool")
                and not (m["role"] == "tool" and i + 1 < len(conv) and conv[i + 1]["role"] == "tool")]
        prompts = [self.render(conv[:i + 1], tools, kwargs) for i in ends]
        for (a, pa), (b, pb) in zip(zip(ends, prompts), zip(ends[1:], prompts[1:])):
            k = next((n for n, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
            self.assertTrue(pb.startswith(pa), f"the prompt after message {b} stops matching the one after {a} at char {k}: "
                                               f"{pa[max(0, k - 40):k + 40]!r} vs {pb[max(0, k - 40):k + 40]!r}")
        return prompts

    def test_with_tools_every_prompt_starts_with_the_one_before(self):
        self.chain(conversation(), TOOLS)

    def test_with_tools_and_the_clients_sampling_kwargs(self):
        # the desktop app and pi send reasoning_effort / chat_template_kwargs; none of them may move the history
        for kw in ({}, {"reasoning_effort": "high"}, {"enable_thinking": False}, {"reasoning_effort": "low", "enable_thinking": True}):
            self.chain(conversation(), TOOLS, kw)

    def test_plain_chat_text_turns_chain(self):
        conv = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}, {"role": "user", "content": "e"}]
        self.chain(conv, None)

    def test_without_tools_the_prompt_is_the_one_main_rendered(self):
        # the replay is for requests with tools only
        conv = conversation()[:2] + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
        old = inkling_dialect.finish_prompt(
            self.tpl.render(inkling_dialect.prepare_messages(conv), tools=None, **inkling_dialect.template_kwargs({})), has_tools=False)
        self.assertEqual(self.render(conv, None), old)

    def test_the_prompt_ends_where_the_engine_answers_and_has_no_marker(self):
        prompt = self.render(conversation(), TOOLS)
        self.assertTrue(prompt.endswith("<|message_model|><|content_thinking|><|end_message|>"))
        self.assertNotIn(inkling_dialect._EMPTY_THINK_MARK, prompt)
        plain = self.render(conversation()[:2], None)
        self.assertTrue(plain.endswith("<|message_model|><|content_text|>"))

    def test_a_short_helper_request_does_not_change_the_conversations_prompt(self):
        # LobeHub / Hermes send a title request between two turns: it is another prompt, and the conversation's next prompt must still
        # extend its own previous one (the engine's job is then to keep that conversation's cache; see test_inkling_slot_routing)
        conv = conversation()
        before = self.render(conv[:4], TOOLS)
        helper = self.render([{"role": "user", "content": "Name this chat in 3 words."}], None)
        self.assertNotEqual(helper[:60], before[:60])
        after = self.render(conv[:6], TOOLS)
        self.assertTrue(after.startswith(before))

    def test_a_client_that_sends_the_thinking_back_is_not_doubled(self):
        # the server hides the thinking, so a client has none to send; one that sends its own gets it rendered once, and every
        # later prompt keeps it
        conv = conversation()
        conv[2]["reasoning_content"] = "I should read it."
        a, b = self.render(conv[:4], TOOLS), self.render(conv[:6], TOOLS)
        self.assertTrue(b.startswith(a))
        self.assertEqual(a.count("<|content_thinking|>I should read it.<|end_message|>"), 1)


if __name__ == "__main__":
    unittest.main()
