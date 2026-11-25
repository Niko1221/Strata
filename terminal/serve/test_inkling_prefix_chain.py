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
_EMPTY_THINK = "<|message_model|><|content_thinking|><|end_message|>"
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


class _Base(unittest.TestCase):
    LEVEL = "none"      # the thinking level the requests of the class ask for

    @classmethod
    def setUpClass(cls):
        cls.tpl = frontend.ChatTemplate(TEMPLATE)
        cls.svc = types.SimpleNamespace(hy4=False, inkling=True, template=cls.tpl, effort_end=False)

    def render(self, messages, tools, kwargs=None):
        # the path a request takes: Service.render_prompt for an Inkling engine
        return server.Service.render_prompt(self.svc, messages, tools, {"reasoning_effort": self.LEVEL, **(kwargs or {})})

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


class TestRealTemplatePrefixChain(_Base):
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
            self.tpl.render(inkling_dialect.prepare_messages(conv), tools=None, **inkling_dialect.template_kwargs({"reasoning_effort": "none"})), has_tools=False)
        self.assertEqual(self.render(conv, None), old)

    def test_the_prompt_ends_where_the_engine_answers_and_has_no_marker(self):
        prompt = self.render(conversation(), TOOLS)
        self.assertTrue(prompt.endswith("<|message_model|><|content_thinking|><|end_message|>"))
        self.assertNotIn(inkling_dialect._EMPTY_THINK_MARK, prompt)
        plain = self.render(conversation()[:2], None)
        self.assertTrue(plain.endswith("<|message_model|><|content_text|>"))

    def test_a_forced_tool_call_turn_chains_onto_the_next_request(self):
        # tool_choice naming a tool: the prompt is the rendered conversation + the empty thinking turn + the forced opening (serve.prepare);
        # the next request's history renders that call in full and must start with it
        conv = conversation()[:2]
        first = self.render(conv, TOOLS) + inkling_dialect.force_text("<tool_call>\n<function=read_file>\n")
        self.assertTrue(first.endswith('<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":'))
        self.assertIn(_EMPTY_THINK, first)
        nxt = conv + [{"role": "assistant", "content": None, "tool_calls": [call(1, "read_file", path="setup.py")]},
                      {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"}]
        self.assertTrue(self.render(nxt, TOOLS).startswith(first))

    def test_numbers_and_words_render_the_models_own_effort_number(self):
        for ask, line in ((0, "0"), ("none", "0"), (0.5, "0.5"), (0.35, "0.35"), ("0.3", "0.3"), ("xhigh", "0.99"), (0.99, "0.99"), (5, "0.99"),
                          ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"), ("high", "0.9")):
            prompt = self.render(conversation()[:2], None, {"reasoning_effort": ask})
            self.assertIn(f"<|message_system|><|content_text|>Thinking effort level: {line}<|end_message|>", prompt, ask)

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


def _flat_tools():
    return [{"name": t["function"]["name"], "description": t["function"]["description"], "parameters": t["function"]["parameters"]} for t in TOOLS]


def openai_requests():
    """The same session as three growing /v1/chat/completions bodies (what pi, Hermes and the desktop agent send)."""
    full = conversation()
    cuts = [2, 4, 6, 9, 11, 13]       # after the first user turn, each tool result block and each later user turn
    return [{"model": "m", "messages": full[:n], "tools": TOOLS} for n in cuts]


def anthropic_requests():
    """... as /v1/messages bodies (Claude Code style): the system prompt on its own, tool_use / tool_result blocks."""
    blocks = [
        {"role": "user", "content": "open setup.py"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "setup.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "print('hi')"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It prints hi. Running it."},
                                          {"type": "tool_use", "id": "t2", "name": "run", "input": {"cmd": "python setup.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        {"role": "user", "content": "thanks"},
    ]
    return [{"model": "m", "system": "You are a coding agent. Use the tools.", "messages": blocks[:n], "tools": [
        {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in TOOLS]}
        for n in (1, 3, 5, 7)]


def responses_requests():
    """... as /v1/responses bodies (Codex style): function_call / function_call_output items."""
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "open setup.py"}]},
        {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": json.dumps({"path": "setup.py"})},
        {"type": "function_call_output", "call_id": "c1", "output": "print('hi')"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It prints hi."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "thanks"}]},
    ]
    return [{"model": "m", "instructions": "You are a coding agent. Use the tools.", "input": items[:n]} for n in (1, 3, 5)]


LEVEL_NUMBER = {"none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7", "high": "0.9", "max": "0.99"}


class _LevelCases:
    """What each thinking level does to the prompt (the model pack's real template): level 0 ends the prompt with the text opener (no tools) or the empty
    thinking turn (tools); a level above 0 renders its number and appends nothing, the model opens its own thinking turn; the history chains either way."""

    def effort_line(self):
        return f"<|message_system|><|content_text|>Thinking effort level: {LEVEL_NUMBER[self.LEVEL]}<|end_message|>"

    def test_the_levels_number_is_in_the_prompt(self):
        for tools in (None, TOOLS):
            self.assertIn(self.effort_line(), self.render(conversation()[:2], tools))

    def test_the_prompt_ends_as_the_level_says(self):
        for tools, zero_end in ((None, "<|message_model|><|content_text|>"), (TOOLS, _EMPTY_THINK)):
            prompt = self.render(conversation()[:2], tools)
            if self.LEVEL == "none":
                self.assertTrue(prompt.endswith(zero_end), prompt[-80:])
            else:
                self.assertTrue(prompt.endswith("<|content_text|>open setup.py<|end_message|>"), prompt[-80:])     # nothing appended
            self.assertNotIn(inkling_dialect._EMPTY_THINK_MARK, prompt)

    def test_history_chains_with_and_without_tools(self):
        self.chain(conversation(), TOOLS)
        plain = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"},
                 {"role": "assistant", "content": "d"}, {"role": "user", "content": "e"}]
        self.chain(plain, None)

    def test_no_empty_thinking_turn_is_replayed_above_level_zero(self):
        prompt = self.render(conversation(), TOOLS)
        n = prompt.count(_EMPTY_THINK)
        self.assertEqual(n, (sum(1 for m in conversation() if m["role"] == "assistant") + 1) if self.LEVEL == "none" else 0)

    def test_the_thinking_a_client_sends_back_is_rendered_once(self):
        conv = conversation()
        conv[2]["reasoning_content"] = "I should read it."
        prompt = self.render(conv[:4], TOOLS)
        self.assertEqual(prompt.count("<|content_thinking|>I should read it.<|end_message|>"), 1)

    def test_a_forced_call_follows_the_prompt(self):
        first = self.render(conversation()[:2], TOOLS) + inkling_dialect.force_text("<tool_call>\n<function=read_file>\n")
        self.assertTrue(first.endswith('<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":'))
        nxt = conversation()[:2] + [{"role": "assistant", "content": None, "tool_calls": [call(1, "read_file", path="setup.py")]},
                                    {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"}]
        self.assertTrue(self.render(nxt, TOOLS).startswith(first))


for _level in ("none", "minimal", "low", "medium", "high", "max"):
    globals()[f"TestLevel_{_level}"] = type(f"TestLevel_{_level}", (_LevelCases, _Base), {"LEVEL": _level})


class TestProtocolsKeepThePrefix(_Base):
    """The prompt a request becomes after serve's own conversion (tool ids dropped, arguments parsed, the system prompt moved), for the
    three wire formats: each one's prompt starts with the previous request's."""

    def prompts(self, requests, convert):
        out = []
        for req in requests:
            messages, tools, kwargs = convert(req)
            out.append(self.render(messages, tools if tools is not None else TOOLS_FLAT, kwargs))
        return out

    def assert_chain(self, prompts, label):
        for i, (pa, pb) in enumerate(zip(prompts, prompts[1:])):
            k = next((n for n, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
            self.assertTrue(pb.startswith(pa), f"{label}: request {i + 2} stops matching request {i + 1} at char {k}: "
                                               f"{pa[max(0, k - 40):k + 40]!r} vs {pb[max(0, k - 40):k + 40]!r}")

    def test_openai_chat_completions(self):
        self.assert_chain(self.prompts(openai_requests(), frontend.openai_to_messages), "openai")

    def test_anthropic_messages(self):
        self.assert_chain(self.prompts(anthropic_requests(), frontend.anthropic_to_messages), "anthropic")

    def test_responses(self):
        from serve import responses
        reqs = responses_requests()
        prompts = []
        for req in reqs:
            messages = responses.input_messages(req)
            if req.get("instructions"):
                messages = [{"role": "system", "content": req["instructions"]}] + messages
            prompts.append(self.render(messages, TOOLS_FLAT, {}))
        self.assert_chain(prompts, "responses")


TOOLS_FLAT = _flat_tools()


if __name__ == "__main__":
    unittest.main()
