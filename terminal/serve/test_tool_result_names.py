"""#7051: a tool result keeps the name of the function whose call it answers, so Inkling's header reads
`<|message_tool|>get_weather<|content_text|>` and not `<|message_tool|><|content_text|>`.

The front end drops the call ids before the template runs, so the template's own match by id can never happen; the name is
copied onto the result first (chat completions, Anthropic messages, Responses API).  Results sent in the other order than the
calls come out in call order, each with its own name.  Other families' messages are not changed.

    python -m pytest serve/test_tool_result_names.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import frontend, responses  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "inkling" / "chat_template.jinja"
TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
         for n in ("get_weather", "get_time")]


def headers(rendered: str) -> list[str]:
    """The text between each <|message_tool|> and its <|content_text|>, and the content after it."""
    out = []
    for part in rendered.split("<|message_tool|>")[1:]:
        head, _, rest = part.partition("<|content_text|>")
        out.append((head, rest.split("<|end_message|>")[0]))
    return out


class ToolResultNames(unittest.TestCase):
    def setUp(self):
        frontend.use_family("inkling2")
        self.addCleanup(frontend.use_family, None)
        self.tpl = ChatTemplate(TEMPLATE)

    def render(self, messages, tools=None):
        return self.tpl.render(messages, tools=tools, add_generation_prompt=False, reasoning_effort="low")

    def openai(self, calls, results):
        return {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}} for cid, name in calls]},
            *[{"role": "tool", "tool_call_id": cid, "content": text} for cid, text in results]], "tools": TOOLS}

    def test_openai_one_call(self):
        messages, tools, _ = openai_to_messages(self.openai([("call_1", "get_weather")], [("call_1", '{"temperature":18}')]))
        self.assertEqual(messages[2]["name"], "get_weather")
        self.assertNotIn("_call_id", messages[2])
        self.assertEqual(headers(self.render(messages, tools)), [("get_weather", '{"temperature":18}')])

    def test_openai_two_calls_answered_in_the_other_order(self):
        req = self.openai([("a1", "get_weather"), ("b2", "get_time")], [("b2", "12:00"), ("a1", "sunny")])
        messages, tools, _ = openai_to_messages(req)
        self.assertEqual([(m["name"], m["content"]) for m in messages if m["role"] == "tool"],
                         [("get_weather", "sunny"), ("get_time", "12:00")])
        self.assertEqual(headers(self.render(messages, tools)), [("get_weather", "sunny"), ("get_time", "12:00")])

    def test_two_rounds_each_result_takes_the_name_of_its_own_round(self):
        req = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "one"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": "get_time", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "two"}], "tools": TOOLS}
        messages, _, _ = openai_to_messages(req)
        self.assertEqual([m["name"] for m in messages if m["role"] == "tool"], ["get_weather", "get_time"])

    def test_a_result_that_matches_no_call_or_two_stays_unnamed(self):
        messages, _, _ = openai_to_messages(self.openai([("a1", "get_weather")], [("zzz", "?")]))
        self.assertNotIn("name", messages[2])
        messages, _, _ = openai_to_messages(self.openai([("dup", "get_weather"), ("dup", "get_time")], [("dup", "?")]))
        self.assertNotIn("name", messages[2])                                # ambiguous: no guess
        req = self.openai([("a1", "get_weather")], [("a1", "ok")])
        req["messages"][2].pop("tool_call_id")
        messages, _, _ = openai_to_messages(req)                             # no id at all (legacy client)
        self.assertNotIn("name", messages[2])

    def anthropic(self, calls, results):
        return {"max_tokens": 50, "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": name, "input": {}} for cid, name in calls]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": text} for cid, text in results]}],
            "tools": [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}} for n in ("get_weather", "get_time")]}

    def test_anthropic_one_call_and_two_in_the_other_order(self):
        messages, tools, _ = anthropic_to_messages(self.anthropic([("toolu_1", "get_weather")], [("toolu_1", "18")]), True)
        self.assertEqual(headers(self.render(messages, tools)), [("get_weather", "18")])
        req = self.anthropic([("a1", "get_weather"), ("b2", "get_time")], [("b2", "12:00"), ("a1", "sunny")])
        messages, tools, _ = anthropic_to_messages(req, True)
        self.assertEqual(headers(self.render(messages, tools)), [("get_weather", "sunny"), ("get_time", "12:00")])

    def test_responses_api(self):
        req = {"input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {"type": "function_call", "call_id": "a1", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call", "call_id": "b2", "name": "get_time", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "b2", "output": "12:00"},
            {"type": "function_call_output", "call_id": "a1", "output": "sunny"}]}
        messages = responses.input_messages(req)
        self.assertEqual(headers(self.render(messages, [t["function"] for t in TOOLS])),
                         [("get_weather", "sunny"), ("get_time", "12:00")])

    def test_other_families_get_no_name(self):
        for family in (None, "hy4"):
            frontend.use_family(family)
            messages, _, _ = openai_to_messages(self.openai([("call_1", "get_weather")], [("call_1", "18")]))
            self.assertNotIn("name", messages[2], family)
            messages, _, _ = anthropic_to_messages(self.anthropic([("t1", "get_weather")], [("t1", "18")]), True)
            self.assertNotIn("name", [m for m in messages if m["role"] == "tool"][0], family)
            msgs = responses.input_messages({"input": [
                {"type": "function_call", "call_id": "a1", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a1", "output": "x"}]})
            self.assertNotIn("name", [m for m in msgs if m["role"] == "tool"][0], family)

    def test_without_the_fix_the_header_is_empty(self):
        # the shape the report showed: the same conversation with the name left out of the tool message
        messages, tools, _ = openai_to_messages(self.openai([("call_1", "get_weather")], [("call_1", "18")]))
        messages[2].pop("name", None)
        self.assertEqual(headers(self.render(messages, tools)), [("", "18")])


if __name__ == "__main__":
    unittest.main()
