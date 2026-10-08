"""HarmonyParser (GPT-OSS's harmony output format): python -m unittest serve.test_harmony"""
import unittest

from serve.frontend import HarmonyParser


def run(text, step=None, reason="stop"):
    """Feed `text` whole, or `step` characters at a time; -> (reasoning, content, calls) joined per kind."""
    p = HarmonyParser()
    evs = []
    pieces = [text] if step is None else [text[i:i + step] for i in range(0, len(text), step)]
    for piece in pieces:
        evs += p.feed(piece)
    evs += p.finish(reason)
    join = lambda kind: "".join(e.text for e in evs if e.kind == kind)
    return join("reasoning"), join("content"), [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


THINK_ANSWER = ("<|channel|>analysis<|message|>User asks 2+2. Easy: 4.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>2 + 2 = **4**.<|return|>")
CALL = ('<|channel|>analysis<|message|>Need the weather tool.<|end|>'
        '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json'
        '<|message|>{"city": "Paris", "unit": "C"}<|call|>')


class Harmony(unittest.TestCase):
    def test_reasoning_then_answer_whole_and_char_by_char(self):
        for step in (None, 1, 3, 7):
            self.assertEqual(run(THINK_ANSWER, step), ("User asks 2+2. Easy: 4.", "2 + 2 = **4**.", []), step)

    def test_a_tool_call(self):
        for step in (None, 1, 5):
            r, c, calls = run(CALL, step)
            self.assertEqual((r, c), ("Need the weather tool.", ""))
            self.assertEqual(calls, [("get_weather", {"city": "Paris", "unit": "C"})])

    def test_recipient_before_the_channel_and_a_preamble(self):
        text = ("<|channel|>commentary<|message|>Checking the files.<|end|>"
                "<|start|>assistant to=functions.ls<|channel|>commentary json<|message|>{\"path\": \".\"}<|call|>")
        self.assertEqual(run(text, 2), ("", "Checking the files.", [("ls", {"path": "."})]))

    def test_the_engine_may_stop_before_the_end_marker(self):
        # the engine ends at <|return|> / <|call|> (end-of-generation tokens) and may not hand them back
        self.assertEqual(run(THINK_ANSWER[:-len("<|return|>")], 1)[1], "2 + 2 = **4**.")
        self.assertEqual(run(CALL[:-len("<|call|>")])[2], [("get_weather", {"city": "Paris", "unit": "C"})])

    def test_a_cut_call_is_text_not_a_call(self):
        _, c, calls = run(CALL[:-len("<|call|>")], reason="length")
        self.assertEqual(calls, [])
        self.assertIn('"Paris"', c)

    def test_no_marker_text_leaks_and_buf_is_clean_mid_reasoning(self):
        p = HarmonyParser()
        evs = p.feed("<|channel|>analysis<|message|>thinking about it <|")
        self.assertEqual("".join(e.text for e in evs), "thinking about it ")   # "<|" held: may start a marker
        self.assertEqual((p.state, p.buf), ("reasoning", "<|"))
        evs = p.feed("x")                                       # not a marker after all
        self.assertEqual("".join(e.text for e in evs), "<|x")
        self.assertEqual(p.buf, "")                              # a clean point: the server may close the thinking
        r, c, _ = run(THINK_ANSWER, 1)
        self.assertNotIn("<|", r + c)

    def test_a_wrapped_up_budget_reads_as_the_answer(self):
        # Service.run closes the thinking with this text when reasoning_budget_tokens runs out
        from serve.server import HARMONY_WRAP_UP
        text = "<|channel|>analysis<|message|>long thought" + HARMONY_WRAP_UP + "Answer."
        r, c, _ = run(text, 4)
        self.assertTrue(r.startswith("long thought") and c == "Answer.", (r, c))


class HarmonyTemplate(unittest.TestCase):
    """GPT-OSS's own chat template (fixtures/gpt-oss-chat_template.jinja, Apache-2.0) through Strata's ChatTemplate."""

    def setUp(self):
        from pathlib import Path
        from serve.frontend import ChatTemplate
        self.t = ChatTemplate(Path(__file__).parent / "fixtures" / "gpt-oss-chat_template.jinja")

    def test_a_tool_round_trip_renders_in_harmony(self):
        from serve.frontend import openai_to_messages
        req = {"messages": [{"role": "user", "content": "Weather in Paris?"},
                            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                            {"role": "tool", "tool_call_id": "c1", "content": "18 C"}],
               "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
                          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}
        messages, tools, _ = openai_to_messages(req)       # Strata's flat tools: the template needs them wrapped
        out = self.t.render(messages, tools=tools, reasoning_effort="low")
        self.assertIn("Reasoning: low", out)
        self.assertIn("Current date: ", out)                # strftime_now, as transformers provides it
        self.assertIn("type get_weather = (_: {", out)
        self.assertIn('<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Paris"}'
                      "<|call|>", out)
        self.assertTrue(out.endswith("<|start|>assistant"))

    def test_capabilities(self):
        caps = self.t.caps
        self.assertTrue(caps["supports_tools"] and caps["supports_tool_calls"] and caps["supports_system_role"])
        self.assertFalse(caps["supports_parallel_tool_calls"])    # the template renders one call per message


if __name__ == "__main__":
    unittest.main()
