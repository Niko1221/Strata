"""Token identity, not decoded spelling, separates reasoning from the answer."""
import inspect
import unittest

from serve.frontend import OutputParser


TOOLS = [{"name": "write", "parameters": {"properties": {"text": {"type": "string"}}}}]
CALL = "<tool_call><function=write><parameter=text>{}</parameter></function></tool_call>"


def parser(**kwargs):
    # Run the same regressions against pre-fix main to demonstrate the text-only bug.
    if "token_aware" in inspect.signature(OutputParser).parameters:
        kwargs["token_aware"] = True
    return OutputParser(**kwargs)


def close(p):
    return p.end_thinking() if hasattr(p, "end_thinking") else p.feed("</think>")


def text(events, kind):
    return "".join(e.text for e in events if e.kind == kind)


class LiteralThinkParserTests(unittest.TestCase):
    def test_plain_tag_stays_reasoning_at_every_chunk_boundary(self):
        source = "Describe `</think>` and <think> literally."
        for split in range(len(source) + 1):
            with self.subTest(split=split):
                p = parser()
                events = p.feed(source[:split]) + p.feed(source[split:])
                events += close(p) + p.feed("\n42") + p.finish()
                self.assertEqual(text(events, "reasoning"), source)
                self.assertEqual(text(events, "content"), "42")

    def test_real_marker_closes_midline_even_in_code(self):
        for source in ("answer ready", "unfinished `", "```python\nx"):
            with self.subTest(source=source):
                p = parser()
                events = p.feed(source) + close(p) + p.feed("42") + p.finish()
                self.assertEqual(text(events, "reasoning"), source)
                self.assertEqual(text(events, "content"), "42")

    def test_real_marker_flushes_partial_tool_opener(self):
        p = parser(tools=TOOLS)
        events = p.feed("Explain <tool_") + close(p) + p.feed("answer") + p.finish()
        self.assertEqual(text(events, "reasoning"), "Explain <tool_")
        self.assertEqual(text(events, "content"), "answer")

    def test_plain_partial_tag_before_marker_is_preserved(self):
        for source in ("literal </thi", "literal \U000f0e02", "literal \x00"):
            with self.subTest(source=source):
                p = parser(tools=TOOLS)
                events = p.feed(source) + close(p) + p.feed("42") + p.finish()
                self.assertEqual(text(events, "reasoning"), source)
                self.assertEqual(text(events, "content"), "42")

    def test_literal_in_reasoning_call_argument_is_not_a_marker(self):
        for width in (1, 2, 7, 1000):
            with self.subTest(width=width):
                p = parser(tools=TOOLS)
                source, events = CALL.format("</think>"), []
                for i in range(0, len(source), width):
                    events += p.feed(source[i:i + width])
                events += close(p) + p.finish()
                calls = [e.call for e in events if e.kind == "tool_call"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].arguments, {"text": "</think>"})

    def test_literal_after_pending_call_keeps_call_quoted(self):
        source = CALL.format("example") + "</think> means end of reasoning"
        p = parser(tools=TOOLS)
        events = p.feed(source) + close(p) + p.feed("42") + p.finish()
        self.assertFalse(any(e.kind == "tool_call" for e in events))
        self.assertEqual(text(events, "reasoning"), source)

    def test_real_marker_inside_unfinished_call_flushes_reasoning(self):
        source = "<tool_call><function=write><parameter=text>partial"
        p = parser(tools=TOOLS)
        events = p.feed(source) + close(p) + p.feed("42") + p.finish()
        self.assertEqual(text(events, "reasoning"), source)
        self.assertFalse(any(e.kind == "tool_call" for e in events))
        self.assertEqual(text(events, "content"), "42")

    def test_real_marker_delivers_pending_call(self):
        p = parser(tools=TOOLS)
        events = p.feed(CALL.format("hello") + "\n") + close(p) + p.finish()
        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"text": "hello"}])

    def test_answer_and_tool_argument_preserve_extra_marker(self):
        p = parser(thinking=False, tools=TOOLS)
        events = p.feed("literal ") + close(p) + p.feed("\n" + CALL.format("</think>")) + p.finish()
        self.assertEqual(text(events, "content"), "literal </think>")
        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"text": "</think>"}])

    def test_text_only_callers_keep_existing_behavior(self):
        p = OutputParser()
        events = p.feed("reason</thi") + p.feed("nk>\nanswer") + p.finish()
        self.assertEqual(text(events, "reasoning"), "reason")
        self.assertEqual(text(events, "content"), "answer")


if __name__ == "__main__":
    unittest.main()
