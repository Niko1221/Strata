"""#4078: a string argument that quotes a whole tool call, or reopens its own parameter, stays exact.

    python -m unittest serve.test_quoted_call_values -v
"""
import json
import unittest

from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser,
                            parse_tool_call)

SCHEMA = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}
QUOTED = (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nx.txt\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")
VALUES = {
    "quoted call": f"example transcript:\n{QUOTED}\nafter block",
    "quoted call at the end": f"example transcript:\n{QUOTED}",
    "heredoc": f"python3 - <<'PY'\nprint('''{QUOTED}''')\nPY",
    "repeated parameter": f"head part\n{PARAM_END}\n{PARAM_START}content>\ntail part",
    "unclosed opener in prose": f"A call starts with `{FUNC_START}NAME>` and its parameters follow.",
    "unclosed opener in code": f'opener = "{FUNC_START}"\n',
    "unclosed quoted call": f"a call begins with {FUNC_START}write>\n{PARAM_START}path>\nx",
    "closed then unclosed quote": f"a {QUOTED} b {FUNC_START}write>\n{PARAM_START}path>\nx",
    "two quoted calls": f"a {QUOTED}\n{QUOTED} b",
}
UNCLOSED = f"see {FUNC_START}write>\n{PARAM_START}path>\nx"


def call_text(value):
    return (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nprobe.txt\n{PARAM_END}\n"
            f"{PARAM_START}content>\n{value}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")


class QuotedCallValues(unittest.TestCase):
    def test_closing_tag_list_and_unclosed_quote_stream_exactly(self):
        doc = f"Closing tags:\n\n{PARAM_END}\n{FUNC_END}\n{CALL_END}\n\nThe document continues."
        for value in (doc, doc + "\n" + UNCLOSED, UNCLOSED + "\n" + doc):
            for size in (1, 7, 100000):
                with self.subTest(value=value, size=size):
                    parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=True)
                    events = []
                    text = call_text(value)
                    for i in range(0, len(text), size):
                        events.extend(parser.feed(text[i:i + size]))
                    events.extend(parser.finish())
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    expected = {"path": "probe.txt", "content": value}
                    self.assertEqual(calls[0].arguments, expected)
                    self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), expected)

    def test_values_are_exact_whole_streamed_and_chunked(self):
        for label, value in VALUES.items():
            text = call_text(value)
            expected = {"path": "probe.txt", "content": value}
            self.assertEqual(parse_tool_call(text[len(CALL_START):-len(CALL_END)], SCHEMA).arguments, expected)
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        events.extend(parser.finish())
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        self.assertEqual([c.arguments for c in calls], [expected])
                        self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "")
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                             expected)

    def test_prose_between_two_calls_still_ends_the_first(self):
        first, second = call_text("one"), call_text("two")
        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=True)
        events = parser.feed(first + "\nnow the second:\n" + second) + parser.finish()
        self.assertEqual([e.call.arguments["content"] for e in events if e.kind == "tool_call"], ["one", "two"])

    def test_unclosed_quote_before_another_parameter_or_call(self):
        edit = {"name": "edit", "parameters": {"properties": {k: {"type": "string"} for k in ("path", "old", "new")}}}
        one = (f"{CALL_START}\n{FUNC_START}edit>\n{PARAM_START}old>\n{UNCLOSED}\n{PARAM_END}\n"
               f"{PARAM_START}new>\ny\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")
        cases = {"next parameter": (one, [{"old": UNCLOSED, "new": "y"}]),
                 "next call": (call_text(UNCLOSED) + "\n" + call_text("two"),
                               [{"path": "probe.txt", "content": UNCLOSED}, {"path": "probe.txt", "content": "two"}])}
        for label, (text, expected) in cases.items():
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        parser = OutputParser(thinking=False, tools=[SCHEMA, edit], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        events.extend(parser.finish())
                        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], expected)

    def test_unclosed_quote_streams_what_the_calls_get(self):
        f = {"name": "f", "parameters": {"properties": {"x": {"type": "string"}}}}
        body = f"{FUNC_START}f>{PARAM_START}x>{FUNC_START}q>{PARAM_START}z>v{PARAM_END}{FUNC_END}"
        nxt = f"{FUNC_START}f>{PARAM_START}x>ok{PARAM_END}{FUNC_END}"
        expected = [{"x": f"{FUNC_START}q>{PARAM_START}z>v"}, {"x": "ok"}]
        for label, text in (("batch", CALL_START + body + nxt + CALL_END), ("bare", body + nxt)):
            with self.subTest(label=label):
                parser = OutputParser(thinking=False, tools=[f], recover=True, stream_tools=True)
                events = []
                for ch in text:                      # a partial next opener must not release the held text
                    events.extend(parser.feed(ch))
                events.extend(parser.finish())
                calls = [e.call for e in events if e.kind == "tool_call"]
                self.assertEqual([c.arguments for c in calls], expected)
                for c in calls:
                    self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args" and e.call.id == c.id)),
                                     c.arguments)

    def test_unclosed_quote_in_a_call_from_the_reasoning(self):
        f = {"name": "f", "parameters": {"properties": {"x": {"type": "string"}}}}
        parser = OutputParser(thinking=True, tools=[f])
        events = parser.feed(f"{CALL_START}{FUNC_START}f>{PARAM_START}x>{UNCLOSED}{PARAM_END}{FUNC_END}{CALL_END}")
        events += parser.finish()
        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"x": UNCLOSED}])

    def test_many_unclosed_quotes_do_not_recurse_without_bound(self):
        text = (f"{CALL_START}{FUNC_START}f>" + "".join(f"{PARAM_START}p{i}>{FUNC_START}x>{PARAM_START}x>x{PARAM_END}"
                                                      for i in range(600)) + f"{FUNC_END}{CALL_END}")
        parser = OutputParser(thinking=False)
        parser.feed(text)
        parser.finish()


if __name__ == "__main__":
    unittest.main()
