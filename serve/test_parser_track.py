"""Long Unicode streams and lookahead must preserve text and code decisions."""
import random
import unittest
from serve.frontend import OutputParser


class TrackTests(unittest.TestCase):
    def test_long_lines_and_quoted_tools_across_stream_boundaries(self):
        quoted = ("prefix café 🙂 " * 1024 + "`<tool_call>`\n\n```xml\n" + "x" * 16384
                  + "<tool_call><function=quoted></function></tool_call>\n```\n")
        text = quoted + "<tool_call><function=write><parameter=text>ok</parameter></function></tool_call>\nDone"
        for width in (17, 63, 64, 65, len(text)):
            for recover in (False, True):
                for stream in (False, True):
                    with self.subTest(width=width, recover=recover, stream=stream):
                        parser = OutputParser(thinking=False, tools=[{"name": "write"}],
                                              recover=recover, stream_tools=stream)
                        events = [event for i in range(0, len(text), width)
                                  for event in parser.feed(text[i:i + width])] + parser.finish()
                        self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                                         [("write", {"text": "ok"})])
                        self.assertEqual("".join(e.text for e in events if e.kind == "content"),
                                         quoted.rstrip("\n") + "Done")

    def test_random_unicode_stream_preserves_reasoning_and_final(self):
        thought = ("Reasoning `inline` café 🙂\n\n  ```python\n</think>quoted\n```\n"
                   "~~~xml\r\n<function=test>\n~~~\n\n" * 8)
        text = thought + "</think>\n\nDone"
        rng = random.Random(1029)
        for _ in range(20):
            parser, events, pos = OutputParser(), [], 0
            while pos < len(text):
                end = pos + rng.randint(1, 17)
                events.extend(parser.feed(text[pos:end]))
                pos = end
            events.extend(parser.finish())
            self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), thought)
            self.assertEqual("".join(e.text for e in events if e.kind == "content"), "Done")
            self.assertFalse(any(e.kind.startswith("tool_") for e in events))


if __name__ == "__main__":
    unittest.main()
