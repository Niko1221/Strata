"""Streaming and whole-call contracts, including large structured arguments."""
import json
import random
import unittest

from serve.frontend import OutputParser, parse_tool_call


class ParserStream(unittest.TestCase):
    schema = [{"name": "write", "parameters": {"properties": {
        "content": {"type": "string"}, "data": {"type": "object"}}}}]

    def check_call(self, body, step):
        parser = OutputParser(thinking=False, tools=self.schema, stream_tools=True)
        text = "<tool_call>" + body + "</tool_call>tail"
        events = []
        for i in range(0, len(text), step):
            events += parser.feed(text[i:i + step])
        events += parser.finish()
        calls = [e.call for e in events if e.kind == "tool_call"]
        expected = parse_tool_call(body, self.schema[0]).arguments
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, expected)
        self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), expected)
        starts = [e.call for e in events if e.kind == "tool_start"]
        self.assertEqual(starts[0].id, calls[0].id)
        self.assertEqual("".join(e.text for e in events if e.kind == "content"), "tail")

    def test_literal_tags_unicode_and_structured_values(self):
        body = ('<function=write>\n<parameter=content>\n'
                '你好 🌊 </parameter> is literal </function></tool_call>\n'
                '</parameter>\n<parameter=data>\n'
                '{"literal":"</parameter> x </tool_call>","v":[1,true,null]}\n'
                '</parameter>\n</function>')
        for step in (1, 2, 7, 13, len(body)):
            with self.subTest(step=step):
                self.check_call(body, step)

    def test_large_raw_and_string_arguments(self):
        for pname, value in (("content", "abc🌊" * 16384), ("data", json.dumps({"v": "abc" * 22000}))):
            self.check_call(f"<function=write><parameter={pname}>\n{value}\n</parameter></function>", 31)

    def test_multiple_calls_and_unfinished_call(self):
        text = '<tool_call><function=write><parameter=content>x</parameter></function></tool_call>'
        parser = OutputParser(thinking=False, tools=self.schema, stream_tools=True)
        events = parser.feed(text + text)
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual([c.arguments for c in calls], [{"content": "x"}] * 2)
        self.assertNotEqual(calls[0].id, calls[1].id)
        parser = OutputParser(thinking=False, tools=self.schema, stream_tools=False)
        pending = '<tool_call><function=write><parameter=content>unfinished'
        for ch in pending:
            parser.feed(ch)
        self.assertEqual(parser.finish()[0].text, pending)

    def test_many_calls_in_one_delta(self):
        text = '<tool_call><function=write><parameter=content>x</parameter></function></tool_call>'
        parser = OutputParser(thinking=False, tools=self.schema, stream_tools=True)
        events = parser.feed(text * 1500)
        self.assertEqual(sum(e.kind == "tool_call" for e in events), 1500)

    def test_random_fragmentation(self):
        text = ('<tool_call><function=write><parameter=data>{"v":[1,2,3]}</parameter>'
                '<parameter=content>line\n</parameter> literal\n</parameter></function></tool_call>')
        for seed in range(20):
            rng = random.Random(seed)
            parser = OutputParser(thinking=False, tools=self.schema, stream_tools=True)
            events, at = [], 0
            while at < len(text):
                count = rng.randint(1, 19)
                events += parser.feed(text[at:at + count])
                at += count
            events += parser.finish()
            call = next(e.call for e in events if e.kind == "tool_call")
            self.assertEqual(call.arguments, {"data": {"v": [1, 2, 3]}, "content": "line\n</parameter> literal"})


if __name__ == "__main__":
    unittest.main()
