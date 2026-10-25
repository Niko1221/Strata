"""Coder-L writes a parameter opener as a bare `<NAME>` after `</parameter>`; a declared NAME opens that parameter."""
import unittest

from serve.frontend import OutputParser

TOOLS = [
    {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"},
                                                    "limit": {"type": "integer"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
READ = ("<tool_call>\n<function=Read>\n<parameter=file_path>\n/x/src/workers/pane-exec.ts\n</parameter>\n"
        "<offset>\n638\n</parameter>\n<limit>\n44\n</parameter>\n</function>\n</tool_call>")
WANT = [("Read", {"file_path": "/x/src/workers/pane-exec.ts", "offset": 638, "limit": 44})]


def calls(text, step=None, stream_tools=False):
    p = OutputParser(thinking=False, tools=TOOLS, recover=True, stream_tools=stream_tools)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    evs += p.finish()
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class BareOpenerTests(unittest.TestCase):
    def test_read_example_every_chunking(self):
        for stream in (False, True):
            for step in (None, 1, 3, 7, 29):
                self.assertEqual(calls(READ, step, stream), WANT, (stream, step))

    def test_plain_text_after_a_closer_stays_in_the_value(self):
        for content in ("a</parameter>\nbold", "<div>\n<p>x</p>\n</div>", "x\n</parameter>\ny <div>y"):
            text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/a.html\n</parameter>\n"
                    "<parameter=content>\n%s\n</parameter>\n</function>\n</tool_call>" % content)
            for step in (None, 1, 7):
                self.assertEqual(calls(text, step), [("Write", {"file_path": "/a.html", "content": content})], step)

    def test_a_tag_after_a_closer_ends_the_value(self):
        text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/a\n</parameter>\n<parameter=content>\n"
                "x\n</parameter>\n<content>\ny\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual(calls(text), [("Write", {"file_path": "/a", "content": "x"})])


if __name__ == "__main__":
    unittest.main()
