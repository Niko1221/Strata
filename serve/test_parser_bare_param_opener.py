"""Coder-L raw outputs (STRATA_LOG_RAW, 2026-10-25): a parameter opened as a bare `<NAME>` and closed by `</parameter>`."""
import unittest

from serve.frontend import OutputParser

TOOLS = [
    {"name": "Grep", "parameters": {"properties": {
        "pattern": {"type": "string"}, "path": {"type": "string"}, "output_mode": {"type": "string"},
        "-n": {"type": "boolean"}, "head_limit": {"type": "integer"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
SAMPLE_A = ("<tool_call>\n<Grep>\n<pattern>\nloopRepeat|loop-guard|loopRepeat\\\\(\n</parameter>\n"
            "<parameter=path>\ntest\n</parameter>\n<parameter=output_mode>\ncontent\n</parameter>\n"
            "<parameter=-n>\ntrue\n</parameter>\n</function>\n</tool_call>")
SAMPLE_B = ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nfoo\n</parameter>\n"
            "<parameter=output_mode>\ncontent\n</parameter>\n<head_limit>\n10\n</parameter>\n</function>\n</tool_call>")


def calls(text, step=None):
    p = OutputParser(thinking=False, tools=TOOLS, recover=True)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    evs += p.finish()
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class BareParamOpenerTests(unittest.TestCase):
    def test_sample_a_keeps_pattern(self):
        for step in (None, 1, 3, 7):
            self.assertEqual(calls(SAMPLE_A, step), [("Grep", {
                "pattern": "loopRepeat|loop-guard|loopRepeat\\\\(", "path": "test",
                "output_mode": "content", "-n": True})], step)

    def test_sample_b_keeps_head_limit(self):
        for step in (None, 1, 3, 7):
            self.assertEqual(calls(SAMPLE_B, step), [("Grep", {
                "pattern": "foo", "output_mode": "content", "head_limit": 10})], step)

    def test_tags_in_a_value_stay_byte_for_byte(self):
        content = "<html>\n<pattern>\nx\n</pattern>\n<div>\n</html>"
        text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\na.html\n</parameter>\n"
                "<parameter=content>\n%s\n</parameter>\n</function>\n</tool_call>" % content)
        for step in (None, 1, 5):
            self.assertEqual(calls(text, step), [("Write", {"file_path": "a.html", "content": content})], step)


if __name__ == "__main__":
    unittest.main()
