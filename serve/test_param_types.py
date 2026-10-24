"""Tool parameters coerced by their declared type (one helper for both parsers) and the damaged `<=NAME>` parameter
opener, from a Coder-L Grep transcript."""
import json
import unittest

from serve.frontend import OutputParser, coerce_param

TOOLS = [{"name": "Grep", "parameters": {"properties": {
    "pattern": {"type": "string"}, "output_mode": {"type": "string"}, "-n": {"type": "boolean"},
    "head_limit": {"type": "integer"}, "glob": {"type": "array"}}}}]


def call(*params):
    body = "".join("<parameter=%s>\n%s\n</parameter>\n" % p for p in params)
    return "<tool_call>\n<function=Grep>\n" + body + "</function>\n</tool_call>"


TRUE_N = call(("pattern", "loop guard: you ran this command"), ("output_mode", "content"), ("-n", "True"))
DAMAGED1 = ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nloop guard\n</parameter>\n<=output_mode>\n"
            "files_with_matches\n</parameter>\n</function>\n</tool_call>")
DAMAGED2 = ("<tool_call>\n<function=Grep>\n<parameter=pattern>\nloop guard: you ran this command\n</parameter>\n"
            "<=output_mode>\ncontent\n</parameter>\n</function>\n</tool_call>")
CUT = "<tool_call>\n<function=Grep>\n<parameter=pattern>\nloop guard: you ran this command"


def run(text, stream, step=None, finish=True):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=True)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    return evs + (p.finish() if finish else [])


def calls(evs):
    return [e.call.arguments for e in evs if e.kind == "tool_call"]


def streamed(evs):
    return "".join(e.data for e in evs if e.kind == "tool_args") if evs and hasattr(evs[0], "data") else None


class CoerceTests(unittest.TestCase):
    def test_boolean_words(self):
        for v, want in (("True", True), (" false ", False), ("YES", True), ("no", False), ("1", True), ("0", False),
                        ("true", True)):
            self.assertIs(coerce_param(v, "boolean"), want, v)
        self.assertEqual(coerce_param("maybe", "boolean"), "maybe")
        self.assertEqual(coerce_param("2", "boolean"), "2")

    def test_numbers_arrays_and_fallbacks(self):
        self.assertEqual(coerce_param("12", "integer"), 12)
        self.assertEqual(coerce_param("1.5", "number"), 1.5)
        self.assertEqual(coerce_param('["a"]', "array"), ["a"])
        self.assertEqual(coerce_param('{"a":1}', "object"), {"a": 1})
        self.assertEqual(coerce_param("abc", "integer"), "abc")
        self.assertEqual(coerce_param("True", "string"), "True")


class ParserTests(unittest.TestCase):
    def each(self, text):
        for stream in (False, True):
            for step in (None, 1, 2, 3, 5, 11):
                yield stream, step, calls(run(text, stream, step))

    def test_true_for_boolean(self):
        for stream, step, got in self.each(TRUE_N):
            self.assertEqual(got, [{"pattern": "loop guard: you ran this command", "output_mode": "content",
                                    "-n": True}], (stream, step))

    def test_true_for_string_stays(self):
        text = call(("pattern", "True"))
        for stream, step, got in self.each(text):
            self.assertEqual(got, [{"pattern": "True"}], (stream, step))

    def test_damaged_param_opener(self):
        for text, want in ((DAMAGED1, {"pattern": "loop guard", "output_mode": "files_with_matches"}),
                           (DAMAGED2, {"pattern": "loop guard: you ran this command", "output_mode": "content"})):
            for stream, step, got in self.each(text):
                self.assertEqual(got, [want], (stream, step))

    def test_undeclared_name_stays_text(self):
        text = call(("pattern", "x")).replace("</function>", "<=bogus>\nv\n</parameter>\n</function>")
        for stream, step, got in self.each(text):
            self.assertEqual(got, [{"pattern": "x\n</parameter>\n<=bogus>\nv"}], (stream, step))

    def test_well_formed_unchanged(self):
        text = call(("pattern", "a"), ("head_limit", "5"), ("glob", '["*.py"]'))
        for stream, step, got in self.each(text):
            self.assertEqual(got, [{"pattern": "a", "head_limit": 5, "glob": ["*.py"]}], (stream, step))

    def test_streamed_args_are_the_same_json(self):
        for text, want in ((TRUE_N, {"pattern": "loop guard: you ran this command", "output_mode": "content", "-n": True}),
                           (DAMAGED1, {"pattern": "loop guard", "output_mode": "files_with_matches"})):
            for step in (None, 1, 4):
                evs = run(text, True, step)
                self.assertEqual(json.loads("".join(e.text for e in evs if e.kind == "tool_args")), want, step)

    def test_cut_turn_leaves_args_unclosed(self):
        # finish() on a turn cut inside a streamed call keeps the announced call's JSON open (#211): the source of
        # the client's `{"pattern":"loop guard: you ran this command` (no closer, no tool_call event)
        evs = run(CUT, True, 7)
        self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"),
                         '{"pattern":"loop guard: you ran this command')
        self.assertEqual(calls(evs), [])


if __name__ == "__main__":
    unittest.main()
