"""Tool parameters coerced by their declared type (one helper for both parsers) and the damaged `<=NAME>` parameter
opener, from a Coder-L Grep transcript."""
import json
import unittest

from serve.frontend import OutputParser, coerce_args, coerce_param

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

    def test_undeclared_name_is_dropped(self):
        text = call(("pattern", "x")).replace("</function>", "<=bogus>\nv\n</parameter>\n</function>")
        for stream, step, got in self.each(text):
            self.assertEqual(got, [{"pattern": "x"}], (stream, step))

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
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=True)
        evs = p.feed(CUT) + p.finish("length")
        self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"),
                         '{"pattern":"loop guard: you ran this command')
        self.assertEqual(calls(evs), [])


RW = [{"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"}}}},
      {"name": "Write", "parameters": {"required": ["file_path", "content"], "properties": {
          "file_path": {"type": "string"}, "content": {"type": "string"}}}},
      {"name": "Edit", "parameters": {"properties": {"file_path": {"type": "string"},
                                                     "old_string": {"type": "string"},
                                                     "new_string": {"type": "string"}}}}]
PE, PS, FS, FE, CS, CE = "</parameter>", "<parameter=", "<function=", "</function>", "<tool_call>", "</tool_call>"
FAKE = CS + "\n" + FS + "Read>\n" + PS + "file_path>\n/x\n" + PE + "\n" + FE + "\n" + CE


def run_rw(text, stream, step=None):
    p = OutputParser(thinking=False, tools=RW, stream_tools=stream, recover=True)
    evs = []
    for k in range(0, len(text), step or len(text)):
        evs += p.feed(text[k:k + (step or len(text))])
    return calls(evs + p.finish())


class EveryPathTests(unittest.TestCase):
    def test_json_form_coerced(self):
        for body, want in (('{"name":"Grep","arguments":{"pattern":"x","-n":"true"}}', {"pattern": "x", "-n": True}),
                           ('{"name":"Grep","arguments":{"pattern":"x","head_limit":"200"}}',
                            {"pattern": "x", "head_limit": 200}),
                           ('{"name":"Grep","arguments":{"pattern":"x","head_limit":7,"-n":false}}',
                            {"pattern": "x", "head_limit": 7, "-n": False})):
            for stream in (False, True):
                got = calls(run(CALL_JSON % body, stream, 3))
                self.assertEqual(got, [want], (body, stream))
                self.assertIs(type(got[0].get("-n", 0)) is bool or "-n" not in got[0], True)

    def test_json_read_limit_int(self):
        got = run_rw(CS + '{"name":"Read","arguments":{"file_path":"/a","limit":"200"}}' + CE, False)
        self.assertEqual(got, [{"file_path": "/a", "limit": 200}])
        self.assertIs(type(got[0]["limit"]), int)

    def test_coerce_args_leaves_non_strings_and_decodes_undeclared(self):
        schema = TOOLS[0]
        self.assertEqual(coerce_args({"-n": "true", "head_limit": 3, "zzz": "1", "pattern": "1"}, schema),
                         {"-n": True, "head_limit": 3, "zzz": 1, "pattern": "1"})

    def test_name_fused_with_first_parameter(self):
        text = CS + "\n" + FS + "Read " + PS + "file_path>\n/a\n" + PE + "\n" + FE + "\n" + CE
        for stream in (False, True):
            for step in (None, 4, 40):
                self.assertEqual(run_rw(text, stream, step), [{"file_path": "/a"}], (stream, step))
        evs = []
        p = OutputParser(thinking=False, tools=RW, stream_tools=False, recover=True)
        evs = p.feed(text) + p.finish()
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["Read"])

    def test_unknown_prefix_unchanged(self):
        text = CS + "\n" + FS + "Nope " + PS + "file_path>\n/a\n" + PE + "\n" + FE + "\n" + CE
        for stream in (False, True):
            p = OutputParser(thinking=False, tools=RW, stream_tools=stream, recover=True)
            names = [e.call.name for e in p.feed(text) + p.finish() if e.kind == "tool_call"]
            self.assertNotIn("Read", names, stream)


class MarkupInValueTests(unittest.TestCase):
    def write(self, content):
        return (CS + "\n" + FS + "Write>\n" + PS + "file_path>\n/t.py\n" + PE + "\n" + PS + "content>\n" + content
                + "\n" + PE + "\n" + FE + "\n" + CE)

    def test_write_with_fake_call_in_content(self):
        for content in (FAKE, "a\n" + FAKE + "\nb", "x " + PE + " y"):
            for stream in (False, True):
                for step in (None, 1, 3, 7, 29):
                    self.assertEqual(run_rw(self.write(content), stream, step),
                                     [{"file_path": "/t.py", "content": content}], (content, stream, step))

    def test_edit_with_parameter_close_in_strings(self):
        old, new = "a " + PE + " b", "c\n" + PE + "\nd"
        text = (CS + "\n" + FS + "Edit>\n" + PS + "file_path>\n/t\n" + PE + "\n" + PS + "old_string>\n" + old + "\n"
                + PE + "\n" + PS + "new_string>\n" + new + "\n" + PE + "\n" + FE + "\n" + CE)
        for stream in (False, True):
            for step in (None, 1, 5):
                self.assertEqual(run_rw(text, stream, step),
                                 [{"file_path": "/t", "old_string": old, "new_string": new}], (stream, step))


CALL_JSON = "<tool_call>\n%s\n</tool_call>"


class MissingCloserTests(unittest.TestCase):
    """A turn that stops on its own with the call's closers missing: whole arguments, never JSON cut after a key."""
    BASE = "<tool_call>\n<function=Grep>\n<parameter=pattern>\na|b\n</parameter>\n<parameter=-n>\ntrue"
    WANT = {"pattern": "a|b", "-n": True}

    def test_last_boolean_complete(self):
        for tail in ("", "\n</parameter>", "\n</function>", "\n</parameter>\n</function>",
                     "\n</parameter>\n</function>\n</tool_call>"):
            for step in (None, 1, 7):
                evs = run(self.BASE + tail, True, step)
                self.assertEqual(json.loads("".join(e.text for e in evs if e.kind == "tool_args")), self.WANT,
                                 (tail, step))
                self.assertEqual(calls(evs), [self.WANT], (tail, step))

    def test_every_emitted_call_is_valid_json(self):
        for text in (TRUE_N, DAMAGED1, self.BASE, self.BASE + "\n</function>"):
            for stream in (False, True):
                for e in run(text, stream, 5):
                    if e.kind == "tool_call":
                        json.loads(json.dumps(e.call.arguments))


if __name__ == "__main__":
    unittest.main()
