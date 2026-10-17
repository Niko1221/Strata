"""serve/test_tool_args_stream.py - "stream_all_tool_args": a tool call's non-string argument in pieces.

By default a parameter that is not a string (an array, an object, a number) is held until it ends and goes out as one
"tool_args" piece.  For an `edits`-style argument - an array of objects - that leaves the stream quiet for minutes
while the model writes it, and the app shows nothing of it until it ends.  With the switch the value goes out as the
JSON text the model writes, in pieces.  A client concatenates the pieces either way (the final "tool_call" of a call
that was streamed is not sent again), so what it parses is the same arguments: every test here adds the pieces back
and compares them with the call the parser reports.

A parameter whose schema is not a plain "string" is not necessarily JSON (#2057 review): a union type, an anyOf, no
declared type, or a model that writes `abc` for an integer.  Such a value waits for its end and goes out quoted,
with the switch on as well as off - ValuesThatAreNotJsonText pins that its pieces add up too.

    python -m unittest serve.test_tool_args_stream -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "edit", "parameters": {"properties": {
    "path": {"type": "string"},
    "edits": {"type": "array"},
    "count": {"type": "integer"},
    "dry": {"type": "boolean"},
    "meta": {"type": "object"},
}}}]

EDITS = [{"target": "два", "text": "два 2\nчетыре 4", "where": "replace"},
         {"target": "три", "text": "x" * 120}]
ARR = json.dumps(EDITS, ensure_ascii=False)
ARGS = {"edits": EDITS, "path": "notes/a.md", "count": 2, "dry": True, "meta": {"src": "chat"}}
WIDTHS = (1, 2, 7, 8, 64, 100_000)


def call_text(edits_json: str = ARR) -> str:
    """The call in the template's form: the model writes JSON between <parameter=edits> and </parameter>."""
    return ("</think>\n\n<tool_call>\n<function=edit>\n<parameter=edits>\n" + edits_json +
            "\n</parameter>\n<parameter=path>\nnotes/a.md\n</parameter>\n<parameter=count>\n2\n</parameter>\n"
            "<parameter=dry>\ntrue\n</parameter>\n<parameter=meta>\n{\"src\": \"chat\"}\n</parameter>\n"
            "</function>\n</tool_call>")


def run(text: str, width: int, stream_all_args: bool) -> list:
    return run_with(TOOLS, text, width, stream_all_args)


def run_with(tools: list[dict], text: str, width: int, stream_all_args: bool) -> list:
    p = OutputParser(thinking=True, tools=tools, stream_tools=True, stream_all_args=stream_all_args)
    evs = []
    for i in range(0, len(text), width):
        evs += p.feed(text[i:i + width])
    return evs + p.finish()


def one_arg_call(prop: str, schema: dict, written: str) -> tuple[list[dict], str]:
    """One tool, one parameter, the value the model writes between its tags: a case of a schema on its own."""
    tools = [{"name": "t", "parameters": {"properties": {prop: schema}}}]
    return tools, ("</think>\n\n<tool_call>\n<function=t>\n<parameter=" + prop + ">\n" + written +
                   "\n</parameter>\n</function>\n</tool_call>")


def pieces(evs) -> list[str]:
    return [e.text for e in evs if e.kind == "tool_args"]


def reported(evs):
    calls = [e.call for e in evs if e.kind == "tool_call"]
    return calls[0].arguments if calls else None


class WholeByDefault(unittest.TestCase):
    """The default path is unchanged: the array goes out whole, in one piece, once it ends."""

    def test_the_array_goes_out_whole(self):
        text = call_text()
        for width in WIDTHS:
            with self.subTest(width=width):
                evs = run(text, width, False)
                self.assertIn(ARR, pieces(evs), "the array arrived in pieces without the switch")
                self.assertEqual(json.loads("".join(pieces(evs))), ARGS)


class InPieces(unittest.TestCase):
    """The switch: the value goes out as the model writes it, and the pieces add up to the same arguments."""

    def test_the_array_arrives_in_pieces(self):
        text = call_text()
        for width in (1, 7, 8, 64):
            with self.subTest(width=width):
                got = pieces(run(text, width, True))
                self.assertNotIn(ARR, got, "the whole array went out in one piece")
                self.assertGreater(len(got), len(pieces(run(text, width, False))),
                                   "the array did not arrive in pieces")

    def test_every_feed_width_adds_up(self):
        text = call_text()
        for width in WIDTHS:
            with self.subTest(width=width):
                evs = run(text, width, True)
                self.assertEqual(reported(evs), ARGS)
                self.assertEqual(json.loads("".join(pieces(evs))), ARGS)

    def test_a_value_that_holds_the_terminator(self):
        """#210: a </parameter> inside the value is part of the value; the pieces still add up."""
        edits = [{"target": "a", "text": "close each value with </parameter> and go on"}]
        text = call_text(json.dumps(edits, ensure_ascii=False))
        for width in (1, 7, 64, 100_000):
            with self.subTest(width=width):
                evs = run(text, width, True)
                self.assertEqual(reported(evs)["edits"], edits)
                self.assertEqual(json.loads("".join(pieces(evs)))["edits"], edits)

    def test_a_number_and_an_object_add_up(self):
        """The hold-back is by the declared type, so `count`, `dry` and `meta` were held whole as well."""
        text = call_text()
        for width in WIDTHS:
            for switch in (False, True):
                with self.subTest(width=width, stream_all_args=switch):
                    evs = run(text, width, switch)
                    self.assertEqual(json.loads("".join(pieces(evs))), reported(evs))

    def test_a_call_of_only_string_parameters_is_unchanged(self):
        """Strings already went out character by character: the switch does not touch them."""
        text = ("</think>\n\n<tool_call>\n<function=edit>\n<parameter=path>\nnotes/a.md\n</parameter>\n"
                "</function>\n</tool_call>")
        for width in WIDTHS:
            with self.subTest(width=width):
                self.assertEqual(pieces(run(text, width, False)), pieces(run(text, width, True)))


class ValuesThatAreNotJsonText(unittest.TestCase):
    """#2057 review: a parameter whose schema is not a plain "string" is not necessarily JSON.

    A union type, an anyOf, no declared type at all, an integer the model fills with `abc`, a boolean it fills with
    `yes`, an empty array: the value went out as written - unquoted - and the pieces a client joins were not valid
    JSON.  The final tool_call of a call that was streamed is not sent again, so the client keeps that broken text
    even though the call the server reports is right.  A value that does not open with [ { " now waits for its end
    and goes out the way it does without the switch; a value that does open with one of them is JSON text and still
    streams in pieces, whatever the schema says."""

    CASES = [
        ("nullable", {"type": ["string", "null"]}, "hello world"),
        ("anyof", {"anyOf": [{"type": "string"}, {"type": "object"}]}, "some text"),
        ("untyped", {}, "/tmp/a b.txt"),
        ("n", {"type": "integer"}, "abc"),
        ("flag", {"type": "boolean"}, "yes"),
        ("empty", {"type": "array"}, ""),
        ("num", {"type": "integer"}, "42"),
        ("nothing", {"type": "null"}, "null"),
    ]

    def test_the_pieces_add_up_to_valid_json(self):
        for prop, schema, written in self.CASES:
            tools, text = one_arg_call(prop, schema, written)
            for width in WIDTHS:
                with self.subTest(prop=prop, width=width):
                    evs = run_with(tools, text, width, True)
                    joined = json.loads("".join(pieces(evs)))      # ValueError before the fix
                    self.assertEqual(joined, reported(evs), "the pieces do not add up to the reported call")

    def test_a_plain_text_value_comes_back_as_the_string_the_model_wrote(self):
        for prop, schema, written in self.CASES[:6]:
            tools, text = one_arg_call(prop, schema, written)
            for width in WIDTHS:
                with self.subTest(prop=prop, width=width):
                    self.assertEqual(json.loads("".join(pieces(run_with(tools, text, width, True)))), {prop: written})

    def test_the_switch_changes_nothing_for_them(self):
        """What a value that is not JSON text sends is the same with the switch on and off."""
        for prop, schema, written in self.CASES:
            tools, text = one_arg_call(prop, schema, written)
            for width in WIDTHS:
                with self.subTest(prop=prop, width=width):
                    self.assertEqual(pieces(run_with(tools, text, width, True)),
                                     pieces(run_with(tools, text, width, False)))

    def test_an_untyped_parameter_holding_json_still_streams(self):
        """The gate is on the text, not on the schema: JSON with no declared type still goes out in pieces."""
        tools, text = one_arg_call("payload", {}, '[{"a": 1}, {"b": 2}]')
        for width in (1, 7, 8, 64):
            with self.subTest(width=width):
                got = pieces(run_with(tools, text, width, True))
                self.assertGreater(len(got), 2, "the JSON value did not arrive in pieces")
                self.assertEqual(json.loads("".join(got)), {"payload": [{"a": 1}, {"b": 2}]})

    def test_a_call_cut_inside_a_plain_text_value_is_unchanged(self):
        """The output ended in the middle of a bare word: the gate does not make that worse - the client holds the
        same head it held without the switch (the value never started, so nothing of it went out)."""
        tools, text = one_arg_call("path", {}, "/tmp/some/long/path")
        cut = text.index("/tmp/") + 8
        for width in (1, 8, 1000):
            with self.subTest(width=width):
                evs = run_with(tools, text[:cut], width, True)
                got = pieces(evs)
                self.assertEqual(got, ['{', '"path":'])
                self.assertEqual(got, pieces(run_with(tools, text[:cut], width, False)))
                self.assertIsNone(reported(evs))          # the call never completed


class CutCall(unittest.TestCase):
    """The output ended inside the array: the client gets what was written instead of only the head."""

    def test_the_pieces_written_before_the_cut_have_gone_out(self):
        full = call_text()
        cut = full.index('"target"') + 40
        after = full.index("<parameter=edits>") + len("<parameter=edits>") + 1
        written = full[after:cut].rstrip("\n")
        for width in (1, 8, 64):
            with self.subTest(width=width):
                evs = run(full[:cut], width, True)
                got = "".join(pieces(evs))
                self.assertTrue(got.startswith('{"edits":'))
                self.assertIn(written, got, "the text written before the cut did not reach the client")
                self.assertIsNone(reported(evs))          # the call never completed
                self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])

    def test_without_the_switch_only_the_head_has_gone_out(self):
        full = call_text()
        cut = full.index('"target"') + 40
        for width in (1, 8, 64):
            with self.subTest(width=width):
                self.assertEqual("".join(pieces(run(full[:cut], width, False))), '{"edits":')


if __name__ == "__main__":
    unittest.main(verbosity=2)
