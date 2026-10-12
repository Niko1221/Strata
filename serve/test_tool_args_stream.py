"""serve/test_tool_args_stream.py - "stream_all_tool_args": a tool call's non-string argument in pieces.

By default a parameter that is not a string (an array, an object, a number) is held until it ends and goes out as one
"tool_args" piece.  For an `edits`-style argument - an array of objects - that leaves the stream quiet for minutes
while the model writes it, and the app shows nothing of it until it ends.  With the switch the value goes out as the
JSON text the model writes, in pieces.  A client concatenates the pieces either way (the final "tool_call" of a call
that was streamed is not sent again), so what it parses is the same arguments: every test here adds the pieces back
and compares them with the call the parser reports.

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
    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, stream_all_args=stream_all_args)
    evs = []
    for i in range(0, len(text), width):
        evs += p.feed(text[i:i + width])
    return evs + p.finish()


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
