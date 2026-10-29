"""#4071: every `<tool_call>` block that goes back as text is counted and named, fenced or not.

Observed with Qwen3.8-Flash-Next IQ3_S served by Strata 0.2.7 and used through a
Chat Completions client that declares no tools (Open WebUI, asked to "make an SVG and
save it"): the model wrote a complete call for `write_file` inside a ```xml fence.
The undeclared-name rule (#3075) keeps it text, but the fence rule (#1058) passed it
without even incrementing `unread` - so neither the serve warning nor the
strata_rejected_tool_calls marker (#3796) ever mentioned it, and the client could not
tell "the model wrote a call and it was dropped" from "the model chose not to call".

The rule kept: fenced and undeclared blocks STAY text (a quoted example must not
become a call). What changes is visibility - each block counts once, and
`unread_info` says why: "undeclared" (a name the request did not offer), "in_code"
(a call-shaped block for a declared name inside a fence or inline code), or
"malformed" (opened like a call with no `<function=` name to read).

    python -m unittest serve.test_unread_call_marker -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="60" fill="red"/></svg>'
P = "<" + "parameter="   # built at runtime so this source never looks like a call


def call_block(name: str) -> str:
    return ("<tool_call>\n<function=" + name + ">\n"
            + P + "path>\ncircle.svg\n</" + "parameter>"
            + P + "content>\n" + SVG + "\n</" + "parameter>"
            + "</function>\n</tool_call>")


def run(parser: OutputParser, text: str, chunk: int = 0):
    """Feed `text` (whole, or in `chunk`-sized pieces) -> (content, call names, unread, names, info)."""
    events = []
    if chunk:
        for k in range(0, len(text), chunk):
            events += parser.feed(text[k:k + chunk])
    else:
        events += parser.feed(text)
    events += parser.finish()
    return ("".join(e.text for e in events if e.kind == "content"),
            [e.call.name for e in events if e.kind == "tool_call"],
            parser.unread, list(parser.unread_names), list(parser.unread_info))


class FencedBlockIsCounted(unittest.TestCase):
    """The #4071 shape: a complete call inside a ``` fence, request declares no tools."""

    def test_fenced_undeclared_call_is_named_with_a_reason(self):
        content, calls, unread, names, info = run(
            OutputParser(thinking=False, tools=None, declared_only=True),
            "Here it is:\n```xml\n" + call_block("write_file") + "\n```\n")
        self.assertIn("<tool_call>", content)            # stays text, as the model wrote it (#3075 rule kept)
        self.assertEqual(calls, [])                     # never a call: the client has no such tool to run
        self.assertEqual(unread, 1)                     # #4071: counted, where before this was silent
        self.assertEqual(names, ["write_file"])
        self.assertEqual(info, [{"name": "write_file", "reason": "undeclared"}])

    def test_fenced_declared_call_is_in_code(self):
        content, calls, unread, names, info = run(
            OutputParser(thinking=False, tools=[{"name": "write_file"}]),
            "Example:\n```xml\n" + call_block("write_file") + "\n```\n")
        self.assertIn("<tool_call>", content)            # #1058: a quoted example stays text
        self.assertEqual(calls, [])
        self.assertEqual(info, [{"name": "write_file", "reason": "in_code"}])  # the client can still learn it

    def test_inline_code_block_is_counted_once(self):
        _, _, unread, _, info = run(
            OutputParser(thinking=False, tools=None, declared_only=True),
            "The format is `" + call_block("write_file").replace("\n", "") + "` and then prose.")
        self.assertEqual(unread, 1)
        self.assertEqual(info, [{"name": "write_file", "reason": "undeclared"}])

    def test_streamed_counts_match_whole(self):
        for fence in ("```xml\n", "```\n"):
            raw = "Here:\n" + fence + call_block("write_file") + "\n```\n"
            whole = run(OutputParser(thinking=False, tools=None, declared_only=True), raw)
            for chunk in (1, 3, 17):
                streamed = run(OutputParser(thinking=False, tools=None, declared_only=True), raw, chunk)
                with self.subTest(fence=fence, chunk=chunk):
                    self.assertEqual(streamed, whole)   # same content, same calls, same counts


class OtherPathsStillCount(unittest.TestCase):
    """The counts that already worked keep working, now with the reason attached."""

    def test_bare_undeclared_call(self):
        _, calls, unread, names, info = run(
            OutputParser(thinking=False, tools=None, declared_only=True),
            "Sure.\n" + call_block("write_file"))
        self.assertEqual((calls, unread, names), ([], 1, ["write_file"]))
        self.assertEqual(info, [{"name": "write_file", "reason": "undeclared"}])

    def test_declared_call_is_still_a_call(self):
        _, calls, unread, _, info = run(
            OutputParser(thinking=False, tools=[{"name": "write_file"}]),
            call_block("write_file"))
        self.assertEqual(calls, ["write_file"])
        self.assertEqual((unread, info), (0, []))

    def test_prose_naming_the_tag_stays_clean(self):
        _, calls, unread, _, info = run(
            OutputParser(thinking=False, tools=None, declared_only=True),
            "I will use a <tool_call> block to save it.")
        self.assertEqual((calls, unread, info), ([], 0, []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
