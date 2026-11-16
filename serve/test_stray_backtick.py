"""#4923: a stray backtick in the reasoning must not hide the tool call that follows it.

The inline-code test counted backticks across the whole paragraph.  A model that writes one unpaired backtick -
"let me use `run_commands to check" - left that count odd for the rest of the paragraph, so every later
`<tool_call>` read as a quoted example and was kept as reasoning text.  The client then received thinking only,
never a tool_use block, and waited in "thinking" for a call the engine had already generated and thrown away.

The rule these tests pin: the runs of backticks are paired across the paragraph, and a span stays open only while
the run that opened it ENDED its line.  Pairing alone is what lost the call in the first place; carrying every run
that ends a line is too narrow and loses the wrapped `example` whose CLOSING run ends its line - both shapes are
pinned below so neither side can regress silently.
"""
import unittest

from serve.frontend import OutputParser

TOOL = {"name": "run_commands", "parameters": {"type": "object", "properties": {
    "commands": {"type": "array", "items": {"type": "string"}}}, "required": ["commands"]}}
CALL = "<tool_call>\n<function=run_commands>\n<parameter=commands>\n[\"ls\"]\n</parameter>\n</function>\n</tool_call>"


def events(text, thinking=True, tools=(TOOL,)):
    parser = OutputParser(thinking, list(tools))
    out = [e for piece in (text[i:i + 7] for i in range(0, len(text), 7)) for e in parser.feed(piece)]
    return out + parser.finish("stop"), parser


def kinds(out):
    return [e.kind for e in out]


def text_of(out, kind):
    return "".join(e.text or "" for e in out if e.kind == kind)


class StrayBacktickTests(unittest.TestCase):
    def test_call_after_a_stray_backtick_is_a_call(self):
        out, _ = events("Let me use `run_commands to check.\n" + CALL + "\n")
        self.assertIn("tool_call", kinds(out))
        self.assertEqual(out[kinds(out).index("tool_call")].call.name, "run_commands")
        self.assertNotIn("<tool_call>", text_of(out, "reasoning"))

    def test_call_after_a_stray_backtick_on_an_earlier_line(self):
        out, _ = events("I'll check `first.\nThen the second step.\n" + CALL)
        self.assertIn("tool_call", kinds(out))

    def test_two_stray_backticks_do_not_hide_a_call(self):
        """Two strays leave the count even, which is why the old count happened to work here - it must keep working
        for the right reason and not by accident."""
        out, _ = events("the ls` and cat` tools.\n" + CALL)
        self.assertIn("tool_call", kinds(out))

    def test_wrapped_span_closed_at_end_of_line_does_not_hide_a_call(self):
        """The shape the too-narrow rule lost: a wrapped `example whose CLOSING run ends its line.  The span is
        closed when the call arrives, so the call is a call."""
        out, _ = events("a wrapped `example\nthat closes here`\n" + CALL)
        self.assertIn("tool_call", kinds(out))
        self.assertNotIn("<tool_call>", text_of(out, "reasoning"))

    def test_wrapped_span_closed_mid_line_does_not_hide_a_call(self):
        out, _ = events("a wrapped `example\nthat closes here` and some words.\n" + CALL)
        self.assertIn("tool_call", kinds(out))

    def test_ambiguous_open_span_resolves_towards_the_call(self):
        """The one shape the rule cannot settle without lookahead: "a `span that stays open" and the stray
        "use `run_commands to check" are the same thing - one run, not at the end of its line, nothing closing it
        later.  The rule reads both as no open span and delivers the call, which is the cheaper error for an agentic
        client: a kept-as-text call ends the turn with nothing to run and the client waits forever."""
        out, _ = events("a `span that stays open\nand more text\n" + CALL + "\n")
        self.assertIn("tool_call", kinds(out))

    def test_span_opened_at_end_of_line_hides_a_call_on_that_line(self):
        """The other half of the same rule: a run that ENDS its line does open a span, and a call written on the
        next line is inside it - a quoted example, kept as text."""
        out, _ = events("look at this `\n" + CALL + "\n")
        self.assertNotIn("tool_call", kinds(out))
        self.assertIn("<tool_call>", text_of(out, "reasoning"))

    def test_span_opened_on_one_line_closed_on_the_next_is_a_call(self):
        out, _ = events("a `span that stays\nopen until here`\n" + CALL)
        self.assertIn("tool_call", kinds(out))

    def test_thinking_that_ends_on_a_backtick_is_flushed(self):
        out, parser = events("...summarize my thoughts... `")
        self.assertEqual(text_of(out, "reasoning"), "...summarize my thoughts... `")
        self.assertEqual(parser.buf, "")

    def test_quoted_call_stays_text(self):
        out, _ = events("The format is `<tool_call>` and then a line.\n")
        self.assertNotIn("tool_call", kinds(out))
        self.assertIn("`<tool_call>`", text_of(out, "reasoning"))

    def test_call_inside_a_fence_stays_text(self):
        out, _ = events("```xml\n" + CALL + "\n```\n")
        self.assertNotIn("tool_call", kinds(out))
        self.assertIn("<tool_call>", text_of(out, "reasoning"))

    def test_fence_closes_and_a_later_call_is_a_call(self):
        out, _ = events("```xml\n" + CALL + "\n```\n" + CALL)
        self.assertEqual(kinds(out).count("tool_call"), 1)

    def test_answer_state_inline_code_still_quotes(self):
        out, _ = events("The format is `<tool_call>` here.\n", thinking=False)
        self.assertNotIn("tool_call", kinds(out))

    def test_answer_state_stray_backtick_does_not_hide_a_call(self):
        out, _ = events("I'll use `run_commands now.\n" + CALL, thinking=False)
        self.assertIn("tool_call", kinds(out))

    def test_streamed_and_whole_outputs_agree(self):
        """Whatever the piece boundaries, the client sees the same thinking, the same answer, and the same call."""
        def delivered(text, size):
            parser = OutputParser(True, [TOOL])
            out = [e for i in range(0, len(text), size) for e in parser.feed(text[i:i + size])]
            out += parser.finish("stop")
            calls = [(c.call.name, c.call.arguments) for c in out if c.kind == "tool_call"]
            return {k: text_of(out, k) for k in ("reasoning", "content")}, calls

        # The call is the last thing the thinking holds: it is an act, not a quote (#804).
        text = "Let me use `run_commands to check.\n" + CALL + "\n"
        whole = delivered(text, len(text))
        self.assertEqual(whole[1], [("run_commands", {"commands": ["ls"]})])
        for size in (1, 2, 3, 5, 11, 97):
            self.assertEqual(delivered(text, size), whole, f"{size}-char pieces")

        # A call the thinking writes and then talks on over is a quote by design - and reads the same way
        # at every piece size, backtick or not.
        quoted = "Let me use `run_commands to check.\n" + CALL + "\nand that is the format.\n"
        whole = delivered(quoted, len(quoted))
        self.assertEqual(whole[1], [])
        for size in (1, 3, 11):
            self.assertEqual(delivered(quoted, size), whole, f"{size}-char pieces")


if __name__ == "__main__":
    unittest.main()
