"""serve/test_code_span_quotes.py - #<issue>: the code-span rule must guard the think pair on the reasoning side.

Tag spellings are IMPORTED from serve.frontend (never hand-typed: the literals in this file must
byte-match the parser's constants).  The parser is called the way the server calls it
(serve/server.py): OutputParser(thinking=True, tools=..., stream_tools=True, recover=False),
feed(delta) per chunk, finish(None) at the end of the turn.

    python -m unittest serve.test_code_span_quotes -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START,  # noqa: E402
                            PARAM_END, PARAM_START, THINK_END, OutputParser)

SCHEMA = [{"name": "noop", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}]
VALUE = "plain value"


def call_text(value: str = VALUE) -> str:
    return (CALL_START + "\n" + FUNC_START + "noop>\n" + PARAM_START + "x>\n" + value + "\n"
            + PARAM_END + "\n" + FUNC_END + "\n" + CALL_END + "\n")


def parse(chunks: list[str]) -> list:
    p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, recover=False)
    evs: list = []
    for c in chunks:
        evs += p.feed(c)
    evs += p.finish(None)
    return evs


def calls(evs) -> list:
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def visible(evs, kind: str) -> str:
    return "".join(e.text for e in evs if e.kind == kind)


# --------------------------------------------------------------------------------- G: recover on (the config path)
# The parser is built exactly as serve/server.py builds it when the config sets tool_call_recovery true:
# OutputParser(thinking=True, tools=..., stream_tools=True, recover=True).

def parse_rec(chunks: list[str]) -> list:
    p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, recover=True)
    evs: list = []
    for c in chunks:
        evs += p.feed(c)
    evs += p.finish(None)
    return evs


class WellFormedCallWholeOneFeed(unittest.TestCase):
    """A well-formed call, whole, one feed() -> exactly one tool_call, value intact, no text."""

    def test_whole_call_one_feed(self):
        evs = parse([call_text()])
        self.assertEqual(calls(evs), [("noop", {"x": VALUE})])
        self.assertEqual(visible(evs, "content") + visible(evs, "reasoning"), "")


class WellFormedCallSplitAcrossChunks(unittest.TestCase):
    """The same call split across chunks -> the same expectation."""

    def test_same_call_split_across_chunks(self):
        text = call_text()
        evs = parse([text[:6], text[6:30], text[30:55], text[55:]])
        self.assertEqual(calls(evs), [("noop", {"x": VALUE})])
        self.assertEqual(visible(evs, "content") + visible(evs, "reasoning"), "")


class LiteralOpenTagInsideParameterValue(unittest.TestCase):
    """A call with the literal opener inside a parameter value -> exactly one tool_call, the
    value intact (#210 rule), nothing emitted twice."""

    def test_open_tag_inside_value(self):
        value = "value with " + CALL_START + " inside"
        evs = parse([call_text(value)])
        self.assertEqual(calls(evs), [("noop", {"x": value})])
        self.assertEqual(visible(evs, "content") + visible(evs, "reasoning"), "")


class OpenTagAtLineStartInProse(unittest.TestCase):
    """The opener at a line start in reasoning, ordinary prose after it, then the thinking
    close, then the answer -> the quote stays reasoning text, the answer arrives as content, each
    once."""

    def test_line_start_quote_stays_reasoning(self):
        prose = CALL_START + " opens the format, but this line is prose."
        evs = parse([prose + "\n" + THINK_END + "\n\nThe answer."])
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "reasoning"), prose + "\n")
        self.assertEqual(visible(evs, "content"), "The answer.")
        self.assertEqual(len([e for e in evs if e.kind == "reasoning"]), 1)
        self.assertEqual(len([e for e in evs if e.kind == "content"]), 1)


class QuotedOpenTagMidSentence(unittest.TestCase):
    """Control: prose containing the opener mid-sentence in content -> stays content, once."""

    def test_mid_sentence_quote_stays_content(self):
        sentence = "I will use the " + CALL_START + " tag now. Done."
        evs = parse([THINK_END + "\n\n" + sentence])
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "content"), sentence)
        self.assertEqual(visible(evs, "reasoning"), "")


class RecoverOpenTagMidSentence(unittest.TestCase):
    """Recover on - a sentence containing the open-tag spelling mid-text, more text, end of stream:
    ALL of it comes out as content, no call event."""

    def test_all_text_comes_out_as_content(self):
        sentence = "First part. " + CALL_START + " tail text. The end."
        evs = parse_rec([THINK_END + "\n\n" + sentence])
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "content"), sentence)


class RecoverBareOpenTagAlone(unittest.TestCase):
    """Recover on - the bare open-tag spelling alone -> text out, no call event."""

    def test_bare_open_tag_is_text(self):
        evs = parse_rec([THINK_END + "\n\n" + CALL_START])
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "content"), CALL_START)


class RecoverWellFormedCallStillACall(unittest.TestCase):
    """Recover on - a full well-formed call is still a call event (documented ceiling; no regression)."""

    def test_full_call_still_a_call(self):
        evs = parse_rec([THINK_END + "\n\n" + call_text()])
        self.assertEqual(calls(evs), [("noop", {"x": VALUE})])


class ThinkCloseInsideInlineCodeStaysReasoning(unittest.TestCase):
    """#1058's rule on the reasoning side: a quoted close-tag spelling inside an inline code span is
    text, not the end of the thinking: all of it stays on one channel, nothing is lost."""

    def test_close_spelling_inside_inline_code_stays_reasoning(self):
        chunks = ["Thinking about it, `" + THINK_END, "` is text, ", "and so is the rest."]
        evs = parse(chunks)
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "content"), "")
        self.assertEqual(visible(evs, "reasoning"), "".join(chunks))


class ThinkCloseInsideFenceStaysReasoning(unittest.TestCase):
    """The same rule inside a fenced block: a quoted close-tag spelling on a line of the fence is
    text; the thinking does not end there."""

    def test_close_spelling_inside_fence_stays_reasoning(self):
        chunks = ["Plan:\n", "```\n", "close " + THINK_END + " here\n", "```\n", "still thinking.\n"]
        evs = parse(chunks)
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "content"), "")
        self.assertEqual(visible(evs, "reasoning"), "".join(chunks))


class PlainThinkCloseStillCloses(unittest.TestCase):
    """Guard for the normal path: a close-tag spelling outside code still ends the thinking."""

    def test_plain_close_closes(self):
        evs = parse(["Thinking.\n" + THINK_END + "\n\nThe answer."])
        self.assertEqual(calls(evs), [])
        self.assertEqual(visible(evs, "reasoning"), "Thinking.\n")
        self.assertEqual(visible(evs, "content"), "The answer.")


class ThinkQuoteBoundaries(unittest.TestCase):
    """Require BOTH preserved reasoning and a separate final answer at every two-chunk split."""

    def assert_reply(self, reasoning, *, recover=False, reason="stop"):
        raw = reasoning + THINK_END + "\n\nOK"
        chunkings = [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(len(raw) + 1)]
        for chunks in chunkings:
            with self.subTest(chunks=len(chunks), split=len(chunks[0]) if chunks else 0):
                parser = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, recover=recover)
                evs = [e for chunk in chunks for e in parser.feed(chunk)] + parser.finish(reason)
                self.assertEqual(visible(evs, "reasoning"), reasoning)
                self.assertEqual(visible(evs, "content"), "OK")
                self.assertEqual(calls(evs), [])

    def test_single_and_multiple_backtick_runs(self):
        for width in (1, 2, 3, 4):
            for recover in (False, True):
                with self.subTest(width=width, recover=recover):
                    ticks = "`" * width
                    self.assert_reply("Quote " + ticks + THINK_END + ticks + " is text.\n", recover=recover)

    def test_backtick_inside_double_backtick_span(self):
        self.assert_reply("Quote ``a ` " + THINK_END + " b`` is text.\n")

    def test_escaped_backtick_does_not_hide_actual_close(self):
        self.assert_reply("Escaped \\` character.\n")

    def test_even_backslashes_leave_a_real_code_opener(self):
        self.assert_reply("Quote \\\\`" + THINK_END + "` is text.\n")

    def test_backslash_inside_inline_code_does_not_escape_its_end(self):
        self.assert_reply("Quote `" + THINK_END + "\\` followed by reasoning.\n")

    def test_unmatched_inline_does_not_hide_final_answer(self):
        self.assert_reply("An unfinished `example\n")

    def test_unmatched_fence_does_not_hide_final_answer(self):
        for opener in ("```python", "````python", "~~~python"):
            for reason in ("stop", "length", "cancel"):
                with self.subTest(opener=opener, reason=reason):
                    self.assert_reply("Plan:\n" + opener + "\nx = 1\n", reason=reason)

    def test_cut_undecided_quote_does_not_promote_held_calls(self):
        body = "Answer text\n<tool_call>\n<function=noop>\n</function>\n</tool_call>\n"
        raw = "Plan:\n```xml\n" + THINK_END + "\n\n" + body
        for reason in ("length", "cancel", "error", "disconnect"):
            for width in (1, 7, len(raw)):
                with self.subTest(reason=reason, width=width):
                    parser = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, recover=True)
                    events = [e for i in range(0, len(raw), width) for e in parser.feed(raw[i:i + width])]
                    self.assertFalse(any(e.kind.startswith("tool_") for e in events))
                    events += parser.finish(reason)
                    self.assertFalse(any(e.kind.startswith("tool_") for e in events))
                    self.assertEqual(visible(events, "content"), body)

    def test_natural_stop_still_settles_held_call(self):
        raw = "Plan:\n```xml\n" + THINK_END + "\n\n<tool_call>\n<function=noop>\n</function>\n</tool_call>"
        for reason in (None, "stop"):
            parser = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, recover=True)
            events = parser.feed(raw) + parser.finish(reason)
            self.assertEqual(calls(events), [("noop", {})])

    def test_committed_boundary_still_allows_complete_call_at_length(self):
        parser = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True, recover=True)
        raw = "Ready.\n" + THINK_END + "\n\n<tool_call>\n<function=noop>\n</function>\n</tool_call>"
        events = parser.feed(raw) + parser.finish("length")
        self.assertEqual(calls(events), [("noop", {})])

    def test_cut_quote_does_not_create_calls_on_all_http_apis(self):
        from serve.server import ByteTokenizer
        from serve.test_reasoning_rescue import CALL, OverHttp
        text = "Plan:\n```xml\n" + THINK_END + "\n\nOK\n" + CALL
        uses = OverHttp().ask(text + "more", max_tokens=len(ByteTokenizer().encode(text, parse_special=True)))
        self.assertEqual(set(uses.values()), {0}, uses)

    def test_closed_fences_preserve_quotes_and_final_answer(self):
        for char, width in (("`", 3), ("`", 4), ("~", 3), ("~", 5)):
            with self.subTest(char=char, width=width):
                fence = char * width
                self.assert_reply("Plan:\n" + fence + "python\nquote " + THINK_END + " here\n" + fence + "\nDone.\n")

    def test_shorter_fence_does_not_close_a_longer_fence(self):
        self.assert_reply("Plan:\n````\nquote " + THINK_END + "\n```\nstill code " + THINK_END + "\n````\nDone.\n")

    def test_crlf_fences_preserve_quotes(self):
        self.assert_reply("Plan:\r\n```python\r\nquote " + THINK_END + "\r\n```\r\nDone.\r\n")

    def test_fence_with_trailing_prose_does_not_close(self):
        self.assert_reply("Plan:\n```\nquote " + THINK_END + "\n``` not a close\nstill code\n```\nDone.\n")

    def test_multiple_quotes_are_delivered_once(self):
        self.assert_reply("Quote ``" + THINK_END + " then " + THINK_END + "`` is text.\n")

    def test_multi_line_inline_code(self):
        self.assert_reply("Quote `first\n" + THINK_END + " last` is text.\n")

    def test_pending_quote_prefix_is_not_duplicated(self):
        parser = OutputParser()
        evs = parser.feed("Prefix `" + THINK_END)
        self.assertEqual(visible(evs, "reasoning"), "Prefix `")
        evs += parser.feed("` is text.\n" + THINK_END + "\n\nOK")
        evs += parser.finish("stop")
        self.assertEqual(visible(evs, "reasoning"), "Prefix `" + THINK_END + "` is text.\n")
        self.assertEqual(visible(evs, "content"), "OK")

    def test_completed_inline_quote_at_end_of_stream_stays_reasoning(self):
        raw = "Quote ``" + THINK_END + "``"
        for reason in ("stop", "length", "cancel"):
            parser = OutputParser()
            evs = parser.feed(raw) + parser.finish(reason)
            self.assertEqual(visible(evs, "reasoning"), raw)
            self.assertEqual(visible(evs, "content"), "")

    def test_closed_fence_at_end_of_stream_stays_reasoning(self):
        raw = "Plan:\n```\n" + THINK_END + "\n```"
        parser = OutputParser()
        evs = parser.feed(raw) + parser.finish("stop")
        self.assertEqual(visible(evs, "reasoning"), raw)
        self.assertEqual(visible(evs, "content"), "")

    def test_anthropic_has_separate_thinking_and_text_blocks(self):
        from serve.server import anthropic_collect, anthropic_events
        raw = "Plan:\n```python\nx = 1\n" + THINK_END + "\n\nOK"
        parser = OutputParser()
        evs = parser.feed(raw) + parser.finish("stop")
        class ScriptedService:
            def model_for(self, req):
                return "test"
            def run(self, *args, **kwargs):
                yield from (("event", ev) for ev in evs)
                yield "done", {"finish": "stop", "completion_tokens": len(raw)}
        result = anthropic_collect(anthropic_events(ScriptedService(), {}, [], True, [], 1000, None))
        self.assertEqual([b["type"] for b in result["content"]], ["thinking", "text"])
        self.assertEqual(result["content"][1]["text"], "OK")
        self.assertNotIn(THINK_END, result["content"][0]["thinking"])

    def test_service_delivers_special_and_plain_token_quotes(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        class Tokenizer(ByteTokenizer):
            SPECIALS = ByteTokenizer.SPECIALS + [THINK_END]
            ALWAYS = (THINK_END,)
        for tok in (ByteTokenizer(), Tokenizer()):
            raw = "Quote ``" + THINK_END + "`` stays reasoning.\n" + THINK_END + "\n\nOK"
            svc = Service(MockEngine(tok, raw), tok, None)
            events = list(svc.run([], True, [], 1000, {}, threading.Event()))
            evs = [ev for kind, ev in events if kind == "event"]
            self.assertEqual(visible(evs, "content"), "OK")
            self.assertEqual(visible(evs, "reasoning"), "Quote ``" + THINK_END + "`` stays reasoning.\n")

    def test_replay_real_thinking_only_response_with_escaped_backtick(self):
        reasoning = ('The user wants me to write exactly "Escaped \\` character." in my thinking, '
                     'without adding any other backticks or explanation. Then my final answer must be exactly "OK".\n')
        answer = "Escaped \\` character.\n\nOK"
        raw = reasoning + THINK_END + "\n\n" + answer
        parser = OutputParser()
        evs = [e for char in raw for e in parser.feed(char)] + parser.finish("stop")
        self.assertEqual(visible(evs, "reasoning"), reasoning)
        self.assertEqual(visible(evs, "content"), answer)

    def test_replay_real_thinking_only_response_with_unfinished_fence(self):
        reasoning = ("Let me be careful here. The thinking should contain exactly:\nPlan:\n```python\nx = 1\n\n"
                     "And then end the thinking. The final answer must be exactly OK.\n")
        self.assert_reply(reasoning)

    def test_large_fence_quote_stream_is_complete(self):
        reasoning = "Plan:\n```\n" + THINK_END + "\n" + "unchanged code line\n" * 5000 + "```\nDone.\n"
        raw = reasoning + THINK_END + "\n\nOK"
        parser = OutputParser()
        evs = [e for i in range(0, len(raw), 23) for e in parser.feed(raw[i:i + 23])] + parser.finish("stop")
        self.assertEqual(visible(evs, "reasoning"), reasoning)
        self.assertEqual(visible(evs, "content"), "OK")



class SharedCodeContext(unittest.TestCase):
    def test_escape_consumes_one_backtick_not_the_entire_opening_run(self):
        # CommonMark 6.1: an escaped tick is text; the remaining ticks can open code.
        for slashes in (1, 3):
            for width in (1, 2, 3):
                thought = "Example: " + "\\" * slashes + "`" * (width + 1) + THINK_END + "`" * width + " stays reasoning.\n"
                raw = thought + THINK_END + "\n\nOK"
                for recover in (False, True):
                    for chunks in ([raw], list(raw), *([raw[:i], raw[i:]] for i in range(len(raw) + 1))):
                        with self.subTest(slashes=slashes, width=width, recover=recover, first=len(chunks[0])):
                            parser = OutputParser(thinking=True, tools=SCHEMA, recover=recover)
                            evs = [ev for chunk in chunks for ev in parser.feed(chunk)] + parser.finish("stop")
                            self.assertEqual(visible(evs, "reasoning"), thought)
                            self.assertEqual(visible(evs, "content"), "OK")

    def test_escaped_first_tick_cannot_turn_a_quoted_call_into_an_action(self):
        for width in (1, 2, 3):
            self.check_quote("Example: \\" + "`" * (width + 1) + "{call}" + "`" * width)

    def test_real_call_after_escaped_tick_quote_is_delivered(self):
        quoted = "Example: \\" + "``" + call_text().rstrip("\n") + "`\n"
        for thinking in (False, True):
            for recover in (False, True):
                raw = quoted + call_text()
                for chunks in ([raw], list(raw)):
                    parser = OutputParser(thinking=thinking, tools=SCHEMA, stream_tools=True, recover=recover)
                    evs = [ev for chunk in chunks for ev in parser.feed(chunk)] + parser.finish("stop")
                    self.assertEqual(calls(evs), [("noop", {"x": VALUE})])
                    self.assertEqual(visible(evs, "reasoning" if thinking else "content"),
                                     quoted if thinking else quoted.rstrip("\n"))

    def check_quote(self, quote):
        raw_call = call_text().rstrip("\n")
        quoted = quote.format(call=raw_call) + "\nDone."
        for thinking in (False, True):
            for recover in (False, True):
                # Whole, character-by-character and every two-chunk split exercise delimiter boundaries.
                for chunks in ([quoted], list(quoted), *([quoted[:i], quoted[i:]] for i in range(len(quoted) + 1))):
                    with self.subTest(thinking=thinking, recover=recover, chunks=len(chunks), first=len(chunks[0])):
                        parser = OutputParser(thinking=thinking, tools=SCHEMA, stream_tools=True, recover=recover)
                        evs = [ev for chunk in chunks for ev in parser.feed(chunk)] + parser.finish("stop")
                        self.assertFalse(any(ev.kind.startswith("tool_") for ev in evs))
                        self.assertEqual(visible(evs, "reasoning" if thinking else "content"), quoted)

    def test_tool_call_inside_matching_backtick_runs_is_text(self):
        for width in (1, 2, 3, 4):
            ticks = "`" * width
            self.check_quote("Example: " + ticks + "{call}" + ticks)

    def test_tool_call_inside_long_fences_is_text(self):
        for fence in ("````", "~~~~"):
            self.check_quote("Plan:\n" + fence + "\nshorter ```\n{call}\n" + fence)

    def test_real_call_after_double_backtick_quote_is_delivered(self):
        quoted = "Example: ``" + call_text().rstrip("\n") + "``\n"
        for thinking in (False, True):
            for recover in (False, True):
                raw = quoted + call_text()
                for chunks in ([raw], list(raw)):
                    parser = OutputParser(thinking=thinking, tools=SCHEMA, stream_tools=True, recover=recover)
                    evs = [ev for chunk in chunks for ev in parser.feed(chunk)] + parser.finish("stop")
                    self.assertEqual(calls(evs), [("noop", {"x": VALUE})])
                    self.assertEqual(visible(evs, "reasoning" if thinking else "content"), quoted.rstrip("\n") if not thinking else quoted)

    def test_repeated_same_line_think_quotes_scan_bounded_text(self):
        from unittest.mock import patch
        import serve.frontend as frontend
        for count in (64, 128, 256, 512):
            reasoning = ("Quote `" + THINK_END + "` stays text. ") * count + "\n"
            raw = reasoning + THINK_END + "\n\nOK"
            measured = [0]
            original = frontend._CodeContext.feed
            def scan(instance, text):
                measured[0] += len(text)
                return original(instance, text)
            instrumentation = patch.object(frontend._CodeContext, "feed", scan)
            parser = OutputParser()
            with instrumentation:
                evs = [ev for offset in range(0, len(raw), 23) for ev in parser.feed(raw[offset:offset + 23])]
                evs += parser.finish("stop")
            self.assertEqual(visible(evs, "reasoning"), reasoning)
            self.assertEqual(visible(evs, "content"), "OK")
            self.assertLessEqual(measured[0], 3 * len(raw), (count, measured[0], len(raw)))

class ServerOwnedThinkingBoundary(unittest.TestCase):
    def run_service(self, thought, *, budget=0, answer="```\nOK", sampling=None):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        engine = MockEngine(tok, [thought, answer])
        svc = Service(engine, tok, None)
        events = list(svc.run([], True, [], 2000, {"reasoning_budget_tokens": budget, **(sampling or {})}, threading.Event()))
        return engine, [ev for kind, ev in events if kind == "event"], dict(events[-1][1])

    def test_replay_live_budget_cut_inside_open_python_fence(self):
        # 2026-10-10 direct backend, reasoning_budget_tokens=155: the model's
        # incomplete reasoning example was closed by the final answer's fence.
        thought = "The final answer should be exactly:\n```python\nprint("
        answer = "```python\nprint(4)\n```"
        engine, evs, done = self.run_service(thought, budget=len(thought), answer=answer)
        self.assertEqual(engine.turns, 2)
        self.assertEqual(visible(evs, "content"), answer)

    def test_pending_quote_does_not_disable_the_budget(self):
        thought = "Plan:\n```\nquoted " + THINK_END + "\n" + "x" * 200
        engine, evs, done = self.run_service(thought, budget=60)
        self.assertEqual(engine.turns, 2)
        self.assertEqual(visible(evs, "content"), "```\nOK")
        self.assertIn(THINK_END, visible(evs, "reasoning"))

    def test_explicit_boundary_keeps_quoted_model_text(self):
        parser = OutputParser()
        raw = "Plan:\n```\nquoted " + THINK_END + "\nstill code\n"
        evs = parser.feed(raw)
        evs += parser.feed("\n" + THINK_END + "\n\n", protocol=True)
        evs += parser.feed("```\nOK") + parser.finish("stop")
        self.assertEqual(visible(evs, "reasoning"), raw + "\n")
        self.assertEqual(visible(evs, "content"), "```\nOK")



if __name__ == "__main__":
    unittest.main()
