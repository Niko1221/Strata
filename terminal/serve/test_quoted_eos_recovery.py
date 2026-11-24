"""serve/test_quoted_eos_recovery.py - #5839: an end token the model writes out as text must not end the turn.

Qwen3.8-Flash-Next writes `<|im_end|>` when it means to mention it - inside a code example, right after "is" or 是, or as
the first word of its answer - and the turn ended there, with its reasoning only or with no answer at all.  The opt-in
"quoted_eos_recovery" keeps such a token as its ordinary spelling and writes on, and reads a `</think>` the reasoning
quotes as reasoning text.  With it off nothing changes.  The helpers here are shared with serve/test_think_eos_ban.py.

    python -m unittest serve.test_quoted_eos_recovery -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import test_server  # noqa: E402  (a module: no test of it is collected here)
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IM_END = "<|im_end|>"
CALL = '<tool_call>\n<function=probe>\n<parameter=step>\n2\n</parameter>\n</function>\n</tool_call>'
SCHEMA = [{"name": "probe", "parameters": {"type": "object", "properties": {"step": {"type": "integer"}}}}]


class SessionEngine(MockEngine):
    """Replays one token list over however many passes the server needs, each from where the last one stopped (an engine
    continuing the prefix it holds), then the end-of-turn token.  Records every prompt and every sampling it is given."""
    info = {}

    def __init__(self, tokenizer, text):
        super().__init__(tokenizer, "")
        self.ids = tokenizer.encode(text, parse_special=True) + tokenizer.encode(IM_END, parse_special=True)
        self.pos, self.samplings, self.prompts = 0, [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.turns += 1
        self.samplings.append(dict(sampling or {}))
        self.prompts.append(list(ids))
        # (set before the tokens: the service closes a pass at its end-of-turn token, before the code after the loop)
        self.last = {"prompt_tokens": len(ids), "reused": 0, "finish": "length"}
        for t in self.ids[self.pos:self.pos + max_new]:
            self.pos += 1
            if cancel.is_set():
                return
            yield t


def service_for(engine, tokenizer, **switches):
    """A Service on `engine` with the named switches set (the server's defaults for the rest)."""
    service = Service(engine, tokenizer, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    for name, value in switches.items():
        setattr(service, name, value)
    return service


def run(service, tokenizer, thinking=True, tools=None, cap=2000, sampling=None, force=None, cancel=None):
    """Service.run's events as {reasoning, content, calls, done}; the server's log lines are not printed."""
    with contextlib.redirect_stdout(io.StringIO()):
        events = list(service.run(tokenizer.encode("hi"), thinking, tools, cap, dict(sampling or {}),
                                  cancel or threading.Event(), force=force))

    def text(kind):
        return "".join(e.text or "" for tag, e in events if tag == "event" and e.kind == kind)
    return {"reasoning": text("reasoning"), "content": text("content"),
            "calls": [e.call for tag, e in events if tag == "event" and e.kind == "tool_call"],
            "done": events[-1][1]}


class QuotationContext(unittest.TestCase):
    """OutputParser: where an end token is a quoted marker, and where a quoted </think> is reasoning text."""

    def test_open_and_closed_examples(self):
        for text, expected in [
            ('Compare `', True), ('Compare `x`', False), ('Use "', True), ('Use "x"', False),
            ('Use "a\\"b', True), ('Use "a\\\\"', False),
            ('```json\n{"raw":"', True), ('```\nOK\n```', False),
            ('~~~\nOK\n~~~', False), ('```\nOK\n```\n', False),
            ('The result is 4.', False),
        ]:
            with self.subTest(text=text):
                parser = OutputParser(thinking=True)
                parser.feed(text)
                self.assertEqual(parser.quoting_control_token(), expected)

    def test_unfinished_tool_arguments_keep_their_existing_semantics(self):
        parser = OutputParser(thinking=False, tools=SCHEMA, stream_tools=True)
        parser.feed('<tool_call>\n<function=probe>\n<parameter=step>\n"')
        self.assertFalse(parser.quoting_control_token())

    def test_opt_in_requires_a_line_break_after_the_reasoning_end_tag(self):
        parser = OutputParser(thinking=True)
        parser.line_think_end = True
        events = parser.feed('The tag </think> is quoted prose.\n</think>\n\nOK') + parser.finish('stop')
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'reasoning'),
                         'The tag </think> is quoted prose.\n')
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'content'), 'OK')

    def test_a_multiline_quoted_think_tag_keeps_the_reasoning_open(self):
        parser = OutputParser(thinking=True)
        parser.line_think_end = True
        thought = 'Example `<|im_start|>think\n</think>\nend` remains quoted.\n'
        events = parser.feed(thought + '</think>\n\nOK') + parser.finish('stop')
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'reasoning'), thought)
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'content'), 'OK')

    def test_an_unfenced_json_string_keeps_its_markers_as_reasoning(self):
        parser = OutputParser(thinking=True)
        parser.line_think_end = True
        thought = '{\n  "raw": "Assistant: ...\n</think>\n<|im_end|>"\n}\n'
        events = parser.feed(thought + '</think>\n\nOK') + parser.finish('stop')
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'reasoning'), thought)
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'content'), 'OK')

    def test_a_think_marker_on_its_own_line_in_an_answer_code_block_is_preserved(self):
        parser = OutputParser(thinking=True)
        parser.line_think_end = True
        answer = '```text\n</think>\n```\nOK'
        events = parser.feed('Done.\n</think>\n\n' + answer) + parser.finish('stop')
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'content'), answer)

    def test_without_the_switch_the_parser_reads_as_before(self):
        parser = OutputParser(thinking=True)
        self.assertFalse(parser.line_think_end)
        events = parser.feed('The tag </think> is quoted prose.\n</think>\n\nOK') + parser.finish('stop')
        self.assertEqual(''.join(e.text or '' for e in events if e.kind == 'reasoning'), 'The tag ')   # the first bare tag ends the thinking


class LeadIn(unittest.TestCase):
    """OutputParser.mention_lead_in: the words that introduce a marker, and no others."""

    def lead_in(self, text):
        parser = OutputParser(thinking=False)
        parser.feed(text)
        return parser.mention_lead_in()

    def test_the_introductions_are_named(self):
        for text in ("The end token is", "ChatML uses", "also called", "wrapped with", "known as", "the marker", "e.g.",
                     "output:", "For example:", "A, B,", "原样拼写：", "结束标记是", "例如", "直接输出", "称为", "这里使用",
                     "即", "以及", "总结如下：", "它就是", "模型遇到", "包含", "改成", "遇到", "换成", "到这里结束于"):
            with self.subTest(text=text):
                self.assertTrue(self.lead_in(text), text)

    def test_the_words_an_answer_may_end_on_are_not(self):
        for text in ("", "The answer is 42", "9.9 更大。", "Done", "北京", "I think yes", "whatever you like", "that is why",
                     "- item one", "Key: value", "```", "注意言行举止和行为", "这体现了它的作用", "该功能不可用", "这样做可以",
                     "这是因为", "我认为", "可以正常运行", "结束。", "x =>", "a = b", "，", "—",
                     IM_END, "<|endoftext|>"):
            with self.subTest(text=text):
                self.assertFalse(self.lead_in(text), text)

    def test_a_marker_on_its_own_line_reads_the_line_before(self):
        parser = OutputParser(thinking=False)
        parser.feed("原样拼写：\n")
        self.assertTrue(parser.mention_lead_in())          # the answer's line break is held back with the line
        parser = OutputParser(thinking=True)
        parser.feed("原样拼写：\n")
        self.assertTrue(parser.mention_lead_in())          # in the thinking the line before is kept
        parser = OutputParser(thinking=False)
        parser.feed("第一行。\n\n")
        self.assertFalse(parser.mention_lead_in())

    def test_where_the_answer_starts_nothing_of_the_thinking_is_carried_over(self):
        parser = OutputParser(thinking=True)
        parser.line_think_end = True
        parser.feed("The list:\n")
        self.assertTrue(parser.mention_lead_in())           # still in the thinking: the line before counts
        parser.feed("</think>\n\n")
        self.assertFalse(parser.mention_lead_in())          # the answer started: the thinking's line is gone


class Recovery(unittest.TestCase):
    """Service.run with "quoted_eos_recovery" on: the marker is kept as text and the same reply goes on."""

    def reply(self, scripts, *, on=True, thinking=True, cap=1000, sampling=None, cancel=None, tools=None):
        tokenizer = ByteTokenizer()
        engine = MockEngine(tokenizer, scripts)
        service = service_for(engine, tokenizer, quoted_eos_recovery=on)
        return tokenizer, engine, run(service, tokenizer, thinking, tools, cap, sampling, cancel=cancel)

    def test_a_quoted_end_token_in_reasoning_resumes_the_same_reply(self):
        for marker in (IM_END, "<|endoftext|>"):
            with self.subTest(marker=marker):
                tokenizer, engine, r = self.reply(['Compare `' + marker, '` carefully.</think>\n\nOK'])
                self.assertEqual(r["reasoning"], 'Compare `' + marker + '` carefully.')
                self.assertEqual(r["content"], 'OK')
                self.assertEqual(engine.turns, 2)
                literal = tokenizer.encode(marker)         # the next pass's prompt ends with the marker's ordinary spelling
                self.assertEqual(engine.last_prompt[-len(literal):], literal)
                self.assertEqual(r["done"]["quoted_eos_recoveries"], 1)

    def test_in_the_thinking_a_marker_after_an_introduction_is_kept_too(self):
        tokenizer, engine, r = self.reply(["Actually ChatML uses " + IM_END, " to end a message.\n</think>\n\nOK"])
        self.assertEqual(r["reasoning"], "Actually ChatML uses " + IM_END + " to end a message.\n")
        self.assertEqual(r["content"], 'OK')
        self.assertEqual(engine.turns, 2)

    def test_fenced_json_content_is_complete(self):
        _, engine, r = self.reply(['```json\n{"raw":"', '"}\n```\nOK'], thinking=False)
        self.assertEqual(r["content"], '```json\n{"raw":"' + IM_END + '"}\n```\nOK')
        self.assertEqual(engine.turns, 2)

    def test_an_answer_that_starts_with_the_marker_is_not_empty(self):
        _, engine, r = self.reply(['', ' is the end token.'], thinking=False)
        self.assertEqual(r["content"], IM_END + ' is the end token.')
        self.assertEqual((r["done"]["finish"], engine.turns), ("stop", 2))
        _, engine, r = self.reply(['', ''], thinking=False)   # nothing but the marker: the turn ends after it
        self.assertEqual((r["content"], r["done"]["finish"], engine.turns), (IM_END, "stop", 2))

    def test_the_following_tool_call_runs_once(self):
        _, engine, r = self.reply(['Compare `', '`.</think>\n\n' + CALL], tools=SCHEMA)
        self.assertEqual([(c.name, c.arguments) for c in r["calls"]], [("probe", {"step": 2})])
        self.assertEqual(engine.turns, 2)

    def test_an_end_token_right_after_a_kept_one_ends_the_turn(self):
        """An unclosed quote must not keep the model writing markers: what comes right after a kept one is the end."""
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "The answer: `" + IM_END)
        r = run(service_for(engine, tokenizer, quoted_eos_recovery=True), tokenizer, thinking=False)
        self.assertEqual((r["content"], r["done"]["finish"], r["done"]["quoted_eos_recoveries"]),
                         ("The answer: `" + IM_END, "stop", 1))
        engine = SessionEngine(tokenizer, "```\n" + "\n".join([IM_END] * 6) + "\n```")
        r = run(service_for(engine, tokenizer, quoted_eos_recovery=True), tokenizer, thinking=False)
        self.assertEqual((r["content"], r["done"]["finish"]), ("```\n" + "\n".join([IM_END] * 4) + "\n", "stop"))
        for text in ("```text\n" + IM_END + "\n<|endoftext|>\n</think>\n```\nOK",
                     "Compare `" + IM_END + "` with `" + IM_END + "` and `" + IM_END + "`."):
            with self.subTest(text=text[:24]):             # a list of markers, or text between them: all are kept
                engine = SessionEngine(tokenizer, text)
                self.assertEqual(run(service_for(engine, tokenizer, quoted_eos_recovery=True), tokenizer,
                                    thinking=False)["content"], text)

    def test_in_the_thinking_the_budget_ends_a_loop_of_markers(self):
        """Without think_eos_ban the thinking can meet end tokens; where only markers come, the budget's wrap-up ends it."""
        tokenizer = ByteTokenizer()

        class QuoteLoopEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                text = "OK" if tokenizer.decode(ids).endswith("</think>\n\n") else "```\n" if self.turns == 0 else ""
                self.turns += 1
                yield from (tokenizer.encode(text) + tokenizer.encode(IM_END, parse_special=True))[:max_new]
        r = run(service_for(QuoteLoopEngine(tokenizer, ""), tokenizer, quoted_eos_recovery=True), tokenizer,
                cap=500, sampling={"reasoning_budget_tokens": 100})
        self.assertEqual(r["content"], "OK")

    def test_off_leaves_the_reply_and_its_records_as_they_were(self):
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "Compare `" + IM_END + "`.</think>\n\nOK")
        service = service_for(engine, tokenizer)
        r = run(service, tokenizer)
        self.assertEqual((r["reasoning"], r["content"], r["done"]["finish"], engine.turns), ("Compare `", "", "stop", 1))
        self.assertNotIn("quoted_eos_recoveries", r["done"])
        self.assertNotIn("quoted_eos_recoveries", service.history[-1])
        self.assertNotIn("think_eos_bans", r["done"])
        self.assertNotIn("think_eos_bans", service.history[-1])

    def test_a_closed_quote_or_fence_ends_normally(self):
        for text in ('The answer is `4`', 'The answer is "4"', '```\n4\n```'):
            with self.subTest(text=text):
                _, engine, r = self.reply([text, 'UNEXPECTED'], thinking=False)
                self.assertEqual(r["content"], text)
                self.assertEqual(engine.turns, 1)

    def test_an_answer_that_ends_on_an_ordinary_word_ends_there(self):
        for text in ("The answer is 42", "注意言行举止和行为", "这体现了它的作用", "该功能不可用", "Done。"):
            with self.subTest(text=text):
                tokenizer = ByteTokenizer()
                engine = SessionEngine(tokenizer, text)
                r = run(service_for(engine, tokenizer, quoted_eos_recovery=True), tokenizer, thinking=False)
                self.assertEqual((r["content"], r["done"]["finish"], engine.turns), (text, "stop", 1))

    def test_a_marker_after_an_introduction_or_on_the_next_line_is_kept(self):
        for head, rest in (("The end token is ", " and it ends the turn."), ("结束标记是", "，表示一条消息结束。"),
                           ("Qwen uses ", " to end a message.")):
            with self.subTest(head=head):
                tokenizer = ByteTokenizer()
                engine = SessionEngine(tokenizer, head + IM_END + rest)
                r = run(service_for(engine, tokenizer, quoted_eos_recovery=True), tokenizer, thinking=False)
                self.assertEqual((r["content"], r["done"]["finish"]), (head + IM_END + rest, "stop"))
        tokenizer = ByteTokenizer()
        engine = SessionEngine(tokenizer, "原样拼写：\n" + IM_END + "\n第一个标记。")
        r = run(service_for(engine, tokenizer, quoted_eos_recovery=True), tokenizer, thinking=False)
        self.assertEqual(r["content"], "原样拼写：\n" + IM_END + "\n第一个标记。")

    def test_explicit_stop_strings_still_stop_at_the_requested_spelling(self):
        _, engine, r = self.reply(['Example `', '` OK'], thinking=False, sampling={"stop": IM_END})
        self.assertEqual(r["content"], 'Example `')
        self.assertEqual(r["done"]['stop_sequence'], IM_END)
        self.assertEqual(engine.turns, 1)

    def test_recovery_is_bounded_and_stays_within_the_token_cap(self):
        _, engine, r = self.reply(['```\n', 'x'], thinking=False, cap=300)
        self.assertLess(engine.turns, 300)
        self.assertGreater(r["done"]['quoted_eos_recoveries'], 16)
        self.assertEqual(r["done"]['finish'], 'length')
        self.assertLessEqual(r["done"]['completion_tokens'], 300)

    def test_many_quoted_markers_can_finish_within_the_callers_budget(self):
        scripts = ['Compare `'] + ['` and `'] * 20 + ['`.</think>\n\nOK']
        _, engine, r = self.reply(scripts, cap=2000)
        self.assertEqual(r["done"]['quoted_eos_recoveries'], 21)
        self.assertEqual(r["done"]['finish'], 'stop')
        self.assertEqual(r["content"], 'OK')

    def test_no_room_or_cancellation_does_not_start_another_generation(self):
        _, engine, r = self.reply(['`', 'UNEXPECTED'], cap=2)
        self.assertEqual(engine.turns, 1)
        self.assertLessEqual(r["done"]['completion_tokens'], 2)
        cancel = threading.Event()
        cancel.set()
        _, engine, r = self.reply(['`', 'UNEXPECTED'], cancel=cancel)
        self.assertEqual(engine.turns, 0)


class Routes(unittest.TestCase):
    """Through the real server, on both APIs, streamed and whole: the marker is kept and the call after it goes out once."""
    # The request is the server tests' own (test_server.UsageAndStatus.request reads `base` off the test).
    request = test_server.UsageAndStatus.request

    def test_openai_and_anthropic_streamed_and_whole(self):
        for route in ('/v1/chat/completions', '/v1/messages'):
            for stream in (False, True):
                with self.subTest(route=route, stream=stream):
                    tokenizer = ByteTokenizer()
                    engine = SessionEngine(tokenizer, 'Compare `' + IM_END + '`.</think>\n\n' + CALL)
                    httpd = serve(service_for(engine, tokenizer, quoted_eos_recovery=True), port=0)
                    self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
                    try:
                        tools = [{"type": "function", "function": SCHEMA[0]}] if route.endswith('completions') else [
                            {"name": "probe", "input_schema": SCHEMA[0]['parameters']}]
                        body = {"model": "m", "messages": [{"role": "user", "content": "next"}], "tools": tools,
                                "max_tokens": 1000, "stream": stream, "thinking": {"type": "enabled", "budget_tokens": 500}}
                        status, raw = self.request(route, body)
                        raw = raw.decode()
                        self.assertEqual(status, 200)
                        self.assertEqual(engine.turns, 2)
                        self.assertIn('probe', raw)
                        self.assertIn('tool_calls' if route.endswith('completions') else 'tool_use', raw)
                    finally:
                        httpd.shutdown()
                        httpd.server_close()


if __name__ == '__main__':
    unittest.main()
