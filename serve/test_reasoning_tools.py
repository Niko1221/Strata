"""serve/test_reasoning_tools.py - #804 (strict form of PR #787): a tool call the model writes inside its reasoning.

A call inside the reasoning counts only when tools are declared and the name is one of them; anything else stays
reasoning text.  Without tools the parser is byte-identical to before.

#1058 adds a gate: such a call must also begin a line, be followed by nothing but whitespace and further such
calls, and the turn must end by itself (a stop token, not max_tokens or a cancel) - a call the model only quoted while
thinking (an example, a fenced block, a turn cut mid-thought) stays reasoning text.

    python -m unittest serve.test_reasoning_tools -v
"""
from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_server import CTX, UnfinishedToolCall  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CALL = "<tool_call>\n<function=read_file>\n<parameter=limit>\n120\n</parameter>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>"
OTHER = CALL.replace("read_file", "not_declared")
TOOLS = [{"name": "read_file", "parameters": {"properties": {"limit": {"type": "integer"}, "path": {"type": "string"}}}}]


def run(text, width, tools=TOOLS, thinking=True, stream_tools=True, natural=True):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools)
    evs = []
    for i in range(0, len(text), width):
        evs += p.feed(text[i:i + width])
    return evs + p.finish(natural=natural)


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class Parser(unittest.TestCase):
    def test_a_declared_call_that_ends_the_turn_is_a_call(self):
        for text, n in (("before\n\n" + CALL, 1), ("before\n\n" + CALL + "\n", 1), (CALL, 1),
                        ("before\n\n" + CALL + "\n" + CALL + "\n", 2)):
            for width in (1, 2, 7, 100_000):
                with self.subTest(text=text[:12], n=n, width=width):
                    evs = run(text, width)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([(c.name, c.arguments) for c in calls],
                                     [("read_file", {"limit": 120, "path": "file.txt"})] * n)
                    self.assertEqual(joined(evs, "reasoning"), text.split("<tool_call>")[0])
                    self.assertEqual(joined(evs, "content"), "")
                    self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_args")])   # never streamed

    def test_the_counts(self):
        """#1058: GET /metrics' totals come from these: calls delivered, and declared calls kept as text."""
        for text, natural, want in (("x\n" + CALL, True, [1, 0]), ("x\n" + CALL + "\n" + CALL, True, [2, 0]),
                                    ("x\n" + CALL, False, [0, 1]), ("x " + CALL + "\n", True, [0, 1]),
                                    ("x\n" + CALL + " more", True, [0, 1]), ("x " + OTHER, True, [0, 0])):
            for width in (1, 100_000):
                with self.subTest(text=text[:4], natural=natural, width=width):
                    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
                    for i in range(0, len(text), width):
                        p.feed(text[i:i + width])
                    p.finish(natural=natural)
                    self.assertEqual(p.rcalls, want)

    def test_a_quoted_call_stays_reasoning(self):
        """#1058: what the model only quoted while thinking is never a call - its text comes back whole."""
        cases = {"more thought after it": ("before " + CALL + " after</think>\n\nanswer", True),
                 "mid-sentence": ("I would write " + CALL + "\n", True),
                 "prose after it": ("The file is small.\n" + CALL + "\nlet me see what comes back.", True),
                 "fenced": ("Format:\n```xml\n" + CALL + "\n```\nThat is how a call looks.", True),
                 "the thinking ends after it": ("Plan:\n" + CALL + "\n</think>\n\nanswer", True),
                 "cut by max_tokens": ("I could run\n" + CALL + "\n", False),
                 "two, prose between": ("Both:\n" + CALL + " and also " + CALL, True),
                 "two, the second mid-line": ("Both:\n" + CALL + CALL, True)}
        for name, (text, natural) in cases.items():
            for width in (1, 3, 100_000):
                with self.subTest(name, width=width):
                    evs = run(text, width, natural=natural)
                    self.assertFalse([e for e in evs if e.kind in ("tool_call", "tool_start", "tool_args")])
                    self.assertEqual(joined(evs, "reasoning"), text.split("</think>")[0])
                    self.assertEqual(joined(evs, "content"), "answer" if "</think>" in text else "")

    def test_an_undeclared_name_stays_reasoning(self):
        text = "see " + OTHER + " done</think>\n\nanswer"
        for width in (1, 5, 100_000):
            with self.subTest(width=width):
                evs = run(text, width)
                self.assertFalse([e for e in evs if e.kind == "tool_call"])
                self.assertEqual(joined(evs, "reasoning"), "see " + OTHER + " done")
                self.assertEqual(joined(evs, "content"), "answer")

    def test_no_tools_declared_is_unchanged(self):
        text = "before " + CALL + " after</think>\n\nanswer"
        for tools in (None, []):
            for width in (1, 7, 100_000):
                with self.subTest(tools=tools, width=width):
                    evs = run(text, width, tools=tools)
                    self.assertFalse([e for e in evs if e.kind == "tool_call"])
                    self.assertEqual(joined(evs, "reasoning"), "before " + CALL + " after")
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_an_unfinished_call_is_never_a_call(self):
        for text in ("before<tool_call>\n<function=read_file>\n",
                     "before<tool_call>\n<function=read_file>\n<parameter=path>\nfile.txt\n</parameter>\n"):
            for width in (1, 100_000):
                with self.subTest(text=text[-20:], width=width):
                    evs = run(text, width)
                    self.assertFalse([e for e in evs if e.kind == "tool_call"])
                    self.assertEqual(joined(evs, "reasoning"), text)

    def test_the_thinking_ending_inside_a_call_makes_it_text(self):
        text = "a<tool_call>\n<function=read_file>\nhmm</think>\n\nthe answer"
        for width in (1, 100_000):
            with self.subTest(width=width):
                evs = run(text, width)
                self.assertFalse([e for e in evs if e.kind == "tool_call"])
                self.assertEqual(joined(evs, "reasoning"), "a<tool_call>\n<function=read_file>\nhmm")
                self.assertEqual(joined(evs, "content"), "the answer")

    def test_a_call_in_the_answer_is_unchanged(self):
        text = "think</think>\n\nok " + CALL
        for width in (1, 100_000):
            evs = run(text, width)
            self.assertEqual(len([e for e in evs if e.kind == "tool_call"]), 1)
            self.assertEqual(joined(evs, "reasoning"), "think")

    def test_a_tag_that_never_closes_does_not_hold_the_thinking_back(self):
        from serve import frontend
        text = "a <tool_call> " + "thinking on and on " * 3000 + "</think>\n\nanswer"
        with unittest.mock.patch.object(frontend, "RCALL_MAX", 1000):
            evs = run(text, 50)
        first_reasoning_at = next(i for i, e in enumerate(evs) if e.kind == "reasoning" and len(e.text) > 100)
        self.assertLess(first_reasoning_at, 100)                     # streamed, not kept until </think>
        self.assertEqual(joined(evs, "reasoning"), text.split("</think>")[0])
        self.assertEqual(joined(evs, "content"), "answer")

    def test_a_partial_tag_is_held(self):
        p = OutputParser(thinking=True, tools=TOOLS)
        evs = p.feed("hello <tool_c")
        self.assertEqual(joined(evs, "reasoning"), "hello ")
        self.assertEqual(joined(p.feed("ats are nice</think>"), "reasoning"), "<tool_cats are nice")


class OverHttp(unittest.TestCase):
    SCRIPT = "Let me look.\n\n<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n" \
             "</tool_call>\n"

    def test_a_call_cut_by_max_tokens_is_text(self):
        """#1058 through the server: the same call, but the turn runs on and max_tokens cuts it right after the call."""
        helper = UnfinishedToolCall("test_parser")
        helper.PROPS = {"path": {"type": "string"}}
        a = helper.answers(self.SCRIPT + "More thought " * 20,
                           max_tokens=len(ByteTokenizer().encode(self.SCRIPT, parse_special=True)))
        self.assertEqual((a["openai", False], a["anthropic", False]), (("length", []), ("max_tokens", [])))
        self.assertEqual((a["openai", True], a["anthropic", True]), (("length", ""), ("max_tokens", "")))

    def test_finish_reasons_and_block_order(self):
        helper = UnfinishedToolCall("test_parser")
        helper.PROPS = {"path": {"type": "string"}}
        script = self.SCRIPT
        a = helper.answers(script)
        self.assertEqual((a["openai", False][0], [json.loads(x) for x in a["openai", False][1]]),
                         ("tool_calls", [{"path": "file.txt"}]))
        self.assertEqual(a["anthropic", False], ("tool_use", [{"path": "file.txt"}]))
        # the order of the Anthropic stream's blocks: thinking, then the call that ended the turn
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            body = {"model": "x", "max_tokens": 500, "stream": True, "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"name": "write", "input_schema": {"type": "object", "properties": helper.PROPS}}]}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                                   "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
            evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
            order = [e["content_block"]["type"] for e in evs if e["type"] == "content_block_start"]
            self.assertEqual(order, ["thinking", "tool_use"])
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/metrics", timeout=30) as r:
                totals = json.loads(r.read())["totals"]
            self.assertEqual((totals["reasoning_calls_delivered"], totals["reasoning_calls_kept_as_text"]), (1, 0))
            tool = [e["delta"]["partial_json"] for e in evs if e["type"] == "content_block_delta"
                    and e["delta"]["type"] == "input_json_delta"]
            self.assertEqual(json.loads("".join(tool)), {"path": "file.txt"})
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
