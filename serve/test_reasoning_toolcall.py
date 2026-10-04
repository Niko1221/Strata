"""serve/test_reasoning_toolcall.py - a tool call written inside the thinking, without </think> first, ends the
thinking implicitly (OutputParser._implicit_call); one the model only writes ABOUT stays reasoning.

    python -m unittest serve.test_reasoning_toolcall -v

The real cases are serve/fixtures/casos_tool_call_raciocinio.json: Qwen3.8-Flash-Next (IQ3_S) replies recorded by
Hermes in which the call ended up in reasoning_content with no tool_calls.  There were no real mention-only replies
to use as controls, so the controls below are written by hand.
"""
from __future__ import annotations

import contextlib
import io
import json
import random
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "serve/fixtures/casos_tool_call_raciocinio.json").read_text())["falhas"]
CALL = "<tool_call>\n<function=execute_code>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>"

# The model writes ABOUT the call format: none of these is a call, so no tool_call and the text stays as it was.
CONTROLS = {
    "backticks mid-sentence": "I must emit `<tool_call>` then `<function=execute_code>` - the format.\n",
    "mid-sentence, whole call": "The format is <tool_call>\n<function=execute_code>\n</function>\n</tool_call> ok.\n",
    "backticks at line start": "Plan:\n`<tool_call>\n<function=execute_code>` comes next.\n",
    "line start, no <function=": "Next step:\n<tool_call> is the tag the format starts with.\n",
    "line start, other tag after": "Example:\n<tool_call>\n<parameter=code>\nx\n</parameter>\n",
    "indented": "As a list:\n  <tool_call>\n  <function=execute_code>\n",
    "markdown quote": "The doc says:\n> <tool_call>\n> <function=execute_code>\n",
    "``` code block": "Format:\n```xml\n" + CALL + "\n```\nThat is how a call looks.\n",
    "~~~ code block": "Format:\n~~~\n" + CALL + "\n~~~\nDone.\n",
    "tag right before </think>": "So I call it:\n<tool_call>\n",
}


def parse(text, step, stream_tools, thinking=True):
    """Feed `text` in pieces of `step` characters (0: random sizes); return (reasoning, content, calls, parser)."""
    p = OutputParser(thinking=thinking, tools=None, stream_tools=stream_tools)
    evs, i, rnd = [], 0, random.Random(step)
    while i < len(text):
        n = step or rnd.randint(1, 9)
        evs += p.feed(text[i:i + n])
        i += n
    evs += p.finish()
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [e.call for e in evs if e.kind == "tool_call"]
    return reasoning, content, calls, p


WAYS = [(step, st) for step in (1, 0, 7, 1_000_000) for st in (False, True)]   # streaming and whole replies


class RealCases(unittest.TestCase):
    def test_the_recorded_failures_give_a_tool_call(self):
        self.assertEqual(len(CASES), 4)
        for k, case in enumerate(CASES):
            text = case["reasoning_content"] + case["content"]      # what the model generated after <think>
            before = text[:text.index("<tool_call>")]
            name = text.split("<function=", 1)[1].split(">", 1)[0]
            for step, st in WAYS:
                with self.subTest(case=case["quando"], step=step, stream_tools=st):
                    reasoning, content, calls, p = parse(text, step, st)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].name, name)
                    self.assertEqual(reasoning, before)          # the text before the call stays reasoning
                    self.assertEqual(content.strip(), "")
                    self.assertEqual(p.implicit_ends, 1)
                    if name == "tool_call":                      # Hermes' batch tool: a JSON list
                        self.assertIsInstance(calls[0].arguments["calls"], list)

    def test_unfinished_implicit_call_stays_reasoning(self):
        text = CASES[1]["reasoning_content"]
        cut = text[:text.index("</parameter>")]                  # the reply ended inside the call
        reasoning, content, calls, _ = parse(cut, 1_000_000, False)
        self.assertEqual((reasoning, content, calls), (cut, "", []))


class Controls(unittest.TestCase):
    def test_mentions_are_not_calls(self):
        for name, text in CONTROLS.items():
            for step, st in WAYS:
                with self.subTest(control=name, step=step, stream_tools=st):
                    reasoning, content, calls, p = parse(text, step, st)
                    self.assertEqual(calls, [])
                    self.assertEqual(reasoning, text)
                    self.assertEqual(content, "")
                    self.assertEqual(p.implicit_ends, 0)

    def test_mention_then_a_real_call_after_think(self):
        # the real call after </think> is read as before, once; the mentions stay in the reasoning
        for name, text in CONTROLS.items():
            for step, st in WAYS:
                with self.subTest(control=name, step=step, stream_tools=st):
                    reasoning, content, calls, p = parse(text + "</think>\n\n" + CALL, step, st)
                    self.assertEqual([c.name for c in calls], ["execute_code"])
                    self.assertEqual(reasoning, text)
                    self.assertEqual(p.implicit_ends, 0)

    def test_after_a_code_block_closes_a_call_counts(self):
        text = "Format:\n```\n<tool_call>\n```\nNow for real:\n" + CALL
        for step, st in WAYS:
            with self.subTest(step=step, stream_tools=st):
                reasoning, _, calls, p = parse(text, step, st)
                self.assertEqual([c.name for c in calls], ["execute_code"])
                self.assertEqual(reasoning, "Format:\n```\n<tool_call>\n```\nNow for real:\n")

    def test_a_call_as_the_first_text_of_the_thinking(self):
        reasoning, _, calls, _ = parse(CALL, 1, True)
        self.assertEqual((reasoning, [c.name for c in calls]), ("", ["execute_code"]))

    def test_without_thinking_nothing_changes(self):
        reasoning, content, calls, p = parse("Here:\n" + CALL, 1, False, thinking=False)
        self.assertEqual((reasoning, content, [c.name for c in calls], p.implicit_ends), ("", "Here:", ["execute_code"], 0))


class TemplateHistory(unittest.TestCase):
    """serve/chat_template.jinja: an earlier reply whose call stayed inside the thinking is rendered with </think>
    before the call (defense in depth); everything else renders as before (serve/chat_golden.json)."""
    TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
    WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    KWARGS = {"tool call and response": {"tools": WEATHER}, "no generation prompt": {"add_generation_prompt": False},
              "thinking disabled": {"enable_thinking": False}}     # what each golden case was rendered with

    def render_turn(self, assistant):
        out = self.TEMPLATE.render([{"role": "user", "content": "hi"}, assistant, {"role": "user", "content": "go"}])
        return out[out.index("<|im_start|>assistant"):out.rindex("<|im_start|>user")]

    def test_golden(self):
        cases = json.loads((ROOT / "serve/chat_golden.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 10)
        for c in cases:
            with self.subTest(case=c["name"]):
                self.assertEqual(self.TEMPLATE.render(c["messages"], **self.KWARGS.get(c["name"], {})), c["hf"])

    def test_call_left_in_reasoning_content(self):
        for case in CASES:
            r = case["reasoning_content"]
            with self.subTest(case=case["quando"]):
                turn = self.render_turn({"role": "assistant", "content": "", "reasoning_content": r})
                before, call = r[:r.index("<tool_call>")].strip(), r[r.index("<tool_call>"):]
                self.assertEqual(turn, f"<|im_start|>assistant\n<think>\n{before}\n</think>\n\n{call}<|im_end|>\n")

    def test_open_think_in_content(self):
        r = CASES[1]["reasoning_content"]
        turn = self.render_turn({"role": "assistant", "content": "<think>\n" + r})
        before, call = r[:r.index("<tool_call>")].strip(), r[r.index("<tool_call>"):]
        self.assertEqual(turn, f"<|im_start|>assistant\n<think>\n{before}\n</think>\n\n{call}<|im_end|>\n")

    def test_mentions_render_as_before(self):
        for name, text in CONTROLS.items():
            with self.subTest(control=name):
                turn = self.render_turn({"role": "assistant", "content": "ok", "reasoning_content": text})
                self.assertEqual(turn, f"<|im_start|>assistant\n<think>\n{text.strip()}\n</think>\n\nok<|im_end|>\n")
        # a reply whose call WAS extracted keeps its reasoning as it is, even when it ends with a quoted call
        quoted = "The format is:\n" + CALL
        turn = self.render_turn({"role": "assistant", "content": "", "reasoning_content": quoted, "tool_calls": [
            {"type": "function", "function": {"name": "execute_code", "arguments": {"code": "print(1)"}}}]})
        self.assertTrue(turn.startswith(f"<|im_start|>assistant\n<think>\n{quoted}\n</think>\n\n"))


class ServerCountsAndLogs(unittest.TestCase):
    """The server logs each implicit end in servidor.log and counts it in GET /metrics (totals and the request)."""

    def test_metrics_and_log(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, [CASES[2]["reasoning_content"], "</think>\n\nok"], max_context=32768)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(svc.totals["implicit_reasoning_ends"], 0)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        tools = [{"type": "function", "function": {"name": "tool_call", "parameters": {"type": "object"}}}]
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                replies = []
                for _ in range(2):
                    body = json.dumps({"model": "m", "max_tokens": 4000, "tools": tools,
                                       "messages": [{"role": "user", "content": "hi"}]})
                    req = urllib.request.Request(base + "/v1/chat/completions", data=body.encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        replies.append(json.loads(r.read())["choices"][0])
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                m = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        first = replies[0]
        self.assertEqual(first["finish_reason"], "tool_calls")
        self.assertEqual(first["message"]["tool_calls"][0]["function"]["name"], "tool_call")
        self.assertNotIn("<tool_call>", first["message"].get("reasoning_content") or "")
        self.assertEqual(replies[1]["finish_reason"], "stop")
        self.assertEqual(m["totals"]["implicit_reasoning_ends"], 1)
        self.assertEqual([r["implicit_reasoning_ends"] for r in m["requests"]], [0, 1])
        self.assertIn("[strata] implicit end of thinking: 1 tool call(s)", log.getvalue())


if __name__ == "__main__":
    unittest.main()
