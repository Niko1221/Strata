"""Temporary regression coverage for output rollover; remove after verification."""
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, CTX_SLACK, Service, serve
from serve.usage import UsageLedger


class IncrementalByteTokenizer(ByteTokenizer):
    """Use the real tokenizer's streaming byte contract, avoiding the test stub's quadratic decode."""
    def token_bytes(self, token):
        return bytes([token]) if token < 256 else self.SPECIALS[token - 256].encode()


class BoundedEngine:
    """Enforce native GEN's real finite context while producing a much longer reply."""
    max_context = 256

    def __init__(self, tok, text):
        self.tokens = tok.encode(text + "<|im_end|>", parse_special=True)
        self.pos = 0
        self.calls = []
        self.last = None
        self.cancel_at = None

    def generate(self, ids, maximum, sampling, cancel, embeddings=None):
        assert maximum >= 1 and len(ids) + maximum + CTX_SLACK <= self.max_context
        self.calls.append((list(ids), maximum, embeddings))
        first = len(self.calls) == 1
        produced = 0
        stopped = False
        try:
            while produced < maximum and self.pos < len(self.tokens) and not cancel.is_set():
                token = self.tokens[self.pos]
                self.pos += 1
                produced += 1
                stopped = token == 257
                yield token
                if self.cancel_at and self.pos >= self.cancel_at:
                    cancel.set()
        finally:
            self.last = dict(prompt_tokens=len(ids), prompt_read=len(ids) - (2 if first else 0),
                             reused=2 if first else 0, generated=produced,
                             prompt_ms=10.0, decode_ms=20.0,
                             finish="stop" if stopped else "cancel" if cancel.is_set() else "length")


class UnlimitedRollover(unittest.TestCase):
    def service(self, text="abcde " * 180, unlimited=True):
        tok = IncrementalByteTokenizer()
        engine = BoundedEngine(tok, text)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")),
                      unlimited_output=unlimited)
        svc.repeat_stop_tokens = 0
        return svc, engine

    def test_prepare_is_logically_unbounded_and_rejects_oversized_input(self):
        svc, _ = self.service()
        ids, _, maximum = svc.prepare([{"role": "user", "content": "hi"}], None,
                                      {"enable_thinking": False}, 1)
        self.assertEqual(maximum, -1)
        self.assertLess(len(ids), 256)
        with self.assertRaisesRegex(ValueError, "exceeds the context window"):
            svc.prepare([{"role": "user", "content": "x" * 256}], None, {}, 1)

    def test_stream_crosses_many_windows_without_losing_or_counting_tokens_twice(self):
        svc, engine = self.service()
        with tempfile.TemporaryDirectory() as folder:
            svc.usage = UsageLedger(Path(folder) / "usage.json", "test")
            events = list(svc.run([65] * 30, False, None, -1, {}, threading.Event()))
            content = "".join(event.text or "" for kind, event in events
                              if kind == "event" and event.kind == "content")
            done = events[-1][1]
            self.assertEqual(content, "abcde " * 180)
            self.assertEqual(done["finish"], "stop")
            self.assertEqual(done["completion_tokens"], 1081)
            self.assertEqual(done["reused"], 2)
            self.assertGreater(len(engine.calls), 6)
            for prompt, budget, _ in engine.calls:
                self.assertEqual(prompt[:30], [65] * 30)
                self.assertLessEqual(len(prompt) + budget + CTX_SLACK, engine.max_context)
            total = svc.usage.snapshot()
            self.assertEqual(total["requests"], 1)
            self.assertEqual(total["input_tokens"], 30)
            self.assertEqual(total["cached_tokens"], 2)
            self.assertEqual(total["output_tokens"], 1081)
            self.assertEqual(svc.history[-1]["context_rollovers"], len(engine.calls) - 1)
            self.assertLessEqual(svc.history[-1]["context_tokens"], 248)

    def test_default_mode_still_respects_positive_cap(self):
        svc, engine = self.service(unlimited=False)
        events = list(svc.run([65] * 30, False, None, 16, {}, threading.Event()))
        self.assertEqual(events[-1][1]["finish"], "length")
        self.assertEqual(events[-1][1]["completion_tokens"], 16)
        self.assertEqual(len(engine.calls), 1)

    def test_cancel_stops_between_rollovers(self):
        svc, engine = self.service()
        engine.cancel_at = 320
        events = list(svc.run([65] * 30, False, None, -1, {}, threading.Event()))
        self.assertEqual(events[-1][1]["finish"], "cancel")
        self.assertEqual(events[-1][1]["completion_tokens"], 320)
        self.assertEqual(len(engine.calls), 2)

    def test_stop_string_split_across_windows(self):
        svc, engine = self.service("a" * 215 + "END" + "never sent" * 50)
        events = list(svc.run([65] * 30, False, None, -1, {"stop": "END"}, threading.Event()))
        content = "".join(event.text or "" for kind, event in events
                          if kind == "event" and event.kind == "content")
        self.assertEqual(content, "a" * 215)
        self.assertEqual(events[-1][1]["finish"], "stop")
        self.assertLessEqual(len(engine.calls), 2)

    def test_unicode_survives_partial_utf8_at_rollover(self):
        svc, _ = self.service("🙂" * 150)
        events = list(svc.run([65] * 31, False, None, -1, {}, threading.Event()))
        content = "".join(event.text or "" for kind, event in events
                          if kind == "event" and event.kind == "content")
        self.assertEqual(content, "🙂" * 150)

    def test_reasoning_budget_and_close_retry_keep_working(self):
        svc, engine = self.service("reason " * 100 + "</think>\nanswer")
        svc.reasoning_budget_tokens = 20
        events = list(svc.run([65] * 30, True, None, -1, {}, threading.Event()))
        self.assertTrue(any(kind == "event" and event.kind == "content" for kind, event in events))
        self.assertEqual(events[-1][1]["finish"], "stop")
        self.assertGreater(len(engine.calls), 3)

    def test_eos_does_not_start_another_window(self):
        svc, engine = self.service("short reply")
        events = list(svc.run([65] * 30, False, None, -1, {}, threading.Event()))
        self.assertEqual(events[-1][1]["finish"], "stop")
        self.assertEqual(len(engine.calls), 1)

    def test_apis_ignore_client_cap_and_report_complete_output(self):
        for path, budget_key in (("/v1/chat/completions", "max_tokens"),
                                 ("/v1/messages", "max_tokens"),
                                 ("/v1/responses", "max_output_tokens")):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    svc, engine = self.service()
                    httpd = serve(svc, port=0)
                    body = {"model": "test", "stream": stream, budget_key: 1,
                            "chat_template_kwargs": {"enable_thinking": False},
                            "thinking": {"type": "disabled"}}
                    body.update({"input": "hi"} if path == "/v1/responses" else
                                {"messages": [{"role": "user", "content": "hi"}]})
                    try:
                        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=10) as response:
                            raw = response.read().decode()
                        if stream:
                            chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                            if path == "/v1/chat/completions":
                                text = "".join(c["choices"][0].get("delta", {}).get("content", "")
                                               for c in chunks if c.get("choices"))
                            elif path == "/v1/messages":
                                text = "".join(c.get("delta", {}).get("text", "") for c in chunks)
                            else:
                                text = "".join(c.get("delta", "") for c in chunks
                                               if c.get("type") in ("response.output_text.delta", "response.reasoning_text.delta"))
                            self.assertEqual(text, "abcde " * 180)
                        else:
                            self.assertIn("abcde", raw)
                        self.assertEqual(svc.history[-1]["finish"], "stop")
                        self.assertEqual(svc.history[-1]["output_tokens"], 1081)
                        self.assertGreater(len(engine.calls), 4)
                    finally:
                        httpd.shutdown()
                        httpd.server_close()

    def test_more_than_118k_output_tokens_complete(self):
        svc, engine = self.service("abcde\n" * 24000)
        engine.max_context = 2048
        content = []
        for kind, event in svc.run([65] * 30, False, None, -1, {}, threading.Event()):
            if kind == "event" and event.kind == "content":
                content.append(event.text or "")
            elif kind == "done":
                self.assertEqual(event["completion_tokens"], 144001)
                self.assertEqual(event["finish"], "stop")
        self.assertEqual("".join(content), "abcde\n" * 24000)
        self.assertGreater(len(engine.calls), 100)

    def test_tool_call_parser_survives_many_windows(self):
        args = "abcde " * 180
        svc, engine = self.service('<tool_call>\n<function=echo><parameter=text>' + args +
                                   '</parameter></function>\n</tool_call>')
        tools = [{"type": "function", "function": {"name": "echo", "parameters": {
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
        events = list(svc.run([65] * 30, False, tools, -1, {}, threading.Event()))
        calls = [event.call for kind, event in events if kind == "event" and event.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments["text"], args)
        self.assertGreater(len(engine.calls), 6)


if __name__ == "__main__":
    unittest.main()
