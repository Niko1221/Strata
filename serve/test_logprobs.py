"""OpenAI logprobs on /v1/chat/completions, from the engine's LP lines: python -m unittest serve.test_logprobs.

The engine (src/program/generate.cpp) writes an `LP <logprob> [id:logprob ...]` line right after
each `T <id>` when the request carries `logprobs=K`; this is the server half that turns those
lines into OpenAI's logprobs, and the parts that are easy to get wrong:

  - `allowed_token_ids`: only the named ids, renormalised over them (vLLM's semantics).
  - `response_format: json_schema` rewrites the content, so the tokens no longer spell what is
    returned; the entries are re-aligned or dropped, never returned wrong.
"""
from __future__ import annotations

import collections
import json
import queue
import threading
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, logprob_span,
                          serve, top_logprobs, allowed_ids)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ChatTemplate(ROOT / "serve" / "chat_template.jinja")


class Tokenizer(ByteTokenizer):
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class ThinkTokenizer(Tokenizer):
    # two extra special tokens that decode as the thinking marks; ALWAYS so the byte-level
    # encode() maps them wherever they appear ("" would loop forever in startswith, so they
    # are real strings, like the engine's own think tokens are)
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")


class LpEngine(MockEngine):
    """MockEngine with the LP line of each token on lp_lines before the token is yielded, as the
    engine leaves it; token i has the log-probability -(i + 1) / 100 and four named alternatives."""

    def __init__(self, tok, script, lines=True):
        super().__init__(tok, script, max_context=16384)
        self.with_lines, self.sampling = lines, None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling, self.lp_lines = dict(sampling or {}), collections.deque()
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.with_lines:
                self.lp_lines.append((
                    t, f"LP {-(i + 1) / 100:.6f} 65:-0.500000 72:-1.250000 "
                       f"88:-3.000000 91:-4.200000\n"))
            yield t


SCHEMA = {"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": {
    "type": "object", "properties": {"rent": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["rent", "quote"], "additionalProperties": False}}}


class Keys(unittest.TestCase):
    """What the request asks for, and what reaches the engine line."""

    def test_what_the_request_asks(self):
        self.assertEqual(top_logprobs({"logprobs": True}), 0)
        self.assertEqual(top_logprobs({"logprobs": True, "top_logprobs": 3}), 3)
        self.assertEqual(top_logprobs({"logprobs": True, "top_logprobs": 50}), 20)
        self.assertEqual(top_logprobs({"logprobs": 2}), 2)          # the legacy completions form
        for req in ({}, {"logprobs": False}, {"logprobs": None}, {"top_logprobs": 3}):
            self.assertIsNone(top_logprobs(req))

    def test_allowed_ids(self):
        self.assertEqual(allowed_ids({"allowed_token_ids": [1, 2, 3]}), [1, 2, 3])
        for req in ({}, {"allowed_token_ids": "1,2"}, {"allowed_token_ids": [True]}, {"allowed_token_ids": [-1]}):
            self.assertEqual(allowed_ids(req), [])

    def test_the_engine_key_only_when_asked(self):
        self.assertIn(" logprobs=0", StrataEngine.sampling_keys({"logprobs": True}))
        self.assertIn(" logprobs=5", StrataEngine.sampling_keys({"logprobs": True, "top_logprobs": 5}))
        self.assertNotIn("logprobs", StrataEngine.sampling_keys({}))
        self.assertNotIn("logprobs", StrataEngine.sampling_keys({"logprobs": False, "temperature": 0}))

    def test_allowed_reaches_the_engine_line(self):
        keys = StrataEngine.sampling_keys({"logprobs": True, "top_logprobs": 2,
                                           "allowed_token_ids": [7, 8, 9]})
        self.assertIn(" logprobs=2 ", keys)
        self.assertIn(" allowed=7,8,9", keys)


class Generate(unittest.TestCase):
    """The token's LP line is read next to it, and only when it was asked for."""

    def engine(self, *lines):
        e = StrataEngine.__new__(StrataEngine)
        e.proc = __import__("types").SimpleNamespace(
            stdin=__import__("io").StringIO(), poll=lambda: None)
        e.lines, e.can_stop, e.max_context, e.silence_s = queue.Queue(), False, 262144, 0
        for line in lines:
            e.lines.put(line)
        return e

    def test_a_token_is_yielded_with_its_lp_line_read(self):
        e = self.engine("PP 3 3", "T 72", "LP -0.100000 72:-0.1 65:-2.5", "T 105",
                        "LP -0.200000 105:-0.2 65:-3", "T 257", "LP -0.010000",
                        "DONE 3 3 0 3 stop 0 0 0")
        seen = []
        for t in e.generate([1, 2, 3], 8, {"logprobs": True, "top_logprobs": 2}, threading.Event()):
            if t is not None:
                seen.append((t, [x[0] for x in e.lp_lines]))
        self.assertEqual(seen, [(72, [72]), (105, [72, 105]), (257, [72, 105, 257])])
        self.assertIn(" logprobs=2 ", e.proc.stdin.getvalue())
        self.assertEqual(e.lp_lines[1][1], "LP -0.200000 105:-0.2 65:-3")

    def test_an_engine_without_lp_lines(self):
        e = self.engine("T 72", "T 105", "DONE 3 3 0 2 stop 0 0 0")
        toks = [t for t in e.generate([1], 8, {"logprobs": True}, threading.Event()) if t is not None]
        self.assertEqual(toks, [72, 105])
        self.assertEqual(list(e.lp_lines), [])
        self.assertEqual(e.last["finish"], "stop")

    def test_not_asked_reads_as_before(self):
        e = self.engine("T 72", "LP -0.1", "T 105", "DONE 3 3 0 2 stop 0 0 0")
        toks = [t for t in e.generate([1], 8, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [72, 105])
        self.assertEqual(list(e.lp_lines), [])

    def test_a_consumer_that_stops_early_does_not_lose_the_done_read_after_the_token(self):
        # the DONE line was read while looking for the token's LP line: the drain must take it,
        # not wait for another
        e = self.engine("T 72", "DONE 3 3 0 1 length 0 0 0", None)
        gen = e.generate([1], 1, {"logprobs": True}, threading.Event())
        self.assertEqual(next(gen), 72)
        gen.close()
        self.assertEqual(e.last["finish"], "length")


class Api(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.engine = LpEngine(self.tok, "Café costs €5.")
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, stream=False):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False}, **body}
        if stream:
            body["stream"] = True
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        if not stream:
            return json.loads(raw)
        return [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]

    def test_the_tokens_spell_the_content(self):
        out = self.post({"logprobs": True, "top_logprobs": 1})
        choice = out["choices"][0]
        content = choice["message"]["content"]
        lps = choice["logprobs"]["content"]
        self.assertEqual(content, "Café costs €5.")
        self.assertEqual("".join(e["token"] for e in lps), content)
        self.assertEqual(len(lps), len(content.encode()))       # one per byte token; no stop token
        self.assertEqual(b"".join(bytes(e["bytes"]) for e in lps), content.encode())
        e_acute = [e for e in lps if len(e["bytes"]) == 1 and e["bytes"][0] >= 0x80]
        self.assertEqual([e["token"] for e in e_acute[:2]], ["", "é"])   # split across two tokens
        self.assertEqual(lps[0]["logprob"], -0.01)
        self.assertEqual(lps[0]["top_logprobs"][0]["token"], "A")
        self.assertEqual(self.engine.sampling.get("logprobs"), True)

    def test_streamed(self):
        chunks = self.post({"logprobs": True}, stream=True)
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        toks = [e["token"] for c in chunks if c.get("choices")
                for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
        self.assertEqual(content, "Café costs €5.")
        self.assertEqual("".join(toks), content)

    def test_with_thinking_each_part_is_spelled_by_its_tokens(self):
        tok = ThinkTokenizer()
        svc = Service(LpEngine(tok, "Two times three.</think>\n\nSix."), tok,
                      TEMPLATE)
        httpd = serve(svc, port=0)
        try:
            self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
            for stream in (False, True):
                out = self.post({"logprobs": True, "chat_template_kwargs": {"enable_thinking": True}},
                                 stream)
                chunks = out if stream else [{"choices": [{"delta": out["choices"][0]["message"],
                                                           "logprobs": out["choices"][0]["logprobs"]}]}]
                for key in ("content", "reasoning_content"):
                    text = "".join(c["choices"][0]["delta"].get(key) or "" for c in chunks if c.get("choices"))
                    toks = "".join(e["token"] for c in chunks if c.get("choices")
                                   for e in (c["choices"][0].get("logprobs") or {}).get(key, []))
                    self.assertTrue(text)
                    self.assertEqual(toks, text, (key, stream))
                    self.assertNotIn("think>", toks)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_not_asked_no_logprobs(self):
        out = self.post({})
        self.assertNotIn("logprobs", out["choices"][0])

    def test_an_engine_without_lp_lines_gives_none(self):
        self.engine.with_lines = False
        out = self.post({"logprobs": True})
        self.assertEqual(out["choices"][0]["message"]["content"], "Café costs €5.")
        self.assertNotIn("logprobs", out["choices"][0])

    def test_allowed_token_ids_are_renormalised_over_the_named_ids(self):
        out = self.post({"logprobs": True, "allowed_token_ids": [65, 72]})
        lps = out["choices"][0]["logprobs"]["content"]
        self.assertTrue(lps)
        for e in lps:
            ids = [t["token"] for t in e["top_logprobs"]]
            self.assertEqual(ids, ["A", "H"])                    # only the named ones
        import math
        total = sum(math.exp(t["logprob"]) for t in lps[0]["top_logprobs"])
        self.assertAlmostEqual(total, 1.0, places=6)             # renormalised: they sum to 1

    def test_json_schema_answer_as_written_with_its_logprobs(self):
        # MockEngine appends the stop token to the script it is built with, and reports
        # finish_reason=length when it runs out -- so the engine is rebuilt with this answer
        # rather than re-pointed at a raw token list (which would drop the stop token).
        written = '```json\n{"rent": "$1,250", "quote": "x"}\n```'
        self.engine = self.svc.engine = LpEngine(self.tok, written)
        out = self.post({"logprobs": True, "temperature": 0, "response_format": SCHEMA})
        choice = out["choices"][0]
        content = choice["message"]["content"]
        self.assertEqual(content, '{"rent": "$1,250", "quote": "x"}')   # the JSON as written
        self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"]), content)
        streamed = self.post({"logprobs": True, "response_format": SCHEMA}, stream=True)
        toks = [e["token"] for c in streamed if c.get("choices")
                for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
        self.assertEqual("".join(toks), content)
        plain = self.post({"temperature": 0, "response_format": SCHEMA})    # not asked: as before
        self.assertEqual(plain["choices"][0]["message"]["content"],
                         '{"rent":"$1,250","quote":"x"}')
        self.assertNotIn("logprobs", plain["choices"][0])


class Span(unittest.TestCase):
    def entries(self, *toks):
        return [{"token": t} for t in toks]

    def test_span(self):
        es = self.entries("```", "json", "\n", "{", "}", "\n", "```")
        self.assertEqual([e["token"] for e in logprob_span("```json\n{}\n```", "{}", es)], ["{", "}"])
        self.assertEqual(len(logprob_span("{}", "{}", self.entries("{", "}"))), 2)

    def test_no_span(self):
        self.assertIsNone(logprob_span("x{}", "{}", self.entries("x{", "}")))   # no boundary at the start
        self.assertIsNone(logprob_span("{}x", "{}", self.entries("{", "}x")))   # none at the end
        self.assertIsNone(logprob_span("{}", "{}", self.entries("{")))           # do not spell it


if __name__ == "__main__":
    unittest.main()
