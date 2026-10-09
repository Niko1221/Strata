"""serve/test_kv_pool.py - the server's side of the shared KV pool (--kv-pool-tokens): the engine's "KV pool full" refusal
is a retryable 503 (with Retry-After) on every API, a decoding lane the pool ended is reported truncated, the engine's
POOL line reaches /metrics as live.kv_pool, and an ERR ends the request at once (mock engines: no GPU, no pack).

    python -m unittest serve.test_kv_pool -v
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (POOL_RETRY_S, ByteTokenizer, MockEngine, PoolFull, Service, StrataEngine,  # noqa: E402
                          engine_error, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
POOL_MSG = "KV pool full: the decoding lanes hold all 524288 cells of --kv-pool-tokens; try again when one finishes"


class RefusingEngine(MockEngine):
    """The engine answers ERR KV pool full to the request: before any token (`beat` False, the headers are not sent yet) or
    after a prompt-progress heartbeat (`beat` True: the stream has begun)."""

    def __init__(self, *a, beat=False, message=POOL_MSG, **kw):
        super().__init__(*a, **kw)
        self.beat, self.message = beat, message

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if self.beat:
            yield None
        raise engine_error(self.message)


class DoneEngine(MockEngine):
    """The mock engine whose `last` comes from a DONE line, as StrataEngine parses it."""

    def __init__(self, *a, done_lines=(), **kw):
        super().__init__(*a, **kw)
        self.done_lines = list(done_lines)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            StrataEngine._parse_done(self, self.done_lines.pop(0))


def start(engine_cls, **kw):
    tok = ByteTokenizer()
    svc = Service(engine_cls(tok, "ok", max_context=CTX, **kw), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)
    return svc, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, dict(e.headers), e.read().decode()


MSGS = [{"role": "user", "content": "hi"}]


class ErrorMapping(unittest.TestCase):
    def test_pool_full_is_its_own_exception(self):
        e = engine_error(POOL_MSG)
        self.assertIsInstance(e, PoolFull)
        self.assertIsInstance(e, ValueError)          # old handlers that catch ValueError still see it
        self.assertEqual(str(e), POOL_MSG)

    def test_other_refusals_stay_plain(self):
        for msg in ("bad request: max_new", "the KV pool is full", "context exceeded", "", "kv pool full"):
            e = engine_error(msg)
            self.assertNotIsInstance(e, PoolFull, msg)
            self.assertIsInstance(e, ValueError)


class RefusedBeforeTheStream(unittest.TestCase):
    """The pool's refusal reaches the client as 503 + Retry-After + code kv_pool_full; a prompt that can never fit is
    still a 400 (the server refuses that one itself, before the engine)."""

    def setUp(self):
        self.svc, self.httpd, self.base = start(RefusingEngine)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def check_503(self, code, headers, text):
        self.assertEqual(code, 503, text)
        self.assertEqual(headers.get("Retry-After"), str(POOL_RETRY_S))
        return json.loads(text)

    def test_chat_completions(self):
        body = self.check_503(*post(self.base, "/v1/chat/completions", {"model": "m", "messages": MSGS, "max_tokens": 20}))
        self.assertEqual(body["error"]["code"], "kv_pool_full")
        self.assertEqual(body["error"]["type"], "server_error")
        self.assertIn("KV pool full", body["error"]["message"])

    def test_anthropic_messages(self):
        body = self.check_503(*post(self.base, "/v1/messages", {"model": "m", "messages": MSGS, "max_tokens": 20}))
        self.assertEqual(body["error"]["code"], "kv_pool_full")

    def test_responses_not_streamed(self):
        body = self.check_503(*post(self.base, "/v1/responses", {"model": "m", "input": "hi", "max_output_tokens": 20}))
        self.assertEqual(body["error"]["code"], "kv_pool_full")

    def test_a_plain_engine_refusal_is_not_retryable(self):
        svc, httpd, base = start(RefusingEngine, message="something else refused it")
        try:
            code, headers, text = post(base, "/v1/chat/completions", {"model": "m", "messages": MSGS, "max_tokens": 20})
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(code, 400, text)
        self.assertNotIn("Retry-After", headers)

    def test_a_prompt_longer_than_the_context_is_a_400_not_a_503(self):
        code, headers, text = post(self.base, "/v1/chat/completions",
                                   {"model": "m", "messages": MSGS, "max_tokens": CTX * 2})
        self.assertEqual(code, 400, text)
        self.assertNotIn("Retry-After", headers)


class RefusedInTheStream(unittest.TestCase):
    """After the first heartbeat the headers are out: the refusal is an error event that says what it is."""

    def setUp(self):
        self.svc, self.httpd, self.base = start(RefusingEngine, beat=True)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_chat_completions_event(self):
        code, _, text = post(self.base, "/v1/chat/completions", {"model": "m", "messages": MSGS, "max_tokens": 20, "stream": True})
        self.assertEqual(code, 200)
        event = next(json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: {") and '"error"' in l)
        self.assertEqual(event["error"]["code"], "kv_pool_full")
        self.assertEqual(event["error"]["retry_after"], POOL_RETRY_S)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_anthropic_event_is_overloaded(self):
        code, _, text = post(self.base, "/v1/messages", {"model": "m", "messages": MSGS, "max_tokens": 20, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("event: error", text)
        event = json.loads(text.split("event: error\ndata: ", 1)[1].split("\n", 1)[0])
        self.assertEqual(event["error"]["type"], "overloaded_error")

    def test_responses_event_carries_the_code(self):
        code, _, text = post(self.base, "/v1/responses", {"model": "m", "input": "hi", "max_output_tokens": 20, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("kv_pool_full", text)
        self.assertIn("response.failed", text)

    def test_a_refusal_in_a_stream_is_not_an_overloaded_error_when_it_is_another_error(self):
        svc, httpd, base = start(RefusingEngine, beat=True, message="verify: layer 31 never rang")
        try:
            _, _, text = post(base, "/v1/messages", {"model": "m", "messages": MSGS, "max_tokens": 20, "stream": True})
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertIn('"api_error"', text)
        self.assertNotIn("overloaded_error", text)


class TruncatedByThePool(unittest.TestCase):
    """A decoding lane the full pool cannot grow ends with finish "pool": the API says "length" and truncated: true
    (llama.cpp's field); a request that ended by itself has no such field."""

    def test_truncated_flag(self):
        svc, httpd, base = start(DoneEngine, done_lines=["DONE 4 20 40.0 30.0 pool", "DONE 4 20 40.0 30.0 stop",
                                                        "DONE 4 20 40.0 30.0 pool"])
        try:
            code, _, text = post(base, "/v1/chat/completions", {"model": "m", "messages": MSGS, "max_tokens": 1})
            self.assertEqual(code, 200, text)
            out = json.loads(text)
            self.assertEqual(out["choices"][0]["finish_reason"], "length")
            self.assertIs(out.get("truncated"), True)
            code, _, text = post(base, "/v1/chat/completions", {"model": "m", "messages": MSGS, "max_tokens": 1})
            self.assertNotIn("truncated", json.loads(text))
            code, _, text = post(base, "/v1/chat/completions",
                                 {"model": "m", "messages": MSGS, "max_tokens": 1, "stream": True})
            self.assertEqual(code, 200)
            self.assertIn('"truncated": true', text)
        finally:
            httpd.shutdown()
            httpd.server_close()


class PoolLine(unittest.TestCase):
    """The engine's `POOL <cells> <free> <main> <slot>,<slot>,...` line is kept for /metrics and never reaches a request."""

    def pump(self, lines):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc = SimpleNamespace(stdout=iter(lines), terminate=lambda: None, poll=lambda: None)
        eng.lines = queue.Queue()
        eng.slot_q = [queue.Queue() for _ in range(4)]
        eng.pool = None
        eng.ended = False
        eng.last_err = None
        eng._pump()
        return eng

    def test_pool_line_is_parsed_and_not_forwarded(self):
        eng = self.pump(["POOL 524288 507904 4096 0,8192,4096,0\n", "T 5\n", "DONE 1 2 3.0 4.0 stop\n"])
        self.assertEqual(eng.pool, {"cells": 524288, "free": 507904, "main": 4096, "slots": [0, 8192, 4096, 0]})
        got = []
        while True:
            line = eng.lines.get_nowait()
            if line is None:
                break
            got.append(line)
        self.assertEqual(got, ["T 5\n", "DONE 1 2 3.0 4.0 stop\n"])

    def test_no_slots_and_the_last_line_wins(self):
        eng = self.pump(["POOL 8192 8192 0 -\n", "POOL 8192 4096 4096 -\n"])
        self.assertEqual(eng.pool, {"cells": 8192, "free": 4096, "main": 4096, "slots": []})

    def test_a_garbled_line_changes_nothing(self):
        eng = self.pump(["POOL 8192 4096 4096 -\n", "POOL 8192\n", "POOL x y z -\n"])
        self.assertEqual(eng.pool["free"], 4096)

    def test_metrics_shows_it(self):
        svc, httpd, base = start(MockEngine)
        try:
            svc.engine.pool = {"cells": 8192, "free": 4096, "main": 4096, "slots": [0, 0]}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                live = json.loads(r.read())["live"]
            self.assertEqual(live["kv_pool"]["free"], 4096)
            svc.engine.pool = None
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertNotIn("kv_pool", json.loads(r.read())["live"])
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
