"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.kvcache import KvCache, parse_kv  # noqa: E402
from serve.server import CTX_SLACK, ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, request_timings, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "x" * 2000                              # longer than the old 1024 fallback: one token per byte


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MaxTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def call(self, api, text="hi", **budget):
        """-> (status, body, prompt tokens, completion tokens); `budget` is merged into the request as given."""
        msgs = [{"role": "user", "content": text}]
        if api == "openai":
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, **budget})
            u = b.get("usage", {})
            return s, b, u.get("prompt_tokens"), u.get("completion_tokens")
        s, b = self.post("/v1/messages", {"model": "m", "messages": msgs, **budget})
        u = b.get("usage", {})
        return s, b, u.get("input_tokens"), u.get("output_tokens")

    def test_unset_budget_is_the_rest_of_the_context(self):
        cases = {"openai": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None},
                            {"max_completion_tokens": -1}, {"max_completion_tokens": None, "max_tokens": None}],
                 "anthropic": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None}]}
        for api, budgets in cases.items():
            for budget in budgets:
                with self.subTest(api=api, budget=budget):
                    s, b, pt, ct = self.call(api, **budget)
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
                    self.assertGreater(ct, 1024)          # the whole answer, not cut at the old 1024 fallback

    def test_explicit_budget_is_honoured(self):
        for api, budget in [("openai", {"max_tokens": 50}), ("openai", {"max_completion_tokens": 50}),
                            ("openai", {"max_completion_tokens": 50, "max_tokens": 9}),
                            ("anthropic", {"max_tokens": 50}), ("openai", {"max_tokens": 1500}),
                            ("anthropic", {"max_tokens": 1500})]:
            with self.subTest(api=api, budget=budget):
                want = budget.get("max_completion_tokens") or budget["max_tokens"]
                s, b, _, ct = self.call(api, **budget)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, want)
                self.assertEqual(ct, want)

    def test_explicit_budget_over_the_context_is_rejected(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, _, _ = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 400)
                self.assertIn("exceeds the context", b["error"]["message"])

    def test_unset_budget_with_a_near_full_prompt(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")                  # the template's tokens around the user text
        for api in ("openai", "anthropic"):
            _, _, pa, _ = self.call(api, max_tokens=1)
            over = pa - pt0                          # the Anthropic template may differ slightly
            with self.subTest(api=api, room=5):     # a few tokens left: the budget is exactly those
                text = "y" * (CTX - CTX_SLACK - overhead - over - 5)
                s, b, pt, ct = self.call(api, text=text, max_tokens=-1)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, 5)
                self.assertEqual(ct, 5)
            with self.subTest(api=api, room=0):     # nothing left: rejected, not truncated
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])

    def test_debug_log_shows_the_resolved_budget(self):
        import contextlib
        import io
        os.environ["STRATA_DEBUG"] = "1"
        try:
            for api in ("openai", "anthropic"):
                with self.subTest(api=api):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        _, _, pt, _ = self.call(api, max_tokens=-1)
                    self.assertIn(f"max_new={CTX - CTX_SLACK - pt} ", out.getvalue())
        finally:
            del os.environ["STRATA_DEBUG"]


class FitMaxTokens(unittest.TestCase):
    """PR #24: --fit-max-tokens clamps an explicit budget that overshoots the context instead of a 400."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), fit_max_tokens=True)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = MaxTokens.post
    call = MaxTokens.call

    def test_overshoot_is_clamped_to_the_room(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, pt, ct = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)

    def test_a_budget_that_fits_is_unchanged(self):
        s, b, _, ct = self.call("openai", max_tokens=50)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, 50)

    def test_no_room_is_still_a_400(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")
        s, b, _, _ = self.call("openai", text="y" * (CTX - CTX_SLACK - overhead), max_tokens=100)
        self.assertEqual(s, 400, b)
        self.assertIn("no room to answer", b["error"]["message"])


class ImageMarkers(unittest.TestCase):
    """#150: the text "<|image_pad|>" inside a message is text, not an image's place."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

    def test_literal_marker_with_an_image(self):
        import tempfile
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                          vision=self.FakeVision(d))
            pad = tok.encode("<|image_pad|>", parse_special=True)[0]
            for text in ("the docs say <|image_pad|> marks an image", "plain"):
                with self.subTest(text=text):
                    msgs = [{"role": "user", "content": [{"type": "text", "text": text},
                                                         {"type": "image", "source": "x.png"}]}]
                    ids, _, _ = svc.prepare(msgs, None, {})
                    self.assertEqual(ids.count(pad), 3)          # the image's three rows, nothing else
                    self.assertIn("<|image_pad|> marks" if "docs" in text else "plain", tok.decode(ids))
            svc.embeddings.path.unlink(missing_ok=True)


class StatusNeedsTheKey(unittest.TestCase):
    """#212: /status shows the end of the answer being written, so it needs the key like /v1/*."""

    def test_status(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = "k3y"
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}/status"
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base, timeout=10)
            self.assertEqual(e.exception.code, 401)
            e.exception.close()
            req = urllib.request.Request(base, headers={"Authorization": "Bearer k3y"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
                self.assertNotIn("tail", json.loads(r.read()))
        finally:
            httpd.shutdown()
            httpd.server_close()


class ToolCallTerminators(unittest.TestCase):
    """#210: a value that contains </parameter> or </tool_call> (a file documenting the call format) is kept whole."""
    CONTENT = ("Close each value with </parameter> and the call with </function></tool_call>.\n"
               "<parameter=x>\nnot a parameter\n</parameter>\nend")
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"},
                                                              "content": {"type": "string"}}}}]

    def run_parser(self, stream_tools, step):
        from serve.frontend import OutputParser
        text = ("</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md\n</parameter>\n"
                f"<parameter=content>\n{self.CONTENT}\n</parameter>\n</function>\n</tool_call>")
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish()
        return evs

    def test_values_keep_the_terminators(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.run_parser(stream_tools, step)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].arguments, {"path": "doc.md", "content": self.CONTENT})
                    self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])
                    if stream_tools:
                        streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                        self.assertEqual(json.loads(streamed), {"path": "doc.md", "content": self.CONTENT})


class ClientShapes(unittest.TestCase):
    """What real clients send: Claude Code posts /v1/messages?beta=true (issue #55) and puts hook context into the
    conversation as a mid-conversation system message (issue #56); some OpenAI clients send a late developer message."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingPrompt(tok, "</think>\n\n2", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def prompt_text(self):
        return bytes(i for i in self.engine.last_ids if i < 256).decode("utf-8", "replace")

    def test_query_string(self):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
        for path in ("/v1/messages?beta=true", "/v1/chat/completions?api-version=1", "/v1/messages/?beta=true"):
            status, b = self.post(path, body)
            self.assertEqual(status, 200, (path, b))
        status, _ = self.post("/v1/nothing?beta=true", body)
        self.assertEqual(status, 404)

    def test_anthropic_mid_conversation_system(self):
        status, b = self.post("/v1/messages?beta=true", {
            "model": "x", "max_tokens": 50,
            "system": [{"type": "text", "text": "You are terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "1+1? digits only"}]},
                {"role": "system", "content": [{"type": "text", "text": "<system-reminder>answer in digits</system-reminder>"}]}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertIn("You are terse.", text)
        self.assertIn("<system-reminder>answer in digits</system-reminder>", text)
        self.assertLess(text.index("You are terse."), text.index("1+1?"))          # the first system stays first
        self.assertLess(text.index("1+1?"), text.index("answer in digits"))        # the late one stays in place

    def test_openai_late_developer_and_system(self):
        status, b = self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 50,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "hi"}, {"role": "developer", "content": "Now use digits."},
                         {"role": "system", "content": "Also this."}, {"role": "user", "content": "1+1?"}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        for part in ("Be brief.", "Now use digits.", "Also this.", "1+1?"):
            self.assertIn(part, text)

    def test_leading_system_unchanged(self):
        from serve.frontend import anthropic_to_messages, openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "developer", "content": "D"}, {"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        msgs, _, _ = anthropic_to_messages({"system": "S", "messages": [{"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class SamplingKeys(unittest.TestCase):
    """The GEN line's sampling keys: top_k 0 ("off") or wider than the engine's 64 get the widest list, 64 (they used
    to fall back to the engine default 20); a penalty always carries its window."""

    def keys(self, **sampling):
        return StrataEngine.sampling_keys(sampling).split()

    def test_top_k(self):
        self.assertIn("top_k=10", self.keys(temperature=0.7, top_k=10))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=64))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=0))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=100))
        for bad in (-1, True, 2.5, "20"):
            self.assertFalse([k for k in self.keys(temperature=0.7, top_k=bad) if k.startswith("top_k=")], bad)

    def test_tune_keys(self):
        k = self.keys(temperature=0, strata_tune={"pcie_frac": 0.2, "spec_min_p": 0.7})
        self.assertIn("pcie_frac=0.2", k)
        self.assertIn("spec_min_p=0.7", k)
        bad = self.keys(strata_tune={"pcie_frac": 3, "spec_min_p": True, "pool_workers": 2})
        self.assertFalse([x for x in bad if x.split("=")[0] in ("pcie_frac", "spec_min_p", "pool_workers")])

    def test_penalty_window(self):
        self.assertIn("penalty_last_n=64", self.keys(presence_penalty=1.5))
        self.assertIn("penalty_last_n=4096", self.keys(repetition_penalty=1.1, penalty_last_n=4096))
        self.assertFalse([k for k in self.keys(temperature=0.7) if k.startswith("penalty")])


class GpuChoice(unittest.TestCase):
    """Issue #51: the config's \"gpu\" reaches the engine as CUDA_VISIBLE_DEVICES, numbered like nvidia-smi."""

    def test_env(self):
        from serve.server import child_env
        env = child_env({"gpu": 1})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        plain = child_env({})                     # no choice: the environment as it was (existing installs)
        self.assertEqual(plain.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.assertEqual(plain.get("CUDA_DEVICE_ORDER"), os.environ.get("CUDA_DEVICE_ORDER"))


class RecordingPrompt(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class DyingEngine(MockEngine):
    """Issue #27: an engine that dies after a few tokens of its first answer, and comes back when restarted."""

    def __init__(self, tok, script, max_context):
        super().__init__(tok, script, max_context=max_context)
        self.dead, self.restarts, self.die_after = False, 0, 5

    def alive(self):
        return not self.dead

    def restart(self):
        self.dead, self.die_after = False, None
        self.restarts += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.die_after is not None and i == self.die_after:
                self.dead = True
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class EngineDeath(unittest.TestCase):
    """Issue #27: a dead engine is an error (not "length"), and the next request starts it again."""

    def test_error_then_restart(self):
        tok = ByteTokenizer()
        eng = DyingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def post(body):
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, r.read().decode()
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, e.read().decode()
            msgs = [{"role": "user", "content": "hi"}]
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50, "stream": True})
            self.assertEqual(code, 200)
            self.assertIn('"error"', text)
            self.assertIn("stopped unexpectedly", text)
            self.assertTrue(text.rstrip().endswith("data: [DONE]"))
            self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50})
            self.assertEqual(code, 200, text)
            self.assertEqual(eng.restarts, 1)
            self.assertEqual(json.loads(text)["usage"]["completion_tokens"], 50)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_engine_err_mid_stream(self):
        """The engine's ERR line after the stream started reaches the client as an error event (it used to be a
        400 written into the open stream, which clients read as an empty answer)."""
        class ErrEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield None                                  # a prompt-progress heartbeat: the stream has started
                raise ValueError("verify: layer 31 never rang (an illegal memory access was encountered)")

        tok = ByteTokenizer()
        svc = Service(ErrEngine(tok, ANSWER, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in [("/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 20,
                                                          "messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/messages", {"model": "m", "stream": True, "max_tokens": 20,
                                                 "messages": [{"role": "user", "content": "hi"}]})]:
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    text = r.read().decode()
                self.assertIn("illegal memory access", text, path)
                self.assertNotIn("HTTP/1", text, path)
                self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
        finally:
            httpd.shutdown()
            httpd.server_close()


class LiveRate(unittest.TestCase):
    """The Monitor's Speed readout: live.tok_s is a rate, and a request that never got a DONE keeps no counters.

    It used to be `generated / (now - first_token)` - the mean since the first token, whose first sample is
    1/elapsed.  Against a paced engine that reads five-digit numbers for the first instant of every answer and
    undershoots for the first second after that.  It is now the rate over the last RATE_WINDOW_S, with the mean
    still available as `live.tok_s_mean` for anyone who wants it."""

    PACE_S = 0.02                    # 50 tokens/s: a 30-token answer takes about 0.6 s
    TOKENS = 30

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "x" * self.TOKENS, max_context=CTX, delay_s=self.PACE_S)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read())

    def test_prefill_rate_excludes_cached_tokens(self):
        import io
        import queue
        from types import SimpleNamespace
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO())
        engine.lines = queue.Queue()
        engine.can_stop = False
        engine.max_context = 262144
        engine.prefill_tok_s_mean = 9999.0
        engine.lines.put("PP 10000 12000 2000 1000.0")  # 8000 cached, 2000 newly read in two seconds
        engine.lines.put("DONE 1 12000 4000 10 stop 0 0 8000")
        gen = engine.generate([1], 1, {}, threading.Event())
        self.assertIsNone(next(gen))
        self.assertEqual(engine.progress, (10000, 12000))
        self.assertEqual(engine.prefill_tok_s_mean, 1000.0)
        self.svc.engine = engine
        self.svc.status.update(busy=True, first_token=None)
        self.assertEqual(self.metrics()["live"]["prefill_tok_s_mean"], 1000.0)
        self.assertNotIn("prefill_tok_s", self.metrics()["live"])
        self.assertEqual(self.svc._prefill_tok_s_mean(), 1000.0)
        self.svc.status.update(first_token=time.time(), generated=1)
        self.assertEqual(self.svc._prefill_tok_s_mean(), 0.0)
        self.assertEqual(list(gen), [])
        timings = request_timings(12000, 1, engine.last)
        self.assertEqual(timings["prompt_per_second"], 1000.0)
        engine.lines.put("PP 8000 12000")
        engine.lines.put("DONE 0 12000 0 0 stop 0 0 12000")
        gen = engine.generate([1], 1, {}, threading.Event())
        next(gen)
        self.assertIsNone(engine.prefill_tok_s_mean)
        list(gen)
        self.svc.status["busy"] = False
        self.assertIsNone(self.metrics()["live"]["prefill_tok_s_mean"])

    def test_the_live_number_is_a_rate(self):
        live_samples, stop = [], threading.Event()

        def poll():                                   # what the Monitor polls, at 10 ms
            while not stop.is_set():
                live = self.metrics()["live"]
                if live["state"] == "generating" and live["tok_s"] is not None:
                    live_samples.append((live["generated"], live["tok_s"], live["tok_s_mean"]))
                time.sleep(0.01)

        body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "temperature": 0,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        watcher = threading.Thread(target=poll, daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                usage = json.loads(r.read())["usage"]
        finally:
            stop.set()
            watcher.join(2)
        true_rate = usage["completion_tokens"] / (time.time() - t0)
        self.assertGreaterEqual(len(live_samples), 3, "too few live readings to judge the readout")
        self.assertLess(max(s for _g, s, _m in live_samples), 4 * true_rate,
                        f"live.tok_s peaked at {max(s for _g, s, _m in live_samples):.1f} tok/s "
                        f"for a {true_rate:.1f} tok/s engine")
        self.assertEqual(self.metrics()["live"]["state"], "idle")
        self.assertIsNone(self.metrics()["live"]["tok_s"])

    def test_a_request_without_a_done_keeps_no_engine_counters(self):
        """An engine that dies mid-answer: the previous request's `last` must not become this row's decode rate."""
        class HalfDead(MockEngine):
            last = {"generated": 99, "prompt_tokens": 9, "prompt_ms": 10.0, "decode_ms": 100.0, "finish": "stop"}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if i == 3:
                        raise EngineDied("the engine stopped unexpectedly (exit code -9)")
                    yield t

        tok = ByteTokenizer()
        svc = Service(HalfDead(tok, "x" * self.TOKENS, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                text = r.read().decode()
            self.assertIn('"error"', text)
            row = svc.metrics()["requests"][0]
            self.assertEqual(row["finish"], "error")
            self.assertEqual(row["output_tokens"], 3)
            self.assertIsNone(row["decode_tok_s"], "the previous request's counters were recorded as this one's")
            self.assertIsNone(row["engine_generated"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class SharedSettings(unittest.TestCase):
    """The web app's "Use for other apps too": POST /settings makes its Chat settings every client's defaults."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        class Sampled(RecordingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last_sampling = dict(sampling or {})
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        cls.engine = Sampled(tok, "</think>\n\nhello", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc.shared_path = os.path.join(cls.tmp.name, "strata-x.shared-settings.json")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def req(self, path, body, headers=None, raw=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=raw if raw is not None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra})

    def tearDown(self):
        self.svc.set_shared(None)

    def test_other_apps_get_the_chat_settings(self):
        d = {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "seed": 7, "max_tokens": 77,
             "reasoning_effort": "low", "experimental_speed_projection": False}
        code, b = self.req("/settings", {"defaults": d})
        self.assertEqual(code, 200, b)
        self.assertTrue(b["shared"])
        self.assertTrue(os.path.exists(self.svc.shared_path))
        code, _ = self.chat()                                        # a client that sets nothing
        self.assertEqual(code, 200)
        got = self.engine.last_sampling
        for k in ("temperature", "top_p", "top_k", "seed", "experimental_speed_projection"):
            self.assertEqual(got[k], d[k], k)
        self.assertEqual(self.engine.last_max_new, 77)
        code, _ = self.chat(temperature=0.9, max_tokens=5)          # its own values win
        self.assertEqual(self.engine.last_sampling["temperature"], 0.9)
        self.assertEqual(self.engine.last_max_new, 5)
        r = self.svc.with_shared({"messages": []}, "openai")
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertEqual(self.svc.with_shared({"reasoning_effort": "high"}, "openai")["reasoning_effort"], "high")
        self.assertEqual(self.svc.with_shared({}, "anthropic")["output_config"], {"effort": "low"})

    def test_off_again(self):
        self.req("/settings", {"defaults": {"temperature": 0.3}})
        code, b = self.req("/settings", {"defaults": None})
        self.assertEqual((code, b["shared"]), (200, False))
        self.assertFalse(os.path.exists(self.svc.shared_path))
        self.chat()
        self.assertNotIn("temperature", self.engine.last_sampling)

    def test_only_strata_s_own_page_may_set_them(self):
        code, _ = self.req("/settings", None, {"Content-Type": "text/plain"}, raw=b'{"defaults": {"temperature": 1}}')
        self.assertEqual(code, 415)
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, b = self.req("/settings", {"defaults": {"temperature": 9}})
        self.assertEqual(code, 400)
        self.assertIn("temperature", b["error"]["message"])
        self.assertEqual(self.svc.shared, {})
        host = self.base.split("://", 1)[1]
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://" + host})
        self.assertEqual(code, 200)

    def test_they_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}})[0], 401)
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}},
                                      {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""


class WebApp(unittest.TestCase):
    """The web app (PR #22's dashboard idea, rebuilt): its page and files, and GET /metrics."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_page_and_files(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"\"web/app.js\"", body)   # relative since #82 (works behind a path-prefixed proxy)
        for path, want in (("/web/app.js", "javascript"), ("/web/app.css", "text/css"), ("/web/tokens.css", "text/css"),
                           ("/web/components.css", "text/css"), ("/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, ctype, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)

    def test_cache_tab_is_in_the_page(self):
        """The Cache tab's markup (web plan step 6, design §5): the view, the tab button - hidden until
        /metrics says cache.enabled - and its sprite glyph."""
        code, _, body = self.get("/")
        self.assertEqual(code, 200)
        for want in (b'id="view-cache"', b'id="tab-btn-cache"', b'data-tab="cache" hidden',
                     b'aria-controls="view-cache"', b'aria-labelledby="tab-btn-cache"', b"sprite.svg#i-cache"):
            with self.subTest(want=want):
                self.assertIn(want, body)
        self.assertIn(b'id="i-cache"', self.get("/web/sprite.svg")[2])

    def test_cache_tab_files_render(self):
        """app.js carries the Cache render path and app.css the rows it writes - and /cache is still a JSON
        route, never a file route under /web/."""
        code, ctype, body = self.get("/web/app.js")
        self.assertEqual(code, 200)
        self.assertIn("javascript", ctype)
        for want in (b"function renderCache", b"async function loadCache", b"function startCache",
                     b"function stopCache", b"CACHE_METRICS", b"cache-warn--danger"):
            with self.subTest(want=want):
                self.assertIn(want, body)
        self.assertIn(b".cache-grid", self.get("/web/app.css")[2])
        for path in ("/web/cache", "/cache/app.js", "/cache/../web/app.js"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_only_the_app_files_are_served(self):
        for path in ("/web/..%2Fserver.py", "/web/index.html", "/web/test.py", "/fonts/..%2F..%2Fsetup.py",
                     "/fonts/missing.woff2", "/fonts/x.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_metrics(self):
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=10).read()
        code, ctype, body = self.get("/metrics")
        self.assertEqual(code, 200)
        m = json.loads(body)
        for key in ("engine", "live", "requests", "hardware", "hardware_static", "history", "cache"):
            self.assertIn(key, m)
        self.assertEqual(m["engine"]["max_context"], CTX)
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["output_tokens"], 5)
        # this service has no tier at all: the block says so in the shape the Cache tab hides on (design §5)
        self.assertEqual(m["cache"], {"enabled": False})
        # what the Monitor's Cache column reads.  This engine printed no `KV` line, so the row is UNKNOWN - a dash
        # in the page, never "cold" (design §3, note 1).  A tier-on service's row is asserted in CacheWiring.
        self.assertIn("cache", m["requests"][0])
        self.assertIsNone(m["requests"][0]["cache"])

    def test_monitor_cache_column_and_engine_rss_row_are_in_the_page(self):
        """Step 7's Monitor markup: the request table's Cache column - beside `Reused`, which it exists to explain
        - and the engine-RSS row beside the System RAM row.  The `Reused` header and its column are asserted
        untouched: the new column must not have quietly replaced or re-labelled them."""
        code, _, body = self.get("/")
        self.assertEqual(code, 200)
        for want in (b'<th class="num">Reused</th>', b"<th title=\"Where this request's KV came from", b">Cache</th>",
                     b'<td colspan="9" class="muted">No requests yet</td>',
                     b'id="eng-ram-text"', b'id="sp-eng-ram"'):
            with self.subTest(want=want):
                self.assertIn(want, body)
        # the empty row widened: the table has nine columns now
        self.assertEqual(body.count(b'colspan="9"'), 1)
        app = self.get("/web/app.js")[2]
        for want in (b"function cacheCell", b"const CACHE_SRC", b"cacheCell(r.cache)", b'spark("sp-eng-ram"',
                     b'"eng-ram-text"'):
            with self.subTest(want=want):
                self.assertIn(want, app)

    def test_about_cache_card_is_in_the_page(self):
        """The About card (design §5): hidden markup the app fills from the `cache` block already in /metrics -
        About fetches /cache never and walks the store never."""
        code, _, body = self.get("/")
        self.assertEqual(code, 200)
        for want in (b'<div class="st-card" id="card-cache" hidden>', b'<span class="card-title">NVMe cache</span>',
                     b'id="facts-cache"'):
            with self.subTest(want=want):
                self.assertIn(want, body)
        app = self.get("/web/app.js")[2]
        for want in (b'$("card-cache").hidden', b'facts($("facts-cache")', b"layer-split sessions are not cached",
                     b"a promote stages the whole snapshot in RAM at once", b'renderAbout(eng, hw, st, cache)'):
            with self.subTest(want=want):
                self.assertIn(want, app)

    def test_cache_reports_no_tier(self):
        """GET /cache on a server with no KvCache: 200, {"enabled": false}, and a JSON route - not a file route."""
        code, ctype, body = self.get("/cache")
        self.assertEqual(code, 200)
        self.assertIn("application/json", ctype)
        self.assertEqual(json.loads(body), {"enabled": False})
        self.assertEqual(self.get("/cache/")[0], 200)              # the handler rstrips "/": the same route
        for path in ("/cache/app.js", "/cache/../web/app.js"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)      # nothing under /cache is a file route

    def test_cache_needs_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/cache")[0], 401)
            self.assertEqual(self.get("/cache", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/cache", {"x-api-key": "secret"})[0], 200)
        finally:
            self.svc.api_key = ""

    def test_model_discovery_and_props(self):
        svc = self.svc
        previous = svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared
        try:
            svc.engine.max_context = 262144
            svc.sampling_defaults = {"temperature": 1.0, "repetition_penalty": 1.1}
            svc.shared = {"temperature": 0.7, "max_tokens": 4096}
            for vision in (None, object()):
                svc.vision = vision
                for path in ("/models", "/v1/models"):
                    code, _, body = self.get(path)
                    self.assertEqual(code, 200)
                    models = json.loads(body)["data"]
                    self.assertEqual(len(models), 1)
                    model = models[0]
                    self.assertEqual(model["id"], svc.model)
                    self.assertEqual(model["status"]["value"], "loaded")
                    self.assertEqual(model["meta"]["n_ctx"], 262144)
                    self.assertEqual(model["architecture"]["input_modalities"],
                                     ["text", "image"] if vision else ["text"])
                code, _, body = self.get("/props?model=" + svc.model + "&autoload=false")
                self.assertEqual(code, 200)
                props = json.loads(body)
                self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
                self.assertEqual(props["default_generation_settings"]["params"],
                                 {"temperature": 0.7, "repeat_penalty": 1.1, "n_predict": 4096})
                self.assertEqual(props["chat_template"], (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))
                self.assertEqual(props["modalities"]["vision"], vision is not None)
                self.assertEqual(props["total_slots"], 1)
                self.assertFalse(props["models_autoload"])
            svc.shared = {}
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["default_generation_settings"]["params"]["n_predict"], -1)
            self.assertEqual(self.get("/props?model=not-loaded&autoload=true")[0], 404)
        finally:
            svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared = previous

    def test_discovery_needs_the_api_key(self):
        self.svc.api_key = "secret"
        try:
            for path in ("/models", "/v1/models", "/props", "/slots"):
                self.assertEqual(self.get(path)[0], 401)
                self.assertEqual(self.get(path, {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""

    def test_build_model_path_and_slot_status(self):
        engine = self.svc.engine
        engine.model_path = "models/example.gguf"
        engine.info = {"version": "0.1.21"}
        try:
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["model_path"], engine.model_path)
            self.assertEqual(props["build_info"], "Strata 0.1.21")
            for busy in (True, False):
                with self.svc.status_lock:
                    self.svc.status["busy"] = busy
                code, _, body = self.get("/slots")
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body), [{"id": 0, "n_ctx": CTX, "is_processing": busy}])
        finally:
            with self.svc.status_lock:
                self.svc.status["busy"] = False
            del engine.model_path, engine.info
        props = json.loads(self.get("/props")[2])
        self.assertNotIn("build_info", props)
        self.assertNotIn("model_path", props)

    def test_discovery_does_not_restart_a_dead_engine(self):
        self.svc.engine.alive = lambda: False
        try:
            for path in ("/models", "/v1/models"):
                code, _, body = self.get(path)
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["data"], [])
            self.assertEqual(self.get("/props")[0], 503)
            self.assertEqual(json.loads(self.get("/slots")[2]), [])
        finally:
            del self.svc.engine.alive

    def test_metrics_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/metrics")[0], 401)
            self.assertEqual(self.get("/metrics", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/")[0], 200)                  # the page itself asks for the key
        finally:
            self.svc.api_key = ""


class ClockedEngine(MockEngine):
    """The mock engine with StrataEngine's clock: `last` as the engine's DONE line gives it, the conversation cache
    holding the first REUSED tokens of every prompt."""
    REUSED = 5

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:          # as StrataEngine reads its DONE line: also when the server closes the request at a stop token
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * n,
                         "finish": "stop", "reused": min(self.REUSED, len(ids)), "hits": 9, "lookups": 10}


class UsageAndStatus(unittest.TestCase):
    """What clients read besides the text: the part of the prompt the conversation cache held (OpenAI's
    prompt_tokens_details.cached_tokens, Anthropic's cache_read_input_tokens), llama.cpp's timings, GET /v1/status."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = ClockedEngine(tok, "</think>\n\nok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def chat(self, path, stream=False):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
        status, raw = self.request(path, body)
        self.assertEqual(status, 200)
        if not stream:
            return json.loads(raw)
        return [json.loads(line[6:]) for line in raw.decode().splitlines()
                if line.startswith("data: {")]

    def test_openai(self):
        b = self.chat("/v1/chat/completions")
        u, t = b["usage"], b["timings"]
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(t["cache_n"], ClockedEngine.REUSED)
        self.assertEqual(t["prompt_n"] + t["cache_n"], u["prompt_tokens"])
        self.assertEqual(t["predicted_n"], u["completion_tokens"])
        self.assertAlmostEqual(t["prompt_per_second"], t["prompt_n"] / 0.040, delta=0.1)
        self.assertAlmostEqual(t["predicted_per_second"], 50.0, delta=0.1)            # 20 ms a token

    def test_openai_stream(self):
        last = self.chat("/v1/chat/completions", stream=True)[-1]
        self.assertEqual(last["usage"]["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(last["timings"]["cache_n"], ClockedEngine.REUSED)

    def test_anthropic(self):
        u = self.chat("/v1/messages")["usage"]
        self.assertEqual(u["cache_read_input_tokens"], ClockedEngine.REUSED)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], len(self.engine.last_prompt))
        self.assertGreater(u["output_tokens"], 0)

    def test_v1_status(self):
        self.chat("/v1/chat/completions")
        status, raw = self.request("/v1/status")
        self.assertEqual(status, 200)
        s = json.loads(raw)
        self.assertEqual(s["model"], self.svc.model)
        self.assertEqual(s["context"]["max_positions"], CTX)
        self.assertEqual(s["concurrency"]["serving"], 1)
        self.assertFalse(s["vision"]["available"])
        self.assertEqual(s["activity"]["in_flight"], 0)
        self.assertGreaterEqual(s["activity"]["requests"], 1)
        self.assertEqual(s["last_timings"]["cache_n"], ClockedEngine.REUSED)
        self.assertIn("at", s["last_timings"])

    def test_no_clock(self):
        """An engine without a clock (MockEngine): no timings, nothing cached."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            data = json.dumps({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = json.loads(r.read())
            self.assertNotIn("timings", b)
            self.assertEqual(b["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            self.assertIsNone(svc.v1_status()["last_timings"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class TimingsDrafts(unittest.TestCase):
    """`timings` carries the speculative draft counts (PR #83's fields) only when the engine reported them."""

    def test_draft_fields(self):
        base = {"prompt_ms": 100.0, "decode_ms": 200.0, "generated": 20, "reused": 4}
        t = request_timings(24, 20, dict(base, drafts_offered=15, drafts_accepted=11))
        self.assertEqual((t["draft_n"], t["draft_n_accepted"]), (15, 11))
        self.assertEqual((t["prompt_n"], t["cache_n"]), (20, 4))
        self.assertNotIn("draft_n", request_timings(24, 20, base))
        self.assertIsNone(request_timings(24, 20, {}))

# --------------------------------------------------------------------------------- the cache tiers (serve/kvcache.py)
# The exact lines step 2 prints (src/program/generate.cpp:3225 and :3544), used as the parser's contract.
KV_START_LINE = ("KV start=1 entries=104 entries_bytes=213674598400 delta_entries=1 delta_bytes=1181116416 "
                 "cap=107374182400")
KV_REQUEST_LINE = ("KV src=delta resume=4107 promote_ms=1840 promote_bytes=1010893312 staging_bytes=1010893312 "
                   "dump_ms=412 dump_bytes=1184923648 evict=1 evict_bytes=154000384 sweep=0 sweep_bytes=0 "
                   "refused=0 transfer=0 entries=104 entries_bytes=213674598400 delta_entries=1 "
                   "delta_bytes=1181116416 cap=107374182400 checkpoints=3 live=4131 total_dump_bytes=1184923648 "
                   "total_promote_bytes=1010893312 total_refused=0 total_transfer=0 total_evict_bytes=154000384")
KV_FIELDS = {"src", "resume", "promote_ms", "promote_bytes", "staging_bytes", "dump_ms", "dump_bytes", "evict",
             "evict_bytes", "sweep", "sweep_bytes", "refused", "transfer", "entries", "entries_bytes",
             "delta_entries", "delta_bytes", "cap", "checkpoints", "live", "total_dump_bytes",
             "total_promote_bytes", "total_refused", "total_transfer", "total_evict_bytes"}


class KvLine(unittest.TestCase):
    """The engine's `KV` line: typed, unknown keys kept, and nothing a malformed line can raise."""

    def test_the_startup_line(self):
        self.assertEqual(parse_kv(KV_START_LINE),
                         {"start": 1, "entries": 104, "entries_bytes": 213674598400, "delta_entries": 1,
                          "delta_bytes": 1181116416, "cap": 107374182400})

    def test_a_request_line_is_every_field_the_engine_prints(self):
        kv = parse_kv(KV_REQUEST_LINE)
        self.assertEqual(set(kv), KV_FIELDS)
        self.assertEqual(kv["src"], "delta")                      # a value that is not a number stays a string
        self.assertEqual(kv["resume"], 4107)
        self.assertIsInstance(kv["resume"], int)
        self.assertEqual(kv["promote_ms"], 1840.0)                # the two ms fields are floats
        self.assertIsInstance(kv["promote_ms"], float)
        self.assertEqual(kv["dump_ms"], 412.0)
        self.assertEqual(kv["promote_bytes"], 1010893312)
        self.assertEqual(kv["total_evict_bytes"], 154000384)

    def test_an_unknown_key_is_kept(self):
        kv = parse_kv("KV src=none resume=0 future_fact=7 note=whatever")
        self.assertEqual(kv["future_fact"], 7)
        self.assertEqual(kv["note"], "whatever")

    def test_a_truncated_line_keeps_what_it_can(self):
        self.assertEqual(parse_kv("KV src=delta resume= entries 123"), {"src": "delta"})
        self.assertEqual(parse_kv("KV src=ram resume=41 promote"), {"src": "ram", "resume": 41})

    def test_a_non_numeric_value_does_not_raise(self):
        kv = parse_kv("KV resume=abc promote_ms=xyz dump_bytes= src=none")
        self.assertEqual(kv["src"], "none")
        self.assertEqual(kv["resume"], "abc")                     # kept, and every reader coerces it to None
        self.assertEqual(kv["promote_ms"], "xyz")
        self.assertNotIn("dump_bytes", kv)

    def test_no_line_is_no_facts(self):
        for line in ("KV", "", "   ", "DONE 5 10 1.0 2.0 eos 0 0 0 0 0", None):
            with self.subTest(line=line):
                self.assertEqual(parse_kv(line), {})


def snapshot(path: str, version: int, length: int, size: int) -> None:
    """A v3 snapshot the way kv_nvme.cpp writes one: magic, version, then the prefix length L."""
    Path(path).write_bytes(struct.pack("<IIq", 0x5E564D45, version, length) + b"\0" * max(0, size - 16))


def manifest(path: str, version: int, length: int, n_chunks: int, state_key: int, chunk_keys: list[int]) -> None:
    """A delta head the way kv_delta.cpp writes one: the 264-byte header at its pinned offsets, the body's ids
    and image keys, then the chunk refs, then the 8-byte footer."""
    head = bytearray(264)
    struct.pack_into("<II", head, 0, 0x474F4C44, version)
    struct.pack_into("<q", head, 8, length)
    struct.pack_into("<q", head, 16, 1024)                       # BLOCK
    struct.pack_into("<q", head, 24, n_chunks)
    struct.pack_into("<Q", head, 232, state_key)
    refs = b"".join(struct.pack("<Qq", k, i * 1024) for i, k in enumerate(chunk_keys))
    Path(path).write_bytes(bytes(head) + b"\0" * 40 + refs + b"\0" * 8)


def delta_record(path: str, magic: int, size: int) -> None:
    Path(path).write_bytes(struct.pack("<II", magic, 1) + b"\0" * max(0, size - 8))


class CountingKvCache(KvCache):
    """Counts the directory walks, so the throttle is measured instead of assumed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.walks = 0

    def _walk(self, now: float) -> dict:
        self.walks += 1
        return super()._walk(now)


class KvStore(unittest.TestCase):
    """The serve-side walk (design §2.2): per-class counts and bytes, the format version of each head record,
    the delta tier's `.tmp-` residue, and the free space the engine never reports."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = os.path.join(cls.tmp.name, "kvstore")
        now = time.time()
        os.makedirs(cls.root)
        snapshot(os.path.join(cls.root, "kv-1-1.bin"), 3, 4107, 1000)        # promotable, the warmest prefix
        snapshot(os.path.join(cls.root, "kv-1-2.bin"), 3, 100, 500)          # promotable, the coldest
        snapshot(os.path.join(cls.root, "kv-1-3.bin"), 2, 900, 700)          # another build's format version
        Path(os.path.join(cls.root, "kv-1-4.bin")).write_bytes(struct.pack("<IIq", 0x12345678, 3, 5) + b"\0" * 284)
        os.makedirs(os.path.join(cls.root, "delta/chunks"))
        os.makedirs(os.path.join(cls.root, "delta/states"))
        manifest(os.path.join(cls.root, "delta/log-1-1.manifest"), 1, 217, 2, 0x1111, [0x2222, 0x3333])
        delta_record(os.path.join(cls.root, "delta/chunks/0000000000002222.bin"), 0x4B4E4843, 400)
        delta_record(os.path.join(cls.root, "delta/chunks/0000000000003333.bin"), 0x4B4E4843, 400)
        delta_record(os.path.join(cls.root, "delta/states/0000000000001111.bin"), 0x54415453, 200)
        Path(os.path.join(cls.root, "delta/chunks/.tmp-9-0")).write_bytes(b"x" * 123)   # a crash's residue
        Path(os.path.join(cls.root, "delta/.tmp-9-1")).write_bytes(b"x" * 55)
        os.utime(os.path.join(cls.root, "kv-1-1.bin"), (now - 60, now - 60))
        os.utime(os.path.join(cls.root, "kv-1-2.bin"), (now - 7200, now - 7200))
        for name in ("kv-1-3.bin", "kv-1-4.bin"):        # neither is a prefix: they must not move the ages
            os.utime(os.path.join(cls.root, name), (now - 3600, now - 3600))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cache(self) -> CountingKvCache:
        return CountingKvCache(["--kv-nvme", self.root, "--kv-nvme-max", "100", "--kv-delta", "1"])

    def test_per_class_counts_and_bytes(self):
        on_disk = self.cache().scan()["on_disk"]
        self.assertEqual({k: (v["count"], v["bytes"]) for k, v in on_disk.items()},
                         {"snapshots": (4, 2500), "manifests": (1, 344), "chunks": (2, 800),
                          "states": (1, 200), "residue": (2, 178)})

    def test_a_stale_version_is_a_number_and_stays_on_disk(self):
        s = self.cache().scan()
        self.assertEqual(s["stale"], {"count": 1, "version": 2})
        self.assertEqual(s["on_disk"]["snapshots"]["stale"], 1)
        self.assertEqual(s["on_disk"]["snapshots"]["promotable"], 2)      # counted, but not promotable
        self.assertEqual(s["on_disk"]["snapshots"]["foreign"], 1)         # the file with another magic
        self.assertEqual(s["foreign"], 1)
        self.assertTrue(any("version 2" in w for w in s["warnings"]), s["warnings"])

    def test_residue_is_the_delta_tmp_class_and_is_said(self):
        s = self.cache().scan()
        self.assertEqual(s["on_disk"]["residue"]["count"], 2)
        self.assertEqual(s["on_disk"]["residue"]["bytes"], 178)
        self.assertTrue(any(".tmp-" in w for w in s["warnings"]), s["warnings"])

    def test_mtime_ages_per_class(self):
        snaps = self.cache().scan()["on_disk"]["snapshots"]
        self.assertAlmostEqual(snaps["newest_age_s"], 60.0, delta=5)
        self.assertAlmostEqual(snaps["oldest_age_s"], 7200.0, delta=5)
        self.assertLess(snaps["newest_age_s"], snaps["oldest_age_s"])

    def test_disk_space_is_a_real_number(self):
        s = self.cache().scan()
        self.assertGreater(s["disk_free_bytes"], 0)
        self.assertGreaterEqual(s["disk_total_bytes"], s["disk_free_bytes"])

    def test_stored_prefixes_are_lengths_and_sizes_only(self):
        p = self.cache().scan()["prefixes"]
        self.assertEqual([(x["tier"], x["tokens"], x["age_s"] < 3600) for x in p],
                         [("delta", 217, True), ("v3", 4107, True), ("v3", 100, False)])   # newest first
        self.assertEqual(p[0]["records"], {"manifests": 1, "chunks": 2, "states": 1})
        self.assertEqual(p[0]["bytes"], 344 + 800 + 200)          # manifest + its chunks + its State record
        self.assertEqual(p[1]["records"], {"snapshots": 1})
        self.assertEqual(p[1]["bytes"], 1000)
        self.assertNotIn("kv-1-3.bin", str(p))                    # the stale one is not a promotable prefix
        for row in p:
            self.assertEqual(set(row), {"age_s", "tokens", "tier", "kind", "records", "bytes"})

    def test_two_scans_inside_the_throttle_walk_once(self):
        c = self.cache()
        first = c.scan()
        second = c.scan()
        self.assertEqual(c.walks, 1)
        self.assertIs(first, second)                              # the cached result, with its timestamp
        self.assertEqual(first["scanned_at"], second["scanned_at"])
        c.scan()
        c.scan(force=True)
        self.assertEqual(c.walks, 2)

    def test_a_directory_that_is_not_there_is_a_warning_row(self):
        c = KvCache(["--kv-nvme", os.path.join(self.tmp.name, "not-here")])
        s = c.scan()
        self.assertEqual(s["on_disk_files"], 0)
        self.assertEqual(s["prefixes"], [])
        self.assertEqual(s["disk_free_bytes"], None)
        self.assertTrue(any("not there" in w for w in s["warnings"]), s["warnings"])
        self.assertTrue(any("not there" in w for w in c.detail()["warnings"]), c.detail())

    def test_an_unreadable_directory_is_a_warning_row(self):
        if os.name == "nt" or os.geteuid() == 0:
            self.skipTest("a chmod 000 directory is readable for a superuser")
        locked = os.path.join(self.tmp.name, "locked")
        os.makedirs(locked, exist_ok=True)
        os.chmod(locked, 0o000)
        try:
            s = KvCache(["--kv-nvme", locked]).scan()
            self.assertTrue(any("could not be read" in w for w in s["warnings"]), s["warnings"])
        finally:
            os.chmod(locked, 0o755)

    def test_a_path_that_is_a_file_is_a_warning_row(self):
        not_a_dir = os.path.join(self.tmp.name, "a-file")
        Path(not_a_dir).write_bytes(b"x")
        s = KvCache(["--kv-nvme", not_a_dir]).scan()
        self.assertEqual(s["on_disk_files"], 0)
        self.assertTrue(s["warnings"])

    def test_no_kv_nvme_means_no_tier_and_no_work(self):
        c = CountingKvCache(["--model", "x", "--kv-resident", "0"])      # no --kv-nvme: no tier
        self.assertFalse(c.enabled)
        self.assertEqual(c.summary(), {})
        self.assertEqual(c.detail(), {})
        self.assertEqual(c.scan(), {})
        self.assertEqual(c.scan(force=True), {})
        self.assertEqual(c.series(), {})
        self.assertEqual(c.walks, 0)                              # no directory walk at all
        self.assertIsNone(c.observe(parse_kv(KV_REQUEST_LINE), {"finish": "eos"}))
        self.assertIsNone(c.store_state(parse_kv(KV_START_LINE)))
        self.assertEqual(list(c.events), [])

    def test_the_config_facts(self):
        c = KvCache(["--kv-nvme", self.root])                     # the engine's own defaults
        self.assertEqual((c.enabled, c.mode, c.cap_bytes), (True, "delta", 100 * 2**30))
        self.assertIsNone(c.inert_reason)
        c = KvCache(["--kv-nvme", self.root, "--kv-nvme-max", "0", "--kv-delta", "0"])
        self.assertEqual((c.mode, c.cap_bytes), ("v3", 0))        # 0 = unlimited
        c = KvCache(["--kv-nvme", self.root, "--layer-split", "auto"])
        self.assertIn("layer split", c.inert_reason)
        self.assertIn("layer split", c.summary()["warnings"][0])


class KvEvents(unittest.TestCase):
    """What the `KV` lines produce: event rows, and cumulative totals a missing or garbage line cannot corrupt."""

    def cache(self):
        return KvCache(["--kv-nvme", "/tmp/unused-store", "--kv-nvme-max", "100"])

    def test_a_request_line_becomes_event_rows(self):
        c = self.cache()
        c.observe(parse_kv(KV_REQUEST_LINE), {"finish": "eos"})
        rows = c.summary()["events"]
        self.assertEqual([r["kind"] for r in rows], ["promote", "cascade", "evict"])
        self.assertEqual((rows[0]["src"], rows[0]["tokens"], rows[0]["bytes"], rows[0]["ms"]),
                         ("delta", 4107, 1010893312, 1840.0))
        self.assertEqual(rows[1]["ms"], 412.0)
        self.assertEqual(rows[2]["count"], 1)
        self.assertEqual(rows[0]["finish"], "eos")

    def test_a_cold_request_is_not_a_promote(self):
        c = self.cache()
        c.observe(parse_kv("KV src=none resume=0 promote_ms=0 promote_bytes=0 staging_bytes=0 dump_ms=412 "
                           "dump_bytes=1184923648 evict=0 evict_bytes=0 sweep=0 sweep_bytes=0 refused=1 "
                           "transfer=0 entries=1 entries_bytes=1 delta_entries=0 delta_bytes=0 cap=0 "
                           "checkpoints=0 live=1 total_dump_bytes=1 total_promote_bytes=0 total_refused=1 "
                           "total_transfer=0 total_evict_bytes=0"), {"finish": "eos"})
        self.assertEqual([r["kind"] for r in c.summary()["events"]], ["cascade", "refuse"])
        self.assertEqual(c.summary()["totals"]["promotes"], 0)

    def test_a_transfer_failure_is_a_row(self):
        """The engine prints `transfer=1` with `src=none` (generate.cpp:3849): the line is the server's only
        chance to learn the class before the engine dies."""
        c = self.cache()
        c.observe(parse_kv("KV src=none resume=4107 transfer=1 refused=0 entries=1 entries_bytes=1000 "
                           "total_transfer=1"), {"finish": "error"})
        rows = c.summary()["events"]
        self.assertEqual([(r["kind"], r["src"], r["finish"]) for r in rows], [("transfer", "none", "error")])
        self.assertEqual(c.summary()["totals"]["total_transfer"], 1)

    def test_the_startup_line_is_store_state_not_an_event(self):
        c = self.cache()
        c.observe(parse_kv(KV_START_LINE), {"finish": "eos"})
        s = c.summary()
        self.assertEqual(s["events"], [])
        self.assertEqual(s["promotable"]["entries"], 104)         # merged, because it is store state
        self.assertEqual(s["promotable"]["prefixes"], 105)
        self.assertEqual(s["totals"]["requests_with_kv_line"], 0)

    def test_totals_survive_a_missing_a_garbage_and_a_partial_line(self):
        c = self.cache()
        c.observe(parse_kv(KV_REQUEST_LINE), {"finish": "eos"})
        before = dict(c.summary()["totals"])
        self.assertEqual(before["total_dump_bytes"], 1184923648)
        for kv in (None, {}, parse_kv("KV"), parse_kv("KV resume=abc"), parse_kv("garbage"), [], "KV src=x"):
            with self.subTest(kv=kv):
                c.observe(kv, {"finish": "eos"})
                after = c.summary()["totals"]
                for key in ("total_dump_bytes", "total_promote_bytes", "total_refused", "total_transfer",
                            "total_evict_bytes"):
                    self.assertEqual(after[key], before[key], key)   # last seen wins, and a bad line changes none
                self.assertEqual(c.summary()["events"], c.summary()["events"][:3])   # no row from a bad line

    def test_totals_are_the_engine_s_cumulative_fields_not_a_sum(self):
        c = self.cache()
        c.observe(parse_kv("KV src=none dump_bytes=100 total_dump_bytes=100"), {"finish": "eos"})
        c.observe(parse_kv("KV src=none dump_bytes=250 total_dump_bytes=350"), {"finish": "eos"})
        self.assertEqual(c.summary()["totals"]["total_dump_bytes"], 350)
        self.assertEqual(c.summary()["totals"]["requests"], 2)
        self.assertEqual(c.summary()["totals"]["requests_with_kv_line"], 2)

    def test_summary_and_detail_keys(self):
        c = self.cache()
        c.observe(parse_kv(KV_REQUEST_LINE), {"finish": "eos"})
        s = c.summary()
        self.assertEqual(set(s), {"enabled", "mode", "dir", "cap_bytes", "inert_reason", "promotable",
                                  "ram_tier", "totals", "events", "series", "warnings"})
        self.assertEqual(s["promotable"]["bytes"], 213674598400 + 1181116416)   # cap accounting, from the engine
        self.assertEqual(s["ram_tier"], {"checkpoints": 3, "live_tokens": 4131})
        d = c.detail()
        self.assertTrue(set(s) <= set(d))
        self.assertTrue({"scanned_at", "on_disk", "on_disk_bytes", "stale", "prefixes",
                         "disk_free_bytes"} <= set(d), sorted(d))
        self.assertNotEqual(d["promotable"]["bytes"], d["on_disk_bytes"])        # the two are never merged

    def test_series_samples_one_point_per_call(self):
        c = self.cache()
        c.observe(parse_kv(KV_REQUEST_LINE), {"finish": "eos"})
        first = c.series()
        self.assertEqual(set(first), {"store_bytes", "write_mb", "read_mb", "warm"})
        self.assertEqual([len(v) for v in first.values()], [1, 1, 1, 1])
        self.assertEqual(first["warm"], [105])                     # entries + delta_entries
        self.assertEqual(first["read_mb"], [round(1010893312 / 2**20, 2)])
        second = c.series()
        self.assertEqual([len(v) for v in second.values()], [2, 2, 2, 2])
        self.assertEqual(len(c.summary()["series"]["warm"]), 2)   # summary reads them without appending


# ------------------------------------------------------------------ the wiring: the KV line into the request path
# (web plan step 4): _pump captures the line, Service.run's finally records it, main() attaches the KvCache.
KV_COLD_LINE = ("KV src=none resume=0 promote_ms=0 promote_bytes=0 staging_bytes=0 dump_ms=412 "
                "dump_bytes=1184923648 evict=0 evict_bytes=0 sweep=0 sweep_bytes=0 refused=1 transfer=0 "
                "entries=1 entries_bytes=1 delta_entries=0 delta_bytes=0 cap=0 checkpoints=0 live=1 "
                "total_dump_bytes=1184923648 total_promote_bytes=0 total_refused=1 total_transfer=0 "
                "total_evict_bytes=0")
DONE_LINE = "DONE 2 3 1.0 2.0 eos 0 0 0 0 0 0 0"


class CacheEngine(MockEngine):
    """A fake that plays the wiring the way `StrataEngine` does it: `last_kv` is cleared when the request starts
    and filled after the cascade (where the pump sees the line), `last` is its `DONE` record, `cache` the KvCache
    the service attached.  A turn with no scripted line leaves `last_kv` None."""

    def __init__(self, tok, script, max_context, kv_lines=()):
        super().__init__(tok, script, max_context=max_context)
        self.kv_lines, self.turn = list(kv_lines), 0
        self.last, self.last_kv, self.cache, self.in_request = {}, None, None, False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_kv, self.in_request = None, True
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                yield t
            self.last_kv = self.kv_lines[min(self.turn, len(self.kv_lines) - 1)] if self.kv_lines else None
            self.last = {"generated": 0, "prompt_tokens": len(ids), "prompt_ms": 12.0, "decode_ms": 100.0,
                         "finish": "eos"}
        finally:
            self.in_request = False
            self.turn += 1


class CacheWiring(unittest.TestCase):
    """What one finished request leaves behind: its `cache` row in the history, the disk-tier tokens in the
    totals, and the event rows the attached KvCache built.  No line is recorded as cold (design §3, note 1)."""

    def metrics_after_request(self, kv_lines=()):
        tok = ByteTokenizer()
        eng = CacheEngine(tok, "</think>\n\nhello", CTX, kv_lines)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.cache = KvCache(["--kv-nvme", "/tmp/unused-store", "--kv-nvme-max", "100", "--kv-delta", "1"])
        eng.cache = svc.cache                           # what main() does beside svc.gpu_index
        httpd = serve(svc, port=0)
        try:
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 5}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            return svc, svc.metrics()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_promote_is_recorded_on_the_request_and_in_the_totals(self):
        svc, m = self.metrics_after_request([parse_kv(KV_REQUEST_LINE)])
        self.assertEqual(m["requests"][0]["cache"],
                         {"src": "delta", "resume": 4107, "promote_ms": 1840.0,
                          "promote_bytes": 1010893312, "staging_bytes": 1010893312})
        self.assertEqual(m["totals"]["reused_from_disk"], 4107)      # the tokens the disk tier resumed
        self.assertEqual(m["requests"][0]["reused"], None)           # the RAM tier's `reused` is still its own
        rows = svc.cache.summary()["events"]
        self.assertEqual([r["kind"] for r in rows], ["promote", "cascade", "evict"])
        self.assertEqual(svc.cache.summary()["totals"]["requests_with_kv_line"], 1)
        self.assertEqual(svc.cache.summary()["promotable"]["entries"], 104)   # the line's store fields too

    def test_a_cold_request_is_recorded_and_adds_nothing_to_the_disk_total(self):
        _, m = self.metrics_after_request([parse_kv(KV_COLD_LINE)])
        self.assertEqual(m["requests"][0]["cache"]["src"], "none")
        self.assertEqual(m["requests"][0]["cache"]["resume"], 0)
        self.assertEqual(m["totals"]["reused_from_disk"], 0)

    def test_a_ram_tier_hit_is_not_disk_reuse(self):
        _, m = self.metrics_after_request([parse_kv("KV src=ram resume=64 promote_ms=0 promote_bytes=0 "
                                                    "entries=1 entries_bytes=1 cap=0 total_refused=0")])
        self.assertEqual(m["requests"][0]["cache"]["src"], "ram")
        self.assertEqual(m["totals"]["reused_from_disk"], 0)

    def test_no_kv_line_is_unknown_never_cold(self):
        svc, m = self.metrics_after_request([])
        self.assertIsNone(m["requests"][0]["cache"])
        self.assertEqual(m["totals"]["reused_from_disk"], 0)
        self.assertEqual(svc.cache.summary()["events"], [])
        self.assertEqual(svc.cache.summary()["totals"]["requests"], 1)
        self.assertEqual(svc.cache.summary()["totals"]["requests_with_kv_line"], 0)

    def test_an_engine_that_never_heard_of_the_tiers_still_answers(self):
        """MockEngine (and any Engine without the wiring): no `last_kv` attribute at all, and the request path
        does not care."""
        tok = ByteTokenizer()
        svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 5}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            m = svc.metrics()
            self.assertIsNone(m["requests"][0]["cache"])
            self.assertEqual(m["totals"]["reused_from_disk"], 0)
        finally:
            httpd.shutdown()
            httpd.server_close()


class CacheRoute(unittest.TestCase):
    """Step 5: the `cache` block in GET /metrics and the read-only GET /cache, over a tier-on service.

    The split is the whole design (design §4): /metrics carries KvCache.summary() - memory only - and /cache
    carries KvCache.detail(), which is the only caller of scan()/_walk().  The cache here is a CountingKvCache, so
    "cheap" is measured (walks before and after each route) rather than asserted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = os.path.join(cls.tmp.name, "kvstore")
        now = time.time()
        os.makedirs(cls.root)
        snapshot(os.path.join(cls.root, "kv-1-1.bin"), 3, 4107, 1000)          # promotable
        snapshot(os.path.join(cls.root, "kv-1-2.bin"), 2, 900, 700)            # another build's version: stale
        os.makedirs(os.path.join(cls.root, "delta/chunks"))
        os.makedirs(os.path.join(cls.root, "delta/states"))
        manifest(os.path.join(cls.root, "delta/log-1-1.manifest"), 1, 217, 1, 0x1111, [0x2222])
        delta_record(os.path.join(cls.root, "delta/chunks/0000000000002222.bin"), 0x4B4E4843, 400)
        delta_record(os.path.join(cls.root, "delta/states/0000000000001111.bin"), 0x54415453, 200)
        Path(os.path.join(cls.root, "delta/chunks/.tmp-9-0")).write_bytes(b"x" * 123)   # a crash's residue
        os.utime(os.path.join(cls.root, "kv-1-1.bin"), (now - 3600, now - 3600))
        os.utime(os.path.join(cls.root, "delta/log-1-1.manifest"), (now - 60, now - 60))

        tok = ByteTokenizer()
        cls.engine = CacheEngine(tok, "\\n</think>\\n\\nhello", CTX, [parse_kv(KV_REQUEST_LINE)])
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.cache = CountingKvCache(["--kv-nvme", cls.root, "--kv-nvme-max", "100", "--kv-delta", "1"])
        cls.svc.cache = cls.cache
        cls.engine.cache = cls.cache                            # what main() does beside svc.gpu_index
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    def post(self):
        """One real request through the HTTP layer: the pump sees the scripted KV line, Service.run's finally
        observes it.  Nothing here calls observe() directly."""
        before = len(self.cache.events)
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=30).read()
        return list(self.cache.events)[before:]     # the rows THIS request produced

    def test_metrics_carries_the_summary_and_walks_nothing(self):
        # the class shares one service, so every assertion about a growing counter is relative to this request
        ev, reqs, disk = len(self.cache.events), self.cache.totals["requests_with_kv_line"], \
            self.svc.totals["reused_from_disk"]
        rows = self.post()
        self.assertEqual([r["kind"] for r in rows], ["promote", "cascade", "evict"])
        walks = self.cache.walks
        code, m = self.get("/metrics")
        self.assertEqual(code, 200)
        self.assertEqual(self.cache.walks, walks)                   # summary() did no filesystem work
        c = m["cache"]
        self.assertEqual((c["enabled"], c["mode"], c["dir"]), (True, "delta", self.root))
        self.assertEqual(c["cap_bytes"], 100 * 2**30)
        self.assertIsNone(c["inert_reason"])
        self.assertEqual(c["promotable"]["entries"], 104)           # the engine's own store fields, from its line
        self.assertEqual(c["promotable"]["bytes"], 213674598400 + 1181116416)
        self.assertEqual(c["ram_tier"], {"checkpoints": 3, "live_tokens": 4131})
        self.assertEqual([r["kind"] for r in c["events"]][ev:], ["promote", "cascade", "evict"])
        self.assertEqual(c["totals"]["requests_with_kv_line"], reqs + 1)
        self.assertEqual(c["totals"]["total_evict_bytes"], 154000384)
        self.assertEqual(set(c["series"]), {"store_bytes", "write_mb", "read_mb", "warm"})
        self.assertEqual(c["warnings"], [])
        # the walk's tables are not here, and neither is the request row the Monitor's Cache column will read
        for key in ("on_disk", "on_disk_bytes", "on_disk_files", "prefixes", "stale", "foreign",
                    "disk_free_bytes", "disk_total_bytes", "scanned_at"):
            self.assertNotIn(key, c)
        self.assertEqual(m["requests"][0]["cache"]["src"], "delta")
        self.assertEqual(m["totals"]["reused_from_disk"], disk + 4107)

    def test_cache_carries_the_walk(self):
        reqs = self.cache.totals["requests_with_kv_line"]
        rows = self.post()
        walks = self.cache.walks
        code, d = self.get("/cache")
        self.assertEqual(code, 200)
        self.assertEqual(self.cache.walks, walks + 1)               # the walk happened, here and only here
        self.get("/cache")
        self.assertEqual(self.cache.walks, walks + 1)               # throttled: the second read reuses it
        self.assertEqual({k: (v["count"], v["bytes"]) for k, v in d["on_disk"].items()},
                         {"snapshots": (2, 1700), "manifests": (1, 328), "chunks": (1, 400),
                          "states": (1, 200), "residue": (1, 123)})
        self.assertEqual(d["on_disk"]["snapshots"]["promotable"], 1)
        self.assertEqual(d["stale"], {"count": 1, "version": 2})    # on disk, not promotable (design §5.3)
        self.assertEqual([(p["tier"], p["tokens"], p["bytes"]) for p in d["prefixes"]],
                         [("delta", 217, 328 + 400 + 200), ("v3", 4107, 1000)])   # newest first
        self.assertGreater(d["disk_free_bytes"], 0)
        self.assertTrue(any("version 2" in w for w in d["warnings"]), d["warnings"])
        self.assertEqual([r["kind"] for r in d["events"]][-3:], [r["kind"] for r in rows])
        self.assertEqual(d["totals"]["requests_with_kv_line"], reqs + 1)
        self.assertNotEqual(d["promotable"]["bytes"], d["on_disk_bytes"])   # the two books are never merged
        self.assertTrue({"enabled", "mode", "dir", "cap_bytes", "totals", "series"} <= set(d))

    def test_cache_is_read_only(self):
        """No store mutation anywhere in the serve protocol (design §4, §8): /cache answers GET and nothing else."""
        data = b'{"dir": "x"}'
        walks = self.cache.walks
        req = urllib.request.Request(self.base + "/cache", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                self.fail(f"POST /cache answered {r.status}")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)
        self.assertEqual(self.cache.walks, walks)                   # and it did not walk anything either


class StubProc:
    """A stand-in engine process: `stdout` is a scripted list of lines, `stdin` records what was written.
    With `gate`, the pump does not outrun the request: no line is printed before `generate()` has written its
    GEN line (which is what sets `in_request`), so a scripted KV line arrives exactly as a real one does."""

    def __init__(self, lines, gate=False, pid=4242):
        self.written, self.started = [], threading.Event()
        self.pid = pid                                # a stand-in process has an id too (telemetry reads it)
        self.stdout = self._read(lines, gate)

    def _read(self, lines, gate):
        for line in lines:
            if gate:
                self.started.wait(10)
            yield line

    def kill(self):
        pass

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    @property
    def stdin(self):
        return self

    def write(self, s):
        self.written.append(s)
        self.started.set()

    def flush(self):
        pass


def scripted_engine(lines, in_request=False, cache=None, pump="sync"):
    """A `StrataEngine` with no process, so the pump's routing over a scripted stdout is what is under test.
    `pump="sync"` runs it on this thread (a scripted stdout ends, so it returns with the queue already filled);
    `pump="thread"` runs it as the real engine does - on its own thread, gated on the request's GEN line."""
    eng = object.__new__(StrataEngine)
    eng.proc = StubProc(lines, gate=pump == "thread")
    eng.lines, eng.can_stop, eng.ended = queue.Queue(), True, False
    eng.last, eng.last_kv, eng.store_kv, eng.cache = {}, None, {}, cache
    eng.in_request, eng.progress, eng.spawn = in_request, None, None
    if pump == "sync":
        eng._pump()
    else:
        threading.Thread(target=eng._pump, daemon=True).start()
    return eng


def queued(eng):
    """What the request's line queue actually holds (the sentinel that closes the stream is not a line)."""
    out = []
    while not eng.lines.empty():
        line = eng.lines.get_nowait()
        if line is not None:
            out.append(line)
    return out


class KvPump(unittest.TestCase):
    """`_pump` is where the line is captured: the early-stop drain discards queue entries, and the startup line
    arrives after READY.  A KV line never reaches the request's queue, and nothing it says can raise."""

    def test_the_startup_line_is_store_state_not_an_event(self):
        cache = KvCache(["--kv-nvme", "/tmp/unused-store", "--kv-nvme-max", "100"])
        eng = scripted_engine([KV_START_LINE], cache=cache)
        self.assertIsNone(eng.last_kv)                              # no request was running
        self.assertEqual((eng.store_kv["entries"], eng.store_kv["cap"]), (104, 107374182400))
        self.assertEqual(queued(eng), [])                           # never forwarded as a token line
        self.assertEqual(cache.summary()["promotable"]["entries"], 104)   # a line before any request still lands

    def test_a_startup_line_read_before_the_cache_exists_is_replayed_at_attach(self):
        """main() attaches the KvCache AFTER the engine is running, so the pump can consume `KV start=1` with no
        cache to feed - `attach_cache()` must replay what the pump already saw.  Without the replay a tier-on
        server reports `enabled: true` with every store fact null until some request line happens to arrive;
        found on a live server, and only there - every test above attaches the cache by hand, BEFORE the line."""
        eng = scripted_engine([KV_START_LINE])                   # no cache exists yet: the pump reads it alone
        self.assertIsNone(eng.cache)
        self.assertEqual(eng.store_kv["entries"], 104)           # it kept what it saw
        cache = KvCache(["--kv-nvme", "/tmp/unused-store", "--kv-nvme-max", "100"])
        self.assertIsNone(cache.summary()["promotable"]["entries"])          # the store state is not there yet
        eng.attach_cache(cache)
        self.assertEqual(cache.summary()["promotable"]["entries"], 104)      # replayed at attach
        self.assertEqual(cache.summary()["promotable"]["delta_bytes"], 1181116416)
        eng.attach_cache(None)                                   # and detaching, or never attaching, is fine

    def test_a_request_line_while_in_request_becomes_this_request_s_event(self):
        eng = scripted_engine(["T 11", KV_REQUEST_LINE, "T 12", DONE_LINE], in_request=True)
        self.assertEqual((eng.last_kv["src"], eng.last_kv["resume"]), ("delta", 4107))
        self.assertEqual(queued(eng), ["T 11", "T 12", DONE_LINE])
        self.assertEqual(eng.store_kv["entries"], 104)              # its store fields are merged too

    def test_a_src_line_outside_a_request_is_store_state(self):
        eng = scripted_engine([KV_REQUEST_LINE], in_request=False)  # e.g. the drain of a request that already ended
        self.assertIsNone(eng.last_kv)
        self.assertEqual(eng.store_kv["delta_entries"], 1)

    def test_a_garbage_line_and_a_huge_value_cannot_break_the_pump(self):
        huge = "KV src=delta resume=" + "9" * 400
        eng = scripted_engine(["KV ", "KV resume=abc", "KV " + "x" * 4000, "KV = = =", huge, "T 7", DONE_LINE],
                              in_request=True)
        self.assertEqual(queued(eng), ["T 7", DONE_LINE])
        self.assertEqual(eng.last_kv["src"], "delta")
        self.assertEqual(eng.last_kv["resume"], int("9" * 400))     # typed, and costs nothing

    def test_a_kv_line_cannot_break_the_request_path(self):
        """The gate makes the scripted lines arrive while the request runs, as a real engine prints them."""
        eng = scripted_engine(["T 11", "KV ", "KV resume=abc", KV_REQUEST_LINE, "T 12", DONE_LINE],
                              pump="thread")
        out = list(eng.generate([1, 2, 3], 4, {}, threading.Event()))
        self.assertEqual(out, [11, 12])                             # the tokens, and nothing else
        self.assertEqual((eng.last["generated"], eng.last["finish"]), (2, "eos"))   # DONE parsed as before
        self.assertEqual(eng.last_kv["src"], "delta")
        self.assertEqual(eng.last_kv["resume"], 4107)
        self.assertFalse(eng.in_request)                            # cleared by generate's finally
        self.assertTrue(eng.proc.written[0].startswith("GEN "))


class RestartKeepsTheCache(unittest.TestCase):
    """A transfer failure restarts the engine (design §5.2), and the KvCache the service attached must survive
    it: `restart()` calls `__init__` again, which would otherwise drop it."""

    class Engine(StrataEngine):
        """The real `restart()` against a stand-in process: only `__init__` is a stand-in, and it resets `cache`
        to None exactly as the real one does.  Each stand-in process gets its own pid, as a real one would."""

        pids = iter(range(4242, 4252))

        def __init__(self, exe="strata", args=(), cwd=None, log=None, env=None):
            self.spawn, self.info, self.cache = (exe, list(args), cwd, log, env), {}, None
            self.proc, self.ended = StubProc([], pid=next(self.pids)), False

    def test_restart_keeps_the_cache_attached_and_informed(self):
        eng = self.Engine()
        eng.info = {"kv": 1234}
        cache = eng.cache = KvCache(["--kv-nvme", "/tmp/unused-store", "--kv-nvme-max", "100"])
        cache.store_state(parse_kv(KV_START_LINE))
        eng.restart()
        self.assertIs(eng.cache, cache)
        self.assertEqual(eng.info["kv"], 1234)
        self.assertEqual(eng.cache.summary()["promotable"]["entries"], 104)   # what it knew survives the restart
        eng._kv_line(KV_START_LINE)                               # the new process re-reports its store
        self.assertEqual(eng.cache.summary()["promotable"]["entries"], 104)

    def test_the_pid_follows_the_process_restart(self):
        """`Engine.pid` is a property, not a number captured at startup: `restart()` replaces the process, and
        telemetry reads the engine's RSS through a lookup for exactly that reason (design §5.2, §6)."""
        eng = self.Engine()

        def lookup():                                     # what Service.start_telemetry passes to Telemetry
            return getattr(eng, "pid", None)

        first = eng.pid
        self.assertEqual(lookup(), first)
        eng.restart()
        self.assertNotEqual(eng.pid, first)                     # the old id is not what it reports now
        self.assertEqual(lookup(), eng.pid)                     # and the lookup followed it to the new process


# ------------------------------------------------------------------ the engine's own RSS (serve/telemetry.py)
try:
    import psutil  # noqa: F401
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


def reaped_pid():
    """A pid that was real and is now waited on: reading it is the `NoSuchProcess` path."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


class EngineRss(unittest.TestCase):
    """`rss_used` - the engine process's own resident bytes (design §6's RAM transient, the bump a ~2 GB promote
    staging makes).  The pid is resolved per sample, and a reading that cannot be taken is ABSENT: a missing
    reading must never look like a healthy zero."""

    @staticmethod
    def sampler(pid):
        from serve.telemetry import Telemetry
        return Telemetry(pid=pid)

    @unittest.skipUnless(HAVE_PSUTIL, "the engine's RSS is a psutil reading")
    def test_a_live_pid_gives_rss_used_and_a_history_deque(self):
        tel = self.sampler(os.getpid())
        s = tel.sample()
        self.assertIn("rss_used", s)
        self.assertGreater(s["rss_used"], 0)
        time.sleep(1.2)                                          # one pass of the sampler thread's key list
        snap = tel.snapshot()
        self.assertIn("rss_used", snap["now"])
        self.assertTrue(snap["history"]["rss_used"])             # a deque like every other sampled series

    @unittest.skipUnless(HAVE_PSUTIL, "the engine's RSS is a psutil reading")
    def test_no_engine_process_is_no_reading_never_zero(self):
        for case, pid in (("no pid at all", None), ("an engine with no process", lambda: None),
                          ("a process that is gone", reaped_pid())):
            with self.subTest(case=case):
                self.assertNotIn("rss_used", self.sampler(pid).sample())

    @unittest.skipUnless(HAVE_PSUTIL, "the engine's RSS is a psutil reading")
    def test_the_pid_is_resolved_every_sample_not_captured(self):
        """restart() replaces the process (a transfer failure is exactly that path, design §5.2): a captured int
        would keep reading the dead one."""
        running = [os.getpid()]
        tel = self.sampler(lambda: running[0])
        self.assertIn("rss_used", tel.sample())
        running[0] = None                                        # the engine it pointed at is gone
        self.assertNotIn("rss_used", tel.sample())

    def test_without_psutil_the_key_is_absent(self):
        tel = self.sampler(os.getpid())
        tel.ps = None                                            # the import failed on this machine
        self.assertNotIn("rss_used", tel.sample())

    def test_the_service_hands_the_sampler_a_lookup_not_a_value(self):
        """Service.start_telemetry passes `lambda: getattr(self.engine, "pid", None)`; a mock engine has no
        process, so the Monitor's Engine RAM row stays empty rather than reading the server's own pid."""
        tok = ByteTokenizer()
        svc = Service(RecordingEngine(tok, "\n</think>\n\nhello", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.start_telemetry()
        self.assertTrue(callable(svc.telemetry.pid))
        self.assertIsNone(svc.telemetry.pid())
        self.assertNotIn("rss_used", svc.telemetry.sample())

    @unittest.skipUnless(HAVE_PSUTIL, "the engine's RSS is a psutil reading")
    def test_a_real_engine_process_reaches_the_sampled_key(self):
        """The whole path for a real `StrataEngine`: its `pid` property -> the lookup -> `rss_used`.  The stand-in
        process is this test process, so the reading is this process's own."""
        eng = scripted_engine([])
        eng.proc.pid = os.getpid()
        svc = Service(eng, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.start_telemetry()
        self.assertEqual(svc.telemetry.pid(), os.getpid())
        self.assertGreater(svc.telemetry.sample()["rss_used"], 0)


if __name__ == "__main__":
    unittest.main()
