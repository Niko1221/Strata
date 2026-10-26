"""Request inspection through HTTP; no native model or GPU is required."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                self.assertNotIn("api_monitor", json.loads(r.read()))   # the default answer is unchanged
            for path in ("/api/requests", "/api-monitor", "/api/context"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        # a request the server refuses is an error record: the strict mode (fit_max_tokens off) refuses a max_tokens
        # that overshoots the context, the default shortens it (serve/test_server.py MaxTokens)
        self.svc.fit_max_tokens = False
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.svc.fit_max_tokens = True
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_context_of_an_engine_that_does_not_report_it(self):
        code, body = self.request("/api/context")
        self.assertEqual(code, 200)
        self.assertEqual((body["available"], body["slots"]), (False, []))
        self.assertTrue(self.request("/health")[1]["api_monitor"])

    def test_context_tokens_and_their_text(self):
        # "é" is two bytes, one token each: the first token's piece is empty, the second completes it
        ids = [256, *"hi é".encode(), 999, 257]
        self.engine.context_view = lambda: [{"name": "Slot 0", "slot": 0, "state": "idle", "tokens": list(ids)},
                                            {"name": "Slot 1", "slot": 1, "state": "idle", "tokens": None}]
        code, body = self.request("/api/context")
        self.assertEqual(code, 200)
        a, b = body["slots"]
        self.assertEqual(a["tokens"], ids)
        self.assertEqual(a["pieces"][:6], ["<|im_start|>", "h", "i", " ", "", "é"])
        self.assertEqual(len(a["pieces"]), len(ids))
        self.assertEqual(a["pieces"][-1], "<|im_end|>")
        self.assertEqual(b["pieces"], None)                 # not known (a disk session was restored)
        self.assertEqual(body["max_context"], self.svc.reported_ctx())

    def test_context_needs_the_key(self):
        self.svc.api_key = "k"
        self.assertEqual(self.request("/api/context")[0], 401)
        self.assertEqual(self.request("/api/context", headers={"Authorization": "Bearer k"})[0], 200)

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        for el in ("ctx-view", "ctx-refresh", "ctx-save", "ctx-binary", "ctx-slot", "ctx-sum", "ctx-note", "ctx-pager",
                   "ctx-prev", "ctx-next", "ctx-range"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', script)

    def test_the_main_page_links_to_the_monitor(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("api-monitor-link", "req-monitor"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)
        self.assertIn('href="api-monitor"', html)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class HeldTokens(unittest.TestCase):
    """The Context view's tokens without --batch: what the engine keeps after a GEN, from the lines it printed."""

    @staticmethod
    def bare(lines, held=(), batch=0):
        """A StrataEngine without a process: its output lines are queued up front."""
        import io
        import queue
        from types import SimpleNamespace
        from serve.server import StrataEngine
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        engine.lines, engine.can_stop, engine.max_context, engine.batch = queue.Queue(), True, 4096, batch
        engine.held, engine.held_note, engine.held_busy = list(held), None, False
        for line in lines:
            engine.lines.put(line)
        return engine

    def held(self, engine):
        (slot,) = engine.context_view()
        return slot["tokens"], slot["unknown"]

    def test_the_engine_keeps_the_prompt_and_all_but_the_last_token(self):
        engine = self.bare(["RESUME 2", "T 7", "T 8", "T 9", "DONE 3 3 1 1 stop 0 0 1"])
        gen = engine.generate([1, 2, 3], 8, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        self.assertEqual(engine.context_view()[0]["state"], "busy")      # live while the request runs
        self.assertEqual(self.held(engine)[0], [1, 2, 3, 7])
        self.assertEqual(list(gen), [8, 9])
        self.assertEqual(engine.context_view()[0]["state"], "idle")
        self.assertEqual(self.held(engine), ([1, 2, 3, 7, 8], None))     # 9 was printed, never fed

    def test_a_stop_counts_the_tokens_decoded_after_it(self):
        # the client stops after the first token; the engine had decoded two more before it saw the STOP
        engine = self.bare(["T 7", "T 8", "T 9", "DONE 3 3 1 1 cancel 0 0 0"])
        gen = engine.generate([1, 2, 3], 8, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        gen.close()
        self.assertIn("STOP", engine.proc.stdin.getvalue())
        self.assertEqual(self.held(engine), ([1, 2, 3, 7, 8], None))

    def test_deepseek_has_fed_every_token_it_printed_when_stopped(self):
        engine = self.bare(["T 7", "T 8", "DONE 2 3 1 1 cancel 0 0 0"])
        engine.stop_feeds_last = True                     # set_family("deepseek41")
        gen = engine.generate([1, 2, 3], 8, {}, threading.Event())
        next(gen)
        gen.close()
        self.assertEqual(self.held(engine), ([1, 2, 3, 7, 8], None))
        engine = self.bare(["T 7", "T 8", "DONE 2 3 1 1 stop 0 0 0"])  # an end token: not fed, as the others
        engine.stop_feeds_last = True
        list(engine.generate([1, 2, 3], 8, {}, threading.Event()))
        self.assertEqual(self.held(engine), ([1, 2, 3, 7], None))

    def test_a_stop_while_the_prompt_is_read_is_not_known(self):
        engine = self.bare(["PP 512 4096 100", "DONE 0 4096 100 0 cancel 0 0 0"])
        cancel = threading.Event()
        gen = engine.generate(list(range(4096)), 8, {}, cancel)
        self.assertIsNone(next(gen))                      # a prompt chunk: a heartbeat
        cancel.set()
        self.assertEqual(list(gen), [])
        self.assertEqual(self.held(engine), (None, "stopped"))

    def test_an_error_leaves_the_cache_as_it_was(self):
        engine = self.bare(["ERR bad request"], held=[5, 6])
        with self.assertRaises(ValueError):
            list(engine.generate([1, 2, 3], 8, {}, threading.Event()))
        self.assertEqual(self.held(engine), ([5, 6], None))

    def test_non_finite_logits_drop_it(self):
        from serve.server import EngineNonFinite
        engine = self.bare(["T 7", "DONE 1 3 1 1 nonfinite 0 0 0"])
        with self.assertRaises(EngineNonFinite):
            list(engine.generate([1, 2, 3], 8, {}, threading.Event()))
        self.assertEqual(self.held(engine), (None, "dropped"))

    def test_a_busy_batch_slot_is_shown_when_it_ends(self):
        engine = self.bare([], held=[9], batch=2)
        engine.slot_held, engine.slot_busy, engine.slot_used = [[1, 2], []], [False, True], [5.0, 6.0]
        engine.slot_live = [None, {"state": "decoding"}]
        view = [{k: e[k] for k in ("name", "state", "tokens", "unknown")} for e in engine.context_view()]
        self.assertEqual(view, [
            {"name": "Solo session", "state": "idle", "tokens": [9], "unknown": None},   # a request alone runs here
            {"name": "Slot 0", "state": "idle", "tokens": [1, 2], "unknown": None},
            {"name": "Slot 1", "state": "decoding", "tokens": None, "unknown": "busy"}])
        engine.own_solo_session = False                   # the Qwen3.6 path: a request alone runs in a slot's session
        self.assertEqual(engine.context_view()[0]["name"], "Last request alone (in one of the slots' sessions)")


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


if __name__ == "__main__":
    unittest.main()
