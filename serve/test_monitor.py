"""Request inspection through HTTP; no native model or GPU is required."""
import json
import shutil
import subprocess
import tempfile
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
            for path in ("/api/requests", "/api-monitor"):
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
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
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

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    PRESSURE = ("strata serve: conversation cache: skip parking (physical RAM admission; need 2128 MiB plus "
                "32768 MiB floor, 34449 MiB available, source meminfo; evicted 38, 0 still parked)\n")

    def test_pressure_refusal_clears_empty_cache_without_inventing_eviction_total(self):
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK.replace("parked=1", "parked=38").replace("evictions=0", "evictions=41"),
                           encoding="utf-8")
            c = ConvCacheLog()
            before = c.poll(str(log), 0)
            with log.open("a", encoding="utf-8") as f:
                f.write(self.PRESSURE)
            after = c.poll(str(log), 0)
            self.assertEqual((after["parked"], after["bytes"]), (0, 0))
            # The 38 counts only physical-pressure evictions. make_room() can have evicted others first.
            self.assertIsNone(after["evictions"])
            for key in ("parks", "restores", "last_event", "last_tokens", "last_at"):
                self.assertEqual(after[key], before[key], key)
            self.assertEqual(c.poll(str(log), 0), after)  # polling again must not count the refusal twice

    def test_pressure_survivors_have_unknown_bytes_until_an_authoritative_event(self):
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK + self.PRESSURE.replace("0 still parked", "1 still parked"), encoding="utf-8")
            c = ConvCacheLog()
            state = c.poll(str(log), 0)
            self.assertEqual(state["parked"], 1)
            self.assertIsNone(state["bytes"])
            self.assertIsNone(state["evictions"])
            with log.open("a", encoding="utf-8") as f:
                f.write(self.RESTORE)
            state = c.poll(str(log), 0)
            self.assertEqual((state["parked"], state["bytes"], state["restores"]), (2, 209715200, 1))
            self.assertIsNone(state["evictions"])  # a restore line does not report that cumulative counter
            with log.open("a", encoding="utf-8") as f:
                f.write(self.PARK.replace("evictions=0", "evictions=79"))
            state = c.poll(str(log), 0)
            self.assertEqual((state["parked"], state["bytes"], state["evictions"], state["parks"]),
                             (1, 104857600, 79, 2))

    def test_refusals_without_counts_do_not_keep_stale_occupancy(self):
        from serve.server import ConvCacheLog
        refusals = (
            "skip parking (snapshot 2128 MiB exceeds available budget)",
            "skip parking (physical RAM admission; need 2128 MiB plus 32768 MiB floor, or telemetry unavailable)",
            "skip parking (RAM floor after capture: 32768 MiB needed, 31000 MiB available, source meminfo)",
            "allocation failed; skip parking",
        )
        for refusal in refusals:
            with self.subTest(refusal=refusal), tempfile.TemporaryDirectory() as d:
                log = Path(d) / "strata.log"
                log.write_text(self.PARK + "strata serve: conversation cache: " + refusal + "\n", encoding="utf-8")
                state = ConvCacheLog().poll(str(log), 0)
                self.assertIsNone(state["parked"])
                self.assertIsNone(state["bytes"])
                self.assertIsNone(state["evictions"])
                self.assertEqual((state["parks"], state["restores"]), (1, 0))

    def test_pressure_refusal_waits_for_a_complete_line_and_resets_on_new_run(self):
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK + self.PRESSURE.rstrip("\n"), encoding="utf-8")
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), 0)["parked"], 1)
            with log.open("a", encoding="utf-8") as f:
                f.write("\n")
            self.assertEqual(c.poll(str(log), 0)["parked"], 0)
            state = c.poll(str(log), log.stat().st_size)
            self.assertEqual((state["parked"], state["bytes"], state["evictions"], state["parks"]), (0, 0, 0, 0))

    def test_minimum_length_skip_preserves_counters_and_reports_its_count(self):
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK + "strata serve: conversation cache: skip parking "
                           "(100 tokens, below the 4096 minimum; parked=1)\n", encoding="utf-8")
            state = ConvCacheLog().poll(str(log), 0)
            self.assertEqual((state["parked"], state["parks"], state["evictions"]), (1, 1, 0))
            self.assertIsNone(state["bytes"])  # the count says nothing about retained K/V bytes

    def test_dropped_copies_do_not_keep_stale_bytes(self):
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK + "strata serve: conversation cache: dropped 1 superseded copy of this "
                           "conversation; parked=0\n", encoding="utf-8")
            state = ConvCacheLog().poll(str(log), 0)
            self.assertEqual((state["parked"], state["parks"], state["evictions"]), (0, 1, 0))
            self.assertIsNone(state["bytes"])  # unlike the pressure gate, this can still hold retained K/V

    def test_metrics_reports_pressure_refusal_and_recovers_on_next_park(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK, encoding="utf-8")
            tok = ByteTokenizer()
            engine = MockEngine(tok, "Hello.", max_context=4096)
            engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 50}
            engine.log_path, engine.log_start = str(log), 0
            svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
            httpd = serve(svc, port=0)
            url = f"http://127.0.0.1:{httpd.server_address[1]}/metrics"
            def read():
                with urllib.request.urlopen(url, timeout=10) as response:
                    return json.loads(response.read())["conversation_cache"]
            try:
                self.assertEqual(read()["parked"], 1)
                with log.open("a", encoding="utf-8") as f:
                    f.write(self.PRESSURE.replace("evicted 38", "evicted 1"))
                state = read()
                self.assertEqual((state["parked"], state["bytes"], state["parks"]), (0, 0, 1))
                self.assertIsNone(state["evictions"])
                with log.open("a", encoding="utf-8") as f:
                    f.write(self.PARK.replace("evictions=0", "evictions=1"))
                state = read()
                self.assertEqual((state["parked"], state["bytes"], state["evictions"], state["parks"]),
                                 (1, 104857600, 1, 2))
            finally:
                httpd.shutdown()
                httpd.server_close()

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_card_hides_unknown_bars_and_shows_them_again_when_known(self):
        source = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        render = source[source.index("function renderConvCache("):source.index("function renderTotals(")]
        script = ("const elements = {}; const $ = id => elements[id] ||= {style: {}};\n"
                  "const fmt = n => n == null ? 'unknown' : String(n); const gb = fmt;\n"
                  "const facts = () => {}; const since = () => '';\n" + render +
                  "const base = {enabled: true, slots: 50, budget_mib: 1024};\n"
                  "renderConvCache({...base, parked: null, bytes: null, evictions: null});\n"
                  "const unknown = JSON.parse(JSON.stringify(elements));\n"
                  "renderConvCache({...base, parked: 0, bytes: 0, evictions: 0});\n"
                  "const empty = JSON.parse(JSON.stringify(elements));\n"
                  "renderConvCache({...base, parked: 25, bytes: 536870912, evictions: 1});\n"
                  "console.log(JSON.stringify({unknown, empty, known: elements}));")
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        states = json.loads(result.stdout)
        for key in ("cc-slots-bar", "cc-mem-bar"):
            self.assertTrue(states["unknown"][key].get("hidden", False))
            self.assertFalse(states["empty"][key]["hidden"])
            self.assertEqual(states["empty"][key]["style"]["width"], "0%")
            self.assertFalse(states["known"][key]["hidden"])
            self.assertEqual(states["known"][key]["style"]["width"], "50%")
        self.assertIn("unknown", states["unknown"]["cc-slots-text"]["textContent"])
        self.assertIn("unknown", states["unknown"]["cc-mem-text"]["textContent"])

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
