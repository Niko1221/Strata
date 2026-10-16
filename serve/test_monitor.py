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

    def test_the_disk_and_system_prompt_metrics(self):
        from serve.server import ConvCacheLog, conversation_cache_view
        import tempfile
        disk = ("strata serve: conversation cache: spill dir ready (2 conversations, 300 MiB, when-full=evict-oldest, "
                "max-age=0 d, 1 oversized kept, 0 stale kept, 0 foreign kept, 0 orphans kept, 3 disk evictions, "
                "0 age evictions)\n")
        spilled = "strata serve: conversation cache: spilled 4000 tokens (120 MiB); disk=420 MiB disk_evictions=4\n"
        sysp = ("strata serve: system prompt cache: ready (2 variants, 300 MiB, slots=2, max-age=0 d, 0 foreign kept, "
                "0 stale kept, 0 orphans kept, 0 oversized kept)\n"
                "strata serve: system prompt cache: hit 5000 tokens (2 variants)\n"
                "strata serve: system prompt cache: prefix changed (miss, 2 variants live); reprocessing and rewriting\n"
                "strata serve: system prompt cache: persisted the 2048-token root (1 variants, 120 MiB)\n"
                "strata serve: system prompt cache: shutdown: 2 variants, 300 MiB, hits=3 misses=1, tokens_saved=7000, "
                "hash_changes=1, evicted_by_space=0, evicted_by_age=2\n")
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(disk + spilled + sysp, encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
        self.assertEqual((st["disk"]["conversations"], st["disk"]["bytes"], st["disk"]["evictions"]), (2, 420 * 1048576, 4))
        self.assertEqual(st["disk"]["spills"], 1)
        self.assertEqual(st["disk"]["kept"]["oversized"], 1)
        self.assertEqual((st["sysprompt"]["variants"], st["sysprompt"]["bytes"]), (2, 300 * 1048576))
        self.assertEqual((st["sysprompt"]["hits"], st["sysprompt"]["misses"]), (3, 1))
        self.assertEqual((st["sysprompt"]["hash_changes"], st["sysprompt"]["slots"]), (1, 2))
        self.assertEqual((st["sysprompt"]["evicted_by_age"], st["sysprompt"]["tokens_saved"]), (2, 7000))
        # the view adds the two objects and keeps every key it had
        view = conversation_cache_view({"conversation_cache_mib": 8192, "conversation_cache_slots": 4,
                                        "conversation_cache_disk_mib": 8192, "system_prompt_cache_mib": 2048}, [], {}, st)
        self.assertEqual((view["enabled"], view["budget_mib"], view["slots"], view["parked"]), (True, 8192, 4, 0))
        self.assertTrue(view["disk"]["enabled"])
        self.assertEqual((view["disk"]["bytes"], view["disk"]["spills"], view["disk"]["restores"]), (420 * 1048576, 1, 0))
        # the parking split by destination, one row per configured tier (RAM, and the single spill folder today)
        self.assertEqual([d["tier"] for d in view["destinations"]], ["ram", "disk"])
        self.assertEqual([d["label"] for d in view["destinations"]], ["RAM", "SSD 1"])
        self.assertEqual((view["destinations"][0]["conversations"], view["destinations"][1]["conversations"]), (0, 2))
        self.assertEqual(view["destinations"][0]["budget_mib"], 8192)
        self.assertFalse(conversation_cache_view({"conversation_cache_mib": 0}, [], {}, {})["destinations"])
        self.assertEqual((view["system_prompt_cache"]["hits"], view["system_prompt_cache"]["variants"]),
                         (3, 2))
        for key in ("hits", "misses", "variants", "bytes", "evicted_by_age", "evicted_by_space", "hash_changes"):
            self.assertIn(key, view["system_prompt_cache"])

    def test_the_system_prompt_folder_and_budget_come_from_the_info_line(self):
        from serve.server import parse_info_line, conversation_cache_view
        # the engine reports the folder it opened and the budget it runs with (after its own defaults) on the INFO
        # line, QUOTED like the spill folder because a path can hold spaces
        info = {}
        parse_info_line(info, 'INFO context=16384 vram_free_mib=12000 conversation_cache_mib=8192 '
                              'system_prompt_cache_dir="D:\\Sys Cache\\f5" system_prompt_cache_mib=2048 '
                              'system_prompt_cache_slots=2 engine=0.1.99')
        self.assertEqual(info["system_prompt_cache_dir"], r"D:\Sys Cache\f5")   # the folder's spaces kept
        self.assertEqual(info["system_prompt_cache_mib"], 2048)                 # a number, not the bare string
        sp = conversation_cache_view(info, [], {}, {})["system_prompt_cache"]   # no log yet: the fields alone say it
        self.assertEqual((sp["enabled"], sp["dir"], sp["budget_mib"]),
                         (True, r"D:\Sys Cache\f5", 2048))

    def test_an_engine_without_the_system_prompt_fields_leaves_them_null(self):
        from serve.server import ConvCacheLog, conversation_cache_view
        import tempfile
        with tempfile.TemporaryDirectory() as d:                    # an engine from before the fields
            log = Path(d) / "strata.log"
            log.write_text("strata serve: system prompt cache: ready (2 variants, 300 MiB, slots=2, max-age=0 d, "
                           "0 foreign kept, 0 stale kept, 0 orphans kept, 0 oversized kept)\n", encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
        sp = conversation_cache_view({"conversation_cache_mib": 8192}, [], {}, st)["system_prompt_cache"]
        self.assertIsNone(sp["dir"])                                # unknown, not invented
        self.assertIsNone(sp["budget_mib"])
        self.assertTrue(sp["enabled"])                             # the log still proves the cache ran
        self.assertEqual(sp["variants"], 2)                        # and a figure that did arrive is kept

    def test_an_unreported_counter_is_not_a_measured_zero(self):
        from serve.server import ConvCacheLog, conversation_cache_view
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            # the authoritative "shutdown:" line (hits=/misses=/tokens_saved=) is only written when the engine exits
            # cleanly, and it usually does not: a counter it never reported is UNKNOWN, not a measured 0
            log.write_text("strata serve: system prompt cache: ready (2 variants, 300 MiB, slots=2, max-age=0 d, "
                           "0 foreign kept, 0 stale kept, 0 orphans kept, 0 oversized kept)\n"
                           "strata serve: system prompt cache: hit 5000 tokens (2 variants)\n", encoding="utf-8")
            sp = conversation_cache_view({"conversation_cache_mib": 8192}, [], {},
                                         ConvCacheLog().poll(str(log), 0))["system_prompt_cache"]
            self.assertIsNone(sp["tokens_saved"])                  # never reported: not displayed as 0
            self.assertIsNone(sp["hash_changes"])
            self.assertIsNone(sp["misses"])
            self.assertEqual(sp["hits"], 1)                        # the hit line did arrive: a real count
            self.assertEqual(sp["variants"], 2)
            # and a counter the engine DID report as zero is a measurement: "unknown" and "measured zero" differ
            log.write_text("strata serve: system prompt cache: ready (2 variants, 300 MiB, slots=2, max-age=0 d, "
                           "0 foreign kept, 0 stale kept, 0 orphans kept, 0 oversized kept)\n"
                           "strata serve: system prompt cache: shutdown: 2 variants, 300 MiB, hits=0 misses=0, "
                           "tokens_saved=0, hash_changes=0, evicted_by_space=0, evicted_by_age=0\n", encoding="utf-8")
            sp = conversation_cache_view({"conversation_cache_mib": 8192}, [], {},
                                         ConvCacheLog().poll(str(log), 0))["system_prompt_cache"]
            self.assertEqual((sp["hits"], sp["tokens_saved"]), (0, 0))     # measured zeros, not None
            self.assertIsNotNone(sp["tokens_saved"])

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note", "cc-dest", "state-slots"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


class ParkingLifecycleAndAdmission(unittest.TestCase):
    """The pieces the on-disk tier added under the card and that the series left untested: the engine's INFO line
    into `info` (parse_info_line), the disk load / discard / park-mirror lines that move ConvCacheLog's `loading`
    and its disk tier, `spill_on` (the park-mode mirror flag and its absence), and `live.admission` - the field that
    says whether an admission is reading a new prompt or loading a parking back from disk, or nothing at all."""

    LOAD = "strata serve: conversation cache: loading 4096 tokens from disk\n"
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=1 bytes=104857600\n"
    MIRROR = ("strata serve: conversation cache: park mirror queued (4000 tokens, copy 1.2 ms); writes=3 collapsed=1 "
              "throttled=0 disk=420 MiB disk_evictions=4 conversations=2\n")

    def test_parse_info_line_reads_the_engines_info_line(self):
        from serve.server import parse_info_line
        info = {}
        parse_info_line(info, 'INFO logit_bias=1 context=16384 kv=int8 vram_free_mib=12000 '
                              'conversation_cache_mib=8192 conversation_cache_disk_mib=2048 conversation_cache_disk=1 '
                              'conversation_cache_spill_dir="D:\\Strata Cache\\spill" engine=0.1.99')
        self.assertEqual(info["context"], 16384)                    # a number becomes an int
        self.assertEqual(info["kv"], "int8")                        # a word stays a word
        self.assertEqual(info["engine"], "0.1.99")                  # a version is not a number
        self.assertEqual(info["conversation_cache_disk"], 1)
        self.assertEqual(info["conversation_cache_spill_dir"], r"D:\Strata Cache\spill")   # the folder's spaces kept
        # a key the parser was not written for still lands; three keys here prove it stops at the right tokens
        parse_info_line(info, "INFO a_new_knob=17 context=4096 conversation_cache_mib=0")
        self.assertEqual((info["a_new_knob"], info["context"], info["conversation_cache_mib"]), (17, 4096, 0))
        # an engine with no spill folder quotes an empty one: the key right after it still parses
        parse_info_line(info, 'INFO conversation_cache_spill_dir="" engine=0.1.100')
        self.assertEqual((info["conversation_cache_spill_dir"], info["engine"]), ("", "0.1.100"))

    def test_parse_info_line_leaves_a_short_line_alone(self):
        from serve.server import parse_info_line
        info = {"context": 4096}
        before = dict(info)
        parse_info_line(info, "")                                   # nothing to read at all
        parse_info_line(info, "INFO")                              # the tag with no key=value after it
        self.assertEqual(info, before)                              # nothing invented, nothing raised

    def test_the_disk_load_sets_and_a_discard_clears_the_loading(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.LOAD, encoding="utf-8")
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), 0)["loading"], 4096)   # a parking is streaming back from disk
            with open(log, "a", encoding="utf-8") as f:              # the image was refused: no load runs now
                f.write("strata serve: conversation cache: discard unreadable disk conversation (bad header)\n")
            self.assertEqual(c.poll(str(log), 0)["loading"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.LOAD)
            self.assertEqual(c.poll(str(log), 0)["loading"], 4096)
            with open(log, "a", encoding="utf-8") as f:              # the load's other close: it did not fit
                f.write("strata serve: conversation cache: disk conversation exceeds the RAM cache budget "
                        "after loading\n")
            self.assertEqual(c.poll(str(log), 0)["loading"], 0)

    def test_the_restored_line_ends_the_disk_load(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.LOAD, encoding="utf-8")
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), 0)["loading"], 4096)
            with open(log, "a", encoding="utf-8") as f:              # the load's own end: the parking is seated
                f.write(self.RESTORE)
            st = c.poll(str(log), 0)
            self.assertEqual((st["loading"], st["restores"]), (0, 1))

    def test_the_park_mirror_line_moves_the_disk_tier(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.MIRROR, encoding="utf-8")
            disk = ConvCacheLog().poll(str(log), 0)["disk"]
            self.assertTrue(disk["ready"])                           # the mirror line alone proves the tier is on
            self.assertEqual((disk["bytes"], disk["evictions"]), (420 * 1048576, 4))
            self.assertEqual((disk["conversations"], disk["spills"]), (2, 1))

    def test_spill_on_is_read_and_the_older_line_leaves_it_absent(self):
        from serve.server import ConvCacheLog, conversation_cache_view
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("strata serve: conversation cache: spill dir ready (2 conversations, 300 MiB, "
                           "when-full=evict-oldest, max-age=0 d, spill-on=park, divergence=64 tok, park-throttle=30 s, "
                           "1 oversized kept, 0 stale kept, 0 foreign kept, 0 orphans kept, 3 disk evictions, "
                           "0 age evictions)\n", encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            self.assertEqual(st["disk"]["spill_on"], "park")         # the park-mode mirror flag
            view = conversation_cache_view({"conversation_cache_mib": 8192, "conversation_cache_disk_mib": 8192,
                                            "conversation_cache_spill_dir": r"D:\spill"}, [], {}, st)
            self.assertEqual(view["disk"]["spill_on"], "park")       # the card and its disk-row both carry it
            self.assertEqual((view["disk"]["when_full"], view["disk"]["max_age_days"]), ("evict-oldest", 0))
            self.assertEqual(view["disk"]["kept"]["oversized"], 1)
            disk_row = next(x for x in view["destinations"] if x["tier"] == "disk")
            self.assertEqual((disk_row["spill_on"], disk_row["dir"]), ("park", r"D:\spill"))
            # an engine from before --conversation-cache-spill-on writes the line with no spill-on=: it stays absent
            log.write_text("strata serve: conversation cache: spill dir ready (0 conversations, 0 MiB, "
                           "when-full=evict-oldest, max-age=0 d, 1 oversized kept, 0 stale kept, 0 foreign kept, "
                           "0 orphans kept, 0 disk evictions, 0 age evictions)\n", encoding="utf-8")
            self.assertIsNone(ConvCacheLog().poll(str(log), 0)["disk"]["spill_on"])

    def test_live_admission_through_metrics(self):
        from serve.server import admission_field
        self.assertIsNone(admission_field("generating", {"loading": 4096}))   # the field is only about an admission
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=4096)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                log = Path(d) / "strata.log"
                log.write_text("", encoding="utf-8")
                engine.log_path, engine.log_start = str(log), 0

                def admission():
                    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                        return json.loads(r.read())["live"]["admission"]

                self.assertIsNone(admission())                       # idle: no admission, so nothing to be loading
                svc.status["busy"] = True                            # an admission is running (before its first token)
                self.assertEqual(admission(), "reading")             # ...a new prompt
                with open(log, "a", encoding="utf-8") as f:          # ...or a parking streaming back from disk
                    f.write(self.LOAD)
                self.assertEqual(admission(), "loading")
                with open(log, "a", encoding="utf-8") as f:
                    f.write(self.RESTORE)                            # the load ended: the field must not stay stuck
                self.assertEqual(admission(), "reading")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_page_treats_a_missing_admission_field_as_no_admission(self):
        js = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        # live.admission is read only when it is a string: null, or the field missing on an older shim, means "no
        # word to add" and the slot keeps its own - it must never print "undefined" or throw
        self.assertIn('typeof live.admission === "string"', js)
        self.assertIn('admission === "loading" ? "loading" : "reading"', js)


class Cascade(unittest.TestCase):
    """Delta 6: the cascade (RAM -> warm -> cold) as the Monitor sees it. The engine names the level it writes to,
    so a row is drawn per level that exists - and with NO warm level the log, the rows and the labels are exactly
    the ones of a single disk tier (the byte-compatibility case, asserted here explicitly)."""

    WARM = ("strata serve: conversation cache: warm dir ready (12 conversations, 9800 MiB, mib=10240, "
            "when-full=evict-oldest, max-age=0 d, 5 degraded to cold, 1 oversized kept, 0 stale kept, "
            "0 foreign kept, 0 orphans kept)\n")
    SPILL = "strata serve: conversation cache: spilled 4000 tokens (120 MiB); disk=420 MiB disk_evictions=4\n"
    MIRROR = ("strata serve: conversation cache: park mirror queued (4000 tokens, copy 1.2 ms); writes=3 collapsed=1 "
              "throttled=0 disk=420 MiB disk_evictions=4 conversations=2\n")
    SPILL_WARM = "strata serve: conversation cache: spilled 4000 tokens (120 MiB); warm=9600 MiB warm_degraded=6\n"
    MIRROR_WARM = ("strata serve: conversation cache: park mirror queued (4000 tokens, copy 1.2 ms); writes=3 "
                   "collapsed=1 throttled=0 warm=9600 MiB warm_degraded=6 conversations=15\n")
    DEGRADED = ("strata serve: conversation cache: degraded 2 conversations to the cold level; warm=15 conversations "
                "/ 9600 MiB; cold=3 conversations / 240 MiB, cold_evictions=1\n")

    def _poll(self, text):
        import tempfile
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(text, encoding="utf-8")
            return ConvCacheLog().poll(str(log), 0)

    def test_the_warm_line_and_its_own_writes_are_read(self):
        st = self._poll(self.WARM + self.SPILL_WARM + self.MIRROR_WARM)
        w = st["warm"]
        self.assertTrue(w["ready"])
        self.assertEqual((w["spills"], w["conversations"]), (2, 15))       # the mirror line's own count wins
        self.assertEqual((w["bytes"], w["degraded"]), (9600 * 1048576, 6))
        self.assertEqual((w["budget_mib"], w["kept"]["oversized"]), (10240, 1))
        self.assertFalse(st["disk"]["ready"])                              # the cold level wrote nothing yet

    def test_the_degradation_moves_both_levels(self):
        st = self._poll(self.WARM + self.DEGRADED)
        w, d = st["warm"], st["disk"]
        self.assertEqual((w["degraded"], w["conversations"]), (5 + 2, 15))  # the ready line's count plus the event
        self.assertEqual((w["bytes"], d["bytes"]), (9600 * 1048576, 240 * 1048576))
        self.assertEqual((d["conversations"], d["evictions"]), (3, 1))      # the cold level's row moves too

    def test_the_three_rows_with_a_warm_level_and_two_without_it(self):
        from serve.server import conversation_cache_view
        empty = {}
        st = self._poll(self.WARM + self.DEGRADED)
        info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4, "conversation_cache_disk_mib": 460800,
                "conversation_cache_spill_dir": r"D:\AI\servers\strata\slots\xl",
                "conversation_cache_warm_dir": r"C:\AI\servers\strata\warm", "conversation_cache_warm_mib": 10240}
        view = conversation_cache_view(info, [], {}, st)
        self.assertEqual([d["tier"] for d in view["destinations"]], ["ram", "warm", "cold"])
        self.assertEqual([d["label"] for d in view["destinations"]], ["RAM", "SSD 1", "SSD 2"])
        warm_row, cold_row = view["destinations"][1], view["destinations"][2]
        self.assertEqual((warm_row["conversations"], warm_row["budget_mib"], warm_row["dir"]),
                         (15, 10240, r"C:\AI\servers\strata\warm"))
        self.assertEqual((cold_row["conversations"], cold_row["budget_mib"], cold_row["dir"]),
                         (3, 460800, r"D:\AI\servers\strata\slots\xl"))
        self.assertEqual(view["warm"]["degraded"], 7)
        # byte-compatibility: NO warm level configured -> the single disk tier keeps its own row and its label
        view2 = conversation_cache_view({k: v for k, v in info.items() if "warm" not in k}, [], {},
                                        self._poll(self.SPILL + self.MIRROR))
        self.assertEqual([d["tier"] for d in view2["destinations"]], ["ram", "disk"])
        self.assertEqual([d["label"] for d in view2["destinations"]], ["RAM", "SSD 1"])
        self.assertEqual(view2["destinations"][1]["bytes"], 420 * 1048576)
        self.assertFalse(view2["warm"]["enabled"])                        # and no warm row is invented
        self.assertEqual(len(conversation_cache_view(empty, [], {}, {})["destinations"]), 0)

    def test_a_warm_dir_with_spaces_survives_the_info_line(self):
        from serve.server import parse_info_line, conversation_cache_view
        info = {}
        parse_info_line(info, 'INFO conversation_cache_mib=8192 conversation_cache_spill_dir="" '
                              'conversation_cache_warm_dir="C:\\Strata Cache\\warm" conversation_cache_warm_mib=10240 '
                              'engine=0.1.99')
        self.assertEqual(info["conversation_cache_warm_dir"], r"C:\Strata Cache\warm")   # the folder's spaces kept
        self.assertEqual((info["conversation_cache_warm_mib"], info["engine"]), (10240, "0.1.99"))
        view = conversation_cache_view(info, [], {}, {})
        self.assertTrue(view["warm"]["enabled"])
        self.assertEqual(view["warm"]["dir"], r"C:\Strata Cache\warm")

    def test_a_cold_load_is_an_admission_in_progress(self):
        from serve.server import admission_field
        st = self._poll("strata serve: conversation cache: loading 4096 tokens from cold\n")
        self.assertEqual(st["loading"], 4096)                             # the engine said "cold": it is loading
        self.assertEqual(admission_field("reading", st), "loading")
        st2 = self._poll("strata serve: conversation cache: loading 4096 tokens from nothing\n")
        self.assertEqual(st2["loading"], 0)                              # an unknown source is not an admission


class SlotsOnTheCards(unittest.TestCase):
    """The batch slots on the Monitor's two cards: the Model state list - one entry per CONFIGURED slot, the idle
    ones at 0 t/s - and the Context fill clocks, one small gauge per slot under the aggregate. The engine's /metrics
    carries live.slots (one entry per batch slot) and engine.batch_slots; a server that sends neither leaves both
    cards standing instead of breaking them."""

    def _metrics(self, engine):
        svc = Service(engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                return json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_metrics_carries_one_entry_per_configured_slot(self):
        engine = MockEngine(ByteTokenizer(), "ok", max_context=4096)
        engine.batch = 4
        engine.info = {"batch_slots": 4}
        engine.slots_view = lambda: [
            {"slot": 0, "state": "decoding", "prompt_tokens": 512, "generated": 128, "tok_s": 15.2},
            {"slot": 1, "state": "idle", "held_tokens": 0},
            {"slot": 2, "state": "reading", "prompt_tokens": 1024, "generated": 0},
            {"slot": 3, "state": "idle", "held_tokens": 4096},
        ]
        m = self._metrics(engine)
        self.assertEqual(m["engine"]["batch_slots"], 4)
        self.assertEqual(m["live"]["parallel"], 4)
        self.assertEqual([s["slot"] for s in m["live"]["slots"]], [0, 1, 2, 3])   # every slot the config declares
        self.assertEqual(m["live"]["slots"][1], {"slot": 1, "state": "idle", "held_tokens": 0})
        self.assertEqual(m["live"]["slots"][2]["state"], "reading")
        self.assertEqual(m["live"]["slots"][3]["held_tokens"], 4096)             # the idle slot's held state is there

    def test_an_engine_without_batch_slots_sends_no_slot_list(self):
        m = self._metrics(MockEngine(ByteTokenizer(), "ok", max_context=4096))
        self.assertNotIn("slots", m["live"])           # an engine without batching: no list to draw - the cards degrade
        self.assertNotIn("batch_slots", m["engine"])

    def test_the_state_card_lists_every_configured_slot(self):
        js = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        # one entry per slot the configuration declares, idle included and reading "0 t/s": a --batch 4 engine shows
        # s#1..s#4, never only the ones working
        self.assertIn('if (s.state === "idle") return `${slot} 0 t/s`', js)
        self.assertNotIn("active.length >= 2", js)     # the old "only 2+ active slots" gate is gone

    def test_the_context_card_has_one_clock_per_slot(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="ctx-slots"', html)          # where the small clocks land
        self.assertIn("ctx-mini", js)                  # the small clock markup; the big one stays the aggregate
        # each slot's own share of the same context: the request it runs, or the tokens it holds for a next turn
        self.assertIn('s.state === "idle" ? (s.held_tokens || 0) : ((s.prompt_tokens || 0) + (s.generated || 0))', js)

    def test_the_page_treats_a_missing_slots_array_as_no_slots(self):
        js = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        # live.slots is read only when it IS an array: null, or the field missing on an older shim (an engine without
        # batching), means "no slots" - the state line stays empty and no clocks are drawn, and nothing prints
        # "undefined" or throws
        self.assertIn("Array.isArray(live.slots) ? live.slots.filter((s) => s) : []", js)
        self.assertIn("left.hidden = !list.length", js)   # both flank columns hide together; the big clock stays


class TheAggregateClockAndTheSlotTexts(unittest.TestCase):
    """The two shape fixes on those cards. Context fill: the big clock is the SUM of the small ones - the same
    live.slots entries, over the capacity the same list declares - so its percentage and the `used / capacity`
    under it explain the small clocks instead of contradicting them (43 + 0 + 0 + 0 % is 11%). Model state: the
    per-slot texts sit on the card's top right, right aligned and evenly spaced by a flex gap, each in its own
    element; the phase text keeps the left of the row. The small clocks are 1.5x the 54px they were drawn at,
    label included."""

    WEB = Path(__file__).parent / "web"

    def _files(self):
        return ((self.WEB / "app.js").read_text(encoding="utf-8"),
                (self.WEB / "app.css").read_text(encoding="utf-8"),
                (self.WEB / "index.html").read_text(encoding="utf-8"))

    def test_the_big_clock_adds_the_slots_up(self):
        js, _, _ = self._files()
        # both clocks call the same slotContextTokens, and the aggregate's denominator follows the same list: the
        # slots it drew, times one context each
        self.assertIn("used = slots.reduce((sum, s) => sum + slotContextTokens(s), 0)", js)
        self.assertIn("capacity = ctx * slots.length", js)
        self.assertIn('$("ctx-sub").textContent = capacity ?', js)
        self.assertIn("${kfmt(used)} / ${ctxfmt(capacity)}", js)
        # an engine without slots still shows the single request's own fill: the same sum with nothing to add
        self.assertIn('else if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);', js)

    def test_the_slot_texts_are_evenly_spaced_against_the_right_edge(self):
        js, css, html = self._files()
        # one element per entry: the gap spaces them, evenly for every width - a joined string would collapse to
        # single spaces (HTML) and read as one run
        self.assertIn("slotsLine.replaceChildren(...parts.map", js)
        self.assertNotIn('parts.join(" · ")', js)
        self.assertIn("display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--st-s-4);", css)
        self.assertRegex(css, r"\.state-slots\[hidden\] \{ display: none; \}")   # the class would beat [hidden] alone
        # the row keeps what was on its left: the phase text first, then the right side, slots over the running total
        row = html[html.index('class="state-card__row"'):html.index('id="state-progress"')]
        self.assertLess(row.index('id="state-label"'), row.index("state-card__tail"))
        self.assertLess(row.index('id="state-slots"'), row.index('id="state-detail"'))

    def test_the_small_clocks_grew_one_and_a_half_times(self):
        _, css, _ = self._files()
        tokens = (self.WEB / "tokens.css").read_text(encoding="utf-8")
        self.assertIn("--st-fs-xs:12px;", tokens)
        self.assertIn("--st-fs-lg:18px;", tokens)                          # 12 x 1.5, a token, not a number
        self.assertIn(".ctx-mini { position: relative; width: calc(54px * 1.5); height: calc(54px * 1.5); }", css)
        self.assertIn(".ctx-mini__gauge { width: calc(54px * 1.5); height: calc(54px * 1.5); }", css)
        self.assertIn(".ctx-mini__slot { font-size: var(--st-fs-lg);", css)   # the label grew with the ring
        self.assertIn(".ctx-mini__pct { font-size: var(--st-fs-lg);", css)
        # the clocks now live in a column that flanks the big one, not in the wrapped row that sat under it
        self.assertIn(".ctx-slots { display: flex; flex-direction: column; align-items: center; gap: var(--st-s-2); flex: none; }", css)


class TheTachometerLayout(unittest.TestCase):
    """The Context fill card's disposition: the aggregate clock keeps the middle and the per-slot clocks flank it,
    one column a side and vertically centred on it - a dashboard tachometer, not the row of clocks that sat under
    the big one.  The left column takes the odd one (2 of 3, 1 of 2, the lone one of 1); a right column with
    nothing to draw stays hidden, so the card never reserves a gap for a clock that is not there.  Sizes and
    numbers are untouched: this is shape only."""

    WEB = Path(__file__).parent / "web"

    def _files(self):
        return ((self.WEB / "app.js").read_text(encoding="utf-8"),
                (self.WEB / "app.css").read_text(encoding="utf-8"),
                (self.WEB / "index.html").read_text(encoding="utf-8"))

    def test_the_small_clocks_flank_the_big_one(self):
        js, css, html = self._files()
        # the wrapper and its three cells: a left column, the big clock, a right column - in that order
        self.assertIn('class="ctx-dials"', html)
        self.assertIn('id="ctx-slots"', html) and self.assertIn('id="ctx-slots-right"', html)
        dials = html[html.index('class="ctx-dials"'):html.index('class="bar-row"')]
        self.assertLess(dials.index('id="ctx-slots"'), dials.index('class="ctx-gauge"'))
        self.assertLess(dials.index('class="ctx-gauge"'), dials.index('id="ctx-slots-right"'))
        # the row centres its three cells and holds their middles level; the columns are a stack a side
        self.assertIn(".ctx-dials { display: flex; align-items: center; justify-content: center; gap: var(--st-s-2);", css)
        self.assertIn(".ctx-slots { display: flex; flex-direction: column; align-items: center;", css)
        self.assertIn(".ctx-slots--right[hidden] { display: none; }", css)   # a full column, hidden, leaves no gap

    def test_the_left_column_takes_the_odd_slot(self):
        js, _, _ = self._files()
        # ceil(n/2) to the left: 1 -> 1|0, 2 -> 1|1, 3 -> 2|1, 4 -> 2|2, the pair even around the middle
        self.assertIn("const mid = Math.ceil(list.length / 2);", js)
        self.assertIn("fillCtxColumn(left, list.slice(0, mid), ctx);", js)
        self.assertIn("fillCtxColumn(right, list.slice(mid), ctx);", js)
        # a right column with nothing stays hidden; the left still blanks when the engine declares no slot at all
        self.assertIn("left.hidden = !list.length;", js)
        self.assertIn("right.hidden = !list.length;", js)

    def test_no_slot_counts_still_draws_the_big_clock_alone(self):
        js, _, _ = self._files()
        # the degrade is untouched: no live.slots -> both columns hidden, the big clock an only child, centred
        self.assertIn("if (!list.length) { left.textContent = \"\"; right.textContent = \"\"; return; }", js)
        self.assertNotIn("wrap.hidden", js)                                 # the old single-container name is gone


class TheSlotMeterAndTheAggregate(unittest.TestCase):
    """The slot texts against the aggregate (the Monitor report of 2026-10-15): they read live.slots[].tok_s - the
    engine's PER-SLOT rates - while the aggregate (the state card's own "N tokens at X tok/s", and `live.tok_s`) is
    the sum of the requests in flight, so the two can disagree.  A request that is ALONE decodes on the engine's
    solo path (generate_batched: `alone` -> GEN), which uses no slot's sessions, and then every configured slot is
    idle and reads 0 t/s under an aggregate that moves.  Reproduced here as the /metrics the two sources produce in
    that case (the same state the fake batch engine of test_parallel reaches with --batch 4 and one request: state
    generating, running 1, in_slots 0, outside_slots 1, every slot idle) - and NOT flaky: the slots list and the
    live rate are handed in, not raced for.

    live.solo_tok_s names that rate, and only then: with a slot decoding the slots carry it, and an engine with no
    batch slots (or an older shim) sends no field at all - the page adds nothing rather than "undefined"."""

    def _service(self, slots, live_requests=()):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=4096)
        engine.batch, engine.info = 4, {"batch_slots": 4}
        engine.waiting = engine.admitting = 0
        engine.slots_view = lambda: list(slots)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        now = time.time()
        for st, rate in live_requests:
            with svc.status_lock:
                svc.live_reqs[id(st)] = (st, rate)              # what Service.run registers while a request runs
        return svc, engine

    @staticmethod
    def _decoding(generated=40):
        """One request in flight and decoding: what the shim's own rate is computed from (its first token arrived)."""
        import collections
        now = time.time()
        st = {"busy": True, "first_token": now - 1.0, "started": now - 1.2, "generated": generated,
              "prompt_tokens": 62, "max_tokens": 256, "phase": None, "queued": 0}
        return st, collections.deque([(now - 1.0, generated - 30), (now, generated)])   # Service.run's rate samples

    def _metrics(self, svc):
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                return json.loads(r.read())["live"]
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_lone_request_moves_the_aggregate_with_every_slot_idle(self):
        idle = [{"slot": b, "state": "idle", "held_tokens": 0} for b in range(4)]
        svc, _ = self._service(idle, [self._decoding()])
        live = self._metrics(svc)
        self.assertEqual([s["state"] for s in live["slots"]], ["idle"] * 4)   # the capture the owner sent
        self.assertGreater(live["tok_s"], 0)                                 # ...under a moving aggregate
        self.assertEqual(live["outside_slots"], 1)
        self.assertEqual(live["solo_tok_s"], live["tok_s"])                   # named: this IS what the aggregate is
        self.assertEqual(live["state"], "generating")

    def test_a_slot_that_is_decoding_carries_the_rate_and_nothing_is_named_solo(self):
        slots = [{"slot": 0, "state": "decoding", "prompt_tokens": 512, "generated": 128, "tok_s": 15.2},
                 {"slot": 1, "state": "idle", "held_tokens": 0},
                 {"slot": 2, "state": "reading", "prompt_tokens": 1024, "generated": 0},
                 {"slot": 3, "state": "idle", "held_tokens": 4096}]
        svc, _ = self._service(slots, [self._decoding()])
        live = self._metrics(svc)
        self.assertEqual(live["slots"][0]["tok_s"], 15.2)                     # the slot's own rate: the page shows it
        self.assertNotIn("solo_tok_s", live)                                  # in a slot: no separate entry to add
        # a slot that HOLDS the request is enough, whatever it is doing with it (here it is being read into it):
        # the request is that slot's, so the aggregate and the slots agree and nothing is named solo
        svc2, _ = self._service([{"slot": 0, "state": "reading", "prompt_tokens": 1024, "generated": 0},
                                 {"slot": 1, "state": "idle", "held_tokens": 0}], [self._decoding()])
        live2 = self._metrics(svc2)
        self.assertEqual(live2["outside_slots"], 0)                           # the request in flight IS that slot's
        self.assertNotIn("solo_tok_s", live2)

    def test_an_engine_without_batch_slots_sends_no_slot_field(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        live = self._metrics(svc)
        self.assertNotIn("slots", live)
        self.assertNotIn("solo_tok_s", live)              # no slots, nothing to contradict: no field either

    def test_the_page_names_the_solo_rate_and_hides_it_when_the_field_is_missing(self):
        js = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        # read only when it IS a number, the rule live.admission follows: an older shim sends none
        self.assertIn('typeof live.solo_tok_s === "number"', js)
        self.assertIn("parts.push(`solo ${fmt(live.solo_tok_s, 1)} t/s`)", js)
        # one element per entry as before, and the idle slots keep their own "0 t/s" text
        self.assertIn('if (s.state === "idle") return `${slot} 0 t/s`', js)
        self.assertIn("slotsLine.replaceChildren(...parts.map", js)


class TheRamTheSlotsHold(unittest.TestCase):
    """The Conversation cache card's RAM row (the same report): it counted ONE of the two things that live in that
    RAM.  The budget (--conversation-cache-mib) is the RAM the slots' conversations live in; the row's own figure is
    what is PARKED in it, which the cascade writes to the warm level instead - so with the parking on disk it read
    0 / 0,0 GB while the slots held their K/V.  The row now carries the slots' side: their held tokens (measured,
    live.slots) and their bytes as an ESTIMATE, marked as one (kv_estimated) with its source (kv_source), scaled
    from the engine's own pinned K/V ("KV streaming: ... the K/V in X.XX GiB of pinned RAM", over the context it
    pins).  An engine that says the whole K/V is in VRAM (kv_resident=0) gets a measured 0 instead; one that said
    nothing leaves both unknown and the row is byte for byte the one it always was."""

    # what `strata generate` writes once per start, on stderr
    KV = ("strata generate: KV streaming: 81920 of 262144 cells per QSA layer in VRAM, the K/V in 3.20 GiB of "
          "pinned RAM\n")
    PARK = ("strata serve: conversation cache: parked on disk 5000 tokens in 12.0 ms; parked=0 bytes=0 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")

    def _parked(self, text=KV + PARK):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(text, encoding="utf-8")
            return ConvCacheLog().poll(str(log), 0)

    def _ram(self, info, slots, parked=None):
        from serve.server import conversation_cache_view
        view = conversation_cache_view(info, [], {}, parked if parked is not None else self._parked(), slots)
        return next(x for x in view["destinations"] if x["tier"] == "ram")

    def test_the_engine_pinned_kv_is_read_from_its_own_line(self):
        from serve.server import ConvCacheLog
        kv = self._parked()["kv_host"]
        self.assertEqual((kv["resident_cells"], kv["cells"], kv["gib"]), (81920, 262144, 3.20))
        self.assertEqual(kv["bytes"], int(3.20 * 1073741824))
        # the periodic block-read line is a different line and must not be read as this one
        other = "strata serve: KV streaming: 91.24% of 4000000 block reads hit VRAM, 12.5 MiB read from RAM\n"
        self.assertIsNone(self._parked(other)["kv_host"])
        # an engine run without --kv-resident prints neither: unknown, never invented
        self.assertIsNone(self._parked(self.PARK)["kv_host"])

    def test_the_ram_row_carries_the_slots_side_of_the_same_ram(self):
        info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4, "kv_resident": 81920,
                "context": 262144, "conversation_cache_disk_mib": 8192}
        slots = [{"slot": 0, "state": "idle", "held_tokens": 40000},
                 {"slot": 1, "state": "idle", "held_tokens": 20000},
                 {"slot": 2, "state": "decoding", "prompt_tokens": 5000, "generated": 5000},
                 {"slot": 3, "state": "idle", "held_tokens": 0}]
        ram = self._ram(info, slots)
        # measured: the tokens the slots hold (an idle slot keeps its conversation, a busy one runs one) and how
        # many slots hold one at all
        self.assertEqual(ram["held_tokens"], 70000)
        self.assertEqual((ram["slots_in_use"], ram["batch_slots"]), (3, 4))
        # and the parked side is untouched: the engine's log says nothing is parked in RAM (the cascade sent it on)
        self.assertEqual((ram["conversations"], ram["bytes"]), (0, 0))
        # the bytes: an estimate, and marked as one, with the source named - 70000 x (3.20 GiB / 262144)
        self.assertLess(abs(ram["kv_bytes"] - int(round(70000 * 3.20 * 1073741824 / 262144))), 2)
        self.assertTrue(ram["kv_estimated"])
        self.assertIn("70000 tokens", ram["kv_source"])
        self.assertIn("pinned K/V", ram["kv_source"])
        # the view still carries the engine's own figure, so /metrics shows what the estimate was scaled from
        from serve.server import conversation_cache_view
        view = conversation_cache_view(info, [], {}, self._parked(), slots)
        self.assertEqual(view["kv_host"]["bytes"], int(3.20 * 1073741824))

    def test_an_engine_that_keeps_the_whole_kv_in_vram_gets_a_zero_that_was_measured(self):
        info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4, "kv_resident": 0,
                "context": 262144}
        slots = [{"slot": 0, "state": "idle", "held_tokens": 40000}]
        ram = self._ram(info, slots, parked={})
        self.assertEqual(ram["kv_bytes"], 0)               # kv_resident=0 is the engine's own "all in VRAM"
        self.assertFalse(ram["kv_estimated"])              # a statement, not an estimate: the row adds no "est."
        self.assertIn("VRAM", ram["kv_source"])
        self.assertEqual(ram["held_tokens"], 40000)        # the measured part is still there

    def test_a_engine_that_said_nothing_leaves_them_unknown(self):
        ram = self._ram({"conversation_cache_mib": 8192, "conversation_cache_slots": 4}, [{"slot": 0, "state": "idle",
                                                                                          "held_tokens": 40000}],
                        parked={})
        self.assertIsNone(ram["kv_bytes"])                 # unknown, never a fabricated 0
        self.assertIsNone(ram["kv_source"])
        self.assertFalse(ram["kv_estimated"])

    def test_the_page_shows_the_slots_side_and_says_it_is_an_estimate(self):
        js = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('typeof d.kv_bytes === "number" && d.kv_estimated === true', js)
        self.assertIn("${gb(d.bytes)} parked + ${gb(d.kv_bytes)} est. / ${gb(budget)} GB", js)
        self.assertIn('const title = d.kv_source ? ` title="${esc(d.kv_source)}"` : "";', js)
        self.assertIn("pct(d.bytes + (est ? d.kv_bytes : 0), budget)", js)
        # without the estimate the row is byte for byte the one it always was (the degrade, as for admission)
        self.assertIn('budget ? `${gb(d.bytes)} / ${gb(budget)} GB` : `${gb(d.bytes)} GB`', js)


if __name__ == "__main__":
    unittest.main()
