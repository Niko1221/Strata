"""CPU-only native-pipe/HTTP consumer fixtures; never construct a live engine."""
import io
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import server

if os.environ.get("P01_E2E_SERVER_UNDER_TEST"):
    import importlib.util
    original_root = server.ROOT
    server_spec = importlib.util.spec_from_file_location("b1_server_mutant", os.environ["P01_E2E_SERVER_UNDER_TEST"])
    server_mutant = importlib.util.module_from_spec(server_spec)
    server_spec.loader.exec_module(server_mutant)
    server_mutant.ROOT = original_root  # copied module has the same product data root
    server = server_mutant

if os.environ.get("P01_E2E_UNDER_TEST"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("e2e_mutant", os.environ["P01_E2E_UNDER_TEST"])
    mutant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutant)
    server.perf_e2e = mutant


def native(lines, module=server):
    engine = object.__new__(module.StrataEngine)
    engine.alive = lambda: True
    engine.exit_code = lambda: 0
    engine.sampling_keys = lambda sampling: ""
    engine.batch, engine.silence_s = 0, 0
    engine.max_context = 131072
    engine.proc = SimpleNamespace(stdin=io.StringIO(), pid=4242)
    engine.lines = queue.Queue()
    for line in lines:
        engine.lines.put(line)
    engine.last = {}
    engine.info = {}
    engine.can_stop = True
    return engine


def unit_observer(name, ids=None):
    ids = [1] * 4096 if ids is None else ids
    audit = server.perf_e2e
    temp = Path(tempfile.mkdtemp(dir=os.environ["TEMP"]))
    settings = audit.Settings(temp, name, audit.token_hash(ids), "a" * 64, prompt_tokens=len(ids))
    obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
    obs.prepared(ids, 256, 131072)
    return settings, obs, ids


def finish_unit(settings, obs, count):
    obs.service_terminal("stop")
    obs.http_event({"choices": [{"finish_reason": "stop"}],
                    "usage": {"prompt_tokens": settings.prompt_tokens, "completion_tokens": count}})
    obs.http_end()
    server.perf_e2e.end(obs)
    return json.loads((settings.directory / "request-000001.json").read_text())


class E2EConsumerTests(unittest.TestCase):
    def exercise_http(self, *, finish="stop", reserve=256, capacity=131072, error=False, count=1,
                      max_samples=256, mismatch=False, actual_service=False, late_cancel=False, synthetic=False,
                      stream=True, write_error=False):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "run", audit.token_hash(ids), "a" * 64,
                                  max_samples=max_samples)
        lines = ["T 7"] * count + [f"DONE {count + int(mismatch)} 4096 1 1 {finish}"]
        engine = native(lines)
        engine.max_context = capacity
        svc = server.Service(engine, server.ByteTokenizer(), SimpleNamespace())
        svc.perf131k_e2e = settings
        svc.load = lambda: None
        svc.ensure_loaded = lambda: None
        svc.encode_prompt = lambda *args: ids
        def run(*args, **kwargs):
            emitted = list(engine.generate(ids, args[3], {}, threading.Event()))
            if error:
                raise server.EngineDied("fixture error; private text must not be recorded")
            server.perf_e2e.current().service_terminal(finish)
            yield "event", server.Event("content", "PUBLIC-FIXTURE")
            yield "done", {"finish": finish, "completion_tokens": len(emitted), "reasoning_tokens": 0}
        svc.run = run
        if actual_service:
            svc.run = svc._run  # the real Service consumer; only native pipes are fixtures
            if synthetic:
                svc.reasoning_close_retry = True
                svc.stop_ids = {7}  # fixture EOS is counted in usage, excluded from continuation prompt
                continuation_n = len(ids) + len(svc.tok.encode(server.REASONING_CLOSE))
                engine.lines.put("T 8")
                engine.lines.put(f"DONE 1 {continuation_n} 1 1 stop")
            original_generate = engine.generate
            def actual_native(*args, **kwargs):
                yield from original_generate(*args, **kwargs)
                if late_cancel:
                    args[3].set()  # native DONE stop has already been consumed
            engine.generate = actual_native
        handler = object.__new__(server.make_handler(svc))
        handler.path = "/v1/chat/completions"
        handler.headers = {}
        handler._authorized = lambda: True
        handler._foreign_page = lambda: False
        handler._no_local_images = lambda messages: None
        handler._watch_client = lambda cancel: None
        handler._cors = lambda: None
        handler.send_response = lambda code: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler._body = lambda: json.dumps({"messages": [], "stream": stream, "max_tokens": reserve})
        class BrokenWrites(io.BytesIO):
            def write(self, data):
                raise OSError("fixture write failure")
        handler.wfile = BrokenWrites() if write_error else io.BytesIO()
        with patch("subprocess.Popen", side_effect=AssertionError("real process launch forbidden")), \
             patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            if write_error:
                with self.assertRaises(OSError):
                    handler.do_POST()
            else:
                handler.do_POST()
        record = json.loads((settings.directory / "request-000001.json").read_text())
        self.assertIsNone(audit.current())
        return record, handler.wfile.getvalue(), settings

    def test_observer_is_connected_to_product_consumer(self):
        self.assertTrue(hasattr(server, "perf_e2e"), "P01-B1 observer is not connected")

    def test_native_terminal_is_required_even_if_http_done_arrives(self):
        self.assertTrue(hasattr(server, "perf_e2e"), "P01-B1 observer is not connected")
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        if temp:  # retain fixtures and evidence; no recursive cleanup
            settings = audit.Settings(Path(temp), "fixture", audit.token_hash(ids), "a" * 64)
            obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
            obs.prepared(ids, 256, 131072)
            engine = native(["T 7", None])
            with self.assertRaises(server.EngineDied):
                list(engine.generate(ids, 256, {}, threading.Event()))
            obs.http_event({"choices": [{"finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 4096, "completion_tokens": 1}})
            obs.http_end()
            audit.end(obs)
            record = json.loads((settings.directory / "request-000001.json").read_text())
            self.assertEqual(record["status"], "INCOMPLETE")
            self.assertIn("missing_native_terminal", record["reasons"])

    def test_real_prepare_captures_effective_input_and_reserve(self):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "prepare", audit.token_hash(ids), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        engine = native([])
        svc = server.Service(engine, server.ByteTokenizer(), SimpleNamespace())
        svc.encode_prompt = lambda *args: ids
        try:
            actual, thinking, reserve = svc.prepare([], None, {}, 256)
            self.assertEqual((len(actual), reserve), (4096, 256))
            self.assertTrue(obs.prepared_seen)
        finally:
            audit.end(obs)

    def test_real_http_consumer_rejects_error_followed_by_done(self):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "http", audit.token_hash(ids), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        engine = native(["T 7", "DONE 1 4096 1 1 stop"])
        svc = server.Service(engine, server.ByteTokenizer(), SimpleNamespace())
        svc.load = lambda: None  # never start any real engine
        svc.encode_prompt = lambda *args: ids
        def run(*args, **kwargs):
            list(engine.generate(ids, 256, {}, threading.Event()))
            raise server.EngineDied("fixture transport failure")
            yield  # generator body runs inside the real SSE handler
        svc.run = run
        handler = object.__new__(server.make_handler(svc))
        handler._no_local_images = lambda messages: None
        handler._watch_client = lambda cancel: None
        handler._sse = lambda: None
        handler.wfile = io.BytesIO()
        try:
            handler._openai({"messages": [], "stream": True, "max_tokens": 256})
            self.assertIn(b"[DONE]", handler.wfile.getvalue())
        finally:
            audit.end(obs)
        record = json.loads((settings.directory / "request-000001.json").read_text())
        self.assertIn("http_error_or_disconnect", record["reasons"])

    def test_complete_real_post_native_and_http_path(self):
        record, wire, settings = self.exercise_http()
        self.assertEqual(record["status"], "COMPLETE_OBSERVED_SERVER_PATH")
        self.assertEqual(record["native_t"], 1)
        self.assertIn(b"[DONE]", wire)
        self.assertIsNone(record["client_arrival_times"])
        self.assertIsNone(record["GPU_execution_time"])
        self.assertNotIn("PUBLIC-FIXTURE", json.dumps(record))
        self.assertEqual(json.loads((settings.directory / "summary.json").read_text())["completed"], 1)

    def test_clock_origin_and_observed_process_identity_are_explicit(self):
        record, _, _ = self.exercise_http()
        self.assertGreater(record["server_clock_start_ns"], 0)
        self.assertEqual(record["server_pid"], os.getpid())
        self.assertEqual(record["native_pid"], 4242)  # fixture attribute, no live PID lookup
        self.assertTrue(record["started_utc"].endswith("+00:00"))

    def test_nonstream_success_is_server_write_evidence(self):
        record, wire, _ = self.exercise_http(stream=False)
        self.assertEqual(record["status"], "COMPLETE_OBSERVED_SERVER_PATH")
        self.assertEqual(json.loads(wire)["usage"]["completion_tokens"], 1)

    def test_failed_nonstream_write_does_not_report_a_written_delta(self):
        record, _, _ = self.exercise_http(stream=False, write_error=True)
        self.assertEqual(record["status"], "INCOMPLETE")
        self.assertIsNone(record["first_http_delta_s"])

    def test_cancel_cannot_become_success_through_http_stop_mapping(self):
        record, wire, _ = self.exercise_http(finish="cancel")
        self.assertIn(b'"finish_reason": "stop"', wire)
        self.assertIn("native_unsuccessful_finish", record["reasons"])

    def test_real_service_late_cancel_after_native_stop_is_not_complete(self):
        record, wire, _ = self.exercise_http(actual_service=True, late_cancel=True)
        self.assertEqual(record["native_finish"], "stop")
        self.assertIn(b'"finish_reason": "stop"', wire)
        self.assertEqual(record["status"], "INCOMPLETE")
        self.assertIn("service_unsuccessful_finish", record["reasons"])

    def test_real_service_reasoning_close_insertion_and_continuation_are_counted(self):
        record, wire, _ = self.exercise_http(actual_service=True, synthetic=True)
        self.assertEqual(record["native_segments"], 2)
        self.assertGreater(record["synthetic_tokens"], 0)
        self.assertEqual(record["http_completion_tokens"],
                         record["native_delivered_t"] + record["synthetic_tokens"])
        self.assertIn(b"[DONE]", wire)

    def test_error_followed_by_done_is_incomplete_in_real_post(self):
        record, wire, _ = self.exercise_http(error=True)
        self.assertIn(b"[DONE]", wire)
        self.assertIn("http_error_or_disconnect", record["reasons"])

    def test_native_count_mismatch_is_detected(self):
        record, _, _ = self.exercise_http(mismatch=True)
        self.assertIn("native_count_mismatch", record["reasons"])

    def test_original_reserve_and_actual_capacity_are_observations(self):
        record, _, _ = self.exercise_http(reserve=128, capacity=32768)
        self.assertEqual(record["requested_reserve"], 128)
        self.assertEqual(record["capacity"], 32768)
        self.assertIn("capacity_mismatch", record["reasons"])

    def test_timestamp_overflow_is_explicit(self):
        record, _, _ = self.exercise_http(count=2, max_samples=1)
        self.assertEqual(len(record["token_times_s"]), 1)
        self.assertEqual(record["token_samples_dropped"], 1)
        self.assertIn("timestamp_overflow", record["reasons"])

    def test_fit_shortening_is_recorded_and_rejected_without_changing_behavior(self):
        record, _, _ = self.exercise_http(capacity=4224)
        self.assertEqual(record["requested_reserve"], 256)
        self.assertEqual(record["effective_reserve"], 120)
        self.assertIn("effective_reserve_mismatch", record["reasons"])

    def test_missing_optional_native_counters_are_unknown(self):
        record, _, _ = self.exercise_http()
        self.assertEqual(record["optional_native_counters"],
                         {"drafts_accepted": None, "drafts_offered": None, "reused": None})

    def test_request_output_count_is_bounded_and_loss_is_recorded(self):
        audit = server.perf_e2e
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "bounded", audit.token_hash([1] * 4096), "a" * 64,
                                  max_requests=2)
        for _ in range(3):
            audit.end(audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256}))
        self.assertEqual(len(list(settings.directory.glob("request-*.json"))), 2)
        summary = json.loads((settings.directory / "summary.json").read_text())
        self.assertEqual(summary["dropped_requests"], 1)

    def test_write_error_is_loss_not_generation_control_flow(self):
        audit = server.perf_e2e
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "io", audit.token_hash([1] * 4096), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        with patch.object(Path, "open", side_effect=OSError("fixture disk failure")):
            audit.end(obs)
        self.assertEqual(settings.completed, 0)
        self.assertGreater(settings.write_errors, 0)
        self.assertIsNone(audit.current())

    def test_drain_counts_native_tokens_separately_from_delivered_tokens(self):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "drain", audit.token_hash(ids), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        obs.prepared(ids, 256, 131072)
        engine = native(["T 7", "T 8", "DONE 2 4096 1 1 stop"])
        gen = engine.generate(ids, 256, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        gen.close()
        obs.http_event({"choices": [{"finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4096, "completion_tokens": 1}})
        obs.http_end()
        obs.service_terminal("stop")
        audit.end(obs)
        record = json.loads((settings.directory / "request-000001.json").read_text())
        self.assertEqual((record["native_t"], record["native_generated"], record["native_delivered_t"]), (2, 2, 1))
        self.assertEqual(record["status"], "COMPLETE_OBSERVED_SERVER_PATH")

    def test_default_off_protocol_and_prepare_match_frozen_product(self):
        import importlib.util
        baseline_path = Path("H:/Strata/evidence/p01-b1-20261026/freeze/server.py")
        baseline_spec = importlib.util.spec_from_file_location("frozen_server_b1", baseline_path)
        baseline = importlib.util.module_from_spec(baseline_spec)
        snapshots = []
        with patch("subprocess.Popen", side_effect=AssertionError("no process")), \
             patch("socket.socket.connect", side_effect=AssertionError("no network")), \
             patch.object(server.perf_e2e.time, "perf_counter_ns", side_effect=AssertionError("OFF clock")):
            baseline_spec.loader.exec_module(baseline)
            for module in (baseline, server):
                engine = native(["PP 4096 4096", "RESUME 0", "T 7", "T 8",
                                 "DONE 2 4096 4 6 length 1 2 0"], module)
                output = list(engine.generate([1] * 4096, 256, {}, threading.Event()))
                svc = module.Service(engine, module.ByteTokenizer(), SimpleNamespace())
                svc.encode_prompt = lambda *args: [1] * 4096
                engine.max_context = 4224
                prepared = svc.prepare([], None, {}, 256)
                snapshots.append((output, engine.proc.stdin.getvalue(), engine.last, prepared))
        self.assertEqual(snapshots[0], snapshots[1])

    def test_synthetic_and_multiple_native_segments_have_distinct_counts(self):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "segments", audit.token_hash(ids), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        obs.prepared(ids, 256, 131072)
        for prompt, reserve in ((ids, 256), (ids + [7, 8], 254)):
            engine = native(["T 7", f"DONE 1 {len(prompt)} 1 2 stop 0 1 0"])
            list(engine.generate(prompt, reserve, {}, threading.Event()))
        obs.synthetic_token()
        obs.http_event({"choices": [{"finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4096, "completion_tokens": 3}})
        obs.http_end()
        obs.service_terminal("stop")
        audit.end(obs)
        record = json.loads((settings.directory / "request-000001.json").read_text())
        self.assertEqual(record["status"], "COMPLETE_OBSERVED_SERVER_PATH")
        self.assertEqual((record["native_generated"], record["synthetic_tokens"],
                          record["http_completion_tokens"]), (2, 1, 3))
        self.assertEqual(record["native_host_prompt_ms_sum"], 2)
        self.assertEqual(record["optional_native_counters"]["drafts_offered"], 2)

    def test_nonfinite_terminal_and_missing_measurements_stay_unknown(self):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "nan", audit.token_hash(ids), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        obs.prepared(ids, 256, 131072)
        obs.native_begin(ids, 256)
        obs.native_token(7)
        obs.native_done("DONE 1 4096 nan 1 stop")
        audit.end(obs)
        record = json.loads((settings.directory / "request-000001.json").read_text())
        self.assertIn("invalid_native_terminal", record["reasons"])
        self.assertIsNone(record["native_generated"])
        self.assertIsNone(record["native_host_prompt_ms_sum"])

    def test_huge_and_unsafe_configuration_is_rejected_before_output(self):
        audit = server.perf_e2e
        temp = Path(tempfile.mkdtemp(dir=os.environ["TEMP"]))
        config = temp / "too-big.json"
        config.write_bytes(b" " * 16385)
        with self.assertRaises(ValueError):
            audit.load_settings(config)
        with self.assertRaises(ValueError):
            audit.Settings(Path("C:/"), "unsafe", "a" * 64, "a" * 64)
        with self.assertRaises(ValueError):
            audit.Settings(temp, "../escape", "a" * 64, "a" * 64)

    def test_thread_contexts_do_not_mix_requests(self):
        audit = server.perf_e2e
        temp = Path(tempfile.mkdtemp(dir=os.environ["TEMP"]))
        settings = audit.Settings(temp, "threads", audit.token_hash([1] * 4096), "a" * 64)
        barrier = threading.Barrier(2, timeout=2)
        failures = []
        def fixture():
            obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
            try:
                barrier.wait()
                if audit.current() is not obs:
                    failures.append("context mixed")
            finally:
                audit.end(obs)
        threads = [threading.Thread(target=fixture) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(settings.completed, 2)
        self.assertIsNone(audit.current())

    def test_ordered_segment_identity_binds_32k_continuations(self):
        chains = []
        for suffix in (2, 2, 3):
            settings, obs, ids = unit_observer("identity", [1] * 32768)
            for prompt, reserve in ((ids, 256), (ids + [suffix], 255)):
                engine = native(["T 7", f"DONE 1 {len(prompt)} 1 1 stop"])
                list(engine.generate(prompt, reserve, {}, threading.Event()))
            record = finish_unit(settings, obs, 2)
            self.assertEqual(record["status"], "COMPLETE_OBSERVED_SERVER_PATH")
            self.assertTrue(record.get("segment_chain_sha256"), "continuation identity is not bound")
            self.assertEqual(record["segment_samples"][1]["prompt_tokens"], 32769)
            chains.append(record["segment_chain_sha256"])
        self.assertEqual(chains[0], chains[1])
        self.assertNotEqual(chains[0], chains[2])

    def test_optional_counter_overflow_is_unknown_and_incomplete(self):
        settings, obs, ids = unit_observer("overflow")
        limit = (1 << 63) - 1
        for _ in range(2):
            engine = native(["T 7", f"DONE 1 4096 1 1 stop {limit} {limit} 0"])
            list(engine.generate(ids, 256, {}, threading.Event()))
        record = finish_unit(settings, obs, 2)
        self.assertIn("counter_overflow", record["reasons"])
        self.assertIsNone(record["optional_native_counters"]["drafts_offered"])

    def test_shared_watcher_notification_cannot_split_status_from_reasons(self):
        settings, obs, ids = unit_observer("race")
        engine = native(["T 7", "DONE 1 4096 1 1 stop"])
        list(engine.generate(ids, 256, {}, threading.Event()))
        obs.service_terminal("stop")
        obs.http_event({"choices": [{"finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4096, "completion_tokens": 1}})
        obs.http_end()
        entered, release, notifying, notified = (threading.Event() for _ in range(4))
        class PausingReasons(set):
            def __bool__(self):
                entered.set()
                release.wait(2)
                return False
        obs.reasons = PausingReasons(obs.reasons)
        finisher = threading.Thread(target=obs.finish)
        def watcher():
            notifying.set()
            obs.note(outcome="disconnected")
            notified.set()
        finisher.start()
        self.assertTrue(entered.wait(2))
        writer = threading.Thread(target=watcher)
        writer.start()
        self.assertTrue(notifying.wait(2))
        notified.wait(0.1)  # old implementation mutates mid-snapshot; fixed one waits for its cutoff
        release.set()
        finisher.join(2)
        writer.join(2)
        self.assertFalse(finisher.is_alive() or writer.is_alive())
        server.perf_e2e.end(obs)  # clear the creator thread's context; finish itself is idempotent
        record = json.loads((settings.directory / "request-000001.json").read_text())
        self.assertEqual(record["status"] == "COMPLETE_OBSERVED_SERVER_PATH", not bool(record["reasons"]))
        before = set(obs.reasons)
        obs.note(error={"message": "notification after cutoff"})
        self.assertEqual(set(obs.reasons), before)
        self.assertTrue(record.get("observation_cutoff"))

    def test_disabled_begin_has_no_clock_file_or_worker_side_effect(self):
        with patch.object(server.perf_e2e.time, "perf_counter_ns", side_effect=AssertionError("clock")), \
             patch.object(Path, "open", side_effect=AssertionError("file")), \
             patch("threading.Thread.start", side_effect=AssertionError("worker")):
            self.assertIsNone(server.perf_e2e.begin(None, "/v1/chat/completions", {}))
            self.assertIsNone(server.perf_e2e.load_settings(None))

    def test_invalid_native_input_does_not_interrupt_existing_consumer(self):
        audit = server.perf_e2e
        ids = [1] * 4096
        temp = tempfile.mkdtemp(dir=os.environ["TEMP"])
        settings = audit.Settings(Path(temp), "large", audit.token_hash(ids), "a" * 64)
        obs = audit.begin(settings, "/v1/chat/completions", {"max_tokens": 256})
        large = [1] * 32769
        try:
            obs.prepared(large, 256, 131072)
            engine = native(["T 7", "DONE 1 32769 1 1 stop"])
            self.assertEqual(list(engine.generate(large, 256, {}, threading.Event())), [7])
        finally:
            audit.end(obs)


if __name__ == "__main__":
    unittest.main()
