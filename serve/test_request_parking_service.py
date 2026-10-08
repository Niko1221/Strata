"""Owned STOP/journal/unload/resume checks without a model or GPU."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from serve.request_parking import RequestParking
from serve.server import (Service, RequestParkControl, RequestParkCancelled, EngineDied,
                          MockEngine, StrataEngine, RequestParkCapacity)
from serve.test_live_memory import engine as protocol_engine
from serve.test_live_memory import service as protocol_service
from serve.test_reasoning_loop_recovery import Tokenizer


class ScriptedEngine(MockEngine):
    def __init__(self, tok, segments):
        super().__init__(tok, [""], max_context=100000)
        self.segments = list(segments)
        self.calls = []
        self.proc = SimpleNamespace(poll=lambda: 0, kill=lambda: None)
        self.can_stop = True
        self.info = {}
        self.down = False
        self.unloads = self.reloads = 0
        self.service = None

    def alive(self):
        return not self.down

    def generate(self, ids, max_new, sampling, cancel, park=None, **kwargs):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        tokens, suspend = self.segments.pop(0)
        for token in tokens:
            if cancel.is_set():
                return
            yield token
        if suspend == "error":
            raise EngineDied("fatal CUDA failure")
        self.last = {"generated": len(tokens), "prompt": len(ids), "finish": "cancel" if suspend else "length"}
        if suspend and park is not None:
            park.requested.set()
            park.stop_sent = park.acknowledged = True

    def unload(self):
        assert self.service.fifo.locked(), "unload escaped request ownership"
        self.down = True
        self.unloads += 1

    def restart(self, tries=1, startup_cancel=None):
        assert self.service.fifo.locked(), "reload escaped request ownership"
        self.reloads += 1
        self.proc = SimpleNamespace(poll=lambda: 0, kill=lambda: None)
        self.down = False


class ParkingProtocolTests(unittest.TestCase):
    def make_engine(self, lines):
        engine = protocol_engine()
        engine.can_stop = True
        engine.silence_s = 1
        for line in lines:
            engine.lines.put(line)
        return engine

    def test_stop_drains_every_accepted_token_before_acknowledging(self):
        engine = self.make_engine(["T 10\n", "T 11\n", "BACKGROUND status=waiting lease_remaining_ms=100\n",
                                   "T 12\n", "DONE 3 2 1 2 cancel\n"])
        park = RequestParkControl(engine.proc)
        gen = engine.generate([1, 2], 10, {}, threading.Event(), park=park)
        self.assertEqual(next(gen), 10)
        park.requested.set()
        self.assertEqual(list(gen), [11, None, 12])
        self.assertTrue(park.acknowledged)
        self.assertEqual(engine.proc.stdin.getvalue().count("STOP"), 1)

    def test_mismatched_done_cannot_authorize_reload(self):
        engine = self.make_engine(["T 10\n", "DONE 2 1 0 1 cancel\n"])
        park = RequestParkControl(engine.proc)
        park.requested.set()
        with self.assertRaisesRegex(EngineDied, "exact streamed"):
            list(engine.generate([1], 10, {}, threading.Event(), park=park))
        self.assertFalse(park.acknowledged)

    def test_completed_normally_is_not_a_suspension(self):
        engine = self.make_engine(["T 10\n", "DONE 1 1 0 1 length\n"])
        park = RequestParkControl(engine.proc)
        park.requested.set()
        self.assertEqual(list(engine.generate([1], 10, {}, threading.Event(), park=park)), [10])
        self.assertFalse(park.acknowledged)

    def test_error_or_eof_cannot_authorize_reload(self):
        for tail, exception in [("ERR CUDA failed\n", ValueError), (None, EngineDied)]:
            with self.subTest(tail=tail):
                engine = self.make_engine(["T 10\n", tail])
                park = RequestParkControl(engine.proc)
                park.requested.set()
                with self.assertRaises(exception):
                    list(engine.generate([1], 10, {}, threading.Event(), park=park))
                self.assertFalse(park.acknowledged)

    def test_wrong_process_control_is_rejected(self):
        engine = self.make_engine([])
        with self.assertRaises(ValueError):
            list(engine.generate([1], 10, {}, threading.Event(), park=RequestParkControl(object())))

    def test_withdrawn_startup_does_not_spawn(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch("serve.server.popen") as start:
            with self.assertRaises(RequestParkCancelled):
                StrataEngine("unused", [], startup_cancel=cancel)
        start.assert_not_called()

    def test_withdrawn_ready_wait_ends_its_process(self):
        import io
        cancel = threading.Event()
        proc = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO())
        proc.poll.return_value = None
        proc.kill.side_effect = lambda: setattr(proc.poll, "return_value", 1)
        def pump(process, out):
            cancel.set()
            out.put(None)
        with mock.patch("serve.server.popen", return_value=proc), mock.patch("serve.server.contain"), \
                mock.patch.object(StrataEngine, "_ready_pump", side_effect=pump):
            with self.assertRaises(RequestParkCancelled):
                StrataEngine("unused", [], startup_cancel=cancel)
        proc.kill.assert_called_once()
        proc.wait.assert_called()

    def test_pressure_dwell_needs_fresh_hard_floor_and_same_process(self):
        svc = protocol_service()
        svc.engine.info["background_control"] = 1
        svc.request_parking = RequestParking({"enabled": True, "directory": str(Path.cwd()), "pressure_seconds": 2})
        svc.coadaptive = mock.Mock(active=True)
        svc.coadaptive.fairness_decision.return_value = {"delay_ms": 0, "reason": "quiet"}
        svc.memory_policy.safety_decision = mock.Mock(return_value={"action": "wait", "reason": "vram_floor"})
        control = RequestParkControl(svc.engine.proc)
        svc.parking_active = {"control": control, "footprint": {}, "cancel": threading.Event()}
        svc._observe_background({"sampled_at": 100}, 100)
        svc._observe_background({"sampled_at": 100}, 102)  # duplicate cannot earn dwell
        self.assertFalse(control.requested.is_set())
        for stamp in (103, 104, 105):
            svc._observe_background({"sampled_at": stamp}, stamp)
        self.assertTrue(control.requested.is_set())
        svc.memory_policy.safety_decision.return_value = {"action": "run", "reason": "quiet"}
        svc._observe_background({"sampled_at": 106}, 106)
        self.assertFalse(control.requested.is_set())

    def test_reload_footprint_restores_shrunken_ram_and_gpu_caps(self):
        svc = protocol_service()
        svc.engine.proc.pid = 123
        svc.engine.native_capacity = {"proc": svc.engine.proc, "sampled_at": time.time(),
            "resident_mib": 1024, "cache_mib": 100, "usage_mib": 6000, "total_mib": 8192, "free_mib": 2192}
        svc.parking_start_cache_mib = 500
        svc.request_parking = RequestParking()
        info = SimpleNamespace(rss=3 * 2**30, private=5 * 2**30, vms=9 * 2**30)
        with mock.patch("psutil.Process", return_value=SimpleNamespace(memory_info=lambda: info)):
            result = svc._parking_footprint()
        self.assertEqual(result, {"ram_bytes": (3 + 41 + 2) * 2**30,
                                  "commit_bytes": (5 + 41 + 2) * 2**30, "gpu_bytes": 6400 * 2**20})
        svc.engine.native_capacity["sampled_at"] = 1
        with self.assertRaisesRegex(ValueError, "fresh native"):
            svc._parking_footprint()


class ParkingServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-parking-service-")
        self.addCleanup(self.temp.cleanup)
        self.tok = Tokenizer()

    def service(self, segments, *, ready=True):
        engine = ScriptedEngine(self.tok, segments)
        svc = Service(engine, self.tok, None)
        engine.service = svc
        svc.request_parking = RequestParking({"enabled": True, "directory": self.temp.name})
        svc._parking_identity = mock.Mock(return_value={"runtime": "test", "context": 100000})
        svc._parking_footprint = mock.Mock(return_value={"ram_bytes": 100, "commit_bytes": 100, "gpu_bytes": 100})
        svc.ensure_loaded = mock.Mock()
        svc._invalidate_live_memory = mock.Mock()
        svc._record_live_load = mock.Mock()
        svc.telemetry = SimpleNamespace(capacity=mock.Mock(return_value={"sampled_at": time.time()}))
        svc.request_parking.admission = mock.Mock(return_value=SimpleNamespace(
            observe=mock.Mock(return_value={"ready": ready, "reason": "test"})))
        return svc

    def test_utf8_split_preserves_exact_prefix_seed_budget_and_one_done(self):
        raw = list("Grüße 世界 ✓ done".encode())
        svc = self.service([(raw[:3], True), (raw[3:], False)])
        original = {"temperature": .8}
        events = list(svc.run([1, 2], False, [], 100, original, threading.Event()))
        text = "".join(event.text or "" for kind, event in events if kind == "event" and event.kind == "content")
        self.assertEqual(text, "Grüße 世界 ✓ done")
        first, second = svc.engine.calls
        self.assertEqual(second[0], first[0] + raw[:3])
        self.assertEqual(second[1], 100 - 3)
        self.assertEqual(second[2]["seed"], first[2]["seed"])
        self.assertNotIn("seed", original)
        self.assertEqual(sum(kind == "done" for kind, _ in events), 1)
        self.assertEqual((svc.engine.unloads, svc.engine.reloads), (1, 1))
        self.assertFalse(svc.fifo.locked())
        self.assertIsNone(svc.parking_active)
        self.assertFalse(list(Path(self.temp.name).iterdir()))

    def test_tool_arguments_split_keeps_one_call_identity(self):
        text = '<tool_call>\n<function=terminal><parameter=command>echo ok</parameter></function>\n</tool_call>'
        raw = list(text.encode())
        split = text.index("echo") + 2
        svc = self.service([(raw[:split], True), (raw[split:], False)])
        events = list(svc.run([1], False, [{"name": "terminal", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}}}],
                              1000, {}, threading.Event()))
        starts = [ev for kind, ev in events if kind == "event" and ev.kind == "tool_start"]
        calls = [ev for kind, ev in events if kind == "event" and ev.kind == "tool_call"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(starts[0].call.id, calls[0].call.id)

    def test_cancel_while_suspended_releases_fifo_without_reload(self):
        svc = self.service([(list(b"abc"), True)], ready=False)
        cancel = threading.Event()
        gen = svc.run([1], False, [], 100, {}, cancel)
        while next(gen)[0] != "ping":
            pass
        self.assertTrue(svc.engine.down)
        self.assertTrue(svc.fifo.locked())
        self.assertTrue(svc.status["busy"])
        cancel.set()
        tail = list(gen)
        self.assertEqual(tail[-1][1]["finish"], "cancel")
        self.assertEqual(svc.engine.reloads, 0)
        self.assertFalse(svc.fifo.locked())
        self.assertFalse(list(Path(self.temp.name).iterdir()))

    def test_disconnect_while_suspended_removes_journal_and_releases_fifo(self):
        svc = self.service([(list(b"abc"), True)], ready=False)
        gen = svc.run([1], False, [], 100, {}, threading.Event())
        while next(gen)[0] != "ping":
            pass
        gen.close()
        self.assertFalse(svc.fifo.locked())
        self.assertIsNone(svc.parking_active)
        self.assertEqual(svc.engine.reloads, 0)
        self.assertFalse(list(Path(self.temp.name).iterdir()))

    def test_native_failure_does_not_unload_reload_or_replay(self):
        svc = self.service([(list(b"abc"), "error")])
        with self.assertRaises(EngineDied):
            list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual((svc.engine.unloads, svc.engine.reloads), (0, 0))
        self.assertEqual(len(svc.engine.calls), 1)
        self.assertFalse(svc.fifo.locked())

    def test_journal_write_failure_never_unloads_a_valid_engine(self):
        svc = self.service([(list(b"abc"), True)])
        journal = mock.Mock()
        journal.write.side_effect = OSError("disk full")
        svc.request_parking.create_journal = mock.Mock(return_value=journal)
        with self.assertRaisesRegex(OSError, "disk full"):
            list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(svc.engine.unloads, 0)
        journal.remove.assert_called_once()

    def test_reloading_failure_never_enters_second_gen(self):
        svc = self.service([(list(b"abc"), True)])
        svc.engine.restart = mock.Mock(side_effect=RuntimeError("allocation failed"))
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(len(svc.engine.calls), 1)
        self.assertTrue(svc.engine.down)
        self.assertFalse(svc.memory_loading)
        self.assertFalse(svc.fifo.locked())

    def test_reloaded_identity_change_fails_without_replay(self):
        svc = self.service([(list(b"abc"), True)])
        svc._parking_identity.side_effect = [{"runtime": "test", "context": 100000},
                                              {"runtime": "changed", "context": 100000}]
        with self.assertRaisesRegex(RuntimeError, "identity"):
            list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(svc.engine.reloads, 0)
        self.assertEqual(len(svc.engine.calls), 1)

    def test_identity_change_during_load_releases_the_new_engine(self):
        svc = self.service([(list(b"abc"), True)])
        original = {"runtime": "test", "context": 100000}
        svc._parking_identity.side_effect = [original, original, {"runtime": "changed", "context": 100000}]
        with self.assertRaisesRegex(ValueError, "identity"):
            list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(svc.engine.reloads, 1)
        self.assertEqual(svc.engine.unloads, 2)
        self.assertTrue(svc.engine.down)
        self.assertEqual(len(svc.engine.calls), 1)

    def test_admission_keeps_larger_pre_stop_measurements(self):
        svc = self.service([(list(b"abc"), True), (list(b"def"), False)])
        svc._parking_footprint.side_effect = [
            {"ram_bytes": 200, "commit_bytes": 100, "gpu_bytes": 100},
            {"ram_bytes": 100, "commit_bytes": 300, "gpu_bytes": 500}]
        list(svc.run([1], False, [], 100, {}, threading.Event()))
        svc.request_parking.admission.assert_called_with({"ram_bytes": 200, "commit_bytes": 300, "gpu_bytes": 500})

    def test_lost_pre_stop_telemetry_keeps_running_engine_and_exact_prefix(self):
        svc = self.service([(list(b"abc"), True), (list(b"def"), False)])
        svc._parking_footprint.side_effect = [{"ram_bytes": 100, "commit_bytes": 100, "gpu_bytes": 100},
                                             ValueError("fresh native footprint is unavailable")]
        events = list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual((svc.engine.unloads, svc.engine.reloads), (0, 0))
        self.assertEqual(svc.engine.calls[1][0], [1] + list(b"abc"))
        self.assertEqual(events[-1][1]["completion_tokens"], 6)

    def test_cancel_during_reload_finishes_owned_cleanup_before_fifo_release(self):
        svc = self.service([(list(b"abc"), True)])
        cancel = threading.Event()
        def reload(tries=1, startup_cancel=None):
            cancel.set()
            startup_cancel.wait(2)
            raise RequestParkCancelled()
        svc.engine.restart = reload
        events = list(svc.run([1], False, [], 100, {}, cancel))
        self.assertEqual(events[-1][1]["finish"], "cancel")
        self.assertTrue(svc.engine.down)
        self.assertFalse(svc.fifo.locked())
        self.assertEqual(len(svc.engine.calls), 1)
        self.assertFalse(any(t.name == "strata-request-lifecycle" for t in threading.enumerate()))

    def test_first_capacity_can_arrive_after_ready(self):
        svc = self.service([(list(b"ok"), False)])
        svc._parking_footprint.side_effect = [ValueError("fresh native footprint is unavailable"),
                                            {"ram_bytes": 100, "commit_bytes": 100, "gpu_bytes": 100}]
        events = list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(events[0], ("ping", None))
        self.assertEqual(svc._parking_footprint.call_count, 2)
        self.assertIn("seed", svc.engine.calls[0][2])

    def test_cancel_before_first_capacity_never_enters_gen(self):
        svc = self.service([])
        svc._parking_footprint.side_effect = ValueError("fresh native footprint is unavailable")
        cancel = threading.Event()
        gen = svc.run([1], False, [], 100, {}, cancel)
        self.assertEqual(next(gen), ("ping", None))
        cancel.set()
        self.assertEqual(list(gen)[-1][1]["finish"], "cancel")
        self.assertEqual(svc.engine.calls, [])
        self.assertFalse(svc.fifo.locked())

    def test_only_pre_ready_capacity_failure_gets_safe_bounded_retry(self):
        svc = self.service([(list(b"abc"), True), (list(b"def"), False)])
        original = svc.engine.restart
        attempts = []
        def restart(tries=1, startup_cancel=None):
            attempts.append(1)
            if len(attempts) == 1:
                svc.engine.max_context = 0
                raise RuntimeError("CUDA out of memory before READY")
            svc.engine.max_context = 100000
            original(tries, startup_cancel)
        svc.engine.restart = restart
        admission = svc.request_parking.admission.return_value
        admission.failed = mock.Mock()
        events = list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(len(attempts), 2)
        admission.failed.assert_called_once()
        self.assertEqual(len(svc.engine.calls), 2)  # first call was not replayed
        self.assertEqual(svc.engine.calls[1][0], [1] + list(b"abc"))
        self.assertEqual(events[-1][1]["completion_tokens"], 6)

    def test_real_identity_survives_zero_context_during_failed_startup(self):
        svc = self.service([(list(b"abc"), True), (list(b"def"), False)])
        del svc._parking_identity
        svc.engine.known_ctx = 100000
        original = svc.engine.restart
        attempts = []
        def restart(tries=1, startup_cancel=None):
            attempts.append(1)
            if len(attempts) == 1:
                svc.engine.max_context = 0
                raise RuntimeError("CUDA out of memory before READY")
            svc.engine.max_context = 100000
            original(tries, startup_cancel)
        svc.engine.restart = restart
        svc.request_parking.admission.return_value.failed = mock.Mock()
        with mock.patch("serve.server.runtime_key", return_value="test"):
            events = list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(events[-1][1]["completion_tokens"], 6)

    def test_repeated_pre_ready_capacity_failure_is_bounded(self):
        svc = self.service([(list(b"abc"), True)])
        svc._parking_reload = mock.Mock(side_effect=RequestParkCapacity("out of memory"))
        svc.request_parking.admission.return_value.failed = mock.Mock()
        with self.assertRaises(RequestParkCapacity):
            list(svc.run([1], False, [], 100, {}, threading.Event()))
        self.assertEqual(svc._parking_reload.call_count, 3)
        self.assertEqual(len(svc.engine.calls), 1)

    def test_lifecycle_disconnect_waits_for_worker_cleanup(self):
        svc = self.service([])
        finished = threading.Event()
        def operation(stopped):
            stopped.wait(5)
            finished.set()
        gen = svc._parking_lifecycle(operation, threading.Event())
        self.assertEqual(next(gen), ("ping", None))
        gen.close()
        self.assertTrue(finished.is_set())
        self.assertFalse(any(t.name == "strata-request-lifecycle" for t in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
