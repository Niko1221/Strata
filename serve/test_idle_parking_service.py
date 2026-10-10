"""Idle lifecycle admission and HTTP preparation races, without a model/GPU."""
import http.client
import json
import io
import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from serve.server import (Service, IdleParking, ByteTokenizer, EngineDied, EngineStuck,
                          RequestParkCancelled, RequestParkCapacity, GpuBusy, Vision, serve)
from serve.test_request_parking_service import ScriptedEngine
from serve.test_reasoning_loop_recovery import Tokenizer
from serve.test_live_memory import service as native_service

GIB, MIB = 2**30, 2**20


def snapshot(stamp, ram=2, commit=20, gpu=500):
    return {"sampled_at": stamp, "ram_total": 64 * GIB, "ram_used": (64 - ram) * GIB,
            "ram_commit_available": commit * GIB, "ram_commit_required": True,
            "native_free_mib": gpu, "gpu_mem_total": 8192 * MIB,
            "gpu_mem_used": (8192 - gpu) * MIB}


class IdleConfigTests(unittest.TestCase):
    def test_defaults_disabled_and_bad_values_rejected(self):
        self.assertFalse(IdleParking().enabled)
        for cfg in ({"enabled": "yes"}, {"directory": "/tmp"},
                    {"pressure_ram_available_gib": float("nan")}, {"unknown": 1},
                    {"pressure_vram_free_mib": 249}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                IdleParking(cfg)

    def test_pressure_requires_advancing_fresh_measurements(self):
        policy = IdleParking({"enabled": True, "pressure_seconds": 2})
        self.assertFalse(policy.pressure(snapshot(100), 100)[0])
        self.assertFalse(policy.pressure(snapshot(100), 102)[0])
        self.assertFalse(policy.pressure(snapshot(101), 103)[0])
        self.assertTrue(policy.pressure(snapshot(103), 103)[0])
        self.assertFalse(policy.pressure(snapshot(104, ram=8), 104)[0])
        self.assertFalse(policy.pressure(snapshot(104), 120)[0])
        for key in ("native_free_mib", "ram_used", "ram_commit_available", "sampled_at"):
            bad = snapshot(121)
            bad.pop(key)
            self.assertFalse(policy.pressure(bad, 121)[0])

    def test_reload_includes_pressure_reserve_so_own_release_is_not_recovery(self):
        policy = IdleParking({"enabled": True, "pressure_ram_available_gib": 10})
        admission = policy.admission({"ram_bytes": 35 * GIB, "commit_bytes": 35 * GIB, "gpu_bytes": 6 * GIB})
        self.assertEqual(admission.required["ram_bytes"], 45 * GIB)
        low = snapshot(100, ram=44, commit=60, gpu=7900)
        self.assertEqual(admission.observe(low, 100)["reason"], "reload_footprint_unavailable")
        for now in range(101, 112):
            recovered = admission.observe(snapshot(now, ram=50, commit=60, gpu=7900), now)
        self.assertTrue(recovered["ready"])

    def test_commit_pressure_alone_earns_dwell(self):
        policy = IdleParking({"enabled": True, "pressure_seconds": 2})
        self.assertFalse(policy.pressure(snapshot(100, ram=20, commit=2), 100)[0])
        self.assertTrue(policy.pressure(snapshot(102, ram=20, commit=2), 102)[0])

    def test_requires_active_single_loaded_engine_and_no_hook(self):
        svc = native_service()
        svc.engine.can_stop = True
        svc.engine.info["background_control"] = 1
        svc.engine.spawn[1].extend(["--pcie-frac", ".5"])
        cfg = {"enabled": True, "mode": "live", "idle_parking": {"enabled": True}}
        svc.before_load = "release-other-model"
        with self.assertRaisesRegex(ValueError, "before_load"):
            svc.configure_coadaptive(cfg)
        svc.before_load = None
        svc.configure_coadaptive(cfg)
        self.assertTrue(svc.idle_parking.enabled)


class IdleServiceTests(unittest.TestCase):
    def service(self):
        tok = Tokenizer()
        engine = ScriptedEngine(tok, [([65], False)])
        svc = Service(engine, tok, None)
        engine.service = svc
        svc.idle_parking = IdleParking({"enabled": True, "pressure_seconds": 2})
        svc.idle_parking_status = {"enabled": True, "state": "ready"}
        svc._parking_footprint = mock.Mock(return_value={"ram_bytes": 35 * GIB,
            "commit_bytes": 35 * GIB, "gpu_bytes": 6 * GIB})
        svc._parking_identity = mock.Mock(return_value={"runtime": "test"})
        svc._invalidate_live_memory = mock.Mock()
        svc._record_live_load = mock.Mock()
        svc.telemetry = SimpleNamespace(capacity=mock.Mock(return_value=snapshot(time.time())))
        svc.memory_snapshot = mock.Mock(return_value=snapshot(100))
        return svc

    def park(self, svc):
        for now in (100, 101, 102):
            svc.memory_snapshot.return_value = snapshot(now)
            with mock.patch("serve.server.time.time", return_value=now):
                result = svc.observe_idle_pressure()
        self.assertTrue(result)
        self.assertFalse(svc.loaded())
        self.assertEqual(svc.idle_parking_status["state"], "suspended")

    def ready(self, svc, ready=True):
        admission = SimpleNamespace(observe=mock.Mock(return_value={"ready": ready, "reason": "test"}),
                                    failed=mock.Mock())
        svc.idle_parked["admission"] = admission
        return admission

    def test_disabled_and_reservation_prevent_idle_unload(self):
        svc = self.service()
        svc.idle_parking.enabled = False
        self.assertFalse(svc.observe_idle_pressure())
        svc.idle_parking.enabled = True
        with svc.preparation_reservation():
            for now in (100, 101, 102):
                svc.memory_snapshot.return_value = snapshot(now)
                self.assertFalse(svc.observe_idle_pressure())
            self.assertEqual(svc.preparing_requests, 1)
            self.assertEqual(svc.unload(), "busy")
        self.assertEqual(svc.preparing_requests, 0)
        self.assertEqual(svc.engine.unloads, 0)

    def test_busy_fifo_never_waits_and_missing_footprint_never_unloads(self):
        svc = self.service()
        svc.fifo.acquire()
        self.assertFalse(svc.observe_idle_pressure())
        svc.fifo.release()
        svc._parking_footprint.side_effect = ValueError("no footprint")
        for now in (100, 101, 102):
            svc.memory_snapshot.return_value = snapshot(now)
            with mock.patch("serve.server.time.time", return_value=now):
                self.assertFalse(svc.observe_idle_pressure())
        self.assertEqual(svc.engine.unloads, 0)
        self.assertEqual(svc.idle_parking_status["state"], "unavailable")

    def test_wait_heartbeat_cancellation_keeps_engine_unloaded_and_releases_fifo(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        cancel = threading.Event()
        waiting = svc.load_events(cancel)
        self.assertEqual(next(waiting), ("ping", None))
        self.assertEqual(svc.engine.reloads, 0)
        cancel.set()
        with self.assertRaises(RequestParkCancelled):
            list(waiting)
        self.assertFalse(svc.fifo.locked())
        self.assertFalse(svc.memory_loading)
        self.assertEqual(svc.preparing_requests, 0)
        self.assertIsNotNone(svc.idle_parked)

    def test_disconnect_closes_wait_without_loading_or_leaking_ownership(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        waiting = svc.load_events(threading.Event())
        next(waiting)
        waiting.close()
        self.assertFalse(svc.fifo.locked())
        self.assertFalse(svc.memory_loading)
        self.assertEqual(svc.preparing_requests, 0)
        self.assertEqual(svc.engine.reloads, 0)

    def test_success_reloads_once_and_preserves_vram_command(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc)
        svc.vram_reserve = 400
        svc.engine.vram = mock.Mock()
        list(svc.load_events(threading.Event()))
        self.assertTrue(svc.loaded())
        self.assertIsNone(svc.idle_parked)
        self.assertEqual(svc.engine.reloads, 1)
        svc.engine.vram.assert_called_once_with(400)
        svc._record_live_load.assert_called_once_with("idle_pressure_resumed")

    def test_no_replay_on_noncapacity_failure_and_bounded_capacity_retry(self):
        svc = self.service()
        self.park(svc)
        admission = self.ready(svc)
        svc._idle_reload = mock.Mock(side_effect=RequestParkCapacity("out of memory"))
        with self.assertRaises(GpuBusy):
            list(svc.load_events(threading.Event()))
        self.assertEqual(svc._idle_reload.call_count, 3)
        self.assertEqual(admission.failed.call_count, 3)
        self.assertEqual(svc.engine.calls, [])
        svc._idle_reload = mock.Mock(side_effect=EngineDied("illegal memory access"))
        with self.assertRaises(EngineDied):
            list(svc.load_events(threading.Event()))
        self.assertEqual(svc._idle_reload.call_count, 1)

    def test_failed_partial_unload_refuses_requests(self):
        svc = self.service()
        self.park(svc)
        svc.engine.down = False
        with self.assertRaises(EngineStuck):
            list(svc.load_events(threading.Event()))
        self.assertEqual(svc.engine.reloads, 0)

    def test_identity_change_before_reload_refuses_allocation(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc)
        svc._parking_identity.return_value = {"runtime": "changed"}
        with self.assertRaisesRegex(ValueError, "identity"):
            list(svc.load_events(threading.Event()))
        self.assertEqual(svc.engine.reloads, 0)

    def test_direct_prepare_cannot_encode_before_admission(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        svc._prepare = mock.Mock()
        cancel = threading.Event()
        svc.request_owner.cancel = cancel
        cancel.set()
        with self.assertRaises(RequestParkCancelled):
            svc.prepare([], None, {})
        svc._prepare.assert_not_called()
        self.assertEqual(svc.preparing_requests, 0)

    def test_prepare_admission_forwards_heartbeat_and_closes_on_write_failure(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        svc._prepare = mock.Mock()
        svc.request_owner.preparation_heartbeat = mock.Mock(side_effect=BrokenPipeError())
        with self.assertRaises(BrokenPipeError):
            svc.prepare([], None, {})
        svc.request_owner.preparation_heartbeat.assert_called_once_with()
        svc._prepare.assert_not_called()
        self.assertFalse(svc.memory_loading)
        self.assertFalse(svc.fifo.locked())
        self.assertEqual(svc.preparing_requests, 0)

    def test_explicit_unload_captures_guard_and_minimum_vram(self):
        svc = self.service()
        svc.min_free_vram_mib = 8000
        self.assertEqual(svc.unload(), "unloaded")
        self.assertEqual(svc.idle_parked["admission"].required["gpu_bytes"], 8000 * MIB)
        self.assertEqual(svc.idle_parking_status["state"], "suspended")

    def test_active_only_manual_and_timed_unload_keep_next_request_admission(self):
        for idle_for in (None, 1):
            with self.subTest(idle_for=idle_for):
                svc = self.service()
                svc.idle_parking.enabled = False
                svc.request_parking.enabled = True
                svc.last_request_at = time.time() - 100
                self.assertEqual(svc.unload(idle_for=idle_for), "unloaded")
                self.assertIsNotNone(svc.idle_parked)
                self.ready(svc)
                list(svc.load_events(threading.Event()))
                self.assertTrue(svc.loaded())
                self.assertIsNone(svc.idle_parked)
                self.assertEqual(svc.engine.reloads, 1)

    def test_shutdown_withdraws_admission_without_loading(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        waiting = svc.load_events(threading.Event())
        next(waiting)
        svc.stop_background()
        with self.assertRaises(RequestParkCancelled):
            list(waiting)
        self.assertEqual(svc.engine.reloads, 0)

    def test_dead_vision_without_recorded_footprint_is_not_restarted_blindly(self):
        svc = self.service()
        svc.vision = mock.Mock()
        svc.vision.alive.return_value = False
        with self.assertRaises(GpuBusy):
            list(svc.load_events(threading.Event()))
        svc.vision.restart.assert_not_called()

    def test_concurrent_loads_have_one_native_owner_and_one_reload(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc)
        entered, release = threading.Event(), threading.Event()
        original = svc._idle_reload
        errors = []
        def reload(stopped):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test reload did not release")
            original(stopped)
        svc._idle_reload = reload
        def load():
            try:
                svc.load()
            except BaseException as exc:
                errors.append(exc)
        first = threading.Thread(target=load)
        second = threading.Thread(target=load)
        first.start()
        self.assertTrue(entered.wait(2))
        second.start()
        self.assertEqual(svc.unload(), "busy")
        release.set()
        first.join(3)
        second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(svc.engine.reloads, 1)
        self.assertEqual(svc.preparing_requests, 0)

    def test_prepare_failure_releases_reservation(self):
        svc = self.service()
        svc._prepare = mock.Mock(side_effect=ValueError("bad picture"))
        with self.assertRaisesRegex(ValueError, "bad picture"):
            svc.prepare([], [], {})
        self.assertEqual(svc.preparing_requests, 0)

    def interleaved_vision(self, svc):
        """Another request parks/cancels while this owner downloads a picture."""
        vision = SimpleNamespace(down=False)
        vision.alive = lambda: not vision.down
        vision.unload = lambda: setattr(vision, "down", True)
        vision.restart = lambda: setattr(vision, "down", False)
        svc.vision = vision
        def download(_):
            with svc.fifo:
                svc.engine.unload()
                vision.unload()
                svc.idle_parked = {"footprint": svc._parking_footprint(),
                    "identity": svc._parking_identity(), "admission": None}
                self.ready(svc)
            return b"fixture"
        return vision, download

    def test_tool_image_rechecks_admission_after_download(self):
        svc = self.service()
        vision, download = self.interleaved_vision(svc)
        observed = []
        vision.encode = lambda *_: observed.append((svc.loaded(), vision.alive(), svc.fifo.locked()))
        svc.encode_prompt = mock.Mock(return_value=[1])
        messages = [{"role": "tool", "content": [{"type": "image", "source": "https://fixture.invalid/image"}]}]
        with mock.patch.object(Vision, "download", side_effect=download), \
                mock.patch("serve.server.images_of", return_value=[]):
            svc.prepare(messages, None, {})
        self.assertEqual(observed, [(True, True, True)])
        self.assertEqual(svc.engine.reloads, 1)
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(svc.preparing_requests, 0)

    def test_user_image_rechecks_admission_after_download(self):
        svc = self.service()
        vision, download = self.interleaved_vision(svc)
        observed = []
        def encode_all(_):
            observed.append((svc.loaded(), vision.alive(), svc.fifo.locked()))
            raise ValueError("fixture reached guarded encoder")
        vision.encode_all = encode_all
        svc.encode_prompt = mock.Mock(return_value=[1])
        with mock.patch.object(Vision, "download", side_effect=download), \
                mock.patch("serve.server.images_of", return_value=["https://fixture.invalid/image"]):
            with self.assertRaisesRegex(ValueError, "fixture reached guarded encoder"):
                svc.prepare([], None, {})
        self.assertEqual(observed, [(True, True, True)])
        self.assertEqual(svc.engine.reloads, 1)
        self.assertEqual(svc.preparing_requests, 0)

    def test_vision_readmission_heartbeats_and_cancels_without_encoding(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        cancel = threading.Event()
        svc.request_owner.cancel = cancel
        svc.request_owner.preparation_heartbeat = mock.Mock(side_effect=cancel.set)
        with self.assertRaises(RequestParkCancelled):
            with svc._vision_turn():
                self.fail("capacity wait reached encoder")
        svc.request_owner.preparation_heartbeat.assert_called_once_with()
        self.assertEqual(svc.engine.reloads, 0)
        self.assertFalse(svc.memory_loading)
        self.assertFalse(svc.fifo.locked())

    def test_vision_queue_cancellation_never_reaches_encoder(self):
        svc = self.service()
        cancel = threading.Event()
        cancel.set()
        svc.request_owner.cancel = cancel
        svc.fifo.acquire()
        try:
            with self.assertRaises(RequestParkCancelled):
                with svc._vision_turn():
                    self.fail("cancelled queue reached encoder")
            self.assertTrue(svc.fifo.locked(), "must not release another owner's FIFO")
        finally:
            svc.fifo.release()

    def test_vision_heartbeat_write_failure_closes_admission(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        svc.request_owner.preparation_heartbeat = mock.Mock(side_effect=BrokenPipeError())
        with self.assertRaises(BrokenPipeError):
            with svc._vision_turn():
                self.fail("disconnected wait reached encoder")
        self.assertFalse(svc.memory_loading)
        self.assertFalse(svc.fifo.locked())

    def test_active_only_cancellation_retains_guard_for_next_request(self):
        from serve.test_request_parking_service import ParkingServiceTests
        fixture = ParkingServiceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        svc = fixture.service([(list(b"abc"), True)], ready=False)
        self.assertFalse(svc.idle_parking.enabled)
        cancel = threading.Event()
        gen = svc.run([1], False, [], 100, {}, cancel)
        while next(gen)[0] != "ping":
            pass
        cancel.set()
        list(gen)
        self.assertIsNotNone(svc.idle_parked)
        self.assertTrue(svc._preparation_admission_enabled())
        self.assertFalse(svc.loaded())
        next_cancel = threading.Event()
        svc.request_owner.cancel = next_cancel
        svc.request_owner.preparation_heartbeat = next_cancel.set
        with self.assertRaises(RequestParkCancelled):
            with svc._vision_turn():
                self.fail("canceled active request allowed an unadmitted encode")
        self.assertEqual(svc.engine.reloads, 0)
        self.assertFalse(svc.fifo.locked())

    def test_vision_checks_newly_parked_state_after_waiting_for_fifo(self):
        svc = self.service()
        svc.idle_parking.enabled = False
        queued = threading.Event()
        cancel = threading.Event()
        errors, encoded = [], []
        svc.fifo.acquire()
        def encode():
            svc.request_owner.cancel = cancel
            def heartbeat():
                queued.set()
                if svc.idle_parked is not None:
                    cancel.set()
            svc.request_owner.preparation_heartbeat = heartbeat
            try:
                with svc._vision_turn():
                    encoded.append(True)
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=encode)
        worker.start()
        try:
            self.assertTrue(queued.wait(2))
            svc.engine.unload()
            svc.idle_parked = {"footprint": svc._parking_footprint(), "identity": svc._parking_identity(),
                               "admission": None}
            self.ready(svc, False)
        finally:
            svc.fifo.release()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(encoded, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestParkCancelled)
        self.assertFalse(svc.fifo.locked())


class IdleFootprintTests(unittest.TestCase):
    def footprint(self, *, rss, arena, peak=None, vision_rss=0, vision_peak=None):
        svc = native_service()
        svc.engine.proc.pid = 123
        args = svc.engine.spawn[1]
        args[args.index("--resident-budget-gib") + 1] = "32"
        svc.engine.native_capacity = {"proc": svc.engine.proc, "sampled_at": time.time(),
            "resident_mib": arena * 1024, "cache_mib": 100, "usage_mib": 6000,
            "total_mib": 8192, "free_mib": 2192}
        native_info = SimpleNamespace(rss=rss * GIB, private=40 * GIB, vms=50 * GIB)
        if peak is not None:
            native_info.peak_wset = peak * GIB
        records = {123: native_info}
        if vision_rss:
            svc.vision = SimpleNamespace(alive=lambda: True, proc=SimpleNamespace(pid=456), spawn=(["vision"], None, {}))
            records[456] = SimpleNamespace(rss=vision_rss * GIB, private=vision_rss * GIB,
                                           vms=vision_rss * GIB)
            if vision_peak is not None:
                records[456].peak_wset = vision_peak * GIB
        with mock.patch("psutil.Process", side_effect=lambda pid: SimpleNamespace(memory_info=lambda: records[pid])):
            return svc._parking_footprint()

    def test_trimmed_arena_preserves_process_peak_and_separate_commit(self):
        footprint = self.footprint(rss=16, arena=32, peak=36)
        self.assertEqual(footprint["ram_bytes"], 38 * GIB)
        self.assertEqual(footprint["commit_bytes"], 42 * GIB)

    def test_shrunken_arena_does_not_add_delta_to_old_full_peak(self):
        self.assertEqual(self.footprint(rss=20, arena=16, peak=36)["ram_bytes"], 38 * GIB)

    def test_native_and_vision_highwater_are_both_retained(self):
        self.assertEqual(self.footprint(rss=20, arena=16, peak=36, vision_rss=1,
                                       vision_peak=2)["ram_bytes"], 40 * GIB)

    def test_missing_or_invalid_peak_retains_cap_and_current_vision(self):
        for peak in (None, float("nan"), float("inf"), -1):
            with self.subTest(peak=peak):
                self.assertEqual(self.footprint(rss=16, arena=32, peak=peak,
                                               vision_rss=1)["ram_bytes"], 35 * GIB)


class IdleHTTPTests(unittest.TestCase):
    service = IdleServiceTests.service
    park = IdleServiceTests.park
    ready = IdleServiceTests.ready
    interleaved_vision = IdleServiceTests.interleaved_vision
    def setUp(self):
        self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def start(self, svc):
        with mock.patch.object(svc, "start_telemetry"):
            self.httpd = serve(svc, port=0)

    def connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=3)

    def test_invalid_static_input_returns400_before_wait_or_headers(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        self.start(svc)
        conn = self.connection()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True, "reasoning_budget_tokens": "bad"}), {"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 400)
            self.assertEqual(response.getheader("Content-Type"), "application/json")
            response.read()
        finally:
            conn.close()
        self.assertEqual(svc.engine.reloads, 0)
        self.assertEqual(svc.preparing_requests, 0)

    def test_streamed_admission_heartbeat_and_disconnect_before_prepare(self):
        svc = self.service()
        self.park(svc)
        self.ready(svc, False)
        svc._prepare = mock.Mock()
        self.start(svc)
        conn = self.connection()
        conn.request("POST", "/v1/chat/completions", json.dumps({
            "messages": [{"role": "user", "content": "hi"}], "stream": True}),
            {"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.readline(), b": waiting for model capacity\n")
        response.close()
        conn.close()
        deadline = time.monotonic() + 3
        while svc.preparing_requests and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(svc.preparing_requests, 0)
        self.assertEqual(svc.engine.reloads, 0)
        svc._prepare.assert_not_called()
        self.assertFalse(svc.fifo.locked())

    def test_exhausted_capacity_retry_is_terminal_sse_error_without_second_headers(self):
        svc = self.service()
        self.park(svc)
        admission = self.ready(svc)
        reads = iter([False, True, True, True])
        admission.observe.side_effect = lambda *_: {"ready": next(reads), "reason": "test"}
        svc._idle_reload = mock.Mock(side_effect=RequestParkCapacity("out of memory"))
        self.start(svc)
        conn = self.connection()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": "hi"}], "stream": True}),
                {"Content-Type": "application/json"})
            response = conn.getresponse()
            body = response.read()
            self.assertEqual(response.status, 200)
            self.assertIn(b": waiting for model capacity", body)
            self.assertIn(b"bounded capacity retries", body)
            self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
            self.assertNotIn(b"HTTP/", body)
        finally:
            conn.close()
        self.assertEqual(svc.preparing_requests, 0)
        self.assertFalse(svc.fifo.locked())

    def test_noncapacity_startup_error_is_structured_and_never_retried(self):
        svc = self.service()
        self.park(svc)
        admission = self.ready(svc)
        reads = iter([False, True])
        admission.observe.side_effect = lambda *_: {"ready": next(reads), "reason": "test"}
        svc._idle_reload = mock.Mock(side_effect=RuntimeError("invalid model header"))
        self.start(svc)
        conn = self.connection()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": "hi"}], "stream": True}),
                {"Content-Type": "application/json"})
            response = conn.getresponse()
            body = response.read()
            self.assertIn(b"invalid model header", body)
            self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
            self.assertNotIn(b"HTTP/", body)
        finally:
            conn.close()
        self.assertEqual(svc._idle_reload.call_count, 1)

    def test_request_reservation_covers_load_to_prepare_and_generation_gap(self):
        svc = self.service()
        entered, release = threading.Event(), threading.Event()
        def prepare(*_):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test preparation did not release")
            return [1], False, 8
        svc._prepare = prepare
        self.start(svc)
        results = []
        def request():
            conn = self.connection()
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps({
                    "messages": [{"role": "user", "content": "hi"}]}),
                    {"Content-Type": "application/json"})
                response = conn.getresponse()
                results.append((response.status, response.read()))
            finally:
                conn.close()
        thread = threading.Thread(target=request)
        thread.start()
        self.assertTrue(entered.wait(2))
        self.assertGreater(svc.preparing_requests, 0)
        self.assertEqual(svc.unload(), "busy")
        self.assertFalse(svc.observe_idle_pressure())
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0][0], 200)
        self.assertEqual(svc.engine.unloads, 0)
        self.assertEqual(svc.preparing_requests, 0)

    def test_image_readmission_after_download_keeps_http_heartbeat_and_disconnect(self):
        svc = self.service()
        vision, interleave = self.interleaved_vision(svc)
        vision.encode_all = mock.Mock()
        svc.encode_prompt = mock.Mock(return_value=[1])
        def download(source):
            data = interleave(source)
            self.ready(svc, False)
            return data
        self.start(svc)
        conn = self.connection()
        with mock.patch.object(Vision, "download", side_effect=download):
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps({
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {
                        "url": "https://fixture.invalid/image"}}]}], "stream": True}),
                    {"Content-Type": "application/json"})
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.readline(), b": waiting for model capacity\n")
                response.close()
            finally:
                conn.close()
            deadline = time.monotonic() + 3
            while svc.preparing_requests and time.monotonic() < deadline:
                time.sleep(.02)
        vision.encode_all.assert_not_called()
        self.assertEqual(svc.engine.reloads, 0)
        self.assertEqual(svc.preparing_requests, 0)
        self.assertFalse(svc.fifo.locked())
        self.assertFalse(svc.memory_loading)


class VisionTeardownTests(unittest.TestCase):
    def encoder(self, alive=True):
        vision = Vision.__new__(Vision)
        vision.proc = mock.Mock(stdin=io.StringIO())
        vision.proc.poll.return_value = None if alive else 0
        vision.stopped = False
        return vision

    def test_unkillable_process_handle_is_retained_and_replacement_refused(self):
        vision = self.encoder()
        original = vision.proc
        original.wait.side_effect = subprocess.TimeoutExpired("encoder", 10)
        vision._start = mock.Mock()
        with self.assertRaises(EngineStuck):
            vision.restart()
        self.assertIs(vision.proc, original)
        self.assertTrue(vision.alive())
        vision._start.assert_not_called()

    def test_failed_ready_retains_unkillable_process(self):
        vision = self.encoder()
        original = vision.proc
        original.wait.side_effect = subprocess.TimeoutExpired("encoder", 10)
        vision.spawn = (["unused"], None, {})
        vision.dir = "."
        vision._readline = mock.Mock(side_effect=RuntimeError("did not start"))
        with mock.patch("serve.server.popen", return_value=original), mock.patch("serve.server.contain"):
            with self.assertRaises(EngineStuck):
                vision._start()
        self.assertIs(vision.proc, original)

    def test_verified_death_clears_handle(self):
        vision = self.encoder()
        original = vision.proc
        original.wait.side_effect = lambda **_: setattr(original.poll, "return_value", 0)
        vision.close()
        self.assertIsNone(vision.proc)
        self.assertTrue(vision.stopped)

    def test_unloaded_public_counters_masked_but_internal_descriptors_retained(self):
        svc = native_service()
        svc.engine.known_ctx = 65536
        svc.engine.max_context = 65536
        svc.engine.native_capacity = {"proc": svc.engine.proc, "free_mib": 300}
        svc.engine.background_state = {"state": "running"}
        info_before = dict(svc.engine.info)
        svc.engine.unloaded = True
        svc.engine.ended = True
        svc.engine.proc.poll = lambda: 0
        self.assertEqual(svc.memory_status()["native_capacity"], {})
        self.assertIsNone(svc.memory_status()["background_native"])
        metrics = svc.metrics()
        self.assertNotIn("arena_mib", metrics["engine"])
        self.assertEqual(metrics["engine"]["max_context"], 65536)
        self.assertEqual(svc.engine.info, info_before)


if __name__ == "__main__":
    unittest.main()
