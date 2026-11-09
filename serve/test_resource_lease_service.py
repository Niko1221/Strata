"""Supervisor handoff lifecycle, preparation concurrency and HTTP authority."""
import contextlib
import http.client
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from serve import server
from serve.resource_lease import ToolLeases
from serve.test_resource_lease import acquire_request, capacity

TOKEN = "isolated-test-control-" * 3


class LeaseServiceTests(unittest.TestCase):
    def setUp(self):
        from serve.test_idle_parking_service import IdleServiceTests
        self.svc = IdleServiceTests().service()
        self.now = 0
        with mock.patch.dict("os.environ", {"STRATA_RESOURCE_LEASE_TOKEN": TOKEN}):
            self.svc.tool_leases = ToolLeases({"enabled": True}, clock=lambda: self.now)
        self.svc.telemetry.capacity.side_effect = lambda: capacity(time.time())
        self.httpd = None

    def tearDown(self):
        self.svc.stop_background()
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def acquire(self):
        return self.svc.resource_lease_action(acquire_request())

    def release(self, row):
        return self.svc.resource_lease_action({"action": "release", "lease_token": row["lease_token"]})

    def state(self, row):
        return self.svc.resource_lease_action({"action": "status", "lease_token": row["lease_token"]})["state"]

    def start(self):
        with mock.patch.object(self.svc, "start_telemetry"):
            self.httpd = server.serve(self.svc, port=0)

    def connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=3)

    def post(self, request, headers=None, method="POST"):
        conn = self.connection()
        try:
            conn.request(method, "/v1/resource-lease", json.dumps(request) if method == "POST" else None,
                         {"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN,
                          **(headers or {})})
            response = conn.getresponse()
            return response.status, json.loads(response.read()), response.getheader("Access-Control-Allow-Origin")
        finally:
            conn.close()

    def test_acquire_only_queues_no_request_or_lifecycle_replay(self):
        row = self.acquire()
        self.assertEqual(row["state"], "pending")
        self.assertEqual(self.svc.engine.unloads, 0)
        self.assertTrue(self.svc.observe_tool_lease())
        self.assertEqual(self.svc.engine.unloads, 1)
        self.assertFalse(self.svc.loaded())
        self.assertEqual(self.state(row), "ready")
        self.release(row)
        self.assertEqual(self.svc.engine.reloads, 0)
        self.svc.idle_parked["admission"].observe = lambda *_: {"ready": True}
        self.svc.load()
        self.assertEqual(self.svc.engine.reloads, 1)

    def test_existing_preparation_may_finish_but_new_preparation_waits_outside_fifo(self):
        svc = self.svc
        with svc.preparation_reservation():
            row = self.acquire()
            self.assertFalse(svc.observe_tool_lease())
            with svc.preparation_reservation():
                self.assertEqual(svc.preparing_requests, 1)
        cancel, heartbeat, finished = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def waiter():
            events = svc.load_events(cancel)
            try:
                for _ in events:
                    heartbeat.set()
            except server.RequestParkCancelled:
                pass
            except BaseException as exc:
                errors.append(exc)
            finally:
                events.close()
                finished.set()
        worker = threading.Thread(target=waiter)
        worker.start()
        self.assertTrue(heartbeat.wait(2))
        self.assertEqual(svc.preparing_requests, 0)
        self.assertFalse(svc.fifo.locked())
        self.assertTrue(svc.observe_tool_lease())
        self.assertEqual(self.state(row), "ready")
        cancel.set()
        worker.join(2)
        self.assertTrue(finished.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(svc.preparing_requests, 0)

    def test_expiry_during_footprint_sampling_does_not_unload_or_install_parked_state(self):
        row = self.acquire()
        footprint = self.svc._parking_footprint.return_value
        def expired():
            self.now = 31
            return footprint
        self.svc._parking_footprint.side_effect = expired
        self.assertFalse(self.svc.observe_tool_lease())
        self.assertEqual(self.svc.engine.unloads, 0)
        self.assertIsNone(self.svc.idle_parked)
        self.assertEqual(self.state(row), "expired")

    def test_release_during_unload_waits_for_confirmed_death(self):
        row = self.acquire()
        entered, finish = threading.Event(), threading.Event()
        original = self.svc._parking_unload
        def slow_unload(stopped):
            entered.set()
            if not finish.wait(2):
                raise RuntimeError("test deadline")
            original(stopped)
        self.svc._parking_unload = slow_unload
        worker = threading.Thread(target=self.svc.observe_tool_lease)
        worker.start()
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.release(row)["state"], "unloading")
        self.assertTrue(self.svc.tool_leases.blocked())
        finish.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.svc.tool_leases.blocked())
        self.assertEqual(self.state(row), "released")
        self.assertEqual(self.svc.engine.reloads, 0)

    def test_missing_footprint_and_uncertain_vision_death_never_grant_ready(self):
        row = self.acquire()
        self.svc._parking_footprint.side_effect = ValueError("missing")
        self.assertFalse(self.svc.observe_tool_lease())
        self.assertEqual(self.state(row), "failed")
        self.assertEqual(self.svc.engine.unloads, 0)
        self.svc._parking_footprint.side_effect = None
        row = self.acquire()
        self.svc.vision = SimpleNamespace(alive=lambda: True, unload=lambda: None)
        self.assertFalse(self.svc.observe_tool_lease())
        self.assertEqual(self.state(row), "failed")
        self.assertIsNotNone(self.svc.idle_parked)
        with self.assertRaises(server.EngineStuck):
            self.svc.load()

    def test_direct_generation_does_not_join_fifo_ahead_of_pending_lease(self):
        row = self.acquire()
        cancel = threading.Event()
        gen = self.svc.run([1], False, None, 8, {}, cancel)
        self.assertEqual(next(gen), ("ping", None))
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertEqual(self.svc.preparing_requests, 0)
        self.assertFalse(self.svc.fifo.locked())
        self.assertTrue(self.svc.observe_tool_lease())
        cancel.set()
        with self.assertRaises(server.RequestParkCancelled):
            next(gen)
        gen.close()
        self.release(row)

    def test_separate_control_auth_origin_rejection_and_public_redaction(self):
        self.svc.api_key = "different-model-api-key"
        self.svc.cors_origins = ["*"]
        self.start()
        req = acquire_request()
        for headers, expected in (({"Authorization": "Bearer different-model-api-key"}, 401),
                                  ({"Origin": "http://127.0.0.1"}, 403), ({"Origin": "null"}, 403),
                                  ({"Sec-Fetch-Site": "same-origin"}, 403)):
            status, _, cors = self.post(req, headers)
            self.assertEqual(status, expected)
            self.assertIsNone(cors)
        status, row, _ = self.post(req)
        self.assertEqual(status, 200)
        self.assertEqual(row["state"], "pending")
        status, public, _ = self.post({}, {"Authorization": ""}, method="GET")
        self.assertEqual(status, 200)
        self.assertNotIn("lease_id", public)
        self.assertNotIn("lease_token", public)
        status, _, _ = self.post({"action": "status", "lease_token": "wrong"})
        self.assertEqual(status, 403)

    def test_stream_wait_has_heartbeat_cancels_without_reservation_or_load(self):
        self.acquire()
        self.start()
        self.svc._prepare = mock.Mock(side_effect=AssertionError("must not prepare"))
        conn = self.connection()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": "hello"}], "stream": True}),
                {"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b"supervisor resource lease", response.readline())
            self.assertEqual(self.svc.preparing_requests, 0)
            self.assertFalse(self.svc.fifo.locked())
            self.svc.stop_background()
            response.read()
        finally:
            conn.close()
        self.svc._prepare.assert_not_called()
        self.assertEqual(self.svc.engine.reloads, 0)

    def test_two_waiting_requests_share_one_guarded_reload_after_release(self):
        row = self.acquire()
        self.assertTrue(self.svc.observe_tool_lease())
        self.svc.idle_parked["admission"].observe = lambda *_: {"ready": True}
        heard = [threading.Event(), threading.Event()]
        errors = []
        def load(index):
            try:
                for _ in self.svc.load_events(threading.Event()):
                    heard[index].set()
            except BaseException as exc:
                errors.append(exc)
        workers = [threading.Thread(target=load, args=(i,)) for i in range(2)]
        for worker in workers:
            worker.start()
        try:
            self.assertTrue(all(beat.wait(2) for beat in heard))
            self.assertFalse(self.svc.fifo.locked())
            self.assertEqual(self.svc.preparing_requests, 0)
            self.release(row)
        finally:
            for worker in workers:
                worker.join(3)
        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertEqual(errors, [])
        self.assertEqual(self.svc.engine.reloads, 1)
        self.assertEqual(self.svc.preparing_requests, 0)

    def test_readiness_respects_higher_configured_reserves_and_missing_commit(self):
        from serve.memory_policy import MemoryPolicy
        row = self.acquire()
        self.svc.memory_policy = MemoryPolicy({"min_ram_headroom_gib": 45}, resident_cap_gib=32,
                                             vram_reserve_mib=7500)
        self.assertTrue(self.svc.observe_tool_lease())
        self.assertEqual(self.state(row), "pending")
        self.svc.telemetry.capacity.side_effect = lambda: capacity(time.time(), ram=50, gpu=7900)
        self.assertEqual(self.state(row), "ready")
        self.svc.telemetry.capacity.side_effect = lambda: {
            **capacity(time.time(), ram=50, gpu=7900), "ram_commit_required": True,
            "ram_commit_available": None}
        self.assertEqual(self.state(row), "pending")

    def test_direct_load_under_fifo_fails_instead_of_blocking_unload_owner(self):
        self.acquire()
        with self.svc.fifo:
            with self.assertRaisesRegex(server.GpuBusy, "supervisor"):
                list(self.svc._admit_loaded(threading.Event()))
        self.assertTrue(self.svc.observe_tool_lease())

    def test_cancelled_waiter_cannot_enter_after_owner_releases(self):
        row = self.acquire()
        cancel = threading.Event()
        events = self.svc.load_events(cancel)
        self.assertEqual(next(events), ("ping", None))
        cancel.set()
        self.release(row)
        with self.assertRaises(server.RequestParkCancelled):
            next(events)
        events.close()
        self.assertEqual(self.svc.engine.reloads, 0)
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.preparing_requests, 0)


if __name__ == "__main__":
    unittest.main()
