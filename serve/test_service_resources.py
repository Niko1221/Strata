"""Resource preset persistence, HTTP admission and live allocation ownership."""
import http.client
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.server import Service, ByteTokenizer, serve
from serve.test_live_memory import engine, ack
from serve.test_memory_policy import sample


class ResourceServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "strata.json"
        self.path.write_text(json.dumps({"untouched": {"keep": True}}), encoding="utf-8")
        self.svc = Service(engine(), ByteTokenizer(), None)
        self.svc.config_path = str(self.path)
        self.svc.configure_memory({"enabled": True, "mode": "live", "pressure_seconds": 4})
        self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.tmp.cleanup()

    def start(self):
        with mock.patch.object(self.svc, "start_telemetry"):
            self.httpd = serve(self.svc, port=0)

    def request(self, method, body=None, headers=None, path="/v1/resources"):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=5)
        try:
            connection.request(method, path, json.dumps(body).encode() if body is not None else None,
                               headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_resource_route_exists_and_requires_existing_api_key(self):
        self.svc.api_key = "synthetic-resource-key"
        self.start()
        self.assertEqual(self.request("GET")[0], 401)
        code, body = self.request("GET", headers={"Authorization": "Bearer synthetic-resource-key"})
        self.assertEqual(code, 200)
        self.assertFalse(body["enabled"])
        self.assertTrue(body["available"])

    def test_http_save_preserves_allocation_and_unrelated_config(self):
        before = dict(self.svc.memory_policy.current)
        proc = self.svc.engine.proc
        self.start()
        code, body = self.request("POST", {"enabled": True, "selection": "busy"},
                                  {"Content-Type": "application/json"})
        self.assertEqual(code, 200)
        self.assertEqual(body["effective"], "busy")
        self.assertEqual(body["headroom_gib"], 8)
        self.assertEqual(self.svc.memory_policy.current, before)
        self.assertIs(self.svc.engine.proc, proc)
        self.assertEqual(self.svc.engine.proc.stdin.getvalue(), "")
        self.assertEqual(json.loads(self.path.read_text())["untouched"], {"keep": True})
        self.assertEqual(json.loads(self.path.read_text())["resource_presets"],
                         {"enabled": True, "selection": "busy"})

    def test_cross_origin_and_non_json_requests_cannot_change_targets(self):
        before = self.path.read_bytes()
        self.start()
        for headers, expected in (({"Content-Type": "text/plain"}, 415),
                                  ({"Content-Type": "application/json", "Origin": "https://foreign.invalid"}, 403)):
            self.assertEqual(self.request("POST", {"enabled": True, "selection": "full"}, headers)[0], expected)
        self.assertEqual(self.path.read_bytes(), before)

    def test_validation_and_persistence_failures_leave_live_state_unchanged(self):
        self.start()
        before = self.path.read_bytes()
        for body in ({}, {"enabled": True}, {"selection": "full"},
                     {"enabled": True, "selection": "invalid"},
                     {"enabled": True, "selection": "full", "ram_target_percent": 99}):
            self.assertEqual(self.request("POST", body, {"Content-Type": "application/json"})[0], 400)
        with mock.patch("serve.server.runconfig.save", side_effect=OSError("synthetic write failure")):
            self.assertEqual(self.request("POST", {"enabled": True, "selection": "full"},
                                          {"Content-Type": "application/json"})[0], 503)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.svc.resource_status()["enabled"])

    def test_auto_retargets_unloaded_service_without_loading_model(self):
        self.svc.configure_resources({"enabled": True, "selection": "auto"})
        self.svc.engine.unloaded = True
        proc = self.svc.engine.proc
        for now in range(10, 19, 2):
            reading = sample(now)
            reading["workload"] = {"complete": True,
                                   "cpu_percent": 0, "rss_bytes": 9 * 2**30}
            self.svc.memory_snapshot = mock.Mock(return_value=reading)
            with mock.patch("serve.server.time.time", return_value=now):
                self.svc.observe_memory()
        self.assertEqual(self.svc.resource_status()["effective"], "busy")
        self.assertEqual(self.svc.memory_policy.headroom, 8)
        self.assertIs(self.svc.engine.proc, proc)
        self.assertTrue(self.svc.engine.unloaded)
        self.assertEqual(proc.stdin.getvalue(), "")

    def test_configured_manual_selection_restores_on_new_service(self):
        self.svc.set_resources({"enabled": True, "selection": "daily"})
        other = Service(engine(), ByteTokenizer(), None)
        other.configure_memory({"enabled": True, "mode": "live"})
        other.configure_resources(json.loads(self.path.read_text())["resource_presets"])
        self.assertEqual(other.resource_status()["selection"], "daily")
        self.assertEqual(other.memory_policy.headroom, 4)

    def test_legacy_backup_is_not_overwritten_by_preset_save(self):
        backup = self.path.with_name(self.path.name + ".bak")
        backup.write_bytes(b"unrelated existing backup")
        self.svc.set_resources({"enabled": True, "selection": "full"})
        self.assertEqual(backup.read_bytes(), b"unrelated existing backup")

    def test_retarged_pending_ack_keeps_actual_and_native_error_retry(self):
        self.svc.set_resources({"enabled": True, "selection": "full"})
        old_plan = {"resident_budget_gib": 40, "vram_reserve_mib": 256, "reason": "sustained_pressure"}
        self.svc.memory_live_pending = {"id": 7, "proc": self.svc.engine.proc, "plan": old_plan,
                                       "resident_before_gib": 42,
                                       "resource_generation": self.svc.resource_generation}
        self.svc.memory_retry_at = time.time() + 600
        retry = self.svc.memory_retry_at
        self.svc.set_resources({"enabled": True, "selection": "busy"})
        self.assertEqual(self.svc.memory_live_pending["id"], 7)
        self.assertEqual(self.svc.memory_retry_at, retry)
        self.svc.engine.memory_acks.put((self.svc.engine.proc,
                                         self.svc.engine._memory_ack(ack(7, resident=40960, reserve=256))))
        with self.svc.memory_lock:
            self.svc._drain_memory_acks(time.time())
        self.assertEqual(self.svc.memory_policy.current["resident_budget_gib"], 40)
        self.assertEqual(self.svc.memory_policy.reserve_floor, 1536)
        self.assertEqual(self.svc.memory_retry_at, retry)
        self.assertIsNone(self.svc.memory_live_pending)
        self.assertIsNone(self.svc.memory_policy.pressure_recovery_ceiling_gib)

    def test_private_workload_is_not_in_public_metrics(self):
        self.svc.telemetry = mock.Mock()
        self.svc.telemetry.snapshot.return_value = {"now": {"cpu": 0, "workload": {"private": "fixture"}},
                                                   "history": {"workload": ["fixture"]}, "static": {}}
        result = self.svc.metrics()
        self.assertNotIn("workload", result["hardware"])
        self.assertNotIn("workload", result["history"])
        self.assertIn("resources", result)


if __name__ == "__main__":
    unittest.main()
