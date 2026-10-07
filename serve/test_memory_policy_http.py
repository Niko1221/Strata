"""Request-boundary memory control; no GPU or model files are required."""
import json
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, serve
from serve.test_server import UnloadableEngine
from serve.telemetry import Telemetry

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


class BudgetEngine(UnloadableEngine):
    def __init__(self, tok):
        super().__init__(tok, "</think>\n\nok", max_context=4096)
        self.spawn = ("fake", ["--resident-budget-gib", "42", "--vram-reserve-mib", "1536",
                               "--max-context", "4096"], None, None, None)
        self.info = {"arena_mib": 42 * 1024, "expert_cache_mib": 10 * 1024}
        self.unloads = 0

    def unload(self):
        self.unloads += 1
        super().unload()

    def restart(self):
        super().restart()
        self.info["arena_mib"] = float(self.spawn[1][1]) * 1024


class MemoryPolicyHttp(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = BudgetEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.reading = {"sampled_at": time.time(), "ram_total": 64 * GIB, "ram_used": 60 * GIB,
                        "gpu_mem_total": 24 * GIB, "gpu_mem_used": 22 * GIB}

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def configure(self):
        self.svc.configure_memory({"enabled": True})
        self.svc.memory_policy.applied({"resident_budget_gib": 42, "vram_reserve_mib": 1536,
                                       "reason": "test"}, time.time() - 700)
        self.svc.memory_snapshot = lambda fresh=False: dict(self.reading)

    def test_idle_observation_never_unloads_or_changes_engine(self):
        self.configure()
        before = list(self.engine.spawn[1])
        self.svc.observe_memory()
        self.assertEqual(self.engine.unloads, 0)
        self.assertEqual(self.engine.spawn[1], before)
        self.assertTrue(self.engine.alive())

    def test_pending_plan_changes_budget_only_at_request_boundary(self):
        self.configure()
        self.svc.memory_pending = {"resident_budget_gib": 35, "vram_reserve_mib": 1792,
                                   "reason": "sustained_pressure"}
        self.svc.load()
        self.assertEqual((self.engine.unloads, self.engine.starts), (1, 1))
        self.assertEqual(self.engine.spawn[1][1], "35")
        self.assertEqual(self.engine.spawn[1][3], "1792")
        self.assertIsNone(self.svc.memory_pending)
        self.assertEqual(self.svc.memory_policy.current["resident_budget_gib"], 35)
        self.svc.load()
        self.assertEqual(self.engine.starts, 1)

    def test_busy_work_is_not_replanned_and_refresh_is_deferred(self):
        self.configure()
        self.svc.memory_pending = {"resident_budget_gib": 35, "vram_reserve_mib": 1536, "reason": "test"}
        for state in ("busy", "queued"):
            self.svc.status[state] = True
            with self.svc.fifo:
                self.svc.ensure_loaded()
            self.assertEqual(self.engine.unloads, 0)
            self.svc.status[state] = False
        status, result = self.post("/v1/memory/refresh", {"model": self.svc.model})
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(self.engine.unloads, 0)

    def test_observation_during_load_cannot_erase_plan(self):
        self.configure()
        plan = {"resident_budget_gib": 35, "vram_reserve_mib": 1536, "reason": "test"}
        self.svc.memory_pending = plan
        restart = self.engine.restart
        def load_with_observer():
            self.svc.observe_memory()
            self.assertEqual(self.svc.memory_pending, plan)
            restart()
        self.engine.restart = load_with_observer
        self.svc.load()
        self.assertEqual(self.svc.memory_policy.current["resident_budget_gib"], 35)
        self.assertFalse(self.svc.memory_loading)

    def test_failed_load_does_not_commit_allocation(self):
        self.configure()
        self.svc.memory_pending = {"resident_budget_gib": 35, "vram_reserve_mib": 1536, "reason": "test"}
        self.engine.restart = mock.Mock(side_effect=RuntimeError("failed allocation"))
        with self.assertRaisesRegex(RuntimeError, "failed allocation"):
            self.svc.load()
        self.assertEqual(self.svc.memory_policy.current["resident_budget_gib"], 42)
        self.assertFalse(self.svc.memory_loading)

    def test_refresh_rejects_wrong_model_and_foreign_origin(self):
        self.configure()
        self.assertEqual(self.post("/v1/memory/refresh", {"model": "foreign"})[0], 404)
        self.assertEqual(self.post("/v1/memory/refresh", {"model": self.svc.model},
                                  {"Origin": "https://foreign.example"})[0], 403)
        self.svc.api_key = "fixture-key"
        self.assertEqual(self.post("/v1/memory/refresh", {})[0], 401)
        self.assertEqual(self.post("/v1/memory/refresh", {},
                                  {"Authorization": "Bearer fixture-key"})[0], 200)
        self.assertEqual(self.post("/v1/memory/refresh", {},
                                  {"Authorization": "Bearer fixture-key", "Origin": "https://foreign.example"})[0], 403)
        self.assertEqual(self.engine.unloads, 0)

    def test_disabled_policy_preserves_existing_lifecycle(self):
        self.svc.configure_memory({"enabled": False})
        self.assertEqual(self.svc.memory_status(), {"enabled": False})
        self.svc.observe_memory()
        self.assertEqual(self.engine.unloads, 0)

    def test_memory_configuration_rejects_native_layer_split(self):
        self.engine.spawn[1].extend(["--layer-split", "auto"])
        with self.assertRaisesRegex(ValueError, "one GPU"):
            self.svc.configure_memory({"enabled": True})
        self.assertIsNone(self.svc.memory_policy)

    def test_unloaded_load_uses_fresh_capacity_after_previous_engine_exit(self):
        self.engine.unload()
        self.svc.configure_memory({"enabled": True})
        stale = dict(self.reading)
        fresh = {**stale, "ram_used": 12 * GIB, "sampled_at": time.time()}
        self.svc.telemetry = mock.Mock()
        self.svc.telemetry.snapshot.return_value = {"now": stale}
        self.svc.telemetry.capacity.return_value = fresh
        self.svc.load()
        self.svc.telemetry.capacity.assert_called_once_with()
        self.svc.telemetry.snapshot.assert_not_called()
        self.assertEqual(self.svc.memory_policy.current["resident_budget_gib"], 42)

    def test_already_loaded_startup_adopts_existing_budget(self):
        self.svc.configure_memory({"enabled": True})
        self.assertEqual(self.svc.memory_policy.current["resident_budget_gib"], 42)
        self.assertEqual(self.svc.memory_policy.current["vram_reserve_mib"], 1536)
        self.assertEqual(self.engine.unloads, 0)
        self.assertEqual(self.engine.starts, 0)


class TelemetryCapacity(unittest.TestCase):
    def telemetry(self):
        telemetry = Telemetry.__new__(Telemetry)
        telemetry.ps = mock.Mock()
        telemetry.ps.virtual_memory.return_value = SimpleNamespace(total=64 * GIB, available=20 * GIB)
        telemetry.fallback = mock.Mock()
        reader = mock.Mock()
        reader.ok.return_value = True
        reader.read.return_value = {"mem_total": 24 * GIB, "mem_used": 10 * GIB}
        telemetry.gpus = [(0, reader)]
        telemetry.gpu = reader
        telemetry.now, telemetry.hist = {"old": True}, {"old": [1]}
        telemetry._disk_prev = (1, 2, 3)
        telemetry.extra = mock.Mock(side_effect=AssertionError("capacity called generation callback"))
        return telemetry

    def test_fresh_capacity_does_not_touch_cached_reading_history_or_deltas(self):
        telemetry = self.telemetry()
        with mock.patch("serve.telemetry.time.time", return_value=123), \
                mock.patch("serve.telemetry._commit_capacity", return_value={"ram_commit_available": 12 * GIB}):
            capacity = telemetry.capacity()
        self.assertEqual(capacity, {"sampled_at": 123, "ram_total": 64 * GIB, "ram_used": 44 * GIB,
                                    "gpu_mem_total": 24 * GIB, "gpu_mem_used": 10 * GIB,
                                    "ram_commit_available": 12 * GIB})
        self.assertEqual(telemetry.now, {"old": True})
        self.assertEqual(telemetry.hist, {"old": [1]})
        self.assertEqual(telemetry._disk_prev, (1, 2, 3))
        telemetry.extra.assert_not_called()

    def test_sensor_failure_uses_ram_fallback_and_freezes_missing_gpu_capacity(self):
        telemetry = self.telemetry()
        telemetry.ps.virtual_memory.side_effect = OSError("sensor gone")
        telemetry.fallback.ram.return_value = (12 * GIB, 64 * GIB)
        telemetry.gpu.read.side_effect = OSError("GPU gone")
        capacity = telemetry.capacity()
        self.assertEqual(capacity["ram_used"], 12 * GIB)
        self.assertIsNone(capacity["gpu_mem_used"])
        self.assertIsNone(capacity["gpu_mem_total"])
        telemetry.fallback.ram.side_effect = OSError("fallback gone")
        self.assertIsNone(telemetry.capacity()["ram_total"])

    def test_regular_samples_include_timestamp_for_policy_freshness(self):
        telemetry = self.telemetry()
        telemetry.extra = None
        telemetry._disk = lambda: (None, None)
        with mock.patch("serve.telemetry.time.time", return_value=321):
            self.assertEqual(telemetry.sample()["sampled_at"], 321)


if __name__ == "__main__":
    unittest.main()
