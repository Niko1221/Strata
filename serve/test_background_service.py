"""Lease delivery and native process-budget feedback without a GPU or model."""
import unittest
from unittest import mock

from serve.server import Service, StrataEngine
from serve.test_live_memory import service, engine, propose, deliver, ack


class BackgroundServiceTests(unittest.TestCase):
    def controlled(self):
        svc = service()
        svc.engine.info["background_control"] = 1
        svc.coadaptive = mock.Mock(active=True)
        svc.coadaptive.fairness_decision.return_value = {"delay_ms": 10, "reason": "external_cpu"}
        svc.coadaptive.observe.return_value = False
        svc.memory_policy.safety_decision = mock.Mock(return_value={"action": "wait", "reason": "critical_ram"})
        return svc

    def test_renew_wait_while_memory_request_pending(self):
        svc = self.controlled()
        svc.memory_live_pending = {"proc": svc.engine.proc, "id": 1}
        with mock.patch("serve.server.time.time", return_value=601):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "BACKGROUND 5000 10 1\n")
        with mock.patch("serve.server.time.time", return_value=603):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue().count("BACKGROUND"), 2)

    def test_recovery_releases_wait_immediately(self):
        svc = self.controlled()
        svc._observe_background({}, 10)
        svc.memory_policy.safety_decision.return_value = {"action": "run", "reason": "stable"}
        svc.coadaptive.fairness_decision.return_value = {"delay_ms": 0, "reason": "quiet"}
        svc._observe_background({}, 10.1)
        self.assertEqual(svc.engine.proc.stdin.getvalue().splitlines(),
                         ["BACKGROUND 5000 10 1", "BACKGROUND 0 0 0"])
        svc._observe_background({}, 20)
        self.assertEqual(svc.engine.proc.stdin.getvalue().count("BACKGROUND"), 2)

    def test_disable_live_mode_releases_existing_lease(self):
        svc = self.controlled()
        svc._observe_background({}, 10)
        svc.coadaptive.active = False
        svc._observe_background({}, 11)
        self.assertTrue(svc.engine.proc.stdin.getvalue().endswith("BACKGROUND 0 0 0\n"))

    def test_old_native_binary_gets_no_unknown_command(self):
        svc = self.controlled()
        del svc.engine.info["background_control"]
        svc._observe_background({}, 10)
        self.assertEqual(svc.engine.proc.stdin.getvalue(), "")
        self.assertFalse(svc.background_control["supported"])

    def test_invalid_and_stale_process_commands_cannot_write(self):
        e = engine()
        e.info["background_control"] = 1
        for args in [(5001, 0, True), (5000, 101, False), (0, 0, True), (5000, 0, 1)]:
            with self.assertRaises(ValueError):
                e.request_background(*args, e.proc)
        with self.assertRaises(OSError):
            e.request_background(5000, 0, True, object())
        self.assertEqual(e.proc.stdin.getvalue(), "")

    def test_local_budget_is_conservative_including_zero(self):
        svc = self.controlled()
        svc.telemetry = mock.Mock()
        svc.telemetry.snapshot.return_value = {"now": {"sampled_at": 100}}
        del svc.memory_snapshot  # use the real method instead of the fixture stub
        for free in (200, 0):
            svc.engine.native_capacity = {
                "free_mib": 600, "total_mib": 8192, "resident_mib": 1000,
                "cache_mib": 500, "budget_mib": 7000, "usage_mib": 7000 - free,
                "budget_free_mib": free, "proc": svc.engine.proc, "sampled_at": 100}
            with mock.patch("serve.server.time.time", return_value=100):
                reading = svc.memory_snapshot()
            self.assertEqual(reading["native_free_mib"], free)
            self.assertEqual(reading["native_cuda_free_mib"], 600)
            self.assertEqual(reading["gpu_mem_used"], (8192 - free) * 2**20)

    def test_capacity_parser_rejects_partial_duplicate_or_bad_budget(self):
        base = "CAPACITY free_mib=600 total_mib=8192 resident_mib=1024 cache_mib=512"
        for tail in (" budget_mib=0", " free_mib=600",
                     " budget_mib=1 usage_mib=0 budget_free_mib=2",
                     " budget_mib=1 usage_mib=-1 budget_free_mib=0"):
            self.assertIsNone(StrataEngine._capacity_reading(base + tail, None, 100))
        good = StrataEngine._capacity_reading(
            base + " budget_mib=0 usage_mib=8000 budget_free_mib=0", None, 100)
        self.assertEqual(good["budget_free_mib"], 0)
        self.assertIsNotNone(StrataEngine._capacity_reading(base, None, 100))

    def test_invalidation_drops_previous_process_lease(self):
        svc = self.controlled()
        svc._observe_background({}, 10)
        svc._invalidate_live_memory()
        self.assertIsNone(svc.background_sent)
        self.assertEqual(svc.background_control["reason"], "engine_unavailable")

    def test_background_heartbeat_does_not_break_session_save(self):
        e = engine()
        e.silence_s = 10
        e.lines.put("BACKGROUND status=waiting lease_remaining_ms=4000\n")
        e.lines.put("BACKGROUND status=running\n")
        e.lines.put("SAVED 30 4096 1.5\n")
        with mock.patch.object(e, "_write", wraps=e._write) as write:
            self.assertEqual(e.session_file("save", "session.dat"), {"tokens": 30, "bytes": 4096, "ms": 1.5})
        write.assert_called_once_with("SAVE session.dat")
        self.assertFalse(e.ended)

    def test_background_heartbeat_is_not_a_generated_token(self):
        e = engine()
        e.silence_s = 10
        e.lines.put("BACKGROUND status=waiting lease_remaining_ms=4000\n")
        e.lines.put("BACKGROUND status=running\n")
        e.lines.put("T 42\n")
        e.lines.put("DONE 1 2 0 1 stop\n")
        import threading
        values = list(e.generate([1, 2], 10, {}, threading.Event()))
        self.assertEqual(values, [None, None, 42])

    def pending(self, svc):
        svc.memory_live_pending = {"id": 7, "proc": svc.engine.proc, "sent_at": 100, "progress_at": 105,
                                   "plan": {"resident_budget_gib": 40, "vram_reserve_mib": 1536},
                                   "resident_before_gib": 42}
        svc.engine.native_capacity = {"proc": svc.engine.proc, "sampled_at": 135, "memory_pending_id": 0,
                                      "memory_last_terminal_id": 7,
                                      "memory_reserve_mib": 1536, "resident_mib": 40960,
                                      "cache_mib": 1024, "free_mib": 600}

    def test_lost_terminal_ack_reconciles_actual_native_sizes(self):
        svc = service()
        self.pending(svc)
        svc._reconcile_memory_control(134)
        self.assertIsNotNone(svc.memory_live_pending)
        svc._reconcile_memory_control(135)
        self.assertIsNone(svc.memory_live_pending)
        self.assertEqual(svc.engine.info["arena_mib"], 40960)
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 40)
        self.assertEqual(svc.memory_error, "terminal_ack_missing")

    def test_reconcile_does_not_guess_active_or_stale_control_complete(self):
        svc = service()
        self.pending(svc)
        svc.engine.native_capacity["memory_pending_id"] = 7
        svc._reconcile_memory_control(135)
        self.assertIsNotNone(svc.memory_live_pending)
        self.assertEqual(svc.memory_last_reason, "native_resize_delayed")
        svc.engine.native_capacity["memory_pending_id"] = 0
        svc._reconcile_memory_control(145)
        self.assertIsNotNone(svc.memory_live_pending)

    def test_queued_but_not_started_command_cannot_be_reconciled_complete(self):
        svc = service()
        self.pending(svc)
        svc.engine.native_capacity["memory_last_terminal_id"] = 6
        svc._reconcile_memory_control(135)
        self.assertIsNotNone(svc.memory_live_pending)
        self.assertEqual(svc.memory_last_reason, "control_feedback_missing")

    def test_identical_failed_pressure_backoffs_but_stronger_relief_can_proceed(self):
        svc = service()
        propose(svc, 600)
        deliver(svc, ack(status="error").strip() + " error=ram_resize\n", now=601)
        propose(svc, 602)
        self.assertEqual(svc.engine.proc.stdin.getvalue().count("MEMORY"), 1)
        svc.memory_policy.observe.return_value = {"resident_budget_gib": 39, "vram_reserve_mib": 1536,
                                                 "reason": "sustained_pressure"}
        with mock.patch("serve.server.time.time", return_value=602):
            svc.observe_memory()
        self.assertEqual(svc.engine.proc.stdin.getvalue().count("MEMORY"), 2)

    def test_stricter_pressure_supersedes_growth_only_when_native_supports_it(self):
        svc = service()
        self.pending(svc)
        svc.engine.info["memory_supersede"] = 1
        svc.memory_policy.observe = mock.Mock(return_value={"resident_budget_gib": 39,
                                                            "vram_reserve_mib": 1536,
                                                            "reason": "sustained_pressure"})
        with mock.patch("serve.server.time.time", return_value=110):
            svc.observe_memory()
        self.assertIn("MEMORY 1 39936 1536", svc.engine.proc.stdin.getvalue())
        self.assertEqual(svc.memory_live_pending["plan"]["resident_budget_gib"], 39)


if __name__ == "__main__":
    unittest.main()
