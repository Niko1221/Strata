"""Adversarial resident-lease ordering with real service/policy and fake native IO."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from serve.resource_lease import ToolLeases, LeaseError
from serve.test_live_memory import service, ack
from serve.test_resource_lease import acquire_request, capacity


class ResidentLeaseRaceTests(unittest.TestCase):
    def setUp(self):
        self.wall, self.mono = 100., 0.
        self.svc = svc = service()
        svc.engine.info.update(memory_hold=1, memory_supersede=1, arena_mib=32768,
                               expert_cache_mib=1024, vram_reserve_mib=800)
        svc.memory_policy.live_actual(32768, 800, 0, "loaded", completed=True, loaded=True)
        svc.memory_policy.update_resource_limits(3, 320)
        svc.memory_policy.reclaim_gpu_headroom = True
        svc.resource_limits = (3, 320)
        svc.coadaptive = SimpleNamespace(active=True, observe=lambda *a, **kw: False,
                                        limits=lambda: (3, 320))
        svc._observe_background = mock.Mock()
        with mock.patch.dict("os.environ", {"STRATA_RESOURCE_LEASE_TOKEN": "unit-test-control-" * 3}):
            svc.tool_leases = ToolLeases({"enabled": True}, clock=lambda: self.mono)
        svc.telemetry = SimpleNamespace(capacity=lambda: self.snapshot)
        svc.memory_snapshot = mock.Mock(side_effect=lambda **kw: dict(self.snapshot))
        self.patch_time = mock.patch("serve.server.time.time", side_effect=lambda: self.wall)
        self.patch_mono = mock.patch("serve.server.time.monotonic", side_effect=lambda: self.mono)
        self.patch_time.start()
        self.patch_mono.start()
        self.advance()

    def tearDown(self):
        self.patch_mono.stop()
        self.patch_time.stop()

    def advance(self, *, ram=16, gpu=2048, pending=0, resident=None):
        self.wall += 1
        self.snapshot = dict(capacity(self.wall, ram=ram, gpu=gpu), native_free_mib=gpu,
                             native_sampled_at=self.wall)
        self.svc.engine.native_capacity = {"proc": self.svc.engine.proc, "sampled_at": self.wall,
            "resident_mib": resident if resident is not None else self.svc.engine.info["arena_mib"],
            "cache_mib": self.svc.engine.info["expert_cache_mib"], "free_mib": gpu, "total_mib": 8192,
            "memory_pending_id": pending, "memory_last_terminal_id": 0, "memory_reserve_mib": 800}

    def acquire(self, mode="relieve"):
        return self.svc.resource_lease_action(acquire_request(mode=mode))

    def terminal_ack(self, request_id, resident=32768, reserve=800):
        e = self.svc.engine
        e.memory_acks.put((e.proc, e._memory_ack(ack(request_id, resident=resident, reserve=reserve))))

    def test_inherited_growth_ack_must_be_sealed_before_readiness(self):
        svc = self.svc
        old = {"id": 9, "proc": svc.engine.proc, "plan": {"resident_budget_gib": 34,
               "vram_reserve_mib": 800, "reason": "stable_headroom"},
               "resident_before_gib": 32, "resource_generation": svc.resource_generation,
               "sent_at": self.wall, "progress_at": self.wall}
        svc.memory_live_pending = old
        self.advance(pending=9)
        row = self.acquire()
        self.assertIsNone(svc.memory_policy.lease_ceiling())
        self.assertEqual(svc.tool_leases.operation["operation_id"], 9)
        self.terminal_ack(9, resident=34816)
        self.advance(resident=34816)
        svc.observe_tool_lease()
        self.assertEqual(svc.memory_policy.lease_ceiling()["arena_gib"], 34)
        self.assertIsNone(svc.tool_leases.operation)
        self.advance(resident=34816)
        svc.observe_tool_lease()
        result = svc.tool_leases.action({"action": "status", "lease_token": row["lease_token"]})
        self.assertEqual(result["state"], "ready")

    def test_held_command_cannot_orphan_operation_during_new_pressure(self):
        svc = self.svc
        self.advance(ram=8, gpu=2048)
        self.acquire()
        svc.observe_tool_lease()
        self.advance(ram=8, gpu=2048)
        svc.observe_memory()
        first = svc.memory_live_pending
        self.assertIsNotNone(first)
        self.assertEqual(svc.tool_leases.operation["operation_id"], first["id"])
        written = svc.engine.proc.stdin.getvalue()
        self.assertTrue(written.rstrip().endswith(" hold"))
        self.advance(ram=2, gpu=200, pending=first["id"])
        svc.observe_memory()
        self.assertEqual(svc.memory_live_pending["id"], svc.tool_leases.operation["operation_id"])
        self.assertEqual(svc.tool_resident["operation"][0], svc.tool_leases.operation["operation_id"])
        # Deferring an unresolved owned command is safe. A later implementation
        # may replace it only with an atomically matching lease operation.
        self.assertEqual(svc.engine.proc.stdin.getvalue(), written)
        self.terminal_ack(first["id"], resident=28672, reserve=2048)
        self.advance(ram=16, gpu=2048, resident=28672)
        svc.observe_tool_lease()
        self.assertIsNone(svc.tool_leases.operation)

    def test_settled_terminal_resident_lease_does_not_disable_normal_policy(self):
        svc = self.svc
        row = self.acquire()
        svc.observe_tool_lease()
        svc.resource_lease_action({"action": "release", "lease_token": row["lease_token"]})
        self.assertFalse(svc.tool_leases.blocked())
        self.assertIsNone(svc.memory_policy.lease_ceiling())
        with mock.patch.object(svc.memory_policy, "observe", wraps=svc.memory_policy.observe) as observe:
            self.advance()
            svc.observe_memory()
            observe.assert_called_once()

    def test_settled_legacy_unload_lease_does_not_disable_normal_policy(self):
        svc = self.svc
        row = self.acquire(mode="unload")
        svc.resource_lease_action({"action": "release", "lease_token": row["lease_token"]})
        with mock.patch.object(svc.memory_policy, "observe", wraps=svc.memory_policy.observe) as observe:
            self.advance()
            svc.observe_memory()
            observe.assert_called_once()

    def test_fresh_capacity_taken_inside_call_is_not_rejected_as_future(self):
        svc = self.svc
        row = self.acquire()
        def fresh(**kwargs):
            self.advance()
            return dict(self.snapshot)
        svc.memory_snapshot.side_effect = fresh
        for _ in range(3):
            svc.observe_tool_lease()
        result = svc.tool_leases.action({"action": "status", "lease_token": row["lease_token"]})
        self.assertEqual(result["state"], "ready")

    def test_former_ready_is_revoked_when_native_capacity_disappears(self):
        svc = self.svc
        row = self.acquire()
        svc.observe_tool_lease()
        ready = svc.resource_lease_action({"action": "status", "lease_token": row["lease_token"]})
        self.assertEqual(ready["state"], "ready")
        self.snapshot["native_free_mib"] = None
        svc.engine.native_capacity = None
        with self.assertRaises(LeaseError):
            svc.resource_lease_action({"action": "start", "lease_token": row["lease_token"]})
        self.assertNotEqual(svc.tool_leases.current["state"], "ready")
        self.assertEqual(svc.tool_leases.current["phase"], "admission")

    def test_strict_missing_native_hold_capability_cannot_grant_or_unload(self):
        svc = self.svc
        svc.engine.info.pop("memory_hold")
        row = self.acquire()
        with mock.patch.object(svc, "_parking_unload") as unload:
            svc.observe_tool_lease()
            unload.assert_not_called()
        result = svc.resource_lease_action({"action": "status", "lease_token": row["lease_token"]})
        self.assertEqual((result["state"], result["reason"]), ("failed", "resident_relief_unsupported"))
        self.assertIsNone(svc.memory_policy.lease_ceiling())

    def test_short_tool_return_requires_native_sample_after_release(self):
        svc = self.svc
        row = self.acquire()
        svc.observe_tool_lease()
        svc.resource_lease_action({"action": "start", "lease_token": row["lease_token"]})
        # Both readings remain less than five seconds old, but the native
        # reading precedes the tool's release and cannot certify its aftermath.
        self.wall += .5
        svc.resource_lease_action({"action": "release", "lease_token": row["lease_token"]})
        self.snapshot["sampled_at"] = self.wall
        self.mono = 10
        wait = svc._admit_resident_return(threading.Event())
        self.assertEqual(next(wait), ("ping", None))
        self.assertIsNotNone(svc.tool_resume)
        self.advance()
        with self.assertRaises(StopIteration):
            next(wait)
        self.assertIsNone(svc.tool_resume)
        self.assertFalse(svc.tool_resuming)

    def test_release_with_pending_control_keeps_barrier_until_matching_ack(self):
        svc = self.svc
        self.advance(ram=8)
        row = self.acquire()
        svc.observe_tool_lease()
        self.advance(ram=8)
        svc.observe_memory()
        pending = svc.memory_live_pending
        self.assertIsNotNone(pending)
        svc.resource_lease_action({"action": "release", "lease_token": row["lease_token"]})
        self.assertTrue(svc.tool_leases.blocked())
        self.assertIsNotNone(svc.memory_policy.lease_ceiling())
        with self.assertRaises(LeaseError):
            self.acquire()
        self.terminal_ack(pending["id"] + 1)
        svc.observe_tool_lease()
        self.assertEqual(svc.memory_live_pending["id"], pending["id"])
        self.assertTrue(svc.tool_leases.blocked())
        self.terminal_ack(pending["id"], resident=28672)
        self.advance(resident=28672)
        svc.observe_tool_lease()
        self.assertFalse(svc.tool_leases.blocked())
        self.assertIsNone(svc.memory_policy.lease_ceiling())
        self.assertIsNone(svc.tool_resident)

    def test_expired_execution_keeps_owner_ceiling_until_confirmed_exit(self):
        svc = self.svc
        row = self.acquire()
        svc.observe_tool_lease()
        svc.resource_lease_action({"action": "start", "lease_token": row["lease_token"]})
        self.mono = 31
        svc.observe_tool_lease()
        self.assertTrue(svc.tool_leases.blocked())
        self.assertIsNotNone(svc.memory_policy.lease_ceiling())
        with self.assertRaises(LeaseError):
            self.acquire()
        svc.resource_lease_action({"action": "release", "lease_token": row["lease_token"]})
        self.assertFalse(svc.tool_leases.blocked())
        self.assertIsNone(svc.memory_policy.lease_ceiling())

    def test_long_healthy_execution_gets_relief_window_for_new_pressure(self):
        svc = self.svc
        row = svc.resource_lease_action(acquire_request(mode="relieve",
            execution_ram_floor_gib=4, execution_vram_floor_mib=320))
        svc.observe_tool_lease()
        svc.resource_lease_action({"action": "start", "lease_token": row["lease_token"]})
        self.mono = 20
        svc.resource_lease_action({"action": "renew", "lease_token": row["lease_token"], "ttl_seconds": 30})
        self.mono = 35
        self.advance(ram=3.5)
        svc.observe_tool_lease()
        # A newly observed half-GiB deficit during a renewed long compile can
        # be relieved by its 32-GiB resident cache. Healthy elapsed execution
        # must not consume the deadline for a pressure episode that just began.
        self.assertFalse(svc.tool_leases.demand()["terminal"])
        svc.observe_memory()
        self.assertIsNotNone(svc.memory_live_pending)
        self.assertEqual(svc.tool_leases.operation["action"], "relieve")

    def test_inherited_ack_does_not_seal_pre_ack_residency_sample(self):
        svc = self.svc
        svc.memory_live_pending = {"id": 9, "proc": svc.engine.proc,
            "plan": {"resident_budget_gib": 34, "vram_reserve_mib": 800, "reason": "stable_headroom"},
            "resident_before_gib": 32, "resource_generation": svc.resource_generation,
            "sent_at": self.wall, "progress_at": self.wall}
        self.advance(pending=9)
        self.acquire()
        # The ACK can reach the service before the next native CAPACITY line.
        # A prior idle CAPACITY must not become the new acknowledged ceiling.
        svc.engine.native_capacity["memory_pending_id"] = 0
        self.wall += .5
        self.snapshot["sampled_at"] = self.wall
        self.terminal_ack(9, resident=34816)
        svc.observe_tool_lease()
        self.assertEqual(svc.memory_policy.current["resident_budget_gib"], 34)
        self.assertIsNone(svc.memory_policy.lease_ceiling())
        self.advance(resident=34816)
        svc.observe_tool_lease()
        self.assertEqual(svc.memory_policy.lease_ceiling()["arena_gib"], 34)

    def test_idle_pressure_rechecks_lease_after_fifo_before_unload_commit(self):
        svc = self.svc
        svc.idle_parking = SimpleNamespace(enabled=True, pressure=lambda *args: (True, "ram_pressure"),
            admission=lambda footprint: SimpleNamespace(required={"gpu_bytes": 0}))
        real_fifo = svc.fifo
        def acquire_then_lease(**kwargs):
            acquired = real_fifo.acquire(**kwargs)
            if acquired:
                self.acquire()
            return acquired
        svc.fifo = SimpleNamespace(acquire=acquire_then_lease, release=real_fifo.release)
        with mock.patch.object(svc, "_parking_footprint", return_value={}), \
                mock.patch.object(svc, "_parking_identity", return_value={}), \
                mock.patch.object(svc, "_parking_unload") as unload:
            self.assertFalse(svc.observe_idle_pressure())
            unload.assert_not_called()
        self.assertTrue(svc.tool_leases.blocked())
        self.assertIsNone(svc.idle_parked)

    def test_continuous_pressure_timeout_is_not_extended_by_status_polls(self):
        svc = self.svc
        svc.resource_lease_action(acquire_request(mode="relieve", ttl_seconds=120))
        svc.observe_tool_lease()
        for moment in (5, 20, 34.999):
            self.mono = moment
            self.advance(ram=11)
            svc.observe_tool_lease()
            self.assertEqual(svc.tool_resident["pressure_since"], 5)
            self.assertFalse(svc.tool_leases.demand()["terminal"])
        self.mono = 35
        self.advance(ram=11)
        svc.observe_tool_lease()
        self.assertEqual(svc.tool_leases.demand()["reason"], "live_relief_insufficient")
        self.assertTrue(svc.tool_leases.demand()["terminal"])

    def test_verified_healthy_capacity_resets_only_the_pressure_episode(self):
        svc = self.svc
        svc.resource_lease_action(acquire_request(mode="relieve", ttl_seconds=120))
        svc.observe_tool_lease()
        self.mono = 5
        self.advance(ram=11)
        svc.observe_tool_lease()
        self.assertEqual(svc.tool_resident["pressure_since"], 5)
        self.mono = 20
        self.advance(ram=16)
        svc.observe_tool_lease()
        self.assertIsNone(svc.tool_resident["pressure_since"])
        for moment in (60, 89.999):
            self.mono = moment
            self.advance(ram=11)
            svc.observe_tool_lease()
            self.assertEqual(svc.tool_resident["pressure_since"], 60)
            self.assertFalse(svc.tool_leases.demand()["terminal"])

    def test_unresolved_control_timeout_keeps_identity_and_ceiling(self):
        svc = self.svc
        self.advance(ram=8)
        svc.resource_lease_action(acquire_request(mode="relieve", ttl_seconds=120))
        svc.observe_tool_lease()
        svc.observe_memory()
        pending = svc.memory_live_pending
        self.assertIsNotNone(pending)
        for moment in (35, 50, 64.999):
            self.mono = moment
            self.advance(ram=8, pending=pending["id"])
            svc.observe_tool_lease()
            self.assertEqual(svc.tool_resident["control_wait_since"], 35)
            self.assertFalse(svc.tool_leases.demand()["terminal"])
        self.mono = 65
        svc.observe_tool_lease()
        self.assertEqual(svc.tool_leases.demand()["reason"], "live_relief_deadline")
        self.assertTrue(svc.tool_leases.blocked())
        self.assertEqual(svc.memory_live_pending["id"], pending["id"])
        self.assertEqual(svc.tool_leases.operation["operation_id"], pending["id"])
        self.assertIsNotNone(svc.memory_policy.lease_ceiling())


if __name__ == "__main__":
    unittest.main()
