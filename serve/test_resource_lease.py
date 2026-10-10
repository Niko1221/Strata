"""Owner/expiry/idempotency tests without model processes or hardware."""
import json
import unittest
import uuid
from unittest import mock

from serve.resource_lease import ToolLeases, LeaseError

GIB, MIB = 2**30, 2**20


def capacity(now=100, ram=40, gpu=7000, commit=40, commit_required=True):
    return {"sampled_at": now, "ram_total": 64 * GIB, "ram_used": (64 - ram) * GIB,
            "gpu_mem_total": 8192 * MIB, "gpu_mem_used": (8192 - gpu) * MIB,
            "ram_commit_available": commit * GIB, "ram_commit_required": commit_required}


def acquire_request(**changes):
    return {"action": "acquire", "request_id": str(uuid.uuid4()), "mode": "unload",
            "ram_headroom_gib": 12, "vram_headroom_mib": 1000, "ttl_seconds": 30, **changes}


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        with mock.patch.dict("os.environ", {"STRATA_RESOURCE_LEASE_TOKEN": "test-control-" * 4}):
            self.lease = ToolLeases({"enabled": True}, clock=lambda: self.now)

    def acquire(self, **changes):
        return self.lease.acquire(acquire_request(**changes), capacity(), 100)

    def act(self, row, action, **changes):
        return self.lease.action({"action": action, "lease_token": row["lease_token"], **changes})

    def test_disabled_and_secret_validation(self):
        self.assertFalse(ToolLeases().enabled)
        for cfg in ({"enabled": 1}, {"enabled": True, "token_env": "NONEXISTENT_LEASE_TOKEN"},
                    {"token": "secret"}, {"max_ttl_seconds": 301}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                ToolLeases(cfg)
        self.assertTrue(self.lease.authenticated("test-control-" * 4))
        self.assertFalse(self.lease.authenticated("wrong"))

    def test_idempotent_retry_binds_parameters_and_does_not_renew(self):
        req = acquire_request()
        first = self.lease.acquire(req, capacity(), 100)
        self.now = 10
        retry = self.lease.acquire(req, {}, 100)
        self.assertEqual(first["lease_token"], retry["lease_token"])
        self.assertEqual(retry["expires_in_seconds"], 20)
        with self.assertRaises(LeaseError) as cm:
            self.lease.acquire({**req, "ram_headroom_gib": 20}, capacity(), 100)
        self.assertEqual(cm.exception.status, 409)
        with self.assertRaises(LeaseError):
            self.acquire()

    def test_stale_owner_cannot_release_successor(self):
        first = self.acquire()
        self.now = 31
        self.assertFalse(self.lease.blocked())
        second = self.acquire()
        self.act(first, "release")
        self.assertTrue(self.lease.blocked())
        self.assertEqual(self.act(second, "status")["state"], "pending")
        with self.assertRaises(LeaseError):
            self.act(first, "renew", ttl_seconds=30)

    def test_expiry_and_release_during_unload_keep_barrier_until_teardown_finishes(self):
        for action in ("expired", "released"):
            with self.subTest(action=action):
                self.setUp()
                row = self.acquire()
                owned = self.lease.begin_unload()
                if action == "expired":
                    self.now = 31
                else:
                    self.act(row, "release")
                self.assertTrue(self.lease.blocked())
                self.assertEqual(self.act(row, "status")["state"], "unloading")
                self.lease.unloaded(owned, 101)
                self.assertFalse(self.lease.blocked())
                self.assertEqual(self.act(row, "status")["state"], action)

    def test_expired_pending_lease_never_begins_unload(self):
        self.acquire()
        self.now = 31
        self.assertIsNone(self.lease.begin_unload())

    def test_ready_needs_full_unload_fresh_post_exit_capacity_and_all_floors(self):
        row = self.acquire()
        owned = self.lease.begin_unload()
        self.lease.unloaded(owned, 101)
        def observe(reading, unloaded=True):
            self.lease.observe_ready(reading, 102, unloaded=unloaded, ram_floor=3, gpu_floor=320, commit_floor=3)
            return self.act(row, "status")["state"]
        for reading in (capacity(100), capacity(103), {}, capacity(102, ram=11),
                        capacity(102, gpu=999), capacity(102, commit=2)):
            self.assertEqual(observe(reading), "pending")
        self.assertEqual(observe(capacity(102), False), "pending")
        self.assertEqual(observe(capacity(102)), "ready")
        self.assertEqual(observe(capacity(102, ram=1)), "pending")

    def test_capacity_validation_and_public_status_do_not_expose_owner(self):
        for changes in ({"ram_headroom_gib": 100}, {"vram_headroom_mib": 9000},
                        {"ram_headroom_gib": float("nan")}, {"ttl_seconds": True}, {"mode": "quiet"}):
            with self.subTest(changes=changes), self.assertRaises(LeaseError):
                self.acquire(**changes)
        with self.assertRaises(LeaseError):
            self.lease.acquire(acquire_request(), capacity(90), 100)
        row = self.acquire()
        public = json.dumps(self.lease.public())
        self.assertNotIn(row["lease_token"], public)
        self.assertNotIn(row["lease_id"], public)
        self.assertNotIn("test-control", public)
        status = self.act(row, "status")
        self.assertNotIn("lease_token", status)
        self.assertFalse(status["availability_is_reservation"])

    def test_optional_commit_sensor_does_not_block_otherwise_ready_lease(self):
        row = self.acquire()
        owned = self.lease.begin_unload()
        self.lease.unloaded(owned, 101)
        for reading in (capacity(102, commit=0, commit_required=False),
                        {k: v for k, v in capacity(102, commit_required=False).items()
                         if k != "ram_commit_available"}):
            with self.subTest(reading=reading):
                self.lease.observe_ready(reading, 102, unloaded=True, ram_floor=3,
                                         gpu_floor=320, commit_floor=3)
                self.assertEqual(self.act(row, "status")["state"], "ready")


class ResidentLeaseTests(unittest.TestCase):
    """C1 contract checks without pretending a fake ACK releases real memory."""

    setUp, acquire, act = LeaseTests.setUp, LeaseTests.acquire, LeaseTests.act

    def resident(self, **changes):
        row = self.acquire(mode="auto", **changes)
        self.assertTrue(self.lease.select_action(row["lease_id"], "none", "sufficient_headroom"))
        self.assertTrue(self.lease.seal_residency(row["lease_id"], 42, 32768, 264))
        return row

    def observe(self, row, reading=None, **changes):
        if reading is None:
            reading = {**capacity(102), "native_sampled_at": 102, "native_free_mib": 1200}
        self.lease.observe_resident_ready(reading, 102, lease_id=row["lease_id"], engine_identity=42,
                                          drained_at=101, ram_floor=3, gpu_floor=250, commit_floor=3, **changes)
        return self.act(row, "status")

    def test_start_changes_only_phase_targets_without_renewing_or_increasing_ceilings(self):
        row = self.resident(execution_ram_floor_gib=3, execution_vram_floor_mib=250)
        with self.assertRaises(LeaseError):
            self.act(row, "start")
        self.assertEqual(self.observe(row)["phase"], "admission")
        self.now = 1
        started = self.act(row, "start")
        self.assertEqual((started["state"], started["phase"], started["selected_action"]),
                         ("ready", "execution", "none"))
        self.assertEqual(started["expires_in_seconds"], 29)
        demand = self.lease.demand()
        self.assertEqual((demand["ram_target_gib"], demand["vram_target_mib"]), (3, 250))
        self.assertEqual(demand["residency"]["ram_cap_mib"], 32768)
        # Idempotent owner start does not extend authority or repeat admission.
        self.now = 2
        self.assertEqual(self.act(row, "start")["expires_in_seconds"], 28)
        # The real tool consuming its admitted working set must not reimpose 12GiB/1000MiB.
        reading = {**capacity(102, ram=4), "native_sampled_at": 102, "native_free_mib": 300}
        self.assertEqual(self.observe(row, reading)["state"], "ready")

    def test_unspecified_execution_floor_never_infers_incremental_tool_demand(self):
        row = self.resident()
        self.observe(row)
        self.act(row, "start")
        demand = self.lease.demand()
        self.assertEqual((demand["ram_target_gib"], demand["vram_target_mib"]), (12, 1000))
        reading = {**capacity(102, ram=4), "native_sampled_at": 102, "native_free_mib": 300}
        self.assertEqual(self.observe(row, reading)["state"], "pending")

    def test_execution_floors_are_validated_and_bound_to_idempotency(self):
        for changes in ({"execution_ram_floor_gib": 13}, {"execution_vram_floor_mib": 1001},
                        {"execution_ram_floor_gib": -1}, {"execution_vram_floor_mib": True},
                        {"execution_ram_floor_gib": float("inf")}, {"execution_ram_floor_gib": 10**1000}):
            with self.subTest(changes=changes), self.assertRaises(LeaseError):
                self.acquire(mode="auto", **changes)
        req = acquire_request(mode="auto", execution_ram_floor_gib=3)
        original = self.lease.acquire(req, capacity(), 100)
        self.assertEqual(self.lease.acquire(req, {}, 100)["lease_token"], original["lease_token"])
        with self.assertRaises(LeaseError):
            self.lease.acquire({**req, "execution_ram_floor_gib": 4}, capacity(), 100)

    def test_expired_execution_stays_blocked_until_owner_confirms_tool_cleanup(self):
        for mode in ("auto", "relieve"):
            with self.subTest(mode=mode):
                self.setUp()
                row = self.acquire(mode=mode)
                self.lease.select_action(row["lease_id"], "none", "sufficient")
                self.lease.seal_residency(row["lease_id"], 42, 32768, 264)
                self.observe(row)
                self.act(row, "start")
                self.now = 31
                self.assertEqual(self.act(row, "status")["state"], "expired")
                self.assertTrue(self.lease.blocked())
                self.assertEqual(self.lease.public()["barrier_reason"], "tool_exit_unconfirmed")
                with self.assertRaises(LeaseError):
                    self.acquire()
                for action, args in (("start", {}), ("renew", {"ttl_seconds": 30})):
                    with self.assertRaises(LeaseError):
                        self.act(row, action, **args)
                self.act(row, "release")
                self.assertFalse(self.lease.blocked())
                successor = self.acquire()
                self.act(row, "release")
                self.assertTrue(self.lease.blocked())
                self.assertEqual(self.act(successor, "status")["state"], "pending")

    def test_legacy_unload_with_start_keeps_legacy_expiration_semantics(self):
        row = self.acquire(execution_ram_floor_gib=3, execution_vram_floor_mib=250)
        self.lease.unloaded(self.lease.begin_unload(), 101)
        self.lease.observe_ready(capacity(102), 102, unloaded=True, ram_floor=3, gpu_floor=250, commit_floor=3)
        self.act(row, "start")
        self.lease.observe_ready(capacity(102, ram=4, gpu=300), 102, unloaded=True,
                                 ram_floor=3, gpu_floor=250, commit_floor=3)
        self.assertEqual(self.act(row, "status")["state"], "ready")
        self.now = 31
        self.assertFalse(self.lease.blocked())

    def test_native_operations_survive_release_expiry_failure_and_stale_ack(self):
        for terminal in ("release", "expire", "fail"):
            with self.subTest(terminal=terminal):
                self.setUp()
                row = self.resident()
                owner = row["lease_id"]
                self.assertTrue(self.lease.begin_operation(owner, "relieve", 7, 42))
                self.assertTrue(self.lease.begin_operation(owner, "relieve", 7, 42))
                self.assertFalse(self.lease.begin_operation(owner, "relieve", 8, 42))
                if terminal == "release":
                    self.act(row, "release")
                elif terminal == "expire":
                    self.now = 31
                else:
                    self.lease.fail("operation_deadline")
                self.assertTrue(self.lease.blocked())
                self.assertTrue(self.lease.demand()["terminal"])
                self.assertEqual(self.lease.demand()["residency"]["gpu_cap_mib"], 264)
                for lease_id, op, process in (("other", 7, 42), (owner, 8, 42), (owner, 7, 43)):
                    self.assertFalse(self.lease.complete_operation(lease_id, op, process))
                with self.assertRaises(LeaseError):
                    self.acquire()
                self.assertTrue(self.lease.complete_operation(owner, 7, 42))
                self.assertFalse(self.lease.blocked())
                successor = self.resident()
                self.assertFalse(self.lease.complete_operation(owner, 7, 42))
                self.assertEqual(self.lease.demand()["lease_id"], successor["lease_id"])

    def test_pending_growth_drain_blocks_ready_even_if_capacity_is_sufficient(self):
        row = self.resident()
        self.assertTrue(self.lease.begin_operation(row["lease_id"], "drain", "previous-growth", 42))
        self.assertEqual(self.observe(row)["state"], "pending")
        self.assertFalse(self.lease.select_action(row["lease_id"], "none", "unsafe_pending_growth"))
        self.lease.complete_operation(row["lease_id"], "previous-growth", 42)
        self.assertEqual(self.observe(row)["state"], "ready")

    def test_ceiling_cannot_increase_or_switch_process_and_snapshots_are_not_mutable(self):
        row = self.resident()
        self.assertTrue(self.lease.seal_residency(row["lease_id"], 42, 16384, 128))
        self.assertTrue(self.lease.seal_residency(row["lease_id"], 42, 65536, 1024))
        self.assertFalse(self.lease.seal_residency(row["lease_id"], 43, 100, 100))
        self.assertFalse(self.lease.seal_residency("old-owner", 42, 100, 100))
        demand = self.lease.demand()
        self.assertEqual(demand["residency"], {"engine_identity": 42, "ram_cap_mib": 16384, "gpu_cap_mib": 128})
        demand["residency"]["gpu_cap_mib"] = 9999
        self.assertEqual(self.lease.demand()["residency"]["gpu_cap_mib"], 128)
        self.lease.begin_operation(row["lease_id"], "drain", 7, 42)
        demand = self.lease.demand()
        demand["operation"]["operation_id"] = 99
        self.assertEqual(self.lease.demand()["operation"]["operation_id"], 7)

    def test_resident_readiness_requires_drain_both_fresh_samples_and_native_capacity(self):
        row = self.resident()
        base = {**capacity(102), "native_sampled_at": 102, "native_free_mib": 1200}
        for reading in ({}, {**base, "sampled_at": 100}, {**base, "native_sampled_at": 100},
                        {**base, "native_sampled_at": 103}, {**base, "native_free_mib": None},
                        {**base, "native_free_mib": 0}, {**base, "native_free_mib": -1},
                        {**base, "native_free_mib": float("nan")}, {**base, "ram_used": 63 * GIB},
                        {**base, "ram_commit_available": 2 * GIB}, {**base, "ram_commit_required": "false"}):
            with self.subTest(reading=reading):
                self.assertEqual(self.observe(row, reading)["state"], "pending")
        self.assertEqual(self.observe(row, base)["state"], "ready")
        self.lease.observe_resident_ready(base, 102, lease_id=row["lease_id"], engine_identity=43,
                                          drained_at=101, ram_floor=3, gpu_floor=250, commit_floor=3)
        self.assertEqual(self.act(row, "status")["state"], "pending")
        self.observe(row, base)
        self.now = 6
        with self.assertRaises(LeaseError):
            self.act(row, "start")

    def test_resident_ready_requires_selected_action_and_sealed_identity(self):
        row = self.acquire(mode="auto")
        self.assertEqual(self.observe(row)["state"], "pending")
        self.lease.select_action(row["lease_id"], "none", "sufficient")
        self.assertEqual(self.observe(row)["state"], "pending")
        self.lease.seal_residency(row["lease_id"], 42, 32768, 264)
        self.assertEqual(self.observe(row)["state"], "ready")

    def test_strict_relief_rejects_unload_and_auto_can_escalate(self):
        row = self.acquire(mode="relieve")
        with self.assertRaises(LeaseError):
            self.lease.select_action(row["lease_id"], "unload", "unreachable")
        with self.assertRaises(LeaseError):
            self.lease.begin_operation(row["lease_id"], "unload", 8, 42)
        self.assertFalse(self.lease.needs_unload())
        self.assertIsNone(self.lease.begin_unload())
        self.lease.fail("strict_relief_unreachable")
        self.assertEqual(self.act(row, "status")["state"], "failed")
        row = self.acquire(mode="auto")
        self.assertFalse(self.lease.needs_unload())
        self.lease.select_action(row["lease_id"], "unload", "relief_insufficient")
        self.assertTrue(self.lease.needs_unload())
        self.lease.unloaded(self.lease.begin_unload(), 101)
        self.lease.observe_ready(capacity(102), 102, unloaded=True, ram_floor=3, gpu_floor=250, commit_floor=3)
        self.assertEqual(self.act(row, "status")["selected_action"], "unload")
        self.assertEqual(self.act(row, "status")["state"], "ready")

    def test_failed_teardown_stays_blocked_until_confirmed_death_without_resurrection(self):
        row = self.acquire()
        owner = self.lease.begin_unload()
        self.lease.fail("uncertain_process_death")
        self.assertTrue(self.lease.blocked())
        self.assertEqual(self.act(row, "status")["state"], "failed")
        self.lease.unloaded(owner, 101)
        self.assertFalse(self.lease.blocked())
        self.assertEqual(self.act(row, "status")["state"], "failed")

    def test_auto_teardown_supersedes_pending_work_and_rejects_old_ack(self):
        row = self.resident()
        owner = row["lease_id"]
        self.lease.begin_operation(owner, "relieve", 7, 42)
        self.assertTrue(self.lease.select_action(owner, "unload", "native_operation_deadline"))
        self.assertTrue(self.lease.needs_unload())
        self.assertEqual(self.lease.begin_unload(), owner)
        self.assertFalse(self.lease.complete_operation(owner, 7, 42))
        self.assertTrue(self.lease.blocked())
        self.assertEqual(self.act(row, "status")["state"], "unloading")
        self.act(row, "release")
        self.lease.unloaded(owner, 101)
        self.assertFalse(self.lease.blocked())

    def test_auto_execution_may_escalate_without_changing_phase_or_releasing_tool_hold(self):
        row = self.resident(execution_ram_floor_gib=3, execution_vram_floor_mib=250)
        self.observe(row)
        self.act(row, "start")
        self.assertTrue(self.lease.select_action(row["lease_id"], "unload", "external_pressure"))
        self.lease.unloaded(self.lease.begin_unload(), 101)
        self.lease.observe_ready(capacity(102, ram=4, gpu=300, commit=4), 102,
                                 unloaded=True, ram_floor=3, gpu_floor=250, commit_floor=3)
        status = self.act(row, "status")
        self.assertEqual((status["state"], status["phase"], status["selected_action"]),
                         ("ready", "execution", "unload"))
        self.assertTrue(status["execution_hold"])
        self.now = 31
        self.assertTrue(self.lease.blocked())
        self.act(row, "release")
        self.assertFalse(self.lease.blocked())

    def test_commit_and_physical_must_each_meet_the_active_phase_target(self):
        row = self.resident(execution_ram_floor_gib=3)
        reading = {**capacity(102, ram=40, commit=4), "native_sampled_at": 102, "native_free_mib": 1200}
        self.assertEqual(self.observe(row, reading)["state"], "pending")
        self.observe(row)
        self.act(row, "start")
        self.assertEqual(self.observe(row, reading)["state"], "ready")

    def test_resume_allowances_are_explicit_bounded_config_not_request_fields(self):
        self.assertEqual((self.lease.resume_ram_working_gib, self.lease.resume_vram_working_mib,
                          self.lease.resume_timeout_seconds), (2, 256, 30))
        for cfg in ({"resume_ram_working_gib": -1}, {"resume_vram_working_mib": True},
                    {"resume_timeout_seconds": 0}, {"resume_timeout_seconds": 301},
                    {"resume_ram_working_gib": float("nan")}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                ToolLeases(cfg)
        with self.assertRaises(LeaseError):
            self.acquire(resume_ram_working_gib=0)


if __name__ == "__main__":
    unittest.main()
