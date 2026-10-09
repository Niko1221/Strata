"""Owner/expiry/idempotency tests without model processes or hardware."""
import json
import unittest
import uuid
from unittest import mock

from serve.resource_lease import ToolLeases, LeaseError

GIB, MIB = 2**30, 2**20


def capacity(now=100, ram=40, gpu=7000, commit=40):
    return {"sampled_at": now, "ram_total": 64 * GIB, "ram_used": (64 - ram) * GIB,
            "gpu_mem_total": 8192 * MIB, "gpu_mem_used": (8192 - gpu) * MIB,
            "ram_commit_available": commit * GIB}


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


if __name__ == "__main__":
    unittest.main()
