"""Owner-bound cache ceilings for resident foreground leases; no hardware needed."""
import unittest

from serve.memory_policy import GIB, MIB, MemoryPolicy


class LeaseCeilingTests(unittest.TestCase):
    def policy(self):
        p = MemoryPolicy({"enabled": True, "mode": "live", "recovery_seconds": 30},
                         resident_cap_gib=42, vram_reserve_mib=800)
        p.update_resource_limits(4, 320)
        p.reclaim_gpu_headroom = True
        p.live_actual(32 * 1024, 800, 0, "loaded", completed=True, loaded=True)
        return p

    def observe(self, p, now, free_ram=16, free_gpu=2048, arena=None):
        snapshot = {"sampled_at": now, "ram_total": 64 * GIB,
                    "ram_used": (64 - free_ram) * GIB, "gpu_mem_total": 8192 * MIB,
                    "gpu_mem_used": (8192 - free_gpu) * MIB}
        return p.observe(snapshot, True, {"arena_mib": (arena if arena is not None else
                         p.current["resident_budget_gib"]) * 1024, "expert_cache_mib": 1024}, now)

    def test_abundant_headroom_cannot_grow_either_cache_through_long_lease(self):
        p = self.policy()
        self.assertTrue(p.update_lease_ceiling("owner", 32, 1024))
        for now in range(1, 2001):
            self.assertIsNone(self.observe(p, now))
        self.assertEqual(p.last_reason, "lease_ceiling_active")
        self.assertIsNone(p.growth_since)
        self.assertIsNone(p.gpu_growth_since)
        self.assertEqual(p.current["resident_budget_gib"], 32)

    def test_pressure_shrinks_ram_without_opportunistic_gpu_target_growth(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        plan = self.observe(p, 1, free_ram=2, free_gpu=1234.1)
        self.assertEqual(plan, {"resident_budget_gib": 30, "vram_reserve_mib": 1235,
                                "reason": "sustained_pressure"})
        p.live_actual(30 * 1024, 1235, 1, "sustained_pressure", completed=True)
        for now in range(2, 1002):
            self.assertIsNone(self.observe(p, now))
        self.assertEqual(p.current["resident_budget_gib"], 30)

    def test_gpu_pressure_still_reclaims_without_ram_growth(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        plan = self.observe(p, 1, free_gpu=200)
        self.assertEqual(plan, {"resident_budget_gib": 32, "vram_reserve_mib": 320,
                                "reason": "sustained_pressure"})

    def test_shrink_to_zero_remains_reclaimable_and_does_not_regrow(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        p.live_actual(1024, 800, 1, "sustained_pressure", completed=True)
        plan = self.observe(p, 2, free_ram=1)
        self.assertEqual(plan["resident_budget_gib"], 0)
        p.live_actual(0, plan["vram_reserve_mib"], 2, "sustained_pressure", completed=True)
        for now in range(3, 100):
            self.assertIsNone(self.observe(p, now))

    def test_same_owner_refresh_only_tightens_and_never_resets_pressure(self):
        p = self.policy()
        p.pressure_since = 10
        self.assertTrue(p.update_lease_ceiling("owner", 32, 1024))
        self.assertEqual(p.pressure_since, 10)
        self.assertFalse(p.update_lease_ceiling("owner", 34, 2048))
        self.assertEqual(p.lease_ceiling(), {"arena_gib": 32, "gpu_cache_mib": 1024})
        self.assertTrue(p.update_lease_ceiling("owner", 30, 900))
        self.assertEqual(p.lease_ceiling(), {"arena_gib": 30, "gpu_cache_mib": 900})
        self.assertEqual(p.pressure_since, 10)

    def test_other_owner_cannot_modify_or_clear(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        before = p.status()
        with self.assertRaises(ValueError):
            p.update_lease_ceiling("other", 20, 512)
        with self.assertRaises(ValueError):
            p.clear_lease_ceiling("other")
        self.assertEqual(p.status(), before)

    def test_clear_resets_recovery_once_then_requires_full_fresh_dwell(self):
        p = self.policy()
        for now in range(1, 30):
            self.assertIsNone(self.observe(p, now))
        p.recovery_since = 1
        p.pressure_since = 7
        p.update_lease_ceiling("owner", 32, 1024)
        self.assertIsNone(p.recovery_since)
        self.assertEqual(p.pressure_since, 7)
        self.assertTrue(p.clear_lease_ceiling("owner"))
        for now in range(30, 60):
            self.assertIsNone(self.observe(p, now))
            self.assertFalse(p.clear_lease_ceiling("owner"))
        self.assertEqual(p.growth_since, 30)
        self.assertEqual(self.observe(p, 60)["reason"], "stable_headroom")

    def test_after_shrink_release_grows_from_actual_not_old_capture(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        p.live_actual(28 * 1024, 2048, 1, "sustained_pressure", completed=True)
        p.clear_lease_ceiling("owner")
        for now in range(2, 32):
            self.assertIsNone(self.observe(p, now))
        self.assertEqual(self.observe(p, 32)["resident_budget_gib"], 30)

    def test_invalid_values_are_atomic(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        for owner, arena, gpu in (("", 32, 1024), (None, 32, 1024), (True, 32, 1024),
                                  ("owner", -1, 1024), ("owner", 43, 1024),
                                  ("owner", float("nan"), 1024), ("owner", True, 1024),
                                  ("owner", 32, -1), ("owner", 32, float("inf")),
                                  ("owner", 32, False)):
            with self.subTest(owner=owner, arena=arena, gpu=gpu):
                before = p.status()
                with self.assertRaises(ValueError):
                    p.update_lease_ceiling(owner, arena, gpu)
                self.assertEqual(p.status(), before)

    def test_only_coadaptive_live_loaded_mode_accepts_ceiling(self):
        for mode, enabled, fixed, reclaim, loaded in (
                ("live", False, True, True, True), ("reload", True, True, True, True),
                ("live", True, False, True, True), ("live", True, True, False, True),
                ("live", True, True, True, False)):
            p = MemoryPolicy({"enabled": enabled, "mode": mode})
            if fixed:
                p.update_resource_limits(4, 320)
            p.reclaim_gpu_headroom = reclaim
            if loaded:
                p.live_actual(32 * 1024, 800, 0, "loaded", completed=True, loaded=True)
            with self.subTest(mode=mode, enabled=enabled, fixed=fixed, reclaim=reclaim, loaded=loaded):
                before = p.status()
                with self.assertRaises(ValueError):
                    p.update_lease_ceiling("owner", 32, 1024)
                self.assertEqual(p.status(), before)

    def test_status_and_getter_do_not_expose_owner_or_mutable_state(self):
        p = self.policy()
        self.assertIsNone(p.lease_ceiling())
        self.assertNotIn("lease_ceiling", p.status())
        p.update_lease_ceiling("private-owner", 32, 1024)
        self.assertNotIn("private-owner", str(p.status()))
        p.lease_ceiling()["arena_gib"] = 100
        p.status()["lease_ceiling"]["gpu_cache_mib"] = 100
        self.assertEqual(p.lease_ceiling(), {"arena_gib": 32, "gpu_cache_mib": 1024})

    def test_stale_unknown_and_load_cannot_remove_ceiling(self):
        p = self.policy()
        p.update_lease_ceiling("owner", 32, 1024)
        self.assertIsNone(p.observe({}, True, {}, 100))
        self.assertIsNone(p.plan_for_load({}, 100))
        self.assertEqual(p.last_reason, "lease_ceiling_active")
        p.live_actual(30 * 1024, 800, 101, "native_error", completed=True)
        self.assertIsNotNone(p.lease_ceiling())


if __name__ == "__main__":
    unittest.main()
