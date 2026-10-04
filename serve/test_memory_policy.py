"""Deterministic resource-policy checks; no model or hardware is required."""
import unittest

try:
    from serve.memory_policy import GIB, MIB, MemoryPolicy
except ModuleNotFoundError:  # also support the repository's direct script convention
    from memory_policy import GIB, MIB, MemoryPolicy


def sample(now, used=50, total=64, gpu_used=20, gpu_total=24):
    return {"sampled_at": now, "ram_used": used * GIB, "ram_total": total * GIB,
            "gpu_mem_used": gpu_used * GIB, "gpu_mem_total": gpu_total * GIB}


class PolicyTests(unittest.TestCase):
    def policy(self, resident=42, reserve=1536):
        policy = MemoryPolicy({"enabled": True})
        policy.applied({"resident_budget_gib": resident, "vram_reserve_mib": reserve}, 0)
        return policy

    def observations(self, policy, start, stop, used=50, arena=42, gpu_used=20):
        result = None
        for now in range(start, stop + 1):
            result = policy.observe(sample(now, used=used, gpu_used=gpu_used), True,
                                    {"arena_mib": arena * 1024}, now)
        return result

    def test_default_disabled(self):
        policy = MemoryPolicy()
        self.assertIsNone(policy.plan_for_load(sample(0), 0))
        self.assertIsNone(policy.observe(sample(0), True, {"arena_mib": 42000}, 0))

    def test_load_obeys_os_headroom_and_cap(self):
        policy = self.policy()
        self.assertEqual(policy.plan_for_load(sample(1, used=4), 1)["resident_budget_gib"], 42)
        self.assertEqual(policy.plan_for_load(sample(1, used=20), 1)["resident_budget_gib"], 36.5)
        self.assertEqual(policy.plan_for_load(sample(1, used=63), 1)["resident_budget_gib"], 1)
        self.assertEqual(policy.plan_for_load(sample(1), 1)["vram_reserve_mib"], 1536)

    def test_95_point_4_percent_is_not_immediate_unload(self):
        policy = self.policy()
        self.assertIsNone(policy.observe(sample(601, used=64 * .954), True,
                                         {"arena_mib": 42 * 1024}, 601))
        self.assertEqual(policy.status()["reason"], "pressure_debounce")
        self.assertEqual(policy.current["resident_budget_gib"], 42)

    def test_sustained_pressure_proposes_shrink_only(self):
        policy = self.policy()
        self.assertIsNone(self.observations(policy, 600, 659, used=61.056))
        result = self.observations(policy, 660, 660, used=61.056)
        self.assertEqual(result["reason"], "sustained_pressure")
        self.assertLess(result["resident_budget_gib"], 42)
        self.assertEqual(result["vram_reserve_mib"], 1536)
        self.assertEqual(policy.current["resident_budget_gib"], 42)  # proposal is pure

    def test_absolute_os_headroom_is_stronger_than_percentage(self):
        policy = self.policy()
        result = self.observations(policy, 600, 660, used=60, arena=42)
        self.assertLess(60 / 64, .95)
        self.assertEqual(result["reason"], "sustained_pressure")
        self.assertEqual(result["resident_budget_gib"], 40.5)

    def test_sustained_gpu_pressure_increases_reserve(self):
        policy = self.policy()
        result = self.observations(policy, 600, 660, gpu_used=23.999)
        # A 24 GiB device at 99.995% needs only ~245 MiB to reach 99%; this
        # remains below the material 256 MiB threshold, so no reload loop.
        self.assertIsNone(result)
        policy = MemoryPolicy({"enabled": True, "vram_target_percent": 98})
        policy.applied({"resident_budget_gib": 42, "vram_reserve_mib": 1536}, 0)
        result = self.observations(policy, 600, 660, gpu_used=23.99)
        self.assertGreaterEqual(result["vram_reserve_mib"], 1536 + 256)
        self.assertEqual(result["resident_budget_gib"], 42)

    def test_growth_stable_capped_and_gpu_floor_retained(self):
        policy = self.policy(resident=32, reserve=2560)
        self.assertIsNone(self.observations(policy, 600, 719, used=35, arena=32))
        result = self.observations(policy, 720, 720, used=35, arena=32)
        self.assertEqual(result["reason"], "stable_headroom")
        self.assertEqual(result["resident_budget_gib"], 42)
        self.assertEqual(result["vram_reserve_mib"], 1536)

    def test_cooldown_after_successful_apply(self):
        policy = self.policy()
        result = self.observations(policy, 600, 660, used=61.056)
        policy.applied(result, 660)
        self.assertIsNone(self.observations(policy, 661, 900, used=25,
                                            arena=result["resident_budget_gib"]))
        self.assertEqual(policy.status()["reason"], "cooldown")
        result = self.observations(policy, 901, 1260, used=25,
                                   arena=result["resident_budget_gib"])
        self.assertIsNotNone(result)

    def test_early_sustained_pressure_bypasses_growth_cooldown(self):
        policy = self.policy()
        self.assertIsNone(self.observations(policy, 1, 60, used=61.056))
        result = self.observations(policy, 61, 61, used=61.056)
        self.assertEqual(result["reason"], "sustained_pressure")
        self.assertLess(result["resident_budget_gib"], 42)
        policy.applied(result, 61)
        # Applying a shrink starts a fresh debounce. A further material loss
        # can shrink again only after another full stable-pressure window.
        self.assertIsNone(self.observations(policy, 62, 121, used=63,
                                            arena=result["resident_budget_gib"]))
        second = self.observations(policy, 122, 122, used=63,
                                   arena=result["resident_budget_gib"])
        self.assertEqual(second["reason"], "sustained_pressure")
        self.assertLess(second["resident_budget_gib"], result["resident_budget_gib"])
        policy.applied(second, 122)
        # Recovered capacity remains growth-cooldown bound after either shrink.
        self.assertIsNone(self.observations(policy, 123, 500, used=25,
                                            arena=second["resident_budget_gib"]))
        self.assertEqual(policy.status()["reason"], "cooldown")
        recovered = self.observations(policy, 501, 722, used=25,
                                      arena=second["resident_budget_gib"])
        self.assertEqual(recovered["reason"], "stable_headroom")

    def test_loaded_runtime_does_not_recharge_startup_allowance(self):
        policy = MemoryPolicy({"enabled": True, "overhead_ram_gib": 6})
        # Before startup: 20 GiB external workload. Reserve 6 GiB for dense,
        # vision and runtime buffers, and retain 5.5 GiB OS headroom.
        plan = policy.plan_for_load(sample(0, used=20), 0)
        self.assertEqual(plan["resident_budget_gib"], 32.5)
        policy.applied(plan, 1)
        # After startup: the same workload plus the actual arena and the 6 GiB
        # runtime allocations. Those allocations are already in ram_used.
        for now in range(600, 1000):
            self.assertIsNone(policy.observe(sample(now, used=58.5), True,
                                             {"arena_mib": 32.5 * 1024}, now))
        self.assertEqual(policy.current["resident_budget_gib"], 32.5)
        self.assertEqual(policy.status()["reason"], "stable")

    def test_no_same_budget_reload_under_pressure_at_minimum(self):
        policy = self.policy(resident=1)
        self.assertIsNone(self.observations(policy, 600, 900, used=63, arena=1))
        self.assertEqual(policy.status()["reason"], "stable")

    def test_missing_stale_and_nonfinite_freeze_and_reset(self):
        policy = self.policy()
        self.observations(policy, 600, 650, used=61.056)
        for snapshot in (None, {}, sample(640), sample(651, used=float("nan"))):
            self.assertIsNone(policy.observe(snapshot, True, {"arena_mib": 43008}, 651))
            self.assertIsNone(policy.pressure_since)
        self.assertIsNone(self.observations(policy, 652, 660, used=61.056))
        self.assertIsNone(policy.plan_for_load(sample(0), 100))

    def test_no_growth_when_allocation_unknown_or_unloaded(self):
        policy = self.policy(resident=20)
        self.assertIsNone(policy.observe(sample(600, used=25), True, {}, 600))
        self.assertIsNone(policy.observe(sample(601, used=25), False, {"arena_mib": 20480}, 601))

    def test_replayed_out_of_order_and_gap_cannot_satisfy_debounce(self):
        policy = self.policy()
        self.observations(policy, 600, 650, used=61.056)
        self.assertIsNone(policy.observe(sample(649, used=61.056), True,
                                         {"arena_mib": 43008}, 651))
        self.assertIsNone(self.observations(policy, 652, 660, used=61.056))
        self.assertIsNone(policy.observe(sample(720, used=61.056), True,
                                         {"arena_mib": 43008}, 720))
        self.assertEqual(policy.pressure_since, 720)

    def test_tiny_headroom_change_and_ordinary_idle_do_not_reload(self):
        policy = self.policy(resident=32)
        self.assertIsNone(self.observations(policy, 600, 1000, used=57.5, arena=32))
        self.assertIsNone(self.observations(policy, 1001, 1100, used=51, arena=32))

    def test_unsafe_configuration_rejected(self):
        for config in ({"ram_target_percent": 99}, {"min_ram_headroom_gib": 1},
                       {"cooldown_seconds": 2}, {"vram_target_percent": 100}):
            with self.assertRaises(ValueError):
                MemoryPolicy(config)

    def test_stable_new_gpu_headroom_replans_same_arguments_once(self):
        policy = self.policy()
        self.assertTrue(policy.record_loaded(sample(1, gpu_used=22), 1))
        # Baseline cannot follow a transient new free-space observation.
        self.assertFalse(policy.record_loaded(sample(2, gpu_used=20), 2))
        info = {"arena_mib": 42 * 1024, "expert_cache_mib": 12000}
        for now in range(600, 720):
            self.assertIsNone(policy.observe(sample(now, gpu_used=21), True, info, now))
        result = policy.observe(sample(720, gpu_used=21), True, info, 720)
        self.assertEqual(result["reason"], "stable_gpu_headroom")
        self.assertEqual(result["resident_budget_gib"], 42)
        self.assertEqual(result["vram_reserve_mib"], 1536)
        policy.applied(result, 720)
        self.assertIsNone(policy.gpu_baseline)
        self.assertFalse(policy.record_loaded(sample(719, gpu_used=21), 721))
        self.assertTrue(policy.record_loaded(sample(721, gpu_used=21), 721))
        for now in range(722, 1500):
            self.assertIsNone(policy.observe(sample(now, gpu_used=21), True, info, now))

    def test_gpu_growth_requires_known_loaded_cache_and_idle_baseline(self):
        policy = self.policy()
        info = {"arena_mib": 42 * 1024, "expert_cache_mib": 12000}
        for now in range(600, 730):
            self.assertIsNone(policy.observe(sample(now, gpu_used=20), True, info, now))
        self.assertTrue(policy.record_loaded(sample(731, gpu_used=22), 731))
        for now in range(732, 900):
            self.assertIsNone(policy.observe(sample(now, gpu_used=20), True,
                                             {"arena_mib": 42 * 1024}, now))

    def test_short_gpu_workspace_recovery_and_small_change_do_not_reload(self):
        policy = self.policy()
        policy.record_loaded(sample(1, gpu_used=22), 1)
        info = {"arena_mib": 42 * 1024, "expert_cache_mib": 12000}
        for now in range(600, 719):
            self.assertIsNone(policy.observe(sample(now, gpu_used=21), True, info, now))
        self.assertIsNone(policy.observe(sample(719, gpu_used=22), True, info, 719))
        for now in range(720, 1000):
            self.assertIsNone(policy.observe(sample(now, gpu_used=21.6), True, info, now))


if __name__ == "__main__":
    unittest.main()
