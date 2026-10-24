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
                       {"min_ram_headroom_gib": 1.999},
                       {"pressure_seconds": 1.999},
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


class FastPressurePolicyTests(unittest.TestCase):
    def policy(self):
        policy = MemoryPolicy({"enabled": True, "mode": "live", "pressure_seconds": 4},
                              vram_reserve_mib=0)
        policy.live_actual(32 * 1024, 256, 100, "load_budget", completed=True, loaded=True)
        return policy

    def reading(self, stamp):
        return sample(stamp, used=20, gpu_used=15.999, gpu_total=16)

    def test_four_seconds_gpu_pressure_shrinks_before_growth_cooldown(self):
        policy = self.policy()
        info = {"arena_mib": 32 * 1024}
        # RAM has room to grow, while an external GPU allocation exceeds 99%.
        for now in (101, 103):
            self.assertIsNone(policy.observe(self.reading(now), True, info, now))
        plan = policy.observe(self.reading(105), True, info, 105)
        self.assertEqual(plan, {"resident_budget_gib": 32, "vram_reserve_mib": 419,
                                "reason": "sustained_pressure"})
        self.assertEqual(policy.current, {"resident_budget_gib": 32, "vram_reserve_mib": 256})

    def test_replayed_or_stale_samples_require_new_four_second_window(self):
        for invalid_stamp in (101, 99):
            with self.subTest(invalid_stamp=invalid_stamp):
                policy = self.policy()
                info = {"arena_mib": 32 * 1024}
                for now in (101, 103):
                    self.assertIsNone(policy.observe(self.reading(now), True, info, now))
                self.assertIsNone(policy.observe(self.reading(invalid_stamp), True, info, 105))
                for now in (106, 108):
                    self.assertIsNone(policy.observe(self.reading(now), True, info, now))
                plan = policy.observe(self.reading(110), True, info, 110)
                self.assertEqual(plan["reason"], "sustained_pressure")
                self.assertEqual(plan["resident_budget_gib"], 32)
                self.assertGreater(plan["vram_reserve_mib"], 256)


class HeadroomPolicyTests(unittest.TestCase):
    def test_explicit_two_gib_minimum_retains_five_percent_on_large_host(self):
        policy = MemoryPolicy({"enabled": True, "min_ram_headroom_gib": 2})
        # 44 GiB free, minus 3.2 GiB (5% of 64) and the 2 GiB startup allowance.
        self.assertEqual(policy.plan_for_load(sample(0, used=20), 0)["resident_budget_gib"], 38.8)

    def test_explicit_two_gib_minimum_bounds_small_host(self):
        policy = MemoryPolicy({"enabled": True, "min_ram_headroom_gib": 2})
        # On a 16 GiB host the absolute floor exceeds the 0.8 GiB percentage.
        self.assertEqual(policy.plan_for_load(sample(0, used=4, total=16), 0)["resident_budget_gib"], 8)

    def test_omitted_minimum_preserves_five_point_five_gib_default(self):
        policy = MemoryPolicy({"enabled": True})
        self.assertEqual(policy.plan_for_load(sample(0, used=20), 0)["resident_budget_gib"], 36.5)
        self.assertEqual(policy.status()["min_ram_headroom_gib"], 5.5)

    def test_loaded_percentage_pressure_shrinks_after_full_debounce(self):
        policy = MemoryPolicy({"enabled": True, "mode": "live", "min_ram_headroom_gib": 2})
        policy.live_actual(32 * 1024, 1536, 0, "load_budget", completed=True, loaded=True)
        info = {"arena_mib": 32 * 1024}
        self.assertIsNone(policy.observe(sample(1, used=64 * .95), True, info, 1))
        self.assertEqual(policy.status()["reason"], "stable")
        # 2.125 GiB free exceeds the absolute 2 GiB floor, but misses 5% by
        # a material amount. Pressure can reclaim RAM before growth cooldown.
        for now in range(2, 62):
            self.assertIsNone(policy.observe(sample(now, used=61.875), True, info, now))
        plan = policy.observe(sample(62, used=61.875), True, info, 62)
        self.assertEqual(plan, {"resident_budget_gib": 30.925, "vram_reserve_mib": 1536,
                                "reason": "sustained_pressure"})
        self.assertEqual(policy.current["resident_budget_gib"], 32)


class PostLoadPolicyTests(unittest.TestCase):
    def loaded_policy(self, mode="live", reserve=1536):
        policy = MemoryPolicy({"enabled": True, "mode": mode, "overhead_ram_gib": 4})
        policy.context_ram_gib = 1.538
        policy.context_startup_reserve_gib = 4
        policy.live_actual(17144, reserve, 100, "load_budget", completed=True, loaded=True)
        return policy

    def observe(self, policy, now, used=43.7421875, gpu_used=20, arena=17144, stamp=None):
        return policy.observe(sample(now if stamp is None else stamp, used=used, gpu_used=gpu_used),
                              True, {"arena_mib": arena, "expert_cache_mib": 5344}, now)

    def test_cold_load_reclaims_safe_ram_before_growth_cooldown(self):
        policy = self.loaded_policy()
        plan = self.observe(policy, 101)
        self.assertEqual(plan, {"resident_budget_gib": 31.5, "vram_reserve_mib": 1536,
                                "reason": "post_load_headroom"})
        self.assertEqual(policy.current["resident_budget_gib"], 17144 / 1024)
        self.assertEqual(policy.last_applied, 100)

    def test_reconciliation_keeps_gpu_reserve_even_when_gpu_has_room(self):
        policy = self.loaded_policy(reserve=2560)
        plan = self.observe(policy, 101)
        self.assertEqual(plan["resident_budget_gib"], 31.5)
        self.assertEqual(plan["vram_reserve_mib"], 2560)

    def test_completed_reconciliation_does_not_bypass_later_growth_limits(self):
        policy = self.loaded_policy()
        self.assertEqual(self.observe(policy, 101)["reason"], "post_load_headroom")
        policy.live_actual(32256, 1536, 102, "post_load_headroom", completed=True)
        for now in range(103, 224):
            self.assertIsNone(self.observe(policy, now, used=25, arena=32256))
        self.assertEqual(policy.status()["reason"], "cooldown")
        for now in range(224, 702):
            self.assertIsNone(self.observe(policy, now, used=25, arena=32256))
        plan = self.observe(policy, 702, used=25, arena=32256)
        self.assertEqual(plan["reason"], "stable_headroom")
        self.assertEqual(plan["resident_budget_gib"], 42)

    def test_progress_error_and_ordinary_apply_do_not_rearm_reconciliation(self):
        for reason, completed in (("native_progress", False), ("native_error", False),
                                  ("sustained_pressure", True)):
            with self.subTest(reason=reason):
                policy = self.loaded_policy()
                self.assertEqual(self.observe(policy, 101)["reason"], "post_load_headroom")
                policy.live_actual(20000, 1536, 102, reason, completed=completed)
                self.assertIsNone(self.observe(policy, 103, arena=20000))
                self.assertEqual(policy.status()["reason"], "growth_debounce")

    def test_invalid_or_preload_samples_preserve_first_loaded_observation(self):
        cases = ((101, {}, 17144), (110, sample(99), 17144),
                 (101, sample(100), 17144), (101, sample(99), 17144),
                 (101, sample(101), None), (101, sample(101), 0))
        for now, reading, arena in cases:
            with self.subTest(now=now, reading=reading, arena=arena):
                policy = self.loaded_policy()
                self.assertIsNone(policy.observe(reading, True, {"arena_mib": arena}, now))
                plan = self.observe(policy, now + 1)
                self.assertEqual(plan["reason"], "post_load_headroom")

    def test_replayed_sample_cannot_consume_loaded_reconciliation(self):
        policy = MemoryPolicy({"enabled": True, "mode": "live"})
        reading = sample(101, used=43.7421875)
        self.assertIsNone(policy.observe(reading, False, {}, 101))
        policy.live_actual(17144, 1536, 100, "load_budget", completed=True, loaded=True)
        self.assertIsNone(policy.observe(reading, True, {"arena_mib": 17144}, 101))
        self.assertEqual(self.observe(policy, 102)["reason"], "post_load_headroom")

    def test_first_pressure_or_small_ram_gain_consumes_reconciliation(self):
        for first_used, first_gpu in ((60, 20), (57.5, 20), (43.7421875, 23.99)):
            with self.subTest(used=first_used, gpu_used=first_gpu):
                policy = self.loaded_policy()
                self.assertIsNone(self.observe(policy, 101, used=first_used, gpu_used=first_gpu))
                self.assertIsNone(self.observe(policy, 102))
                self.assertEqual(policy.status()["reason"], "growth_debounce")

    def test_reload_mode_keeps_existing_debounce_and_cooldown(self):
        policy = self.loaded_policy(mode="reload")
        for now in range(101, 222):
            self.assertIsNone(self.observe(policy, now))
        self.assertEqual(policy.status()["reason"], "cooldown")
        for now in range(222, 700):
            self.assertIsNone(self.observe(policy, now))
        self.assertEqual(self.observe(policy, 700)["reason"], "stable_headroom")


class FixedResourcePolicyTests(unittest.TestCase):
    def policy(self, resident=32, reserve=1536, headroom=4, target_reserve=700, loaded=False):
        policy = MemoryPolicy({"enabled": True, "mode": "live", "pressure_seconds": 4,
                               "recovery_seconds": 30}, resident_cap_gib=55)
        policy.live_actual(resident * 1024, reserve, 100, "initial_budget",
                           completed=True, loaded=loaded)
        policy.update_resource_limits(headroom, target_reserve)
        return policy

    def observe(self, policy, now, used=40, gpu_used=20, stamp=None, info=None):
        if info is None:
            info = {"arena_mib": policy.current["resident_budget_gib"] * 1024,
                    "expert_cache_mib": 8192}
        return policy.observe(sample(now if stamp is None else stamp, used=used, gpu_used=gpu_used),
                              True, info, now)

    def quiet(self, policy, start, stop, **kwargs):
        for now in range(start, stop + 1):
            self.assertIsNone(self.observe(policy, now, **kwargs), f"unexpected plan at {now}")

    def test_fixed_ram_budget_does_not_keep_legacy_percentage_headroom(self):
        policy = MemoryPolicy({"enabled": True, "mode": "live"})
        policy.update_resource_limits(2, 256)
        self.assertEqual(policy.plan_for_load(sample(0, used=20), 0),
                         {"resident_budget_gib": 40, "vram_reserve_mib": 256,
                          "reason": "load_budget"})

    def test_fixed_growth_waits_thirty_seconds_and_is_bounded(self):
        policy = self.policy()
        self.quiet(policy, 101, 130)
        self.assertEqual(self.observe(policy, 131),
                         {"resident_budget_gib": 34, "vram_reserve_mib": 1408,
                          "reason": "stable_headroom"})
        self.assertEqual(policy.current, {"resident_budget_gib": 32, "vram_reserve_mib": 1536})
        self.assertEqual(policy.last_applied, 100)

    def test_cold_targets_use_fixed_ram_and_gpu_reserves_without_allocation(self):
        policy = MemoryPolicy({"enabled": True, "mode": "live"}, resident_cap_gib=55)
        for headroom, reserve, resident in ((2, 256, 40), (4, 700, 38), (8, 1536, 34)):
            with self.subTest(headroom=headroom, reserve=reserve):
                policy.update_resource_limits(headroom, reserve)
                plan = policy.plan_for_load(sample(0, used=20, gpu_used=4, gpu_total=64), 0)
                self.assertEqual(plan, {"resident_budget_gib": resident,
                                       "vram_reserve_mib": reserve, "reason": "load_budget"})
                self.assertIsNone(policy.current)
                self.assertIsNone(policy.last_applied)
        disabled = MemoryPolicy()
        disabled.update_resource_limits(2, 256)
        self.assertIsNone(disabled.plan_for_load(sample(0), 0))
        self.assertFalse(disabled.enabled)

    def test_fixed_targets_suppress_independent_ram_and_gpu_percent_pressure(self):
        policy = self.policy(reserve=256, headroom=2, target_reserve=256)
        # 2.6 GiB free RAM and 512 MiB free VRAM meet both fixed targets,
        # despite exceeding the legacy 95% RAM and 99% GPU ceilings.
        for now in range(101, 140):
            self.assertIsNone(policy.observe(sample(now, used=61.4, gpu_used=63.5, gpu_total=64),
                                             True, {"arena_mib": 32 * 1024}, now))
        self.assertEqual(policy.last_reason, "stable")
        self.assertIsNone(policy.pressure_since)

    def test_raised_targets_reset_windows_and_shrink_before_old_cooldown(self):
        policy = self.policy(reserve=256, headroom=2, target_reserve=256)
        self.quiet(policy, 101, 129, used=58)
        policy.update_resource_limits(8, 1536)
        self.assertIsNone(policy.growth_since)
        self.assertIsNone(policy.gpu_growth_since)
        self.quiet(policy, 130, 133, used=58)
        self.assertEqual(self.observe(policy, 134, used=58),
                         {"resident_budget_gib": 30, "vram_reserve_mib": 1536,
                          "reason": "sustained_pressure"})
        self.assertEqual(policy.current, {"resident_budget_gib": 32, "vram_reserve_mib": 256})
        self.assertEqual(policy.last_applied, 100)

    def test_reserve_floor_retarget_can_reclaim_without_ram_growth(self):
        policy = self.policy(reserve=256)
        # The native allocation still reports the old 256 MiB target. Retarget
        # to 700 MiB even if external free VRAM already exceeds that target.
        gpu_used = 24 - 800 / 1024
        self.quiet(policy, 101, 104, gpu_used=gpu_used)
        self.assertEqual(self.observe(policy, 105, gpu_used=gpu_used),
                         {"resident_budget_gib": 32, "vram_reserve_mib": 700,
                          "reason": "sustained_pressure"})

    def test_pressure_uses_fixed_gpu_deficit_and_never_grows_ram(self):
        policy = self.policy(reserve=700)
        gpu_used = 24 - 300 / 1024
        self.quiet(policy, 101, 104, gpu_used=gpu_used)
        self.assertEqual(self.observe(policy, 105, gpu_used=gpu_used),
                         {"resident_budget_gib": 32, "vram_reserve_mib": 1100,
                          "reason": "sustained_pressure"})

    def test_small_renewed_pressure_cancels_both_growth_windows(self):
        policy = self.policy()
        self.quiet(policy, 101, 129)
        # This RAM deficit cannot earn a material shrink; it still cancels growth.
        self.assertIsNone(self.observe(policy, 130, used=60.1))
        self.quiet(policy, 131, 160)
        self.assertEqual(self.observe(policy, 161)["reason"], "stable_headroom")

    def test_raw_two_gib_ram_margin_and_material_gpu_difference_are_required(self):
        policy = self.policy(reserve=700)
        gpu_used = 24 - 700 / 1024
        # The rounded budget has 2 GiB room, but raw RAM headroom has only 1.9996.
        self.quiet(policy, 101, 135, used=58.0004, gpu_used=gpu_used)
        self.quiet(policy, 136, 165, used=58, gpu_used=gpu_used)
        self.assertEqual(self.observe(policy, 166, used=58, gpu_used=gpu_used),
                         {"resident_budget_gib": 34, "vram_reserve_mib": 700,
                          "reason": "stable_headroom"})
        policy = self.policy(reserve=732)
        self.quiet(policy, 101, 135, used=60, gpu_used=24 - 731 / 1024)
        self.quiet(policy, 136, 165, used=60, gpu_used=24 - 732 / 1024)
        self.assertEqual(self.observe(policy, 166, used=60, gpu_used=24 - 732 / 1024),
                         {"resident_budget_gib": 32, "vram_reserve_mib": 700,
                          "reason": "stable_headroom"})

    def test_each_cache_earns_its_own_fresh_growth_window(self):
        # Stable GPU room must not lend thirty seconds to newly available RAM.
        policy = self.policy()
        self.quiet(policy, 101, 130, used=60)
        self.assertEqual(self.observe(policy, 131),
                         {"resident_budget_gib": 32, "vram_reserve_mib": 1408,
                          "reason": "stable_headroom"})
        # Stable RAM room must not lend its window to newly available GPU room.
        policy = self.policy()
        gpu_used = 24 - 700 / 1024
        self.quiet(policy, 101, 130, gpu_used=gpu_used)
        self.assertEqual(self.observe(policy, 131),
                         {"resident_budget_gib": 34, "vram_reserve_mib": 1536,
                          "reason": "stable_headroom"})

    def test_completed_or_limited_actual_ack_starts_a_new_post_ack_window(self):
        for actual, reserve in ((34, 1408), (33.999, 1408), (33, 128)):
            with self.subTest(actual=actual, reserve=reserve):
                policy = self.policy()
                self.quiet(policy, 101, 130)
                plan = self.observe(policy, 131)
                policy.live_actual(actual * 1024, reserve, 132, plan["reason"], completed=True)
                policy.complete_live_plan(plan, 32, "ram_capacity_or_rounding")
                self.assertEqual(policy.current["resident_budget_gib"], actual)
                self.assertEqual(policy.current["vram_reserve_mib"], reserve)
                self.assertEqual(policy.last_applied, 132)
                self.assertIsNone(policy.pressure_recovery_ceiling_gib)
                self.assertIsNone(self.observe(policy, 133, stamp=132))
                if reserve < 700:
                    self.quiet(policy, 134, 137)
                    next_plan = self.observe(policy, 138)
                    self.assertEqual(next_plan["reason"], "sustained_pressure")
                    self.assertEqual(next_plan["resident_budget_gib"], actual)
                else:
                    self.quiet(policy, 134, 163)
                    next_plan = self.observe(policy, 164)
                    self.assertEqual(next_plan["resident_budget_gib"], actual + 2)
                    self.assertEqual(next_plan["vram_reserve_mib"], reserve - 128)

    def test_loaded_ack_has_no_unbounded_immediate_reconciliation(self):
        policy = self.policy(loaded=True)
        self.quiet(policy, 101, 130)
        self.assertEqual(self.observe(policy, 131)["resident_budget_gib"], 34)
        # Loading again while fixed targets remain selected uses the same window.
        policy.live_actual(20 * 1024, 1536, 132, "load_budget", completed=True, loaded=True)
        self.quiet(policy, 133, 162)
        self.assertEqual(self.observe(policy, 163)["resident_budget_gib"], 22)

    def test_replay_staleness_gap_and_unknown_arena_reset_fixed_growth(self):
        cases = ((125, 124, None, False), (134, 125, None, False),
                 (130, 130, None, True), (125, 125, {}, False),
                 (125, 125, {"arena_mib": 0}, False))
        for now, stamp, info, fresh_gap in cases:
            with self.subTest(now=now, stamp=stamp, info=info):
                policy = self.policy()
                self.quiet(policy, 101, 124)
                self.assertIsNone(self.observe(policy, now, stamp=stamp, info=info))
                due = now + 30 if fresh_gap else now + 31
                self.quiet(policy, now + 1, due - 1)
                self.assertEqual(self.observe(policy, due)["reason"], "stable_headroom")

    def test_changed_limits_clear_old_episode_without_falsifying_actual_state(self):
        policy = MemoryPolicy({"enabled": True, "mode": "live", "min_ram_headroom_gib": 7.5,
                               "pressure_seconds": 4, "recovery_seconds": 30},
                              resident_cap_gib=55, vram_reserve_mib=2048)
        policy.live_actual(42 * 1024, 2048, 100, "initial_budget", completed=True)
        self.quiet(policy, 101, 104, used=63.5)
        pressure = self.observe(policy, 105, used=63.5)
        policy.live_actual(pressure["resident_budget_gib"] * 1024, 2048, 106,
                           "sustained_pressure", completed=True)
        policy.complete_live_plan(pressure, 42)
        self.assertEqual(policy.pressure_recovery_ceiling_gib, 42)
        self.quiet(policy, 107, 135)
        self.assertIsNotNone(policy.recovery_since)
        before = dict(policy.current), policy.last_applied, policy.last_sample
        policy.update_resource_limits(2, 256)
        self.assertEqual((policy.current, policy.last_applied, policy.last_sample), before)
        self.assertIsNone(policy.recovery_since)
        self.assertIsNone(policy.pressure_recovery_ceiling_gib)
        self.assertEqual(policy.status()["resource_targets"],
                         {"headroom_gib": 2, "vram_reserve_mib": 256})
        # Old pressure ACKs may still report actual sizes; they cannot arm a
        # legacy pressure episode under newly selected fixed limits.
        policy.complete_live_plan(pressure, 42)
        self.assertIsNone(policy.pressure_recovery_ceiling_gib)
        policy.live_actual(30 * 1024, 256, 136, "stable_headroom", completed=True)
        policy.update_resource_limits()
        self.assertFalse(policy.status()["fixed_resource_limits"])
        self.assertIsNone(policy.status()["resource_targets"])
        self.assertEqual(policy.headroom, 7.5)
        self.assertEqual(policy.reserve_floor, 2048)
        self.assertEqual(policy.current, {"resident_budget_gib": 30, "vram_reserve_mib": 256})
        self.assertEqual(policy.last_applied, 136)

    def test_same_targets_preserve_window_and_invalid_pairs_are_atomic(self):
        policy = self.policy()
        self.quiet(policy, 101, 129)
        window = policy.growth_since, policy.gpu_growth_since
        before = policy.status()
        for headroom, reserve in ((None, 700), (4, None), (True, 700), (1.9, 700),
                                  (129, 700), (float("nan"), 700), (4, True),
                                  (4, -1), (4, 700.5), (4, "700"), (4, float("inf"))):
            with self.subTest(headroom=headroom, reserve=reserve), self.assertRaises(ValueError):
                policy.update_resource_limits(headroom, reserve)
            self.assertEqual(policy.status(), before)
            self.assertEqual((policy.growth_since, policy.gpu_growth_since), window)
        policy.update_resource_limits(4, 700)
        self.assertEqual((policy.growth_since, policy.gpu_growth_since), window)
        self.assertIsNone(self.observe(policy, 130))
        self.assertEqual(self.observe(policy, 131)["reason"], "stable_headroom")

    def test_fixed_gpu_space_does_not_replan_same_arguments_or_follow_oscillation(self):
        policy = self.policy(resident=55, reserve=256, headroom=2, target_reserve=256)
        self.assertFalse(policy.record_loaded(sample(101, gpu_used=22), 101))
        for now in range(101, 400):
            self.assertIsNone(self.observe(policy, now, gpu_used=20 if now % 2 else 22))
        self.assertEqual(policy.last_reason, "stable")
        self.assertIsNone(policy.gpu_baseline)


class PressureRecoveryPolicyTests(unittest.TestCase):
    def policy(self, recovery=30, mode="live", reserve=1536):
        config = {"enabled": True, "mode": mode, "pressure_seconds": 4}
        if recovery is not None:
            config["recovery_seconds"] = recovery
        policy = MemoryPolicy(config, resident_cap_gib=55)
        policy.live_actual(42 * 1024, reserve, 100, "load_budget", completed=True, loaded=True)
        return policy

    def observe(self, policy, now, used=40, gpu_used=20, stamp=None, info=None):
        if info is None:
            info = {"arena_mib": policy.current["resident_budget_gib"] * 1024,
                    "expert_cache_mib": 8192}
        return policy.observe(sample(now if stamp is None else stamp, used=used, gpu_used=gpu_used),
                              True, info, now)

    def quiet(self, policy, start, stop, **kwargs):
        for now in range(start, stop + 1):
            self.assertIsNone(self.observe(policy, now, **kwargs), f"unexpected plan at {now}")

    def complete(self, policy, plan, now, actual=None, limitation=None):
        before = policy.current["resident_budget_gib"]
        actual = plan["resident_budget_gib"] if actual is None else actual
        policy.live_actual(actual * 1024, plan["vram_reserve_mib"], now,
                           plan["reason"], completed=True)
        policy.complete_live_plan(plan, before, limitation)

    def shrunk(self, recovery=30, mode="live", reserve=1536):
        policy = self.policy(recovery, mode, reserve)
        self.quiet(policy, 101, 104, used=63.5)
        plan = self.observe(policy, 105, used=63.5)
        self.assertEqual(plan["reason"], "sustained_pressure")
        self.assertEqual(plan["resident_budget_gib"], 37)
        self.complete(policy, plan, 106)
        return policy

    def test_recovery_setting_is_optional_and_bounded(self):
        for recovery in (None, 0, 30, 3600):
            with self.subTest(recovery=recovery):
                policy = self.policy(recovery)
                self.assertEqual(policy.status()["recovery_seconds"], recovery or 0)
        for recovery in (-1, 1, 29.999, 3601, True, "30", float("nan")):
            with self.subTest(recovery=recovery), self.assertRaises(ValueError):
                self.policy(recovery)

    def test_live_recovery_waits_thirty_fresh_seconds_then_restores_two_gib(self):
        policy = self.shrunk()
        self.quiet(policy, 107, 136)
        plan = self.observe(policy, 137)
        self.assertEqual(plan, {"resident_budget_gib": 39, "vram_reserve_mib": 1536,
                                "reason": "pressure_recovery"})
        self.assertEqual(policy.current["resident_budget_gib"], 37)
        self.assertEqual(policy.last_applied, 106)
        self.assertLess(137 - policy.last_applied, policy.cooldown)
        # A GPU sizing limitation does not invalidate a completed RAM-only step.
        self.complete(policy, plan, 138, limitation="gpu_pressure_cap")
        self.quiet(policy, 139, 168)
        self.assertEqual(self.observe(policy, 169)["resident_budget_gib"], 41)

    def test_repeated_shrinks_keep_first_actual_ceiling_and_each_step_waits_again(self):
        policy = self.shrunk()
        self.quiet(policy, 107, 110, used=63.5)
        pressure = self.observe(policy, 111, used=63.5)
        self.assertEqual(pressure["resident_budget_gib"], 32)
        self.complete(policy, pressure, 112)
        start = 113
        for actual in (34, 36, 38, 40, 42):
            self.quiet(policy, start, start + 29)
            recovery = self.observe(policy, start + 30)
            self.assertEqual(recovery["reason"], "pressure_recovery")
            self.assertEqual(recovery["resident_budget_gib"], actual)
            self.assertEqual(recovery["vram_reserve_mib"], 1536)
            self.complete(policy, recovery, start + 31)
            start += 32
        # Extra available capacity cannot raise the old pressure episode's ceiling.
        self.quiet(policy, start, start + 119)
        self.assertEqual(policy.current["resident_budget_gib"], 42)

    def test_material_safe_gain_and_ceiling_gain_are_required_without_gpu_growth(self):
        policy = self.shrunk(reserve=2560)
        # 7.499 GiB free minus the 5.5 GiB headroom permits only 1.999 GiB.
        self.quiet(policy, 107, 140, used=56.501, gpu_used=2)
        # _budget rounds this 1.9996 GiB room to 2.000; raw room remains too small.
        # Keep advancing samples with no gap so freshness cannot hide the boundary.
        self.quiet(policy, 141, 174, used=56.5004, gpu_used=2)
        self.quiet(policy, 175, 204, used=56.5, gpu_used=2)
        plan = self.observe(policy, 205, used=56.5, gpu_used=2)
        self.assertEqual(plan["resident_budget_gib"], 39)
        self.assertEqual(plan["vram_reserve_mib"], 2560)
        self.complete(policy, plan, 206)
        self.quiet(policy, 207, 236)
        plan = self.observe(policy, 237)
        self.assertEqual(plan["resident_budget_gib"], 41)
        self.complete(policy, plan, 238)
        # The last 1 GiB to the ceiling does not qualify for a fast step.
        self.quiet(policy, 239, 274)

    def test_replayed_stale_gap_and_unknown_arena_restart_recovery_window(self):
        cases = ((127, 126, None, False), (137, 127, None, False),
                 (132, 132, None, True), (127, 127, {}, False),
                 (127, 127, {"arena_mib": 0}, False))
        for now, stamp, info, fresh_gap in cases:
            with self.subTest(now=now, stamp=stamp, info=info):
                policy = self.shrunk()
                self.quiet(policy, 107, 126)
                self.assertIsNone(self.observe(policy, now, stamp=stamp, info=info))
                # A valid sample after a gap may itself start the new window.
                due = now + 30 if fresh_gap else now + 31
                self.quiet(policy, now + 1, due - 1)
                self.assertEqual(self.observe(policy, due)["reason"], "pressure_recovery")
        policy = self.shrunk()
        self.assertIsNone(self.observe(policy, 107, stamp=106))
        self.quiet(policy, 108, 137)
        self.assertEqual(self.observe(policy, 138)["reason"], "pressure_recovery")

    def test_even_small_renewed_pressure_resets_and_material_pressure_has_priority(self):
        policy = self.shrunk()
        self.quiet(policy, 107, 135)
        # Free RAM falls below headroom, but the 0.1 GiB deficit is not a shrink.
        self.assertIsNone(self.observe(policy, 136, used=58.6))
        self.quiet(policy, 137, 166)
        self.assertEqual(self.observe(policy, 167)["reason"], "pressure_recovery")
        for used, gpu_used in ((63.5, 20), (40, 23.99)):
            with self.subTest(used=used, gpu_used=gpu_used):
                policy = self.shrunk()
                self.quiet(policy, 107, 135)
                self.quiet(policy, 136, 139, used=used, gpu_used=gpu_used)
                pressure = self.observe(policy, 140, used=used, gpu_used=gpu_used)
                self.assertEqual(pressure["reason"], "sustained_pressure")
                self.assertLessEqual(pressure["resident_budget_gib"], 37)
                self.assertGreaterEqual(pressure["vram_reserve_mib"], 1536)

    def test_partial_error_and_capacity_limited_completion_do_not_credit_full_recovery(self):
        for completion in ("error", "undershoot", "capacity_limit"):
            with self.subTest(completion=completion):
                policy = self.shrunk()
                self.quiet(policy, 107, 136)
                plan = self.observe(policy, 137)
                if completion == "error":
                    policy.live_actual(38 * 1024, 1536, 138, "native_error")
                    self.assertEqual(policy.last_applied, 106)
                else:
                    actual = 39 - 1 / 1024 if completion == "undershoot" else 39
                    limitation = "ram_capacity_or_rounding" if completion == "capacity_limit" else None
                    self.complete(policy, plan, 138, actual, limitation)
                self.assertEqual(policy.current["resident_budget_gib"],
                                 38 if completion == "error" else actual)
                self.quiet(policy, 139, 170)

    def test_load_ordinary_completion_and_gpu_only_shrink_do_not_arm_old_episode(self):
        for completion in ("load", "ordinary", "gpu_only"):
            with self.subTest(completion=completion):
                policy = self.shrunk()
                if completion == "load":
                    policy.live_actual(30 * 1024, 1536, 110, "load_budget", completed=True, loaded=True)
                    # Consume the distinct post-load reconciliation with no RAM gain.
                    self.assertIsNone(self.observe(policy, 111, used=58.5))
                    start = 112
                else:
                    plan = {"resident_budget_gib": 37, "vram_reserve_mib": 2048,
                            "reason": "stable_headroom"}
                    self.complete(policy, plan, 110)
                    if completion == "gpu_only":
                        plan = {"resident_budget_gib": 37, "vram_reserve_mib": 2560,
                                "reason": "sustained_pressure"}
                        self.complete(policy, plan, 111)
                    start = 112
                self.quiet(policy, start, start + 31)

    def test_default_disabled_and_reload_mode_keep_normal_growth_limits(self):
        for recovery, mode in ((None, "live"), (0, "live"), (30, "reload")):
            with self.subTest(recovery=recovery, mode=mode):
                policy = self.shrunk(recovery, mode)
                self.quiet(policy, 107, 705)
                plan = self.observe(policy, 706)
                self.assertEqual(plan["reason"], "stable_headroom")
                self.assertEqual(plan["resident_budget_gib"], 55)


if __name__ == "__main__":
    unittest.main()
