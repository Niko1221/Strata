"""Background safety and fairness policy, without a model or OS mutations."""
import unittest

from serve.coadaptive import CoAdaptive
from serve.memory_policy import GIB, MIB, MemoryPolicy


def reading(now, ram=8, gpu=400, commit=None):
    snapshot = {"sampled_at": now, "ram_total": 64 * GIB, "ram_used": (64 - ram) * GIB,
                "gpu_mem_total": 8 * GIB, "gpu_mem_used": 8 * GIB - gpu * MIB,
                "native_capacity_required": True, "native_free_mib": gpu}
    if commit is not None:
        snapshot.update(ram_commit_required=True, ram_commit_available=commit * GIB)
    return snapshot


def policy(resident=32):
    p = MemoryPolicy({"enabled": True, "mode": "live"}, 32, 448)
    p.live_actual(resident * 1024, 448, 0, "loaded", completed=True, loaded=True)
    p.update_resource_limits(4, 320)
    p.reclaim_gpu_headroom = True
    return p


class BackgroundMemoryTests(unittest.TestCase):
    def test_soft_ram_pressure_reclaims_first_sample_without_waiting(self):
        p = policy()
        sample = reading(1, ram=3.9)
        proposal = p.observe(sample, True, {"arena_mib": 32768}, 1)
        self.assertEqual(proposal["reason"], "sustained_pressure")
        self.assertEqual(proposal["resident_budget_gib"], 31.875)
        self.assertEqual(proposal["vram_reserve_mib"], 400)
        self.assertEqual(p.safety_decision(sample, 1)["action"], "run")

    def test_soft_gpu_pressure_reclaims_first_sample_without_stopping(self):
        p = policy()
        sample = reading(1, gpu=280)
        proposal = p.observe(sample, True, {"arena_mib": 32768}, 1)
        self.assertEqual(proposal["resident_budget_gib"], 32)
        self.assertEqual(proposal["vram_reserve_mib"], 320)
        self.assertEqual(p.safety_decision(sample, 1)["action"], "run")

    def test_release_all_resident_weights_then_grow_from_empty(self):
        p = policy(resident=2)
        proposal = p.observe(reading(1, ram=.5), True, {"arena_mib": 2048}, 1)
        self.assertEqual(proposal["resident_budget_gib"], 0)
        p.live_actual(0, 400, 1, "sustained_pressure", completed=True)
        for stamp in range(2, 32):
            self.assertIsNone(p.observe(reading(stamp), True, {"arena_mib": 0}, stamp))
        proposal = p.observe(reading(32), True, {"arena_mib": 0}, 32)
        self.assertEqual(proposal["resident_budget_gib"], 2)
        self.assertEqual(proposal["reason"], "stable_headroom")

    def test_commit_pressure_is_independent_of_free_physical_ram(self):
        p = policy()
        sample = reading(1, ram=20, commit=.5)
        proposal = p.observe(sample, True, {"arena_mib": 32768}, 1)
        self.assertEqual(proposal["resident_budget_gib"], 28.5)
        decision = p.safety_decision(sample, 1)
        self.assertEqual(decision["action"], "wait")
        self.assertEqual(decision["effective_ram_free_mib"], 512)

    def test_unknown_native_reading_cannot_allocate(self):
        p = policy()
        sample = reading(1)
        sample["native_free_mib"] = None
        self.assertIsNone(p.observe(sample, True, {"arena_mib": 32768}, 1))
        self.assertEqual(p.safety_decision(sample, 1)["action"], "wait")

    def test_legacy_resource_presets_keep_the_existing_pressure_window(self):
        p = policy()
        p.reclaim_gpu_headroom = False
        sample = reading(1, ram=2)
        self.assertIsNone(p.observe(sample, True, {"arena_mib": 32768}, 1))
        self.assertEqual(p.safety_decision(sample, 1)["reason"], "disabled")

    def test_partial_ram_growth_has_backoff_but_relief_remains_immediate(self):
        p = policy(resident=28)
        for stamp in range(1, 32):
            proposal = p.observe(reading(stamp), True, {"arena_mib": 28672}, stamp)
        self.assertEqual(proposal["resident_budget_gib"], 30)
        p.live_actual(28 * 1024, 400, 31, "stable_headroom", completed=True)
        p.complete_live_plan(proposal, 28, "ram_capacity_or_rounding")
        for stamp in range(32, 72):
            other = p.observe(reading(stamp), True, {"arena_mib": 28672}, stamp)
            if other is not None:
                self.assertEqual(other["resident_budget_gib"], 28)  # GPU recovery remains legal.
        relief = p.observe(reading(72, ram=2), True, {"arena_mib": 28672}, 72)
        self.assertEqual(relief["resident_budget_gib"], 26)

    def test_material_new_capacity_can_release_ram_backoff(self):
        p = policy(resident=28)
        p.ram_growth_block = {"free_ram_gib": 8, "until": 999}
        for stamp in range(1, 32):
            proposal = p.observe(reading(stamp, ram=10), True, {"arena_mib": 28672}, stamp)
        self.assertIsNone(p.ram_growth_block)
        self.assertEqual(proposal["resident_budget_gib"], 30)

    def test_pressure_does_not_grow_gpu_from_fractional_free_sensor(self):
        p = policy()
        proposal = p.observe(reading(1, ram=3, gpu=400.5), True, {"arena_mib": 32768}, 1)
        self.assertEqual(proposal["vram_reserve_mib"], 401)


class BackgroundSafetyTests(unittest.TestCase):
    def test_critical_floor_waits_first_sample_and_reports_limitation(self):
        p = policy()
        decision = p.safety_decision(reading(1, gpu=249), 1, {"arena_mib": 32768}, "prefill_cache_floor")
        self.assertEqual(decision["action"], "wait")
        self.assertEqual(decision["reason"], "non_evictable_vram_floor")
        self.assertEqual(decision["min_vram_free_mib"], 250)

    def test_zero_ram_floor_is_reported_as_unsupported(self):
        p = policy(resident=0)
        decision = p.safety_decision(reading(1, ram=.75), 1, {"arena_mib": 0})
        self.assertEqual(decision["reason"], "non_evictable_ram_floor")

    def test_short_sensor_gap_does_not_pause_previously_healthy_task(self):
        p = policy()
        self.assertEqual(p.safety_decision(reading(1), 1)["action"], "run")
        for stamp in range(2, 8):
            sample = reading(stamp)
            sample["native_free_mib"] = None
            self.assertEqual(p.safety_decision(sample, stamp)["reason"], "telemetry_grace")
        sample = reading(8)
        sample["native_free_mib"] = None
        self.assertEqual(p.safety_decision(sample, 8)["action"], "wait")

    def test_host_emergency_bypasses_missing_gpu_sensor_grace(self):
        p = policy()
        p.safety_decision(reading(1), 1)
        sample = reading(2, ram=.5)
        sample["native_free_mib"] = None
        self.assertEqual(p.safety_decision(sample, 2)["reason"], "ram_floor")
        self.assertEqual(p.last_safety["action"], "wait")

    def test_recovery_requires_margin_and_five_fresh_seconds(self):
        p = policy()
        p.safety_decision(reading(1, gpu=100), 1)
        for stamp in range(2, 10):
            self.assertEqual(p.safety_decision(reading(stamp, gpu=260), stamp)["action"], "wait")
        for stamp in range(10, 15):
            self.assertEqual(p.safety_decision(reading(stamp), stamp)["action"], "wait")
        self.assertEqual(p.safety_decision(reading(15), 15)["action"], "run")

    def test_replayed_samples_do_not_earn_recovery(self):
        p = policy()
        p.safety_decision(reading(1, gpu=100), 1)
        p.safety_decision(reading(2), 2)
        for stamp in range(3, 8):
            self.assertEqual(p.safety_decision(reading(2), stamp)["action"], "wait")

    def test_recovered_capacity_after_gap_restarts_quiet_window(self):
        p = policy()
        p.safety_decision(reading(1, gpu=100), 1)
        p.safety_decision(reading(2), 2)
        self.assertEqual(p.safety_decision(reading(20), 20)["action"], "wait")
        for stamp in range(21, 25):
            self.assertEqual(p.safety_decision(reading(stamp), stamp)["action"], "wait")
        self.assertEqual(p.safety_decision(reading(25), 25)["action"], "run")

    def test_stale_commit_cannot_authorize_resume(self):
        p = policy()
        p.safety_decision(reading(1, commit=.5), 1)
        sample = reading(2)
        sample.update(ram_commit_required=True, ram_commit_available=None)
        self.assertEqual(p.safety_decision(sample, 2)["action"], "wait")


class FairnessDelayTests(unittest.TestCase):
    def controller(self, mode="live"):
        return CoAdaptive({"enabled": True, "mode": mode})

    def sample(self, stamp, cpu=0, gpu=None, complete=True):
        return {"sampled_at": stamp, "gpu_util": 100,
                "workload": {"cpu_percent": cpu, "gpu_percent": gpu, "complete": complete}}

    def test_shadow_mode_never_changes_runtime(self):
        self.assertEqual(self.controller("shadow").fairness_decision(self.sample(1, cpu=90), 1)["delay_ms"], 0)

    def test_strata_device_utilization_does_not_cause_delay(self):
        self.assertEqual(self.controller().fairness_decision(self.sample(1), 1)["delay_ms"], 0)

    def test_attributed_work_yields_bounded_time_without_route_change(self):
        c = self.controller()
        self.assertEqual(c.fairness_decision(self.sample(1, cpu=60), 1)["delay_ms"], 10)
        self.assertEqual(c.fairness_decision(self.sample(2, gpu=100), 2)["delay_ms"], 20)
        self.assertEqual(c.fairness_decision(self.sample(3, cpu=100, gpu=100), 3)["delay_ms"], 20)
        self.assertEqual(c.selected_fraction, c.base_fraction)

    def test_missing_or_stale_load_cannot_authorize_throttle(self):
        c = self.controller()
        self.assertEqual(c.fairness_decision(self.sample(1, cpu=90, complete=False), 1)["delay_ms"], 0)
        self.assertEqual(c.fairness_decision(self.sample(1, cpu=90), 7)["delay_ms"], 0)
        self.assertEqual(c.fairness_decision(self.sample(1, cpu=float("nan")), 1)["delay_ms"], 0)


class RoutingBudgetTests(unittest.TestCase):
    def setup_route(self):
        from serve.routing_costs import RoutingCosts
        from serve.test_routing_costs import profile
        c = CoAdaptive({"enabled": True, "mode": "live"}, pcie_frac=.37)
        c.runtime_key = "runtime"
        c.routing = RoutingCosts(profile())
        snapshot = reading(10)
        snapshot["workload"] = {"complete": True, "cpu_percent": 40, "gpu_percent": 0}
        native = {"sampled_at": 10, "resident_mib": 32768, "cache_mib": 736, "free_mib": 4000}
        return c, snapshot, native

    def test_os_budget_blocks_promotion_despite_optimistic_cuda_headroom(self):
        for budget in (0, 249, -1, None, float("nan")):
            with self.subTest(budget=budget):
                c, snapshot, native = self.setup_route()
                native["budget_free_mib"] = budget
                self.assertEqual(c.choose_route(snapshot, native, 128, 128, 11), .37)

    def test_absent_or_ample_optional_budget_preserves_measured_routing(self):
        for budget in (None, 400):
            c, snapshot, native = self.setup_route()
            if budget is not None:
                native["budget_free_mib"] = budget
            self.assertEqual(c.choose_route(snapshot, native, 128, 128, 11), .65)

    def test_stale_native_capacity_cannot_promote_even_with_ample_budget(self):
        c, snapshot, native = self.setup_route()
        native.update(sampled_at=1, budget_free_mib=4000)
        self.assertEqual(c.choose_route(snapshot, native, 128, 128, 11), .37)

    def test_missing_required_commit_cannot_promote_to_cpu(self):
        c, snapshot, native = self.setup_route()
        for row in c.routing.profile["records"]:
            if row["fraction"] == .15:
                row["decode_ms_per_token"] = 10
        snapshot.update(ram_commit_required=True, ram_commit_available=None)
        native["budget_free_mib"] = 0
        self.assertEqual(c.choose_route(snapshot, native, 128, 128, 11), .37)


if __name__ == "__main__":
    unittest.main()
