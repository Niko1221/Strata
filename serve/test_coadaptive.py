"""Direction and admission tests, including the opposite-pressure cases."""
import unittest
from serve.coadaptive import CoAdaptive, GIB, MIB
from serve.test_live_memory import service
from serve.memory_policy import MemoryPolicy


def reading(now, cpu=0, free_gpu=2048, free_ram=16, gpu_util=0, external_gpu=None):
    return {"sampled_at": now, "ram_total": 64 * GIB, "ram_used": (64 - free_ram) * GIB,
            "gpu_mem_total": 8 * GIB, "gpu_mem_used": 8 * GIB - free_gpu * MIB, "gpu_util": gpu_util,
            "workload": {"complete": True, "cpu_percent": cpu, "gpu_percent": external_gpu}}


class CoAdaptiveTests(unittest.TestCase):
    def controller(self):
        return CoAdaptive({"enabled": True, "mode": "live"}, ram_headroom_gib=4, vram_reserve_mib=256, pcie_frac=.37)

    def test_enabled_defaults_to_observation_without_actuation(self):
        c = CoAdaptive({"enabled": True}, pcie_frac=.37)
        self.assertEqual(self.settle(c, cpu=65)["mode"], "yield_cpu")
        self.assertFalse(c.active)
        s = service()
        s.engine.spawn[1].extend(["--pcie-frac", ".37"])
        s.configure_coadaptive({"enabled": True})
        self.assertFalse(s.memory_policy.reclaim_gpu_headroom)
        self.assertIsNone(s.resource_limits)

    def test_new_destination_pressure_cancels_promotion_without_quiet_dwell(self):
        for initial, changed in (({"cpu": 65}, {"cpu": 65, "external_gpu": 90}),
                                 ({"external_gpu": 90}, {"cpu": 65, "external_gpu": 90})):
            c = self.controller()
            self.settle(c, **initial)
            c.observe(reading(10, **changed), 10)
            self.assertEqual(c.mode, "balanced")
            self.assertAlmostEqual(c.fraction(10), .37)

    def settle(self, controller, **kwargs):
        for now in range(10):
            controller.observe(reading(now, **kwargs), now)
        return controller.status(9)

    def test_compiler_cpu_load_favors_gpu_without_eviction(self):
        c = self.controller()
        self.assertEqual(self.settle(c, cpu=65)["mode"], "yield_cpu")
        self.assertEqual(c.limits(), (4, 256))
        self.assertAlmostEqual(c.fraction(9), .37)  # pressure alone cannot prove routing benefit

    def test_external_gpu_load_favors_cpu_and_preserves_ram(self):
        c = self.controller()
        self.assertEqual(self.settle(c, external_gpu=90)["mode"], "yield_gpu")
        self.assertEqual(c.limits(), (4, 768))
        self.assertAlmostEqual(c.fraction(9), .37)

    def test_ram_pressure_with_gpu_room_favors_gpu(self):
        c = self.controller()
        self.assertEqual(self.settle(c, free_ram=3)["mode"], "yield_cpu")
        self.assertEqual(c.limits()[1], 256)

    def test_both_busy_does_not_invent_spare_compute(self):
        c = self.controller()
        self.assertEqual(self.settle(c, cpu=80, external_gpu=90)["mode"], "balanced")

    def test_stratas_own_gpu_load_is_not_external_pressure(self):
        c = self.controller()
        self.assertEqual(self.settle(c, cpu=60, gpu_util=100)["mode"], "yield_cpu")

    def test_low_capacity_without_attributed_compute_does_not_inflate_reserve(self):
        c = self.controller()
        self.assertEqual(self.settle(c, free_gpu=240, gpu_util=100)["mode"], "balanced")
        self.assertEqual(c.limits(), (4, 256))

    def test_capacity_pressure_cancels_cpu_promotion_without_claiming_gpu_work(self):
        c = self.controller()
        self.settle(c, cpu=65)
        c.observe(reading(10, cpu=65, free_gpu=240), 10)
        self.assertEqual(c.mode, "balanced")
        self.assertEqual(c.limits(), (4, 256))

    def test_missing_stale_or_replayed_sensor_cannot_authorize_route(self):
        c = self.controller()
        self.settle(c, cpu=60)
        self.assertIsNone(c.fraction(15))
        c.observe(reading(9, cpu=60), 10)
        self.assertIsNone(c.fraction(10))

    def test_recovery_to_baseline_requires_sustained_fresh_quiet(self):
        c = self.controller()
        self.settle(c, cpu=60)
        for now in range(10, 40):
            c.observe(reading(now), now)
        self.assertEqual(c.mode, "yield_cpu")
        c.observe(reading(40), 40)
        self.assertEqual(c.mode, "balanced")

    def test_service_has_one_target_owner_and_keeps_process(self):
        s = service()
        s.engine.spawn[1].extend(["--pcie-frac", ".37"])
        proc = s.engine.proc
        s.configure_coadaptive({"enabled": True})
        self.assertTrue(s.coadaptive.enabled)
        self.assertIs(s.engine.proc, proc)
        with self.assertRaises(ValueError):
            s.configure_resources({"enabled": True, "selection": "auto"})

    def test_no_silent_routing_override_without_explicit_baseline(self):
        with self.assertRaises(ValueError):
            service().configure_coadaptive({"enabled": True})

    def test_free_gpu_can_grow_again_at_the_same_reserve_after_ram_release(self):
        p = MemoryPolicy({"enabled": True, "mode": "live"}, 32, 256)
        p.update_resource_limits(4, 256)
        p.reclaim_gpu_headroom = True
        p.applied({"resident_budget_gib": 32, "vram_reserve_mib": 256}, 0)
        for now in range(1, 32):
            plan = p.observe(reading(now, free_ram=4, free_gpu=1024), True, {"arena_mib": 32 * 1024}, now)
        self.assertEqual(plan["resident_budget_gib"], 32)
        self.assertEqual(plan["vram_reserve_mib"], 256)
        self.assertEqual(plan["reason"], "stable_headroom")


if __name__ == "__main__":
    unittest.main()
