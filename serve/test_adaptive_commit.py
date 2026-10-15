"""Commit-pressure regressions: plentiful physical RAM is not allocation capacity."""
import unittest
from serve.memory_policy import GIB, MemoryPolicy
from serve.test_memory_policy import sample


class CommitPressureTests(unittest.TestCase):
    def policy(self):
        p = MemoryPolicy({"enabled": True, "mode": "live", "pressure_seconds": 2}, 36, 256)
        p.applied({"resident_budget_gib": 36, "vram_reserve_mib": 256}, 0)
        return p

    def test_startup_uses_commit_even_with_abundant_physical_ram(self):
        p = self.policy()
        s = dict(sample(1, used=4), ram_commit_available=10 * GIB)
        self.assertEqual(p.plan_for_load(s, 1)["resident_budget_gib"], 2.5)

    def test_live_commit_pressure_releases_ram_without_growing_gpu(self):
        for fixed in (False, True):
            p = self.policy()
            if fixed:
                p.update_resource_limits(4, 256)
            for now in range(1, 4):
                s = dict(sample(now, used=40), ram_commit_available=2 * GIB)
                plan = p.observe(s, True, {"arena_mib": 36 * 1024}, now)
            self.assertEqual(plan["reason"], "sustained_pressure")
            self.assertLess(plan["resident_budget_gib"], 36)
            self.assertGreaterEqual(plan["vram_reserve_mib"], 256)

    def test_unavailable_or_invalid_required_commit_cannot_earn_growth(self):
        for value in (None, -1, float("nan"), True):
            p = self.policy()
            s = dict(sample(1, used=4), ram_commit_required=True, ram_commit_available=value)
            self.assertIsNone(p.plan_for_load(s, 1))
            self.assertIsNone(p.observe(s, True, {"arena_mib": 36 * 1024}, 1))
            self.assertEqual(p.last_reason, "telemetry_unavailable")

    def test_linux_without_commit_sensor_retains_physical_policy(self):
        self.assertEqual(self.policy().plan_for_load(sample(1, used=4), 1)["resident_budget_gib"], 36)


if __name__ == "__main__":
    unittest.main()
