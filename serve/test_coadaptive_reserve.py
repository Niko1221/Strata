"""Absolute running reserve semantics, independent from startup fitting."""
import unittest

from serve.memory_policy import GIB, MIB, MemoryPolicy


class AbsoluteReserveTests(unittest.TestCase):
    def policy(self, current=448):
        policy = MemoryPolicy({"enabled": True, "mode": "live", "pressure_seconds": 4}, 32, current)
        policy.live_actual(32 * 1024, current, 100, "loaded", completed=True)
        policy.update_resource_limits(4, 320)
        policy.reclaim_gpu_headroom = True
        return policy

    def sample(self, timestamp, free_ram=10, free_gpu=400):
        return dict(sampled_at=timestamp, ram_total=64 * GIB, ram_used=(64 - free_ram) * GIB,
                    gpu_mem_total=8 * GIB, gpu_mem_used=8 * GIB - free_gpu * MIB)

    def settle(self, policy, **kwargs):
        plan = None
        for timestamp in range(101, 106):
            plan = policy.observe(self.sample(timestamp, **kwargs), True, {"arena_mib": 32768}, timestamp)
        return plan

    def test_native_target_does_not_add_deficit_to_startup_reserve(self):
        plan = self.settle(self.policy(), free_gpu=280)
        self.assertEqual(plan["vram_reserve_mib"], 320)
        self.assertEqual(plan["resident_budget_gib"], 32)

    def test_renewed_pressure_at_same_target_can_reclaim(self):
        plan = self.settle(self.policy(current=320), free_gpu=300)
        self.assertEqual(plan["vram_reserve_mib"], 320)

    def test_ram_only_reclaim_holds_actual_gpu_headroom(self):
        plan = self.settle(self.policy(), free_ram=2.5, free_gpu=400)
        self.assertEqual(plan["resident_budget_gib"], 30.5)
        self.assertEqual(plan["vram_reserve_mib"], 400)

    def test_combined_pressure_does_not_double_count_gpu_deficit(self):
        plan = self.settle(self.policy(), free_ram=2.5, free_gpu=280)
        self.assertEqual(plan["resident_budget_gib"], 30.5)
        self.assertEqual(plan["vram_reserve_mib"], 320)

    def test_ram_only_recovery_cannot_spend_unearned_gpu_headroom(self):
        policy = self.policy(current=320)
        policy.live_actual(30 * 1024, 320, 100, "shrink", completed=True)
        policy.gpu_capacity_ceiling = True
        for timestamp in range(101, 132):
            plan = policy.observe(self.sample(timestamp, free_ram=8, free_gpu=700), True,
                                  {"arena_mib": 30720}, timestamp)
        self.assertEqual(plan["resident_budget_gib"], 32)
        self.assertEqual(plan["vram_reserve_mib"], 700)


if __name__ == "__main__":
    unittest.main()
