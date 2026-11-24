"""Inkling-2-Mini fit planner: what N request slots (`--parallel N`) cost on each backend, the largest N that fits, and a plain answer when the
requested N does not fit at the context (an Arc A750 8 GB crashed at `--parallel 32`: dalloc 256 MiB, out of device memory, 0 of 3072 experts in VRAM).

The cost model follows the engines' allocation code (read in inkling_fit.py): CUDA and HIP keep a 4096-token window of global K/V per slot on the
device, SYCL allocates every slot's K/V at the full context in FP16 up front, the workspaces grow from 8 slots on."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from strata_control import inkling_fit as F

GIB = F.GIB_TO_GB


class PerBackendSlotCost(unittest.TestCase):
    def test_sycl_gives_every_slot_the_full_context_in_fp16(self):
        # 32 slots x 32768 tokens x 32 KiB: 34 GB of K/V that the old plan counted as 2 GB (FP8, 4096-token window)
        sycl = F.global_kv_gb(32768, 32, backend="sycl")
        self.assertAlmostEqual(sycl, 32 * 32768 * F.GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9, delta=0.01)
        cuda = F.global_kv_gb(32768, 32, fp8_kv=True, backend="cuda")
        self.assertAlmostEqual(cuda, 32 * 4096 * F.GLOBAL_KV_BYTES_PER_TOK_FP8 / 1e9, delta=0.01)
        self.assertGreater(sycl, 10 * cuda)

    def test_hip_and_cuda_keep_a_4096_token_window_per_slot_from_16_slots(self):
        for be in ("hip", "cuda", None):
            self.assertAlmostEqual(F.global_kv_gb(131072, 16, backend=be), 16 * 4096 * F.GLOBAL_KV_BYTES_PER_TOK_FP8 / 1e9, delta=0.01)

    def test_sycl_has_no_fp8_kv(self):
        self.assertFalse(F.auto_fp8_kv(8.0, 32768, 64, backend="sycl"))
        self.assertTrue(F.auto_fp8_kv(8.0, 32768, 64, backend="cuda"))
        self.assertAlmostEqual(F.swa_sconv_gb(32, backend="sycl"), 32 * F.SWA_AND_SCONV_GB_FP16, delta=0.01)

    def test_the_drafter_kv_is_per_slot_on_sycl_only(self):
        one = F.dspark_gb(32768, True, 1, backend="sycl")
        many = F.dspark_gb(32768, True, 8, backend="sycl")
        self.assertGreater(many, 7 * (one - F.DSPARK_BASE_GB))
        self.assertEqual(F.dspark_gb(32768, True, 8, backend="cuda"), F.dspark_gb(32768, True, 1, backend="cuda"))

    def test_the_workspaces_grow_from_8_slots(self):
        for be in (None, "cuda", "hip", "sycl"):
            below = F.non_expert_gb(32768, 7, host_offload_global_kv=True, backend=be)
            at = F.non_expert_gb(32768, 8, host_offload_global_kv=True, backend=be)
            self.assertGreater(at - below, F.SLOT_WORKSPACE_EXTRA_GB * 0.9, be)

    def test_one_slot_is_what_it_was(self):
        # the single-slot plan of every backend still starts where it started (SQ-10 on a 4 GiB card, experts in RAM)
        for be in (None, "cuda"):
            pl = F.place("SQ-10", 3.7, 16, False, 32768, backend=be, ram_free_gib=10.0)
            self.assertTrue(pl.get("kv_fits", True), (be, pl))


class AgainstTheEngine(unittest.TestCase):
    """Numbers from strata-inkling --serve --parallel N (RTX 3060 12 GB, SQ-10, ctx 32768, drafter off): the memory ledger's runner + K/V and the host K/V."""
    ENGINE_RUNNER_KV_MIB = {1: 1920, 4: 2420, 8: 3074, 16: 3022, 32: 4626, 64: 8018}
    ENGINE_HOST_KV_MIB = {4: 3520, 8: 7808, 16: 7722, 32: 16104, 64: 32472}

    def test_the_slot_cost_follows_the_engines_ledger(self):
        def planner_gib(n):
            fp8 = F.auto_fp8_kv(11.6, 32768, n, "on", backend="cuda")
            return F.non_expert_gb(32768, n, dspark=False, fp8_kv=fp8, hybrid=True, host_offload_global_kv=True, backend="cuda") / GIB
        for n, mib in self.ENGINE_RUNNER_KV_MIB.items():
            engine_delta = (mib - self.ENGINE_RUNNER_KV_MIB[1]) / 1024.0
            planner_delta = planner_gib(n) - planner_gib(1)
            self.assertAlmostEqual(planner_delta, engine_delta, delta=0.7, msg=f"{n} slots")
            if n >= 32:     # never under the engine by more than the safety margin once the slots dominate
                self.assertGreater(planner_delta, engine_delta - 0.4, msg=f"{n} slots")

    def test_the_host_kv_is_every_slots_whole_context(self):
        for n, mib in self.ENGINE_HOST_KV_MIB.items():
            fp8 = F.auto_fp8_kv(11.6, 32768, n, "on", backend="cuda")
            total = F.global_kv_total_gb(32768, n, fp8_kv=fp8) * 1e9 / (1 << 20)
            # the planner counts all of it; the engine keeps a small window on the card (0.6 GiB at 4 slots, 1.25 GiB at 64) and maps the rest
            self.assertGreaterEqual(total, mib * 0.99, msg=f"{n} slots")
            self.assertLessEqual(total, mib + 1500, msg=f"{n} slots")

    def test_the_ram_for_the_host_kv_limits_the_slots(self):
        small = F.max_parallel("SQ-10", 11.6, 32.0, False, 32768, backend="cuda", desktop=True, ram_free_gib=28.0)["max_parallel"]
        big = F.max_parallel("SQ-10", 11.6, 200.0, False, 32768, backend="cuda", desktop=True, ram_free_gib=190.0)["max_parallel"]
        self.assertLess(small, 64)
        self.assertEqual(big, 64)      # the engine did start all of 4, 8, 16, 32 and 64 slots on this card with plenty of RAM


class LargestParallel(unittest.TestCase):
    def test_arc_a750_8gb_parallel_32_does_not_fit(self):
        r = F.max_parallel("SQ-10", 7.5, 32.0, False, 32768, requested=32, backend="sycl", desktop=True, ram_free_gib=28.0)
        self.assertFalse(r["fits"])
        self.assertLess(r["max_parallel"], 32)
        self.assertGreaterEqual(r["max_parallel"], 1)
        self.assertIn("--parallel 32 does not fit", r["message"])
        self.assertIn("32768", r["message"])
        self.assertIn(str(r["max_parallel"]), r["message"])

    def test_the_same_request_on_cuda_with_host_kv_is_not_refused_on_12_gib(self):
        r = F.max_parallel("SQ-10", 11.6, 32.0, False, 32768, requested=32, backend="cuda", desktop=True, ram_free_gib=28.0)
        self.assertTrue(r["fits"], r)
        self.assertEqual(r["message"], f"{r['max_parallel']} slots fit at a context of 32768 tokens")

    def test_not_even_one_slot(self):
        r = F.max_parallel("SQ-10", 3.7, 32.0, False, 32768, requested=4, backend="sycl", ram_free_gib=28.0)
        self.assertEqual(r["max_parallel"], 0)
        self.assertFalse(r["fits"])
        self.assertIn("not even one slot", r["message"])

    def test_a_smaller_context_makes_room_and_the_answer_names_it(self):
        r = F.max_parallel("SQ-15", 15.5, 32.0, False, 32768, requested=32, backend="sycl", desktop=True, ram_free_gib=28.0)
        self.assertFalse(r["fits"])
        self.assertGreater(r["max_context_at_requested"], 0)
        self.assertLess(r["max_context_at_requested"], 32768)
        self.assertIn(str(r["max_context_at_requested"]), r["message"])
        # and at that context the request really fits
        self.assertTrue(F.parallel_fits("SQ-15", 15.5, 32.0, False, r["max_context_at_requested"], 32, backend="sycl", desktop=True,
                                        ram_free_gib=28.0))

    def test_more_card_never_means_fewer_slots(self):
        for be in (None, "cuda", "hip", "sycl"):
            last = -1
            for vram in (3.7, 5.5, 7.5, 9.5, 11.6, 15.5, 23.2):
                n = F.max_parallel("SQ-10", vram, 32.0, False, 32768, backend=be, ram_free_gib=28.0)["max_parallel"]
                self.assertGreaterEqual(n, last, (be, vram))
                last = n

    def test_a_longer_context_never_means_more_slots(self):
        for be in ("hip", "sycl"):
            last = 10 ** 6
            for ctx in (4096, 8192, 32768, 131072):
                n = F.max_parallel("SQ-10", 11.6, 32.0, False, ctx, backend=be, ram_free_gib=28.0)["max_parallel"]
                self.assertLessEqual(n, last, (be, ctx))
                last = n

    def test_the_count_is_capped_at_the_engine_maximum(self):
        r = F.max_parallel("SQ-20", 23.2, 64.0, False, 8192, requested=200, backend="cuda", ram_free_gib=56.0)
        self.assertLessEqual(r["max_parallel"], F.MAX_PARALLEL)
        self.assertLessEqual(r["requested"], F.MAX_PARALLEL)

    def test_every_slot_counted_is_a_slot_that_fits(self):
        # the answer is consistent with parallel_fits at N and N + 1
        for be in ("cuda", "hip", "sycl"):
            r = F.max_parallel("SQ-10", 7.5, 32.0, False, 32768, backend=be, desktop=True, ram_free_gib=28.0)
            n = r["max_parallel"]
            if n:
                self.assertTrue(F.parallel_fits("SQ-10", 7.5, 32.0, False, 32768, n, backend=be, desktop=True, ram_free_gib=28.0))
            if n < F.MAX_PARALLEL:
                self.assertFalse(F.parallel_fits("SQ-10", 7.5, 32.0, False, 32768, n + 1, backend=be, desktop=True, ram_free_gib=28.0))

    def test_unified_memory_counts_the_sycl_slots_too(self):
        sycl = F.place("SQ-10", None, 24.0, True, 32768, False, 32, backend="sycl")
        plain = F.place("SQ-10", None, 24.0, True, 32768, False, 32, backend=None)
        self.assertLess(sycl["vram_gb"], plain["vram_gb"])      # the same pool holds less of the model when every slot has its full-context K/V
        self.assertGreater(sycl["ssd_gb"], plain["ssd_gb"])


if __name__ == "__main__":
    unittest.main()
