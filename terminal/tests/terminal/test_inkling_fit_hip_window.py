"""#7022: on a discrete HIP card the plan keeps room for a K/V window (32K tokens a slot), not the whole context.

The engine's K/V pools start empty and grow on demand (VisitKvPool::grow_pool, ink_hip_backend.hip.cpp): when the card is full the pool moves to
host-mapped pinned memory, so a longer chat is slower and never a failure.  Counting the whole context (0.2.21) took the card from the experts: an
RX 9070 XT (16 GB) with SQ-15 planned 3.94 GiB of experts at 128K and none at 256K (0.2.20: 11.10 at 128K).  Every other backend, and the unified-memory
path, plans the context it is given."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

import setup as S
from strata_control import inkling_fit as F

CARD = 15.9           # an RX 9070 XT: what the planner is given (AMD has no free-memory figure, the registry size counts)


def hip(build="SQ-15", ctx=32768, **kw):
    kw.setdefault("ram_free_gib", 45.0)
    return F.place(build, CARD, 64, False, ctx, True, parallel=kw.pop("parallel", 1), backend="hip", **kw)


class HipWindow(unittest.TestCase):
    def test_a_longer_context_plans_like_32k(self):
        base = hip(ctx=32768)
        for ctx in (65536, 131072, 262144, 1048576):
            self.assertEqual(hip(ctx=ctx), base, ctx)

    def test_the_16_gb_card_keeps_its_experts_at_128k_and_256k(self):
        flags = S.inkling_expert_flags(hip(ctx=131072))
        self.assertEqual(flags, ["--vram-experts-gb", "9.11", "--ram-experts-gb", "2.33"])
        self.assertEqual(S.inkling_expert_flags(hip(ctx=262144)), flags)

    def test_up_to_32k_nothing_changes(self):
        # the window starts at 32K: 8K and 32K are the plans 0.2.21 wrote
        self.assertEqual(S.inkling_expert_flags(hip(ctx=32768)), ["--vram-experts-gb", "9.11", "--ram-experts-gb", "2.33"])
        small = F.non_expert_gb(8192, 1, vision=True, dspark=True, hybrid=True, backend="hip")
        self.assertLess(small, F.non_expert_gb(32768, 1, vision=True, dspark=True, hybrid=True, backend="hip"))

    def test_the_window_is_per_slot(self):
        one = F.non_expert_gb(131072, 1, dspark=True, hybrid=True, backend="hip")
        four = F.non_expert_gb(131072, 4, dspark=True, hybrid=True, backend="hip")
        window = F.non_expert_gb(F.HIP_KV_WINDOW_TOKENS, 4, dspark=True, hybrid=True, backend="hip")
        self.assertEqual(four, window)
        self.assertGreater(four, one)

    def test_non_expert_gb_counts_the_window_for_the_k_v_and_the_drafter(self):
        a = F.non_expert_gb(131072, 1, dspark=True, hybrid=True, backend="hip")
        b = F.non_expert_gb(32768, 1, dspark=True, hybrid=True, backend="hip")
        self.assertEqual(a, b)

    def test_other_backends_still_plan_the_whole_context(self):
        for backend in ("cuda", "sycl", "metal", None):
            small = F.non_expert_gb(32768, 1, dspark=True, hybrid=True, backend=backend)
            big = F.non_expert_gb(131072, 1, dspark=True, hybrid=True, backend=backend)
            self.assertGreater(big, small, backend)

    def test_the_unified_memory_path_is_not_windowed(self):
        # Strix Halo (HIP on a unified pool): the K/V is in the same memory, nothing to spill to
        small = F.place("SQ-15", None, 24, True, 32768, True, backend="hip")
        big = F.place("SQ-15", None, 24, True, 262144, True, backend="hip")
        self.assertGreater(small["vram_gb"], big["vram_gb"])

    def test_cuda_plan_still_depends_on_the_context(self):
        a = F.place("SQ-15", CARD, 64, False, 32768, True, backend="cuda", ram_free_gib=45.0)
        b = F.place("SQ-15", CARD, 64, False, 262144, True, backend="cuda", ram_free_gib=45.0)
        self.assertNotEqual(a, b)


class HipKvChoice(unittest.TestCase):
    def kv(self, asked, backend):
        return S.inkling_plan_kv(SimpleNamespace(kv=asked), backend)

    def test_hip_plans_the_type_the_person_chose(self):
        self.assertEqual(self.kv("fp8", "hip"), {"fp8_kv": True})
        self.assertEqual(self.kv("k8v8", "hip"), {"fp8_kv": True})      # setup turns it into fp8 on HIP
        self.assertEqual(self.kv("fp16", "hip"), {"fp8_kv": False})
        self.assertEqual(self.kv("", "hip"), {})
        self.assertEqual(self.kv(None, "hip"), {})
        self.assertEqual(self.kv("auto", "hip"), {})

    def test_other_backends_are_left_alone(self):
        for backend in ("cuda", "sycl", "metal", None):
            for asked in ("fp8", "fp16", "k8v8", "k8v4", ""):
                self.assertEqual(self.kv(asked, backend), {}, (backend, asked))

    def test_fp8_gives_the_card_more_experts(self):
        auto = hip(ctx=131072)
        fp8 = hip(ctx=131072, fp8_kv=True)
        self.assertGreater(fp8["vram_gb"], auto["vram_gb"])
        self.assertEqual(fp8["kv_dtype"], "fp8")
        self.assertEqual(hip(ctx=131072, fp8_kv=False), auto)           # the 16 GB card's rule is FP16: an explicit fp16 changes nothing there


if __name__ == "__main__":
    unittest.main()
