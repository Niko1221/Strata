"""mem_guard 的 CPU 闸门：锚点必须逐点等于实测，边界必须和 bench 一致。

只锁「算得对不对」，不锁「真能跑」—— 后者只能靠 bench 实测。这里的三条判据都
来自 2026-11-01 的实跑记录（48 GB 卡 / pack 16 GiB / 1 行 / no_grad）：

    ce/mezo T=27648  33.0 GiB  ok
    ce/mezo T=49152  47.5 GiB  ok（只剩 0.5 GiB，边界）
    ce/mezo T=52736  OOM（单笔 22.2 GB 分配失败）

以前用解析式 `0.5 + 2e-4·T + c·T²/2^30`：c=3.6 时 49152 只算 43.6（低 3.9，
把「刚够」判成「宽裕」），c=5.3 时 32768 算 37.1（高 1.8，把可跑判成不可跑）。
改成锚点插值后，这两个方向都不会再错。
"""

from __future__ import annotations

import os
import unittest

import mem_guard as M


class TestMezoAnchors(unittest.TestCase):
    def test_anchors_exact(self):
        """锚点必须逐点命中：插值函数在锚点处就等于实测值。"""
        for t, want in M.MEZO_ANCHORS:
            self.assertAlmostEqual(M.estimate_peak_gib(t, 1, "mezo"), want, places=6,
                                   msg=f"T={t}")

    def test_monotonic(self):
        """峰值必须严格随 T 增长 —— 非单调会让二分求上限取到错的根。"""
        prev = 0.0
        for t in range(0, 200000, 512):
            v = M.estimate_peak_gib(t, 1, "mezo")
            self.assertGreaterEqual(v, prev, f"T={t} 比上一档还小")
            prev = v

    def test_49152_fits_52736_does_not(self):
        """边界与实跑一致：49152 通过、52736 拒绝（0.99 安全线）。"""
        limit = M._limit_gib(48.0, "mezo")
        self.assertLessEqual(M.estimate_peak_gib(49152, 1, "mezo"), limit)
        self.assertGreater(M.estimate_peak_gib(52736, 1, "mezo"), limit)

    def test_tail_is_steeper(self):
        """末锚点外推要比「线性」更陡：没测过的点必须偏保守。

        去掉 MEZO_TAIL_PENALTY 时 52736 是 50.16，加上是 50.16 —— 1.10 只作用在
        超出 49152 的部分，这里直接比斜率。
        """
        k_lin = ((M.MEZO_ANCHORS[-1][1] - M.MEZO_ANCHORS[-2][1])
                 / (M.MEZO_ANCHORS[-1][0] - M.MEZO_ANCHORS[-2][0]))
        got = (M.estimate_peak_gib(60000, 1, "mezo")
               - M.estimate_peak_gib(49152, 1, "mezo")) / (60000 - 49152)
        self.assertGreater(got, k_lin * 1.05)

    def test_rows_extra(self):
        """rows=2 只多一份 sqrt 项，不能按两份峰值线性翻倍。"""
        a = M.estimate_peak_gib(27648, 1, "mezo")
        b = M.estimate_peak_gib(27648, 2, "mezo")
        self.assertGreater(b, a)
        self.assertLess(b - a, 2.0)          # 实测 dpo rows=2 只多 ~1.5 GiB

    def test_anchor_hit_flag(self):
        self.assertTrue(M.mezo_anchor_hit(49152))
        self.assertFalse(M.mezo_anchor_hit(45056))   # 插值区，不是实测


class TestEngineRoute(unittest.TestCase):
    def test_128k_matches_measured(self):
        """引擎 130816 token 实测 26.9 GiB（其中 int8 KV 1.68）。"""
        est = M.estimate_peak_gib(130816, route="engine")
        self.assertAlmostEqual(est, 26.9, delta=0.25)

    def test_engine_stays_far_below_hf(self):
        """同长度下引擎的账必须显著低于 HF mezo —— 这不是优化，是两套计量。"""
        self.assertLess(M.estimate_peak_gib(49152, route="engine"),
                        M.estimate_peak_gib(49152, 1, "mezo") - 15)

    def test_kv_auto_steps(self):
        """`--kv auto` 的档位切换点与 generate.cpp / setup.py 的 KV_AUTO_STEPS 一致：
        fp16 到 32K、int8 到 128K、k8v4 到 256K、再往上 q4_0。位宽逐档变窄 ⇒ 显存单调降。"""
        steps = [(8192, "fp16"), (32768, "fp16"), (32769, "int8"),
                 (131072, "int8"), (131073, "k8v4"), (262144, "k8v4"), (262145, "q4_0")]
        for t, want in steps:
            self.assertEqual(M.estimate_engine_gib(t), M.estimate_engine_gib(t, want),
                             f"T={t} 应落在 {want} 档")
        self.assertLess(M.estimate_engine_gib(131072, "int8"),
                        M.estimate_engine_gib(131072, "fp16"))
        self.assertLess(M.estimate_engine_gib(131072, "q4_0"),
                        M.estimate_engine_gib(131072, "int8"))


class TestMaxSeq(unittest.TestCase):
    def test_bisect_consistent(self):
        """max_seq_len 必须和 estimate 同口径：该值通过、再加一档不过。"""
        t = M.max_seq_len("mezo", 1, 48.0)
        self.assertEqual(t % 256, 0)
        limit = M._limit_gib(48.0, "mezo")
        self.assertLessEqual(M.estimate_peak_gib(t, 1, "mezo"), limit)
        self.assertGreater(M.estimate_peak_gib(t + 256, 1, "mezo"), limit)


class TestExclusiveGpu(unittest.TestCase):
    """pipe 拉引擎前的独占检查（free/total 全部注入，不碰真 nvidia-smi）。

    背景：serve 的排队/并行槽位防的是单引擎进程内的请求打爆；pipe 拉第二个引擎
    进程不经过任何排队 —— 实测双驻留 committed 89.7 GiB > 物理 48 GiB，引擎 stall
    自杀。判据是「卡基本是空的」，不是 need（--expert-cache auto 会把空闲吃满）。
    """

    def test_empty_card_passes(self):
        # 桌面占用 ~1 GiB 在容忍内；引擎需求下限远小于空闲
        M.require_exclusive_gpu(26.9, total_gib=48.0, free_mib=47.0 * 1024, log=lambda *a: None)

    def test_resident_engine_refused(self):
        # 服务器引擎驻留后只剩 ~8 GiB：即使 > 需求下限 26.9 不成立、即使 < 也一样拦
        with self.assertRaises(SystemExit):
            M.require_exclusive_gpu(26.9, total_gib=48.0, free_mib=8.0 * 1024, log=lambda *a: None)

    def test_desktop_allowance_boundary(self):
        # 空闲 = total − allowance 恰好过；再少 1 MiB 拦
        boundary = (48.0 - M.EXCLUSIVE_ALLOWANCE_MIB / 1024) * 1024
        M.require_exclusive_gpu(5.0, total_gib=48.0, free_mib=boundary, log=lambda *a: None)
        with self.assertRaises(SystemExit):
            M.require_exclusive_gpu(5.0, total_gib=48.0, free_mib=boundary - 1, log=lambda *a: None)

    def test_need_not_used_as_threshold(self):
        # 需求下限高于空闲时也拦（显存真不够），但错误口径必须与驻留拦不同路径——
        # 这里只验证行为：都拦，退出码同为 2
        for free in (8.0 * 1024, 40.0 * 1024):
            with self.assertRaises(SystemExit) as cm:
                M.require_exclusive_gpu(45.0, total_gib=48.0, free_mib=free, log=lambda *a: None)
            self.assertEqual(cm.exception.code, 2)

    def test_no_nvidia_smi_skips_loudly(self):
        # free=None（nvidia-smi 不可用）：不拦但必须出声，不静默
        lines = []
        M.require_exclusive_gpu(26.9, total_gib=48.0, free_mib=None, log=lines.append)
        self.assertTrue(any("跳过" in s for s in lines))

    def test_env_override_skips_loudly(self):
        lines = []
        os.environ["STRATA_ALLOW_SHARED_GPU"] = "1"
        try:
            M.require_exclusive_gpu(26.9, total_gib=48.0, free_mib=8.0 * 1024, log=lines.append)
        finally:
            del os.environ["STRATA_ALLOW_SHARED_GPU"]
        self.assertTrue(any("STRATA_ALLOW_SHARED_GPU" in s for s in lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
