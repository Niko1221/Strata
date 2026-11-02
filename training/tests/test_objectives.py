"""`objectives.py` 的数学与开关。

    python -m unittest test_objectives

要守住的：
  1. CE/DPO 与原先三处手写版**数值一致**（改结构不能改答案）
  2. 优势只在组内归一化，std=0 的组贡献 0
  3. GRPO：ratio=1 时退化成 −mean(adv)；非对称裁剪真的非对称
  4. TIS 只做常数缩放 —— 传进去的带梯度张量不许回传梯度
  5. DAPO 与长度归一化在长短混排时必须给出不同的数
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import unittest

import numpy as np
import torch

import objectives as O


class SftCe(unittest.TestCase):
    def test_mean_and_sum(self):
        lp = torch.tensor([-2.0, -0.5])
        self.assertAlmostEqual(float(O.sft_loss(lp)), 1.25, places=6)
        self.assertAlmostEqual(float(O.sft_loss(lp, reduction="sum")), 2.5, places=6)

    def test_mask_drops_prompt_positions(self):
        lp = torch.tensor([-9.0, -1.0, -3.0])
        m = torch.tensor([False, True, True])
        self.assertAlmostEqual(float(O.sft_loss(lp, m)), 2.0, places=6)

    def test_an_all_masked_row_is_nan_not_zero(self):
        # 全 mask 掉就是「没有可算的 token」；静默返回 0 会让日志显示 loss=0 而实际没训
        lp = torch.tensor([-1.0, -2.0])
        with self.assertRaises(Exception):
            v = O.sft_loss(lp, torch.tensor([False, False]))
            if torch.isnan(v):
                raise ValueError("nan")

    def test_grad_flows(self):
        lp = torch.tensor([-1.0], requires_grad=True)
        O.sft_loss(lp).backward()
        self.assertAlmostEqual(float(lp.grad), -1.0, places=6)   # d(−mean lp)/d lp = −1/n


class Dpo(unittest.TestCase):
    def test_matches_logsigmoid_away_from_saturation(self):
        m = torch.tensor([0.7, -1.3, 2.0], dtype=torch.float64)
        got = O.dpo_loss(m, beta=0.1)
        want = -torch.nn.functional.logsigmoid(0.1 * m)
        self.assertTrue(torch.allclose(got, want, atol=1e-12))

    def test_saturated_region_stays_finite(self):
        # bf16 下 -logsigmoid(大负数) 会走 0/0；softplus 不会
        m = torch.tensor([-5000.0])
        v = O.dpo_loss(m, beta=0.1)
        self.assertTrue(torch.isfinite(v).all())
        self.assertAlmostEqual(float(v), 500.0, places=3)

    def test_margin_is_chosen_minus_rejected_with_ref(self):
        self.assertAlmostEqual(float(O.dpo_margin(-1.0, -3.0, -1.5, -2.0)), 1.5, places=6)

    def test_margin_token_normalization_is_length_invariant(self):
        # 每侧按各自答案 token 数归一：同一段文本放大 T 倍，margin 不变 —— lr 因此
        # 不随序列长度重标定（2026-11-02：序列级 margin 在 122K 对上发散的根因）。
        m_short = O.dpo_margin(-1.0, -3.0, -1.5, -2.0, c_ntok=1, r_ntok=1)
        self.assertAlmostEqual(float(m_short), 1.5, places=6)
        # 两侧等比放大 1000×、token 数也 ×1000 ⇒ 归一后逐值相等
        m_long = O.dpo_margin(-1000.0, -3000.0, -1500.0, -2000.0, c_ntok=1000, r_ntok=1000)
        self.assertAlmostEqual(float(m_long), 1.5, places=6)
        # 两侧 token 数不同时各除各的
        m_mixed = O.dpo_margin(-2.0, -3.0, -1.0, -1.0, c_ntok=2, r_ntok=4)
        self.assertAlmostEqual(float(m_mixed), (-2.0 + 1.0) / 2 - (-3.0 + 1.0) / 4, places=6)
        # 0 = 不归一（序列级遗留语义）
        self.assertAlmostEqual(float(O.dpo_margin(-2.0, -3.0, -1.0, -1.0)), 1.0, places=6)


class Advantages(unittest.TestCase):
    def test_within_group_only(self):
        # 两组：第一组 [1,0]，第二组 [5,5] 全同。跨组归一化会把第二组也弄出优势
        r = torch.tensor([1.0, 0.0, 5.0, 5.0])
        adv, used, zero = O.group_advantages(r, 2)
        self.assertEqual(used, 1)
        self.assertFalse(zero)
        self.assertGreater(float(adv[0]), 0)
        self.assertLess(float(adv[1]), 0)
        self.assertEqual(float(adv[2]), 0.0)
        self.assertEqual(float(adv[3]), 0.0)

    def test_all_equal_group_contributes_nothing(self):
        r = torch.tensor([2.0, 2.0])
        adv, used, zero = O.group_advantages(r, 2)
        self.assertEqual(used, 0)
        self.assertTrue(zero)
        self.assertEqual(float(adv.abs().sum()), 0.0)

    def test_scale_off_is_the_drgpo_variant(self):
        r = torch.tensor([1.0, 0.0])
        on, _, _ = O.group_advantages(r, 2, scale=True)
        off, _, _ = O.group_advantages(r, 2, scale=False)
        self.assertGreater(float(on.abs()[0]), float(off.abs()[0]))   # std=0.5 < 1 放大
        self.assertAlmostEqual(float(off[0]), 0.5, places=6)          # 只减均值

    def test_adv_dtype_is_float32(self):
        r = torch.tensor([1, 0], dtype=torch.long)
        adv, _, _ = O.group_advantages(r, 2)
        self.assertEqual(adv.dtype, torch.float32)


class GrpoLoss(unittest.TestCase):
    def _one(self, logp, old, adv, mask, **kw):
        t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
        loss, st = O.grpo_loss(t(logp), t(old), t(adv), t(mask), O.GrpoCfg(**kw))
        return float(loss), st

    def test_ratio_one_reduces_to_minus_mean_adv(self):
        loss, st = self._one(-1.0, -1.0, 2.0, 1.0)
        self.assertAlmostEqual(loss, -2.0, places=6)
        self.assertAlmostEqual(st["ratio"], 1.0, places=6)
        self.assertEqual(st["clip_frac"], 0.0)

    def test_asymmetric_clip_lets_the_upward_side_run(self):
        # ratio=exp(0.4)=1.4918，adv=1
        up_sym, _ = self._one(0.4, 0.0, 1.0, 1.0, clip_low=0.2, clip_high=0.2)
        up_asym, _ = self._one(0.4, 0.0, 1.0, 1.0, clip_low=0.2, clip_high=0.28)
        self.assertAlmostEqual(-up_sym, 1.2, places=5)    # clamp 到 1.2
        self.assertAlmostEqual(-up_asym, 1.28, places=5)  # clamp 到 1.28
        self.assertGreater(-up_asym, -up_sym)

    def test_negative_advantage_keeps_its_own_direction(self):
        # adv<0 时 min 挑的是**较大**的那一项，裁剪方向与 adv>0 相反，这是 PPO 的原式
        loss, st = self._one(0.4, 0.0, -1.0, 1.0, clip_low=0.2, clip_high=0.28)
        self.assertAlmostEqual(-loss, -1.4918, places=3)  # 未被裁：ratio*adv 更大
        self.assertEqual(st["clip_frac"], 0.0)

    def test_tis_detaches(self):
        tis = torch.tensor([[2.0]], requires_grad=True)
        logp = torch.tensor([[-1.0]], requires_grad=True)
        loss, _ = O.grpo_loss(logp, torch.tensor([[-1.0]]), torch.tensor([[1.0]]),
                              torch.tensor([[1.0]]), O.GrpoCfg(tis_cap=2.0),
                              tis=tis.clone().requires_grad_(True))
        loss.backward()
        self.assertIsNone(tis.grad)          # 常数缩放，不回传
        self.assertIsNotNone(logp.grad)

    def test_tis_cap_is_applied(self):
        tis = torch.tensor([[10.0]])
        old, adv, mask = torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[1.0]])
        zero = torch.tensor([[0.0]])
        capped, st = O.grpo_loss(zero, old, adv, mask, O.GrpoCfg(tis_cap=2.0), tis=tis)
        uncapped, _ = O.grpo_loss(zero, old, adv, mask, O.GrpoCfg(tis_cap=0.0), tis=tis)
        self.assertAlmostEqual(-float(capped), 2.0, places=6)     # 权重被压到 cap
        self.assertAlmostEqual(-float(uncapped), 1.0, places=6)   # 关掉就是全 1
        self.assertAlmostEqual(st["tis"], 2.0, places=6)

    def test_dapo_differs_from_length_norm(self):
        # 行0：1 个 token，adv=1；行1：4 个 token，adv=0
        logp = torch.zeros(2, 4)
        old = torch.zeros(2, 4)
        adv = torch.zeros(2, 4)
        adv[0, 0] = 1.0
        mask = torch.zeros(2, 4)
        mask[0, 0] = 1.0
        mask[1, :] = 1.0
        tok, _ = O.grpo_loss(logp, old, adv, mask, O.GrpoCfg(norm="token"))
        seq, _ = O.grpo_loss(logp, old, adv, mask, O.GrpoCfg(norm="seq"))
        self.assertAlmostEqual(-float(tok), 0.2, places=6)   # 1/5
        self.assertAlmostEqual(-float(seq), 0.5, places=6)   # (1/1)/2

    def test_dead_row_does_not_dilute_seq_norm(self):
        logp, old = torch.zeros(2, 2), torch.zeros(2, 2)
        adv = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        seq, st = O.grpo_loss(logp, old, adv, mask, O.GrpoCfg(norm="seq"))
        self.assertAlmostEqual(-float(seq), 1.0, places=6)   # 只除活着的行
        self.assertEqual(st["tokens"], 2)

    def test_stats_count_clipped_tokens(self):
        logp = torch.tensor([[0.9, 0.0]])       # ratio = e^0.9 = 2.46，两侧都超
        old = torch.zeros(1, 2)
        adv = torch.ones(1, 2)
        _, st = O.grpo_loss(logp, old, adv, torch.ones(1, 2), O.GrpoCfg())
        # ratio = [e^0.9=2.46, e^0=1.0]：第一格超上界，第二格没动 ⇒ 恰好一半被裁
        self.assertAlmostEqual(st["clip_frac"], 0.5, places=6)


class CfgValidation(unittest.TestCase):
    def test_rejects_bad_norm(self):
        with self.assertRaises(ValueError):
            O.GrpoCfg(norm="drgrpo")

    def test_rejects_zero_clip(self):
        with self.assertRaises(ValueError):
            O.GrpoCfg(clip_low=0.0)

    def test_rejects_tis_cap_below_one(self):
        # cap<1 会连「同分布」的 token 一起压权重，把梯度整体缩小 —— 那是 lr 的活
        with self.assertRaises(ValueError):
            O.GrpoCfg(tis_cap=0.5)


class RolloutMask(unittest.TestCase):
    def test_lengths(self):
        m = O.grpo_rollout_mask(torch.tensor([1, 3]), 4)
        self.assertEqual(m.tolist(), [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

    def test_truncated_row_is_fully_dropped(self):
        # 被 max_tokens 切断的回答在精确奖励下必是 0 分；当负样本学就是学「没说完不对」
        m = O.grpo_rollout_mask(torch.tensor([2, 2]), 2,
                                truncated=torch.tensor([False, True]))
        self.assertEqual(m.tolist(), [[1.0, 1.0], [0.0, 0.0]])

    def test_existing_mask_is_respected(self):
        m = O.grpo_rollout_mask(torch.tensor([2, 2]), 2, truncated=torch.tensor([True, False]),
                                mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(m.tolist(), [[0.0, 0.0], [1.0, 1.0]])


class Truncation(unittest.TestCase):
    """截断判定：两个条件缺一不可。判错不会报错，只会让梯度学错东西。"""

    def test_exactly_full_but_ends_with_eos_is_complete(self):
        # 正好说满预算、自己收尾 —— 是正常样本，不能当截断丢掉
        self.assertFalse(O.is_truncated([5, 6, 9], 3, 9))

    def test_full_and_no_eos_is_truncated(self):
        self.assertTrue(O.is_truncated([5, 6, 7], 3, 9))

    def test_short_unfinished_is_not_truncated(self):
        # 没到预算就停了，是模型自己选的停，不是被切断
        self.assertFalse(O.is_truncated([5, 6], 8, 9))

    def test_empty_is_not_truncated(self):
        self.assertFalse(O.is_truncated([], 8, 9))

    def test_without_eos_falls_back_to_length(self):
        self.assertTrue(O.is_truncated([5, 6, 7], 3, None))


class NumpyIsTorch(unittest.TestCase):
    """零阶（引擎）那条线只拿得到 numpy/float，反传线拿 torch。

    同一个损失函数必须给同一个数 —— 否则「同算法换路线」只是句口号，两条线的日志
    没法比。这里逐项对拍；`std` 的 ddof 是唯一一个两库默认值不同的地方（见 _std1）。
    """

    def test_ce_matches(self):
        logp = [-0.3, -1.2, -2.5, -0.7]
        self.assertAlmostEqual(float(O.sft_loss(torch.tensor(logp))),
                               float(O.sft_loss(np.array(logp))), places=6)

    def test_dpo_matches_including_saturation(self):
        for margin in (-40.0, -3.0, 0.0, 3.0, 40.0):
            got = float(O.dpo_loss(np.float64(margin), 0.1))
            want = float(O.dpo_loss(torch.tensor(margin), 0.1))
            self.assertAlmostEqual(got, want, places=5)

    def test_group_advantages_uses_the_same_std(self):
        rewards = [1.0, 0.0, 0.0, 1.0, 0.2, 0.4]
        a_np, u_np, _ = O.group_advantages(np.array(rewards), 2)
        a_t, u_t, _ = O.group_advantages(torch.tensor(rewards), 2)
        self.assertEqual((u_np, u_t), (3, 3))
        for got, want in zip(a_np.tolist(), a_t.tolist()):
            self.assertAlmostEqual(got, want, places=4)

    def test_grpo_loss_matches(self):
        lp = [[-0.1, -0.5, -1.0], [-0.4, -0.2, -0.9]]
        old = [[-0.2, -0.4, -1.1], [-0.3, -0.3, -0.8]]
        adv = [[1.0, -1.0, 0.5], [-0.5, 1.0, -1.0]]
        mask = [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
        got, st_np = O.grpo_loss(np.array(lp), np.array(old), np.array(adv), np.array(mask))
        want, st_t = O.grpo_loss(torch.tensor(lp), torch.tensor(old),
                                 torch.tensor(adv), torch.tensor(mask))
        self.assertAlmostEqual(float(got), float(want), places=6)
        for k in ("ratio", "clip_frac", "tis", "tokens", "adv_abs"):
            self.assertAlmostEqual(st_np[k], st_t[k], places=5, msg=k)


if __name__ == "__main__":
    unittest.main()
