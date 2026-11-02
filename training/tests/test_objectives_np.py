"""`objectives.py` 的**零阶侧**数学（numpy 路径），不 import torch。

    python -m unittest test_objectives_np

为什么和 `test_objectives.py` 分开：
  * 零阶线（`feat/train-mezo`）在引擎前向上算损失，拿到的是 **numpy** 数组，
    这条分支不带 torch，那个文件里的用例一条也跑不起来；
  * 那个文件守的是「numpy ↔ torch 数值一致」（对拍要在同时有两条线的分支上跑）。

所以这里只守**零阶真正会走的那条路**的数值语义：
  1. CE：mean / sum / none 三种口径，mask 只丢 prompt 位置
  2. DPO：softplus 写法在饱和区不发散；margin 按答案 token 数归一后与长度无关
  3. 优势只在组内归一化，std=0 的组贡献 0（并且要能报出「这步没梯度」）
  4. GRPO：ratio=1 退化成 −mean(adv)；裁剪**非对称**（负优势不因 ratio 大被抬高）
  5. DAPO（token 口径）与长度归一化（seq 口径）在长短混排上给出不同的数
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
import warnings

import numpy as np

import objectives as O


class SftCe(unittest.TestCase):
    def test_mean_and_sum(self):
        lp = np.array([-2.0, -0.5], dtype=np.float32)
        self.assertAlmostEqual(float(O.sft_loss(lp)), 1.25, places=6)
        self.assertAlmostEqual(float(O.sft_loss(lp, reduction="sum")), 2.5, places=6)

    def test_none_keeps_the_token_axis(self):
        lp = np.array([-2.0, -0.5], dtype=np.float32)
        got = O.sft_loss(lp, reduction="none")
        self.assertEqual(got.shape, (2,))
        np.testing.assert_allclose(got, [2.0, 0.5], rtol=0, atol=1e-6)

    def test_mask_drops_prompt_positions(self):
        lp = np.array([-9.0, -1.0, -3.0], dtype=np.float32)
        m = np.array([False, True, True])
        self.assertAlmostEqual(float(O.sft_loss(lp, m)), 2.0, places=6)

    def test_an_all_masked_row_is_nan_not_zero(self):
        # 全 mask 掉 = 没有可算的 token。静默返回 0 会让日志显示 loss=0 而实际没训。
        lp = np.array([-1.0, -2.0], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = O.sft_loss(lp, np.array([False, False]))
        self.assertTrue(np.isnan(float(v)))


class Dpo(unittest.TestCase):
    def test_matches_the_closed_form(self):
        m = np.array([0.7, -1.3, 2.0], dtype=np.float64)
        got = np.asarray(O.dpo_loss(m, beta=0.1), dtype=np.float64)
        want = np.logaddexp(0.0, -0.1 * m)          # −log σ(x) = softplus(−x)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_saturated_region_stays_finite(self):
        # 序列级 margin 在 122K 对上能到几千；−logsigmoid 会 0/0，softplus 不会
        got = np.asarray(O.dpo_loss(np.array([-5000.0]), beta=0.1), dtype=np.float64)
        self.assertTrue(np.isfinite(got).all())
        self.assertAlmostEqual(float(got[0]), 500.0, places=6)

    def test_margin_token_normalization_is_length_invariant(self):
        # 两侧各按**自己**的答案 token 数归一：短对（4/8 token）与 8 倍长的对
        # （32/64 token，logp 同步放大 8 倍）必须给出同一个 margin
        short = O.dpo_margin(-2.0, -6.0, 0.0, 0.0, c_ntok=4, r_ntok=8)
        long_ = O.dpo_margin(-16.0, -48.0, 0.0, 0.0, c_ntok=32, r_ntok=64)
        self.assertAlmostEqual(float(short), 0.25, places=6)
        self.assertAlmostEqual(float(long_), 0.25, places=6)

    def test_margin_is_chosen_minus_rejected_with_ref(self):
        got = O.dpo_margin(-2.0, -5.0, -2.5, -4.0)      # (0.5) − (−1.0)
        self.assertAlmostEqual(float(got), 1.5, places=6)


class Advantages(unittest.TestCase):
    def test_within_group_only(self):
        # 两组各 3 条，scale=False：优势是组内中心化，不跨组
        adv, used, zero_std = O.group_advantages(np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
                                                 group_size=3, scale=False)
        self.assertEqual(used, 2)
        self.assertFalse(zero_std)
        np.testing.assert_allclose(np.asarray(adv), [-1.0, 0.0, 1.0, -10.0, 0.0, 10.0],
                                   rtol=0, atol=1e-6)

    def test_scale_on_divides_by_unbiased_std(self):
        adv, _, _ = O.group_advantages(np.array([1.0, 2.0, 3.0]), group_size=3)
        sd = np.std([1.0, 2.0, 3.0], ddof=1)            # 无偏（ddof=1），不是两库默认
        np.testing.assert_allclose(np.asarray(adv), [-1.0, 0.0, 1.0], rtol=0, atol=1e-3)
        self.assertGreater(sd, 0.0)

    def test_all_equal_group_contributes_nothing(self):
        adv, used, zero_std = O.group_advantages(np.array([5.0, 5.0, 5.0]), group_size=3)
        self.assertEqual(used, 0)
        self.assertTrue(zero_std)                       # 日志必须能打「这步没梯度」
        np.testing.assert_allclose(np.asarray(adv), [0.0, 0.0, 0.0], rtol=0, atol=0)


class GrpoLoss(unittest.TestCase):
    def _call(self, logp, old, adv, mask=None, **cfg):
        mask = np.ones_like(np.asarray(logp), dtype=np.float32) if mask is None else mask
        return O.grpo_loss(np.asarray(logp, dtype=np.float32),
                           np.asarray(old, dtype=np.float32),
                           np.asarray(adv, dtype=np.float32),
                           np.asarray(mask, dtype=np.float32),
                           O.GrpoCfg(**cfg))

    def test_ratio_one_reduces_to_minus_mean_adv(self):
        loss, st = self._call([-1.0, -1.0], [-1.0, -1.0], [2.0, 0.0])
        self.assertAlmostEqual(float(loss), -1.0, places=6)
        self.assertAlmostEqual(st["ratio"], 1.0, places=6)
        self.assertEqual(st["tokens"], 2)

    def test_mask_excludes_prompt_tokens(self):
        # 第一个位置是 prompt：它的 logp 与 adv 都不该进分母
        loss, st = self._call([-9.0, -1.0], [-9.0, -1.0], [100.0, 2.0],
                              mask=np.array([0.0, 1.0]))
        self.assertAlmostEqual(float(loss), -2.0, places=6)
        self.assertEqual(st["tokens"], 1)

    def test_positive_advantage_is_clipped(self):
        # ratio=2.0 超过 1+0.2；正优势取 min ⇒ 被裁到 1.2
        loss, st = self._call([np.log(2.0)], [0.0], [1.0])
        self.assertAlmostEqual(float(loss), -1.2, places=5)
        self.assertAlmostEqual(st["clip_frac"], 1.0, places=6)

    def test_negative_advantage_is_not_clipped_upward(self):
        # 负优势下 unclipped=−2 比 clipped=−1.2 更小 ⇒ min 取前者，不算裁剪
        loss, st = self._call([np.log(2.0)], [0.0], [-1.0])
        self.assertAlmostEqual(float(loss), 2.0, places=5)
        self.assertAlmostEqual(st["clip_frac"], 0.0, places=6)

    def test_dapo_and_seq_norm_differ_on_mixed_lengths(self):
        # 两行：一条 1 个有效 token、优势 1；一条 3 个有效 token、优势 0。
        # token 口径 = −1/4；seq 口径 = −((1/1) + (0/3))/2 = −1/2
        logp = np.zeros((2, 3), dtype=np.float32)
        adv = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        mask = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        dapo, st = O.grpo_loss(logp, logp, adv, mask, O.GrpoCfg(norm="token"))
        seq, _ = O.grpo_loss(logp, logp, adv, mask, O.GrpoCfg(norm="seq"))
        self.assertEqual(st["tokens"], 4)
        self.assertAlmostEqual(float(dapo), -0.25, places=6)
        self.assertAlmostEqual(float(seq), -0.5, places=6)


class RolloutMask(unittest.TestCase):
    def test_truncated_rows_get_no_gradient(self):
        m = O.grpo_rollout_mask(np.array([2, 3]), max_len=4,
                                truncated=np.array([True, False]))
        np.testing.assert_allclose(np.asarray(m),
                                   [[0, 0, 0, 0], [1, 1, 1, 0]], rtol=0, atol=0)

    def test_is_truncated_needs_a_full_budget_and_no_eos(self):
        self.assertTrue(O.is_truncated([5, 6, 7], max_new=3, eos_id=9))
        self.assertFalse(O.is_truncated([5, 6, 9], max_new=3, eos_id=9))   # 自己收尾
        self.assertFalse(O.is_truncated([5, 6], max_new=3, eos_id=9))      # 没用满预算


if __name__ == "__main__":
    unittest.main()
