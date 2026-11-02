"""`update_rules.py` 的 CPU 单测：一个 loss_fn，两条路线。

这里能测的是**接口与正确性**，不是显存/速度（那在 B2/B4 基准里）。三类断言：

  1. 方向向量 z：同 seed 可复现（±ε 必须是同一个 d）、跨参数不同（否则方向退化成全 1）、
     换 seed 就换方向；
  2. 方向导数：线性目标上 `z` 必须等于解析值 `<g, d>`（差一个 ε 量级的舍入）；
  3. 更新：反传与零阶各自在二次目标上靠近最优点；`lr=0` 时参数必须回到原处
     （±ε 扰动忘了复原是零阶最典型的静默错误 —— 它不会报错，只是每步多点位移）。

跑法：`python test_update_rules.py`（不需要 GPU、不需要 transformers）。
"""

import unittest

import torch

import update_rules as U
from update_rules import BackwardRule, MezoRule, make_rule, zo_direction


def quad_fn(w: torch.Tensor, target: float = 3.0):
    """f(w) = (w − target)²，返回闭包（每步读当前参数，这是统一接口的全部要求）。"""
    def fn():
        return ((w - target) ** 2).sum()
    return fn


class TestDirection(unittest.TestCase):
    def test_direction_reproducible(self):
        """同一个 seed 两次必须给出逐位相同的方向 —— `+ε` 与 `−ε` 靠这一条才对齐。"""
        params = [torch.zeros(4), torch.zeros(3, 2)]
        a = zo_direction(params, seed=7)
        b = zo_direction(params, seed=7)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_direction_differs_across_params(self):
        """不同参数必须拿到不同的 z。

        回归 2026-11-01 的真 bug：旧实现把 `torch.manual_seed(seed)` 写在循环体内，
        每个参数都重置 RNG，于是所有参数拿到**同一个** z 前缀 —— 方向变成「全参数同向」，
        方向导数不再是梯度的无偏估计，且不会报任何错。
        """
        params = [torch.zeros(8), torch.zeros(4)]
        zs = zo_direction(params, seed=0)
        overlap = min(zs[0].numel(), zs[1].numel())
        self.assertFalse(torch.equal(zs[0][:overlap], zs[1][:overlap]),
                         "两个参数的方向相同 ⇒ 每个参数都在被同一个 seed 重新播种")

    def test_direction_changes_with_seed(self):
        params = [torch.zeros(8)]
        self.assertFalse(torch.equal(zo_direction(params, 0)[0], zo_direction(params, 1)[0]))

    def test_direction_dtype_and_device_follow_param(self):
        params = [torch.zeros(4, dtype=torch.float32)]
        self.assertEqual(zo_direction(params, 0)[0].dtype, torch.float32)


class TestMezoDirectionalDerivative(unittest.TestCase):
    def test_z_matches_analytic_on_linear(self):
        """线性目标 f(w)=<g,w>：方向导数解析值就是 <g,d>。"""
        w = torch.zeros(6)
        g = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -0.25])
        rule = MezoRule([w], lr=0.0, eps=1e-3, seed=0)
        d = zo_direction([w], seed=0)[0]              # 第 1 步用 seed + 0（queries=1）
        want = float((g * d).sum())
        got = rule.step(lambda: (g * w).sum())["z"]
        self.assertAlmostEqual(got, want, places=3)

    def test_queries_average_matches_analytic(self):
        """queries=K：z 是 K 个方向导数的平均，解析值 = mean_j <g, d_j>。

        这条同时锁住 seed 的排布（base = seed + (step−1)·queries）：K 个方向必须**不同**，
        否则「多方向降方差」只是把同一次测量重复 K 遍。
        """
        w = torch.zeros(6)
        g = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -0.25])
        rule = MezoRule([w], lr=0.0, eps=1e-3, seed=0, queries=3)
        ds = [zo_direction([w], seed=k)[0] for k in range(3)]
        self.assertFalse(torch.equal(ds[0], ds[1]), "queries>1 时各方向必须不同")
        want = sum(float((g * d).sum()) for d in ds) / 3
        got = rule.step(lambda: (g * w).sum())["z"]
        self.assertAlmostEqual(got, want, places=3)

    def test_lr_zero_restores_params(self):
        """lr=0 时参数必须回到原处：±ε 扰动漏了复原是零阶最典型的静默错误。"""
        w = torch.tensor([0.5, -1.5, 2.0])
        w0 = w.clone()
        rule = MezoRule([w], lr=0.0, eps=1e-3, seed=3)
        for _ in range(3):
            rule.step(quad_fn(w))
        self.assertLess(float((w - w0).abs().max()), 1e-6,
                        f"扰动没复原：最大位移 {float((w - w0).abs().max()):.2e}")

    def test_converges_on_quadratic(self):
        w = torch.zeros(5)
        rule = MezoRule([w], lr=0.05, eps=1e-3, seed=1)
        d0 = float(w.sub(3.0).abs().max())
        for _ in range(400):
            rule.step(quad_fn(w))
        d1 = float(w.sub(3.0).abs().max())
        self.assertLess(d1, d0 * 0.5, f"零阶没靠近最优点：{d0:.3f} → {d1:.3f}")

    def test_momentum_buffer_carries_history(self):
        """momentum: 第二步的 buffer 必须含第一步的项（不是只用当前 g）。

        方向每步重采样，所以「累计位移更大」不成立（试过，会假失败）。
        改为白盒：用 lr=1e-3 与 lr=0 跑同一 seed 的两步，前者第二步的 buffer 若等于
        后者第二步的 g，就说明 momentum 没接上。
        """
        w = torch.zeros(4)
        m = MezoRule([w], lr=1e-3, eps=1e-3, seed=2, momentum=0.9)
        m.step(quad_fn(w))
        b1 = m._buf[0].clone()
        m.step(quad_fn(w))
        b2 = m._buf[0].clone()

        # 第 2 步的「当前 g」：用 lr=0 的同 seed rule 取 z 与 d（参数不动，判据干净）
        ref = MezoRule([torch.zeros(4)], lr=0.0, eps=1e-3, seed=2, momentum=0.0)
        ref.step(quad_fn(ref.params[0]))
        z2 = ref.step(quad_fn(ref.params[0]))["z"]
        g2 = z2 * zo_direction(ref.params, seed=2 + 1)[0]   # 第 2 步 base = seed + 1
        self.assertFalse(torch.allclose(b1, torch.zeros_like(b1)), "第一步 buffer 不该是 0")
        self.assertFalse(torch.allclose(b2, g2, atol=1e-3), "第二步 buffer 只剩当前 g ⇒ momentum 没接上")

    def test_no_grad_left_on_params(self):
        """零阶全程不产生 `.grad`（这正是它显存贴住推理基线的原因）。"""
        w = torch.zeros(3, requires_grad=True)
        rule = MezoRule([w], lr=0.01, eps=1e-3, seed=0)
        rule.step(quad_fn(w))
        self.assertIsNone(w.grad)


class TestBackwardRule(unittest.TestCase):
    def test_moves_toward_target(self):
        w = torch.zeros(4)
        rule = BackwardRule([w.requires_grad_(True)], lr=0.1)
        rule.step(quad_fn(w))
        self.assertTrue(bool((w > 0).all()), "反传第一步应当朝最优点的方向移动")

    def test_reports_grad_norm(self):
        w = torch.zeros(3)
        rule = BackwardRule([w.requires_grad_(True)], lr=0.1)
        m = rule.step(quad_fn(w))
        self.assertIn("grad_norm", m)
        # 每个元素梯度 2·(0−3) = −6，3 个 ⇒ norm = 6√3
        self.assertAlmostEqual(m["grad_norm"], 6.0 * 3 ** 0.5, places=3)

    def test_clip_limits_norm(self):
        """`clip_grad_norm_` 返回的是**裁剪前**的范数（PyTorch 语义），所以两条都要测：
        上报值反映真实梯度大小，参数上的梯度才被裁到 clip。"""
        w = torch.zeros(3, requires_grad=True)
        rule = BackwardRule([w], lr=0.0, clip=0.5)
        m = rule.step(quad_fn(w))
        self.assertAlmostEqual(m["grad_norm"], 6.0 * 3 ** 0.5, places=3)
        self.assertLessEqual(float(w.grad.norm()), 0.5 + 1e-5)


class TestCommonInterface(unittest.TestCase):
    """两条路线共享的契约：`step(loss_fn) -> metrics`，metrics 必须有 `loss`。"""

    def test_both_rules_return_loss(self):
        for rule in (BackwardRule([torch.zeros(2, requires_grad=True)], lr=0.01),
                     MezoRule([torch.zeros(2)], lr=0.01, eps=1e-3, seed=0)):
            w = rule.params[0]
            m = rule.step(quad_fn(w))
            self.assertIn("loss", m)
            self.assertIsInstance(m["loss"], float)

    def test_loss_fn_needs_no_grad_handling(self):
        """同一个闭包，两条路线都能跑：建不建图由 rule 决定，闭包自己不写 no_grad。"""
        for rule in (BackwardRule([torch.zeros(2, requires_grad=True)], lr=0.01),
                     MezoRule([torch.zeros(2)], lr=0.01, eps=1e-3, seed=0)):
            w = rule.params[0]
            rule.step(quad_fn(w))     # 不抛 "element 0 of tensors does not require grad"

    def test_make_rule_dispatch(self):
        self.assertIsInstance(make_rule("backward", [torch.zeros(2, requires_grad=True)], lr=1e-4), BackwardRule)
        self.assertIsInstance(make_rule("zeroth_order", [torch.zeros(2)], lr=1e-6), MezoRule)
        with self.assertRaises(ValueError):
            make_rule("sgd", [torch.zeros(2)], lr=1e-4)

    def test_mezo_casts_to_fp32(self):
        """bf16 下 eps=1e-3 只有 ~3 位有效数字，零阶必须先转 fp32。"""
        w = torch.zeros(4, dtype=torch.bfloat16)
        rule = MezoRule([w], lr=0.0, eps=1e-3, seed=0)
        self.assertEqual(rule.params[0].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
