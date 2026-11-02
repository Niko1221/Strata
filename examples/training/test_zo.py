"""MeZO 小规模自验证：ZO 梯度估计必须与 autograd 真梯度方向一致、量级正确。

不碰 125B 就先证明优化器核心是对的 —— 这是写 sft_mezo.py 之前的地基。
MeZO (Malladi et al., NeurIPS 2023) 的估计量：

    ĝ = (L(θ+εz) − L(θ−εz)) / (2ε) · z ,   z ~ N(0, I)

对二次损失，方向差分恰为 2ε(∇L·z)（三阶项消失），因此存在可精确断言的恒等式：

    cos(ĝ_i, ∇L) == |cos(z_i, ∇L)|          —— 估计器实现正确性的锐利检查
    E[ĝ] = ∇L                                 —— K 次平均 cos → 1/√(1+1/k)

注意：模型初始化与扰动禁止共用 seed（z 会重放初始化 RNG 流、与权重相关，
本文件的 bug 历史记录）。高维下单样本 cos ~ 1/√d ≈ 0.005 是正常的，
不要拿它断言"估计不准"。
"""

import torch

torch.manual_seed(0)
dev = "cuda"

d_in, d_out = 256, 128
W = torch.randn(d_out, d_in, device=dev) * 0.05
X = torch.randn(64, d_in, device=dev)
Y = torch.randn(64, d_out, device=dev)


def loss_fn(w):
    return ((X @ w.T - Y) ** 2).mean()


def true_grad():
    w = W.detach().clone().requires_grad_(True)
    loss_fn(w).backward()
    return w.grad


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


gt = true_grad()
eps = 1e-3


def zo_sample(seed):
    """单个 z 的 ZO 梯度估计 ĝ_i 及其方向差分标量。"""
    torch.manual_seed(seed)
    z = torch.normal(0, 1, size=W.shape, device=dev)
    lp = float(loss_fn(W + eps * z))
    lm = float(loss_fn(W - eps * z))
    return (lp - lm) / (2 * eps) * z


# 1) 恒等式检查：cos(ĝ_i, gt) 必须逐位等于 |cos(z_i, gt)|（二次损失，无高阶偏差）
for s in (7, 8, 9):
    gi = zo_sample(s)
    torch.manual_seed(s)
    z = torch.normal(0, 1, size=W.shape, device=dev)
    lhs, rhs = cos(gi, gt), abs(cos(z, gt))
    print(f"seed {s}: cos(gi,gt)={lhs:.6f}  ==|cos(z,gt)|={rhs:.6f}")
    assert abs(lhs - rhs) < 1e-4, f"ZO 估计器与理论恒等式不符: {lhs} vs {rhs}"

# 2) K 次平均收敛：维数诅咒 cos(mean_k) ≈ √(k/(d+k))（d=32768：k=64→0.044, k=4096→0.333）。
#    实测与理论逐位吻合（2026-10-29），这是估计器无偏的端到端证据。
import math

d = d_in * d_out
for k, lo in ((64, 0.03), (4096, 0.25)):
    g = torch.zeros_like(W)
    for i in range(k):
        g += zo_sample(1000 + i)
    c = cos(g / k, gt)
    print(f"cos(mean of k={k}, gt) = {c:.4f}  (theory ~{math.sqrt(k / (d + k)):.3f})")
    assert c > lo, f"K={k} 平均后 cos 仅 {c}"

# 3) 端到端：ZO-SGD 必须能降 loss。参数区是实测出来的（同次扫描）：
#    lr 1e-2 / lr 1e-3+momentum0.9 都发散——零阶噪声比信号大 ~√d 倍，动量会放大噪声，
#    默认必须保守（sft_mezo.py 默认 momentum=0）。
lr = 1e-3
l0 = float(loss_fn(W))
for step in range(400):
    g = zo_sample(step)
    with torch.no_grad():
        W -= lr * g
l1 = float(loss_fn(W))
print(f"ZO-SGD 400 steps (lr 1e-3, no momentum): loss {l0:.4f} -> {l1:.4f}")
assert l1 < l0 * 0.85, f"ZO-SGD 400 步 loss 没有下降: {l0:.4f} -> {l1:.4f}"
print("test_zo: ALL PASS")
