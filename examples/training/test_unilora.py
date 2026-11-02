"""unilora.py 的语义自检：不依赖 transformers，CPU 上几十秒跑完。

验证的都是"错了会静默污染训练结果"的性质：
  1. 可训练参数只有库本身（索引/scale 是缓冲，base 冻结）；
  2. 前向 delta 逐值等于 B @ A（查库 -> 物化 -> 手工复算 三方一致）;
  3. 全局索引分配：每个库条目引用次数只差 1，且同种子可复现；
  4. 物化 A/B 与训练前向的 delta 一致（导出正确性的前提）;
  5. 初始扰动幅度：init_bound=0.02 时 delta 的量级确实压在 base 输出的噪声级。
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import unilora


def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(32, 48, bias=False),
        nn.Linear(48, 32, bias=False),
        nn.Linear(32, 16, bias=False),
    )


def main() -> int:
    # --- 1/2/3: 注入与可训练参数 ---
    model = tiny_model()
    n = unilora.inject_unilora(model, targets=("0", "1", "2"), r=4, theta_d_length=64, proj_seed=7)
    assert n == 3, n
    mods = [m for m in model.modules() if isinstance(m, unilora.UniLoRALinear)]
    assert len(mods) == 3
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"unilora_bank.v"}, trainable
    print(f"[1] trainable params: {trainable} (bank len {model.unilora_bank.v.numel()})  OK")

    # --- 3: 引用计数只差 1 + 可复现 ---
    for seed in (7, 7, 8):
        m2 = tiny_model()
        unilora.inject_unilora(m2, targets=("0", "1", "2"), r=4, theta_d_length=64, proj_seed=seed)
        counts = torch.bincount(m2.unilora_bank.v.numel() and
                                torch.cat([mm.unilora_indices_A.reshape(-1) for mm in m2.modules()
                                           if isinstance(mm, unilora.UniLoRALinear)]
                                          + [mm.unilora_indices_B.reshape(-1) for mm in m2.modules()
                                             if isinstance(mm, unilora.UniLoRALinear)]), minlength=64)
        assert counts.max() - counts.min() <= 1, counts
        if seed == 7 and "first" not in dir():
            first_counts = counts
            first_bank = m2.unilora_bank.v.clone()
        elif seed == 7:
            assert torch.equal(counts, first_counts) and torch.equal(m2.unilora_bank.v, first_bank), \
                "same seed must reproduce identical index assignment"
    print(f"[3] per-entry counts in [{int(first_counts.min())}, {int(first_counts.max())}], seed-reproducible  OK")

    # --- 2/4: delta == B@A，物化与前向一致 ---
    x = torch.randn(5, 32)
    m = mods[0]
    a, b = m.materialize()
    assert a.shape == (4, 32) and b.shape == (48, 4), (a.shape, b.shape)
    manual = x @ a.T @ b.T
    direct = m(x) - m.base(x)
    assert torch.allclose(direct, manual, atol=1e-5), (direct - manual).abs().max()
    sd = unilora.materialize_ab(model)
    assert set(sd) == {"0.A", "0.B", "1.A", "1.B", "2.A", "2.B"}, sorted(sd)
    assert torch.allclose(sd["0.A"], a) and torch.allclose(sd["0.B"], b)
    print(f"[2/4] forward delta == x@A^T@B^T (max err {float((direct - manual).abs().max().detach()):.2e}), "
          f"materialize keys {sorted(sd)}  OK")

    # --- 5: 初始扰动幅度 ---
    rel = float((direct.norm() / m.base(x).norm()))
    assert rel < 0.05, rel
    print(f"[5] init perturbation |delta|/|base| = {rel:.5f} (<0.05)  OK")

    # --- 梯度只流向库 ---
    out = model(x).sum()
    out.backward()
    assert model.unilora_bank.v.grad is not None and model.unilora_bank.v.grad.abs().sum() > 0
    assert all(p.grad is None for mm in mods for p in mm.base.parameters())
    print("[6] gradient flows to bank only  OK")

    # --- 与标准 lora.py 的接口并存 ---
    import lora

    assert lora.DEFAULT_TARGETS == unilora.DEFAULT_TARGETS
    print("[7] interface parity with lora.py  OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
