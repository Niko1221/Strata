"""两条更新路线，一个接口：`rule.step(loss_fn) -> metrics`。

为什么要有这个文件：`sft.py`（反传）与 `hf_ref.py`（零阶）各自写了一遍「怎么把 loss
变成参数更新」。两支的**目标函数**已经统一到 `objectives.py` 了，但更新那一半还是两份：
反传那边是 `loss.backward(); opt.step()`，零阶那边是「±ε 扰动两次前向 + 方向导数 + 手写
SGD」。于是「同一算法换路线」要改两处，而日志字段、momentum 语义、fp32 cast 时机各说各话。

统一到哪一层：**`loss_fn` 是唯一接口**。

    loss_fn() -> 标量 Tensor        # 闭包里只有「用当前参数做前向 → objective → 标量」

要不要建图由 rule 决定，loss_fn 自己不关心 —— 这就是两条路线能共用同一份算法的全部原因。
反传 rule 在 `enable_grad` 里调它再 `.backward()`；零阶 rule 在 `no_grad` 里调两次。
**反过来不成立**（不能让零阶假装产出 `.grad`）：
  * 方向导数 `z = (L(θ+εd) − L(θ−εd)) / 2ε` 是**一个标量**，不是逐参数梯度；
  * 塞进 `p.grad` 后 AdamW 的 `v` 会把同一个常数反复累积，更新方向退化成 `d` 的符号，
    那已经不是 MeZO 了 —— 所以零阶只能用 SGD(+momentum)。

`metrics` 的公共键是 `loss`；反传多 `grad_norm`，零阶多 `z` / `loss_minus`。日志因此可以
用同一行格式打两种路线。
"""

from __future__ import annotations

import types

import torch

from lora import LoRALinear


# ---------------------------------------------------------------- 零阶的方向向量

def zo_cast_fp32(model) -> int:
    """LoRA A/B 转 fp32，并给模块绑 dtype 安全的 forward。

    两个原因：
      1) ZO 更新步长 lr~1e-6，bf16 相对精度 ~2^-8，小幅更新会整体舍入丢失；
      2) fp32 A/B × bf16 x 的 `F.linear` 会 dtype 报错，需显式换边。
    """
    n = 0

    def fwd(self, x):
        out = self.base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(self.A.dtype), self.A), self.B)
        return out + delta.to(out.dtype) * self.scale

    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.data = m.A.data.float()
            m.B.data = m.B.data.float()
            m.forward = types.MethodType(fwd, m)
            n += 1
    return n


def zo_direction(params, seed: int) -> list[torch.Tensor]:
    """按 seed 重新生成**一整条**方向 z（与 MeZO 官方实现同构）。

    这里必须用**一个** generator 顺序消费：同一个 seed 只要参数顺序不变，就复现同一条 z，
    于是 `+ε` 与 `−ε` 两次调用拿到的是同一个方向。

    ⚠️ 2026-11-01 修：旧实现把 `torch.manual_seed(seed)` 写在**循环体内**，每个参数都
    重置 RNG ⇒ 所有参数拿到同一个 z 前缀，方向退化成「全参数同向」，方向导数不再是梯度的
    无偏估计。判据见 `test_update_rules.py::test_direction_differs_across_params`
    （两个不同 shape 的参数，前 min(shape) 个元素**不能**相等）。
    """
    if not params:
        return []
    gen = torch.Generator(device=params[0].device)
    gen.manual_seed(int(seed))
    return [torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params]


# ---------------------------------------------------------------- 两条路线

class BackwardRule:
    """标准反传：建图 → backward → 梯度裁剪 → AdamW。"""

    kind = "backward"

    def __init__(self, params, lr: float = 1e-4, clip: float = 1.0,
                 weight_decay: float = 0.0, betas=(0.9, 0.999)):
        self.params = list(params)
        if not any(p.requires_grad for p in self.params):
            raise ValueError(
                "BackwardRule 需要一个 requires_grad=True 的参数（LoRA 注入后才有；"
                "冻结全部参数却走反传，loss 不会带 grad_fn，只会报 'does not require grad'）")
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.clip = clip

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    def backward(self, loss) -> float:
        """只累加梯度、不更新（micro-batch / gradient accumulation 用）。

        逐条 backward 的显存档位与「先把一个 batch 拼成一个大 loss」不同 —— 前者在两条
        之间能释放大部分激活图，这也是 `sft.py` 保留 accumulation 循环的原因。
        """
        loss.backward()
        return float(loss.detach())

    def finish(self) -> dict:
        gn = float(torch.nn.utils.clip_grad_norm_(self.params, self.clip)) if self.clip else float("nan")
        self.opt.step()
        return {"grad_norm": gn}

    def step(self, loss_fn) -> dict:
        self.zero_grad()
        with torch.enable_grad():
            loss = loss_fn()
        self.backward(loss)
        m = self.finish()
        m["loss"] = float(loss.detach())
        return m


class MezoRule:
    """零阶（MeZO）：θ±εd 两次前向取方向导数，SGD(+momentum)。

    `loss_fn` 里**不能**出现 `.backward()` —— 这里整段在 `no_grad` 下跑，图不会存在。

    `queries > 1` 时每步采 K 个独立方向、各测一次方向导数再平均（原 `hf_ref.py`
    的 `--queries`）：每方向 2 次前向，方差降 √K，时间线性上升。
    """

    kind = "zeroth_order"

    def __init__(self, params, lr: float = 3e-6, eps: float = 1e-3,
                 momentum: float = 0.0, seed: int = 0, cast_fp32: bool = True,
                 queries: int = 1, weight_decay: float = 0.0):
        self.params = list(params)
        if cast_fp32:
            for p in self.params:
                p.data = p.data.float()
        self.lr = lr
        self.eps = eps
        self.momentum = momentum
        self.seed = int(seed)
        self.weight_decay = weight_decay
        self.queries = max(1, int(queries))
        self._buf = [torch.zeros_like(p) for p in self.params] if momentum else None
        self.steps = 0

    def _perturb(self, dirs, scale: float) -> None:
        for p, d in zip(self.params, dirs):
            p.data.add_(d, alpha=scale * self.eps)

    def step(self, loss_fn) -> dict:
        self.steps += 1
        eps = self.eps
        base = self.seed + (self.steps - 1) * self.queries
        qs = []                       # [(dirs, 方向导数)]
        loss_plus = 0.0
        loss_minus = 0.0
        for k in range(self.queries):
            dirs = zo_direction(self.params, base + k)
            self._perturb(dirs, +1.0)
            with torch.no_grad():
                lp = float(loss_fn())
            self._perturb(dirs, -2.0)
            with torch.no_grad():
                lm = float(loss_fn())
            self._perturb(dirs, +1.0)          # 复原到 θ
            qs.append((dirs, (lp - lm) / (2.0 * eps)))
            loss_plus += lp
            loss_minus += lm
        z = sum(v for _, v in qs) / self.queries

        with torch.no_grad():
            for i, p in enumerate(self.params):
                # 梯度的无偏估计：每条方向各自乘回去再平均（不是先平均 z 再乘一个方向）
                g = torch.zeros_like(p)
                for dirs, v in qs:
                    g.add_(dirs[i], alpha=v)
                if self.queries > 1:
                    g.div_(self.queries)
                if self._buf is not None:
                    b = self._buf[i]
                    if self.weight_decay > 0:
                        b.mul_(self.weight_decay)
                    b.mul_(self.momentum).add_(g)
                    p.data.add_(b, alpha=-self.lr)
                else:
                    p.data.add_(g, alpha=-self.lr)
        return {"loss": loss_plus / self.queries, "loss_minus": loss_minus / self.queries,
                "z": z, "queries": self.queries}


def make_rule(kind: str, params, *, lr: float, **kw):
    """按名字造 rule，参数原样透传（两条路线的 kw 不通用，多给会立刻 TypeError）。"""
    if kind == "backward":
        return BackwardRule(params, lr=lr, **kw)
    if kind in ("zeroth_order", "mezo"):
        return MezoRule(params, lr=lr, **kw)
    raise ValueError(f"未知路线 {kind!r}（只认 backward / zeroth_order）")
