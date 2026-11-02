"""最小 LoRA：只在冻结的 Linear 旁边挂一对低秩矩阵，不动原权重。

不依赖 peft，是因为要挂的地方不全是标准 `nn.Linear`：
Qwen3.5-MoE 的线性注意力（`in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b`）
和共享专家（`gate_proj` / `up_proj` / `down_proj`）都得能单独选。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 默认挂载点：10 层全注意力 + 30 层线性注意力的投影，加共享专家
DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        for p in base.parameters():
            p._no_weight_decay = True
        self.r = r
        self.scale = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.A), self.B)
        return out + delta.to(out.dtype) * self.scale

    def extra_repr(self) -> str:
        return f"r={self.r}, scale={self.scale}, base={tuple(self.base.weight.shape)}"


def inject_lora(model: nn.Module, targets: tuple[str, ...] = DEFAULT_TARGETS, r: int = 16, alpha: int = 32,
                dropout: float = 0.0, prefix: str = "") -> int:
    """就地替换。返回替换个数。prefix 可限制只替换某个子树（如 'mtp.'）。"""
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        if prefix and not name.startswith(prefix):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        n += 1
    return n


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def trainable_summary(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train, "pct": 100.0 * train / max(total, 1)}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ".A" == k[-2:] or ".B" == k[-2:]}
