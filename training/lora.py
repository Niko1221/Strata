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
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0,
                 init_seed: int = 0):
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
        # A 的初始化必须用**局部 Generator**，不能吃全局 RNG。
        # 2026-11-07 真机实测：同一 `--seed 0`、同一数据、同一份代码连训两次，
        # 权重 sha256 不同、相对 L2 差 1.41（≈ 权重本身量级），根因就在这里 ——
        # `nn.init.kaiming_uniform_` 用的是 torch 全局 RNG，而训练入口只固定了数据
        # 顺序的那个 Generator（`torch.Generator().manual_seed(args.seed)`），从没
        # `torch.manual_seed` 过；PyTorch 进程启动时用系统熵给全局 RNG 播种，于是每次
        # 的 A 都是不同的随机投影，整条训练轨迹随之不同。固定全局 RNG 只是给调用方
        # 打补丁，源头在构造 A 的这一行：换成显式 seed 的局部生成器，任何调用方都确定。
        #
        # kaiming_uniform_(a=sqrt(5)) 的闭式界：gain=1/sqrt(3)、fan_in=A.size(1)，
        # bound = sqrt(3)*gain/sqrt(fan_in) = 1/sqrt(fan_in)。下面与它逐值等价
        # （tests/test_curriculum.py 里对拍 torch.nn.init.kaiming_uniform_）。
        bound = 1.0 / math.sqrt(self.A.size(1))
        gen = torch.Generator(device=dev).manual_seed(int(init_seed))
        with torch.no_grad():        # nn.init.* 内部也是这样绕开 leaf Variable 的 in-place 限制
            self.A.uniform_(-bound, bound, generator=gen)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.A), self.B)
        return out + delta.to(out.dtype) * self.scale

    def extra_repr(self) -> str:
        return f"r={self.r}, scale={self.scale}, base={tuple(self.base.weight.shape)}"


def inject_lora(model: nn.Module, targets: tuple[str, ...] = DEFAULT_TARGETS, r: int = 16, alpha: int = 32,
                dropout: float = 0.0, prefix: str = "", seed: int = 0) -> int:
    """就地替换。返回替换个数。prefix 可限制只替换某个子树（如 'mtp.'）。

    `seed` 只决定 A 的随机初始化（按模块序号 `seed + n` 逐层展开），不碰全局 RNG；
    传同一个 seed 就得到逐张量相同的初始 adapter，A/B 差分实验才成立。
    """
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
        setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout,
                                          init_seed=int(seed) + n))
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


def load_lora_state(model: nn.Module, path: str) -> int:
    """把上一阶段存下的 adapter 装回来（课程续训：身份 → 数据集）。

    只认 `lora_state_dict` 存出来的那种（键与本次 inject 的完全同名），所以不做命名映射。
    **形状/键对不上就报错，不静默跳过**：rank、alpha、targets 任一处变了，那就是另一份
    adapter，悄悄 load 一半会让 D 阶段的「身份掉了多少」变得没法解释。

    返回载入的张量个数（= 当前模型里 LoRA 键的个数）。
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"init-adapter: {path} 不是 state_dict")
    cur = lora_state_dict(model)
    if not cur:
        raise SystemExit("init-adapter: 模型里没有 LoRA 键（先 inject_lora 再载入）")
    miss = [k for k in cur if k not in sd]
    extra = [k for k in sd if k not in cur]
    bad = [(k, tuple(sd[k].shape), tuple(cur[k].shape)) for k in cur
           if k in sd and tuple(sd[k].shape) != tuple(cur[k].shape)]
    if miss or extra or bad:
        raise SystemExit(
            "init-adapter 对不上：缺 %d 键、多 %d 键、形状不符 %d 个"
            "（rank / alpha / targets / lora_method 与存的那份不同？）\n  缺：%s\n  形状：%s"
            % (len(miss), len(extra), len(bad), miss[:3], bad[:3]))
    model.load_state_dict({k: sd[k] for k in cur}, strict=False)
    return len(cur)
