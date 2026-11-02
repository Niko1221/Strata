"""Uni-LoRA：整个模型只训一个共享向量（Li et al., NeurIPS 2025 Spotlight）。

论文：https://arxiv.org/abs/2506.00799（已进 HF PEFT，`UniLoraConfig`）。
本实现不依赖 peft —— 和 lora.py 同一个理由：挂载点不全是标准 `nn.Linear`
（线性注意力的 `in_proj_*`、共享专家都要能单独选），且这里只需要训练路径。

机制（与上游 PEFT 逐条对齐，见 NLU/peft/src/peft/tuners/unilora/）：

    全模型唯一可训练参数是一个长度 d 的向量库 theta_d（默认 256，U(-bound, bound)）。
    每个被适配层持有两块**固定**索引缓冲：
        indices_A : (r, n_in)   indices_B : (n_out, r)
    前向时查库构造当层的 LoRA 矩阵：
        A = theta_d[indices_A] * scales_A      # (r, n_in)
        B = theta_d[indices_B] * scales_B      # (n_out, r)
        delta = B @ A，输出 = base(x) + x @ A^T @ B^T
    索引由种子生成器做**全局均匀分配**（上游 model.py::generate_index：每个库条目
    出现次数 = floor(总元素/d)，余量随机不重复补给后整体洗牌），保证每个条目被
    引用次数几乎相等；scales = 1/sqrt(count) 做等距归一化（上游 update_norm）。

与标准 LoRA 的两个必须知道的差异：

    1. **初始 delta 不为零**。标准 LoRA 用 B=0 保证第 0 步等价 base；Uni-LoRA 的
       A、B 都从库里查出来，第 0 步就是一个小的随机扰动。论文靠极小的 init bound
       （默认 0.02）+ 等距归一化把它压到不影响 base 行为的程度，这是设计的一部分，
       不要"修"成 B=0 —— 库共享意味着没有哪一层能独立归零。
    2. **没有 alpha/rank 缩放**。上游 forward 里没有 s 因子，幅度全部由 scales 和
       库本身承担。导出回引擎时把 adapter 的 alpha 设成等于 rank（s=1），delta 才
       逐值等于训练时的 B@A（见 lora_to_gguf.py）。

导出：训练完后用 `materialize_ab()` 把每层 A/B 物化成稠密矩阵，键名与 lora.py 的
state_dict 约定一致（`<module>.A` / `<module>.B`），所以 lora_to_gguf.py 不用改。

显存意义要如实说：Uni-LoRA 省的是**可训练参数侧**（梯度 + AdamW 状态 + 存档），
不省 base 权重。对 35B 级别，标准 LoRA(r=16) 的优化器状态本来就只有几百 MiB，
这里省得有限；参数量优势真正兑现是在超大模型 / 超多 adapter 存档的场景。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import DEFAULT_TARGETS, freeze_all  # noqa: F401  (re-export for sft/grpo 接口一致)


class UniLoRABank(nn.Module):
    """全模型共享的向量库。挂在模型根上（`model.unilora_bank`），只含一个参数。"""

    def __init__(self, length: int, init_bound: float):
        super().__init__()
        self.init_bound = init_bound
        self.v = nn.Parameter(torch.empty(length))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.v.uniform_(-self.init_bound, self.init_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # 容器模块：只持有共享参数，不参与计算图

    def extra_repr(self) -> str:
        return f"length={self.v.numel()}, init_bound={self.init_bound}"


class UniLoRALinear(nn.Module):
    """base Linear + 查库式 LoRA。索引/scale 是固定缓冲，库里只有一个参数可训。"""

    def __init__(self, base: nn.Linear, bank: UniLoRABank, r: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        for p in base.parameters():
            p._no_weight_decay = True
        # 不把 bank 注册成本层子模块：普通实例属性即可。注册成子模块会让
        # named_parameters 的去重命中第一个遍历路径（如 `0.bank.v`），键名随层序漂移；
        # 用普通属性后，唯一命名就是模型根上的 `unilora_bank.v`。
        object.__setattr__(self, "bank", bank)
        self.r = r
        # 先放占位索引（形状正确、内容随机），inject_unilora 第二阶段做全局重排
        self.unilora_indices_A = torch.randint(0, bank.v.numel(), (r, base.in_features))
        self.unilora_indices_B = torch.randint(0, bank.v.numel(), (base.out_features, r))
        self.unilora_scales_A = torch.ones(r, base.in_features)
        self.unilora_scales_B = torch.ones(base.out_features, r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _ab(self) -> tuple[torch.Tensor, torch.Tensor]:
        """A = 库[indices_A] * scales_A，B 同理。gather 对 bank.v 可导。"""
        v = self.bank.v
        a = v[self.unilora_indices_A] * self.unilora_scales_A.to(v.dtype)
        b = v[self.unilora_indices_B] * self.unilora_scales_B.to(v.dtype)
        return a, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        a, b = self._ab()
        delta = F.linear(F.linear(self.drop(x), a.to(x.dtype)), b.to(x.dtype))
        return out + delta.to(out.dtype)

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """物化成稠密 A (r, n_in) / B (n_out, r)，供导出（键名对齐 lora.py 约定）。"""
        with torch.no_grad():
            return self._ab()

    def extra_repr(self) -> str:
        return f"r={self.r}, base={tuple(self.base.weight.shape)}"


def _assign_global_indices(modules: list[UniLoRALinear], theta_d_length: int, proj_seed: int) -> None:
    """上游 UniLoRAModel.__init__ 的全局索引分配：每个库条目引用次数尽量相等。

    numpy default_rng + 同样的 repeat/choice/shuffle 流程，保证与上游同种子同分布。
    """
    total = sum(m.unilora_indices_A.numel() + m.unilora_indices_B.numel() for m in modules)
    rng = np.random.default_rng(proj_seed)
    base_count, remaining = divmod(total, theta_d_length)
    data = np.repeat(np.arange(theta_d_length), base_count)
    if remaining:
        data = np.concatenate([data, rng.choice(theta_d_length, size=remaining, replace=False)])
    rng.shuffle(data)
    counts = np.bincount(data, minlength=theta_d_length)
    sqrt_inv = 1.0 / np.sqrt(counts.astype(np.float64))

    ptr = 0
    all_idx: list[np.ndarray] = []
    for m in modules:
        for which in ("A", "B"):
            idx = getattr(m, f"unilora_indices_{which}")
            n = idx.numel()
            chunk = torch.from_numpy(data[ptr:ptr + n].copy()).view_as(idx)
            setattr(m, f"unilora_indices_{which}", chunk)
            all_idx.append(chunk.reshape(-1).numpy())
            ptr += n
    assert ptr == total
    # 等距归一化：每个索引按其全局引用次数 1/sqrt(count) 缩放（上游 update_norm）
    for m in modules:
        for which in ("A", "B"):
            idx_t = getattr(m, f"unilora_indices_{which}")
            idx = idx_t.reshape(-1).numpy()
            scales = torch.from_numpy(sqrt_inv[idx].reshape(idx_t.shape).copy()).float()
            setattr(m, f"unilora_scales_{which}", scales)


def inject_unilora(
    model: nn.Module,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    r: int = 4,
    theta_d_length: int = 256,
    proj_seed: int = 42,
    init_bound: float = 0.02,
    dropout: float = 0.0,
    prefix: str = "",
) -> int:
    """就地替换。返回替换个数。两阶段：先铺层，再做全局索引分配。"""
    if theta_d_length <= 0:
        raise ValueError("theta_d_length must be positive")
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
        setattr(parent, child, UniLoRALinear(module, _ensure_bank(model, theta_d_length, init_bound), r, dropout))
        n += 1
    if n:
        mods = [m for m in model.modules() if isinstance(m, UniLoRALinear)]
        _assign_global_indices(mods, theta_d_length, proj_seed)
        _move_bank_and_buffers(model)
    return n


def _ensure_bank(model: nn.Module, theta_d_length: int, init_bound: float) -> UniLoRABank:
    bank = getattr(model, "unilora_bank", None)
    if bank is None:
        bank = UniLoRABank(theta_d_length, init_bound)
        model.add_module("unilora_bank", bank)
    elif bank.v.numel() != theta_d_length:
        raise ValueError(f"bank already sized {bank.v.numel()} != {theta_d_length}")
    return bank


def _move_bank_and_buffers(model: nn.Module) -> None:
    """inject 可能发生在模型已上 GPU 之后（sft.py 的顺序正是如此）：
    bank 参数和各层的索引/scale 都建在 CPU，这里统一搬到第一个真实参数的设备。"""
    dev = next((p.device for p in model.parameters() if p.is_floating_point() and p.numel() > 0), None)
    if dev is None:
        return
    bank = getattr(model, "unilora_bank", None)
    if bank is not None:
        bank.to(dev)
    for m in model.modules():
        if isinstance(m, UniLoRALinear):
            m.unilora_indices_A = m.unilora_indices_A.to(dev)
            m.unilora_indices_B = m.unilora_indices_B.to(dev)
            m.unilora_scales_A = m.unilora_scales_A.to(dev)
            m.unilora_scales_B = m.unilora_scales_B.to(dev)


def trainable_summary(model: nn.Module) -> dict[str, float]:
    """与 lora.trainable_summary 同名接口。可训练参数应只有库本身。"""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train, "pct": 100.0 * train / max(total, 1)}


def materialize_ab(model: nn.Module) -> dict[str, torch.Tensor]:
    """物化全部 A/B，键 `<module>.A` / `<module>.B` —— 与 lora_state_dict / lora_to_gguf 兼容。"""
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, UniLoRALinear):
            a, b = module.materialize()
            out[f"{name}.A"] = a.detach().cpu().float()
            out[f"{name}.B"] = b.detach().cpu().float()
    return out


def unilora_ckpt(model: nn.Module, proj_seed: int, theta_d_length: int, init_bound: float) -> dict:
    """恢复训练所需的最小集合：库 + 索引/scale。索引可由种子重建，这里仍显式存，防种子丢失。"""
    bank = model.unilora_bank
    state = {
        "theta_d_length": theta_d_length,
        "proj_seed": proj_seed,
        "init_bound": init_bound,
        "bank": bank.v.detach().cpu(),
        "modules": {},
    }
    for name, m in model.named_modules():
        if isinstance(m, UniLoRALinear):
            state["modules"][name] = {
                "indices_A": m.unilora_indices_A.cpu(),
                "indices_B": m.unilora_indices_B.cpu(),
                "scales_A": m.unilora_scales_A.cpu(),
                "scales_B": m.unilora_scales_B.cpu(),
            }
    return state
