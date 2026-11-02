"""把 HF 的融合专家权重量化成 NF4，前向时只解当前命中的专家。

HF 的 `Qwen3_5MoeExperts` 把 256 个专家放在两个 3D Parameter 里：
  gate_up_proj (E, 2*I, H)、down_proj (E, H, I)
bitsandbytes 只认 nn.Linear，这两个张量永远是 bf16 —— 实测 35B 因此要 62.6 GiB。

这里的 forward 与 HF 逐行等价（同一个 for expert_idx 循环），唯一的区别是
权重按专家切片现场反量化：命中几个专家就只解几个，每个 2M 参数 = 4MB，
前向峰值不随专家总数增长。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import nf4


class QuantExperts(nn.Module):
    """NF4 融合专家。权重不进 state_dict，只持久化 LoRA。"""

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int, act_fn, blocksize: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.blocksize = blocksize
        # 普通属性而非 buffer：避免 model.to(dtype) 把 uint8 包体转成 bf16
        self.gu_packed: torch.Tensor | None = None
        self.gu_absmax: torch.Tensor | None = None
        self.dn_packed: torch.Tensor | None = None
        self.dn_absmax: torch.Tensor | None = None

    @classmethod
    def from_hf(cls, mod: nn.Module) -> "QuantExperts":
        return cls(mod.num_experts, mod.hidden_dim, mod.intermediate_dim, mod.act_fn)

    # --- 存储 ---
    def shapes(self) -> dict[str, tuple[int, ...]]:
        gu = (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        dn = (self.num_experts, self.hidden_dim, self.intermediate_dim)
        return {"gu": gu, "dn": dn}

    def alloc(self, device: torch.device) -> None:
        for key, shape in self.shapes().items():
            setattr(self, f"{key}_packed", torch.empty(nf4.packed_shape(shape, self.blocksize), dtype=torch.uint8, device=device))
            setattr(self, f"{key}_absmax", torch.empty(nf4.absmax_shape(shape, self.blocksize), dtype=torch.bfloat16, device=device))

    @torch.no_grad()
    def load_fp(self, key: str, weight: torch.Tensor) -> None:
        """把 bf16 的 (E, O, I) 量化进键为 key 的槽位。"""
        shape = self.shapes()[key]
        assert tuple(weight.shape) == shape, (key, tuple(weight.shape), shape)
        packed, absmax = nf4.quantize(weight, self.blocksize)
        getattr(self, f"{key}_packed").copy_(packed)
        getattr(self, f"{key}_absmax").copy_(absmax)

    def nbytes(self) -> int:
        return sum(
            getattr(self, f"{k}_{s}").numel() * getattr(self, f"{k}_{s}").element_size()
            for k in ("gu", "dn")
            for s in ("packed", "absmax")
        )

    # --- 计算 ---
    def _weight(self, key: str, e: int) -> torch.Tensor:
        shape = self.shapes()[key][1:]
        return nf4.dequantize(
            getattr(self, f"{key}_packed")[e],
            getattr(self, f"{key}_absmax")[e],
            shape,
            self.blocksize,
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self._weight("gu", int(expert_idx))).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self._weight("dn", int(expert_idx)))
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
