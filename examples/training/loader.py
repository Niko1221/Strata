"""完整加载 Qwen3.6-35B-A3B：多模态 + MTP + 4bit 专家。

为什么不能只用 `Qwen3_5MoeForCausalLM`：
  transformers 对 `^mtp.*` 和 `^model.visual.*` 都列在
  `_keys_to_ignore_on_load_unexpected` 里 —— 它会**静默丢弃**这两部分。
  丢掉的后果不是省显存（vision 0.89GB + mtp 1.69GB，只占 3.6%），
  而是训出来的权重回不去：引擎的 `--vision` / `--mtp` 直接失效，
  多模态与投机解码这两个能力在训练产物里凭空消失。

所以这里：
  - vision  → HF 自带 `Qwen3_5MoeVisionModel`，原样保留（冻结）；
  - mtp     → HF 没有实现，按权重的 19 个键自己造模块树（冻结，但其专家也走 4bit）；
  - lang    → 40 层里 30 层 linear_attention + 10 层 full_attention，专家权重量化成 NF4。

流式加载：WSL 只有 31GB 内存，67GB bf16 不可能整体读进内存，所以按 shard
逐个 tensor 读 → 搬到 GPU → 量化 → 写入 → 释放。
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q36
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeRMSNorm,
)

from qexperts import QuantExperts

EXPERT_SUFFIX = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")


class MTP(nn.Module):
    """DeepSeek 式多 token 预测头：embed 与 hidden 各过一层 norm，拼接后降到 H，
    再走一层全注意力 DecoderLayer，最后 norm。权重键与 `mtp.*` 一一对应。"""

    def __init__(self, tcfg):
        super().__init__()
        cfg = copy.deepcopy(tcfg)
        cfg.layer_types = ["full_attention"]  # mtp 层是全注意力（有 q_norm/k_norm）
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, 0)])
        self.norm = Qwen3_5MoeRMSNorm(h, eps=eps)


def _swap_experts(module: nn.Module, where: str, out: dict[str, QuantExperts]) -> None:
    for name, child in module.named_children():
        path = f"{where}.{name}" if where else name
        if type(child).__name__ == "Qwen3_5MoeExperts":
            q = QuantExperts.from_hf(child)
            if where:
                parent = module
                setattr(parent, name, q)
            out[path] = q
        else:
            _swap_experts(child, path, out)


def build_skeleton(model_dir: str) -> tuple[nn.Module, dict[str, QuantExperts]]:
    """meta 设备上的完整模型骨架 + 专家槽位表。不分配任何真实显存。"""
    cfg = AutoConfig.from_pretrained(model_dir)
    with init_empty_weights():
        model = Qwen3_5MoeForConditionalGeneration(cfg)
        model.mtp = MTP(cfg.text_config)
    experts: dict[str, QuantExperts] = {}
    _swap_experts(model, "", experts)
    return model, experts


def key_index(model_dir: str) -> dict[str, str]:
    with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def materialize(model: nn.Module, experts: dict[str, QuantExperts], device: torch.device, dtype: torch.dtype) -> None:
    """把 meta 骨架变成真实张量；专家槽位单独分配（uint8 包体不能被 .to(dtype) 碰）。"""
    model.to_empty(device=device)
    model.to(dtype)
    for q in experts.values():
        q.alloc(device)


def stream_load(
    model: nn.Module,
    experts: dict[str, QuantExperts],
    model_dir: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    log=print,
) -> dict[str, int]:
    weight_map = key_index(model_dir)
    params = dict(model.named_parameters())
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k, shard in weight_map.items():
        by_shard[shard].append(k)

    consumed: list[str] = []
    t0 = time.time()
    hashed = 0
    for shard in sorted(by_shard):
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in sorted(by_shard[shard]):
                tensor = f.get_tensor(key)
                target = None
                leaf = key.rsplit(".", 1)[-1]
                if leaf in ("gate_up_proj", "down_proj") and key.rsplit(".", 2)[-2] == "experts":
                    target = experts[key.rsplit(".", 1)[0]]
                    qkey = "gu" if leaf == "gate_up_proj" else "dn"
                if target is not None:
                    target.load_fp(qkey, tensor.to(device))
                else:
                    param = params.get(key)
                    if param is None:
                        raise KeyError(f"checkpoint key with no module slot: {key}")
                    param.data.copy_(tensor.to(device=device, dtype=dtype))
                consumed.append(key)
                hashed += tensor.numel() * tensor.element_size()
                del tensor
        log(f"  {shard}: {len(by_shard[shard]):4d} tensors  {hashed / 1e9:6.1f} GB  {time.time() - t0:5.0f}s")

    missing = sorted(set(weight_map) - set(consumed))
    if missing:
        raise RuntimeError(f"{len(missing)} keys not consumed, e.g. {missing[:3]}")
    return {"tensors": len(consumed), "gib": int(hashed / 1e9), "seconds": int(time.time() - t0)}


def nbytes_of(model: nn.Module, experts: dict[str, QuantExperts]) -> dict[str, float]:
    expert_bytes = sum(q.nbytes() for q in experts.values())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"experts_gib": expert_bytes / 2**30, "other_gib": param_bytes / 2**30, "total_gib": (expert_bytes + param_bytes) / 2**30}
