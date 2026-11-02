"""把 Qwen4ExpTextQSAIndexer.forward 的 batch×seq_length Python 循环向量化。

## 为什么

transformers 的 QSA 索引器（`modeling_qwen4_exp.py:729`）是：

```python
for batch_idx in range(batch_size):
    for query_idx in range(seq_length):
        local_visible_indices = torch.nonzero(visible_token_indices[b, 0, q], ...)
        num_complete_blocks = local_visible_indices.shape[-1] // compress_ratio
        ...
        key_groups = raw_keys[b].index_select(0, block_token_indices.flatten())
        pooled_keys = self.k_layernorm(key_groups.view(...).float().mean(dim=1).to(dtype))
        block_key_states = apply_rotary_pos_emb(pooled_keys.unsqueeze(1), cos[group_starts], sin[group_starts])
        scores = torch.relu(q[b, q].float() @ block_key_states.float().T).sum(-1) / sqrt(D)
        selected_block_indices = scores.topk(...).indices
```

两个致命点：

1. `pooled_keys / k_layernorm / rope` 只依赖 **block**，与 query 无关，却每个 query 重算
   一遍 —— T=2048 就是 2047 倍冗余。
2. `torch.nonzero(...)` 后读 `.shape[-1]`、`selected_tokens.numel()` 都是 **DtoH 同步**。
   实测 T=2048 一步有 24,661 次 `Memcpy DtoH (Device -> Pinned)` 与 24,763 次
   `cudaStreamSynchronize`，正好是 12 个 QSA 层 × 2047 query。每次同步都要排空前
   面的 kernel，GPU 因此只有 5.3 s 忙 / 17.4 s 墙钟（74% 空转）。

## 做了什么

纯 causal 掩码下 `visible(q) = [0, 1, ..., q]`，于是
`block_token_indices = [[0..R-1], [R..2R-1], ...]` 对所有 query 完全相同：
block 池化、k_layernorm、rope **只算一次**；分数一次 `einsum` + 一次 `topk`；
最终掩码用 gather/scatter 拼出来。语义与原循环一致（同一批算子，只是不再逐 query 重算）。

## 边界

只在「无 past_key_values、kv_length == seq_length、掩码恰好是下三角全 True」时启用；
其它情况（生成、padding、图像变长）退回原实现，不做猜测。数值上唯一的差别是
`topk` 在 **等值** (relu 出来的 0) 上的 tie-break 可能与逐 query 调用不同 ——
`torch.topk` 官方不保证稳定序，逐 query 本身也没有跨长度的一致性。
`STRATA_QSA_CHECK=1` 会把两条路径的掩码逐元素比一次并打印差异比例。
"""
from __future__ import annotations

import os

import torch


def _apply(mod_cls, log=print) -> bool:
    if getattr(mod_cls, "_strata_fast", False):
        return True
    orig = mod_cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        if os.environ.get("STRATA_QSA_FAST", "1") == "0" or past_key_values is not None:
            return orig(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        try:
            return _fast(self, hidden_states, position_embeddings, attention_mask)
        except _Fallback as exc:
            if os.environ.get("STRATA_QSA_DEBUG", "0") != "0":
                print(f"[qsa] fallback: {exc}", flush=True)
            return orig(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    mod_cls.forward = forward
    mod_cls._strata_fast = True
    mod_cls._strata_orig_forward = orig
    log("QSA indexer: vectorized (per-query loop removed)")
    return True


class _Fallback(Exception):
    pass


def _fast(self, hidden_states, position_embeddings, attention_mask):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch_size, seq_length, _ = hidden_states.shape
    if batch_size != 1:
        raise _Fallback("batch != 1")
    if attention_mask is None:
        raise _Fallback("no mask")
    kv_length = attention_mask.shape[-1]
    if kv_length != seq_length:
        raise _Fallback(f"kv_length {kv_length} != seq {seq_length}")

    R = self.compress_ratio
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
    q = q.reshape(batch_size, seq_length, -1, self.index_head_dim)
    raw_keys = token_k.reshape(batch_size, seq_length, -1, self.index_head_dim).squeeze(2)
    q = self.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    V = visible[0, 0]                                    # (S, S) 每行 = 该 query 的可见集
    if not torch.equal(V, torch.tril(torch.ones_like(V))):
        raise _Fallback("mask is not plain causal")

    nb_max = seq_length // R
    if nb_max == 0:
        raise _Fallback("no complete block")

    # ---- 1) block 池化：与原实现同一条算子链，只是所有 query 共用一次 ----
    starts = torch.arange(nb_max, device=q.device) * R
    pooled = raw_keys[0][: nb_max * R].reshape(nb_max, R, self.index_head_dim) \
        .float().mean(dim=1).to(raw_keys.dtype)
    pooled = self.k_layernorm(pooled)
    bks = apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts),
        sin=full_sin[0].index_select(0, starts)).squeeze(1)      # (nb, D)

    # ---- 2) 分数：(S, nb, H) 的 dot → relu → 按 head 求和 ----
    dot = torch.einsum("shd,bd->sbh", q[0].float(), bks.float())
    scores = torch.relu(dot).sum(dim=-1) / (self.index_head_dim ** 0.5)   # (S, nb)

    # query i 能用的完整 block 数 = (i+1)//R；其余置 -inf 后再 topk
    i = torch.arange(seq_length, device=q.device)
    nb_i = (i + 1) // R                                        # (S,)
    disallow = torch.arange(nb_max, device=q.device)[None, :] >= nb_i[:, None]
    scores = scores.masked_fill(disallow, float("-inf"))

    k = min(self.block_topk, nb_max)
    top = scores.topk(k, dim=1).indices                        # (S, k)
    # 被选中且确实落在该 query 的可选 block 范围内（topk 会把 -inf 的也返回）
    valid = top < nb_i[:, None]

    # ---- 3) 掩码：选中 block 的全部 token ∪ 尾巴 [nb_i*R, i] ----
    blk = torch.zeros(seq_length, nb_max, dtype=torch.bool, device=q.device)
    blk.scatter_(1, top, valid)
    mask_blocks = blk.repeat_interleave(R, dim=1)               # (S, nb*R)
    tok = torch.arange(seq_length, device=q.device)
    tail = (tok[None, :] >= (nb_i * R)[:, None]) & (tok[None, :] <= i[:, None])
    out = tail.clone()
    out[:, : nb_max * R] |= mask_blocks

    if os.environ.get("STRATA_QSA_CHECK", "0") != "0":
        _check(self, hidden_states, position_embeddings, attention_mask, out)

    mine = out.unsqueeze(0).unsqueeze(1)                       # (1, 1, S, S)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        return torch.where(mine, attention_mask.new_zeros(()), min_dtype)
    return mine


_CHECKS = {"n": 0}


def _check(self, hidden_states, position_embeddings, attention_mask, mine) -> None:
    """与逐 query 原实现逐元素比一次（仅调试用，会把两条路径都跑一遍）。"""
    if _CHECKS["n"] >= 2:
        return
    _CHECKS["n"] += 1
    with torch.no_grad():
        ref = type(self)._strata_orig_forward(
            self, hidden_states, position_embeddings, attention_mask, None)
    ref_b = ref if ref.dtype == torch.bool else ref == 0
    diff = (ref_b != mine.unsqueeze(0).unsqueeze(0)).sum().item()
    tot = mine.numel()
    print(f"[qsa] check: 掩码差异 {diff}/{tot} ({100 * diff / tot:.4f}%)", flush=True)