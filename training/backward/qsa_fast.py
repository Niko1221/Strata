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

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


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


def row_ends(attention_mask: torch.Tensor) -> torch.Tensor:
    """每行最后一个可见位置 `(B, S)`，**不新增 S² 分配**。

    以前这里用 `torch.equal(V, torch.tril(torch.ones_like(V)))` 校验「恰好下三角」，
    S=32768 就是 `ones_like` 1 GiB + `tril` 1 GiB 两个临时张量（S=128K 各 16 GiB，
    直接把「还没轮到 sdpa」变成 OOM）。可见集若是前缀，每行的可见长度就唯一决定
    了它，用一次 row-sum（只读 S² 字节、只写 S 个数）即可。

    返回的 `ends` 还顺便给出窗口偏移：全序列 prefill 是 `ends[j] == j`，
    带 KV cache 的分块 prefill 是 `ends[j] == p + j`（p = 已缓存长度）。
    """
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    if visible.dim() == 4:
        visible = visible[:, 0]
    return visible.sum(-1) - 1


def check_prefix_causal(attention_mask: torch.Tensor) -> torch.Tensor:
    """校验「每行是 [0, p+j] 的前缀」并返回 `ends`，否则抛 `_Fallback`。

    判据是 `ends - ends[:, :1] == arange(S)`：下三角（p=0）、带 cache 的分块
    前缀（p>0）都成立；滑动窗口（行和恒定，减完不递增）与带 padding 的矩形
    掩码都会被挡掉，退回原实现。
    """
    ends = row_ends(attention_mask)
    if ends.dim() != 2:
        raise _Fallback(f"mask rows {tuple(ends.shape)}")
    _, seq_length = ends.shape
    steps = torch.arange(seq_length, device=ends.device).unsqueeze(0).expand_as(ends)
    if not torch.equal(ends - ends[:, :1], steps):
        raise _Fallback("mask is not prefix-causal")
    return ends


def _fast(self, hidden_states, position_embeddings, attention_mask):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch_size, seq_length, _ = hidden_states.shape
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

    # 可见集是「前缀」：全序列 prefill 满足 ends[j] == j。以前这里 `ones_like + tril`
    # 各造一份 S² 临时张量（S=128K 时各 16 GiB），是 128K OOM 的第一块砖，已改成 row-sum。
    ends = check_prefix_causal(attention_mask)                   # (B, S)

    nb_max = seq_length // R
    if nb_max == 0:
        raise _Fallback("no complete block")
    # 分数/topk/掩码的 query 块宽。只影响峰值显存，不影响任何一行的选择结果。
    qchunk = int(os.environ.get("STRATA_QSA_QCHUNK", "4096"))

    # ---- 1) block 池化：与原实现同一条算子链，所有 query、所有行共用一次 ----
    starts = torch.arange(nb_max, device=q.device) * R
    pooled = raw_keys[:, : nb_max * R].reshape(batch_size, nb_max, R, self.index_head_dim) \
        .float().mean(dim=2).to(raw_keys.dtype)
    pooled = self.k_layernorm(pooled)                              # (B, nb, D)
    bks = apply_rotary_pos_emb(
        pooled.unsqueeze(2), cos=full_cos[0].index_select(0, starts),
        sin=full_sin[0].index_select(0, starts)).squeeze(2)        # (B, nb, D)

    # ---- 2) + 3) 按 query 分块算分数、topk、掩码 ----
    # 为什么必须分块：分数张量是 (B, S, nb, H)，nb = S/compress_ratio（4），
    # H = index_n_heads（4）。T=128K 时一次算完是 131072 × 32768 × 4 × 4 B = 68.7 GiB
    # —— 还没轮到 attention 就先 OOM。分块后峰值只与块宽有关（4096 行 ≈ 2.1 GiB），
    # 而 topk 对每一行独立，所以分块不改变任何一行的选择结果。
    i = torch.arange(seq_length, device=q.device)
    nb_i = (ends + 1) // R                                     # (B, S) 每行的完整 block 数
    all_blocks = torch.arange(nb_max, device=q.device)
    k = min(self.block_topk, nb_max)

    # 尾巴 = [nb_i*R, ends]。写成两次 in-place 比较 + 一次与，峰值是 **2 份** (B,S,kv)
    # bool（out + scratch），不是 4 份；S=32K 时每份 1 GiB。
    tok = torch.arange(kv_length, device=q.device)
    out = torch.empty(batch_size, seq_length, kv_length, dtype=torch.bool, device=q.device)
    scratch = torch.empty_like(out)
    torch.le(tok.view(1, 1, -1), ends[:, :, None], out=out)        # tok <= ends
    torch.ge(tok.view(1, 1, -1), (nb_i * R)[:, :, None], out=scratch)   # tok >= nb_i*R
    out.logical_and_(scratch)
    del scratch

    for s0 in range(0, seq_length, qchunk):
        s1 = min(s0 + qchunk, seq_length)
        qc = s1 - s0
        dot = torch.einsum("bqhd,bnd->bqnh", q[:, s0:s1].float(), bks.float())
        scores = torch.relu(dot).sum(dim=-1) / (self.index_head_dim ** 0.5)   # (B, qc, nb)
        # query i 能用的完整 block 数 = (i+1)//R；其余置 -inf 后再 topk
        disallow = all_blocks[None, None, :] >= nb_i[:, s0:s1, None]
        scores = scores.masked_fill(disallow, float("-inf"))

        top = scores.topk(k, dim=2).indices                    # (B, qc, k)
        # 被选中且确实落在该 query 的可选 block 范围内（topk 会把 -inf 的也返回）
        valid = top < nb_i[:, s0:s1, None]

        # 选中 block 的全部 token ∪ 尾巴 [nb_i*R, i]
        blk = torch.zeros(batch_size, qc, nb_max, dtype=torch.bool, device=q.device)
        blk.scatter_(2, top, valid)
        out[:, s0:s1, : nb_max * R] |= blk.repeat_interleave(R, dim=2)

    if os.environ.get("STRATA_QSA_CHECK", "0") != "0":
        _check(self, hidden_states, position_embeddings, attention_mask, out)

    mine = out.unsqueeze(1)                                    # (B, 1, S, S)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        return torch.where(mine, attention_mask.new_zeros(()), min_dtype)
    return mine


_CHECKS = {"n": 0}


def _check(self, hidden_states, position_embeddings, attention_mask, mine) -> None:
    """与逐 query 原实现逐元素比一次（仅调试用，会把两条路径都跑一遍）。

    `mine` 的形状是 (B, 1, S, S)，与 ref 对齐 —— batch 维也要比，否则「各行掩码
    不同」这类错误会被悄悄放过。
    """
    if _CHECKS["n"] >= 2:
        return
    _CHECKS["n"] += 1
    with torch.no_grad():
        ref = type(self)._strata_orig_forward(
            self, hidden_states, position_embeddings, attention_mask, None)
    ref_b = ref if ref.dtype == torch.bool else ref == 0
    diff = (ref_b != mine).sum().item()
    tot = mine.numel()
    print(f"[qsa] check: 掩码差异 {diff}/{tot} ({100 * diff / tot:.4f}%)", flush=True)