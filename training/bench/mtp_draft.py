"""训练线的 MTP 投机解码（draft-then-verify）。

权重源：Strata-data/mtp/（tools/mtp_fetch.py 从 BF16 checkpoint 取的 31 个 mtp.* 张量
+ manifest.json，原始 BF16）。draft 层与主干解码层同构（QSA 注意力 + 超连接 +
512 专家 MoE），直接复用 transformers qwen4_exp 的组件类；embed / lm_head / rotary
与主干共享。

协议（对齐引擎 src/core/mtp.hpp 的 draft 步）：
  cell i 输入 = (主干最终多流残差 R_i（10240，mixer 之前）@ 位置 i, token_{i+1})，
  在 rope 位置 i 预测 token_{i+2}：
    e = fc_embedding(pre_fc_norm_embedding(embed(tok_{i+1})))          # 2560
    hn = pre_fc_norm_hidden(R_i)                                       # 默认整条 10240 一个 RMS
    h = fc_hidden(hn 按 4 流展开)                                       # 每流 2560→2560
    R0 = h + e 广播到 4 流
    R1 = layer(R0, pos=i)（draft 层自有 KV）
    logits = lm_head(mixer(R1)) → token_{i+2}；R1 供下一 draft 步链式使用。

verify / 接受语义（引擎同款：verify window 决定每一个输出 token）：
  greedy：目标 argmax 与 draft 逐位比对，最长前缀接受 + 纠正 token。
  sampling：draft 从 draft 分布采样，逐位以 min(1, q/p) 接受，拒绝位从
  max(q-p, 0) 重采样（Leviathan 投机采样，输出分布严格等于目标分布）。

缓存管理：draft 层用自己的 DynamicCache，每轮 verify 后 crop 回 prefill 末尾，
再按接受结果补喂 catch-up cell（目标 R + 已接受 token），位置严格连续对齐——
等价于引擎「speculative cells 在位置重处理时被覆写」。
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


import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
)


# ---------------------------------------------------------------- 权重装载

def load_mtp_state(mtp_dir: str, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """读 manifest.json + tensors/*.bin（原始 BF16 字节流），返回模块树命名（无 `mtp.` 前缀）。"""
    manifest = json.load(open(os.path.join(mtp_dir, "mtp-manifest.json")))
    state: dict[str, torch.Tensor] = {}
    for entry in manifest:
        name: str = entry["name"]
        shape = entry["shape"]
        raw = open(os.path.join(mtp_dir, "tensors", name + ".bin"), "rb").read()
        assert len(raw) == int(np.prod(shape)) * 2, f"{name}: {len(raw)} B != {shape} BF16"
        # 原始 BF16 字节流：uint16 读入，左移 16 位拼成 fp32（'<f2' 是 fp16，不能直接用）
        arr = np.frombuffer(raw, dtype="<u2").reshape(shape).astype(np.uint32) << 16
        key = name.removeprefix("mtp.").replace("layers.0.", "layer.")
        key = key.replace("hyper_connection_mixer.", "mixer.")
        state[key] = torch.from_numpy(np.ascontiguousarray(arr).view(np.float32)).to(dtype)
    return state


# ---------------------------------------------------------------- draft 模块

class MtpDrafter(nn.Module):
    """一层同构解码层 + fc 双分支 + 顶层 mixer。embed/head/rotary 由 share_main() 注入。"""

    def __init__(self, cfg, pre_fc_norm_mode: str = "full", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        assert pre_fc_norm_mode in ("full", "stream")
        c2 = copy.deepcopy(cfg)
        c2.layer_types = ["qwen_sparse_attention"]  # draft 层是全注意力（有 q_norm/k_norm）；
        # 用主模型的类型串，DynamicCache 才会建出带 indexer 槽的 DynamicIndexedLayer
        c2.ple_layer_ids = []                    # draft 层无 PLE
        if getattr(c2, "_attn_implementation", None) is None:
            c2._attn_implementation = "sdpa"
        h, hc = c2.hidden_size, c2.hc_count
        self.hc = hc
        self.hidden_size = h
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(h, eps=c2.rms_norm_eps)
        # 引擎默认（--mtp-hnorm 关）：整条 hc*h 一个 RMS；"stream" = 每流一个（llama.cpp 口径）
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(
            hc * h, group_size=(h if pre_fc_norm_mode == "stream" else None), eps=c2.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(h, h, bias=False)
        self.fc_hidden = nn.Linear(h, h, bias=False)
        self.layer = Qwen4ExpTextDecoderLayer(c2, layer_idx=0)
        self.mixer = Qwen4ExpTextGatedResidual(c2, use_combine=False)
        self.cache_config = c2   # DynamicCache(config=...) 用它建出带 indexer 槽的层
        self.to(dtype)
        self._embed = None
        self._head = None
        self._rotary = None

    def share_main(self, model) -> None:
        self._embed = model.model.embed_tokens
        self._head = model.lm_head
        self._rotary = model.model.rotary_emb

    def _broadcast_e(self, e: torch.Tensor) -> torch.Tensor:
        """[B,L,h] → [B,L,hc*h]，每个流同一份 e（引擎 add_streams_broadcast 语义）。"""
        B, L, h = e.shape
        return e.unsqueeze(-2).expand(B, L, self.hc, h).reshape(B, L, self.hc * h)

    def cell(self, R: torch.Tensor, tok: torch.Tensor, position_ids_full: torch.Tensor,
             cache: DynamicCache, kv_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """R [B,L,hc*h]；tok [B,L]。返回 (draft 层输出 R1, mixer 后 2560 hidden)。

        position_ids_full [3,B,L_full]：全窗口位置（indexer 要给历史块回 rope，
        与主模型「cache 绑全量 position_ids」同机制）。L_full = cache 行数 + L。
        kv_len = 调用后 cache 的总行数，用于 decode 步的 mask 宽度。
        """
        dt = R.dtype
        e = self._embed(tok)
        e = self.fc_embedding(self.pre_fc_norm_embedding(e))
        hn = self.pre_fc_norm_hidden(R)
        h = self.fc_hidden(hn.unflatten(-1, (self.hc, -1))).flatten(-2)
        R0 = h + self._broadcast_e(e)
        pos3 = position_ids_full if position_ids_full.dim() == 3 else \
            position_ids_full.unsqueeze(0).expand(3, -1, -1)
        pos_emb = self._rotary(R0, pos3)
        Bq = R0.shape[0]
        if R0.shape[1] > 1:
            mask = _causal_additive(R0.shape[1], kv_len, R0.device, dt).expand(Bq, 1, -1, -1)
        else:
            mask = torch.zeros(Bq, 1, 1, kv_len, device=R0.device, dtype=dt)
        R1 = self.layer(R0, position_embeddings=pos_emb, attention_mask=mask,
                        past_key_values=cache)
        mixed = self.mixer(R1)
        return R1, mixed

    def logits_of(self, mixed: torch.Tensor) -> torch.Tensor:
        return self._head(mixed)


def _fix_conv_kernel_sizes(cache: DynamicCache) -> None:
    """GDN 的 conv 状态走自定义更新路径，不写 conv_kernel_size（恒 None）；
    crop 时会 TypeError。这里从 conv_states 的实际形状补齐。"""
    import os as _os
    dbg = _os.environ.get("STRATA_MTP_DEBUG")
    for li, layer in enumerate(cache.layers):
        ck = getattr(layer, "conv_kernel_size", None)
        st = getattr(layer, "conv_states", None)
        if ck is None or not st:
            continue
        items = st.items() if isinstance(st, dict) else enumerate(st)
        for i, s in items:
            if s is not None and ck.get(i) is None:
                ck[i] = s.shape[-1]
        if dbg:
            print(f"[dbg] layer {li} {type(layer).__name__} ck={ck} "
                  f"states={{k: (tuple(v.shape) if v is not None else None) for k, v in (st.items() if isinstance(st, dict) else enumerate(st))}}",
                  flush=True)


def _cache_snapshot(cache: DynamicCache) -> dict:
    """克隆缓存全部层状态（K/V、indexer、conv、GDN recurrent、position_ids）。

    不用 Cache.crop：GDN 混合架构下 conv/recurrent 状态无法按行回滚
    （crop 只裁 conv，recurrent 状态会残留被拒 token 的污染——静默数值错误）。
    verify 有拒绝就走整体恢复，语义与引擎的「位置重处理覆写」一致。
    """
    snap: dict = {"layers": []}
    for layer in cache.layers:
        entry: dict = {}
        for attr in ("keys", "values", "indexer_keys"):
            v = getattr(layer, attr, None)
            entry[attr] = v.detach().clone() if torch.is_tensor(v) else None
        for attr in ("conv_states", "recurrent_states"):
            coll = getattr(layer, attr, None)
            if coll is None:
                continue
            items = coll.items() if isinstance(coll, dict) else enumerate(coll)
            saved = {}
            for i, t in items:
                saved[i] = t.detach().clone() if torch.is_tensor(t) else None
            entry[attr] = (saved, isinstance(coll, dict))
        snap["layers"].append(entry)
    pos = getattr(cache, "position_ids", None)
    snap["position_ids"] = pos.detach().clone() if torch.is_tensor(pos) else None
    return snap


def _cache_restore(cache: DynamicCache, snap: dict) -> None:
    for layer, entry in zip(cache.layers, snap["layers"]):
        for attr in ("keys", "values", "indexer_keys"):
            v = entry.get(attr)
            if v is not None or hasattr(layer, attr):
                setattr(layer, attr, v)
        for attr in ("conv_states", "recurrent_states"):
            if attr not in entry:
                continue
            saved, was_dict = entry[attr]
            coll = getattr(layer, attr)
            if was_dict:
                for i, t in saved.items():
                    coll[i] = t
            else:
                for i, t in saved.items():
                    coll[i] = t
    pos = snap.get("position_ids")
    if pos is not None:
        cache.position_ids = pos


def _causal_additive(q_len: int, kv_len: int, device, dtype) -> torch.Tensor:
    """[1,1,q,kv] 加性 mask：prefill（左上对齐因果）。"""
    m = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    if q_len > 1:
        tri = torch.triu(torch.ones(q_len, kv_len, device=device, dtype=torch.bool),
                         diagonal=1 + kv_len - q_len)
        m.masked_fill_(tri, float("-inf"))
    return m.view(1, 1, q_len, kv_len)


# ---------------------------------------------------------------- 投机解码

@torch.no_grad()
def spec_generate(body, head, drafter: MtpDrafter, ids: torch.Tensor, max_new: int,
                  k: int = 3, mode: str = "greedy", temperature: float = 0.7,
                  generator: torch.Generator | None = None) -> dict:
    """draft-then-verify 生成（单序列 v1）。返回 tokens / accept 统计 / 计时。"""
    assert ids.shape[0] == 1, "v1 只支持单序列"
    dev = ids.device
    t_start = time.time()
    captured: dict[str, torch.Tensor] = {}

    # 顶层 mixer 的 pre-hook：抓它的输入 = 多流残差 R（10240）
    mixer_module = next(m for n, m in body.named_modules() if n.endswith("hyper_connection_mixer"))
    pre = mixer_module.register_forward_pre_hook(lambda m, a, kw: captured.update(R=a[0]), with_kwargs=True)

    # 1) 目标 prefill
    out = body(input_ids=ids, use_cache=True)
    past: DynamicCache = out.past_key_values
    past.activate_past_recording()      # GDN/conv 层的 crop 需要历史录制
    _fix_conv_kernel_sizes(past)        # GDN 自定义更新路径不记 kernel size，crop 会崩
    T = ids.shape[1]
    R_all = captured["R"]                                    # [1, T, 10240]
    nxt = head(out.last_hidden_state[:, -1:])
    if mode == "greedy":
        x_t = int(nxt[0, -1].argmax(-1))
    else:
        x_t = int(torch.softmax(nxt[0, -1].float() / temperature, -1).multinomial(1, generator=generator))
    emitted = [x_t]
    n_target_fwd = 1
    fwd_times: list[float] = []
    draft_times: list[float] = []
    acc_counts: list[int] = []
    tokens_per_fwd: list[int] = []

    # 2) drafter prefill：cells (R_i, x_{i+1}) for i in [0, T-2]
    dcache = DynamicCache(config=drafter.cache_config)   # 1 层，带 indexer 槽
    dcache.activate_past_recording()
    n_pre = T - 1
    drafter.cell(R_all[:, :n_pre], ids[:, 1:T],
                 torch.arange(n_pre, device=dev).view(1, 1, -1).expand(3, 1, -1).contiguous(),
                 dcache, n_pre)
    prefill_rows = n_pre

    # 轮首状态：P = 最后已接受 token 的位置；cur_R = 目标 R 行 @ (P-1)（预测出 x_P 的行）；
    # cur_tok = x_P；cell 位置 = P-1
    P = T
    cur_R = R_all[:, -1:]
    cur_tok = torch.tensor([[x_t]], device=dev)
    seq = [int(t) for t in ids[0]] + emitted                 # 全序列（书籍keeping）

    while len(emitted) < max_new:
        # 轮首：dcache 只保留真值 cells（0..P-2）。本轮 draft 的 spec cells 追加在其后；
        # verify 后按接受结果重喂，缓存永远由目标 R 的 cell 组成（引擎 catchup 语义）。
        # cache 的 crop 约定：负数 = 移除的行数（conv/GDN 层强制）。
        excess = dcache.get_seq_length() - (P - 1)
        if excess > 0:
            dcache.crop(-excess)
        # ---- draft k 个 token（链式，首 cell 用目标 R，其后用 draft 层自身残差）
        t0 = time.time()
        drafts: list[int] = []
        draft_ps: list[torch.Tensor] = []
        R_chain, tok_chain, pos_chain = cur_R, cur_tok, P - 1
        for _ in range(k):
            kv_len = dcache.get_seq_length() + 1
            pos_full = torch.arange(kv_len, device=dev).view(1, 1, -1).expand(3, 1, -1)
            R1, mixed = drafter.cell(R_chain, tok_chain, pos_full, dcache, kv_len)
            lg = drafter.logits_of(mixed[:, -1])[0].float()
            p = torch.softmax(lg / temperature, -1)
            nxt_i = int(lg.argmax(-1)) if mode == "greedy" else int(p.multinomial(1, generator=generator))
            drafts.append(nxt_i)
            draft_ps.append(p)
            R_chain, tok_chain, pos_chain = R1[:, -1:], torch.tensor([[nxt_i]], device=dev), pos_chain + 1
        draft_times.append(time.time() - t0)

        # ---- verify：目标一次前向吃 [x_P, d_1..d_{k-1}]（k 行，位置 P..P+k-1）
        v_tokens = torch.tensor([[seq[P]] + drafts[:-1]], device=dev)
        captured.pop("R", None)
        snap = _cache_snapshot(past)                         # 有拒绝就整体回滚
        t0 = time.time()
        v_out = body(input_ids=v_tokens, past_key_values=past, use_cache=True)
        fwd_times.append(time.time() - t0)
        n_target_fwd += 1
        vR = captured["R"]                                   # [1, k, 10240] @ 位置 P..P+k-1
        v_logits = head(v_out.last_hidden_state)[0].float()  # [k, V]：行 j 判 d_{j+1}
        q = torch.softmax(v_logits / temperature, -1)

        # ---- 接受
        accepted: list[int] = []
        a = 0
        if mode == "greedy":
            g = v_logits.argmax(-1)
            while a < k and int(g[a]) == drafts[a]:
                accepted.append(drafts[a])
                a += 1
            corr = int(g[min(a, k - 1)])
        else:
            corr = None
            for j in range(k):
                pj = draft_ps[j][drafts[j]]
                qj = q[j, drafts[j]]
                if torch.rand((), device=dev, generator=generator) * pj < qj:
                    accepted.append(drafts[j])
                    a += 1
                else:
                    resid = F.relu(q[j] - draft_ps[j])
                    corr = int(resid.multinomial(1, generator=generator))
                    break
            if corr is None:                                  # k 个全接受
                corr = int(q[k - 1].multinomial(1, generator=generator))
        if mode == "greedy" and a == k:
            accepted = drafts[:]                              # d_1..d_k 全接受
            a = k
        acc_counts.append(a)
        emitted.extend(accepted + [corr])
        seq.extend(accepted + [corr])
        tokens_per_fwd.append(a + 1)

        # ---- drafter 缓存收尾（引擎默认语义，STRATA_MTP_CATCHUP_ALL 关）：
        # 接受的 spec cell 留在缓存（位置连续），只裁掉被拒段的尾巴；
        # draft 质量受 R' 残差轻微陈旧影响，但 verify 决定每个输出 token，正确性无涉。
        if k - a - 1 > 0:
            dcache.crop(-(k - a - 1))
        # ---- 状态推进
        newP = P + a + 1                                      # c 的位置
        if k - a - 1 > 0:                                     # 有被拒 draft 的行：整体回滚
            _cache_restore(past, snap)
        cur_R = vR[:, min(a, k - 1):min(a, k - 1) + 1]                          # 预测出 corr 的那一行
        cur_tok = torch.tensor([[corr]], device=dev)
        P = newP

    pre.remove()
    n_steps = len(acc_counts)
    return {
        "tokens": emitted,
        "accept": acc_counts,
        "accept_rate": sum(acc_counts) / max(n_steps, 1),
        "tokens_per_fwd": sum(tokens_per_fwd) / max(n_steps, 1),
        "target_forwards": n_target_fwd,
        "fwd_times": fwd_times,
        "draft_times": draft_times,
        "wall": time.time() - t_start,
    }


# ---------------------------------------------------------------- GRPO 批量采样

@torch.no_grad()
def spec_sample_group(body, head, drafter, ids: torch.Tensor, max_new: int,
                      temperature: float, gen: torch.Generator, k: int = 3) -> dict:
    """GRPO 组采样：K 条共享 prompt 的批量投机解码。

    每轮：drafter 逐行贪心 draft k 个（一批 K 行链式）→ 目标一次前向吃
    [x_P, d_1..d_{k-1}] 网格 → 逐行以 min(1, q/p) 接受（p = draft 的 one-hot），
    拒绝位从 max(q-p,0) 重采样——发射分布严格等于目标分布 ⇒ old logp = log q。
    缓存统一保留 min 行（幸运行多接受的 draft 不发射，上下文无空洞）。

    返回 picked[K][t]、lps[K][t]（log q）、accept 统计与诊断。
    """
    K, T = ids.shape
    dev = ids.device
    captured: dict[str, torch.Tensor] = {}
    mixer = next(m for n, m in body.named_modules() if n.endswith("hyper_connection_mixer"))
    pre = mixer.register_forward_pre_hook(lambda m, a, kw: captured.update(R=a[0]), with_kwargs=True)

    out = body(input_ids=ids, use_cache=True)
    past = out.past_key_values
    R_all = captured["R"]                                    # [K, T, 10240]
    lg0 = head(out.last_hidden_state[:, -1]).float() / temperature
    q0 = torch.softmax(lg0, -1)
    nxt = torch.multinomial(q0, 1, generator=gen)            # [K,1]
    lp0 = torch.log(q0.gather(-1, nxt).squeeze(-1).clamp_min(1e-12))
    picked = [[int(t)] for t in nxt[:, 0]]
    lps = [[float(v)] for v in lp0]

    dcache = DynamicCache(config=drafter.cache_config)
    dcache.activate_past_recording()
    drafter.cell(R_all[:, : T - 1], ids[:, 1:],
                 torch.arange(T - 1, device=dev).view(1, 1, -1).expand(3, K, -1).contiguous(),
                 dcache, T - 1)

    P, cur_R, cur_tok = T, R_all[:, -1:], nxt
    acc_hist: list[int] = []
    while len(picked[0]) < max_new:
        # draft k（一批 K 行链式；drafter 缓存按行独立，K 条互不干扰）
        R_chain, tok_chain = cur_R, cur_tok
        drafts: list[torch.Tensor] = []
        for _ in range(k):
            kv = dcache.get_seq_length() + 1
            pos_full = torch.arange(kv, device=dev).view(1, 1, -1).expand(3, K, -1)
            R1, mixed = drafter.cell(R_chain, tok_chain, pos_full, dcache, kv)
            drafts.append(drafter.logits_of(mixed[:, -1]).argmax(-1))   # [K]
            R_chain, tok_chain = R1[:, -1:], drafts[-1].view(K, 1)
        # verify：目标一次前向吃 [x_P, d_1..d_{k-1}]
        v_tokens = torch.cat([cur_tok, torch.stack(drafts[:-1], 1)], dim=1)   # [K,k]
        captured.pop("R", None)
        snap = _cache_snapshot(past)
        v_out = body(input_ids=v_tokens, past_key_values=past, use_cache=True)
        vR = captured["R"]                                    # [K,k,10240]
        v_logits = head(v_out.last_hidden_state).float() / temperature   # [K,k,V]
        q = torch.softmax(v_logits, -1)
        g = v_logits.argmax(-1)                               # [K,k]
        dmat = torch.stack(drafts, 1)                         # [K,k]
        # 逐行贪心接受前缀
        match = (g == dmat)                                   # [K,k]
        a_vec = match.cumprod(-1).sum(-1)                     # [K] 前缀接受数
        min_a = int(a_vec.min())
        # 发射 min_a 个 draft + 每行自己的 corr（来自行 min_a 的目标分布）
        for j in range(min_a):
            for i in range(K):
                picked[i].append(int(dmat[i, j]))
                lps[i].append(float(torch.log(q[i, j, dmat[i, j]].clamp_min(1e-12))))
        if min_a < k:
            qr = torch.relu(q[:, min_a] - F.one_hot(dmat[:, min_a], q.shape[-1]).to(q))
            qr = qr / qr.sum(-1, keepdim=True).clamp_min(1e-12)
            corr = torch.multinomial(qr, 1, generator=gen)    # [K,1]
            j_lp = min_a
        else:                                                 # k 个全中：bonus 来自行 k-1 的目标分布
            corr = torch.multinomial(q[:, k - 1], 1, generator=gen)
            j_lp = k - 1
        for i in range(K):
            picked[i].append(int(corr[i, 0]))
            lps[i].append(float(torch.log(q[i, j_lp, corr[i, 0]].clamp_min(1e-12))))
        acc_hist.append(min_a)
        # 缓存统一保留行 0..min_a（其上是各 row 自己的 token，长度一致）
        drop = k - (min_a + 1)
        if drop > 0:
            _cache_restore(past, snap)
        # drafter 缓存：裁掉被拒段
        ddrop = k - 1 - min_a
        if ddrop > 0:
            dcache.crop(-ddrop)
        cur_R = vR[:, min_a].unsqueeze(1)
        cur_tok = corr
        P += min_a + 1
    pre.remove()
    n_steps = max(len(acc_hist), 1)
    return {"picked": picked, "lps": lps, "accept": acc_hist,
            "accept_rate": sum(acc_hist) / n_steps,
            "tokens_per_fwd": (sum(acc_hist) + len(acc_hist)) / n_steps}
