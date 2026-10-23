#!/usr/bin/env python3
"""
DeepSeek V4-Flash golden oracle — pure PyTorch reference forward (NO tilelang).

Purpose: produce golden intermediate tensors for validating the Strata C++ engine
(branch dsv4-mvp). Weights are loaded directly from the HF safetensors checkpoint
(ModelOpt NVFP4 experts + FP8 block-quantized projections) and dequantized to
BF16/FP32. The math replicates /data/models/DeepSeek-V4-Flash-0731-NVFP4/
inference/model.py line by line, with the tilelang kernels from kernel.py
replaced by pure-PyTorch reference implementations:

  act_quant / fp4_act_quant  -> identity. These are QAT *simulation* steps
      (quantize->dequantize round trip, inplace=True everywhere in model.py).
      The golden reference computes in full BF16/FP32 precision instead.
  fp8_gemm / fp4_gemm        -> F.linear with the dequantized BF16 weight
      (E4M3 values and E8M0 power-of-two scales are exactly representable, so
      dequantize-to-BF16 is lossless up to the same rounding the kernel would
      produce after its FP32 accumulation).
  sparse_attn                -> dense reference implementing the exact kernel
      semantics: index gather (-1 => -inf score), per-head online-softmax max,
      attn_sink added to the denominator ONLY.
  hc_split_sinkhorn          -> exact torch port of the kernel's iteration order:
      softmax(-1)+eps -> col-norm -> 19x(row-norm, col-norm)  (20 iters total).
  rotate_activation (fast_hadamard_transform, in model.py not kernel.py)
      -> identity. It is only applied to indexer q and indexer-compressor kv
      right before FP4 quant simulation; the scaled Hadamard transform is
      orthogonal, so (Hq).(Hk) == q.k exactly and index scores are unchanged.

Weights are loaded per-layer and released after each layer (peak ~1 GB/layer);
only the routed experts actually activated are dequantized.
MVP scope: DSpark/MTP layers are NOT run.
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

# ----------------------------------------------------------------------------
# Constants (from convert.py — the NVFP4 ground truth)
# ----------------------------------------------------------------------------

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

BF16 = torch.bfloat16
FP32 = torch.float32


# ----------------------------------------------------------------------------
# Config: built from config.json (NOT from model.py ModelArgs defaults — the
# defaults disagree with the checkpoint config on rope_factor, compress_rope_theta,
# original_seq_len, route_scale, n_activated_experts, moe_inter_dim, swiglu_limit,
# n_hash_layers; see MANIFEST.md "semantic notes").
# ----------------------------------------------------------------------------

class Cfg:
    def __init__(self, ckpt):
        c = json.load(open(os.path.join(ckpt, "config.json")))
        self.vocab_size = c["vocab_size"]                    # 129280
        self.dim = c["hidden_size"]                          # 4096
        self.moe_inter_dim = c["moe_intermediate_size"]      # 2048
        self.n_layers = c["num_hidden_layers"]               # 43
        self.n_hash_layers = c["num_hash_layers"]            # 3  -> layers 0,1,2 hash-routed
        self.n_heads = c["num_attention_heads"]              # 64
        self.n_routed_experts = c["n_routed_experts"]        # 256
        self.n_shared_experts = c["n_shared_experts"]        # 1
        self.n_activated_experts = c["num_experts_per_tok"]  # 6
        self.score_func = c["scoring_func"]                  # "sqrtsoftplus"
        self.route_scale = c["routed_scaling_factor"]        # 1.5
        self.swiglu_limit = c["swiglu_limit"]                # 10.0
        self.q_lora_rank = c["q_lora_rank"]                  # 1024
        self.head_dim = c["head_dim"]                        # 512
        self.rope_head_dim = c["qk_rope_head_dim"]           # 64
        self.norm_eps = c["rms_norm_eps"]                    # 1e-6
        self.o_groups = c["o_groups"]                        # 8
        self.o_lora_rank = c["o_lora_rank"]                  # 1024
        self.window_size = c["sliding_window"]               # 128
        self.compress_ratios = tuple(c["compress_ratios"])   # 46 entries; first 43 used
        self.compress_rope_theta = c["compress_rope_theta"]  # 160000
        rs = c["rope_scaling"]
        self.original_seq_len = rs["original_max_position_embeddings"]  # 65536
        self.rope_factor = rs["factor"]                      # 16
        self.beta_fast = rs["beta_fast"]                     # 32
        self.beta_slow = rs["beta_slow"]                     # 1
        self.rope_theta = c["rope_theta"]                    # 10000
        self.index_n_heads = c["index_n_heads"]              # 64
        self.index_head_dim = c["index_head_dim"]            # 128
        self.index_topk = c["index_topk"]                    # 512
        self.hc_mult = c["hc_mult"]                          # 4
        self.hc_sinkhorn_iters = c["hc_sinkhorn_iters"]      # 20
        self.hc_eps = c["hc_eps"]                            # 1e-6
        # oracle-local run geometry (small; only sizes the KV caches)
        self.max_seq_len = 4096
        self.bsz = 1


# ----------------------------------------------------------------------------
# Checkpoint loader with dequantization
# ----------------------------------------------------------------------------

class Loader:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self.handles = {}

    def has(self, name):
        return name in self.wm

    def _fh(self, fname):
        h = self.handles.get(fname)
        if h is None:
            h = safe_open(os.path.join(self.ckpt, fname), framework="pt", device="cpu")
            self.handles[fname] = h
        return h

    def get(self, name):
        return self._fh(self.wm[name]).get_tensor(name)

    def dequant_fp8(self, prefix):
        """FP8 E4M3 weight [out,in] + E8M0 scale [out/128,in/128] -> BF16.
        Formula per convert.py (wo_a branch): per-128x128-block scale broadcast."""
        w = self.get(prefix + ".weight")
        s = self.get(prefix + ".scale")
        out, inn = w.shape
        bo, bi = out // 128, inn // 128
        w = w.float().view(bo, 128, bi, 128) * s.float()[:, None, :, None]
        return w.view(out, inn).to(BF16)

    def dequant_nvfp4(self, prefix):
        """ModelOpt NVFP4 expert weight -> BF16.
        weight: U8 [out, in/2], two E2M1 values per byte (low nibble = even index).
        weight_scale: F8_E4M3 [out, in/16] (group_size 16 along K).
        weight_scale_2: F32 scalar (per-tensor).
        W = FP4_TABLE[nibble] * weight_scale * weight_scale_2   (convert.py FP4_TABLE)."""
        w = self.get(prefix + ".weight")
        ws = self.get(prefix + ".weight_scale")
        ws2 = self.get(prefix + ".weight_scale_2").float()
        out, half = w.shape
        lo = (w & 0x0F).long()
        hi = (w >> 4).long()
        vals = FP4_TABLE[torch.stack([lo, hi], dim=-1)].flatten(1)  # [out, in]
        assert vals.shape == (out, half * 2)
        w = vals * ws.float().repeat_interleave(16, dim=1) * ws2
        return w.to(BF16)


# ----------------------------------------------------------------------------
# RoPE (verbatim port of model.py precompute_freqs_cis / apply_rotary_emb)
# ----------------------------------------------------------------------------

def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn), 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # [seqlen, dim/2] complex64


def rope_apply(t, freqs_cis, rd, inverse=False):
    """Apply rotary to the LAST rd dims of t ([..., D]); out-of-place version of
    model.py's in-place apply_rotary_emb. freqs_cis: [S, rd/2] complex."""
    head, tail = t[..., :-rd], t[..., -rd:]
    x = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
    f = freqs_cis.conj() if inverse else freqs_cis
    if x.ndim == 3:
        f = f.view(1, x.size(1), x.size(-1))
    else:
        f = f.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * f).flatten(-2)
    return torch.cat([head, x.to(t.dtype)], dim=-1)


# ----------------------------------------------------------------------------
# Small reference ops
# ----------------------------------------------------------------------------

def rmsnorm(x, weight, eps):
    dt = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return (weight.float() * (xf * torch.rsqrt(var + eps))).to(dt)


def sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Reference for kernel.py sparse_attn.
    q: [b,s,h,d] BF16, kv: [b,n,d] BF16, attn_sink: [h] FP32, topk_idxs: [b,s,k] int.
    idx == -1 -> score -inf (kv row unused). Denominator includes exp(sink - max)."""
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    k = topk_idxs.shape[-1]
    idx_e = topk_idxs.clamp(min=0).long().unsqueeze(-1).expand(b, s, k, d)
    gathered = torch.gather(kv, 1, idx_e.reshape(b, s * k, d)).view(b, s, k, d)  # [b,s,k,d]
    scores = torch.einsum("bshd,bskd->bhsk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    m = scores.amax(dim=-1, keepdim=True)                          # [b,h,s,1]
    e = (scores - m).exp()                                         # invalid -> 0
    sink = attn_sink.float().view(1, h, 1)
    denom = e.sum(-1) + torch.exp(sink - m.squeeze(-1))  # [b,h,s]
    o = torch.einsum("bhsk,bskd->bshd", e, gathered.float()) / denom.transpose(1, 2).unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Exact torch port of kernel.py hc_split_sinkhorn.
    mixes: [b,s,(2+hc)*hc] FP32. Returns pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]."""
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = mixes[..., 2 * hc_mult:].unflatten(-1, (hc_mult, hc_mult)) * hc_scale[2] \
        + hc_base[2 * hc_mult:].unflatten(-1, (hc_mult, hc_mult))
    # comb = comb.softmax(-1) + eps   (kernel: row-max subtract, exp, row-sum divide, +eps)
    comb = comb.softmax(dim=-1) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, hc_fn, hc_scale, hc_base, norm_eps, hc_mult, sinkhorn_iters, hc_eps):
    # x: [b,s,hc,d] -> y: [b,s,d]; verbatim model.py Block.hc_pre
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
    return y.to(dtype), post, comb


def hc_post(x, residual, post, comb):
    # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc]
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


def hc_head_ref(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    # x: [b,s,hc,d] -> [b,s,d]; verbatim model.py Block.hc_head
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
    return y.to(dtype)


# ----------------------------------------------------------------------------
# Top-k index construction (verbatim model.py)
# ----------------------------------------------------------------------------

def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


# ----------------------------------------------------------------------------
# Compressor (verbatim model.py Compressor.forward; quant steps -> identity)
# st: per-layer persistent state dict. w: weight dict. out_cache: [slots, d] BF16
# (a view into the owning module's kv cache).
# ----------------------------------------------------------------------------

def overlap_transform(tensor, value, ratio, d):
    # tensor: [b,s,r,2d] -> [b,s,2r,d]; verbatim model.py Compressor.overlap_transform
    b, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor


def compressor_forward(st, w, x, start_pos, freqs_cis, ratio, head_dim, rd, eps, out_cache):
    bsz, seqlen, _ = x.size()
    overlap = ratio == 4
    coff = 1 + overlap
    d = head_dim
    dtype = x.dtype
    # compression needs fp32
    xf = x.float()
    kv = xf @ w["wkv"].T       # wkv/wgate loaded BF16 -> FP32
    score = xf @ w["wgate"].T
    if start_pos == 0:
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if overlap else 0
        if overlap and cutoff >= ratio:
            st["kv_state"][:ratio] = kv[0, cutoff - ratio: cutoff]
            st["score_state"][:ratio] = score[0, cutoff - ratio: cutoff] + w["ape"]
        if remainder > 0:
            st["kv_state"][offset: offset + remainder] = kv[0, cutoff:]
            st["score_state"][offset: offset + remainder] = score[0, cutoff:] + w["ape"][:remainder]
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + w["ape"]
        if overlap:
            kv = overlap_transform(kv, 0.0, ratio, d)
            score = overlap_transform(score, float("-inf"), ratio, d)
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
    else:
        should_compress = (start_pos + 1) % ratio == 0
        score = score + w["ape"][start_pos % ratio]
        if overlap:
            st["kv_state"][ratio + start_pos % ratio] = kv[0, 0]
            st["score_state"][ratio + start_pos % ratio] = score[0, 0]
            if should_compress:
                kv_state = torch.cat([st["kv_state"][:ratio, :d], st["kv_state"][ratio:, d:]], dim=0)
                score_state = torch.cat([st["score_state"][:ratio, :d], st["score_state"][ratio:, d:]], dim=0)
                kv = (kv_state * score_state.softmax(dim=0)).sum(dim=0).view(1, 1, -1)
                st["kv_state"][:ratio] = st["kv_state"][ratio:].clone()
                st["score_state"][:ratio] = st["score_state"][ratio:].clone()
        else:
            st["kv_state"][start_pos % ratio] = kv[0, 0]
            st["score_state"][start_pos % ratio] = score[0, 0]
            if should_compress:
                kv = (st["kv_state"] * st["score_state"].softmax(dim=0)).sum(dim=0).view(1, 1, -1)
    if not should_compress:
        return None
    kv = rmsnorm(kv.to(dtype), w["norm"], eps)
    if start_pos == 0:
        fc = freqs_cis[:cutoff:ratio]
    else:
        fc = freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
    kv = rope_apply(kv, fc, rd)
    # rotate_activation + fp4/fp8 quant simulation: identity (see module docstring)
    if start_pos == 0:
        out_cache[: seqlen // ratio] = kv[0]
    else:
        out_cache[start_pos // ratio] = kv[0, 0]
    return kv


# ----------------------------------------------------------------------------
# Indexer (ratio==4 layers; verbatim model.py Indexer.forward)
# ----------------------------------------------------------------------------

def indexer_forward(st, w, x, qr, start_pos, offset, freqs_cis, cfg):
    bsz, seqlen, _ = x.size()
    ratio = cfg.compress_ratios_current  # == 4
    rd = cfg.rope_head_dim
    nh = cfg.index_n_heads
    hd = cfg.index_head_dim
    end_pos = start_pos + seqlen
    fc = freqs_cis[start_pos:start_pos + seqlen]
    q = qr @ w["wq_b"].T                       # dequantized FP8 -> BF16
    q = q.unflatten(-1, (nh, hd))
    q = rope_apply(q, fc, rd)
    # rotate_activation + fp4 quant simulation: identity (orthogonal; see docstring)
    compressor_forward(st["comp"], w["comp"], x, start_pos, freqs_cis, ratio, hd, rd,
                       cfg.norm_eps, st["kv_cache"])
    weights = (x @ w["weights_proj"].T) * (hd ** -0.5 * nh ** -0.5)   # [b,s,nh] BF16
    kv = st["kv_cache"][: end_pos // ratio]
    index_score = torch.einsum("bshd,td->bsht", q, kv)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= \
            torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        index_score = index_score + torch.where(mask, float("-inf"), 0.0)
    topk_idxs = index_score.topk(min(cfg.index_topk, end_pos // ratio), dim=-1)[1]
    if start_pos == 0:
        mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        topk_idxs = torch.where(mask, -1, topk_idxs + offset)
    else:
        topk_idxs = topk_idxs + offset
    return topk_idxs.int()


# ----------------------------------------------------------------------------
# Attention (verbatim model.py Attention.forward)
# ----------------------------------------------------------------------------

def attention_forward(layer_id, w, st, x, start_pos, freqs_cis, cfg):
    bsz, seqlen, _ = x.size()
    win = cfg.window_size
    ratio = cfg.compress_ratios[layer_id]
    rd = cfg.rope_head_dim
    eps = cfg.norm_eps
    fc = freqs_cis[start_pos:start_pos + seqlen]

    # q
    qr = rmsnorm(x @ w["wq_a"].T, w["q_norm"], eps)
    q = (qr @ w["wq_b"].T).unflatten(-1, (cfg.n_heads, cfg.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)   # per-head RMS renorm (model.py:504)
    q = rope_apply(q, fc, rd)

    # win kv
    kv = x @ w["wkv"].T
    kv = rmsnorm(kv, w["kv_norm"], eps)
    kv = rope_apply(kv, fc, rd)
    # act_quant FP8-sim on non-rope dims: identity (see docstring)

    topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    if ratio:
        offset = kv.size(1) if start_pos == 0 else win
        if ratio == 4:
            cfg.compress_ratios_current = ratio
            compress_topk_idxs = indexer_forward(st["indexer"], w["indexer"], x, qr,
                                                 start_pos, offset, freqs_cis, cfg)
        else:
            compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

    if start_pos == 0:
        if seqlen <= win:
            st["kv_cache"][:seqlen] = kv[0]
        else:
            cutoff = seqlen % win
            tail = kv[0, -win:]
            st["kv_cache"][cutoff:win] = tail[: win - cutoff]
            st["kv_cache"][:cutoff] = tail[win - cutoff:]
        if ratio:
            kv_compress = compressor_forward(st["comp"], w["comp"], x, start_pos, freqs_cis,
                                             ratio, cfg.head_dim, rd, eps, st["kv_cache"][win:])
            if kv_compress is not None:
                kv = torch.cat([kv, kv_compress], dim=1)
        o = sparse_attn_ref(q, kv, w["attn_sink"], topk_idxs, cfg.head_dim ** -0.5)
    else:
        st["kv_cache"][start_pos % win] = kv[0, 0]
        if ratio:
            compressor_forward(st["comp"], w["comp"], x, start_pos, freqs_cis,
                               ratio, cfg.head_dim, rd, eps, st["kv_cache"][win:])
        o = sparse_attn_ref(q, st["kv_cache"].unsqueeze(0), w["attn_sink"], topk_idxs,
                            cfg.head_dim ** -0.5)
    o = rope_apply(o, fc, rd, inverse=True)   # inverse rotary on attention OUTPUT

    # grouped low-rank O projection
    o = o.view(bsz, seqlen, cfg.o_groups, -1)
    wo_a = w["wo_a"].view(cfg.o_groups, cfg.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return o.flatten(2) @ w["wo_b"].T


# ----------------------------------------------------------------------------
# MoE (verbatim model.py Gate / Expert / MoE)
# ----------------------------------------------------------------------------

def expert_forward(x, w1, w2, w3, weights, swiglu_limit):
    # x: [t,dim] BF16, w*: dequantized BF16; compute FP32 intermediates like model.py
    dtype = x.dtype
    gate = (x @ w1.T).float()
    up = (x @ w3.T).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return h.to(dtype) @ w2.T


def moe_forward(layer_id, w, x, input_ids, cfg, loader, expert_cache):
    shape = x.size()
    xt = x.view(-1, cfg.dim)
    scores = xt.float() @ w["gate_w"].float().T
    scores = F.softplus(scores).sqrt()          # sqrtsoftplus
    original_scores = scores
    if "gate_b" in w:                            # bias: selection only, layers 3..42
        scores = scores + w["gate_b"]
    if "tid2eid" in w:                           # hash routing, layers 0..2
        indices = w["tid2eid"][input_ids.flatten()]
    else:
        indices = scores.topk(cfg.n_activated_experts, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * cfg.route_scale
    y = torch.zeros_like(xt, dtype=FP32)
    for eid in indices.unique().tolist():
        key = (layer_id, eid)
        if key not in expert_cache:
            p = f"layers.{layer_id}.ffn.experts.{eid}"
            expert_cache[key] = (
                loader.dequant_nvfp4(p + ".w1"),
                loader.dequant_nvfp4(p + ".w2"),
                loader.dequant_nvfp4(p + ".w3"),
            )
        w1, w2, w3 = expert_cache[key]
        idx, top = torch.where(indices == eid)
        y[idx] += expert_forward(xt[idx], w1, w2, w3, weights[idx, top, None], cfg.swiglu_limit)
    y += expert_forward(xt, w["s_w1"], w["s_w2"], w["s_w3"], None, cfg.swiglu_limit)
    return y.type_as(xt).view(shape)


# ----------------------------------------------------------------------------
# Layer weight loading + persistent runtime state
# ----------------------------------------------------------------------------

def load_layer_weights(loader, cfg, i):
    w = {}
    p = f"layers.{i}."
    a = p + "attn."
    w["wq_a"] = loader.dequant_fp8(a + "wq_a")
    w["q_norm"] = loader.get(a + "q_norm.weight")
    w["wq_b"] = loader.dequant_fp8(a + "wq_b")
    w["wkv"] = loader.dequant_fp8(a + "wkv")
    w["kv_norm"] = loader.get(a + "kv_norm.weight")
    w["wo_a"] = loader.dequant_fp8(a + "wo_a")
    w["wo_b"] = loader.dequant_fp8(a + "wo_b")
    w["attn_sink"] = loader.get(a + "attn_sink")
    ratio = cfg.compress_ratios[i]
    if ratio:
        cp = a + "compressor."
        w["comp"] = {
            "wkv": loader.get(cp + "wkv.weight").float(),
            "wgate": loader.get(cp + "wgate.weight").float(),
            "ape": loader.get(cp + "ape").float(),
            "norm": loader.get(cp + "norm.weight"),
        }
        if ratio == 4:
            ip = a + "indexer."
            w["indexer"] = {
                "wq_b": loader.dequant_fp8(ip + "wq_b"),
                "weights_proj": loader.get(ip + "weights_proj.weight"),
                "comp": {
                    "wkv": loader.get(ip + "compressor.wkv.weight").float(),
                    "wgate": loader.get(ip + "compressor.wgate.weight").float(),
                    "ape": loader.get(ip + "compressor.ape").float(),
                    "norm": loader.get(ip + "compressor.norm.weight"),
                },
            }
    w["attn_norm"] = loader.get(p + "attn_norm.weight")
    w["ffn_norm"] = loader.get(p + "ffn_norm.weight")
    w["gate_w"] = loader.get(p + "ffn.gate.weight")
    if loader.has(p + "ffn.gate.bias"):
        w["gate_b"] = loader.get(p + "ffn.gate.bias")
    if loader.has(p + "ffn.gate.tid2eid"):
        w["tid2eid"] = loader.get(p + "ffn.gate.tid2eid")
    for e in ("w1", "w2", "w3"):
        w["s_" + e] = loader.dequant_fp8(p + f"ffn.shared_experts.{e}")
    for side in ("attn", "ffn"):
        w[f"hc_{side}_fn"] = loader.get(p + f"hc_{side}_fn")
        w[f"hc_{side}_base"] = loader.get(p + f"hc_{side}_base")
        w[f"hc_{side}_scale"] = loader.get(p + f"hc_{side}_scale")
    return w


def make_layer_state(cfg, i):
    ratio = cfg.compress_ratios[i]
    win = cfg.window_size
    hd = cfg.head_dim
    st = {}
    n_cache = win + (cfg.max_seq_len // ratio if ratio else 0)
    st["kv_cache"] = torch.zeros(n_cache, hd, dtype=BF16)
    if ratio:
        coff = 2 if ratio == 4 else 1
        st["comp"] = {
            "kv_state": torch.zeros(coff * ratio, coff * hd, dtype=FP32),
            "score_state": torch.full((coff * ratio, coff * hd), float("-inf"), dtype=FP32),
        }
        if ratio == 4:
            ihd = cfg.index_head_dim
            st["indexer"] = {
                "kv_cache": torch.zeros(cfg.max_seq_len // ratio, ihd, dtype=BF16),
                "comp": {
                    "kv_state": torch.zeros(2 * ratio, 2 * ihd, dtype=FP32),
                    "score_state": torch.full((2 * ratio, 2 * ihd), float("-inf"), dtype=FP32),
                },
            }
    return st


# ----------------------------------------------------------------------------
# Full forward (prefill or single-token decode), with dumps
# ----------------------------------------------------------------------------

def run_pass(loader, cfg, states, input_ids, start_pos, freqs_yarn, freqs_plain,
             head_w, norm_w, hc_head_fn, hc_head_base, hc_head_scale,
             dump_dir=None, expert_cache=None):
    ids = torch.tensor(input_ids, dtype=torch.long).view(1, -1)
    seqlen = ids.size(1)
    embed_w = loader.get("embed.weight")
    h = F.embedding(ids, embed_w)                       # [1,s,d] BF16
    del embed_w
    dumps = {"embed": h[0].float().numpy()}
    h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)     # HC expand: 4 identical streams

    for i in range(cfg.n_layers):
        w = load_layer_weights(loader, cfg, i)
        st = states[i]
        freqs = freqs_yarn if cfg.compress_ratios[i] else freqs_plain
        # --- attention sub-block with HC ---
        residual = h
        x, post, comb = hc_pre(h, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"],
                               cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
        x = rmsnorm(x, w["attn_norm"], cfg.norm_eps)
        x = attention_forward(i, w, st, x, start_pos, freqs, cfg)
        h = hc_post(x, residual, post, comb)
        dumps[f"L{i:02d}_post_attn_hc"] = h[0].float().numpy()
        dumps[f"L{i:02d}_post_attn_main"] = h[0].mean(dim=1).float().numpy()
        # --- MoE sub-block with HC ---
        residual = h
        x, post, comb = hc_pre(h, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"],
                               cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
        x = rmsnorm(x, w["ffn_norm"], cfg.norm_eps)
        x = moe_forward(i, w, x, ids, cfg, loader, expert_cache)
        h = hc_post(x, residual, post, comb)
        dumps[f"L{i:02d}_post_moe_hc"] = h[0].float().numpy()
        dumps[f"L{i:02d}_post_moe_main"] = h[0].mean(dim=1).float().numpy()
        del w
        expert_cache.clear()
        gc.collect()

    x = hc_head_ref(h, hc_head_fn, hc_head_scale, hc_head_base, cfg.norm_eps, cfg.hc_eps)
    dumps["hc_head_out"] = x[0].float().numpy()
    x = rmsnorm(x, norm_w, cfg.norm_eps)
    dumps["final_norm"] = x[0].float().numpy()
    logits = x.float() @ head_w.float().T               # [1,s,vocab] FP32 (full_logits=True)
    dumps["logits"] = logits[0].numpy()
    next_id = int(logits[0, -1].argmax().item())

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        for k, v in dumps.items():
            np.save(os.path.join(dump_dir, k + ".npy"), v)
    return next_id, dumps


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/models/DeepSeek-V4-Flash-0731-NVFP4")
    ap.add_argument("--out", default="/data/research/runs/202610-dsv4-golden")
    ap.add_argument("--decode-steps", type=int, default=3)
    ap.add_argument("--raw-prompt", action="store_true",
                    help="encode the bare string instead of the chat template")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    cfg = Cfg(args.ckpt)
    loader = Loader(args.ckpt)

    # --- prompt encoding (checkpoint tokenizer + official chat wrapper) ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    prompt_text = "Write a Python function that adds two numbers."
    if args.raw_prompt:
        prompt_ids = tok.encode(prompt_text)
    else:
        sys.path.insert(0, os.path.join(args.ckpt, "encoding"))
        from encoding_dsv4 import encode_messages
        prompt_str = encode_messages([{"role": "user", "content": prompt_text}],
                                     thinking_mode="chat")
        prompt_ids = tok.encode(prompt_str)
    print(f"prompt ({'raw' if args.raw_prompt else 'chat'}): {prompt_ids}  (len {len(prompt_ids)})")

    # --- rope tables ---
    freqs_yarn = precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len,
                                      cfg.original_seq_len, cfg.compress_rope_theta,
                                      cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    freqs_plain = precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len, 0,
                                       cfg.rope_theta, cfg.rope_factor,
                                       cfg.beta_fast, cfg.beta_slow)

    # --- top-level weights ---
    head_w = loader.get("head.weight")          # BF16 (ParallelHead casts to FP32)
    norm_w = loader.get("norm.weight")
    hc_head_fn = loader.get("hc_head_fn")
    hc_head_base = loader.get("hc_head_base")
    hc_head_scale = loader.get("hc_head_scale")

    states = [make_layer_state(cfg, i) for i in range(cfg.n_layers)]
    expert_cache = {}

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "token_ids.npy"), np.array(prompt_ids, dtype=np.int64))

    # --- prefill ---
    t0 = time.time()
    next_id, _ = run_pass(loader, cfg, states, prompt_ids, 0, freqs_yarn, freqs_plain,
                          head_w, norm_w, hc_head_fn, hc_head_base, hc_head_scale,
                          dump_dir=os.path.join(args.out, "prefill"),
                          expert_cache=expert_cache)
    print(f"prefill done in {time.time()-t0:.1f}s; next_id={next_id} ({tok.decode([next_id])!r})")

    # --- decode ---
    gen = []
    for step in range(args.decode_steps):
        t0 = time.time()
        nid, _ = run_pass(loader, cfg, states, [next_id], len(prompt_ids) + step,
                          freqs_yarn, freqs_plain, head_w, norm_w,
                          hc_head_fn, hc_head_base, hc_head_scale,
                          dump_dir=os.path.join(args.out, f"decode{step+1}"),
                          expert_cache=expert_cache)
        gen.append(nid)
        print(f"decode{step+1} in {time.time()-t0:.1f}s; id={nid} ({tok.decode([nid])!r})")
        next_id = nid

    np.save(os.path.join(args.out, "generated_ids.npy"), np.array(gen, dtype=np.int64))
    decoded = tok.decode(gen)
    print("generated text:", repr(decoded))

    # --- per-layer stats (prefill) ---
    stats = {}
    pd = os.path.join(args.out, "prefill")
    for i in range(cfg.n_layers):
        for tag in ("post_attn_main", "post_moe_main"):
            a = np.load(os.path.join(pd, f"L{i:02d}_{tag}.npy"))
            stats[f"L{i:02d}_{tag}"] = {
                "mean": float(a.mean()), "std": float(a.std()),
                "absmax": float(np.abs(a).max()), "shape": list(a.shape),
            }
    lg = np.load(os.path.join(pd, "logits.npy"))
    stats["logits_prefill"] = {"mean": float(lg.mean()), "std": float(lg.std()),
                               "absmax": float(np.abs(lg).max()), "shape": list(lg.shape)}
    run_info = {
        "prompt_text": prompt_text,
        "prompt_mode": "raw" if args.raw_prompt else "chat",
        "prompt_ids": prompt_ids,
        "generated_ids": gen,
        "generated_text": decoded,
        "decode_steps": args.decode_steps,
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"run_info": run_info, "per_layer_prefill": stats}, f, indent=2)
    print("stats written to", os.path.join(args.out, "stats.json"))


if __name__ == "__main__":
    main()
