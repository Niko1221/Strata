"""ref/glm.py - the glm5-next (GLM-5.3-Flash) reference forward pass, one op at a time.

Phase 1 of docs/GLM5-FLASH.md: the specification every engine kernel is parity-tested against.
It is written from `ref/upstream-llama.cpp/glm5-next.cpp` (the llama.cpp implementation this
architecture's GGUF conversion follows) with the released config's values as defaults, and it
keeps ggml's own conventions so the two can be compared op by op:

  * activations are (features, tokens), features varying fastest, like a ggml tensor's ne[0];
  * the residual stack R is (n_embd, hc, T) and its flat view is (n_embd*hc, T) with n_embd
    fastest, which is exactly what `ggml_reshape_2d` produces;
  * weights come from `ref/load.py`'s `tensor()` as (out, in), so `y = W @ x` IS
    `ggml_mul_mat` - no transposes anywhere;
  * splitting a (dim*n_head, T) projection into heads follows ggml's reshape order
    (element (e, h) lives at row e + dim*h) - see `split_heads`/`join_heads`;
  * every intermediate the engine will need as a parity seam is recorded by `cb()` under the
    name llama.cpp's `cb()` uses for the same node.

THE ONE PASS IS TEACHER-FORCED OVER THE WHOLE SEQUENCE. The KDA recurrence is a plain Python
loop over tokens; the DSA layers build their k-pools over the whole sequence and make pool p
visible to query t only when the pool's last member is <= t (an incomplete pool contributes
only through the select-tail cells, which the released config enables via
`index_kpool_always_select_tail`). Decode and chunked-prefill equivalence is Phase 2's job;
this file fixes WHAT is computed, not how it is scheduled.

Usage:
    python ref/glm.py MODEL.gguf [--tokens 1,2,3] [--dump-dir DIR] [--greedy N]

`MODEL.gguf` is any glm5-next GGUF (the real converted model, or `tools/glm_synth_gguf.py`'s
tiny synthetic one, which is what the parity tests run against).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

from load import Gguf  # noqa: E402  (ref/load.py; pulls in tools/gguf_reader.py)

F32 = np.float32

# Branch-hit coverage, per forward (see tools/cover_audit.py).  A fixture whose forward never
# fires a guarded branch (SwiGLU clamps, the renorm clamp, k-pool discards, near-tied router
# selections) cannot prove anything about that branch - the audit fails a fixture on any zero.
COVER: Counter = Counter()


# ================================ the config ================================

class Glm5Config:
    """Every number a kernel depends on, from the GGUF metadata (glm5-next.* keys) with the
    released GLM-5.3-Flash values as defaults - so a synthetic GGUF only has to write what it
    changes.  Key names are llama.cpp's LLM_KV strings for arch glm5-next."""

    def __init__(self, meta: dict):
        def u(key: str, default):
            v = meta.get("glm5-next." + key)
            return default if v is None else v

        self.n_embd = int(u("embedding_length", 4096))
        self.n_layers = int(u("block_count", 45))
        self.n_head = int(u("attention.head_count", 64))
        # per-layer array: 0 marks a KDA (recurrent) layer, >=1 an MLA layer.  Scalars apply to all.
        self.n_head_kv = u("attention.head_count_kv", None)
        self.n_head_k_mla = int(u("attention.key_length_mla", 256))
        self.n_head_v_mla = int(u("attention.value_length_mla", 256))
        self.q_lora_rank = int(u("attention.q_lora_rank", 1536))
        self.kv_lora_rank = int(u("attention.kv_lora_rank", 512))
        # the K-cache row width contract (deepseek2 convention: key_length = kv_lora + rope); the
        # reference holds the bare latent, so a disagreeing key_length is refused, matching the runner
        _kl = u("attention.key_length", None)
        if _kl is not None and int(_kl) != self.kv_lora_rank:
            raise ValueError(f"attention.key_length {int(_kl)} != kv_lora_rank {self.kv_lora_rank} "
                             "(nope-only MLA holds the bare latent; a rope tail is not supported)")
        self.ssm_d_conv = int(u("ssm.conv_kernel", 4))
        self.kda_head_dim = int(u("kda.head_dim", 128))
        self.kda_gate_lower_bound = float(u("kda.gate_lower_bound", -5.0))

        self.n_expert = int(u("expert_count", 288))
        self.n_expert_used = int(u("expert_used_count", 8))
        n_ff_exp = u("expert_feed_forward_length", 2048)
        self.n_ff_exp = [int(x) for x in n_ff_exp] if isinstance(n_ff_exp, list) else int(n_ff_exp)
        self.n_expert_shared = int(u("expert_shared_count", 1))
        self.n_layer_dense_lead = int(u("leading_dense_block_count", 3))
        self.n_ff = int(u("feed_forward_length", 12288))
        self.expert_weights_scale = float(u("expert_weights_scale", 1.0))
        self.expert_weights_norm = bool(u("expert_weights_norm", False))

        self.idx_n_head = int(u("attention.indexer.head_count", 32))
        self.idx_key_dim = int(u("attention.indexer.key_length", 128))
        self.idx_top_k = int(u("attention.indexer.top_k", 2048))
        self.idx_kpool = int(u("attention.indexer.kpool", 4))
        self.idx_kpool_select_tail = bool(u("attention.indexer.kpool_select_tail", False))
        idx_types = u("attention.indexer.types", None)
        self.idx_full = None if idx_types is None else [bool(int(x)) for x in idx_types]

        self.hc = int(u("hyper_connection.count", 4))
        self.hc_sinkhorn_iters = int(u("hyper_connection.sinkhorn_iterations", 20))
        self.hc_eps = float(u("hyper_connection.epsilon", 1e-6))

        self.norm_rms_eps = float(u("attention.layer_norm_rms_epsilon", 1e-5))
        self.norm_eps = float(u("attention.layer_norm_epsilon", 1e-5))
        swiglu = u("swiglu_clamp_exp", None)
        self.swiglu_limit_exp = float(swiglu) if not isinstance(swiglu, list) else float(swiglu[0])
        swiglu_sh = u("swiglu_clamp_shexp", swiglu)
        self.swiglu_limit_shexp = float(swiglu_sh) if not isinstance(swiglu_sh, list) else float(swiglu_sh[0])

        if self.n_head_kv is None or isinstance(self.n_head_kv, (int, float)):
            kv = 0 if self.n_head_kv is None else int(self.n_head_kv)
            self.n_head_kv = [kv] * self.n_layers
        else:
            self.n_head_kv = [int(x) for x in self.n_head_kv]

        # derived
        self.d_inner = self.kda_head_dim * self.n_head          # KDA width
        self.qk_nope = self.n_head_k_mla                        # nope-only MLA: the rope dim is 0
        self.hc_mix_dim = (2 + self.hc) * self.hc

    def n_ff_exp_of(self, il: int) -> int:
        return self.n_ff_exp[il] if isinstance(self.n_ff_exp, list) else self.n_ff_exp

    def is_recr(self, il: int) -> bool:
        return self.n_head_kv[il] == 0

    def is_indexer_full(self, il: int) -> bool:
        return True if self.idx_full is None else self.idx_full[il]


# ================================ ggml primitives ================================

def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    """ggml_rms_norm: per column, over the FEATURE axis (ne[0]); the weight broadcasts over
    every trailing axis (so a (dim, n_head, T) tensor takes a (dim,) weight)."""
    ms = np.mean(x.astype(np.float64) ** 2, axis=0, keepdims=True)
    y = (x / np.sqrt(ms + eps)).astype(F32)
    if w is None:
        return y
    return (y * w.reshape((-1,) + (1,) * (x.ndim - 1))).astype(F32)


def layer_norm(x: np.ndarray, w: np.ndarray | None, b: np.ndarray | None, eps: float) -> np.ndarray:
    """ggml_norm (LLM_NORM): mean/variance over the feature axis, per column, then w and b."""
    xf = x.astype(np.float64)
    mu = xf.mean(axis=0, keepdims=True)
    var = ((xf - mu) ** 2).mean(axis=0, keepdims=True)
    y = ((xf - mu) / np.sqrt(var + eps)).astype(F32)
    shape = (-1,) + (1,) * (x.ndim - 1)
    if w is not None:
        y = (y * w.reshape(shape)).astype(F32)
    return (y + b.reshape(shape)).astype(F32) if b is not None else y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(F32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x * sigmoid(x)).astype(F32)


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Stable softmax along `axis` (ggml's soft_max works over ne[0]; the caller permutes)."""
    z = x.astype(np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(F32)


def gdn_l2_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """build_gdn_l2_norm: scale(rms_norm(x, eps/n), 1/sqrt(n)) == x / sqrt(||x||^2 + eps)."""
    n = x.shape[0]
    ss = np.sum(x.astype(np.float64) ** 2, axis=0, keepdims=True)
    return (x / np.sqrt(ss + eps)).astype(F32)


def swiglu_clamp(gate: np.ndarray, up: np.ndarray, limit: float) -> np.ndarray:
    """ggml_swiglu_clamp: gate = min(gate, limit); up = clamp(up, -limit, +limit); silu(gate)*up.
    The gate is clamped ABOVE only - that asymmetry is in the kernel (ops.cpp L3450)."""
    COVER["swiglu_gate_clamp_hits"] += int((gate > limit).sum())
    COVER["swiglu_up_clamp_hits"] += int((np.abs(up) > limit).sum())
    g = np.minimum(gate, F32(limit))
    u = np.clip(up, -limit, limit)
    return silu(g) * u


def top_k_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """ggml_top_k + the descending argsort the pool selection runs afterwards: k indices ordered
    by DESCENDING score, ties broken toward the lower index (stable sort on -score)."""
    k = min(k, scores.shape[0])
    return np.argsort(-scores.astype(np.float64), axis=0, kind="stable")[:k].astype(np.int64)


def split_heads(x: np.ndarray, dim: int, n_head: int) -> np.ndarray:
    """(dim*n_head, T) -> (dim, n_head, T) in ggml's reshape order: element (e, h) at row e + dim*h."""
    return np.ascontiguousarray(x.reshape(n_head, dim, -1).transpose(1, 0, 2))


def join_heads(x: np.ndarray) -> np.ndarray:
    """(dim, n_head, T) -> (dim*n_head, T), the same order split_heads undoes."""
    dim, n_head, T = x.shape
    return np.ascontiguousarray(x.transpose(1, 0, 2).reshape(n_head * dim, T))


# ================================ the model ================================

class Glm5Next:
    def __init__(self, path: str | pathlib.Path):
        self.g = Gguf(str(path))
        self.cfg = Glm5Config(self.g.g.metadata)
        self.dumps: dict[str, np.ndarray] = {}

    # ---- weights; `tensor()` is (out, in), so every use below is y = W @ x
    def t(self, name: str) -> np.ndarray:
        return self.g.tensor(name).astype(F32)

    def layer_t(self, il: int, base: str, suffix: str = "weight") -> np.ndarray:
        return self.t(f"blk.{il}.{base}.{suffix}")

    def cb(self, name: str, x: np.ndarray, il: int = -1) -> None:
        self.dumps[f"{name}-{il}" if il >= 0 else name] = np.array(x, dtype=F32, copy=True)

    # ================================ mHC ================================

    def hc_pre(self, R: np.ndarray, fn: np.ndarray, scale: np.ndarray, base: np.ndarray,
               il: int, half: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """build_hc_pre + build_hc_sinkhorn.  R is (n_embd, hc, T); returns (mixed, post, comb).

        mixed = sum_s pre_s * R_s,   pre = sigmoid(affine(mixes[0:hc])) + hc_eps,
        post  = 2 * sigmoid(affine(mixes[hc:2hc])),
        comb[dst, src] = sinkhorn(affine(mixes[2hc + dst + hc*src])).

        The mixes come from ONE rms norm over the whole flattened stack - the four streams are
        normalized JOINTLY, not per stream - projected by hc_fn (hc*n_embd -> hc_mix_dim).
        NOTE the comb row order: src-major (`mixes[2hc + dst + hc*src]`, ggml-h doc L2686), so
        the numpy reshape must transpose - reshape(hc, hc) alone would give dst-major.
        """
        c = self.cfg
        n_embd, hc, T = R.shape
        # ggml's reshape of the (n_embd, hc, T) stack to (hc_dim, T) is STREAM-major (ne0 fastest:
        # element (e, s, t) at e + n_embd*s + hc*n_embd*t), so the numpy flatten must transpose first
        # - R.reshape(hc_dim, T) alone would give embedding-major rows and silently mismatch the
        # conversion's hc_*_fn reduction axis.
        flat = R.transpose(1, 0, 2).reshape(n_embd * hc, T)
        flat_norm = rms_norm(flat, None, c.norm_rms_eps)
        mixes = fn @ flat_norm                                   # (hc_mix_dim, T)
        self.cb(f"hc_mixes_{half}", mixes, il)

        pre = sigmoid(mixes[0:hc] * scale[0:1] + base[0:hc, None]) + F32(c.hc_eps)
        post = 2.0 * sigmoid(mixes[hc:2 * hc] * scale[1:2] + base[hc:2 * hc, None])

        # flat row r = dst + hc*src (src-major): the affine is elementwise over those rows, then
        # numpy's reshape(hc, hc) yields [src, dst] (row = i*hc + j), so transpose to (dst, src).
        flat = mixes[2 * hc:2 * hc + hc * hc] * scale[2] + base[2 * hc:2 * hc + hc * hc][:, None]
        comb = self.hc_sinkhorn(np.ascontiguousarray(flat.reshape(hc, hc, T).transpose(1, 0, 2)))  # (dst, src, T)
        mixed = (R * pre[None, :, :]).sum(axis=1)                # sum_s pre_s * R_s
        self.cb(f"hc_pre_{half}", pre, il)
        self.cb(f"hc_post_{half}", post, il)
        self.cb(f"hc_comb_{half}", comb, il)
        return np.ascontiguousarray(mixed, dtype=F32), post, comb

    def hc_sinkhorn(self, comb: np.ndarray) -> np.ndarray:
        """build_hc_sinkhorn, from the unfused graph path (which the fused op must match):
        softmax over dst, +eps, normalize each dst row, then (iters-1) x [normalize each src
        column, normalize each dst row].  eps is the hc epsilon, added after the softmax and
        inside every divisor.  (ggml.h's comment names the sequence differently; the executable
        unfused path is the authority and Phase 2's fused-vs-unfused parity settles it.)"""
        c = self.cfg
        eps = F32(c.hc_eps)
        comb = softmax(comb, axis=0)                             # softmax over dst
        comb = (comb + eps).astype(F32)

        def norm_dst():                                          # each dst row sums to 1
            return (comb / (comb.sum(axis=1, keepdims=True) + eps)).astype(F32)

        def norm_src():                                          # each src column sums to 1
            return (comb / (comb.sum(axis=0, keepdims=True) + eps)).astype(F32)

        comb = norm_dst()
        for _ in range(1, c.hc_sinkhorn_iters):
            comb = norm_src()
            comb = norm_dst()
        return comb

    def hc_post(self, block_out: np.ndarray, residual: np.ndarray, post: np.ndarray,
                comb: np.ndarray) -> np.ndarray:
        """build_hc_post: out_dst = block_out * post_dst + sum_src comb[dst, src] * residual_src."""
        n_embd, hc, T = residual.shape
        out = np.empty((n_embd, hc, T), dtype=F32)
        for d in range(hc):
            acc = block_out * post[d][None, :]
            for s in range(hc):
                acc = acc + (comb[d, s][None, None, :] * residual[:, s, :]).astype(F32)
            out[:, d, :] = acc
        return out

    # ================================ KDA (linear attention) ================================

    def conv1d_causal(self, x_proj: np.ndarray, conv_w: np.ndarray,
                      state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """glm5_conv1d for one of Q/K/V.  x_proj is (d_inner, T) (the wq/wk/wv projection's
        output), conv_w is (d_conv, d_inner), state is the (d_inner, d_conv-1) history holding
        x_proj values at positions -(d_conv-1)..-1, oldest first.  Returns (silu(conv), new_state).

        Output position t's window is x_proj[t-(d_conv-1) .. t]; taps older than the sequence
        come from the state.  The conv weight's d_conv taps apply oldest-first, which is
        ggml_ssm_conv's contract."""
        d_conv = conv_w.shape[0]
        d_inner, T = x_proj.shape
        windows = np.empty((d_conv, d_inner, T), dtype=F32)
        for tap in range(d_conv):
            src = np.arange(T) - (d_conv - 1) + tap              # position this tap reads
            from_state = src < 0
            windows[tap][:, src >= 0] = x_proj[:, src[src >= 0]]
            if from_state.any():
                windows[tap][:, from_state] = state[:, src[from_state] + (d_conv - 1)]
        out = np.einsum("tc,tck->ck", conv_w.astype(np.float64), windows.astype(np.float64)).astype(F32)
        hist = np.concatenate([state, x_proj], axis=1)[:, -(d_conv - 1):]
        return silu(out), np.ascontiguousarray(hist, dtype=F32)

    def kda_layer(self, cur: np.ndarray, il: int) -> np.ndarray:
        """build_kda_layer.  cur is (n_embd, T), the normed hc-pre mix.

        Order (upstream-llama.cpp/glm5-next.cpp L687-763):
          separate q/k/v projections -> per-stream causal conv -> SiLU
          q, k = l2_norm (v is NOT normalized)
          decay g1 = lb * sigmoid(-(f_b^T (f_a^T x) + dt_b) * A)   (A = ssm_a holds -exp(A_log));
              the lower bound lb < 0 puts exp(g1) in (e^lb, 1), per channel of the KEY axis
          beta  = sigmoid(ssm_beta^T x)                            per head
          recurrent delta rule, per head, state (dk, dv), zeroed for a fresh sequence:
              S = diag(exp(g1)) @ S            the gate scales the KEY axis (ops.cpp L11006)
              S += k (beta * (v - S^T k))^T    the delta update
              o = (1/sqrt(dv)) * S^T q         the 1/sqrt(S_v) scale is INSIDE the kernel
          o = rms_norm(ssm_o_norm) * sigmoid(g2), g2 = g_b^T (g_a^T x) the low-rank output gate;
          then wo.
        """
        c = self.cfg
        dk = c.kda_head_dim
        T = cur.shape[1]
        conv_w = [self.layer_t(il, f"ssm_conv1d_{s}").reshape(self.layer_t(il, f"ssm_conv1d_{s}").shape[0], -1).T
                  for s in ("q", "k", "v")]
        wqkv = [self.layer_t(il, f"attn_{s}") @ cur for s in ("q", "k", "v")]
        zeros = np.zeros((c.d_inner, c.ssm_d_conv - 1), dtype=F32)
        q_c, _ = self.conv1d_causal(wqkv[0], conv_w[0], zeros)
        k_c, _ = self.conv1d_causal(wqkv[1], conv_w[1], zeros.copy())
        v_c, _ = self.conv1d_causal(wqkv[2], conv_w[2], zeros.copy())
        self.cb("kda_q_conv", q_c, il)
        self.cb("kda_k_conv", k_c, il)
        self.cb("kda_v_conv", v_c, il)

        f_a = self.layer_t(il, "ssm_f_a") @ cur                  # (dk, T)
        g1 = self.layer_t(il, "ssm_f_b") @ f_a                   # (d_inner, T)
        g1 = g1 + self.layer_t(il, "ssm_dt", "bias")[:, None]
        A = self.t(f"blk.{il}.ssm_a")                            # (n_head,) holds -exp(A_log)
        g1_arg = -split_heads(g1, dk, c.n_head) * A[None, :, None]     # pre-sigmoid argument
        g1 = c.kda_gate_lower_bound * sigmoid(g1_arg)
        # the bound is structural (sigmoid), so "triggering" is saturation: the gate sitting on
        # the lower bound because the pre-sigmoid argument is deep positive
        COVER["kda_gate_saturated"] += int((g1_arg > 8.0).sum())
        self.cb("kda_g1", g1, il)

        beta = sigmoid(self.layer_t(il, "ssm_beta") @ cur)       # (n_head, T)
        self.cb("kda_beta", beta, il)

        q = gdn_l2_norm(split_heads(q_c, dk, c.n_head))
        k = gdn_l2_norm(split_heads(k_c, dk, c.n_head))
        v = split_heads(v_c, dk, c.n_head)

        o = np.empty((dk, c.n_head, T), dtype=F32)
        scale = F32(1.0 / np.sqrt(dk))
        for h in range(c.n_head):
            S = np.zeros((dk, dk), dtype=np.float64)
            for t in range(T):
                S *= np.exp(g1[:, h, t].astype(np.float64))[:, None]
                delta = (v[:, h, t].astype(np.float64) - S.T @ k[:, h, t].astype(np.float64)) * float(beta[h, t])
                S += np.outer(k[:, h, t].astype(np.float64), delta)
                o[:, h, t] = (S.T @ q[:, h, t].astype(np.float64) * scale).astype(F32)
        self.cb("kda_scan_out", o, il)

        g2 = split_heads(self.layer_t(il, "ssm_g_b") @ (self.layer_t(il, "ssm_g_a") @ cur), dk, c.n_head)
        normed = rms_norm(o, self.layer_t(il, "ssm_norm"), c.norm_rms_eps)
        self.cb("kda_g2", g2, il)
        self.cb("kda_normed", normed, il)
        out = self.layer_t(il, "attn_output") @ join_heads(normed * sigmoid(g2))
        self.cb("kda_out", out, il)
        return out

    # ================================ DSA (nope MLA + k-pool indexer) ================================

    def dsa_layer(self, cur: np.ndarray, il: int) -> np.ndarray:
        """build_dsa_layer + build_kpool_select.  Returns (n_embd, T).

        The cache holds ONLY the compressed latent (kv_lora_rank per token).  Q is absorbed
        through wk_b (score = (wk_b_h^T q_h) . latent) and V is absorbed through wv_b AFTER the
        softmax, so no per-head K/V ever exists.  The kq scale is 1/sqrt(n_embd_head_k_mla).

        The indexer scores POOLS of kpool consecutive tokens: a pool's key is the per-key-dim
        softmax over its members of (gate + ape) weighted average of the member keys - the
        softmax runs SEPARATELY PER KEY-DIM ELEMENT (ggml permutes (kpool, key_dim) and
        soft_maxes over ne[0]).  Pool p is visible to query t only when all its members are
        <= t; the incomplete tail is appended when kpool_select_tail.  Selection is
        top_k/kpool pools per query by descending score, where
        score[t, p] = sum_h relu(iq_h . pooled_p) * weights_h, weights prescaled by
        1/sqrt(key_dim * n_head).  Padded selection slots are excluded from the softmax, which
        is numerically the -inf additive mask the graph uses (exp(-inf) contributes 0).
        """
        c = self.cfg
        T = cur.shape[1]
        qr = rms_norm(self.layer_t(il, "attn_q_a") @ cur, self.layer_t(il, "attn_q_a_norm"), c.norm_rms_eps)
        self.cb("q_resid", qr, il)
        q = split_heads(self.layer_t(il, "attn_q_b") @ qr, c.qk_nope, c.n_head)

        kv_cmpr = rms_norm(self.layer_t(il, "attn_kv_a_mqa") @ cur, self.layer_t(il, "attn_kv_a_norm"),
                           c.norm_rms_eps)
        self.cb("kv_cmpr", kv_cmpr, il)

        wk_b = self.layer_t(il, "attn_k_b")                      # (n_head, kv_lora, qk_nope)
        q_abs = np.empty((c.kv_lora_rank, c.n_head, T), dtype=F32)
        for h in range(c.n_head):
            q_abs[:, h, :] = wk_b[h] @ q[:, h, :]
        self.cb("q_absorbed", q_abs, il)

        # ---- the indexer
        iq = split_heads(self.layer_t(il, "indexer.attn_q_b") @ qr, c.idx_key_dim, c.idx_n_head)
        ik = layer_norm(self.layer_t(il, "indexer.attn_k") @ cur, self.layer_t(il, "indexer.k_norm"),
                        self.layer_t(il, "indexer.k_norm", "bias"), c.norm_eps)
        ig = self.layer_t(il, "indexer_compressor_gate") @ cur        # (idx_key_dim, T)
        self.cb("indexer_q", iq, il)
        self.cb("indexer_k", ik, il)
        self.cb("indexer_gate", ig, il)

        kpool = c.idx_kpool
        n_pool_total = T // kpool                                # pools completed by the sequence's end
        ape = self.layer_t(il, "indexer_compressor_ape").T            # (idx_key_dim, kpool)
        pooled = np.zeros((c.idx_key_dim, max(n_pool_total, 1)), dtype=F32)
        for p in range(n_pool_total):
            members = slice(p * kpool, (p + 1) * kpool)
            w = softmax(ig[:, members] + ape, axis=1)            # per key-dim element, over members
            pooled[:, p] = (ik[:, members] * w).sum(axis=1)
        self.cb("indexer_pool_k", pooled, il)

        weights = (self.layer_t(il, "indexer.proj") @ cur) / F32(np.sqrt(c.idx_key_dim * c.idx_n_head))
        self.cb("indexer_weights", weights, il)

        kq = np.einsum("eht,ep->htp", iq.astype(np.float64), pooled.astype(np.float64))
        score = (np.maximum(kq, 0.0) * weights[:, :, None].astype(np.float64)).sum(axis=0)  # (T, n_pool)
        self.cb("indexer_score", score.T, il)

        n_top_pool = min(n_pool_total, c.idx_top_k // kpool)
        n_tail = (kpool - 1) if c.idx_kpool_select_tail else 0
        COVER["dsa_topk_discard_queries"] += int(sum(
            1 for t in range(T) if (t + 1) // kpool > n_top_pool))
        COVER["dsa_pool_boundaries_crossed"] += T // kpool

        wv_b = self.layer_t(il, "attn_v_b")                      # (n_head, v_head, kv_lora)
        wo = self.layer_t(il, "attn_output")                        # (n_embd, n_head * v_head)
        out = np.zeros((c.n_head * c.n_head_v_mla, T), dtype=F32)
        kq_scale = F32(1.0 / np.sqrt(c.qk_nope))
        for t in range(T):
            n_vis = (t + 1) // kpool                             # pools whose last member is <= t
            sel_pools = top_k_desc(score[t, :n_vis], n_top_pool) if n_vis > 0 else np.zeros(0, np.int64)
            cells = (sel_pools[:, None] * kpool + np.arange(kpool)[None, :]).reshape(-1)
            if n_tail:
                tail = np.arange(n_vis * kpool, t + 1, dtype=np.int64)
                tail = np.pad(tail, (0, n_tail - len(tail)), constant_values=-1)
                cells = np.concatenate([cells, tail])
            latents = kv_cmpr[:, cells]                          # (kv_lora, n_sel), -1 column reads garbage
            valid = cells >= 0
            if not valid.any():
                out[:, t] = 0.0                                  # nothing visible yet (select_tail off, t < kpool-1)
                continue
            s = (q_abs[:, :, t].astype(np.float64).T @ latents[:, valid].astype(np.float64)) * kq_scale
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)                 # (n_head, n_valid)
            ctx = latents[:, valid].astype(np.float64) @ p.T     # (kv_lora, n_head)
            oh = np.stack([wv_b[h] @ ctx[:, h] for h in range(c.n_head)], axis=1)  # (v_head, n_head)
            out[:, t] = oh.T.reshape(-1).astype(F32)
        self.cb("kqv_out", out, il)

        attn = wo @ out
        self.cb("attn_out", attn, il)
        return attn

    # ================================ the FFNs ================================

    def dense_ffn(self, cur: np.ndarray, il: int) -> np.ndarray:
        """Dense SwiGLU (leading layers).  The clamp used here is the SHEXP one: build_ffn reads
        swiglu_clamp_shexp for every non-expert FFN, build_moe_ffn reads swiglu_clamp_exp for
        the routed experts."""
        c = self.cfg
        gate = self.layer_t(il, "ffn_gate") @ cur
        up = self.layer_t(il, "ffn_up") @ cur
        out = self.layer_t(il, "ffn_down") @ swiglu_clamp(gate, up, c.swiglu_limit_shexp)
        self.cb("ffn_out", out, il)
        return out

    def moe_ffn(self, cur: np.ndarray, il: int) -> np.ndarray:
        """build_moe_ffn with sigmoid gating and the DeepSeek selection bias:
        probs = sigmoid(logits) is UNBIASED and is what the weights are made of; the bias is
        added only for the top-k selection; weights are renormalized (clamp at 2**-14) then
        scaled by expert_weights_scale (noaux_tc, norm_topk_prob, routed_scaling_factor).
        """
        c = self.cfg
        logits = self.layer_t(il, "ffn_gate_inp") @ cur
        probs = sigmoid(logits)                                  # (n_expert, T), unbiased
        bias = self.layer_t(il, "exp_probs_b", "bias")
        sel_scores = probs + bias[:, None]
        self.cb("ffn_moe_logits", logits, il)
        self.cb("ffn_moe_probs", probs, il)
        self.cb("ffn_moe_probs_biased", sel_scores, il)

        gate_exps = self.t(f"blk.{il}.ffn_gate_exps.weight")     # (n_expert, n_ff, n_embd)
        up_exps = self.t(f"blk.{il}.ffn_up_exps.weight")
        down_exps = self.t(f"blk.{il}.ffn_down_exps.weight")
        n_ff = gate_exps.shape[1]

        sel_all = np.empty((c.n_expert_used, cur.shape[1]), dtype=np.int64)
        w_all = np.empty((c.n_expert_used, cur.shape[1]), dtype=F32)
        moe = np.zeros_like(cur)
        for t in range(cur.shape[1]):
            sel = top_k_desc(sel_scores[:, t], c.n_expert_used)
            sel_all[:, t] = sel
            w = probs[sel, t]
            if c.expert_weights_norm:
                w_sum = float(w.sum())
                COVER["moe_renorm_clamp_hits"] += int(w_sum < 6.103515625e-5)
                w = w / max(w_sum, 6.103515625e-5)
            # near-tie: the boundary between the last selected and first rejected expert is
            # closer than f32 noise can resolve - the selection can flip on rounding alone
            ordered = np.sort(sel_scores[:, t])[::-1]
            if len(ordered) > c.n_expert_used:
                COVER["moe_selection_near_ties"] += int(
                    ordered[c.n_expert_used - 1] - ordered[c.n_expert_used] < 1e-4)
            w = (w * F32(c.expert_weights_scale)).astype(F32)
            w_all[:, t] = w
            x = cur[:, t]
            acc = np.zeros(c.n_embd, dtype=np.float64)
            for i, e in enumerate(sel):
                h = swiglu_clamp(gate_exps[e] @ x, up_exps[e] @ x, c.swiglu_limit_exp)
                acc += float(w[i]) * (down_exps[e] @ h)
            moe[:, t] = acc.astype(F32)
        self.dumps[f"ffn_moe_topk-{il}"] = sel_all
        self.dumps[f"ffn_moe_weights-{il}"] = w_all
        self.cb("ffn_moe_out", moe, il)

        gate = self.layer_t(il, "ffn_gate_shexp") @ cur
        up = self.layer_t(il, "ffn_up_shexp") @ cur
        shexp = self.layer_t(il, "ffn_down_shexp") @ swiglu_clamp(gate, up, c.swiglu_limit_shexp)
        self.cb("ffn_shexp", shexp, il)
        out = (moe + shexp).astype(F32)
        self.cb("ffn_out", out, il)
        return out

    # ================================ the trunk ================================

    def forward(self, tokens: list[int]) -> np.ndarray:
        """One teacher-forced pass; returns (vocab, T) float32 logits."""
        c = self.cfg
        emb = self.t("token_embd.weight")                        # (vocab, n_embd)
        x = emb[tokens].T.astype(F32)                            # (n_embd, T)
        R = np.repeat(x[:, None, :], c.hc, axis=1)               # (n_embd, hc, T) - hc_init
        self.cb("hc_init", R)

        for il in range(c.n_layers):
            residual = R
            mixed, post, comb = self.hc_pre(R, self.layer_t(il, "hc_attn_fn"), self.layer_t(il, "hc_attn_scale"),
                                            self.layer_t(il, "hc_attn_base"), il, "attn")
            cur = rms_norm(mixed, self.layer_t(il, "attn_norm"), c.norm_rms_eps)
            self.cb("attn_norm", cur, il)

            if c.is_recr(il):
                cur = self.kda_layer(cur, il)
            else:
                # glm5-next.cpp allows a non-full DSA layer to reuse the previous full layer's
                # selection (prev_sel); this reference implements the full indexer only, and the
                # released GLM-5.3-Flash is full on every layer - refuse anything else loudly
                # rather than silently indexing with the wrong weights
                if not c.is_indexer_full(il):
                    raise NotImplementedError(
                        f"layer {il}: shared (non-full) k-pool indexer not implemented; the "
                        "reference and the runner both implement the full indexer only")
                cur = self.dsa_layer(cur, il)

            R = self.hc_post(cur, residual, post, comb)
            self.cb("hc_attn_post", R, il)

            residual = R
            mixed, post, comb = self.hc_pre(R, self.layer_t(il, "hc_ffn_fn"), self.layer_t(il, "hc_ffn_scale"),
                                            self.layer_t(il, "hc_ffn_base"), il, "ffn")
            cur = rms_norm(mixed, self.layer_t(il, "ffn_norm"), c.norm_rms_eps)
            self.cb("ffn_norm", cur, il)

            cur = self.dense_ffn(cur, il) if il < c.n_layer_dense_lead else self.moe_ffn(cur, il)

            R = self.hc_post(cur, residual, post, comb)
            self.cb("l_out", R, il)

        head = R.mean(axis=1)                                    # build_hc_mean: plain mean over streams
        self.cb("hc_head", head)
        cur = rms_norm(head, self.t("output_norm.weight"), c.norm_rms_eps)
        self.cb("result_norm", cur)
        logits = self.t("output.weight") @ cur                   # output is (vocab, n_embd) from tensor()
        self.cb("result_output", logits)
        return logits


def main() -> int:
    ap = argparse.ArgumentParser(description="glm5-next reference forward pass")
    ap.add_argument("model", help="a glm5-next GGUF (real or tools/glm_synth_gguf.py's synthetic)")
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8", help="comma-separated token ids")
    ap.add_argument("--dump-dir", default=None, help="write every cb() seam here as .npy")
    ap.add_argument("--dump-raw-dir", default=None,
                    help="write every cb() seam as raw ggml-order f32 (+ manifest.txt), for glm_layer_parity")
    ap.add_argument("--greedy", type=int, default=0, help="continue greedily for N tokens and print them")
    args = ap.parse_args()

    m = Glm5Next(args.model)
    cfg = m.cfg
    print(f"glm5-next: n_embd {cfg.n_embd}, layers {cfg.n_layers} "
          f"({sum(cfg.is_recr(i) for i in range(cfg.n_layers))} KDA / "
          f"{sum(not cfg.is_recr(i) for i in range(cfg.n_layers))} DSA), "
          f"hc {cfg.hc} (sinkhorn {cfg.hc_sinkhorn_iters}), experts {cfg.n_expert} x top{cfg.n_expert_used}, "
          f"kda head {cfg.kda_head_dim} lb {cfg.kda_gate_lower_bound}, mla {cfg.qk_nope}/kv {cfg.kv_lora_rank}, "
          f"indexer {cfg.idx_n_head}x{cfg.idx_key_dim} top {cfg.idx_top_k} kpool {cfg.idx_kpool}")
    tokens = [int(x) for x in args.tokens.split(",")]
    logits = m.forward(tokens)
    if not np.isfinite(logits).all():
        print("glm.py: NON-FINITE LOGITS")
        return 1
    print(f"logits {logits.shape}: mean {logits.mean():.6f} std {logits.std():.6f} "
          f"argmax per token: {[int(i) for i in logits.argmax(axis=0)]}")
    if args.greedy:
        toks = list(tokens)
        for _ in range(args.greedy):
            toks.append(int(np.argmax(m.forward(toks), axis=0)[-1]))
        print("greedy continuation:", toks[len(tokens):])
    if args.dump_dir:
        d = pathlib.Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, arr in m.dumps.items():
            np.save(d / f"{name}.npy", arr)
        print(f"dumps: {len(m.dumps)} tensors -> {d}")
    if args.dump_raw_dir:
        # Raw f32 binaries for the C++ layer parity (glm_layer_parity): every dump is written in the
        # GGML layout (features fastest) by fully reversing the numpy axes - a (features, T) dump
        # becomes (T, features) on disk, element (e, t) at e + features*t - and the manifest carries
        # each tensor's reversed shape so C++ needs no reshape logic.
        d = pathlib.Path(args.dump_raw_dir)
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for name, arr in m.dumps.items():
            rev = np.ascontiguousarray(arr.transpose(tuple(range(arr.ndim - 1, -1, -1))))
            if arr.dtype == np.int64:
                rev = rev.astype(np.int32)
            rev.astype(F32 if rev.dtype != np.int32 else np.int32).tofile(d / f"{name}.f32")
            lines.append(f"{name} {' '.join(str(x) for x in reversed(arr.shape))}")
        (d / "manifest.txt").write_text("\n".join(lines) + "\n")
        print(f"raw dumps: {len(m.dumps)} tensors -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
