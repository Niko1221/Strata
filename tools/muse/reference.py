"""Independent FP32 Muse Spark text graph, specified by the standard GGUF.

Batch/causal NumPy evaluation (not a port/import of the candidate streaming
runtime). Matrix arrays are [out, in], expert arrays [expert, out, in].
Norm scales, Q scale, router multiplier and alpha/beta are already folded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.gguf_reader import GGUFFile

F = np.float32


def load_f32(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read local fixture tensors; deliberately not a production quantized loader."""
    gguf = GGUFFile(path)
    tensors = {}
    with Path(path).open("rb") as stream:
        for info in gguf.tensors:
            if info.type_name != "F32":
                raise ValueError(f"{info.name}: oracle only accepts F32")
            if info.name in tensors or not info.shape or any(d <= 0 for d in info.shape):
                raise ValueError(f"{info.name}: duplicate or invalid shape")
            stream.seek(gguf.data_start + info.offset)
            raw = stream.read(info.elements * 4)
            if len(raw) != info.elements * 4:
                raise ValueError(f"{info.name}: truncated tensor")
            tensors[info.name] = np.frombuffer(raw, dtype="<f4").reshape(tuple(reversed(info.shape))).copy()
    return gguf.metadata, tensors


def rms(x: np.ndarray, eps: float, weight: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=F) + F(eps))
    return y if weight is None else y * weight


def input_scaled_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return rms(x * weight, eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable also for tests with extreme router scores.
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z)).astype(F)


def rope(x: np.ndarray, base: float) -> np.ndarray:
    """NeoX half-split RoPE; x is [token, head, head_dim]."""
    half = x.shape[-1] // 2
    frequency = F(base) ** (-np.arange(half, dtype=F) / F(half))
    angle = np.arange(x.shape[0], dtype=F)[:, None, None] * frequency[None, None, :]
    c, s = np.cos(angle), np.sin(angle)
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * c - b * s, b * c + a * s), axis=-1)


def route(x: np.ndarray, router: np.ndarray, bias: np.ndarray,
          top_k: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    probabilities = sigmoid(x @ router.T)
    choice = probabilities + bias
    # Explicit stable tie order; the generated fixture has no selection ties.
    selected = np.argsort(-choice, axis=-1, kind="stable")[:, :top_k]
    mixing = np.take_along_axis(probabilities, selected, axis=-1)
    mixing = mixing / (mixing.sum(axis=-1, keepdims=True, dtype=F) + F(1e-15))
    return selected, mixing * F(scale)


def validate(meta: dict, tensors: dict[str, np.ndarray], tokens: list[int]) -> dict:
    """Check shape/metadata contracts before arithmetic; return unprefixed keys."""
    if meta.get("general.architecture") != "muse-spark":
        raise ValueError("architecture must be muse-spark")
    m = {k.removeprefix("muse-spark."): v for k, v in meta.items() if k.startswith("muse-spark.")}
    dimensions = ("block_count", "context_length", "embedding_length", "attention.head_count",
                  "attention.head_count_kv", "attention.key_length", "attention.value_length",
                  "moe_latent_size", "expert_feed_forward_length", "expert_count",
                  "expert_used_count", "attention.sliding_window")
    for key in dimensions:
        if type(m.get(key)) is not int or not 0 < m[key] <= 65536:
            raise ValueError(f"invalid dimension: {key}")
    for key in ("attention.layer_norm_rms_epsilon", "post_norm_epsilon", "rope.freq_base_swa",
                "logit_scale", "final_logit_softcapping", "expert_weights_scale"):
        value = m.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid positive scalar: {key}")
    h, nh, nk, d = (m[k] for k in ("embedding_length", "attention.head_count",
                                   "attention.head_count_kv", "attention.key_length"))
    latent, ff, ne = (m[k] for k in ("moe_latent_size", "expert_feed_forward_length", "expert_count"))
    if nh % nk or d % 2 or m["attention.value_length"] != d or m["expert_used_count"] > ne:
        raise ValueError("invalid GQA/head/expert geometry")
    pattern = m.get("attention.sliding_window_pattern")
    if not isinstance(pattern, list) or len(pattern) != m["block_count"] or any(type(v) is not bool for v in pattern):
        raise ValueError("sliding window pattern must have one bool per layer")
    if m.get("expert_gating_func") != 2 or m.get("expert_weights_norm") is not True:
        raise ValueError("oracle requires sigmoid routing and normalized expert weights")
    emb = tensors.get("token_embd.weight")
    if emb is None or emb.ndim != 2:
        raise ValueError("missing/invalid embedding")
    vocab = emb.shape[0]
    if not tokens or len(tokens) > m["context_length"] or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
        raise ValueError("tokens must be nonempty in-vocabulary integers within context")
    shapes = {"token_embd.weight": (vocab, h), "output_norm.weight": (h,), "output.weight": (vocab, h)}
    layer = {
        "attn_norm.weight": (h,), "attn_q.weight": (nh*d, h),
        "attn_k.weight": (nk*d, h), "attn_v.weight": (nk*d, h),
        "attn_q_norm.weight": (d,), "attn_k_norm.weight": (d,),
        "attn_gate.weight": (nh*d, h), "attn_output.weight": (h, nh*d),
        "attn_resid_alpha.weight": (h,), "attn_resid_beta.weight": (h,),
        "ffn_norm.weight": (h,), "post_ffw_norm.weight": (h,),
        "ffn_resid_alpha.weight": (h,), "ffn_resid_beta.weight": (h,),
        "ffn_gate_inp.weight": (ne, h), "exp_probs_b.bias": (ne,),
        "ffn_latent_down.weight": (latent, h), "ffn_latent_up.weight": (h, latent),
        "ffn_pre_exp_norm.weight": (latent,), "ffn_post_exp_norm.weight": (latent,),
        "ffn_gate_exps.weight": (ne, ff, latent), "ffn_up_exps.weight": (ne, ff, latent),
        "ffn_down_exps.weight": (ne, latent, ff),
    }
    for i in range(m["block_count"]):
        shapes.update({f"blk.{i}.{name}": shape for name, shape in layer.items()})
    if set(tensors) != set(shapes):
        raise ValueError("missing or unexpected tensors")
    for name, shape in shapes.items():
        value = tensors[name]
        if value.shape != shape or value.dtype != F or not np.isfinite(value).all():
            raise ValueError(f"{name}: expected finite FP32 {shape}, got {value.shape}/{value.dtype}")
    return m


def forward(meta: dict, tensors: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    m = validate(meta, tensors, tokens)
    eps, post_eps = m["attention.layer_norm_rms_epsilon"], m["post_norm_epsilon"]
    heads, kvheads, dim = m["attention.head_count"], m["attention.head_count_kv"], m["attention.key_length"]
    n = len(tokens)
    x = rms(tensors["token_embd.weight"][tokens], eps)
    for layer in range(m["block_count"]):
        w = {name.removeprefix(f"blk.{layer}."): value for name, value in tensors.items()
             if name.startswith(f"blk.{layer}.")}
        a = rms(x, eps, w["attn_norm.weight"])
        q = (a @ w["attn_q.weight"].T).reshape(n, heads, dim)
        k = (a @ w["attn_k.weight"].T).reshape(n, kvheads, dim)
        v = (a @ w["attn_v.weight"].T).reshape(n, kvheads, dim)
        q = rms(q, eps, w["attn_q_norm.weight"])
        k = rms(k, eps, w["attn_k_norm.weight"])
        local = m["attention.sliding_window_pattern"][layer]
        if local:
            q, k = rope(q, m["rope.freq_base_swa"]), rope(k, m["rope.freq_base_swa"])
        k = np.repeat(k, heads // kvheads, axis=1)
        v = np.repeat(v, heads // kvheads, axis=1)
        attended = np.empty((n, heads, dim), dtype=F)
        for t in range(n):
            start = max(0, t - m["attention.sliding_window"] + 1) if local else 0
            # Batch all heads, but compute each causal token against its own range.
            scores = np.einsum("hd,shd->hs", q[t], k[start:t+1]) * F(dim ** -0.5)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True, dtype=F)
            attended[t] = np.einsum("hs,shd->hd", probabilities, v[start:t+1])
        attended = rms(attended, eps).reshape(n, heads * dim)
        gate = sigmoid(a @ w["attn_gate.weight"].T)
        attention = (attended * gate) @ w["attn_output.weight"].T
        attention = rms(attention, post_eps)
        stream = x * w["attn_resid_alpha.weight"] + attention * w["attn_resid_beta.weight"]
        ffn = rms(stream, eps, w["ffn_norm.weight"])
        selected, mixing = route(ffn, w["ffn_gate_inp.weight"], w["exp_probs_b.bias"],
                                 m["expert_used_count"], m["expert_weights_scale"])
        latent = rms(ffn @ w["ffn_latent_down.weight"].T, eps, w["ffn_pre_exp_norm.weight"])
        mixed = np.zeros_like(latent)
        for t in range(n):
            for slot, expert in enumerate(selected[t]):
                g = latent[t] @ w["ffn_gate_exps.weight"][expert].T
                u = latent[t] @ w["ffn_up_exps.weight"][expert].T
                expert_out = ((g * sigmoid(g)) * u) @ w["ffn_down_exps.weight"][expert].T
                expert_out = input_scaled_norm(expert_out, w["ffn_post_exp_norm.weight"], post_eps)
                mixed[t] += expert_out * mixing[t, slot]
        branch = mixed @ w["ffn_latent_up.weight"].T
        branch = input_scaled_norm(branch, w["post_ffw_norm.weight"], post_eps)
        x = stream * w["ffn_resid_alpha.weight"] + branch * w["ffn_resid_beta.weight"]
    final = rms(x, eps, tensors["output_norm.weight"])
    logits = (final @ tensors["output.weight"].T) * F(m["logit_scale"])
    cap = F(m["final_logit_softcapping"])
    logits = cap * np.tanh(logits / cap)
    if logits.dtype != F or not np.isfinite(logits).all():
        raise ValueError("nonfinite/non-FP32 oracle output")
    return logits
