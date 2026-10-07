// include/strata/kernels/glm.hpp - GLM-5.3-Flash's own layer arithmetic.
//
// Three things in a glm5-next block have no qwen4exp analogue, and each is here rather than bent into the
// existing kernel families:
//
//   * HYPER-CONNECTIONS (mHC).  qwen4exp's `gr_read`/`gr_write` is a low-rank mixer with a learned gate and a
//     mean injection.  GLM's is a Sinkhorn-normalised 4x4 routing matrix with its own base/scale, computed per
//     token from a 24-wide projection - and it OWNS every residual, so getting it wrong produces plausible
//     garbage rather than a crash.  Ported from ggml's `ggml_hc_pre` / `ggml_hc_post` (ik_llama.cpp
//     ggml/src/ggml.c), not re-derived.
//   * KDA (the 34 linear layers): a delta-rule recurrence with a per-head state, a depthwise conv over three
//     separate q/k/v streams and a bounded decay gate.  Nothing in the engine resembles it.
//   * The SwiGLU clamp and the sigmoid router, which are small but change every FFN in the model.
//
// LAYOUT.  Every buffer is [width, tokens] with the width fastest, which is the engine's convention and ggml's.
// `x` is the mHC residual: `hc` streams of `n_embd` floats each, stream `s` at `s * n_embd`.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The numbers every mHC call needs.  `mix` is `hc * (2 + hc)` - 24 at hc 4 - and the projection that produces
/// it is `hc_*_fn`, a [n_embd * hc, mix] matrix.
struct GlmHcShapes {
    int64_t n_embd = 4096;
    int64_t hc = 4;
    int64_t mix = 24;
    int64_t sinkhorn_iters = 20;
    /// `hyper_connection.epsilon` - the `pre` bias and the eps on every Sinkhorn normalise.  1e-6 in the GGUF.
    float eps = 1e-6f;
    /// `attention.layer_norm_rms_epsilon` - a DIFFERENT number, and the reason this struct needs two.  The
    /// weightless RMSNorm that feeds `hc_*_fn` is the reference's plain `ggml_rms_norm(x, hparams.f_norm_rms_eps)`
    /// and takes the model's norm eps (1e-5 here), not the hyper-connection one.  Running it at 1e-6 measured
    /// 8% high on layer 0's `mixes`, where the flattened row's RMS is ~7e-3 and eps is 10% of its square.
    float norm_eps = 1e-5f;
};

/// `mixes` [mix, T] -> the three per-token maps, each stored width-fastest:
///   `pre`  [hc, T]      sigmoid(a * scale[0] + base[i]) + eps
///   `post` [hc, T]      2 * sigmoid(a * scale[1] + base[hc + i])
///   `comb` [hc*hc, T]   softmax rows, then a column pass, then `iters - 1` row/column rounds, eps on every
///                       normalise - so the last pass is a COLUMN one.  Entry [j*hc + i] is source j, dest i.
void glm_hc_pre(const float* mixes, const float* scale, const float* base, float* pre, float* post, float* comb,
                const GlmHcShapes& s, int64_t T, void* stream);

/// The mHC residual write, which is also the READ of the block's input: `x` [n_embd, T] is the sublayer's
/// output, `residual` [n_embd*hc, T] the streams the sublayer was fed from.  `out[i0 + i*n_embd] =
/// x[i0] * post[i] + sum_j comb[j*hc + i] * residual[i0 + j*n_embd]`.
void glm_hc_post(const float* x, const float* post, const float* residual, const float* comb, float* out,
                 const GlmHcShapes& s, int64_t T, void* stream);

/// The stream the sublayer is fed: `out[i0] = sum_j pre[j] * x[i0 + j*n_embd]`.
void glm_hc_mix(const float* x, const float* pre, float* out, const GlmHcShapes& s, int64_t T, void* stream);

/// The collapse before the head: the MEAN of the streams, not their sum.
void glm_hc_sum(const float* x, float* out, const GlmHcShapes& s, int64_t T, void* stream);

/// The weightless RMSNorm over the flattened streams.  `out_bf16` is the normed activation as bf16 and
/// `out_f32` the SAME value in f32 - both are written from one product, because the `hc_*_fn` weight is Q8_0 in
/// the file (so the GEMV wants a quantized image, built from the f32) while a caller that serves it natively
/// wants the bf16.  `out_f32` may be null.  `T` is the token count.
void glm_hc_norm_bf16(const float* x, uint16_t* out_bf16, float* out_f32, const GlmHcShapes& s, int64_t T,
                      void* stream);

// ---- KDA, the linear layers ----------------------------------------------------------------------------

/// Everything a KDA layer's state needs, in the shape the kernels index it by.
struct GlmKdaShapes {
    int64_t n_embd = 4096;
    int64_t n_head = 64;        ///< 64 heads, each `head_dim` wide, on q, k and v alike
    int64_t head_dim = 128;
    int64_t conv_kernel = 4;
    float eps = 1e-5f;          ///< the RMS epsilon, reused by the L2 normalisation
    float gate_floor = -5.0f;   ///< the decay gate is bounded to (floor, 0)
    int64_t groups = 1;         ///< depthwise: one group per channel
};

/// The bounded decay gate.  `a` is `ssm_a` [n_head] (already `-exp(A_log)` in the file), `raw` [n_v, T] and
/// `dt` [n_v] the bias, both laid out `head_dim` fastest.  `g = floor * sigmoid(-(a_head * (raw + dt)))` with
/// `floor = -5.0` - the gate is in (floor, 0) and the nesting of the signs is the whole content: a port that
/// negates once produces a model that runs and is wrong.  `floor_mag` is the NEGATIVE bound, i.e. -5.0.
void glm_kda_gate(const float* raw, const float* dt, const float* a, float* gate, int64_t n_v, int64_t n_head,
                  int64_t head_dim, float floor_mag, int64_t T, void* stream);

/// The depthwise causal convolution over the CONCATENATION [q; k; v], then SiLU, in place on `qkv` [3*n_v, T].
///
/// The three streams have their own weights (`ssm_conv1d_{q,k,v}`, each [kernel, n_v]) but ONE shared state,
/// because the reference concatenates them into a [kernel, 3*n_v] filter before convolving.  `state` is
/// [kernel - 1, 3*n_v] with the TAP fastest: `state[k*(3*n_v) + ch]` is the input to channel `ch` at time
/// `t - (kernel - 1) + k`, oldest first.  There is no bias tensor in the file, which is why there is none here.
void glm_kda_conv_silu(float* qkv, const float* wq, const float* wk, const float* wv, float* state, int64_t n_v,
                       int64_t T, int64_t kernel, void* stream);

/// The SAME convolution when the three streams are three separate arrays with a token stride of `ld`.
///
/// `glm_kda_conv_silu` is this function with `q, k, v = qkv, qkv + n_v, qkv + 2*n_v` and `ld = 3*n_v`, which is
/// not a special case but the interleaved spelling of it - the arithmetic is identical address for address, so
/// the two callers cannot drift.  The split form exists for the batched prefill path: a projection writes `nt`
/// contiguous columns of width `n_out`, so its output lands as an `nt x n_v` block, and interleaving three of
/// those into the `[q;k;v]` the conv wants would be a fourth kernel and a second pass over the data.
///
/// `state` is unchanged in both forms - `[kernel - 1, 3*n_v]`, the TAP fastest, the three streams sharing it,
/// because the reference concatenates their filters before convolving.
void glm_kda_conv_silu3(float* q, float* k, float* v, const float* wq, const float* wk, const float* wv,
                        float* state, int64_t n_v, int64_t ld, int64_t T, int64_t kernel, void* stream);

/// The KDA recurrence itself - see `src/kernels/cuda/glm_delta.cu` for the six lines and which of them carry
/// `scale`.
///
/// q, k, v, gate and out are all `[n_v, T]` with `n_v = head_dim * n_head` and head_dim fastest; `beta` is
/// [n_head, T]; `state` is per head, [head_dim, head_dim] with the ROW fastest - `state[h*d*d + row + col*d]`.
/// `state` is read and written in place, so the caller owns zeroing it at the start of a sequence.
void glm_kda_delta(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                   float* state, float* out, int64_t head_dim, int64_t n_head, int64_t T, void* stream);

/// L2-normalise the q and k halves in place, per head.  **`eps` IS A FLOOR ON THE NORM, NOT A TERM INSIDE THE
/// SUM**: the reference is `x / max(sqrt(sum(x^2)), eps)`, so a near-zero head is left near-zero rather than
/// amplified to unit length.  Writing `1/sqrt(sum + eps)` - the RMSNorm form, and the one a reader expects -
/// agrees on every head with a normal-sized norm and diverges exactly where it matters.
void glm_kda_l2norm(float* q, float* k, int64_t n_head, int64_t head_dim, float eps, int64_t T, void* stream);

// ---- absorbed NoPE MLA, the 11 full-attention layers ----------------------------------------------------

/// Write one token's (or T tokens') `kv_lora`-wide latent into the fp16 cache at row `pos`.
///
/// **THE CACHE IS FP16 BECAUSE THE REFERENCE'S IS.**  `ggml_cpy(kv_cmpr, k_cache_view)` stores into
/// `kv_self.k_l[il]`, whose type is the `--cache-type-k` setting and f16 by default, so the rounding is part
/// of the model the oracle computes and a fp32 cache here would be a different one.  `x` is `[kv_lora, T]`.
void glm_mla_cache_store(const float* x, uint16_t* cache, int64_t pos, int64_t kv_lora, void* stream);

/// The same write for a run of `T` tokens: `x` is `[kv_lora, T]` and row `t` lands at cache row `pos + t`.
/// A chunk's rows are consecutive by construction - the chunk is the positions `pos .. pos + T - 1` - so this
/// is one launch over the whole chunk rather than `T` launches of a kernel that moves 1 KiB each.
void glm_mla_cache_store_t(const float* x, uint16_t* cache, int64_t pos, int64_t kv_lora, int64_t T, void* stream);

/// The attention itself: `out[l, t, h] = sum_s p(t,s) * K[s, l]` with `p` the softmax over the cache of
/// `scale * sum_l Qcur[l, t, h] * K[s, l]`.
///
/// `q` and `out` are `[kv_lora, T, n_head]` with kv_lora fastest; `k_cache` is `[n_kv, kv_lora]` fp16 with the
/// latent fastest.  `n_kv` is the TOTAL number of tokens in the cache (including this call's) and `pos_base`
/// the absolute position of `t = 0`, so query `t` sees `0 .. pos_base + t`.  `scale` is
/// `1/sqrt(n_embd_head_k_full)` - the per-head width BEFORE absorption (256), not `kv_lora`.
void glm_mla_attn(const float* q, const uint16_t* k_cache, float* out, int64_t n_head, int64_t kv_lora,
                  int64_t n_kv, int64_t T, int64_t pos_base, float scale, void* stream);

// ---- the two elementwise steps the FFN adds ---------------------------------------------------------------

/// `gate = silu(gate) * up`, in place on `gate`.  No clamp: see the note in `glm_elt.cu`.
void glm_swiglu(float* gate, const float* up, int64_t n, void* stream);

/// `dst += src`.
void glm_add_inplace(float* dst, const float* src, int64_t n, void* stream);

/// `dst[i] *= sigmoid(z[i])` - KDA's output gate, which is a SIGMOID here (qwen4exp's GDN uses one too, but
/// only after a different norm; the pair is easy to conflate and neither output looks wrong).
void glm_sigmoid_mul(float* dst, const float* z, int64_t n, void* stream);

/// `y = W x` for a plain F32 `W` laid out `[n_out, n_in]` row-major - the few 2-D weights the file stores at
/// full precision.  glm5-next's router is the one that matters: `ffn_gate_inp.weight` is F32 here, where
/// qwen4exp's is BF16 and goes through `project_bf16`.  Reading an F32 router as BF16 gives 288 plausible
/// logits and a different top-8, which is a different model and not a visible error.
void glm_f32_gemv(const float* x, const float* w, float* y, int64_t n_in, int64_t n_out, void* stream);

/// The sigmoid router, which is not qwen4exp's softmax one:
///
///     probs  = sigmoid(logits)              <- no softmax anywhere in the model
///     select = probs + bias                 <- the bias steers SELECTION only
///     ids    = top_k(select, k)             <- ties by ascending index
///     w      = probs[ids];  w /= sum(w);  w *= scale
///
/// The weights are the UN-BIASED probabilities, normalised by their SUM (not their max), and `scale` (2.5) is
/// applied after.  `bias` may be null.  `ids` is `[k]` int32, `weights` `[k]` f32, both device pointers.
void glm_router_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* weights, int64_t n_expert,
                             int64_t k, float scale, void* stream);

}  // namespace strata::kernels
