// include/strata/qwen35/qwen35.hpp - the Qwen35MoE (Ornith-1.5) forward pass, layer by layer.
//
// **WHY THIS IS A SEPARATE LIBRARY.**  The Qwen4Exp engine (`layer.cpp`, `session.cpp`) is built around the
// pack format, the four-stream hyper-connection residual, PLE and QSA.  None of that exists in Qwen35.  The
// reference for every op here is the pinned llama.cpp revision's
//   * src/models/qwen35moe.cpp        (the layer order)
//   * src/models/delta-net-base.cpp   (gated delta net, conv and l2 norm)
//   * ggml/src/ggml-cpu/ops.cpp       (ggml_compute_forward_gated_delta_net_one_chunk, the exact recurrence)
// and the numbers are validated against them, stage by stage, by the tests beside it.
//
// Weight pointers are ROW-MAJOR with the OUTPUT row contiguous, i.e. GGUF's own layout: for a tensor of GGUF
// shape [ne0, ne1], a row `o` of `ne1` outputs occupies `ne0` contiguous floats at `w + o*ne0`.  Every helper
// below takes `n_in` and `n_out` and follows that convention, so a caller can hand over the mmapped GGUF
// blocks once decoding is added.
#pragma once

#include "strata/core/qwen35.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen35 {

using core::Qwen35Geometry;

// ---------------------------------------------------------------- primitives

/// y = x * w / sqrt(mean(x^2) + eps), the `LLM_NORM_RMS` every Qwen35 block uses.  `w` may be null (no weight).
void rms_norm(const float* x, const float* w, int64_t n, float eps, float* y);
/// The GDN's l2 norm: x / sqrt(sum(x^2) + eps) (llama.cpp's `build_gdn_l2_norm`).
void l2_norm(const float* x, int64_t n, float eps, float* y);
/// y[o] = sum_i W[i,o] * x[i]; a plain matvec, the reference the quantized path must reproduce.
void matvec(const float* w, const float* x, int64_t n_in, int64_t n_out, float* y);
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return std::log1p(std::exp(x)); }

// ---------------------------------------------------------------- one GDN layer

/// The recurrent layer's weights, in GGUF order.  `ssm_conv` is [conv_channels, d_conv] (GGUF [d_conv, conv_channels]):
/// channel `c`'s `d_conv` taps occupy `ssm_conv + c*d_conv`.
struct GdnLayerWeights {
    const float* attn_norm = nullptr;    // [n_embd]
    const float* wqkv = nullptr;         // rows: qkv_dim, each n_embd
    const float* wgate = nullptr;        // rows: value_dim, each n_embd
    const float* ssm_conv = nullptr;     // channel-major: d_conv taps per channel
    const float* ssm_dt = nullptr;       // [v_heads]
    const float* ssm_a = nullptr;        // [v_heads]
    const float* ssm_beta = nullptr;     // rows: v_heads, each n_embd
    const float* ssm_alpha = nullptr;    // rows: v_heads, each n_embd
    const float* ssm_norm = nullptr;     // [head_v_dim]
    const float* ssm_out = nullptr;      // rows: n_embd, each value_dim
};

/// The layer's persistent state.  `conv` is [d_conv-1, conv_channels] oldest-first; `rec` is
/// [v_heads][S_v][S_v] with the element (j, i) at `rec + h*S*S + j*S + i`, the transposed layout the ggml
/// kernel uses (`s_out[j*S_v + i] = S[i][j]`).
struct GdnState {
    std::vector<float> conv;
    std::vector<float> rec;

    void resize(const Qwen35Geometry& g) {
        conv.assign((size_t) (g.ssm_conv_kernel - 1) * g.conv_channels(), 0.0f);
        rec.assign((size_t) g.ssm_dt_rank * g.ssm_state * g.ssm_state, 0.0f);
    }
    void zero() {
        std::fill(conv.begin(), conv.end(), 0.0f);
        std::fill(rec.begin(), rec.end(), 0.0f);
    }
};

/// One token through one GDN layer: `x` (n_embd) -> `out` (n_embd), updating `st`.
void gdn_layer(const Qwen35Geometry& g, const GdnLayerWeights& w, GdnState& st, const float* x, float* out);

// ---------------------------------------------------------------- one full-attention layer

struct AttnLayerWeights {
    const float* attn_norm = nullptr;    // [n_embd]
    const float* wq = nullptr;           // rows: 2*n_head*head_dim (q | gate per head), each n_embd
    const float* wk = nullptr;           // rows: n_head_kv*head_dim
    const float* wv = nullptr;           // rows: n_head_kv*head_dim
    const float* wo = nullptr;           // rows: n_embd, each n_head*head_dim
    const float* q_norm = nullptr;       // [head_dim]
    const float* k_norm = nullptr;       // [head_dim]
};

/// Per-layer KV cache.  `k`/`v` are [max_cells][n_head_kv*head_dim] (one contiguous head block per KV head).
struct AttnState {
    std::vector<float> k, v;
    int64_t n = 0;
    void resize(int64_t max_cells, const Qwen35Geometry& g) {
        const int64_t kv = g.n_head_kv * g.head_dim;
        k.assign((size_t) max_cells * kv, 0.0f);
        v.assign((size_t) max_cells * kv, 0.0f);
        n = 0;
    }
    void zero() { std::fill(k.begin(), k.end(), 0.0f); std::fill(v.begin(), v.end(), 0.0f); n = 0; }
};

/// NEOX partial RoPE on one head vector in place (llama.cpp's `GGML_ROPE_TYPE_NEOX`: rotate the pairs
/// `(v[k], v[k + n_rot/2])` for `k < n_rot/2`, theta_k = pos * base^(-2k/n_rot)).
void rope_neox(float* v, int64_t head_dim, int64_t n_rot, float base, int64_t pos);

/// One token through one full-attention layer: `x` -> `out`, appending to `st`.
void attn_layer(const Qwen35Geometry& g, const AttnLayerWeights& w, AttnState& st, const float* x, float* out);

// ---------------------------------------------------------------- one MoE block

/// One routed expert's three matrices, each output-row-major.
struct ExpertWeights {
    const float* gate = nullptr;   // rows: n_ff_exp, each n_embd
    const float* up = nullptr;     // rows: n_ff_exp, each n_embd
    const float* down = nullptr;   // rows: n_embd, each n_ff_exp
};

struct MoeLayerWeights {
    const float* gate_inp = nullptr;       // rows: n_expert, each n_embd (F32 in the artifact)
    const float* gate_shexp = nullptr;     // rows: n_ff_shexp, each n_embd
    const float* up_shexp = nullptr;       // rows: n_ff_shexp, each n_embd
    const float* down_shexp = nullptr;     // rows: n_embd, each n_ff_shexp
    const float* gate_inp_shexp = nullptr; // [n_embd]
    const ExpertWeights* experts = nullptr;  // [n_expert]
};

/// The block: softmax over ALL experts, top-k, renormalize, silu-gated experts, gated shared expert.
void moe_layer(const Qwen35Geometry& g, const MoeLayerWeights& w, const float* x, float* out);

// ---------------------------------------------------------------- the trunk

/// All 40 layers' weights.  `gdn[l]` is used when `g.is_recurrent(l)`, `attn[l]` otherwise; `moe[l]` always.
struct TrunkWeights {
    const float* token_embd = nullptr;   // rows: n_vocab, each n_embd
    const float* output_norm = nullptr;  // [n_embd]
    const float* output = nullptr;       // rows: n_vocab, each n_embd
    const float* attn_norm = nullptr;        // [n_layers][n_embd], row-major
    const float* post_attn_norm = nullptr;   // [n_layers][n_embd]
    std::vector<GdnLayerWeights> gdn;
    std::vector<AttnLayerWeights> attn;
    std::vector<MoeLayerWeights> moe;
};

/// Persistent per-sequence state: one GDN state per recurrent layer, one KV cache per attention layer.
struct TrunkState {
    std::vector<GdnState> gdn;
    std::vector<AttnState> attn;
    void reset(const Qwen35Geometry& g) {
        gdn.assign((size_t) g.n_layers, {});
        attn.assign((size_t) g.n_layers, {});
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (g.is_recurrent(l)) gdn[l].resize(g);
            else attn[l].resize(g.context_length, g);
        }
    }
    void zero(const Qwen35Geometry& g) {
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (g.is_recurrent(l)) gdn[l].zero();
            else attn[l].zero();
        }
    }
};

/// `token` through the whole trunk at sequence position `pos` (the KV/cache position).  `logits` is n_vocab.
void trunk_forward(const Qwen35Geometry& g, const TrunkWeights& w, TrunkState& st, int64_t token, float* logits);

}  // namespace strata::qwen35
