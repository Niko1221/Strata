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
// **THE WEIGHTS STAY QUANTIZED.**  A `Mat` is a GGUF tensor: raw blocks, a ggml type id and its shape.  A
// type of 0 is plain F32 (what the unit tests use); any other type is decoded by ggml-cpu's own type traits
// (`qwen35_ggml.cpp`), so an IQ4_XS or Q4_K row means exactly what it means in llama.cpp.  Dequantizing the
// 35B model to float would not fit and is never done.
#pragma once

#include "strata/core/qwen35.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::qwen35 {

using core::Qwen35Geometry;

// ---------------------------------------------------------------- a weight matrix

/// One GGUF weight matrix, ROW-MAJOR with the OUTPUT row contiguous (`n_in` values per row, `n_out` rows).
struct Mat {
    const void* data = nullptr;   ///< host (mmapped GGUF) blocks
    const void* dev = nullptr;    ///< the same blocks in VRAM, when the GPU backend uploaded them
    int type = 0;                 ///< ggml type id; 0 = F32 rows
    int64_t n_in = 0, n_out = 0;
    size_t row_bytes = 0;         ///< bytes per output row (set by the loader; computed for F32 here)
    bool empty() const { return data == nullptr; }
};

/// A 1-row view of `m` (the GDN's per-head beta/alpha projections).
Mat mat_row(const Mat& m, int64_t i);
/// `sum_i row(m, 0)[i] * x[i]`.
float mat_dot(const Mat& m, const float* x);
/// `y[o] = sum_i m[o][i] * x[i]`.  Type 0 is a plain float dot; every other type goes through
/// `g_quant_matvec`, which the ggml-backed build installs with `qwen35_enable_ggml()`.
void matvec(const Mat& m, const float* x, float* y);

using QuantMatvecFn = void (*)(int type, const void* w, int64_t n_in, int64_t n_out, const float* x, float* y);
extern QuantMatvecFn g_quant_matvec;
/// GPU (HIP/CUDA) dense matvec.  Return true when it handled `m` (its `dev` pointer is set); `matvec` falls
/// back to `g_quant_matvec` otherwise.  Installed by qwen35_gpu_upload.
using GpuMatvecFn = bool (*)(const Mat& m, const float* x, float* y);
extern GpuMatvecFn g_gpu_matvec;
/// BATCHED GPU matvec: several projections of the SAME width, each with its own input, in one upload/sync/
/// download.  This is what makes the GPU tier worthwhile - a per-matvec sync costs more than the matvec.
/// Returns the number of entries handled (0 = not applicable; the caller falls back to `matvec`).
using GpuBatchFn = int (*)(const Mat* const* mats, const float* const* xs, float* const* ys, int count);
extern GpuBatchFn g_gpu_batch;
/// Dequantize one row of a quantized matrix to `n_in` floats.
using RowDequantFn = void (*)(int type, const void* row, int64_t n, float* out);
extern RowDequantFn g_row_dequant;
/// Dequantize row `row` of `m` to `m.n_in` floats (F32 is a plain copy).
void dequant_row(const Mat& m, int64_t row, float* out);
/// Install the ggml-cpu-backed quantized matvec (src/qwen35/qwen35_ggml.cpp).  Idempotent.
void qwen35_enable_ggml();

// ---------------------------------------------------------------- primitives

/// y = x * w / sqrt(mean(x^2) + eps), the `LLM_NORM_RMS` every Qwen35 block uses.  `w` may be null.
void rms_norm(const float* x, const float* w, int64_t n, float eps, float* y);
/// The GDN's l2 norm: x / sqrt(sum(x^2) + eps) (llama.cpp's `build_gdn_l2_norm`).
void l2_norm(const float* x, int64_t n, float eps, float* y);
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return std::log1p(std::exp(x)); }

// ---------------------------------------------------------------- one GDN layer

/// The recurrent layer's weights.  `ssm_conv` is channel-major: channel `c`'s `d_conv` taps occupy
/// `ssm_conv + c*d_conv`; it stays F32 (the artifact ships it F32).
struct GdnLayerWeights {
    const float* attn_norm = nullptr;    // [n_embd]
    Mat wqkv;                            // qkv_dim rows, n_embd in
    Mat wgate;                           // value_dim rows, n_embd in
    const float* ssm_conv = nullptr;     // channel-major: d_conv taps per channel
    const float* ssm_dt = nullptr;       // [v_heads]
    const float* ssm_a = nullptr;        // [v_heads]
    Mat ssm_beta;                        // v_heads rows, n_embd in
    Mat ssm_alpha;                       // v_heads rows, n_embd in
    const float* ssm_norm = nullptr;     // [head_v_dim]
    Mat ssm_out;                         // n_embd rows, value_dim in
};

/// The layer's persistent state.  `conv` is [d_conv-1, conv_channels] oldest-first; `rec` is
/// [v_heads][S_v][S_v] with element (j, i) at `rec + h*S*S + j*S + i` (the ggml kernel's transposed layout).
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
    Mat wq;                              // 2*n_head*head_dim rows (q | gate per head), n_embd in
    Mat wk;                              // n_head_kv*head_dim rows
    Mat wv;                              // n_head_kv*head_dim rows
    Mat wo;                              // n_embd rows, n_head*head_dim in
    const float* q_norm = nullptr;       // [head_dim]
    const float* k_norm = nullptr;       // [head_dim]
};

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

/// NEOX partial RoPE on one head vector in place.
void rope_neox(float* v, int64_t head_dim, int64_t n_rot, float base, int64_t pos);
/// One token through one full-attention layer: `x` -> `out`, appending to `st`.
void attn_layer(const Qwen35Geometry& g, const AttnLayerWeights& w, AttnState& st, const float* x, float* out);

// ---------------------------------------------------------------- one MoE block

struct ExpertWeights {
    Mat gate;   // n_ff_exp rows, n_embd in
    Mat up;     // n_ff_exp rows, n_embd in
    Mat down;   // n_embd rows, n_ff_exp in
};

struct MoeLayerWeights {
    const float* gate_inp = nullptr;       // [n_expert][n_embd], F32
    Mat gate_shexp;                        // n_ff_shexp rows, n_embd in
    Mat up_shexp;
    Mat down_shexp;                        // n_embd rows, n_ff_shexp in
    const float* gate_inp_shexp = nullptr; // [n_embd], F32
    const ExpertWeights* experts = nullptr;  // [n_expert]
};

void moe_layer(const Qwen35Geometry& g, const MoeLayerWeights& w, const float* x, float* out);

// ---------------------------------------------------------------- the trunk

struct TrunkWeights {
    Mat token_embd;                      // n_vocab rows, n_embd in (a row is one token's embedding)
    const float* output_norm = nullptr;  // [n_embd]
    Mat output;                          // n_vocab rows, n_embd in
    std::vector<const float*> attn_norm;       // per layer, [n_embd]
    std::vector<const float*> post_attn_norm;  // per layer, [n_embd]
    std::vector<GdnLayerWeights> gdn;
    std::vector<AttnLayerWeights> attn;
    std::vector<MoeLayerWeights> moe;
    /// The expert weight storage the `moe[l].experts` arrays point into (one entry per routed expert).
    std::vector<std::vector<ExpertWeights>> expert_store;
    /// Owns the mmap the `Mat`s point into: the loader keeps the GGUF open for the lifetime of the weights.
    std::shared_ptr<void> backing;
    int64_t eos_token = -1;   ///< tokenizer.ggml.eos_token_id, so the serve loop can stop on it
};

struct TrunkState {
    std::vector<GdnState> gdn;
    std::vector<AttnState> attn;
    /// `context` is the KV capacity in cells (the launcher's --max-context); 0 means the model's native
    /// maximum.  The attention states are sized for it, so a 128K deployment holds 128K of KV, not the
    /// model's 262K.
    void reset(const Qwen35Geometry& g, int64_t context) {
        const int64_t cells = context > 0 ? context : g.context_length;
        gdn.assign((size_t) g.n_layers, {});
        attn.assign((size_t) g.n_layers, {});
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (g.is_recurrent(l)) gdn[(size_t) l].resize(g);
            else attn[(size_t) l].resize(cells, g);
        }
    }
    void zero(const Qwen35Geometry& g) {
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (g.is_recurrent(l)) gdn[(size_t) l].zero();
            else attn[(size_t) l].zero();
        }
    }
};

/// `token` through the whole trunk at the state's current position.  `logits` is n_vocab floats.
void trunk_forward(const Qwen35Geometry& g, const TrunkWeights& w, TrunkState& st, int64_t token, float* logits);

/// Build the trunk weights from an Ornith/Qwen35MoE GGUF (mmapped; no copy, no dequantization).  Runs the
/// architecture guard first, so a malformed artifact fails here with a tensor-naming error.
bool load_trunk(const std::string& path, Qwen35Geometry& g, TrunkWeights& w, std::string& err);

/// GPU backend (src/qwen35/qwen35_gpu.cu): create the stream, then upload every DENSE matvec to VRAM and
/// install the matvec hook.  Routed experts are not uploaded (the whole set is ~28 GB) and stay on ggml-cpu.
bool qwen35_gpu_init(std::string& err);
bool qwen35_gpu_upload(TrunkWeights& w, std::string& err);
uint64_t qwen35_gpu_dense_bytes();

}  // namespace strata::qwen35
