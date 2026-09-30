#pragma once

#include <cstdint>

namespace strata::kernels {

/// Qwen3.5 dense full-attention decode path.
///
/// q/k/v are already projected and normalized/rotated by the caller:
///   q: [24,256] F32
///   k/v: [4,256] F32
/// cache: [max_context,4,256] F16 for K and V.
///
/// The function appends the current K/V at `pos`, computes causal GQA attention for all 24
/// query heads, applies the Qwen3.5 sigmoid output gate, and returns [24,256] F32.
/// `scores` is caller-owned scratch of n_head * max_context floats.
/// Qwen3.5's ordinary RMSNorm is zero-centered: y = rms(x) * (1 + weight).
/// The linear-attention gated norm is different and is already implemented by Strata's GDN path.
void qwen35_rms_norm_weighted(float * x, const float * weight, int64_t rows, int64_t cols,
                              float eps, void * stream);

/// Qwen3.5 Gated DeltaNet closing norm: rms(o) * weight * SiLU(z).
void qwen35_gdn_out_norm(const float * o, const float * z, const float * weight,
                         float * y, int64_t heads, int64_t head_dim, float eps, void * stream);

void qwen35_silu_mul(const float * gate, const float * up, float * out, int64_t n, void * stream);

void qwen35_full_attention_step(const float * q, const float * k, const float * v,
                                uint16_t * cache_k, uint16_t * cache_v,
                                int64_t pos, int64_t max_context,
                                int n_head, int n_head_kv, int head_dim,
                                const float * gate,
                                float * scores, float * out, void * stream);

/// Allocate/build the partial NEOX RoPE table used by the text path.
/// Qwen3.5 uses n_rot=64 and theta=10,000,000.
void qwen35_rope_init(int64_t max_context, float ** cos_dev, float ** sin_dev, int ** pos_dev);
void qwen35_rope_free(float * cos_dev, float * sin_dev, int * pos_dev);

/// Fill the current device position used by the RoPE kernel.
void qwen35_rope_set_pos(int * pos_dev, int64_t pos, void * stream);

/// In-place partial NEOX RoPE over `rows x head_dim`, with 64 rotated dimensions.
void qwen35_rope_apply(float * x, int64_t rows, int head_dim, const float * cos_dev,
                       const float * sin_dev, const int * pos_dev, void * stream);

} // namespace strata::kernels
