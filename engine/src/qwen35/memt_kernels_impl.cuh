// MEMT_HOST_TEST runs the same kernel index math with one simulated thread per block.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(MEMT_HOST_TEST)

namespace strata::qwen35::memt::memt_host {
inline int g_tid = 0;        // threadIdx.x
inline int g_bid = 0;        // blockIdx.x
inline int g_nthread = 1;    // blockDim.x
inline float* g_shared = nullptr;
}  // namespace strata::qwen35::memt::memt_host

#define MEMT_TID (::strata::qwen35::memt::memt_host::g_tid)
#define MEMT_BID (::strata::qwen35::memt::memt_host::g_bid)
#define MEMT_NTHREAD (::strata::qwen35::memt::memt_host::g_nthread)
#define MEMT_SHARED float* sh = ::strata::qwen35::memt::memt_host::g_shared
#define MEMT_SYNC ((void) 0)
#define MEMT_GLOBAL
#define MEMT_DEVICE inline
#define MEMT_EXP std::exp
#define MEMT_LOG1P std::log1p
#define MEMT_SQRT std::sqrt
#define MEMT_TANH std::tanh
#else
#define MEMT_TID ((int) threadIdx.x)
#define MEMT_BID ((int) blockIdx.x)
#define MEMT_NTHREAD ((int) blockDim.x)
#define MEMT_SHARED extern __shared__ float sh[]
#define MEMT_SYNC __syncthreads()
#define MEMT_GLOBAL __global__
#define MEMT_DEVICE __device__ __forceinline__
#define MEMT_EXP expf
#define MEMT_LOG1P log1pf
#define MEMT_SQRT sqrtf
#define MEMT_TANH tanhf
#endif

namespace strata::qwen35::memt {
namespace device {

MEMT_DEVICE float logsigmoid(float x) {
    return x >= 0.0f ? -MEMT_LOG1P(MEMT_EXP(-x)) : x - MEMT_LOG1P(MEMT_EXP(x));
}

MEMT_DEVICE float softplus(float x) { return x > 30.0f ? x : MEMT_LOG1P(MEMT_EXP(x)); }

/// Sum `value` across the block; `red` holds MEMT_NTHREAD floats and every
/// thread returns the same total (kThreads is a power of two).
MEMT_DEVICE float block_sum(float value, float* red) {
    const int tid = MEMT_TID;
    red[tid] = value;
    MEMT_SYNC;
    for (int step = MEMT_NTHREAD >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        MEMT_SYNC;
    }
    return red[0];
}

/// decay = -exp(scale_log[h]) * softplus(f_down + bias), elementwise over [T][H*D].
MEMT_GLOBAL void k_gates(const float* fdown, const float* bias, const float* scale_log, float* decay, int total,
                         int heads, int dim) {
    const int i = MEMT_BID * MEMT_NTHREAD + MEMT_TID;
    if (i >= total) return;
    const int hd = heads * dim;
    const int r = i % hd;
    decay[i] = -MEMT_EXP(scale_log[r / dim]) * softplus(fdown[i] + bias[r]);
}

/// k_eff = l2norm(k) * erase, erase = exp(0.5*logsigmoid(erase_raw)); one block per (token, head).
MEMT_GLOBAL void k_key(const float* proj, float* k_eff, int heads, int dim, int64_t stride) {
    const int th = MEMT_BID;
    const int t = th / heads, h = th - t * heads;
    const int64_t hd = (int64_t) heads * dim;
    const int64_t base = (int64_t) t * stride + (int64_t) h * dim;   // proj's row (5*H*D + 2*R wide)
    const int64_t out = (int64_t) t * hd + (int64_t) h * dim;         // k_eff's row (H*D wide)
    MEMT_SHARED;
    float ss = 0.0f;
    for (int d = MEMT_TID; d < dim; d += MEMT_NTHREAD) {
        const float k = proj[base + hd + d];
        ss += k * k;
    }
    ss = block_sum(ss, sh);
    const float inv = 1.0f / MEMT_SQRT(ss + 1e-6f);   // FLA l2norm's epsilon
    for (int d = MEMT_TID; d < dim; d += MEMT_NTHREAD) {
        const float erase_sqrt = MEMT_EXP(0.5f * logsigmoid(proj[base + 3 * hd + d]));
        k_eff[out + d] = proj[base + hd + d] * inv * erase_sqrt;
    }
}

/// The two gate latents, gathered out of proj's wide rows into contiguous
/// [tokens][rank] blocks (the FP32 GEMMs take a matrix, not a stride).
MEMT_GLOBAL void k_latents(const float* proj, float* decay_latent, float* og_latent, int tokens, int64_t stride,
                           int64_t head_dim, int rank) {
    const int i = MEMT_BID * MEMT_NTHREAD + MEMT_TID;
    if (i >= tokens * rank) return;
    const int t = i / rank, r = i - t * rank;
    const int64_t base = (int64_t) t * stride + 5 * head_dim;
    decay_latent[i] = proj[base + r];
    og_latent[i] = proj[base + rank + r];
}

/// att += gated_rmsnorm((l2norm(q) @ state) * dim^-0.5, out_gate); one block per (token, head).
MEMT_GLOBAL void k_read(const float* proj, const float* og, const float* o_norm, const float* state, float* att,
                        int heads, int dim, int64_t stride, float eps, float softcap) {
    const int th = MEMT_BID;
    const int t = th / heads, h = th - t * heads;
    const int64_t hd = (int64_t) heads * dim;
    const int64_t base = (int64_t) t * stride + (int64_t) h * dim;   // proj's row
    const int64_t at = (int64_t) t * hd + (int64_t) h * dim;         // att's / og's row
    const float* S = state + (int64_t) h * dim * dim;
    MEMT_SHARED;   // qn[dim] | raw[dim] | red[MEMT_NTHREAD]
    float* qn = sh;
    float* raw = sh + dim;
    float* red = sh + 2 * dim;

    float ss = 0.0f;
    for (int d = MEMT_TID; d < dim; d += MEMT_NTHREAD) {
        const float q = proj[base + d];
        ss += q * q;
    }
    ss = block_sum(ss, red);
    const float inv = 1.0f / MEMT_SQRT(ss + 1e-6f);   // FLA l2norm's epsilon
    for (int d = MEMT_TID; d < dim; d += MEMT_NTHREAD) qn[d] = proj[base + d] * inv;
    MEMT_SYNC;

    const float scale = 1.0f / MEMT_SQRT((float) dim);
    float sq = 0.0f;
    for (int v = MEMT_TID; v < dim; v += MEMT_NTHREAD) {
        float acc = 0.0f;
        for (int k = 0; k < dim; ++k) acc += qn[k] * S[(int64_t) k * dim + v];
        raw[v] = acc * scale;
        sq += raw[v] * raw[v];
    }
    const float var = block_sum(sq, red) / (float) dim;
    const float rstd = 1.0f / MEMT_SQRT(var + eps);
    for (int v = MEMT_TID; v < dim; v += MEMT_NTHREAD) {
        const float gate = og[at + v];
        const float silu = gate / (1.0f + MEMT_EXP(-gate));
        const float bounded = softcap * MEMT_TANH(silu / softcap);
        att[at + v] += raw[v] * rstd * o_norm[v] * bounded;
    }
}

/// One token at a time: S <- diag(exp(log_decay)) S; kv = S^T k_eff;
/// delta = write*v - kv; S += k_eff delta^T.  One block per head.
MEMT_GLOBAL void k_update(const float* proj, const float* k_eff, const float* decay, float* state, int tokens,
                          int heads, int dim, int64_t stride) {
    const int h = MEMT_BID;
    const int hd = heads * dim;
    MEMT_SHARED;   // gs[dim] | delta[dim]
    float* gs = sh;
    float* delta = sh + dim;
    float* S = state + (int64_t) h * dim * dim;
    for (int t = 0; t < tokens; ++t) {
        const int64_t base = (int64_t) t * stride + (int64_t) h * dim;
        const float* keff = k_eff + (int64_t) t * hd + (int64_t) h * dim;
        const float* lg = decay + (int64_t) t * hd + (int64_t) h * dim;
        for (int d = MEMT_TID; d < dim; d += MEMT_NTHREAD) gs[d] = MEMT_EXP(lg[d]);
        MEMT_SYNC;
        for (int v = MEMT_TID; v < dim; v += MEMT_NTHREAD) {
            float kv = 0.0f;
            for (int k = 0; k < dim; ++k) kv += keff[k] * (gs[k] * S[(int64_t) k * dim + v]);
            const float write = 1.0f / (1.0f + MEMT_EXP(-proj[base + 4 * hd + v]));
            delta[v] = proj[base + 2 * hd + v] * write - kv;   // the gate on the value only
        }
        MEMT_SYNC;
        for (int v = MEMT_TID; v < dim; v += MEMT_NTHREAD) {
            for (int k = 0; k < dim; ++k) {
                float* s = S + (int64_t) k * dim + v;
                *s = gs[k] * (*s) + keff[k] * delta[v];
            }
        }
        MEMT_SYNC;
    }
}

#undef MEMT_TID
#undef MEMT_BID
#undef MEMT_NTHREAD
#undef MEMT_SHARED
#undef MEMT_SYNC
#undef MEMT_GLOBAL
#undef MEMT_DEVICE
#undef MEMT_EXP
#undef MEMT_LOG1P
#undef MEMT_SQRT
#undef MEMT_TANH

}  // namespace device
}  // namespace strata::qwen35::memt
