// src/qwen35/q36_engine.cu - strata-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Strata's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, split-K over 256-cell chunks with a
//     log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface; --tokens-file F reads IDS from a
// file), --bench-pp N / --bench-tg N.  The opt-ins (--kv, --gqa-attn, --mtp-sampled, --snapshots, ...): docs/QWEN36.md.
#include <fstream>
#include <iterator>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_cache.hpp"   // (strata_engine, through strata_prefill)
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "q36_kernels.cuh"
#include "q36_attn.cuh"
#include "q36_pattn.cuh"
#include "q36_gdn.cuh"
#include "q36_sampling.hpp"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;
namespace sampling = strata::qwen35::sampling;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-q36: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    float* const* kc = nullptr;      // device [n_sessions]: this layer's K / V base of each session
    float* const* vc = nullptr;
};

/// The int8 K / V cache (--kv int8): a cell of a KV head is HD int8 values then HD / 32 fp32 scales (llama.cpp's q8_0
/// grid: per 32 values, d = amax / 127, q = round(x / d)), kv_cell_bytes(HD) bytes, cells ctx apart per head as in the
/// fp32 cache.  The fp32 cache's cell is HD floats.
__host__ __device__ constexpr inline size_t kv_cell_bytes(int HD, bool kv8) { return kv8 ? (size_t) HD + (size_t) HD / 32 * 4 : (size_t) HD * 4; }
/// Value i of a cell into the int8 cache row r (every lane of the warp holds one of the 32 values of its block).
__device__ __forceinline__ void kv8_store(uint8_t* r, int HD, int i, float x) {
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a / 127.f;
    reinterpret_cast<int8_t*>(r)[i] = (int8_t) (d > 0.f ? __float2int_rn(x / d) : 0);
    if ((i & 31) == 0) reinterpret_cast<float*>(r + HD)[i >> 5] = d;
}
__device__ __forceinline__ float kv8_load(const uint8_t* r, int HD, int i) {
    return (float) reinterpret_cast<const int8_t*>(r)[i] * reinterpret_cast<const float*>(r + HD)[i >> 5];
}
static_assert(sizeof(q36attn::q8_cell) == kv_cell_bytes(256, true), "the grouped-query kernel's int8 cell");
/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256).  KV: float, or __half (--kv fp16, #3641); KV8 (--kv int8, with V4 k8v4)
/// with KV = float.  (Template arguments, so that the fp32 cache's kernel is the one it was.)
template<class KV, bool KV8, bool V4>
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, KV* kc, KV* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    __shared__ float vbuf[V4 ? 256 : 1];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = reinterpret_cast<KV*>(rs.kc[rs.slot[t]]);
        vc = reinterpret_cast<KV*>(rs.vc[rs.slot[t]]);
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        const float vv = vx[((size_t) t * HKV + kh) * HD + i];
        if (KV8) {   // HD % 32 == 0: whole warps per q8_0 block
            const size_t cell = ((size_t) kh * ctx + pos) * kv_cell_bytes(HD, true);
            kv8_store(reinterpret_cast<uint8_t*>(kc) + cell, HD, i, v);
            if (V4)   // --kv k8v4: HD == 256, every thread of the block here
                q36attn::kv4_store(reinterpret_cast<uint8_t*>(vc) + ((size_t) kh * ctx + pos) * sizeof(q36attn::q4_cell), i, vv, vbuf);
            else
                kv8_store(reinterpret_cast<uint8_t*>(vc) + cell, HD, i, vv);
        } else {
            kc[((size_t) kh * ctx + pos) * HD + i] = (KV) v;
            vc[((size_t) kh * ctx + pos) * HD + i] = (KV) vv;
        }
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.  KV8: the int8 cache's cells.
template<bool KV8>
__global__ void k_attn_part(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    constexpr size_t CB8 = HD + HD / 32 * 4;
    if (p < npos) {
        float a = 0.f;
        if (KV8) {
            const uint8_t* kr = reinterpret_cast<const uint8_t*>(kc) + ((size_t) kvh * ctx + p) * CB8;
            const int8_t* kq = reinterpret_cast<const int8_t*>(kr);
            const float* kd = reinterpret_cast<const float*>(kr + HD);
            const int4* k4 = reinterpret_cast<const int4*>(kq);   // cells are 288 bytes apart: 16-byte aligned
            for (int b = 0; b < HD / 32; ++b) {
                float ab = 0.f;
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int4 w = k4[b * 2 + u];
                    const int8_t* c8 = reinterpret_cast<const int8_t*>(&w);
#pragma unroll
                    for (int z = 0; z < 16; ++z) ab += qs[b * 32 + u * 16 + z] * (float) c8[z];
                }
                a += ab * kd[b];
            }
        } else {
            const float* kp = kc + ((size_t) kvh * ctx + p) * HD;
            for (int i = 0; i < HD; ++i) a += qs[i] * kp[i];
        }
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    if (KV8) {
        const uint8_t* vr = reinterpret_cast<const uint8_t*>(vc) + ((size_t) kvh * ctx + p0) * CB8;
        for (int k = 0; k < n; ++k) acc += sc[k] * kv8_load(vr + (size_t) k * CB8, HD, j);
    } else {
        const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
        for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    }
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, const uint8_t* lent = nullptr,
                           int n_reg = 0x7fffffff) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) q36k::slot_blob(blobs, lent, n_reg, slot_of[x], blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// --hybrid-zero-copy (the hybrid hand-off without copies, also with --expert-prefetch): the routing and the MoE
/// input stored by the SMs into the pinned host buffers (mapped: unified addressing) instead of two device-to-host
/// copies - a copy after a kernel costs the stream an engine switch, ~2-3 us, twice per layer.  Bitwise the same bytes.
/// grid1(n_h / 4).
__global__ void k_route_to_host(const int32_t* ids, const float* h, int32_t* h_ids, float* h_h, int n_ids, int n_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_ids) h_ids[i] = ids[i];
    if (i < n_h / 4) ((float4*) h_h)[i] = ((const float4*) h)[i];
}
/// --hybrid-zero-copy: k_merge_cpu reading the CPU's rows in place (pinned host memory) instead of after two
/// host-to-device copies, the missed entries as a mask argument (n_ent <= 64).  grid (N / 1024, n_ent), block 256.
__global__ void k_merge_mapped(float* parts, const float* cpu, unsigned long long miss, int n) {
    const int e = blockIdx.y;
    if (!((miss >> e) & 1ull)) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 4) ((float4*) (parts + (size_t) e * n))[i] = ((const float4*) (cpu + (size_t) e * n))[i];
}
/// --expert-prefetch: the next layer's router logits (the prediction) for row blockIdx.y, y[o] = W[o] . x -
/// four warps per output (two outputs per block, grid (n_out / 2, rows)) so that it is short beside the stream's kernels.
/// n_in % 512 == 0.
__global__ void k_pred_gemv(const float* W, const float* x, float* y, int n_in, int n_out) {
    __shared__ float part[8];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 2 + warp / 4, q = warp % 4;
    const int per = n_in / 16;   // float4s per quarter
    x += (size_t) blockIdx.y * n_in;
    y += (size_t) blockIdx.y * n_out;
    float acc = 0.f;
    if (o < n_out) {
        const float4* wr = (const float4*) (W + (size_t) o * n_in) + q * per;
        const float4* xr = (const float4*) x + q * per;
        for (int i = lane; i < per; i += 32) {
            const float4 a = wr[i], b = xr[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x < 2 && blockIdx.x * 2 + (int) threadIdx.x < n_out) {
        const int b = threadIdx.x * 4;
        y[blockIdx.x * 2 + threadIdx.x] = part[b] + part[b + 1] + part[b + 2] + part[b + 3];
    }
}
/// A busy-wait of `cycles` clock cycles (tests: STRATA_Q36_SIDE_SKEW), one thread.
__global__ void k_spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
/// --expert-prefetch: a prefetch copy has landed (on the copy's stream, after it): its slot's word = gen.
__global__ void k_pf_landed(int32_t* flag, int32_t gen) {
    *(volatile int32_t*) flag = gen;
    __threadfence();
}
/// ... and, on the engine's stream before a layer's routing, the slot table's changes on the GPU's own timeline: the
/// evicted experts out, then each copy whose word says it has landed in (a copy still in flight stays out: the CPU
/// computes that expert, nothing waits); what it did goes to out[p] (mapped: gen, or 0) for the host's plan.  One
/// thread, in order.
struct PfPub {
    int32_t ev_x[16], x[16], slot[16], gen[16], p[16];
    int nev, n;
};
__global__ void k_pf_publish(int32_t* slot_of, const int32_t* flag, int32_t* out, PfPub u) {
    for (int i = 0; i < u.nev; ++i) slot_of[u.ev_x[i]] = -1;
    for (int i = 0; i < u.n; ++i) {
        const bool landed = ((const volatile int32_t*) flag)[u.p[i]] == u.gen[i];
        if (landed) slot_of[u.x[i]] = u.slot[i];
        out[u.p[i]] = landed ? u.gen[i] : 0;
    }
    __threadfence_system();
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves (KV8: from the int8 cache, V4: its k8v4 V).  grid1(HKV * P * HD).
template<bool KV8, bool V4>
__global__ void k_kv_f16(const float* kc, const float* vc, __half* k16, __half* v16, int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    if (KV8) {
        const size_t cell = ((size_t) kvh * ctx + (size_t) (rest / HD)) * kv_cell_bytes(HD, true);
        const int d = (int) (rest % HD);
        k16[i] = __float2half(kv8_load(reinterpret_cast<const uint8_t*>(kc) + cell, HD, d));
        if (V4) {   // --kv k8v4: V still rotated (the attention output is rotated back, q36attn::unrotate_gate)
            const auto& c = reinterpret_cast<const q36attn::q4_cell*>(vc)[(size_t) kvh * ctx + (size_t) (rest / HD)];
            v16[i] = __float2half(q36attn::v_value(c, d, d >> 5));
        } else {
            v16[i] = __float2half(kv8_load(reinterpret_cast<const uint8_t*>(vc) + cell, HD, d));
        }
        return;
    }
    const size_t src = (size_t) kvh * ctx * HD + rest;
    k16[i] = __float2half(kc[src]);
    v16[i] = __float2half(vc[src]);
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)
/// --spec-min-p: the softmax probability of the row's pick (`pick`, k_argmax's), one block: how sure the draft head is
__global__ void k_pick_prob(const float* x, int n, const int32_t* pick, float* out) {
    __shared__ float bs[1024];
    const float mx = x[*pick];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(x[i] - mx);
    bs[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if ((int) threadIdx.x < k) bs[threadIdx.x] += bs[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = 1.f / bs[0];
}

/// --sample-gpu: the SHORTLIST of a sampled row (q36_sampling.hpp, Cand): the K highest raw logits of each row, the tie
/// rule (higher logit, then lower id), in any order, as (id, logit) pairs, K per row packed.  Two kernels: k_cand_part, a
/// block per 1/32 of the row, keeps its slice's K highest (the global K highest are among them); k_cand_merge, a block
/// per row, takes the K highest of those 32 K.  Both select with cand_select: a radix select of the K-th key over
/// shared memory (4 passes of 8 bits), then the elements above it and the ties at it, the lowest ids first.  The host
/// applies the penalties and runs the rest of the chain on them.
struct Cand2 { int32_t id; float v; };
using sampling::kCandBlocks;
using sampling::kCandSlice;
__device__ __forceinline__ uint32_t cand_key(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u << 1) == 0) u = 0;   // -0 == +0 (the host compares floats)
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/// the K highest of n keys (kf(i)) into sidx[0 .. K) (indices i, any order); ties at the K-th key: the lowest idf(i).
/// kOrdered: idf grows with i.  One block; the caller syncs before and after.
template <bool kOrdered, class KF, class IDF>
__device__ __forceinline__ void cand_select(int n, int K, KF kf, IDF idf, int* sidx) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_rem, s_eq, s_cnt;
    const int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0; s_rem = (unsigned) K; s_cnt = 0; }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const unsigned pre = s_prefix;
        for (int base = 0; base < n; base += blockDim.x) {   // (all lanes of a warp take every turn: __match_any_sync)
            const int i = base + tid;
            const uint32_t u = i < n ? kf(i) : 0u;
            const bool in = i < n && (pass == 0 || (u >> (shift + 8)) == (pre >> (shift + 8)));
            const unsigned bin = (u >> shift) & 255u;
#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
            if (pass == 0) {   // the top byte holds few values: one add per distinct bin of the warp (sm_70+)
                const unsigned m = __match_any_sync(0xffffffffu, in ? bin : 256u);
                if (in && (tid & 31) == __ffs(m) - 1) atomicAdd(&hist[bin], (unsigned) __popc(m));
            } else
#endif
            if (in) {
                atomicAdd(&hist[bin], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned rem = s_rem;
            int b = 255;
            for (; b > 0; --b) {
                if (hist[b] >= rem) break;
                rem -= hist[b];
            }
            s_prefix = pre | ((unsigned) b << shift);
            s_rem = rem;
            s_eq = hist[b];
        }
        __syncthreads();
    }
    const unsigned thr = s_prefix, need = s_rem, eq = s_eq;
    for (int i = tid; i < n; i += blockDim.x) {
        const uint32_t u = kf(i);
        if (u > thr || (u == thr && eq == need)) sidx[atomicAdd(&s_cnt, 1u)] = i;
    }
    __syncthreads();
    if (eq != need && tid == 0) {   // ties across the K-th place: the lowest ids (rare)
        unsigned c = s_cnt;
        if (kOrdered) {
            for (int i = 0, taken = 0; i < n && taken < (int) need; ++i)
                if (kf(i) == thr) { sidx[c++] = i; ++taken; }
        } else {
            int last = -1;
            for (unsigned t = 0; t < need; ++t) {
                int best = -1;
                for (int i = 0; i < n; ++i)
                    if (kf(i) == thr && idf(i) > last && (best < 0 || idf(i) < idf(best))) best = i;
                sidx[c++] = best;
                last = idf(best);
            }
        }
    }
    __syncthreads();
}
__global__ void __launch_bounds__(1024) k_cand_part(const float* __restrict__ lg, int ld, int n, int K, Cand2* part) {
    const float* row = lg + (size_t) blockIdx.y * ld;
    const int S = (n + kCandBlocks - 1) / kCandBlocks, lo = blockIdx.x * S, m = min(n, lo + S) - lo;
    __shared__ uint32_t keys[kCandSlice];
    __shared__ int sidx[sampling::kCandMax];
    for (int i = threadIdx.x; i < m; i += blockDim.x) keys[i] = cand_key(row[lo + i]);
    __syncthreads();
    cand_select<true>(m, K, [&](int i) { return keys[i]; }, [&](int i) { return lo + i; }, sidx);
    Cand2* o = part + ((size_t) blockIdx.y * kCandBlocks + blockIdx.x) * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) o[j] = Cand2{lo + sidx[j], row[lo + sidx[j]]};
}
__global__ void __launch_bounds__(1024) k_cand_merge(const Cand2* __restrict__ part, int K, Cand2* out) {
    const int m = kCandBlocks * K;
    const Cand2* in = part + (size_t) blockIdx.x * m;
    __shared__ uint32_t keys[kCandBlocks * sampling::kCandMax];
    __shared__ int sidx[sampling::kCandMax];
    for (int i = threadIdx.x; i < m; i += blockDim.x) keys[i] = cand_key(in[i].v);
    __syncthreads();
    cand_select<false>(m, K, [&](int i) { return keys[i]; }, [&](int i) { return in[i].id; }, sidx);
    for (int j = threadIdx.x; j < K; j += blockDim.x) out[(size_t) blockIdx.x * K + j] = in[sidx[j]];
}

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
    // --lend-staging: lent_n more slots (numbered from n_reg, after the cache and prefetch slots) in the prompt staging buffers
    uint8_t* lent = nullptr;
    int n_reg = 0x7fffffff, lent_n = 0;
    uint8_t* slot_addr(int s) const { return s < n_reg ? blobs + (size_t) s * stride : lent + (size_t) (s - n_reg) * stride; }
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

constexpr size_t kMmqTail = 4096;
constexpr size_t kHybridHeadroom = (size_t) 256 << 20;   // hybrid placement: free VRAM kept at the low-water mark
constexpr size_t kLazyGrowth = (size_t) 48 << 20;        // ... plus the engine's lazy growth after placement (measured 21 MiB)
constexpr size_t kPlacedAfter = (size_t) 96 << 20;       // ... plus what place_experts allocates after reading the free VRAM (measured 95 MiB)
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "strata-q36 runs Qwen3.6-35B-A3B (qwen35moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15)";
        return false;
    }
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-q36: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-q36: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--mtp): the VRAM it takes is then the same
    // either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on --mtp
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || strata::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(cudaMalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    std::fprintf(stderr, "strata-q36: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    // the side stream of a decode layer's independent matrices (z and the GDN alpha / beta; the attention's k and v): they
    // read the layer's input like the first matrix and nothing reads them before the mixer, so they run beside it.  Same
    // kernels on the same inputs, so the bits are the same.  --side-stream turns it on (off by default).
    cudaStream_t ast = nullptr;
    cudaEvent_t ev_a0 = nullptr, ev_a1 = nullptr;
    bool side_on = false;   // --side-stream
    // tests: STRATA_Q36_SIDE_SKEW=<us> inserts a spin kernel of that length on the side stream before its first kernel, and
    // STRATA_Q36_SIDE_SKEW_JOIN=<us> one on the main stream right before the join wait: the interleavings natural timing never
    // makes (a missing fork or join would show as different logits)
    long long side_skew_us = 0, side_skew_join_us = 0;
    // `ref` (the engine's stream member) is `s` inside the scope: apply() and friends read the member
    struct StreamScope {
        cudaStream_t &ref, keep;
        StreamScope(cudaStream_t& r, cudaStream_t s) : ref(r), keep(r) { ref = s; }
        ~StreamScope() { ref = keep; }
    };
    void side_spin(cudaStream_t s, long long us) {
        if (us <= 0) return;
        int khz = 0;
        CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
        k_spin<<<1, 1, 0, s>>>((long long) khz * us / 1000);
    }
    int ctx = 32768, tmax = 512;
    bool kv8 = false;   // --kv int8: the K / V cache as q8_0 rows (kv_cell_bytes), else fp32
    bool kv_v4 = false; // --kv k8v4: kv8's K, V as rotated q4_0 (q36attn::q4_cell); the grouped-query kernel always
    bool kv_f16 = false;  // --kv fp16; cache pointers/tables hold opaque storage in either format
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    // --sample-gpu: a sampled row comes to the host as a shortlist (k_cand_part, k_cand_merge), not as 1 MB of logits
    bool sample_gpu = false;
    Cand2 *d_cand = nullptr, *h_cand = nullptr, *d_part = nullptr;   // [kVerify + 1][K] the shortlists (device, pinned host)
    int cand_K = 0;   // the K of the LAST cand_launch: cand_row reads with it (call it before the next launch)
    bool smp_prof = false, smp_check = false;     // STRATA_Q36_SAMPLE_PROFILE / STRATA_Q36_SAMPLE_CHECK
    double smp_ms = 0;                            // the sampler's host time of the request (copies, shortlists, chain)
    long long smp_calls = 0, smp_chk_n = 0, smp_chk_bad = 0;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int n_chunk = 0, att_rows = 0;   // attention partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S, kc, vc;   // per layer; the MTP block's K / V at index n_layers: the CURRENT session's (select)
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    float* prob_m = nullptr;   // --spec-min-p: the probability of pick_m under the draft head's softmax
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half *k16 = nullptr, *v16 = nullptr, *q16 = nullptr;   // the prompt attention's FP16 operands
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats when the experts' fit is close)
    // --fused-prompt-attn (opt-in, the int8 cache): the prompt attention in one kernel straight from the cache cells
    // (q36_pattn.cuh), so k16 / v16 / att_s / att_p16 are not allocated; pa_* hold the split partials of a small chunk
    bool fused_pattn = false;
    float *pa_acc = nullptr, *pa_m = nullptr, *pa_l = nullptr;
    int n_sm = 0;
    // --chunked-gdn (opt-in): the prompt's gated delta rule in pieces of 16 tokens (q36_gdn.cuh), its L / P matrices and cumulative gates here
    bool chunked_gdn = false;
    float *gdn_L = nullptr, *gdn_P = nullptr, *gdn_cum = nullptr;
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    // #3641's grouped-query decode attention (q36_attn.cuh; --kv fp16 / k8v4 always, else --gqa-attn, for the fp32 and
    // int8 caches from 8192 cells), and its hybrid expert cache's history window (on with --kv fp16;
    // STRATA_Q36_CACHE_WINDOW=0/1, a test setting)
    bool gqa_attn = false;
    bool cache_window = false;
    size_t rebalance_cursor = 0;
    int cache_swaps = 1024;  // the history window: copy at most this many experts per 32 rows
    int decode_swaps = 8;    // slot replacements per kRebalanceEvery decode rows over all layers (--decode-swaps)
    int gqa_from = 8192;     // the grouped-query kernel from this many cells (graph mode: a context at least this big)
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only, unless --mtp-sampled)
    int mtp_sampled = 0;   // --mtp-sampled: sampled requests draft too (1: match, 2: q; SpecRule)
    float spec_min_p = 0.f;   // --spec-min-p: a draft the block is less sure of ends the step's drafts (0: always mtp_d)
    int last_d = 0;           // the drafts the last spec_step verified (mtp_d, or fewer: the gate, the rebalance boundary)

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    // --snapshots K (opt-in): the recurrent state a prompt read left, K per session.  The DeltaNet
    // layers' S and short-conv state cannot step back; the attention and MTP caches can (cells from the position on are
    // never read), and hlast (the MTP seed) is kept beside the state.  A prompt that parts from the held tokens after a
    // snapshot's position (the reply re-encoded to other ids, a reply the client changed) restores it and reads only
    // the rest instead of the whole conversation.
    struct Snap {
        std::vector<float*> conv, S;        // per layer (null for the attention layers)
        float* hlast = nullptr;
        int64_t len = 0;                    // the held tokens it was taken after (0: none)
        int32_t rdelta = 0;
        std::vector<ImgKey> imgs;
        uint64_t when = 0;
    };
    struct Sess {
        std::vector<float*> conv, S, kc, vc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
        std::vector<Snap> snaps;            // snapshots of its first snaps[k].len held tokens (len 0: none)
    };
    int n_snaps = 0;                        // snapshots per session (--snapshots, 0: off)
    bool snap_host = false;                 // --snapshot-host: the snapshots in pinned host memory (copies over PCIe), not in VRAM
    int64_t snap_min = 4096;                // taken after reads of prompts at least this long (--snapshot-min)
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers
    float **d_conv_tab = nullptr, **d_S_tab = nullptr, **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    int64_t rows_pmax = 0;                                                  // (eager) the highest cell of the rows
    // pictures (--vision): the request's rows replace the embeddings of its <|image_pad|> cells; M-RoPE
    bool vision = false;
    int32_t* d_mtab = nullptr;          // [ctx][3] t, h, w of the current request's prompt cells (vision)
    const int32_t* mtab_now = nullptr;  // set for a forward over prompt cells that need it, else null
    int32_t* d_delta = nullptr;         // device scalar: the current session's rotary offset (decode graphs read it)
    float* d_img = nullptr;             // the request's picture rows [n][N]
    size_t img_cap = 0;                 // floats
    int32_t* d_map = nullptr;           // [tmax] a chunk's row -> picture row (or -1)
    int32_t* h_map = nullptr;           // pinned
    bool map_on = false;                // this forward's embeddings take picture rows
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance
        std::vector<double> heat;    // the same counts never halved, since the warm-up (with count_heat)
        int32_t* slot_of_d = nullptr;
        // --expert-prefetch: the prefetch slots (n_slots + p), the round robin's next, and the table changes
        // not yet on the device (k_pf_publish at the layer's next routing)
        struct PfSlot {
            int x = -1;                // the expert copied in (or being copied in)
            int state = 0;             // 0 empty, 1 loading (copy issued, not seen landed), 2 ready (in the table)
            int32_t gen = 0;           // its copy's number (k_pf_landed writes it)
            bool offered = false;      // the last k_pf_publish offered it (pf_settle reads the outcome)
            long long last_ref = -1;   // the forward (pf_fwd) whose plan last read it
        };
        std::vector<PfSlot> pf;
        std::vector<char> pend_s, pend_x;   // --async-swaps: a regular slot / an expert whose swap copy has not been taken yet
        std::vector<int> prof_order, prank; // --lend-staging: the experts in the expert profile's order (else 0 .. ne - 1), and each one's place in it
        int pf_next = 0;
        std::vector<int32_t> evicted;  // experts out of a prefetch slot, not yet out of the device table
        int32_t* flag_d = nullptr;     // device [P]: the landed copies' numbers
        int32_t* out_h = nullptr;      // mapped [P]: k_pf_publish's outcome
        int32_t* out_d = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
    int pool_warm_k = 0;   // STRATA_Q36_POOL_WARM: predicted non-resident experts per layer posted to the pool's workers (0: off)
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr, dst2 = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {}, ev_d2d[2] = {};
    bool split_copy = false;   // --split-stage-copy: the resident experts' runs from 1/8 resident, on their own stream (else from 1/2)
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    // --async-swaps: the rebalance's slot copies go to a copy stream (swst) into the slot of an expert that already left the
    // table, in batches of kSwapBatch with an event each; an expert enters the host and device tables once its batch's event
    // is done (swap_settle, at the start of a forward, when nothing is in flight), so a slot is never read before it landed
    // --lend-staging: the prompt staging buffers (idle in decode) hold lent_n more expert slots per layer, numbered from L.n_reg;
    // lend_evict() (the stream starts: staging is needed) takes them out of the tables with no copy, after the prompt the
    // rebalance fills them again (lend_active) by the routing counts, the profile's rank for an expert not routed yet
    bool lend_on = false, lend_active = false, lend_prior = true;
    bool lend_settle = false;   // STRATA_Q36_LEND_SETTLE=1: the sync rebalance's lent copies on the engine stream and the after-prompt wait (reproducible); else on the swap stream, taken in when they land (timing-dependent)
    size_t stage_max_layer = 0;   // the staging buffers' used bytes (the zero tail follows), for the tail check
    long long tail_checks = 0, tail_bad = 0;
    cudaEvent_t ev_lend = nullptr;
    long long lend_swaps = 0, lend_evicts = 0, lend_fills = 0;
    double lend_evict_ms = 0;
    int async_mask = 0;   // bit 0: the decode rebalance's copies (--async-swaps), bit 1: the after-prompt rebalance's (--async-after-prompt, alone or with it)
    bool async_swaps = false, arena_pinned = false;
    cudaStream_t swst = nullptr;
    cudaEvent_t ev_swfence = nullptr;
    struct SwapItem { int li, s, x; };
    struct SwapBatch { cudaEvent_t ev; std::vector<SwapItem> items; };
    static constexpr int kSwapBatch = 128;
    std::deque<SwapBatch> swap_q;
    std::vector<cudaEvent_t> swap_ev_free;
    std::vector<SwapItem> sw_new;
    long long sw_taken = 0, sw_batches = 0, chk_calls = 0, chk_bad = 0, chk_bytes = 0, chk_bytes_bad = 0;
    bool pf_check = std::getenv("STRATA_Q36_PF_CHECK") != nullptr;   // debug: the host's slot tables against the device's at every forward
    // the rebalance calls' own cost (the log's "rebalance:" line): [0] decode / verify rows, [1] after a prompt read
    long long rb_calls[2] = {0, 0}, rb_swaps[2] = {0, 0};
    double rb_ms[2] = {0, 0};
    double hy_cpu_ms = 0, hy_plan_ms = 0;   // (hy_cpu_ms: the whole CPU half; hy_plan_ms: its host plan before the pool call)
    long long hy_layers = 0;                // layer calls with at least one pool entry
    // --expert-profile (the main path's file, include/strata/core/expert_cache.hpp; set before place_experts): each
    // hybrid layer's first slots take its most-routed experts from it.  Here a "layer" is the engine's state index (the
    // main layers, then the MTP block).  `profile_ranked`: its pairs as read (--expert-profile-save's prior).
    std::string expert_profile;
    std::vector<std::pair<int32_t, int32_t>> profile_ranked;
    bool count_heat = false;   // --expert-profile-save: Hy::heat counted (what the save ranks by)
    // --expert-prefetch P (opt-in): P of each main hybrid layer's VRAM slots (the same VRAM: P fewer cached experts)
    // hold experts fetched ahead for a decode row or a verify window (up to kPfRows rows: the union of the rows'
    // predictions, the experts most rows predict first).  Layer l+1's router applied to layer l's MoE input
    // (pre-gating, on stream pst beside layer l's experts) names the experts layer l+1 will likely pick (64% of its
    // misses in its top 8 at 4K cells; a Qwen3.6-35B-A3B fine-tune, Q4_K_M, RTX 5080, DDR5-6400 dual channel); the
    // ones without a slot are copied from the pinned host copy into the prefetch slots (stream pfst, the copy engine)
    // once layer l's pool is done and layer l+1's first product is queued - DRAM is idle through layer l+1's mixer, while the pool's reads would share it,
    // and the stream is not kept waiting for the host.  The main path's foresight swap rules (foresight_swap.hpp): a
    // copy is used only once it has landed - k_pf_publish, on the engine's stream before layer l+1's routing, takes it
    // into the table if its copy's word (k_pf_landed) says so, else the CPU computes that expert and the copy serves a
    // later forward, nothing waits (decided on the GPU's timeline: the host enqueues ahead of it); a slot is refilled
    // only when no plan of the running forward read it (a verify window's kernels may still run) and never while its
    // copy is in flight; at most pf_budget copies per forward.  A predicted miss is then a GPU hit; the CPU pool
    // computes only the rest.  pf_topk: how many of the predicted picks are candidates.
    int pf_want = 0;              // --expert-prefetch P (set before place_experts, which makes pf_slots of it)
    int pf_slots = 0, pf_topk = 8;
    bool zc = false;   // --hybrid-zero-copy: the hybrid hand-off without copies (k_route_to_host, k_merge_mapped)
    cudaStream_t pst = nullptr, pfst = nullptr;
    cudaEvent_t ev_pf0 = nullptr, ev_pred = nullptr;
    bool pf_join_pending = false;
    float *d_pflog = nullptr, *h_pflog = nullptr;   // mapped: the next layer's predicted router logits ([2][kPfRows][ne])
    int pf_for = -1;              // the layer the pending prediction is for
    long long pf_fwd = 0;         // main forwards so far (the slots' reuse rule: never one a running forward's plan read)
    bool pf_no_copy = false;   // --prefetch-copies 0 (STRATA_Q36_PREFETCH_COPIES=0): predictions run, no copy is issued
    int pool_warm_want = -1, pf_copies_want = -1;   // --pool-warm K / --prefetch-copies 0|1 (-1: not given; the envs are aliases)
    int pf_budget = 64, pf_used = 0;   // copies per forward, used this forward
    int32_t pf_gen = 0;           // the prefetch copies' numbers (1 .. INT32_MAX, then 1 again; STRATA_Q36_PF_GEN_START for tests)
    long long pf_copies = 0, pf_hits = 0, pf_preds = 0, pf_late = 0;
    std::vector<int> pf_order, pf_idx, pf_cnt;   // the candidates, a row's sort, rows predicting each expert
    std::vector<float> pf_best;                  // each expert's highest predicted logit over the rows
    static constexpr int kPfRows = 8;            // rows predicted (decode, verify windows up to 8 rows)
    int pf_rows = 1;                             // rows in the pending prediction
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every kRebalanceEvery-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    static constexpr int kRebalanceEvery = 32;
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) {
                    hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
                    if (count_heat) hy[li].heat[(size_t) row_ids[li][(size_t) e]] += 1.0;
                }
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(cache_window ? cache_swaps : decode_swaps);
        }
    }
    void after_prompt(int64_t n_tok = 0) {
        if (cache_window) {
            // a long prompt's counts cut to what decode's halving every kRebalanceEvery rows keeps (their order
            // kept), so that an answer's other routing can take slots within that window
            const double limit = 2.0 * kRebalanceEvery * m->top_k;
            for (Hy& H : hy) {
                double total = 0;
                for (double v : H.score) total += v;
                if (total > limit) for (double& v : H.score) v *= limit / total;
            }
        }
        if (lend_on && !warming) lend_active = true;   // the staging is free again: the rebalance below fills the lent slots
        // STRATA_Q36_AFTER_SWAPS_PER_KTOK=f (opt-in): at most f swaps per 1000 tokens just read, under the usual cap
        static const double per_ktok = std::getenv("STRATA_Q36_AFTER_SWAPS_PER_KTOK") ? std::atof(std::getenv("STRATA_Q36_AFTER_SWAPS_PER_KTOK")) : 0.0;
        int budget = hybrid_swaps_after_prompt;
        if (per_ktok > 0 && n_tok > 0) budget = std::min<int>(budget, (int) std::ceil(per_ktok * (double) n_tok / 1000.0));
        rebalance(budget, 1);
        if (lend_on && lend_settle && swst && !warming) {   // the lent slots are all in before the first decode forward
            CK(cudaStreamSynchronize(swst));
            swap_settle();
        }
    }

    /// A cell of one KV head in the cache: fp32, fp16 (--kv f16) or q8_0 (--kv int8) values.
    size_t kv_cell() const {
        return kv8 ? kv_cell_bytes((int) m->g.head_dim, true) : (size_t) m->g.head_dim * (kv_f16 ? sizeof(__half) : sizeof(float));
    }
    /// A V cell: kv_cell(), or a q4_cell with --kv k8v4.
    size_t kv_cell_v() const { return kv_v4 ? sizeof(q36attn::q4_cell) : kv_cell(); }
    float* alloc_kv(size_t cells, size_t cell_bytes) const { return reinterpret_cast<float*>(dalloc_t<uint8_t>(cells * cell_bytes)); }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        for (const Layer& L : m->layers) {
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(alloc_kv((size_t) g.n_head_kv * ctx, kv_cell()));
                ss.vc.push_back(alloc_kv((size_t) g.n_head_kv * ctx, kv_cell_v()));
            } else {
                ss.conv.push_back(dalloc(C * (g.ssm_d_conv - 1)));
                ss.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            ss.kc.push_back(alloc_kv((size_t) g.n_head_kv * ctx, kv_cell()));
            ss.vc.push_back(alloc_kv((size_t) g.n_head_kv * ctx, kv_cell_v()));
        }
        ss.hlast = dalloc((size_t) g.n_embd);
        ss.snaps.resize((size_t) n_snaps);
        // (--snapshot-host: pinned host memory, one block per snapshot; the copies below use cudaMemcpyDefault)
        auto snap_alloc = [&](size_t n) -> float* {
            if (!snap_host) return dalloc(n);
            float* p = nullptr;
            CK(cudaHostAlloc((void**) &p, n * 4, cudaHostAllocDefault));
            std::memset(p, 0, n * 4);
            return p;
        };
        for (Snap& sn : ss.snaps) {         // allocated with the session: before the experts take the rest of the VRAM
            for (const Layer& L : m->layers) {
                sn.conv.push_back(L.attn ? nullptr : snap_alloc(C * (g.ssm_d_conv - 1)));
                sn.S.push_back(L.attn ? nullptr : snap_alloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
            }
            sn.hlast = snap_alloc((size_t) g.n_embd);
        }
    }
    /// The bytes of one snapshot (the DeltaNet layers' conv and S state, hlast).
    size_t snap_bytes() const {
        const auto& g = m->g;
        size_t n = (size_t) g.n_embd;
        for (const Layer& L : m->layers)
            if (!L.attn) n += (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) + (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size;
        return n * 4;
    }

    /// The current session's recurrent state into a snapshot (one of the same position, else a free one, else
    /// the oldest).
    void snapshot() {
        Sess& ss = sess[(size_t) cur];
        if (ss.snaps.empty() || pos <= 0) return;
        ss.pos = pos;
        Snap* d = nullptr;
        for (Snap& sn : ss.snaps)
            if (sn.len == pos) d = &sn;
        for (Snap& sn : ss.snaps)
            if (!d && sn.len == 0) d = &sn;
        if (!d)
            for (Snap& sn : ss.snaps)
                if (!d || sn.when < d->when) d = &sn;
        const auto& g = m->g;
        const cudaMemcpyKind kind = snap_host ? cudaMemcpyDefault : cudaMemcpyDeviceToDevice;   // (host snapshots: by address)
        for (size_t l = 0; l < conv.size() && l < d->conv.size(); ++l) {
            if (!d->conv[l]) continue;
            CK(cudaMemcpyAsync(d->conv[l], conv[l], (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, kind, st));
            CK(cudaMemcpyAsync(d->S[l], S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, kind, st));
        }
        CK(cudaMemcpyAsync(d->hlast, hlast, (size_t) g.n_embd * 4, kind, st));
        CK(cudaStreamSynchronize(st));
        d->len = pos;
        d->rdelta = ss.rdelta;
        d->imgs = ss.imgs;
        d->when = ++use_clock;
    }
    /// Session dst takes snapshot k of session src: its recurrent state, src's first len cache cells (in place
    /// when dst is src), the position and the held tokens up to there.  dst's snapshots past that position no longer hold.
    void restore(int src, int k, int dst) {
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const Snap& sn = a.snaps[(size_t) k];
        const int64_t len = sn.len;
        // K and V cells: kv_cell() / kv_cell_v() (fp32, fp16, int8, or k8v4's int8 K + q4 V)
        const size_t cb = kv_cell(), row = (size_t) ctx * cb, cbv = kv_cell_v(), rowv = (size_t) ctx * cbv;
        const cudaMemcpyKind kind = snap_host ? cudaMemcpyDefault : cudaMemcpyDeviceToDevice;
        for (size_t l = 0; l < b.conv.size(); ++l) {   // (the MTP block's index, past the layers, has a cache only)
            if (l < sn.conv.size() && sn.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], sn.conv[l], (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, kind, st));
                CK(cudaMemcpyAsync(b.S[l], sn.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, kind, st));
            }
            if (src != dst && a.kc[l]) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, (size_t) len * cb, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], rowv, a.vc[l], rowv, (size_t) len * cbv, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
            }
        }
        CK(cudaMemcpyAsync(b.hlast, sn.hlast, (size_t) g.n_embd * 4, kind, st));
        CK(cudaStreamSynchronize(st));
        b.rdelta = sn.rdelta;
        b.imgs = sn.imgs;
        if (src != dst) {
            b.live.assign(a.live.begin(), a.live.begin() + (long) len);
            for (Snap& s_ : b.snaps) s_.len = 0;
        } else {
            b.live.resize((size_t) len);
            for (Snap& s_ : b.snaps)
                if (s_.len > len) s_.len = 0;
        }
        b.pos = len;
        if (dst == cur) load_cur();
    }
    static void drop_snaps(Sess& ss) {
        for (Snap& sn : ss.snaps) sn.len = 0;
    }

    /// The current session: `i`'s state, cache and position become the engine's (conv / S / kc / vc / pos / hlast).
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc;
        hlast = ss.hlast;
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar), for the current session.
    void set_delta(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        CK(cudaMemcpyAsync(d_delta, hpin + 60, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // (the pinned word is reused)
    }

    void init(const Model& model, int ctx_, int tmax_, int n_sessions = 1, bool vision_ = false) {
        m = &model;
        // the release attention unless --gqa-attn (set before init) selects the grouped-query kernel (for the fp32 or
        // int8 cache); --kv fp16 and k8v4 have no other decode kernel.  The expert cache's history window is on with
        // --kv fp16 (STRATA_Q36_CACHE_WINDOW=0/1 overrides it: a test setting).
        if (kv_f16 || kv_v4) gqa_attn = true;   // (k8v4's V has no release-kernel reader)
        cache_window = kv_f16;
        if (const char* v = std::getenv("STRATA_Q36_CACHE_WINDOW")) cache_window = std::atoi(v) != 0;
        // the fp32 cache switches from 8192 cells (as #3641); the int8 kernel (part_q8) with the wider merge is as fast
        // as the release kernel from 4096 cells on (a Qwen3.6-35B-A3B fine-tune, Q4_K_M, RTX 5080: 184 vs 179 t/s at 4K,
        // 165 vs 155 at 8K), so for int8 at once (one kernel at every position: a verify window and a decode step agree)
        gqa_from = kv8 ? 0 : 8192;
        if (const char* v = std::getenv("STRATA_Q36_GQA_FROM")) gqa_from = std::max(0, std::atoi(v));   // tests: the switch-over cell
        if (kv_v4) gqa_from = 0;   // (k8v4: the grouped-query kernel at every cell, whatever the test setting says)
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        if (side_on) {   // --side-stream: the stream and its events exist only then
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(__HIPCC__)   // (HIP: the default priority, as mtp.cpp)
            // the side stream at the highest priority: its small kernels start while the first matrix still holds every SM
            int lo = 0, hi = 0;
            CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            if (!(hi < lo)) {   // one priority level: an equal-priority side stream measured no gain (+0.3%)
                std::fprintf(stderr, "strata-q36: --side-stream needs a GPU with stream priorities: ignored\n");
                side_on = false;
            } else {
                CK(cudaStreamCreateWithPriority(&ast, cudaStreamNonBlocking, hi));
            }
#else
            std::fprintf(stderr, "strata-q36: --side-stream is not available on this build: ignored\n");   // HIP: not measured, and without the priority it does not pay
            side_on = false;
#endif
            if (side_on) for (cudaEvent_t* ev : {&ev_a0, &ev_a1}) CK(cudaEventCreateWithFlags(ev, cudaEventDisableTiming));
        }
        if (const char* v = std::getenv("STRATA_Q36_SIDE_SKEW")) side_skew_us = std::atoll(v);
        if (const char* v = std::getenv("STRATA_Q36_SIDE_SKEW_JOIN")) side_skew_join_us = std::atoll(v);
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
        qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
        qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
        vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
        gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        if (fused_pattn && (!kv8 || kv_v4 || kv_f16)) {
            std::fprintf(stderr, "strata-q36: --fused-prompt-attn needs --kv int8: ignored\n");
            fused_pattn = false;
        }
        if (fused_pattn) {   // mma.sync on FP16 needs sm_80; the kernel is built for 8 query heads of 256 per KV head
            int dev = 0, major = 0;
            CK(cudaGetDevice(&dev));
#if !defined(__HIPCC__)
            CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
#endif
            if (!q36pattn::supported(major, (int) g.n_head, (int) g.n_head_kv, (int) g.head_dim)) {
                std::fprintf(stderr, "strata-q36: --fused-prompt-attn needs an sm_80+ NVIDIA GPU and 8 query heads of 256 "
                             "per KV head: ignored\n");
                fused_pattn = false;
            }
        }
        if (!fused_pattn) att_s = dalloc(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        if (!fused_pattn) att_p16 = dalloc_t<__half>(kAttScoreFloats);
        if (chunked_gdn) {
            if (g.ssm_state_size != 128 || g.ssm_v_heads % g.ssm_k_heads != 0 || g.ssm_value_dim != g.ssm_v_heads * 128) {
                std::fprintf(stderr, "strata-q36: --chunked-gdn is built for 128 x 128 states: ignored\n");
                chunked_gdn = false;
            }
#if defined(__HIPCC__)
            chunked_gdn = false;   // (the kernels are CUDA only)
#endif
        }
        if (chunked_gdn) {
            gdn_L = dalloc(q36gdn::mat_floats((int) T, (int) g.ssm_v_heads));
            gdn_P = dalloc(q36gdn::mat_floats((int) T, (int) g.ssm_v_heads));
            gdn_cum = dalloc(q36gdn::cum_floats((int) T, (int) g.ssm_v_heads));
            std::fprintf(stderr, "strata-q36: chunked gated delta rule for prompt chunks (%.1f MiB of piece matrices)\n",
                         (double) ((2 * q36gdn::mat_floats((int) T, (int) g.ssm_v_heads) + q36gdn::cum_floats((int) T, (int) g.ssm_v_heads)) * 4) / (1 << 20));
        }
        if (fused_pattn) {
            int dev = 0;
            CK(cudaGetDevice(&dev));
            CK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
            pa_acc = dalloc(q36pattn::partial_acc_floats());
            pa_m = dalloc(q36pattn::kPartialRows);
            pa_l = dalloc(q36pattn::kPartialRows);
            std::fprintf(stderr, "strata-q36: fused prompt attention: no FP16 K / V copies or score buffers (%.0f MiB not "
                         "allocated, %.0f MiB of split partials)\n",
                         (double) ((size_t) g.n_head_kv * ctx * g.head_dim * 4 + kAttScoreFloats * 6) / (1 << 20),
                         (double) ((q36pattn::partial_acc_floats() + 2 * q36pattn::kPartialRows) * 4) / (1 << 20));
        }
        if (!kv_f16 && !fused_pattn) {   // --kv f16: the prompt attention reads the cache itself
            k16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
            v16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        }
        q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
        use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            mq_gu = dalloc((size_t) ne * 2 * FF);
            mq_h = dalloc((size_t) ne * FF);
        }
        std::fprintf(stderr, "strata-q36: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (const Layer& L : m->layers) take(L);
            if (m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-q36: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        // 64 query tokens per attention batch; the split-K path only takes batches of <= kGemmT (sized so with
        // the grouped-query kernel: 0.875 GiB less at 262144 cells)
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, gqa_attn ? kGemmT : 64));
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        for (const Layer& L : m->layers) {   // the verify windows' GDN inputs (one set, whichever session runs)
            if (L.attn) {
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                v_qkv.push_back(dalloc((size_t) kVerify * C));
                v_hc.push_back(dalloc((size_t) kVerify * C));
                v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
            }
        }
        if (m->has_mtp) {
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
            prob_m = dalloc(1);
        }
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess);
        {   // the sessions' caches: said before they fail (a slot's K / V at the context, fp32, for the attention layers and
            // the MTP block, and its recurrent state)
            size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
            for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
            const size_t per = (size_t) g.n_head_kv * (kv_cell() + kv_cell_v()) * (size_t) ctx * n_attn +
                               (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn +
                               (snap_host ? 0 : (size_t) n_snaps * snap_bytes());
            size_t fr0 = 0, tot0 = 0;
            CK(cudaMemGetInfo(&fr0, &tot0));
            if (n_sess > 1)
                std::fprintf(stderr, "strata-q36: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in "
                             "all), %.2f GiB free before them\n", n_sess, (double) per / (1 << 30), ctx,
                             (double) per * n_sess / (1 << 30), (double) fr0 / (1 << 30));
            if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr0) {
                std::fprintf(stderr, "strata-q36: not enough VRAM for %d slots at a context of %d (%.2f GiB each, %.2f GiB "
                             "free): use fewer slots (\"parallel\" / --batch) or a shorter --max-context\n", n_sess, ctx,
                             (double) per / (1 << 30), (double) fr0 / (1 << 30));
                std::exit(1);
            }
        }
        for (Sess& ss : sess) alloc_session(ss);
        if (n_snaps > 0)
            std::fprintf(stderr, "strata-q36: recurrent-state snapshots: %d per session, %.1f MiB each (%.1f MiB in all, in %s), "
                         "taken after prompt reads of %lld tokens or more\n", n_snaps, (double) snap_bytes() / (1 << 20),
                         (double) snap_bytes() * n_snaps * n_sess / (1 << 20), snap_host ? "pinned host memory" : "VRAM",
                         (long long) snap_min);
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
            const size_t ns = sess[0].conv.size();
            std::vector<float*> hc(ns * (size_t) n_sess), hs(hc), hk(hc), hv(hc);
            for (size_t li = 0; li < ns; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hc[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                }
            const std::pair<float***, std::vector<float*>*> tabs[4] = {{&d_conv_tab, &hc}, {&d_S_tab, &hs},
                                                                          {&d_kc_tab, &hk}, {&d_vc_tab, &hv}};
            for (const auto& tb : tabs) {
                CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(float*)));
                CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(float*),
                              cudaMemcpyHostToDevice));
            }
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocDefault));
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            d_map = dalloc_t<int32_t>(T);
            CK(cudaHostAlloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocDefault));
        }
        d_delta = dalloc_t<int32_t>(1);
        cur = 0;
        load_cur();
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
        if (kv8 || kv_f16)
            std::fprintf(stderr, "strata-q36: the K / V cache in %s\n", kv_v4 ? "int8 K + rotated q4_0 V" : kv8 ? "int8" : "fp16");
        if (gqa_attn || cache_window)
            std::fprintf(stderr, "strata-q36: grouped-query decode attention %s (from %d cells), expert-cache history window %s\n",
                         gqa_attn ? "on" : "off", kv_f16 ? 0 : gqa_from, cache_window ? "on" : "off");
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (STRATA_Q36_VERIFY=1 compares)
    bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;
    bool verify_k = std::getenv("STRATA_Q36_VERIFY") != nullptr;
    float* vbuf = nullptr;                       // verify scratch
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    void verify_cmp(const char* what, const float* a, const float* b, size_t n, const std::string& tag = "") {
        CK(cudaStreamSynchronize(st));
        std::vector<float> ha(n), hb(n);
        CK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) {
            ++v_bad;
            size_t i = 0;
            while (i < n && std::memcmp(&ha[i], &hb[i], 4) == 0) ++i;
            if (++v_by[what] <= 3)
                std::fprintf(stderr, "strata-q36 VERIFY MISMATCH %s%s: first at %zu of %zu: new %g old %g (bits %08x %08x)\n", what,
                             tag.c_str(), i, n, (double) ha[i], (double) hb[i], *(unsigned*) &ha[i], *(unsigned*) &hb[i]);
        }
    }
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048
    /// rows) for T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the 4-warp block of
    /// native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat& w, const void* q, float* out, int T) {
        if (!fast_k || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const half* dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(wp, dpl, q, out, nb, w.n_out, st);                  \
        else q36k::launch_q8_rows<NCV, 2, false>(wp, dpl, q, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int nc = std::min(8, T - c0);
                K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, vbuf + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                               nc, st);
            }
            verify_cmp("q8_small", out, vbuf, (size_t) T * w.n_out,
                       " n_in " + std::to_string(w.n_in) + " n_out " + std::to_string(w.n_out) + " T " + std::to_string(T));
        }
        return true;
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char* what, const float* hsrc, int T, const std::string& tag) {
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        if (!vq) CK(cudaMalloc(&vq, qb * 8));
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        const long long before = v_bad;
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
        if (v_bad != before && std::getenv("STRATA_Q36_VDEBUG") && v_by[what] <= 2) {   // a mismatching block, on the host
            std::vector<uint8_t> a(qb * (size_t) T), b(qb * (size_t) T);
            std::vector<float> hv((size_t) N * T);
            CK(cudaMemcpy(a.data(), xq, a.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), vq, b.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(), hsrc, hv.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t o = 0; o < a.size(); o += 36)
                if (std::memcmp(&a[o], &b[o], 36) != 0) {
                    const size_t blk = o / 36;
                    float v[32], t[32];
                    for (int l = 0; l < 32; ++l) v[l] = t[l] = hv[blk * 32 + (size_t) l];
                    for (int off = 16; off > 0; off >>= 1) {   // the xor butterfly, lane by lane
                        float u[32];
                        for (int l = 0; l < 32; ++l) u[l] = t[l] + t[l ^ off];
                        std::memcpy(t, u, sizeof(u));
                    }
                    unsigned ds_a, ds_b;
                    const __half t0h = __float2half(t[0]);   // host side: __half_as_ushort is __device__ only
                    unsigned short t0bits;
                    std::memcpy(&t0bits, &t0h, sizeof(t0bits));
                    std::memcpy(&ds_a, &a[o], 4);
                    std::memcpy(&ds_b, &b[o], 4);
                    std::fprintf(stderr, "VDEBUG block %zu: ds new %08x old %08x, qs bytes %s, cpu butterfly sum %.9g (half %04x)\n",
                                 blk, ds_a, ds_b, std::memcmp(&a[o + 4], &b[o + 4], 32) == 0 ? "equal" : "DIFFER", (double) t[0],
                                 (unsigned) t0bits);
                    break;
                }
        }
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                if (!vx_) {
                    CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                    CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
                }
                CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
                k_add_rms<<<T, kThreads, 0, st>>>(vx_, y, L.post_norm, vh_, N, m->eps);
            }
            q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (q36k::QB*) xq, N, m->eps);
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fast_k || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0 ||
            (K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled()))
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const half* da = nullptr;
        const half* db = nullptr;
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (pkd)                                                                                             \
            q36k::k_sh_gu<NCV, 2, true><<<grid, 1024, 0, st>>>(wa, da, wb, db, (const q36k::QB*) xq,        \
                                                               (q36k::QB*) q8, nb, FS);                      \
        else                                                                                                 \
            q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const q36k::QB*) xq, \
                                                                (q36k::QB*) q8, nb, FS);                     \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            if (!vq) CK(cudaMalloc(&vq, K::native_q8_1_bytes(m->g.n_embd, 1) * 8));
            float* vg = vbuf;
            float* vu = vbuf + (size_t) 8 * FS;
            mmq_ref(a, xq, vg, T);
            mmq_ref(b, xq, vu, T);
            K::native_swiglu_quantize_q8_1(vg, vu, vq, FS, T, st);
            verify_cmp("shared_gu q8", (const float*) q8, (const float*) vq, K::native_q8_1_bytes(FS, 1) * (size_t) T / 4,
                       " T " + std::to_string(T));
        }
        return true;
    }
    /// The old products (native MMVQ), for the checks.
    void mmq_ref(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, st);
        }
    }
    void group_old(const Layer& L, const int32_t* slot_of, int T, unsigned long long* o_ptr, int32_t* o_start, int32_t* o_ng,
                   int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok, L.lent, L.n_reg);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                                            o_tok);
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer& L, const int32_t* slot_of, int T) {
        if (!vgrp) CK(cudaMalloc((void**) &vgrp, 2048));
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        CK(cudaStreamSynchronize(st));
        int nn = 0, no = 0;
        CK(cudaMemcpy(&nn, n_groups, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&no, o_ng, 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        bool bad = nn != no;
        if (!bad) {
            std::vector<unsigned long long> a((size_t) nn + 1), b((size_t) nn + 1);
            std::vector<int32_t> sa((size_t) nn + 1), sb((size_t) nn + 1), da(64), db(64), ta(64), tb(64);
            CK(cudaMemcpy(a.data(), grp_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sa.data(), grp_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sb.data(), o_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            const int tot = sa[(size_t) nn];
            CK(cudaMemcpy(da.data(), ent_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(db.data(), o_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ta.data(), ent_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(tb.data(), o_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            bad = std::memcmp(a.data(), b.data(), (size_t) nn * 8) != 0 ||
                  std::memcmp(sa.data(), sb.data(), ((size_t) nn + 1) * 4) != 0 ||
                  std::memcmp(da.data(), db.data(), (size_t) tot * 4) != 0 ||
                  std::memcmp(ta.data(), tb.data(), (size_t) tot * 4) != 0;
        }
        if (bad) {
            ++v_bad;
            if (v_bad <= 20) std::fprintf(stderr, "strata-q36 VERIFY MISMATCH group (T %d, groups %d vs %d)\n", T, nn, no);
        }
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer& L, const int32_t* slot_of, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_k || T > 8 || m->g.n_expert != 256 || Kp != 8) return false;
        if (slot_of)
            q36k::k_route_group<true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, L.blobs, L.stride, grp_ptr,
                                                                 grp_start, n_groups, ent_dst, ent_tok, L.lent, L.n_reg);
        else
            q36k::k_route_group<false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, L.blobs, L.stride, grp_ptr,
                                                                  grp_start, n_groups, ent_dst, ent_tok);
        grouped_by_route = true;
        if (verify_k) {
            // the old top-k into scratch, the ids / weights compared, then the old grouping of the new ids
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            int32_t* vid = (int32_t*) (vbuf + 8 * 8192);
            float* vw = vbuf + 8 * 8192 + 64;
            K::native_router_topk(logits_r, vid, vw, 256, Kp, T, st);
            verify_cmp("route ids", (const float*) ids, (const float*) vid, (size_t) T * Kp, " T " + std::to_string(T));
            verify_cmp("route weights", ewts, vw, (size_t) T * Kp, " T " + std::to_string(T));
            group_verify(L, slot_of, T);
        }
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of)
                q36k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok, L.lent, L.n_reg);
            else
                q36k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout, int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            if (!vx_) {
                CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
            }
            CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
            k_moe_out<<<T, kThreads, 0, st>>>(vx_, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, vh_, N, m->eps);
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            if (Kp == 8 && N == 2048)   // 8 experts, 2048 wide: 256 threads, the values in registers (same bits)
                q36k::k_moe_out_s8<<<T, 256, 0, st>>>(x, eparts, prow, ewts, shared, sgate, sgs, wn, hout, N, m->eps,
                                                      mkq ? (q36k::QB*) xq : nullptr);
            else
                q36k::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps,
                                                          mkq ? (q36k::QB*) xq : nullptr);
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps);
    }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        if (q8_small(w, q, out, T)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    void layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (li == 0 && (async_swaps || lend_on || pf_check) && m->hybrid && main_hybrid) {   // a new main forward: the swap copies that landed enter the tables
            if (async_swaps || lend_on) swap_settle();
            if (pf_check) table_check();
        }
        if (pf_slots && li == 0) {   // --expert-prefetch: a new main forward
            ++pf_fwd;
            pf_used = 0;
        }
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            const bool side = side_on && !use_graph && T <= kGemmT;   // z and alpha / beta on the side stream, beside qkv
            if (side) CK(cudaEventRecord(ev_a0, st));
            apply(L.qkv, xq, qkv_b, T);
            if (side) {
                CK(cudaStreamWaitEvent(ast, ev_a0, 0));
                side_spin(ast, side_skew_us);
                StreamScope on_side(st, ast);
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N, HV);
                apply(L.zg, xq, z, T);
                CK(cudaEventRecord(ev_a1, ast));
            }
            prefetch_issue();   // --expert-prefetch: the last layer's prediction's copies, the stream fed first
            if (!side) {
                apply(L.zg, xq, z, T);
                if (T <= kGemmT) {
                    k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                          HV);
                } else {
                    f32mm(L.alpha, h, al_b, N, HV, T);
                    f32mm(L.beta, h, be_b, N, HV, T);
                    k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
                }
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                if (side) {   // z, alpha, beta: first read here
                    side_spin(st, side_skew_join_us);
                    CK(cudaStreamWaitEvent(st, ev_a1, 0));
                }
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    if (chunked_gdn) {
                        CK(q36gdn::launch(S[li], hc_b, C, al_b, be_b, gdn_y, HK, HV, T, gdn_L, gdn_P, gdn_cum, st));
                    } else {
                        k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                          be_b, gdn_y, HK, HV, T);
                    }
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    if (side) {   // z, alpha, beta: first read here
                        side_spin(st, side_skew_join_us);
                        CK(cudaStreamWaitEvent(st, ev_a1, 0));
                    }
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            const bool side = side_on && !use_graph && T <= kGemmT;   // k and v (512 rows each: launch-bound) beside q
            if (side) CK(cudaEventRecord(ev_a0, st));
            apply(L.wq, xq, qg, T);
            if (side) {
                CK(cudaStreamWaitEvent(ast, ev_a0, 0));
                side_spin(ast, side_skew_us);
                StreamScope on_side(st, ast);
                apply(L.wk, xq, kx, T);
                apply(L.wv, xq, vx, T);
                CK(cudaEventRecord(ev_a1, ast));
            }
            prefetch_issue();   // --expert-prefetch: the last layer's prediction's copies, the stream fed first
            if (side) {
                side_spin(st, side_skew_join_us);
                CK(cudaStreamWaitEvent(st, ev_a1, 0));   // k and v: first read by k_attn_prep
            } else {
                apply(L.wk, xq, kx, T);
                apply(L.wv, xq, vx, T);
            }
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
            }
            if (kv_f16)
                k_attn_prep<__half, false, false><<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate,
                    reinterpret_cast<__half*>(rows_mode ? nullptr : kc[li]),
                    reinterpret_cast<__half*>(rows_mode ? nullptr : vc[li]), H, HKV, HD, m->n_rot, dpos, ctx,
                    m->rope_base, m->eps, d_delta, mtab_now, rs);
            else
                (kv_v4 ? k_attn_prep<float, true, true> : kv8 ? k_attn_prep<float, true, false> : k_attn_prep<float, false, false>)
                    <<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, rows_mode ? nullptr : kc[li],
                                                              rows_mode ? nullptr : vc[li], H, HKV, HD, m->n_rot, dpos, ctx,
                                                              m->rope_base, m->eps, d_delta, mtab_now, rs);
            if (T > kGemmT && fused_pattn) {   // (never set under HIP: init clears it)
                // --fused-prompt-attn: the whole attention of the chunk in one kernel from the int8 cells (q36_pattn.cuh)
                CK(q36pattn::launch(q, reinterpret_cast<const q36attn::q8_cell*>(kc[li]),
                                    reinterpret_cast<const q36attn::q8_cell*>(vc[li]), ctx, H, HKV, T, (int) p0, att,
                                    pa_acc, pa_m, pa_l, n_sm, st));
                k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
            } else if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                if (!kv_f16)
                    (kv_v4 ? k_kv_f16<true, true> : kv8 ? k_kv_f16<true, false> : k_kv_f16<false, false>)
                        <<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(kc[li], vc[li], k16, v16, ctx, P, HD, HKV);
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = kv_f16 ? reinterpret_cast<const __half*>(kc[li]) + (size_t) kvh * ctx * HD
                                                 : k16 + (size_t) kvh * P * HD;
                        const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                       0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                       (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0,
                                                                       1.0f / sqrtf((float) HD));
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Vh = kv_f16 ? reinterpret_cast<const __half*>(vc[li]) + (size_t) kvh * ctx * HD
                                                 : v16 + (size_t) kvh * P * HD;
                        const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD,
                                                       0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                       H * HD, HD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                if (kv_v4)   // k8v4: the rows mix rotated values; rotated back, then gated
                    q36attn::unrotate_gate<<<T * H, 32, 0, st>>>(att, gate);
                else
                    k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    // #3641's grouped-query kernel: a block per KV head reads each cell once for its query heads (the release
                    // kernel reads it once per query head); in graph mode its choice holds for the whole context
                    if (kv_f16)
                        q36attn::part<<<dim3(nc, nt * HKV), kThreads, 0, st>>>(q,
                            reinterpret_cast<const __half*>(rows_mode ? nullptr : kc[li]),
                            reinterpret_cast<const __half*>(rows_mode ? nullptr : vc[li]), dpos, t0, H, HKV, ctx,
                            1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    // (from gqa_from cells: 8192 for fp32, 0 for int8 and k8v4)
                    else if (gqa_attn && H / HKV <= 8 && (graph_mode ? ctx : p0 + t0 + nt) >= gqa_from) {
                        // (q36attn::launch_part_q8x: the cache through shared memory, one pass for all the window's rows of a
                        // session; STRATA_Q36_ATTN_OLD=1 runs part_q8, the same bits)
                        if (kv_v4)   // k8v4: the same kernel on q4 V cells (rotated partials)
                            q36attn::launch_part_q8x(q,
                                reinterpret_cast<const q36attn::q8_cell*>(rows_mode ? nullptr : kc[li]),
                                reinterpret_cast<const q36attn::q4_cell*>(rows_mode ? nullptr : vc[li]), dpos, t0, nt, H, HKV, ctx,
                                1.0f / sqrtf((float) HD), nc, n_chunk, pacc, pm, pl, rs, rows_mode, st);
                        else if (kv8)   // the int8 cache's own kernel (q36_attn.cuh)
                            q36attn::launch_part_q8x(q,
                                reinterpret_cast<const q36attn::q8_cell*>(rows_mode ? nullptr : kc[li]),
                                reinterpret_cast<const q36attn::q8_cell*>(rows_mode ? nullptr : vc[li]), dpos, t0, nt, H, HKV, ctx,
                                1.0f / sqrtf((float) HD), nc, n_chunk, pacc, pm, pl, rs, rows_mode, st);
                        else
                            q36attn::part<<<dim3(nc, nt * HKV), kThreads, 0, st>>>(q, rows_mode ? nullptr : kc[li],
                                rows_mode ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    } else
                        (kv8 ? k_attn_part<true> : k_attn_part<false>)<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q,
                                                                             rows_mode ? nullptr : kc[li],
                                                                             rows_mode ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                             1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    if (kv_v4) {   // rotated rows: merged ungated, then rotated back and gated
                        q36attn::merge<<<dim3(nt * H, HD / 32), kThreads, 0, st>>>(pacc, pm, pl, nullptr, dpos, t0, H, n_chunk, att, rs);
                        q36attn::unrotate_gate<<<nt * H, 32, 0, st>>>(att + (size_t) t0 * H * HD, gate + (size_t) t0 * H * HD);
                    } else if (gqa_attn)   // the same merge spread over 8 x the blocks (q36_attn.cuh)
                        q36attn::merge<<<dim3(nt * H, HD / 32), kThreads, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs);
                    else
                        k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs);
                }
            }
            mm(L.wo, att, y, T);
        }
        add_rms_post(L, T);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        // a decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when the fused
        // router follows; the prompt chunks and the checks keep the separate products
        const bool ext = one || (fast_k && std::getenv("STRATA_Q36_NO_EXT") == nullptr && !verify_k && T <= kGemmT && g.n_expert == 256 && Kp == 8);
        // (the next layer takes the hybrid rows path only for T <= hybrid_t: a bigger batch streams its experts, a prediction would be wasted)
        if (pf_slots && T <= std::min<int>(kPfRows, hybrid_t) && li + 1 < m->layers.size() && lhy(li + 1)) {
            // --expert-prefetch: the next layer's router on this layer's MoE input, on pst beside this layer's experts,
            // into mapped host memory; the stream waits for it before h is rewritten (pf_join)
            CK(cudaEventRecord(ev_pf0, st));
            CK(cudaStreamWaitEvent(pst, ev_pf0, 0));
            k_pred_gemv<<<dim3(((int) g.n_expert + 1) / 2, T), 256, 0, pst>>>(
                m->layers[li + 1].router, h, d_pflog + ((li + 1) % 2) * kPfRows * g.n_expert, N, (int) g.n_expert);
            pf_rows = T;
            CK(cudaEventRecord(ev_pred, pst));
            pf_for = (int) li + 1;
            pf_join_pending = true;
        }
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        if (pf_slots && hy_rows) prefetch_publish(li);
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        // the shared expert, scaled by sigmoid(w . h); a prompt chunk on MMQ runs it before the host waits for the routing
        bool sh_done = false;
        auto shared_expert = [&]() {
            sh_done = true;
            if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
            if (T <= kGemmT) {
                if (!shared_gu(L, T)) {
                    apply(L.sh_gate, xq, sg, T);
                    apply(L.sh_up, xq, su, T);
                    K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);   // SwiGLU and its q8_1 image in one launch
                }
                mmq(L.sh_down, q8, sy, T);
            } else {
                apply(L.sh_gate, xq, sg, T);
                apply(L.sh_up, xq, su, T);
                k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
                mm(L.sh_down, shh, sy, T);
            }
            if (!ext) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        };
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
            shared_expert();   // independent of the routing: the GPU runs it while the host waits for the counts
            CK(cudaEventSynchronize(ev_route));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) {
                    hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
                    if (count_heat) hy[li].heat[(size_t) x] += h_cnt[li * g.n_expert + x];
                }
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            // each token's row once, written to its Kp sorted rows (rinv: entry -> sorted row): the bytes of quantizing per entry
            // (NVFP4 w4a4 activations carry a per-row scale the scatter does not write: quantize() refuses them without one)
            if (MMQ::fp4_activations(L.L.gu_type)) MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            else MMQ::quantize_scatter(h, rinv, ent_tok, mq_x, L.L.gu_type, N, N, T, Kp, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done();
            prow = rinv;
        } else {
            group_entries(L, nullptr, T);
            if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
        }
        if (!sh_done) shared_expert();
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && T > hybrid_t && use_mmq) {
            std::vector<int> seq;
            for (size_t l = 0; l < m->layers.size(); ++l)
                if (lhy(l)) seq.push_back((int) l);
            stream_begin(seq);
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);   // a picture's cells take its rows
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);   // later layers: from k_moe_out
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const bool last = l + 1 == m->layers.size();
            layer(m->layers[l], l, T, p0, pos0, graph_mode, verify, last ? m->out_norm : m->layers[l + 1].attn_norm,
                  last ? hmain : h);
        }
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (draft_cols() == (int) g.n_vocab) {
            head_rows(hn, 1, logits_m, pick_m);
            if (spec_min_p > 0.f) k_pick_prob<<<1, 1024, 0, st>>>(logits_m, (int) g.n_vocab, pick_m, prob_m);
            return;
        }
        quant(hn, N, 1, q8);
        const uint8_t* pk = nullptr;
        for (const auto& p : m->packed)
            if (std::get<0>(p) == hd.d) pk = std::get<1>(p);
        if (pk) {
            const half* dpl = reinterpret_cast<const half*>(pk + (size_t) hd.n_in * hd.n_out);
            q36k::launch_q8_rows<1, 2, true>(pk, dpl, q8, logits_m, 64, draft_n, st);
        } else {
            q36k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        }
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
        if (spec_min_p > 0.f) k_pick_prob<<<1, 1024, 0, st>>>(logits_m, draft_n, pick_m, prob_m);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(m->mtp, m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) head_draft(hmtp + (size_t) (T - 1) * N);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        lend_evict();   // --lend-staging: the staging buffers are needed
        static const bool tail_check = std::getenv("STRATA_Q36_STAGE_TAIL_CHECK") != nullptr;   // debug: the zero tail MMQ reads past a layer
        if (tail_check && stage[0] && stage_max_layer) {
            CK(cudaStreamSynchronize(st));
            CK(cudaStreamSynchronize(cst));
            std::vector<uint8_t> t(kMmqTail);
            for (int b = 0; b < 2; ++b) {
                CK(cudaMemcpy(t.data(), stage[b] + stage_max_layer, kMmqTail, cudaMemcpyDeviceToHost));
                ++tail_checks;
                for (uint8_t v : t)
                    if (v) { ++tail_bad; break; }
            }
        }
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs).  --split-stage-copy takes the runs from 1/8 of
        // the experts resident: a device-to-device copy is a kernel, queued on the copy stream it waits for the compute kernels'
        // SMs and holds up the PCIe copies behind it, so the resident runs go on a high-priority stream of their own (RTX 5080,
        // 78 of 256 resident, 16K: pp 4570 t/s in runs on one stream, 5760 whole, 6090 with the stream of their own)
        if (all || (split_copy ? 8 : 2) * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            // the resident experts' runs: on cst, in order with the PCIe copies, unless --split-stage-copy gave them a stream
            cudaStream_t ds = split_copy ? dst2 : cst;
            if (split_copy) CK(cudaStreamWaitEvent(dst2, ev_used[b], 0));
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, ds));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
            }
            if (split_copy) {
                CK(cudaEventRecord(ev_d2d[b], dst2));
                CK(cudaStreamWaitEvent(cst, ev_d2d[b], 0));
            }
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        // --hybrid-zero-copy (or --expert-prefetch, whose copies hold the copy engine) stores them from a kernel
        if (zc || pf_slots)
            k_route_to_host<<<(T * N / 4 + 255) / 256, 256, 0, st>>>(ids, h, h_ids, h_h, (int) n_ent, T * N);
        else {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaEventRecord(ev_route, st));
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        if (pf_slots) pf_settle(li);   // --expert-prefetch: the copies the GPU took (before the plan reads slot_of)
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;
        std::map<int, size_t> open;   // expert -> its job with room
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer) {
                H.score[(size_t) x] += 1.0;
                if (count_heat) H.heat[(size_t) x] += 1.0;
            }
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (pf_slots && H.slot_of[(size_t) x] >= L.n_slots && H.slot_of[(size_t) x] < L.n_slots + (int) H.pf.size()) {   // a prefetch slot: this forward reads it
                ++pf_hits;
                H.pf[(size_t) (H.slot_of[(size_t) x] - L.n_slots)].last_ref = pf_fwd;
            }
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) x * L.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) {
            pf_join();
            return;
        }
        hy_plan_ms += ms_since(t0);
        ++hy_layers;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        pf_join();
        if (zc && n_ent <= 64 && N % 1024 == 0) {   // --hybrid-zero-copy: the rows read in place
            unsigned long long mask = 0;
            for (int e = 0; e < n_ent; ++e)
                if (h_miss[e]) mask |= 1ull << e;
            k_merge_mapped<<<dim3(N / 1024, n_ent), 256, 0, st>>>(eparts, cpu_out, mask, N);
            return;
        }
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
    }

    /// --expert-prefetch: layer nl's predicted misses (pf_for: its prediction is on the host once ev_pred is)
    /// into its free prefetch slots (pf_free), copies on pfst; prefetch_publish puts the landed ones in nl's table.
    void prefetch_issue() {
        if (pf_for < 0) return;
        const size_t nl = (size_t) pf_for;
        pf_for = -1;
        while (cudaEventQuery(ev_pred) == cudaErrorNotReady) {
        }
        Hy& H = hy[nl];
        const Layer& L = layer_at((int) nl);
        const int ne = (int) m->g.n_expert;
        // the candidates: each row's predicted top pf_topk, the experts more rows predict first (then the higher logit);
        // one row: its top pf_topk in logit order
        pf_cnt.assign((size_t) ne, 0);
        pf_best.assign((size_t) ne, -INFINITY);
        pf_order.clear();
        pf_idx.resize((size_t) ne);
        for (int r = 0; r < pf_rows; ++r) {
            const float* pl = h_pflog + ((nl % 2) * kPfRows + (size_t) r) * ne;
            for (int x = 0; x < ne; ++x) pf_idx[(size_t) x] = x;
            std::partial_sort(pf_idx.begin(), pf_idx.begin() + pf_topk, pf_idx.end(),
                              [&](int a, int b) { return pl[a] > pl[b] || (pl[a] == pl[b] && a < b); });
            for (int k = 0; k < pf_topk; ++k) {
                const int x = pf_idx[(size_t) k];
                if (pf_cnt[(size_t) x]++ == 0) pf_order.push_back(x);
                pf_best[(size_t) x] = std::max(pf_best[(size_t) x], pl[x]);
            }
        }
        std::sort(pf_order.begin(), pf_order.end(), [&](int a, int b) {
            if (pf_cnt[(size_t) a] != pf_cnt[(size_t) b]) return pf_cnt[(size_t) a] > pf_cnt[(size_t) b];
            if (pf_best[(size_t) a] != pf_best[(size_t) b]) return pf_best[(size_t) a] > pf_best[(size_t) b];
            return a < b;
        });
        const int ncand = (int) pf_order.size();
        ++pf_preds;
        for (int k = 0; k < ncand; ++k) {
            const int x = pf_order[(size_t) k];
            if (H.slot_of[(size_t) x] >= 0 || pf_holds(H, x) || H.pend_x[(size_t) x]) continue;   // resident, prefetched, on its way (also into a cache or lent slot)
            const int p = pf_free(H);
            if (p < 0 || !pf_copy(H, L, p, x)) break;
        }
#if defined(STRATA_Q36_CPU_POOL)
        if (pool_warm_k > 0) {   // the next candidates (no slot, no copy): the CPU computes them, so its workers load their rows now
            const uint8_t* wb[strata::kernels::cpu::ExpertPool::kMaxWarm];
            int nw = 0;
            for (int k = 0; k < ncand && nw < pool_warm_k; ++k) {
                const int x = pf_order[(size_t) k];
                if (H.slot_of[(size_t) x] >= 0 || pf_holds(H, x)) continue;
                wb[nw++] = L.hblobs + (size_t) x * L.stride;
            }
            if (nw > 0) pool->warm_post(H.fmt, wb, nw);
        }
#endif
    }
    /// a prefetch slot of H holds x or is loading it
    static bool pf_holds(const Hy& H, int x) {
        for (const Hy::PfSlot& q : H.pf)
            if (q.state != 0 && q.x == x) return true;
        return false;
    }
    /// The next free prefetch slot of H, round robin, or -1: not loading (a copy not seen landed is never overwritten),
    /// not read by a plan of this forward (its kernels may still run; earlier forwards' are done - the host waited for
    /// this forward's routing), and not holding one of the current prediction's candidates.
    int pf_free(const Hy& H) const {
        const int P = (int) H.pf.size();
        for (int i = 0; i < P; ++i) {
            const int p = (H.pf_next + i) % P;
            const Hy::PfSlot& q = H.pf[(size_t) p];
            if (q.state == 1 || q.last_ref >= pf_fwd) continue;
            if (q.state == 2 && pf_cnt[(size_t) q.x] > 0) continue;
            return p;
        }
        return -1;
    }
    /// Expert x's blob into prefetch slot p of H (on pfst); its old expert leaves the table (pending patch).  False when
    /// the forward's copy budget is spent.
    bool pf_copy(Hy& H, const Layer& L, int p, int x) {
        if (pf_no_copy || (pf_budget > 0 && pf_used >= pf_budget)) return false;
        const int S = L.n_slots;
        Hy::PfSlot& q = H.pf[(size_t) p];
        if (q.state == 2 && H.slot_of[(size_t) q.x] == S + p) {
            H.slot_of[(size_t) q.x] = -1;
            H.evicted.push_back(q.x);
        }
        CK(cudaMemcpyAsync(L.blobs + (size_t) (S + p) * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                           cudaMemcpyHostToDevice, pfst));
        // (the word 0 means "not landed": the number skips it when the counter wraps, 2^31 copies on)
        if (pf_gen >= INT32_MAX) std::fprintf(stderr, "strata-q36: the prefetch copies' numbers wrapped (skipping 0)\n");
        pf_gen = pf_gen >= INT32_MAX ? 1 : pf_gen + 1;
        q.gen = pf_gen;
        k_pf_landed<<<1, 1, 0, pfst>>>(H.flag_d + p, q.gen);
        q.x = x;
        q.state = 1;
        q.offered = false;
        H.pf_next = (p + 1) % (int) H.pf.size();
        ++pf_used;
        ++pf_copies;
        return true;
    }
    /// the stream waits for the prediction's read of h before anything rewrites it (where it waits for the pool anyway)
    void pf_join() {
        if (!pf_join_pending) return;
        pf_join_pending = false;
        CK(cudaStreamWaitEvent(st, ev_pred, 0));
    }
    /// ... before layer li's routing: k_pf_publish with the evictions and the loading copies (the GPU takes the landed
    /// ones into the table as it gets there - the host enqueues ahead of the GPU, so it cannot tell)
    void prefetch_publish(size_t li) {
        Hy& H = hy[li];
        const int S = layer_at((int) li).n_slots;
        PfPub u{};
        // (between two publishes at most P evictions and P loading copies, P <= 8: PfPub's 16 hold them)
        if (H.evicted.size() > 16 || H.pf.size() > 16) {
            std::fprintf(stderr, "strata-q36: %zu prefetch evictions, %zu slots: more than k_pf_publish takes\n",
                         H.evicted.size(), H.pf.size());
            std::abort();
        }
        for (int32_t x : H.evicted) u.ev_x[u.nev++] = x;
        H.evicted.clear();
        for (size_t p = 0; p < H.pf.size(); ++p) {
            Hy::PfSlot& q = H.pf[p];
            if (q.state != 1) continue;
            if (H.slot_of[(size_t) q.x] >= 0) {   // rebalance gave it a slot meanwhile: the slot is free (a later copy
                q.state = 0;                       // into it is behind this one on pfst)
                q.x = -1;
                q.offered = false;
                continue;
            }
            u.x[u.n] = q.x;
            u.slot[u.n] = S + (int) p;
            u.gen[u.n] = q.gen;
            u.p[u.n++] = (int) p;
            q.offered = true;
        }
        if (u.n || u.nev) k_pf_publish<<<1, 1, 0, st>>>(H.slot_of_d, H.flag_d, H.out_d, u);
    }
    /// ... and once the routing is on the host (moe_hybrid_cpu): the host's table takes what the GPU did
    void pf_settle(size_t li) {
        Hy& H = hy[li];
        const int S = layer_at((int) li).n_slots;
        for (size_t p = 0; p < H.pf.size(); ++p) {
            Hy::PfSlot& q = H.pf[p];
            if (q.state != 1 || !q.offered) continue;
            q.offered = false;
            if (((volatile int32_t*) H.out_h)[p] != q.gen) {
                ++pf_late;
                continue;
            }
            q.state = 2;
            H.slot_of[(size_t) q.x] = S + (int) p;
        }
    }
    /// --async-swaps: the swaps rebalance chose (sw_new): the copies on swst after the table with the victims gone is on the
    /// device (the fence), in batches, each with an event
    void swap_issue() {
        if (sw_new.empty()) return;
        CK(cudaEventRecord(ev_swfence, st));
        CK(cudaStreamWaitEvent(swst, ev_swfence, 0));
        for (size_t i = 0; i < sw_new.size(); i += kSwapBatch) {
            SwapBatch b;
            const size_t e = std::min(sw_new.size(), i + kSwapBatch);
            for (size_t j = i; j < e; ++j) {
                const SwapItem& it = sw_new[j];
                const Layer& L = layer_at(it.li);
                CK(cudaMemcpyAsync(L.slot_addr(it.s), L.hblobs + (size_t) it.x * L.stride, L.L.bytes, cudaMemcpyHostToDevice, swst));
                b.items.push_back(it);
            }
            if (swap_ev_free.empty()) {
                cudaEvent_t ev;
                CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                swap_ev_free.push_back(ev);
            }
            b.ev = swap_ev_free.back();
            swap_ev_free.pop_back();
            CK(cudaEventRecord(b.ev, swst));
            swap_q.push_back(std::move(b));
            ++sw_batches;
        }
        sw_new.clear();
    }
    /// ... and at the start of a main forward (the previous one is over; a batch still copying stays pending, nobody reads its
    /// slots): the batches whose event is done enter the tables
    void swap_settle() {
        if (swap_q.empty()) return;
        std::vector<char> dirty(hy.size(), 0);
        while (!swap_q.empty()) {
            SwapBatch& b = swap_q.front();
            const cudaError_t qe = cudaEventQuery(b.ev);
            if (qe == cudaErrorNotReady) {
                (void) cudaGetLastError();
                break;
            }
            CK(qe);
            for (const SwapItem& it : b.items) {
                Hy& H = hy[(size_t) it.li];
                const int S = layer_at(it.li).n_slots;
                if (H.slot_of[(size_t) it.x] >= S && H.slot_of[(size_t) it.x] < S + (int) H.pf.size()) {   // it was served from a prefetch slot: that slot is free again
                    H.pf[(size_t) (H.slot_of[(size_t) it.x] - S)].state = 0;
                    H.pf[(size_t) (H.slot_of[(size_t) it.x] - S)].x = -1;
                }
                H.slot_of[(size_t) it.x] = it.s;
                H.pend_s[(size_t) it.s] = 0;
                H.pend_x[(size_t) it.x] = 0;
                dirty[(size_t) it.li] = 1;
                ++sw_taken;
            }
            swap_ev_free.push_back(b.ev);
            swap_q.pop_front();
        }
        const int64_t ne = m->g.n_expert;
        for (size_t li = 0; li < hy.size(); ++li)
            if (dirty[li]) {
                CK(cudaMemcpyAsync(hy[li].slot_of_d, hy[li].slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
                hy[li].evicted.clear();   // the whole table went up
            }
    }
    /// --lend-staging: the prompt stream is about to use the staging buffers: the lent slots go out of the tables (no copy; the
    /// swap copies in flight land first), and the staging copies wait for the table without them
    void lend_evict() {
        if (!lend_on || !lend_active) return;
        const auto t0 = Clock::now();
        if (swst) CK(cudaStreamSynchronize(swst));
        swap_settle();
        const int64_t ne = m->g.n_expert;
        for (size_t li = 0; li < hy.size(); ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            bool ch = false;
            for (int j = 0; j < L.lent_n; ++j) {
                const int s_ = L.n_reg + j, x = H.exp_in_slot[(size_t) s_];
                if (x < 0) continue;
                if (H.slot_of[(size_t) x] == s_) H.slot_of[(size_t) x] = -1;
                H.exp_in_slot[(size_t) s_] = -1;
                ch = true;
            }
            if (ch) CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
        }
        CK(cudaEventRecord(ev_lend, st));
        CK(cudaStreamWaitEvent(cst, ev_lend, 0));
        lend_active = false;
        ++lend_evicts;
        lend_evict_ms += ms_since(t0);
    }
    /// ... and a lent slot with no session history behind it (at load, after the warm-up): each layer's next experts in the
    /// profile's order that are not resident, copied in now
    void lend_fill_profile() {
        if (!lend_on) return;
        lend_evict();
        const int64_t ne = m->g.n_expert;
        for (size_t li = 0; li < hy.size(); ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            size_t k = 0;
            for (int j = 0; j < L.lent_n; ++j) {
                while (k < H.prof_order.size() && H.slot_of[(size_t) H.prof_order[k]] >= 0) ++k;   // (never one that is resident)
                if (k >= H.prof_order.size()) break;
                const int x = H.prof_order[k++], s_ = L.n_reg + j;
                CK(cudaMemcpyAsync(L.lent + (size_t) j * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes, cudaMemcpyHostToDevice, st));
                H.slot_of[(size_t) x] = s_;
                H.exp_in_slot[(size_t) s_] = x;
                ++lend_fills;
            }
            CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
        }
        CK(cudaStreamSynchronize(st));
        lend_active = true;
    }
    /// STRATA_Q36_PF_CHECK=1 (debug): the device slot tables against the host's, and the host's against its slot bookkeeping
    std::vector<uint8_t> bytes_buf;
    void table_check() {
        CK(cudaStreamSynchronize(st));
        const int64_t ne = m->g.n_expert;
        std::vector<int32_t> d((size_t) ne);
        ++chk_calls;
        for (size_t li = 0; li < hy.size(); ++li) {
            const Hy& H = hy[li];
            if (!H.slot_of_d) continue;
            const int S = layer_at((int) li).n_slots;
            CK(cudaMemcpy(d.data(), H.slot_of_d, (size_t) ne * 4, cudaMemcpyDeviceToHost));
            for (int x = 0; x < ne; ++x) {
                const int hs = H.slot_of[(size_t) x];
                bool bad = d[(size_t) x] != hs;
                const int Pl = (int) H.pf.size();
                if (!bad && hs >= 0 && (hs < S || hs >= S + Pl))   // a cache slot or a lent one
                    bad = (size_t) hs >= H.exp_in_slot.size() || H.exp_in_slot[(size_t) hs] != x || H.pend_s[(size_t) hs] ||
                          (hs >= S + Pl && !lend_active);
                if (!bad && hs >= S && hs < S + Pl) bad = H.pf[(size_t) (hs - S)].x != x;
                if (bad && ++chk_bad <= 5)
                    std::fprintf(stderr, "strata-q36 table check: state %zu expert %d: host slot %d, device slot %d\n", li, x, hs,
                                 d[(size_t) x]);
            }
            // the bytes: a few resident slots per layer (a window that moves every forward) and every prefetch slot in the table
            // hold their expert's blob
            const Layer& L = layer_at((int) li);
            std::vector<int> sample;
            for (int k = 0; k < 4 && S > 0; ++k) sample.push_back((int) ((chk_calls * 4 + k + (long long) li * 7) % S));
            for (size_t p = 0; p < H.pf.size(); ++p)
                if (H.pf[p].state == 2) sample.push_back(S + (int) p);
            if (lend_active)   // (the lent slots too)
                for (int k = 0; k < 3 && L.lent_n > 0; ++k) sample.push_back(L.n_reg + (int) ((chk_calls * 3 + k) % L.lent_n));
            for (int s_ : sample) {
                const bool pfs = s_ >= S && s_ < L.n_reg;
                const int x = pfs ? H.pf[(size_t) (s_ - S)].x : H.exp_in_slot[(size_t) s_];
                if (x < 0 || H.slot_of[(size_t) x] != s_ || (!pfs && H.pend_s[(size_t) s_])) continue;
                bytes_buf.resize(L.L.bytes);
                CK(cudaMemcpy(bytes_buf.data(), L.slot_addr(s_), L.L.bytes, cudaMemcpyDeviceToHost));
                ++chk_bytes;
                if (std::memcmp(bytes_buf.data(), L.hblobs + (size_t) x * L.stride, L.L.bytes) != 0 && ++chk_bytes_bad <= 5)
                    std::fprintf(stderr, "strata-q36 table check: state %zu slot %d expert %d: the bytes differ from the host copy\n", li, s_, x);
            }
        }
    }
    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps, int kind = 0) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const auto rb_t0 = Clock::now();
        const long long rb_s0 = hy_swaps;
        const bool as = async_swaps && ((async_mask >> kind) & 1);
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        // the history window (#3641): the budget's first layer rotates, and the layers past it still halve their counts
        const size_t start = cache_window ? rebalance_cursor : 0;
        size_t next = start;
        for (size_t pass = 0; pass < hy.size(); ++pass) {
            if (!cache_window && left <= 0) break;
            const size_t li = (start + pass) % hy.size();
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            if (left <= 0) {
                // (the budget is spent: the counts still halve)
                for (double& v : H.score) v *= 0.5;
                continue;
            }
            const int before = left;
            const int Pl = (int) H.pf.size();                         // (slots S .. S + Pl - 1: prefetch slots)
            const int NL = lend_active ? L.lent_n : 0;                // --lend-staging: slots S + Pl .. : lent, while the staging is free
            const int Scap = S + NL;
            // (with lent slots the profile's rank breaks the ties of the counts: an expert the session has not routed yet)
            const bool prior = lend_on && lend_prior && !H.prank.empty();
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            if (prior)
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    const double sa = H.score[(size_t) a], sb = H.score[(size_t) b];
                    return sa != sb ? sa > sb : H.prank[(size_t) a] < H.prank[(size_t) b];
                });
            else
                std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top Scap, lowest score first: the slots to give away (an empty lent slot first of all)
            auto sc_of = [&](int slot) { const int e = H.exp_in_slot[(size_t) slot]; return e < 0 ? -1.0 : H.score[(size_t) e]; };
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            for (int j = 0; j < NL; ++j) victims.push_back(S + Pl + j);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) { return sc_of(a) < sc_of(b); });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < Scap && left > 0; ++i) {
                const int x = order[(size_t) i];
                const int sx = H.slot_of[(size_t) x];
                if (sx >= 0 && (sx < S || sx >= S + Pl)) continue;
                if (H.pend_x[(size_t) x]) continue;   // its copy is on the way (--async-swaps)
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    if (H.pend_s[(size_t) s]) { ++vi; continue; }   // a copy is on the way into it
                    const int y = H.exp_in_slot[(size_t) s];
                    const double ys = sc_of(s);
                    // (an empty slot takes x if the session routed it, or the profile ranks it)
                    if (y < 0 ? !(H.score[(size_t) x] > 0 || prior) : ys >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    if (y >= 0) H.slot_of[(size_t) y] = -1;
                    const bool lent_dst = s >= S + Pl;
                    if (as || (lent_dst && !lend_settle)) {   // x enters the tables when its copy has landed (swap_settle); y is out of them now
                        H.exp_in_slot[(size_t) s] = x;
                        H.pend_s[(size_t) s] = 1;
                        H.pend_x[(size_t) x] = 1;
                        sw_new.push_back({(int) li, s, x});
                        --left;
                        ++hy_swaps;
                        if (lent_dst) ++lend_swaps;
                        changed = true;
                        break;
                    }
                    if (H.slot_of[(size_t) x] >= S && H.slot_of[(size_t) x] < S + Pl) {   // promoted from a prefetch slot: that slot is free again
                        H.pf[(size_t) (H.slot_of[(size_t) x] - S)].state = 0;
                        H.pf[(size_t) (H.slot_of[(size_t) x] - S)].x = -1;
                    }
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    CK(cudaMemcpyAsync(L.slot_addr(s), L.hblobs + (size_t) x * L.stride, L.L.bytes, cudaMemcpyHostToDevice, st));
                    if (lent_dst) ++lend_swaps;
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed) {
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
                H.evicted.clear();   // the whole table went up
            }
            for (double& v : H.score) v *= 0.5;
            if (cache_window && left < before) next = (li + 1) % hy.size();
        }
        if (cache_window) rebalance_cursor = next;
        // (the tables go up from pageable host memory: the copy is staged before the call returns, so the host may change them
        // at once; the synchronous swaps' copies are on the stream, the async ones wait for the table upload through a fence)
        swap_issue();   // (the async swaps' and the lent slots' copies)
        if (!as) {
            CK(cudaStreamSynchronize(st));
        }
        ++rb_calls[kind];
        rb_swaps[kind] += hy_swaps - rb_s0;
        rb_ms[kind] += ms_since(rb_t0);
    }

    void shrink_att(size_t floats) {
        if (fused_pattn) return;   // (no score buffers)
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }
    void reset_graphs() {
        for (GSet& gset : gsets) {
            for (cudaGraphExec_t* g : {&gset.g1, &gset.gm1})
                if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(cudaGraphExecDestroy(kv.second));
        row_graphs.clear();
    }

    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        const size_t need = rows.size();
        if (need > img_cap) {
            CK(cudaDeviceSynchronize());
            if (d_img) CK(cudaFree(d_img));
            CK(cudaMalloc((void**) &d_img, need * 4));
            img_cap = need;
        }
        CK(cudaMemcpy(d_img, rows.data(), need * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    void clear_images() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        mtab_now = nullptr;
        map_on = false;
        set_delta(0);
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        mtab_now = nullptr;
        map_on = false;
        if (!img_active) return;
        if (c0 < img_prompt_cells) mtab_now = d_mtab;
        bool any = false;
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            const int32_t v = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            h_map[j] = v;
            any = any || v >= 0;
        }
        if (any) {
            CK(cudaMemcpyAsync(d_map, h_map, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            map_on = true;
        }
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  on_chunk(done) runs
    /// after every chunk; false ends the read there (a cancelled request): the tokens processed are returned.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_) {
        const int N = (int) m->g.n_embd;
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            CK(cudaStreamSynchronize(st));   // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (mtab_now || map_on) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, [&] { forward(1, pos, 1, use_graph, false); });
            mtab_now = nullptr;
            map_on = false;
            rows_deferred = false;
            CK(cudaMemcpyAsync(hlast, hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            mtab_now = nullptr;
            map_on = false;
            if (!on_chunk(i + T)) {
                if (m->hybrid) after_prompt(n);
                return i + T;
            }
        }
        if (m->hybrid) after_prompt(n);
        return n;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, cudaGraphExec_t> row_graphs;   // by (rows, groups, kind): graphs of the batch windows
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    int32_t rows_pick[kVerify] = {};
    int rows_G_last = 0;   // groups of the window run last (commit_rows takes their keep counts)

    void run_graph_key(int key, const std::function<void()>& body) {
        if (!use_graph) { body(); return; }
        graph_run(row_graphs[key], body);
    }

    /// One window over the groups' rows.  `verify`: a verify window - the GDN state is read, not written (commit_rows writes
    /// it for the kept rows) and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every
    /// row's logits in `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        CK(cudaMemcpyAsync(tok, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hrow + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hrow + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % kRebalanceEvery == 0) rebalance(cache_window ? cache_swaps : decode_swaps);
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows (the conv history and the
    /// recurrence run again over the rows' kept inputs, as commit() does) and the session advances by keep[g].
    void commit_rows(const Group* gr, const int* keep, int G) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (hrow's slot / first-row words are the window's, unchanged)
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                             d_conv_tab + l * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm,
                                                                m->eps, nullptr, HK, HV, 1, true, d_S_tab + l * n_sess,
                                                                d_gslot, d_grow0, d_gcnt);
            }
        });
        rows_mode = false;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }

    /// The MTP block over rows of several sessions: row r reads token tok at cell `cell` of session `slot` and the seed
    /// hidden at `seed` (device, n_embd floats); its draft - the block's greedy pick after that row - ends up in
    /// mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(tok_m, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        run_graph_key(3000 + T, [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
            k_rms_ld<<<T, kThreads, 0, st>>>(mtp_seed(), N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(m->mtp, m->layers.size(), T, pmax, pos0_m, use_graph, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, st>>>(logits_mr, (int) g.n_vocab, pick_mr, amv, ami, false);
            k_argmax<<<dim3(1, T), kThreads, 0, st>>>(logits_mr, kArgBlocks, pick_mr, amv, ami, true);
        });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t cb = kv_cell(), row = (size_t) ctx * cb, cbv = kv_cell_v(), rowv = (size_t) ctx * cbv;
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, (size_t) a.pos * cb, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], rowv, a.vc[l], rowv, (size_t) a.pos * cbv, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        drop_snaps(b);
        if (dst == cur) load_cur();
    }
    int hybrid_swaps_after_prompt = 2048;
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown: a conversation that continues from them (the server's
    /// hand-over of a stopped request to a batch slot) reuses it instead of reading them again.
    /// `sp` (--mtp-sampled, a sampled request): `draft` replaces the block's greedy pick of draft d by its own choice from
    /// the block's logits (host, the first n token ids; null keeps the greedy pick), `judge` replaces the compare with the
    /// window's greedy picks: it reads the window's rows (host_logits_row) and returns a, writing picks[0 .. a].
    /// `draft` also gives the probability of its choice (what --spec-min-p gates on, as the main path's sampled drafts).
    struct SpecSampled {
        std::function<int32_t(int d, const int32_t* prev, const float* lg, int n, float* prob)> draft;
        std::function<int(int D, const std::vector<int32_t>& drafts, int32_t* picks)> judge;
        // --sample-gpu: the same over shortlists.  kdraft(d, prev) / kwindow(drafts): the K of draft d's / the window's
        // shortlists (0: the row comes whole); draft_c, judge_c: draft and judge from the shortlists (cand_row)
        std::function<int(int d, const int32_t* prev)> kdraft;
        std::function<int(const std::vector<int32_t>& drafts)> kwindow;
        std::function<int32_t(int d, const int32_t* prev, sampling::Cand& cd, size_t n_row, float* prob)> draft_c;
        std::function<int(int D, const std::vector<int32_t>& drafts, int32_t* picks)> judge_c;
    };
    std::vector<float> lg_draft;   // (the block's logits row on the host, for SpecSampled::draft)
    /// the token ids the draft head scores (head_draft's subset, or the whole vocabulary)
    int draft_cols() const {
        const auto& g = m->g;
        const Mat& hd = m->head;
        const bool sub = fast_k && draft_n > 0 && draft_n < g.n_vocab && draft_n % 8 == 0 && hd.type == 8 &&
                         hd.n_in == g.n_embd && g.n_embd / 32 == 64;
        return sub ? draft_n : (int) g.n_vocab;
    }
    /// draft d after its MTP call (pick_m and, with --spec-min-p, prob_m on the device): the block's pick, or sp's
    /// choice; *prob its probability (1 without the gate)
    int32_t take_draft(const SpecSampled* sp, int d, const std::vector<int32_t>& drafts, float* prob) {
        if (sample_gpu && sp != nullptr && sp->draft_c) {   // the shortlist of the block's row, not 1 MB of it
            const int K = sp->kdraft(d, drafts.data());
            if (K > 0) {
                const auto t0 = Clock::now();
                const int n = draft_cols();
                cand_fetch(logits_m, n, n, 1, K);
                sampling::Cand cd;
                cand_row(0, cd);
                const int32_t r = sp->draft_c(d, drafts.data(), cd, (size_t) n, prob);
                if (smp_prof) smp_ms += ms_since(t0);
                return r;
            }
        }
        CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
        if (spec_min_p > 0.f) CK(cudaMemcpyAsync(hpin + 12, prob_m, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        *prob = 1.f;
        if (spec_min_p > 0.f) std::memcpy(prob, hpin + 12, 4);
        if (sp == nullptr || !sp->draft) return hpin[2];
        const auto t0 = Clock::now();
        lg_draft.resize((size_t) draft_cols());
        CK(cudaMemcpyAsync(lg_draft.data(), logits_m, lg_draft.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int32_t r = sp->draft(d, drafts.data(), lg_draft.data(), (int) lg_draft.size(), prob);
        if (smp_prof) smp_ms += ms_since(t0);
        return r;
    }
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr, const SpecSampled* sp = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        // --spec-min-p (the main path's rule, src/program/generate.cpp): draft d joins the window only if the block gave it
        // at least spec_min_p, and the first one below ends the drafting - a step of fewer rows (0 drafts: a decode step)
        float pr = 1.f;
        if (D > 0) {
            drafts[0] = take_draft(sp, 0, drafts, &pr);
            if (pr < spec_min_p) D = 0;
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            if (mtp_graph) graph_run(gs().gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            else forward_mtp(1, pos + d, seeds, true, false);
            drafts[(size_t) d] = take_draft(sp, d, drafts, &pr);
            if (pr < spec_min_p) D = d;   // (ends the loop)
        }
        drafts.resize((size_t) D);
        last_d = D;
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        rows_deferred = true;
        graph_run(gs().gv[T], [&] { forward(T, pos, T, use_graph, true); });
        rows_deferred = false;
        // --sample-gpu: the window's rows as shortlists, copied with the picks (one sync)
        const int Kw = sample_gpu && sp != nullptr && sp->judge_c ? sp->kwindow(drafts) : 0;
        if (Kw > 0) cand_launch(logits, (int) m->g.n_vocab, (int) m->g.n_vocab, T, Kw);
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        if (Kw > 0) {
            const auto t0 = Clock::now();
            if (smp_check) cand_check(logits, (int) m->g.n_vocab, (int) m->g.n_vocab, T, Kw);
            a = sp->judge_c(D, drafts, picks);
            if (smp_prof) smp_ms += ms_since(t0);
        } else if (sp != nullptr && sp->judge) {
            const auto t0 = Clock::now();
            a = sp->judge(D, drafts, picks);
            if (smp_prof) smp_ms += ms_since(t0);
        } else {
            while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
            for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        }
        if (cancel != nullptr && cancel->load()) a = 0;
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        std::vector<int32_t> toks((size_t) tmax, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        if (tmax > kGemmT + 1) process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) {
            std::fill(H.score.begin(), H.score.end(), 0.0);
            std::fill(H.heat.begin(), H.heat.end(), 0.0);
        }
        hy_cpu = hy_gpu = hy_swaps = 0;
        rb_calls[0] = rb_calls[1] = rb_swaps[0] = rb_swaps[1] = 0;
        rb_ms[0] = rb_ms[1] = 0;
        hy_cpu_ms = hy_plan_ms = 0;
        hy_layers = 0;
#if defined(STRATA_Q36_CPU_POOL)
        if (pool) pool->reset_phase_stats();
#endif
        pf_preds = pf_copies = pf_hits = pf_late = 0;
        lend_swaps = lend_evicts = lend_fills = 0;
        lend_evict_ms = 0;
        reset();
        if (lend_on) {   // the warm-up's prompt took the lent slots out: no session history yet
            if (lend_prior) lend_fill_profile();
            else lend_active = true;
        }
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    // ---- --sample-gpu: shortlists (k_cand_part, k_cand_merge)
    /// the shortlists of `rows` rows (stride ld, n wide) on the device and on their way to h_cand (no sync)
    cudaEvent_t ce0 = nullptr, ce1 = nullptr;
    double smp_kernel_ms = 0;   // STRATA_Q36_SAMPLE_PROFILE: the shortlist kernel's own time (plain and draft rows)
    void cand_launch(const float* lg, int ld, int n, int rows, int K) {
        if (!d_cand) {   // (only with --sample-gpu: the buffers are not made otherwise)
            d_cand = dalloc_t<Cand2>((size_t) (kVerify + 1) * sampling::kCandMax);
            d_part = dalloc_t<Cand2>((size_t) (kVerify + 1) * kCandBlocks * sampling::kCandMax);
            CK(cudaHostAlloc((void**) &h_cand, (size_t) (kVerify + 1) * sampling::kCandMax * sizeof(Cand2), cudaHostAllocDefault));
        }
        if (smp_prof) {
            if (!ce0) { CK(cudaEventCreate(&ce0)); CK(cudaEventCreate(&ce1)); }
            CK(cudaEventRecord(ce0, st));
        }
        k_cand_part<<<dim3(kCandBlocks, rows), 1024, 0, st>>>(lg, ld, n, K, d_part);
        k_cand_merge<<<rows, 1024, 0, st>>>(d_part, K, d_cand);
        if (smp_prof) CK(cudaEventRecord(ce1, st));
        CK(cudaMemcpyAsync(h_cand, d_cand, (size_t) rows * K * sizeof(Cand2), cudaMemcpyDeviceToHost, st));
        cand_K = K;
    }
    /// STRATA_Q36_SAMPLE_CHECK: the shortlists against the whole rows by the tie rule (set of ids and their logits)
    void cand_check(const float* lg, int ld, int n, int rows, int K) {
        std::vector<float> row((size_t) n);
        std::vector<int> o((size_t) n);
        for (int r = 0; r < rows; ++r) {
            CK(cudaMemcpy(row.data(), lg + (size_t) r * ld, (size_t) n * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) o[(size_t) i] = i;
            std::partial_sort(o.begin(), o.begin() + K, o.end(), [&](int a, int b) { return sampling::before(row[(size_t) a], a, row[(size_t) b], b); });
            std::vector<int> want(o.begin(), o.begin() + K), got;
            bool ok = true;
            for (int j = 0; j < K; ++j) {
                const Cand2& c = h_cand[(size_t) r * K + j];
                got.push_back(c.id);
                ok = ok && c.id >= 0 && c.id < n && row[(size_t) c.id] == c.v;
            }
            std::sort(want.begin(), want.end());
            std::sort(got.begin(), got.end());
            ++smp_chk_n;
            if (!ok || want != got) {
                ++smp_chk_bad;
                if (smp_chk_bad <= 3) std::fprintf(stderr, "strata-q36 SAMPLE CHECK MISMATCH: row %d of %d, K %d, n %d\n", r, rows, K, n);
            }
        }
    }
    void cand_row(int r, sampling::Cand& cd) {
        cd.id.resize((size_t) cand_K);
        cd.v.resize((size_t) cand_K);
        for (int j = 0; j < cand_K; ++j) {
            const Cand2& c = h_cand[(size_t) r * cand_K + j];
            cd.id[(size_t) j] = c.id;
            cd.v[(size_t) j] = c.v;
        }
    }
    /// STRATA_Q36_SAMPLE_CHECK at start: the kernels on synthetic rows full of exact ties (coarse values, +-0) against the
    /// host's tie rule, for several K
    void cand_selftest() {
        const int n = (int) m->g.n_vocab, rows = 4;
        std::vector<float> h((size_t) rows * n);
        std::mt19937_64 r(11);
        for (int row = 0; row < rows; ++row)
            for (int i = 0; i < n; ++i) {
                float v = (float) (r() % 4096) / 512.f - 4.f;   // 4096 distinct values over 248K entries
                if (row == 1) v = std::round(v);                // only ~9 distinct values: ties across every boundary
                if (row == 2) v = (i % 3) ? 0.f : -0.f;         // all equal (+-0)
                if (row == 3 && i % 1000 == 0) v = 50.f + (float) (i % 7);
                h[(size_t) row * n + i] = v;
            }
        float* d = dalloc((size_t) rows * n);
        CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        const long long bad0 = smp_chk_bad;
        for (int K : {1, 20, 21, 70, 135, sampling::kCandMax}) {
            cand_launch(d, n, n, rows, K);
            CK(cudaStreamSynchronize(st));
            cand_check(d, n, n, rows, K);
        }
        // narrow rows: widths 32 K + r where the last (shorter) slice still holds K (the rows' start is the same data)
        for (int K : {1, 20, 91, 135, sampling::kCandMax})
            for (int r : {0, 1, 3, 17, 100, 333, 700, 1000, 1500, 3000}) {
                const int w = 32 * K + r;
                if (!sampling::cand_rows_ok((size_t) w, K)) continue;
                cand_launch(d, n, w, rows, K);
                CK(cudaStreamSynchronize(st));
                cand_check(d, n, w, rows, K);
            }
        CK(cudaFree(d));
        std::fprintf(stderr, "strata-q36: sample selftest: %lld shortlists on tie-heavy rows, %lld mismatches\n", smp_chk_n,
                     smp_chk_bad - bad0);
        smp_chk_n = smp_chk_bad = 0;
    }
    /// shortlists of the rows on the device, waited for
    void cand_fetch(const float* lg, int ld, int n, int rows, int K) {
        cand_launch(lg, ld, n, rows, K);
        CK(cudaStreamSynchronize(st));
        if (smp_prof) { float ms = 0; CK(cudaEventElapsedTime(&ms, ce0, ce1)); smp_kernel_ms += ms; }
        if (smp_check) cand_check(lg, ld, n, rows, K);
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// STRATA_Q36_VRAM_WATCH=1 (tests): a thread polls the device's free VRAM every few ms; each new low-water mark is said, and the
/// lowest at exit.  It is the number the expert placement's headroom is sized from (the whole device, other programs included).
struct VramWatch {
    std::atomic<bool> stop{false};
    std::atomic<size_t> low{(size_t) -1};
    std::thread th;
    VramWatch() {
        th = std::thread([this] {
            const auto t0 = Clock::now();
            size_t said = (size_t) -1;
            while (!stop.load()) {
                size_t fr = 0, tot = 0;
                if (cudaMemGetInfo(&fr, &tot) == cudaSuccess && fr < low.load()) {
                    low = fr;
                    if (said == (size_t) -1 || fr + ((size_t) 8 << 20) <= said) {   // (a new low by 8 MiB or more)
                        std::fprintf(stderr, "strata-q36: VRAM watch: %.3f GiB free at %.1f s\n", (double) fr / (1 << 30),
                                     ms_since(t0) / 1000.0);
                        said = fr;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    ~VramWatch() {
        stop = true;
        th.join();
        std::fprintf(stderr, "strata-q36: VRAM watch: low-water mark %.3f GiB free\n", (double) low.load() / (1 << 30));
    }
};

/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;
    for (Layer& L : m.layers) all.push_back(&L);
    if (m.has_mtp) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, max_layer = 0;
    for (Layer* L : all) {
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) 768 << 20;
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the card's
    // room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same outputs, a little
    // slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.  STRATA_Q36_KEEP_PACKED=1
    // keeps them (and the hybrid mode).
    size_t packed_bytes = 0;
    for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
    size_t main_need = 0;   // the main layers' experts (the MTP block's can go to the CPU, see below)
    for (Layer& L : m.layers) main_need += L.stride * (size_t) ne + kMmqTail;
    const size_t att_bufs = E.fused_pattn ? 0 : (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // (--fused-prompt-attn: none)
    const size_t slack = ((size_t) 512 << 20) + att_bufs;   // reserve + scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 &&
        (total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack)) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(cudaDeviceSynchronize());
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(cudaFree(std::get<1>(pk)));
        }
        m.packed.clear();
        E.reset_graphs();
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                     "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                     (double) packed_bytes / (1 << 30));
    }
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more, smaller
    // slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (Layer& L : m.layers) total_main += L.stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr) {
        const size_t att_save = att_bufs;   // att_s (4 B) + att_p16 (2 B)
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > small_reserve ? fr - small_reserve : 0;
            mtp_only = total_main + ((size_t) 64 << 20) <= budget2;
            std::fprintf(stderr, "strata-q36: the experts nearly fit: the attention scores buffer at 64 MB and the VRAM "
                         "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                         "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                         "budget instead\n");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(cudaMalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "strata-q36: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = false;                      // the MTP block's layer is hybrid either way
    E.hybrid_t = std::max(1, std::min(hybrid_t, E.use_mmq ? E.tmax : E.tmax));
    if (!E.use_mmq) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    const size_t staging = E.use_mmq && E.main_hybrid ? 2 * (max_layer + kMmqTail) : 0;
    // the CPU half (moe_hybrid_cpu) sees a batch of up to hybrid_t rows when the main layers are hybrid, else (only the MTP
    // block is) every batch up to tmax; its device rows and the host staging are sized by that, not by tmax
    const size_t cpu_rows = (size_t) (E.main_hybrid ? E.hybrid_t : E.tmax);
    const size_t cpu_dev = cpu_rows * (size_t) m.top_k * ((size_t) g.n_embd + 1) * 4;   // ecpu + d_miss
    // the reserve holds the CPU half's device rows of tmax rows (128 MiB at --prefill 2048); they are counted here by the rows
    // that can reach them (cpu_dev), and the difference is given back to the slots: the same headroom, a few more slots.  Only the
    // reserve-derived budget has that room to give back: an explicit --expert-vram budget names the experts' VRAM itself and keeps
    // its old slot pool (cpu_dev_b is 0 there)
    const size_t cpu_credit = vram_gib < 0 ? (size_t) E.tmax * m.top_k * ((size_t) g.n_embd + 1) * 4 : 0;
    const size_t cpu_dev_b = vram_gib < 0 ? cpu_dev : 0;
    // the hybrid mode's slots take the room down to a smaller headroom than the 768 MiB reserve (which the all-in-VRAM decision above keeps):
    // 256 MiB stay free at the low-water mark of the free VRAM (for the driver, the desktop and the next request's buffers), the engine's lazy
    // growth after placement is 21 MiB (measured over a 245K prompt read + MTP decode, a 198K read with two follow-ups, lend-staging's evict
    // and refill, and the first request: STRATA_Q36_VRAM_WATCH=1) and place_experts itself allocates about 95 MiB after reading the free VRAM
    size_t headroom = kHybridHeadroom;   // STRATA_Q36_HEADROOM_MIB=768 gives the old reserve back (a desktop that runs out of VRAM)
    if (const char* hm = std::getenv("STRATA_Q36_HEADROOM_MIB")) headroom = (size_t) std::max(0, std::atoi(hm)) << 20;
    const size_t hy_reserve = headroom + kLazyGrowth + kPlacedAfter + cpu_credit;   // (cpu_credit: the rows' room, added back below)
    const size_t hy_extra = (vram_gib < 0 && !mtp_only && fr > reserve && budget2 == fr - reserve && reserve > hy_reserve) ? reserve - hy_reserve : 0;
    if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
        std::fprintf(stderr, "strata-q36: place: free %.1f MiB, budget %.1f MiB (+%.1f MiB hybrid headroom), staging %.1f, cpu_dev %.1f, credit %.1f, per_slot %.2f MiB\n",
                     fr / 1048576.0, budget2 / 1048576.0, hy_extra / 1048576.0, staging / 1048576.0, cpu_dev / 1048576.0, cpu_credit / 1048576.0, per_slot / 1048576.0);
    size_t slot_bytes = budget2 + hy_extra + cpu_credit > staging + cpu_dev_b ? budget2 + hy_extra + cpu_credit - staging - cpu_dev_b : 0;
    if (slot_bytes + staging + ((size_t) 256 << 20) > fr) slot_bytes = fr > staging + ((size_t) 512 << 20) ? fr - staging - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 + cpu_credit > total_main + cpu_dev_b ? budget2 + cpu_credit - total_main - cpu_dev_b : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    // --expert-prefetch P: P of each main hybrid layer's slots become prefetch slots (the same VRAM; the MTP block
    // keeps S, without them); at least one slot stays a cache slot
    int P = 0;
    if (E.pf_want > 0 && !mtp_only && E.main_hybrid)
        P = std::max(0, std::min(E.pf_want, std::min(S - 1, 8)));   // (PfPub holds 2 P changes)
    S -= P;
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        size_t need = staging + cpu_dev;
        for (Layer* L : all) {
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else {
                const int n = L == &m.mtp ? S : S + P;
                need += n > 0 ? (size_t) n * L->stride + kMmqTail : 0;
            }
        }
        if (need > fr) {
            err = "not enough VRAM left for the hybrid experts' buffers and slots (" + std::to_string((need >> 20)) +
                  " MiB needed, " + std::to_string(fr >> 20) + " MiB free after the weights, the context and the sessions): use "
                  "fewer slots (\"parallel\" / --batch), a shorter --max-context" +
                  (slots > 0 ? ", or fewer --expert-slots" : "");
            return false;
        }
    }
    // host memory: every expert, pinned where the driver allows it (the prompt's stream and the slot copies)
    size_t host = 0;
    for (Layer* L : all)
        if (!mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    uint8_t* arena = nullptr;
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        arena = (uint8_t*) std::malloc(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    E.arena_pinned = pinned;
    if ((E.async_swaps || E.pf_check) && !E.main_hybrid && E.use_graph) {   // (the decode graphs would capture the host-side settle)
        std::fprintf(stderr, "strata-q36: --async-swaps / STRATA_Q36_PF_CHECK need the main layers hybrid (decode runs as graphs here); off\n");
        E.async_swaps = false;
        E.async_mask = 0;
        E.pf_check = false;
    }
    if (E.async_swaps && !pinned) {
        std::fprintf(stderr, "strata-q36: --async-swaps needs the pinned expert copy; off\n");
        E.async_swaps = false;
    }
    if (P > 0 && !pinned) {   // the prefetch copies need the pinned host copy: its slots stay cache slots
        std::fprintf(stderr, "strata-q36: --expert-prefetch needs the pinned expert copy; off\n");
        S += P;
        P = 0;
    }
    size_t off = 0;
    E.hy.resize(all.size());
    // --expert-profile (opt-in): each layer's first slots get its most-routed experts from a saved profile instead of
    // experts 0 .. S - 1; a profile that cannot be read is said, and the default order is used
    std::vector<std::vector<int>> prof;
    if (!E.expert_profile.empty()) {
        std::string perr;
        int64_t pslots = 0;
        if (strata::core::read_expert_profile(E.expert_profile, (int64_t) all.size(), ne, E.profile_ranked, pslots, perr)) {
            prof.assign(all.size(), {});
            std::vector<uint8_t> seen(all.size() * (size_t) ne, 0);   // (a pair listed twice counts once)
            for (const auto& [l, e] : E.profile_ranked)
                if (!seen[(size_t) l * ne + e]++) prof[(size_t) l].push_back((int) e);
            std::fprintf(stderr, "strata-q36: expert profile %s: the initial slots from it\n", E.expert_profile.c_str());
        } else {
            std::fprintf(stderr, "strata-q36: expert profile not used: %s\n", perr.c_str());
            E.profile_ranked.clear();
        }
    }
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
        for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        L.n_slots = S;
        // the initial residents: experts 0 .. S - 1, or the profile's ranking for this layer (then the rest by index)
        std::vector<int> first((size_t) ne);
        for (int64_t e = 0; e < ne; ++e) first[(size_t) e] = (int) e;
        if (!prof.empty() && !prof[li].empty()) {
            std::vector<uint8_t> taken((size_t) ne, 0);
            size_t k = 0;
            for (int e : prof[li]) { first[k++] = e; taken[(size_t) e] = 1; }
            for (int64_t e = 0; e < ne; ++e)
                if (!taken[(size_t) e]) first[k++] = (int) e;
        }
        const int Pl = &L == &m.mtp ? 0 : P;   // (prefetch slots: the main layers only)
        if (S + Pl > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) (S + Pl) + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) (S + Pl) + kMmqTail));
            if (prof.empty()) CK(cudaMemcpy(L.blobs, L.hblobs, L.stride * (size_t) S, cudaMemcpyHostToDevice));
            else
                for (int s = 0; s < S; ++s)
                    CK(cudaMemcpy(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) first[(size_t) s] * L.stride, L.stride,
                                  cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        if (E.lend_on) {   // the profile's order (else 0 .. ne - 1) and each expert's place in it
            H.prof_order = first;
            H.prank.assign((size_t) ne, 0);
            for (int64_t k = 0; k < ne; ++k) H.prank[(size_t) first[(size_t) k]] = (int) k;
        }
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) first[(size_t) s]] = s;
            H.exp_in_slot[(size_t) s] = first[(size_t) s];
        }
        H.score.assign((size_t) ne, 0.0);
        H.pend_s.assign((size_t) std::max(S, 0), 0);
        H.pend_x.assign((size_t) ne, 0);
        H.heat.assign((size_t) ne, 0.0);
        H.pf.assign((size_t) Pl, Engine::Hy::PfSlot{});
        if (Pl > 0) {
            CK(cudaMalloc(&H.flag_d, (size_t) Pl * 4));
            CK(cudaMemset(H.flag_d, 0, (size_t) Pl * 4));
            CK(cudaHostAlloc((void**) &H.out_h, (size_t) Pl * 4, cudaHostAllocMapped));
            std::memset(H.out_h, 0, (size_t) Pl * 4);
            CK(cudaHostGetDevicePointer((void**) &H.out_d, H.out_h, 0));
        }
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = cpu_rows, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;   // (rows moe_hybrid_cpu can see)
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(cudaMalloc(&E.ecpu, T * Kp * N * 4));
    CK(cudaMalloc(&E.d_miss, T * Kp * 4));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
    if (staging) {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        if (E.split_copy) {   // --split-stage-copy: the resident experts' copies (kernels, see stream_kick) on a stream of their own
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(__HIPCC__)   // (HIP: the default priority, as mtp.cpp)
            int prio_lo = 0, prio_hi = 0;   // at the highest priority
            if (cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi) != cudaSuccess) { cudaGetLastError(); prio_hi = 0; }   // (no priorities: a plain stream)
            CK(cudaStreamCreateWithPriority(&E.dst2, cudaStreamNonBlocking, prio_hi));
#else
            CK(cudaStreamCreateWithFlags(&E.dst2, cudaStreamNonBlocking));
#endif
            for (int b = 0; b < 2; ++b) CK(cudaEventCreateWithFlags(&E.ev_d2d[b], cudaEventDisableTiming));
        }
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&E.stage[b], max_layer + kMmqTail));
            CK(cudaMemset(E.stage[b], 0, max_layer + kMmqTail));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
    }
    if (E.lend_on && !(staging && E.main_hybrid)) {
        std::fprintf(stderr, "strata-q36: --lend-staging needs the hybrid mode with prompt staging; off\n");
        E.lend_on = false;
    }
    if (E.lend_on && E.n_sess > 1) {   // (not tested with several sessions)
        std::fprintf(stderr, "strata-q36: --lend-staging is off with --batch %d (the sessions' windows stream prompts between decodes)\n", E.n_sess);
        E.lend_on = false;
    }
    if (E.lend_on) {   // the staging buffers' room as extra slots: n per layer, a layer's in one buffer, from the front
        // (within max_layer: the kMmqTail zero bytes after the biggest layer's blobs stay as they were allocated)
        const size_t cap = max_layer;
        E.stage_max_layer = max_layer;
        int n = (int) ((2 * cap) / per_slot);
        std::vector<std::pair<int, size_t>> place;
        for (; n >= 1; --n) {
            place.clear();
            int b = 0;
            size_t off = 0;
            bool ok = true;
            for (Layer* L : all) {
                const size_t need = ((size_t) n * L->stride + 255) & ~(size_t) 255;
                if (off + need > cap) { ++b; off = 0; }
                if (b >= 2 || off + need > cap) { ok = false; break; }
                place.emplace_back(b, off);
                off += need;
            }
            if (ok) break;
        }
        if (n < 1) {
            std::fprintf(stderr, "strata-q36: --lend-staging: the staging buffers hold no whole slot per layer; off\n");
            E.lend_on = false;
        } else {
            for (size_t li = 0; li < all.size(); ++li) {
                Layer& L = *all[li];
                Engine::Hy& H = E.hy[li];
                const int Pl = (int) H.pf.size();
                L.lent = E.stage[place[li].first] + place[li].second;
                L.lent_n = n;
                L.n_reg = L.n_slots + Pl;
                H.exp_in_slot.resize((size_t) (L.n_reg + n), -1);
                H.pend_s.resize((size_t) (L.n_reg + n), 0);
            }
            CK(cudaEventCreateWithFlags(&E.ev_lend, cudaEventDisableTiming));
            std::fprintf(stderr, "strata-q36: lending the staging buffers: %d extra slots per layer during decode (%.2f GiB), %s\n", n,
                         (double) n * per_slot / (1 << 30), E.lend_prior ? "filled by the expert profile's order, then by the routing counts"
                                                                         : "filled by the routing counts only");
            if (E.lend_prior) E.lend_fill_profile();
            else E.lend_active = true;
        }
    }
    if (E.async_swaps || E.lend_on) {
        CK(cudaStreamCreateWithFlags(&E.swst, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&E.ev_swfence, cudaEventDisableTiming));
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    {   // --prefetch-copies, else the env alias, else the PCIe copies on
        const char* pc = std::getenv("STRATA_Q36_PREFETCH_COPIES");
        const int copies = E.pf_copies_want >= 0 ? E.pf_copies_want : pc ? std::atoi(pc) : 1;
        E.pf_no_copy = copies == 0;
    }
    const char* wk = std::getenv("STRATA_Q36_POOL_WARM");
    const int warm_k = E.pool_warm_want >= 0 ? E.pool_warm_want : wk ? std::atoi(wk) : 0;   // --pool-warm, else the env alias, else off
    if (warm_k > 0) {   // experts per layer the parked pool workers pre-load (needs the lookahead of --expert-prefetch)
        E.pool_warm_k = std::max(0, std::min(C::ExpertPool::kMaxWarm, warm_k));
        if (E.pool_warm_k > 0) E.pool->enable_warm(true);
        if (E.pool_warm_k > 0 && !E.pool->warm_enabled()) E.pool_warm_k = 0;
    }
    if (P > 0) {   // --expert-prefetch
        E.pf_slots = P;
        E.pf_topk = std::min<int>(E.pf_topk, (int) ne);   // (prefetch_issue's partial_sort)
        if (const char* v = std::getenv("STRATA_Q36_PF_GEN_START"))   // tests: start the copies' numbers near the wrap
            E.pf_gen = (int32_t) std::max<long long>(0, std::min<long long>(INT32_MAX - 1, std::atoll(v)));
        CK(cudaStreamCreateWithFlags(&E.pst, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&E.pfst, cudaStreamNonBlocking));
        for (cudaEvent_t* ev : {&E.ev_pf0, &E.ev_pred}) CK(cudaEventCreateWithFlags(ev, cudaEventDisableTiming));
        CK(cudaHostAlloc((void**) &E.h_pflog, (size_t) ne * 4 * 2 * Engine::kPfRows, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &E.d_pflog, E.h_pflog, 0));
        std::fprintf(stderr, "strata-q36: prefetch: %d of the %d slots per layer filled by pre-gating (top %d)\n", P,
                     S + P, E.pf_topk);
    }
    CK(cudaDeviceSynchronize());
    {   // (said: the headroom the reserve is meant to leave, after the slots, the staging pair and the CPU half's rows)
        size_t fr_after = 0, tot_after = 0;
        CK(cudaMemGetInfo(&fr_after, &tot_after));
        std::fprintf(stderr, "strata-q36: VRAM left after the experts' buffers: %.2f GiB (CPU half: %zu rows)\n",
                     (double) fr_after / (1 << 30), cpu_rows);
    }
    if (mtp_only)
        std::fprintf(stderr, "strata-q36: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "strata-q36: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB "
                 "staging); %.1f s\n", S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
// (q36_sampling.hpp: Sampling, sample(), and --mtp-sampled's SpecJudge)
using namespace strata::qwen35::sampling;

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a strata-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
            hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h;
        {   // what the conversation cache compares: a picture is its grid and its key (or its embeddings)
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof grid);
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (serve/server.py).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into that slot's
/// session - the decode windows of the other slots go on between its chunks - then its tokens come as BT lines, one
/// window for every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the request on the
/// control lines.
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;
    // decode windows of the other slots between two chunks of a prompt read: STRATA_Q36_READ_WINDOWS (default 1; a chunk
    // takes ~0.15 s at 512 tokens, a window ~0.012 s: more windows give the decoding slots more of the time a read takes)
    int read_windows = 1;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {
        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
    }

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.  The proposals for the next window come right after
    /// it, from the hidden states the window just produced (the first one of a slot from the state its prompt left).
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        const int N = (int) E.m->g.n_embd;
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        // who verifies a proposal this window: the greedy slots in turn, as many as fit beside one row per slot
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        // a candidate without a proposal gets its first one from the state its last row left
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) act[(size_t) i]];
                first.push_back({act[(size_t) i], sl.x, ss.pos, ss.hlast});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {act[(size_t) i], cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        // the picks: emit, decide how many rows each slot keeps
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                y0 = sample(lg, sl.smp, sl.hist, sl.rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new &&
                       ss.pos + 3 <= E.ctx;   // (ss.pos: the position of the slot's next token: the window did not move it)
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t fed[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                ss.live.push_back(fed[j]);
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "strata batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    ss.active = false;
                    --n_active;
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        // the hidden state after each slot's last kept row (the seed of a later first proposal), and the next proposals
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            CK(cudaMemcpyAsync(ss.hlast, E.hmain + (size_t) (r0 + keep[(size_t) i] - 1) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, E.st));
            Slot& sl = slots[(size_t) b];
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;   // commit_rows moved it: the cell of the token the next window feeds
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.hmain + (size_t) r0 * N});
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) (r0 + 1) * N});
                nx_last.push_back((int) nx.size() - 1);
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) r0 * N});
                nx_last.push_back((int) nx.size() - 1);
            }
            nx_of.push_back(i);
        }
        std::fflush(stdout);
        if (!nx.empty()) {
            // (more rows than the window holds: the proposals come in two calls)
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;   // a slot's rows together
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot: one token each.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) rows.push_back({b, slots[(size_t) b].x});
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        const size_t Nb = (size_t) E.m->g.n_embd * 4;
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = rows[r].slot;
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            if (E.bmtp)   // (the hidden state a later first proposal starts from)
                CK(cudaMemcpyAsync(ss.hlast, E.hmain + r * (size_t) E.m->g.n_embd, Nb, cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            ss.live.push_back(sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                y = sample(lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[r];
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    /// The length of session ss's held tokens when they are a proper start of `ids` with the same pictures, else 0.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        std::vector<ImgKey> in_live;
        for (const ImgKey& k : imgs) {
            if (k.cell + k.n <= (int64_t) L) in_live.push_back(k);
            else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
        }
        return in_live == ss.imgs ? L : 0;
    }

    /// The tokens of `ids` session ss can take without reading them: all it holds when that is a proper start
    /// of `ids` (snap -1), else the longest of its snapshots that is (snap: its index), else 0.
    size_t held_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs,
                    int& snap) const {
        snap = -1;
        const size_t L = match_len(ss, ids, imgs);
        if (L > 0) return L;
        size_t best = 0;
        for (size_t k = 0; k < ss.snaps.size(); ++k) {
            const Engine::Snap& sn = ss.snaps[k];
            const size_t l = (size_t) sn.len;
            if (l == 0 || l <= best || l >= ids.size() || l > ss.live.size() ||
                !std::equal(ss.live.begin(), ss.live.begin() + (long) l, ids.begin()))
                continue;
            std::vector<ImgKey> in_l;
            bool cut = false;
            for (const ImgKey& im : imgs) {
                if (im.cell + im.n <= (int64_t) l) in_l.push_back(im);
                else if (im.cell < (int64_t) l) cut = true;   // the snapshot ends inside a picture
            }
            if (cut || !(in_l == sn.imgs)) continue;
            best = l;
            snap = (int) k;
        }
        return best;
    }

    /// STRATA_REUSE_DIAG=1 (off by default: it logs a few of the conversation's token ids): a prompt that parts from
    /// the tokens a session holds well after their start, 64 or more of them read again (a conversation read again from
    /// the start, or from a snapshot), gets one log line with where, and the ids there (the server says it in text).
    void note_parting(const std::vector<int32_t>& ids, size_t reuse) const {
        static const bool on = [] {
            const char* v = std::getenv("STRATA_REUSE_DIAG");
            return v != nullptr && std::string(v) == "1";
        }();
        if (!on) return;
        size_t lcp = 0, held = 0;
        int at = -1;
        for (int i = 0; i < E.n_sess; ++i) {
            const std::vector<int32_t>& lv = E.sess[(size_t) i].live;
            const size_t n = std::min(lv.size(), ids.size());
            const size_t c = (size_t) (std::mismatch(lv.begin(), lv.begin() + (long) n, ids.begin()).first - lv.begin());
            if (c > lcp) { lcp = c; held = lv.size(); at = i; }
        }
        if (at < 0 || lcp < 1024 || reuse + 64 > lcp) return;
        const std::vector<int32_t>& lv = E.sess[(size_t) at].live;
        auto ids_at = [&](const std::vector<int32_t>& v) {
            std::string o;
            for (size_t j = lcp >= 3 ? lcp - 3 : 0; j < std::min(v.size(), lcp + 4); ++j)
                o += (j == lcp ? " |" : " ") + std::to_string(v[j]);
            return o;
        };
        std::fprintf(stderr, "strata serve: the prompt (%zu tokens) parts from the %zu tokens session %d holds at token %zu "
                     "(%zu reused, %zu read again): held%s, prompt%s\n", ids.size(), held, at, lcp, reuse, lcp - reuse,
                     ids_at(lv).c_str(), ids_at(ids).c_str());
    }

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            int k;
            const size_t l = held_len(ss, ids, imgs, k);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        std::vector<uint64_t> img_keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !E.vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { std::printf("ERR %s\n", ve.c_str()); std::fflush(stdout); return; }
        }
        stop.store(false);
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back)
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        Engine::Sess& ss = E.sess[(size_t) s];
        int src = s, snap = -1;   // snap >= 0: the state is src's snapshot snap
        size_t reuse = held_len(ss, ids, ir.imgs, snap);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                int k;
                const size_t l = held_len(E.sess[(size_t) i], ids, ir.imgs, k);
                if (l > best) { best = l; src = i; snap = k; }
            }
            reuse = best;
        }
        note_parting(ids, reuse);
        if (snap >= 0) E.restore(src, snap, s);
        else if (src != s) E.copy_session(src, s);
        E.select(s);
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
            Engine::drop_snaps(ss);
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (n_active > 0 && done < fresh) {
                poll_control();
                // the slots decoding go on between a read's chunks (the last chunk's
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
            Engine::drop_snaps(ss);
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        // --snapshots: the state this read left, for a later prompt that parts from what is generated next
        if (!cancelled && !ss.snaps.empty() && (int64_t) ids.size() >= E.snap_min) E.snapshot();
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        E.smp_ms = 0;
        E.smp_kernel_ms = 0;
        E.smp_calls = E.smp_chk_n = E.smp_chk_bad = 0;
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones too with --mtp-sampled (SpecJudge: the text is a sample of the target chain); other sampled
        // ones, and requests beside others, one token at a time
        SpecRule rule = SpecRule::off;
        if (smp.needs_logits() && E.mtp_sampled > 0) {
            // q needs a temperature, and spec_prob.hpp's lists hold 64 ids (the server's top_k is 1 .. 64);
            // STRATA_SPEC_MIN_TEMP (Flash-Next's gate): below it the block's greedy drafts (match)
            const bool q_ok = smp.temperature > 0.f && smp.top_k <= 64 && smp.temperature >= strata::core::spec_min_temp();
            rule = E.mtp_sampled == 2 && q_ok ? SpecRule::q : SpecRule::match;
        }
        const bool spec = E.mtp_d > 0 && (!smp.needs_logits() || rule != SpecRule::off) && !adm;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        SpecJudge sj{rule, smp, hist, rng};
        const size_t n_vocab = (size_t) E.m->g.n_vocab;
        // the next token of a sampled request from the logits on the device: the shortlist (--sample-gpu) or the whole row
        auto sample_next = [&]() -> int {
            const auto t0 = Clock::now();
            int t;
            const int K = E.sample_gpu ? cand_need(smp, hist, nullptr, 0, n_vocab) : 0;
            if (K > 0) {
                E.cand_fetch(E.logits, (int) n_vocab, (int) n_vocab, 1, K);
                Cand cd;
                E.cand_row(0, cd);
                t = sample_c(cd, n_vocab, smp, hist, rng);
            } else {
                E.host_logits(lg);
                t = sample(lg, smp, hist, rng);
            }
            if (E.smp_prof) { E.smp_ms += ms_since(t0); ++E.smp_calls; }
            return t;
        };
        Engine::SpecSampled ssp;
        if (rule == SpecRule::q)
            ssp.draft = [&](int d, const int32_t* prev, const float* row, int n, float* prob) {
                return sj.draft(d, prev, row, n, prob);
            };
        if (rule != SpecRule::off)
            ssp.judge = [&](int D, const std::vector<int32_t>& dr, int32_t* pk) {
                return sj.judge([&](int r, std::vector<float>& row) { E.host_logits_row(r, row); }, D, dr, pk);
            };
        if (E.sample_gpu && rule != SpecRule::off) {   // the same over shortlists (the engine falls back per row set: K == 0)
            ssp.kwindow = [&](const std::vector<int32_t>& dr) { return sj.window_k(dr, n_vocab); };
            ssp.judge_c = [&](int D, const std::vector<int32_t>& dr, int32_t* pk) {
                return sj.judge_c([&](int r, Cand& cd) { E.cand_row(r, cd); }, n_vocab, D, dr, pk);
            };
            if (rule == SpecRule::q) {
                ssp.kdraft = [&](int d, const int32_t* prev) { return sj.draft_k(d, prev, (size_t) E.draft_cols()); };
                ssp.draft_c = [&](int d, const int32_t* prev, Cand& cd, size_t n, float* prob) {
                    return sj.draft_c(d, prev, cd, n, prob);
                };
            }
        }
        bool decided = false;   // nx is a sampled speculative step's pick (the logits on the device are not its row's)
        if (adm) {
            if (smp.needs_logits()) nx = sample_next();
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata batch: slot %d takes a prompt of %zu tokens = %zu reused + %lld read in %.0f ms (%.1f "
                         "tok/s)%s\n", slot, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }
        for (;;) {
            if (smp.needs_logits() && !decided) nx = sample_next();
            decided = false;
            auto emit = [&](int t) {   // false: the answer ends here
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (spec && n_active == 0 && E.pos + E.mtp_d + 1 <= E.ctx) {
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop, rule != SpecRule::off ? &ssp : nullptr);
                offered += E.spec_min_p > 0.f ? E.last_d : E.mtp_d;   // (the gate's windows vary: the drafts verified)
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                for (int i = 0; i < a; ++i) ss.live.push_back(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                decided = rule != SpecRule::off;
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata serve: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse,
                     (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
        if (E.smp_prof)
            std::fprintf(stderr, "strata serve: session %d: sampler host time %.1f ms over %lld generated (%.0f us per token)%s\n", s,
                         E.smp_ms, produced, produced > 0 ? 1000.0 * E.smp_ms / (double) produced : 0.0,
                         E.sample_gpu ? " (--sample-gpu)" : "");
        if (E.smp_prof && E.sample_gpu)
            std::fprintf(stderr, "strata serve: session %d: shortlist kernel (plain and draft rows) %.2f ms in all\n", s, E.smp_kernel_ms);
        if (E.smp_check)
            std::fprintf(stderr, "strata serve: session %d: sample check: %lld shortlists, %lld mismatches\n", s, E.smp_chk_n, E.smp_chk_bad);
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E) {
    Serve sv(E);
    return sv.run();
}

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, emb_file;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    double expert_vram = -1.0;
    bool warm = true;
    int expert_slots = 0, batch = 0, draft_vocab = -1, mtp_sampled = 0;
    bool sample_gpu = false;
    float spec_min_p = 0.f;
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    // the opt-ins of this engine path (docs/QWEN36.md): off, they leave the default path as it was
    std::string kv = "fp32", tokens_file, expert_profile, expert_profile_save;
    bool gqa_attn = false, attn_mma = false, fused_pattn = false, chunked_gdn = false, side_stream = false, split_stage_copy = false, zero_copy = false, snapshot_host = false, async_swaps = false, async_after = false;
    int expert_prefetch = 0, decode_swaps = -1, snapshots = 0, pool_warm = -1, prefetch_copies = -1;
    int64_t snapshot_min = 4096;
    bool lend_staging = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-q36: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") tokens_file = next();   // strata's name: --tokens from a file (a long prompt)
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a strata-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-q36: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--mtp-sampled") {   // sampled requests draft too: match (the block's greedy drafts) or q (sampled drafts)
            const std::string v = next();
            mtp_sampled = v == "match" ? 1 : v == "q" ? 2 : v == "off" ? 0 : -1;
            if (mtp_sampled < 0) { std::fprintf(stderr, "strata-q36: --mtp-sampled takes off, match or q\n"); return 2; }
        }
        else if (a == "--sample-gpu") sample_gpu = true;       // sampled rows come to the host as GPU shortlists (K highest logits)
        else if (a == "--spec-min-p") spec_min_p = (float) std::atof(next().c_str());   // the main path's name: the gate
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--kv") kv = next();   // strata's names (checked below): fp16, int8, k8v4; fp32 this path's default
        else if (a == "--gqa-attn") gqa_attn = true;               // the grouped-query decode attention (fp32 / int8 cache)
        else if (a == "--attn-mma") attn_mma = true;               // with --kv int8: its tensor-core kernel (sm_80+, not bit-identical)
        else if (a == "--chunked-gdn") chunked_gdn = true;        // the prompt's gated delta rule in pieces of 16 tokens
        else if (a == "--side-stream") side_stream = true;         // z, alpha / beta, k, v of a decode layer on a second stream
        else if (a == "--split-stage-copy") split_stage_copy = true;   // the prompt's staging copy: resident experts from 1/8, own stream
        else if (a == "--fused-prompt-attn") fused_pattn = true;   // the prompt attention in one kernel (int8 cache)
        else if (a == "--hybrid-zero-copy") zero_copy = true;      // the hybrid hand-off without copies
        else if (a == "--expert-prefetch") expert_prefetch = std::atoi(next().c_str());   // P prefetch slots per layer
        else if (a == "--expert-profile") expert_profile = next();             // the main path's names: the initial slots
        else if (a == "--expert-profile-save") expert_profile_save = next();   // ... and what this run learned, at exit
        else if (a == "--pool-warm" || a == "--prefetch-copies") {   // K predicted experts per layer warmed in the pool workers; 0|1: the PCIe copies
            const std::string v = next();
            char* end = nullptr;
            const long x = std::strtol(v.c_str(), &end, 10);
            const long hi = a == "--pool-warm" ? 8 : 1;
            if (v.empty() || *end != 0 || x < 0 || x > hi) {
                std::fprintf(stderr, "strata-q36: %s takes a whole number from 0 to %ld, not \"%s\"\n", a.c_str(), hi, v.c_str());
                return 2;
            }
            (a == "--pool-warm" ? pool_warm : prefetch_copies) = (int) x;
        }
        else if (a == "--decode-swaps") decode_swaps = std::atoi(next().c_str());   // the hybrid slots' swaps per 32 tokens
        else if (a == "--async-swaps") async_swaps = true;   // the decode slot swaps' copies off the engine stream
        else if (a == "--lend-staging") lend_staging = true;   // the prompt staging buffers as extra expert slots in decode
        else if (a == "--async-after-prompt") async_after = true;   // ... and the after-prompt ones
        else if (a == "--snapshots") snapshots = std::atoi(next().c_str());       // recurrent-state snapshots per session
        else if (a == "--snapshot-min") snapshot_min = std::atoll(next().c_str());
        else if (a == "--snapshot-host") snapshot_host = true;   // the snapshots in pinned host memory: more expert slots
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph]\n"
                         "       (--serve | --tokens IDS | --tokens-file F [--n 64] [--force IDS] [--dump-full-logits F] | "
                         "--bench-pp N --bench-tg N)\n"
                         "       opt-ins (docs/QWEN36.md): [--kv fp32|fp16|int8|k8v4] [--gqa-attn] [--mtp-sampled off|match|q] [--attn-mma] [--fused-prompt-attn] [--chunked-gdn]\n"
                         "       [--side-stream] [--split-stage-copy] [--async-swaps] [--async-after-prompt] [--lend-staging] [--snapshots K] [--snapshot-min N] [--snapshot-host] [--expert-profile F] [--expert-profile-save F] [--spec-min-p P]\n"
                         "       [--expert-prefetch P] [--prefetch-copies 0|1] [--pool-warm K] [--hybrid-zero-copy] [--decode-swaps N] [--sample-gpu]\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
    if (kv != "fp32" && kv != "fp16" && kv != "int8" && kv != "k8v4") {
        // the main path's q4_0 (or any other name) is not a cache of this path: in serve mode said and skipped (the
        // fp32 cache), as an option this engine does not have; elsewhere refused
        if (!serve_mode) { std::fprintf(stderr, "strata-q36: --kv takes fp32, fp16, int8 or k8v4 on this engine path\n"); return 2; }
        std::fprintf(stderr, "strata-q36: ignoring --kv %s (this engine path has fp32, fp16, int8 and k8v4): fp32\n", kv.c_str());
        kv = "fp32";
    }
    const bool kv_v4 = kv == "k8v4", kv8 = kv == "int8" || kv_v4, kv_f16 = kv == "fp16";
    if (!tokens_file.empty()) {
        std::ifstream in(tokens_file);
        if (!in) { std::fprintf(stderr, "strata-q36: --tokens-file: cannot read %s\n", tokens_file.c_str()); return 2; }
        tokens.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
    setenv("STRATA_Q8_PACKED", "1", 0);   // the packed Q8_0 decode copy unless STRATA_Q8_PACKED is set otherwise
    setenv("STRATA_MMQ_KQ", "1", 0);      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("STRATA_Q8_PACKED") == nullptr) _putenv_s("STRATA_Q8_PACKED", "1");   // as on Linux (it was missing here)
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-q36: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-q36: bad --chunk / --max-context\n"); return 2; }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0)) { std::fprintf(stderr, "strata-q36: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-q36: %s\n", e.what());
        return 1;
    }
    Engine E;
    E.use_graph = graph;
    E.kv8 = kv8;
    E.kv_f16 = kv_f16;
    E.kv_v4 = kv_v4;
    if ((kv_f16 || kv_v4) && m.g.n_head / m.g.n_head_kv > 8) {
        std::fprintf(stderr, "strata-q36: --kv fp16 / k8v4 support at most 8 query heads per KV head\n"); return 2;
    }
    E.gqa_attn = gqa_attn;
    if (attn_mma) q36attn::mma_enabled() = true;
    if (q36attn::mma_enabled()) {   // --attn-mma / STRATA_Q36_ATTN_MMA: the tensor-core kernel is for the int8 cache with --gqa-attn, sm_80+
        int major = 8;
#ifndef __HIPCC__
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) major = 8;
#else
        major = 0;   // HIP not built: the kernel is compiled out
#endif
        if (!(kv8 && gqa_attn)) {
            std::fprintf(stderr, "strata-q36: --attn-mma needs --gqa-attn and --kv int8: ignored\n");
            q36attn::mma_enabled() = false;
        } else if (major < 8) {
            std::fprintf(stderr, "strata-q36: --attn-mma needs compute capability 8.0 or newer (this card is sm_%d): ignored\n", major);
            q36attn::mma_enabled() = false;
        }
    }
    E.fused_pattn = fused_pattn;
    E.chunked_gdn = chunked_gdn;
    E.side_on = side_stream;
    E.split_copy = split_stage_copy;
    if (decode_swaps >= 0) E.decode_swaps = decode_swaps;
    E.zc = zero_copy;
    E.pf_want = expert_prefetch;
    E.pool_warm_want = pool_warm;
    E.pf_copies_want = prefetch_copies;
    E.expert_profile = expert_profile;
    E.count_heat = !expert_profile_save.empty();
    // recurrent-state snapshots per session (--snapshots K, up to 4), taken after reads of prompts of --snapshot-min
    // tokens or more (4096); STRATA_Q36_SNAPSHOT_TEST=L[,J] checks one (a test setting, see below)
    E.n_snaps = std::max(0, std::min(4, snapshots));
    E.snap_min = std::max<int64_t>(1, snapshot_min);
    E.snap_host = snapshot_host;
    if (snapshot_host && E.n_snaps == 0)
        std::fprintf(stderr, "strata-q36: --snapshot-host does nothing without --snapshots K\n");
    const char* snap_test = std::getenv("STRATA_Q36_SNAPSHOT_TEST");
    if (snap_test && !serve_mode) E.n_snaps = std::max(E.n_snaps, 1);
    // --async-swaps: the decode rebalance's copies; --async-after-prompt: the after-prompt rebalance's too;
    // STRATA_Q36_ASYNC_SWAPS=1 / 2 / 3 sets the two bits (tests)
    E.lend_on = lend_staging;
    if (const char* ls = std::getenv("STRATA_Q36_LEND_SETTLE")) E.lend_settle = std::atoi(ls) != 0;   // 1: all lent copies settled before the first decode forward
    if (const char* lp = std::getenv("STRATA_Q36_LEND_PRIOR")) E.lend_prior = std::atoi(lp) != 0;   // 0: the counts only (tests)
    E.async_mask = (async_swaps ? 1 : 0) | (async_after ? 2 : 0) |
                   (std::getenv("STRATA_Q36_ASYNC_SWAPS") ? std::atoi(std::getenv("STRATA_Q36_ASYNC_SWAPS")) & 3 : 0);
    E.async_swaps = E.async_mask != 0;
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        const size_t fr_raw = fr;   // before the packed Q8_0 copies count as given back
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve
        const double want = (double) experts + 1.1 * (1ull << 30);
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        const double GiB = (double) (1ull << 30);
        if (fits && expert_vram < 0) {
            // a bigger prompt chunk feeds the MMQ expert kernels more rows per expert (R9700, UD-Q4_K_XL, pp4096: 512 3390, 1024
            // 4495, 2048 5353, 4096 5750 tok/s): the largest that fits beside the experts.  Its buffers grow by about 0.5 MiB per
            // token (measured 0.47: 1.66 GiB more at 4096 than at 512), the sessions' K / V caches by 50 KB per cell and session.
            // With several slots a long prompt would hold the other slots' windows for a whole chunk: 1024 at most there.
            // (measured on a 32 GB R9700 at 8192 cells, 512 chunk: 1.77 GiB of buffers + K / V; the free room after init must still hold
            // every expert and the 0.77 GiB reserve, or place_experts gives the packed decode copies back - slower decode - or goes hybrid)
            const int ns = std::max(1, batch);
            // per cell: a session's K / V, and the fp16 copies the prompt attention keeps (none with --kv fp16)
            // (k8v4: 11 attention layers x 2 heads x (288 + 144) bytes)
            const double kv_one = kv_v4 ? 9504.0 : kv8 ? 12672.0 : kv_f16 ? 22528.0 : 45056.0, kv_shared = kv_f16 ? 0.0 : 5120.0;
            const double sess_extra = std::max(0.0, (double) ctx * (kv_one * ns + kv_shared) - 8192.0 * (kv_one + kv_shared));
            const int cap = ns > 1 ? 1024 : 4096;
            for (int c : {4096, 2048, 1024})
                if (c <= cap && (double) experts + 2.75 * GiB + (double) (c - 512) * 0.5 * 1048576.0 + sess_extra <= (double) fr_raw) {   // (no give-back of the packed decode copies for a bigger chunk)
                    chunk = c;
                    break;
                }
        } else if (!fits) {
            // the hybrid mode: 2048 (a chunk streams every layer's experts over PCIe once); a small card, where the chunk's buffers
            // would take the slots' room, falls back to less
            const double fr_gib = (double) fr / GiB;
            chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;
        }
        if (const char* cv = std::getenv("STRATA_Q36_CHUNK")) chunk = std::max(256, std::min(4096, std::atoi(cv)));   // force a chunk (512: the old default)
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-q36: chunk rule: free %.2f GiB (raw %.2f, with %.2f packed), main experts %.2f GiB, chunk %d\n",
                         fr / 1073741824.0, fr_raw / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0, chunk);
    }
    E.init(m, ctx, chunk, batch, vision);
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, err)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<VramWatch> vram_watch;
    if (std::getenv("STRATA_Q36_VRAM_WATCH")) vram_watch.reset(new VramWatch);
    struct Report {
        Engine& E;
        const std::string& save;   // --expert-profile-save
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "strata-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (!E.m->hybrid) return;
            if (!save.empty()) {
                // what this run learned (#477's order, rank_learned_profile): per layer the experts in its cache slots
                // now, then the rest, each by the routing counted since the warm-up (`heat`), then by the profile it
                // started from, then by index (prefetch slots and layers without a slot table count as not resident)
                const int64_t ne = (int64_t) E.m->g.n_expert, nl = (int64_t) E.hy.size();
                std::vector<uint8_t> resident((size_t) (nl * ne), 0);
                std::vector<double> heat((size_t) (nl * ne), 0.0);
                for (int64_t l = 0; l < nl; ++l) {
                    const Engine::Hy& H = E.hy[(size_t) l];
                    const int S = E.layer_at((int) l).n_slots;
                    for (int64_t e = 0; e < ne && (size_t) e < H.slot_of.size(); ++e) {
                        resident[(size_t) (l * ne + e)] = H.slot_of[(size_t) e] >= 0 && (H.slot_of[(size_t) e] < S || H.slot_of[(size_t) e] >= S + (int) H.pf.size());
                        heat[(size_t) (l * ne + e)] = H.heat[(size_t) e];
                    }
                }
                const auto ranked = strata::core::rank_learned_profile(nl, ne, resident, heat, E.profile_ranked);
                std::string serr;
                if (strata::core::write_expert_profile(save, nl, ne, ranked, serr))
                    std::fprintf(stderr, "strata-q36: expert profile saved to %s\n", save.c_str());
                else std::fprintf(stderr, "strata-q36: expert profile not saved: %s\n", serr.c_str());
            }
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
            std::fprintf(stderr, "strata-q36: rebalance: decode %lld calls, %lld swaps, %.1f ms; after prompts %lld calls, %lld swaps, "
                         "%.1f ms\n", E.rb_calls[0], E.rb_swaps[0], E.rb_ms[0], E.rb_calls[1], E.rb_swaps[1], E.rb_ms[1]);
            if (E.lend_on) {
                std::fprintf(stderr, "strata-q36: lent slots: %lld evictions for the prompt stream (%.1f ms in all), %lld profile fills, %lld swaps into lent slots\n",
                             E.lend_evicts, E.lend_evict_ms, E.lend_fills, E.lend_swaps);
                if (E.tail_checks)
                    std::fprintf(stderr, "strata-q36: staging tail check: %lld tails read, %lld not zero\n", E.tail_checks, E.tail_bad);
            }
            if (E.async_swaps)
                std::fprintf(stderr, "strata-q36: async swaps: %lld copies in %lld batches, %lld taken into the tables, %zu still pending\n",
                             E.hy_swaps, E.sw_batches, E.sw_taken, E.swap_q.size());
            if (E.pf_check)
                std::fprintf(stderr, "strata-q36: table check: %lld forwards compared, %lld table mismatches; %lld slots compared byte for byte, %lld differ\n",
                             E.chk_calls, E.chk_bad, E.chk_bytes, E.chk_bytes_bad);
#if defined(STRATA_Q36_CPU_POOL)
            if (E.hy_layers && E.pool) {
                const auto& P = *E.pool;
                std::fprintf(stderr, "strata-q36: pool: %lld layer calls, %.1f us each (plan %.2f, gate/up %.2f, h-quant %.2f, "
                             "down %.2f); %.1f MB read (%.1f GB/s over gate/up + down; gate/up %.1f, down %.1f); %lld phases, "
                             "%.1f tasks each, park-before %.2f drain %.2f park-after %.2f us per phase\n", E.hy_layers,
                             1e3 * E.hy_cpu_ms / E.hy_layers, 1e3 * E.hy_plan_ms / E.hy_layers,
                             1e3 * P.ms_multi_gu / E.hy_layers, 1e3 * P.ms_multi_q / E.hy_layers,
                             1e3 * P.ms_multi_down / E.hy_layers, (double) P.multi_bytes / 1e6,
                             (double) P.multi_bytes / 1e6 / std::max(1e-9, P.ms_multi_gu + P.ms_multi_down),
                             (double) P.bytes_gu / 1e6 / std::max(1e-9, P.ms_multi_gu),
                             (double) P.bytes_down / 1e6 / std::max(1e-9, P.ms_multi_down), (long long) P.ph_calls,
                             P.ph_calls ? (double) P.ph_tasks / (double) P.ph_calls : 0.0,
                             P.ph_calls ? 1e3 * P.ms_ph_pre / (double) P.ph_calls : 0.0,
                             P.ph_calls ? 1e3 * P.ms_ph_drain / (double) P.ph_calls : 0.0,
                             P.ph_calls ? 1e3 * P.ms_ph_post / (double) P.ph_calls : 0.0);
            }
#endif
#if defined(STRATA_Q36_CPU_POOL)
            if (E.pool && E.pool->warm_enabled()) {
                const auto& P = *E.pool;
                std::fprintf(stderr, "strata-q36: pool warm (%d per layer): %lld sets posted, %lld complete when the call came\n",
                             E.pool_warm_k, (long long) P.warm_posts, (long long) P.warm_full);
                static const char* const nm[3] = {"no expert warmed", "some warmed", "all warmed"};
                for (int k = 0; k < 3; ++k)
                    if (P.wc_calls[k])
                        std::fprintf(stderr, "strata-q36: pool calls with %s: %lld calls, %.1f experts, %.1f us each, %.1f GB/s\n", nm[k],
                                     (long long) P.wc_calls[k], (double) P.wc_jobs[k] / P.wc_calls[k], 1e3 * P.wc_ms[k] / P.wc_calls[k],
                                     (double) P.wc_bytes[k] / 1e6 / std::max(1e-9, P.wc_ms[k]));
            }
#endif
            if (E.pf_slots)
                std::fprintf(stderr, "strata-q36: prefetch: %lld predictions, %lld copies, %lld entries hit a prefetch slot, "
                             "%lld routings found a copy still in flight\n", E.pf_preds, E.pf_copies, E.pf_hits, E.pf_late);
        }
    } report{E, expert_profile_save};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    E.mtp_sampled = mtp_sampled;
    sampling::tie_by_id() = mtp_sampled > 0;   // (--sample-gpu sets it where it is parsed)
    const auto env_on = [](const char* k) { const char* v = std::getenv(k); return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0; };
    E.sample_gpu = sample_gpu || env_on("STRATA_Q36_SAMPLE_GPU");
    if (E.sample_gpu) sampling::tie_by_id() = true;   // the shortlists use the tie rule, so the host rows do too
    E.smp_prof = env_on("STRATA_Q36_SAMPLE_PROFILE");
    E.smp_check = env_on("STRATA_Q36_SAMPLE_CHECK");
    if (E.sample_gpu && E.smp_check) E.cand_selftest();
    E.spec_min_p = std::clamp(spec_min_p, 0.f, 1.f);   // (before the warm-up: the draft step's graph holds the gate's kernel)
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.draft_n = std::max(0, draft_vocab);
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-q36: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-q36: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    if (!tokens_file.empty()) {   // (a file has no argument's length limit: the cache's and the vocabulary's checked)
        for (const int32_t t : prompt)
            if (t < 0 || t >= (int32_t) m.g.n_vocab) {
                std::fprintf(stderr, "strata-q36: --tokens-file: id %d is outside the vocabulary (%lld)\n", t, (long long) m.g.n_vocab);
                return 2;
            }
        if ((int64_t) prompt.size() + n_gen + E.mtp_d > (int64_t) E.ctx) {   // (a speculative step's window past the end)
            std::fprintf(stderr, "strata-q36: --tokens-file: %zu tokens and %d to generate do not fit --max-context %d\n",
                         prompt.size(), n_gen, E.ctx);
            return 2;
        }
    }
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "strata-q36: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "strata-q36: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "strata-q36: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
    const auto t0 = Clock::now();
    if (snap_test && !serve_mode && emb_file.empty()) {
        // STRATA_Q36_SNAPSHOT_TEST=L[,J] (a test setting): the prompt's first L tokens, a snapshot, J other tokens
        // (default 2048: the prompt's own start again, which writes the cache cells and the state past L), the snapshot
        // restored, the rest of the prompt.  Its logits (--force, --dump-full-logits) against a plain run's: what a snapshot-resumed
        // read gives against a fresh one.  J = 0 is the same split without the detour (the reference for the chunking).
        const int64_t L = std::atoll(snap_test);
        const char* cm = std::strchr(snap_test, ',');
        const int64_t J = std::min<int64_t>({cm ? std::atoll(cm + 1) : 2048, (int64_t) prompt.size(), (int64_t) E.ctx - L});
        if (L <= 0 || L >= (int64_t) prompt.size()) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_SNAPSHOT_TEST=%s: L must be inside the prompt (%zu tokens)\n",
                         snap_test, prompt.size());
            return 2;
        }
        E.process(prompt.data(), L, [](int64_t) {});
        const auto ts0 = Clock::now();
        E.snapshot();
        const double snap_ms = ms_since(ts0);
        if (J > 0) E.process(prompt.data(), J, [](int64_t) {});
        const auto tr0 = Clock::now();
        E.restore(E.cur, 0, E.cur);
        std::fprintf(stderr, "strata-q36: snapshot test: one snapshot %.2f ms, one restore %.2f ms (%s)\n", snap_ms, ms_since(tr0),
                     E.snap_host ? "host" : "VRAM");
        E.process(prompt.data() + L, (int64_t) prompt.size() - L, [](int64_t) {});
        std::fprintf(stderr, "strata-q36: snapshot test: %lld tokens, a snapshot, %lld other tokens, the snapshot restored, "
                     "the prompt's other %lld tokens\n", (long long) L, (long long) J, (long long) prompt.size() - L);
    } else {
        E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    }
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.spec_min_p > 0.f ? E.last_d : E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
