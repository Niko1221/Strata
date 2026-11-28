#define DPCT_PROFILING_ENABLED
#define DPCT_COMPAT_RT_VERSION 12080
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
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include <dpct/blas_utils.hpp>
#include <stdexcept>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "q36_kernels.dp.hpp"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <dpct/blas_utils.hpp>

#endif

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
#include <dpct/lib_common_utils.hpp>

#include <chrono>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

// SYCL USM may report allocation failure with nullptr instead of an exception.
inline void* checked_device_alloc(size_t bytes, sycl::queue& q) {
    void* p = sycl::malloc_device(bytes, q);
    if (!p && bytes) throw std::bad_alloc();
    return p;
}
template<class T> T* checked_device_alloc(size_t count, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(count, q);
    if (!p && count) throw std::bad_alloc();
    return p;
}
inline void* checked_host_alloc(size_t bytes, sycl::queue& q) {
    void* p = sycl::malloc_host(bytes, q);
    if (!p && bytes) throw std::bad_alloc();
    return p;
}
template<class T> T* checked_host_alloc(size_t count, sycl::queue& q) {
    T* p = sycl::malloc_host<T>(count, q);
    if (!p && count) throw std::bad_alloc();
    return p;
}


/*
DPCT1009:75: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CK(x) do { const auto code_ = (x); if (code_ != 0) throw std::runtime_error("Qwen3.6 SYCL runtime failure: " + std::to_string(code_)); } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)

// ------------------------------------------------------------------------------------------------ small kernels
__dpct_inline__ float silu(float v) {
    return v / (1.0f + sycl::native::exp(-v));
}
__dpct_inline__ float sigm(float v) {
    return 1.0f / (1.0f + sycl::native::exp(-v));
}
__dpct_inline__ float softplus(float v) {
    return v > 20.f ? v : sycl::log1p(sycl::native::exp(v));
}

float block_sum(float v, float* sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sh[item_ct1.get_local_id(2)] = v;
    /*
    DPCT1065:53: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1) {
        if ((int)item_ct1.get_local_id(2) < s) sh[item_ct1.get_local_id(2)] +=
            sh[item_ct1.get_local_id(2) + s];
        /*
        DPCT1118:14: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sh[0];
    /*
    DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}
float block_max(float v, float* sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sh[item_ct1.get_local_id(2)] = v;
    /*
    DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1) {
        if ((int)item_ct1.get_local_id(2) < s) sh[item_ct1.get_local_id(2)] =
            sycl::fmax(sh[item_ct1.get_local_id(2)],
                       sh[item_ct1.get_local_id(2) + s]);
        /*
        DPCT1118:15: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sh[0];
    /*
    DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
void k_rms(const float* x, const float* w, float* y, int n, float eps,
           float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const float *xr = x + (size_t)item_ct1.get_group(2) * n;
    float *yr = y + (size_t)item_ct1.get_group(2) * n;
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = sycl::rsqrt(s / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps,
              float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const float *xr = x + (size_t)item_ct1.get_group(2) * ldx;
    float *yr = out + (size_t)item_ct1.get_group(2) * ld_out;
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = sycl::rsqrt(s / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) yr[i] = xr[i] * r * w[i];
}
void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps,
               float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    float *xr = x + (size_t)item_ct1.get_group(2) * n;
    const float *yr = y + (size_t)item_ct1.get_group(2) * n;
    float *hr = h + (size_t)item_ct1.get_group(2) * n;
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = sycl::rsqrt(s / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps, float *sh, const float **pk) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_group(2);
    float* xr = x + (size_t) t * n;
    if (item_ct1.get_local_id(2) < K_) // pick k's row: t * K + k, or its sorted
                                       // row (`prow`, the prompt's MMQ order)
        pk[item_ct1.get_local_id(2)] =
            parts + (size_t)(prow ? prow[t * K_ + item_ct1.get_local_id(2)]
                                  : t * K_ + item_ct1.get_local_id(2)) *
                        n;
    /*
    DPCT1065:59: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k)
            acc = sycl::fma((float)(wr[k]), (float)(pk[k][i]), acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = sycl::rsqrt(s / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2))
        h[(size_t)t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) / 32,
              lane = item_ct1.get_local_id(2) % 32;
    const int o = item_ct1.get_group(2) * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const sycl::float4 *wr =
        (const sycl::float4 *)((isb ? Wb : Wa) + (size_t)r * n_in);
    const sycl::float4 *xr =
        (const sycl::float4 *)(x + (size_t)item_ct1.get_group(1) * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const sycl::float4 a = wr[i], b = xr[i];
        acc += a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
    }
    for (int m = 16; m > 0; m >>= 1) acc += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t)item_ct1.get_group(1) * HV + r] = sigm(acc);
    else alpha[(size_t)item_ct1.get_group(1) * HV + r] =
        softplus(acc + dt[r]) * ssm_a[r];
}
void k_add(float* x, const float* y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) x[i] += y[i];
}
void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) / 32,
              lane = item_ct1.get_local_id(2) % 32;
    const int o = item_ct1.get_group(2) * 8 + warp;
    if (o >= n_out) return;
    const sycl::float4 *wr = (const sycl::float4 *)(W + (size_t)o * n_in);
    const sycl::float4 *xr =
        (const sycl::float4 *)(x + (size_t)item_ct1.get_group(1) * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const sycl::float4 a = wr[i], b = xr[i];
        acc += a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
    }
    for (int m = 16; m > 0; m >>= 1) acc += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), acc, m);
    if (lane == 0) y[(size_t)item_ct1.get_group(1) * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
void k_gate_rows(float* sh, const float* gate, int n, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
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

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256).
void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, float* kc, float* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs, float *sh,
                            float *buf) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int head = item_ct1.get_group(2), t = item_ct1.get_group(1);
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = item_ct1.get_local_id(2);
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = sycl::rsqrt(s / (float)HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    /*
    DPCT1065:60: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        /*
        DPCT1098:61: The '*' expression is used instead of the __ldg call. These
        two expressions do not provide the exact same functionality. Check the
        generated code for potential precision and/or performance issues.
        */
        const int rp = mtab ? *(mtab + (size_t)pos * 3 + p % 3) : pos + dl;
        const float theta =
            (float)rp * dpct::pow(base, -2.0f * (float)p / (float)n_rot);
        float sn, cs;
        sn = sycl::sincos(
            theta,
            sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(&cs));
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        kc[((size_t) kh * ctx + pos) * HD + i] = v;
        vc[((size_t) kh * ctx + pos) * HD + i] = vx[((size_t) t * HKV + kh) * HD + i];
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.
void k_attn_part(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs,
                            float *qs, float *sc, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int HD = 256;

    const int c = item_ct1.get_group(2), row = item_ct1.get_group(1);
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = item_ct1.get_local_id(2);
    qs[j] = q[((size_t) t * H + h) * HD + j];
    /*
    DPCT1065:62: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const float* kp = kc + ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a += qs[i] * kp[i];
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? sycl::native::exp(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = sycl::min(kChunkCells, npos - p0);
    float acc = 0.f;
    const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int HD = 256;
    const int row = item_ct1.get_group(2), t = t0 + row / H, h = row % H,
              d = item_ct1.get_local_id(2);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c)
        M = sycl::fmax(M, pm[(size_t)row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = sycl::native::exp(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok, int *cnt, int *off) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    for (int x = item_ct1.get_local_id(2); x < n_expert;
         x += item_ct1.get_local_range(2)) cnt[x] = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int e = item_ct1.get_local_id(2); e < n_ent;
         e += item_ct1.get_local_range(2))
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnt[ids[e]], 1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
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
void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                           int *cnt, int *off) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    for (int x = item_ct1.get_local_id(2); x < n_expert;
         x += item_ct1.get_local_range(2)) cnt[x] = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int e = item_ct1.get_local_id(2); e < n_ent;
         e += item_ct1.get_local_range(2))
        if (slot_of[ids[e]] >= 0)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &cnt[ids[e]], 1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
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
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2);
    if (!miss[e]) return;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2))
        parts[(size_t)e * n + i] = cpu[(size_t)e * n + i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).
void k_kv_f16(const float *kc, const float *vc, sycl::half *k16,
              sycl::half *v16, int ctx, int P, int HD, int HKV) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    k16[i] = sycl::vec<float, 1>(kc[src])
                 .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    v16[i] = sycl::vec<float, 1>(vc[src])
                 .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
void k_attn_softmax16(const float *S, sycl::half *S16, int ld, int T, int P,
                      int pos0, float scale, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_group(2), h = item_ct1.get_group(1);
    const float* row = S + ((size_t) h * T + t) * ld;
    sycl::half *out = S16 + ((size_t)h * T + t) * ld;
    const int n = sycl::min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = item_ct1.get_local_id(2); p < n;
         p += item_ct1.get_local_range(2)) mx = sycl::fmax(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = item_ct1.get_local_id(2); p < n;
         p += item_ct1.get_local_range(2)) sum +=
        sycl::native::exp(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = item_ct1.get_local_id(2); p < P;
         p += item_ct1.get_local_range(2)) out[p] =
        sycl::vec<float, 1>(p < n ? sycl::native::exp(row[p] * scale - mx) * inv
                                  : 0.f)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}
/// att *= sigmoid(gate) (the attention output gate), n values.
void k_gate_mul(float* att, const float* gate, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) att[i] *= sigm(gate[i]);
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    const int m = map[r];
    if (m < 0) return;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2))
        x[(size_t)r * n + i] = img[(size_t)m * n + i];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (e < n_ent)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnt[ids[e]], 1);
}
void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = dpct::max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (e >= n_ent) return;
    const int r =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2,
              float *bv, int *bi) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t)item_ct1.get_group(1) * (stage2 ? 0 : n);
    pv += (size_t)item_ct1.get_group(1) * 128;
    pi += (size_t)item_ct1.get_group(1) * 128;
    out += item_ct1.get_group(1);
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2))
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
             i < n;
             i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[item_ct1.get_local_id(2)] = v;
    bi[item_ct1.get_local_id(2)] = id;
    /*
    DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1) {
        if ((int)item_ct1.get_local_id(2) < s) {
            const float ov = bv[item_ct1.get_local_id(2) + s];
            const int oi = bi[item_ct1.get_local_id(2) + s];
            if (ov > bv[item_ct1.get_local_id(2)] ||
                (ov == bv[item_ct1.get_local_id(2)] &&
                 oi < bi[item_ct1.get_local_id(2)])) {
                bv[item_ct1.get_local_id(2)] = ov;
                bi[item_ct1.get_local_id(2)] = oi;
            }
        }
        /*
        DPCT1118:16: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (item_ct1.get_local_id(2) == 0) {
        if (stage2) *out = bi[0];
        else {
            pv[item_ct1.get_group(2)] = bv[0];
            pi[item_ct1.get_group(2)] = bi[0];
        }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
/*
DPCT1110:17: The total declared local variable size in device function
k_sgemm_nt exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
void k_sgemm_nt(const float *__restrict__ W, const float *__restrict__ x,
                float *__restrict__ out, int K, int n_out, int T, int ldo,
                int kz, float As[16][16 * TM + 4],
                float Bs[16 /*BK*/][68 /*BN + 4*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = item_ct1.get_group(0) * kz,
              ke = sycl::min(
                  K, kb + kz); // split-K: this block's K range, its own partial
    out += (size_t)item_ct1.get_group(0) * T * ldo;

    const int tid = item_ct1.get_local_id(2);
    const int m0 = item_ct1.get_group(1) * BM, n0 = item_ct1.get_group(2) * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            sycl::float4 a = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a =
                *reinterpret_cast<const sycl::float4 *>(
                    x + (size_t)(m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x(); As[lk + 1][lr] = a.y(); As[lk + 2][lr] = a.z();
                As[lk + 3][lr] = a.w();
        }
        sycl::float4 b = sycl::float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b =
            *reinterpret_cast<const sycl::float4 *>(W + (size_t)(n0 + lr) * K +
                                                    k0 + lk);
        Bs[lk][lr] = b.x(); Bs[lk + 1][lr] = b.y(); Bs[lk + 2][lr] = b.z();
            Bs[lk + 3][lr] = b.w();
        /*
        DPCT1118:18: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const sycl::float4 bv =
                *reinterpret_cast<const sycl::float4 *>(&Bs[k][tn]);
            const float br[4] = {bv.x(), bv.y(), bv.z(), bv.w()};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = sycl::fma(ar[i], (float)(br[j]), acc[i][j]);
        }
        /*
        DPCT1118:19: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
void k_to_f16(const float *x, sycl::half *y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] =
        sycl::vec<float, 1>(x[i])
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
void k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float *part,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = item_ct1.get_group(1), r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = item_ct1.get_group(2) * 128 + item_ct1.get_local_id(2);
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && item_ct1.get_group_range(1) > 1;
    if (tiled) {
        t_first = (int)item_ct1.get_group(1) * kConvTile;
        t_end = sycl::min(T, t_first + kConvTile);
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
    const bool qk = (int)item_ct1.get_group(2) < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + sycl::native::exp(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq +=
                dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), sq, o);
            if ((item_ct1.get_local_id(2) & 31) == 0)
                part[item_ct1.get_local_id(2) >> 5] = sq;
            /*
            DPCT1118:20: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:65: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            const float ss = part[0] + part[1] + part[2] + part[3];
            /*
            DPCT1118:21: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:66: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            y *= sycl::rsqrt(ss + eps);
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
/*
DPCT1110:22: The total declared local variable size in device function k_gdn_seq
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
void k_gdn_seq(float *state, const float *h, int C, const float *gate,
               const float *beta, const float *z, const float *gamma, float eps,
               float *y, int h_k, int h_v, int T, bool write, float *sk,
               float *sq, float red_kv[4 /*RG*/][128 /*S*/],
               float red_o[4 /*RG*/][128 /*S*/], float *wsum,
               float *const *state_tab = nullptr,
               const int32_t *slot_rows = nullptr,
               const int32_t *grow0 = nullptr, const int32_t *gcnt = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = item_ct1.get_group(1), r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }

    const int head = item_ct1.get_group(2), col = item_ct1.get_local_id(2),
              rg = item_ct1.get_local_id(1), tid = rg * S + col;
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
        /*
        DPCT1118:23: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:67: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        /*
        DPCT1118:24: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:68: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        /*
        DPCT1118:25: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] +
                  red_o[3][col]) *
                 sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
            dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), sq_part,
                o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        /*
        DPCT1118:26: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:70: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t)t * ZV + head * S + col] =
                oc * scale * gamma[col] *
                (zz / (1.0f + sycl::native::exp(-zz)));
        }
        /*
        DPCT1118:27: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:71: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // sk / sq / the reductions are rewritten by the
                            // next token
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
/*
DPCT1110:28: The total declared local variable size in device function
k_gdn_seq_cols exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void k_gdn_seq_cols(float *state, const float *h, int C, const float *gate,
                    const float *beta, float *o_raw, int h_k, int h_v, int T,
                    float sk[2][128 /*S*/], float sq[2][128 /*S*/],
                    float red_kv[16 /*RG*/][32 /*NC*/],
                    float red_o[16 /*RG*/][32 /*NC*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)

    const int head = item_ct1.get_group(2), cl = item_ct1.get_local_id(2),
              rg = item_ct1.get_local_id(1), tid = rg * NC + cl;
    const int col = item_ct1.get_group(1) * NC + cl;
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
        /*
        DPCT1118:29: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t)(t - 1) * h_v * S + head * S + col] =
                oc * sycl::rsqrt((float)S);
        }
        const float g = sycl::native::exp(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        /*
        DPCT1118:30: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:74: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = sycl::fma((float)g, sv[r], sk[par][rg * RPG + r] * delta);
            o = sycl::fma(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    /*
    DPCT1065:72: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t)(T - 1) * h_v * S + head * S + col] =
            oc * sycl::rsqrt((float)S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
void k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v,
                                                  float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const size_t i =
        ((size_t)item_ct1.get_group(1) * h_v + item_ct1.get_group(2)) * 128 +
        item_ct1.get_local_id(2);
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sq, o);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = sq;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = sycl::rsqrt(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[item_ct1.get_local_id(2)] *
           (zz / (1.0f + sycl::native::exp(-zz)));
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
float *upload_f32(const strata::TensorInfo &t, const uint8_t *src) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
    CK(DPCT_CHECK_ERROR(d = (float *)checked_device_alloc(h.size() * 4, q_ct1)));
    CK(DPCT_CHECK_ERROR(q_ct1.memcpy(d, h.data(), h.size() * 4).wait_and_throw()));
    return d;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool load(const std::string &path, Model &m, std::string &err, bool want_mtp) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
    auto mat = [&](const std::string &name) -> Mat {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
        CK(DPCT_CHECK_ERROR(d = (uint8_t *)checked_device_alloc(nb, q_ct1)));
        CK(DPCT_CHECK_ERROR(
            q_ct1.memcpy(d, gm.shard(s).tensor_data(*t), nb).wait_and_throw()));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(DPCT_CHECK_ERROR(pd =
                                    (uint8_t *)checked_device_alloc(nb, q_ct1)));
            CK(DPCT_CHECK_ERROR(q_ct1.memcpy(pd, packed.data(), nb).wait_and_throw()));
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
        CK(DPCT_CHECK_ERROR(
            y.router_ext = (float *)checked_device_alloc(
                (size_t)(g.n_expert + 1) * g.n_embd * 4, q_ct1)));
        CK(DPCT_CHECK_ERROR(q_ct1.memcpy(y.router_ext, y.router,
                                         (size_t)g.n_expert * g.n_embd * 4)));
        CK(DPCT_CHECK_ERROR(
            q_ct1.memcpy(y.router_ext + (size_t)g.n_expert * g.n_embd,
                         y.sh_gate_inp, (size_t)g.n_embd * 4)));
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
        CK(DPCT_CHECK_ERROR(d = (uint8_t *)checked_device_alloc(nb, q_ct1)));
        CK(DPCT_CHECK_ERROR(
            q_ct1.memcpy(d, gm.shard(s).tensor_data(*t), nb).wait_and_throw()));
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

// oneMKL rejects CUDA's interleaved-C strided batch (strideC=HD, ldc=H*HD).
// Each head occupies disjoint rows of C: issue ordinary GEMMs with the same layout.
void q36_attention_values_gemm(dpct::blas::descriptor_ptr handle,
        const sycl::half* values, const sycl::half* probabilities, float* out,
        int HD, int nt, int P, int ldp, int H, int G) {
    const float one = 1.0f, zero = 0.0f;
    for (int head = 0; head < G; ++head) {
        CK(DPCT_CHECK_ERROR(dpct::blas::gemm(
            handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            HD, nt, P, &one, values, dpct::library_data_t::real_half, HD,
            probabilities + (size_t)head * nt * ldp, dpct::library_data_t::real_half, ldp,
            &zero, out + (size_t)head * HD, dpct::library_data_t::real_float, H * HD,
            dpct::compute_type::f32)));
    }
}

// ------------------------------------------------------------------------------------------------ the engine
float *dalloc(size_t n) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    float* p = nullptr;
    CK(DPCT_CHECK_ERROR(p = (float *)checked_device_alloc(n * 4, q_ct1)));
    CK(DPCT_CHECK_ERROR(q_ct1.memset(p, 0, n * 4).wait_and_throw()));
    return p;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
template <class T> T *dalloc_t(size_t n) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    T* p = nullptr;
    CK(DPCT_CHECK_ERROR(p = (T *)checked_device_alloc(n * sizeof(T), q_ct1)));
    CK(DPCT_CHECK_ERROR(q_ct1.memset(p, 0, n * sizeof(T)).wait_and_throw()));
    return p;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

struct Engine {
    const Model* m = nullptr;
    dpct::queue_ptr st = &dpct::get_in_order_queue();
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
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
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        dpct::experimental::command_graph_exec_ptr g1 = nullptr,
                                                   gv[kVerify + 1] = {},
                                                   gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    strata::prefill::Gemm gemm;
    dpct::blas::descriptor_ptr blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    sycl::half *att_p16 =
        nullptr; // the probabilities in FP16 (kAttScoreFloats halves)
    sycl::half *k16 = nullptr, *v16 = nullptr,
               *q16 = nullptr; // the prompt attention's FP16 operands
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats when the experts' fit is close)
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    sycl::half *x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S, kc, vc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
    };
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
        int32_t* slot_of_d = nullptr;
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
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    dpct::event_ptr ev_route = nullptr;
    std::chrono::time_point<std::chrono::steady_clock> ev_route_ct1;
    uint8_t* stage[2] = {};
    dpct::queue_ptr cst = &dpct::get_in_order_queue();
    dpct::event_ptr ev_copy[2] = {}, ev_used[2] = {};
    std::chrono::time_point<std::chrono::steady_clock> ev_copy_ct1[2];
    std::chrono::time_point<std::chrono::steady_clock> ev_used_ct1[2];
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
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
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }
    void after_rows(int n) try { // a decode step / verify window kept n rows:
                                 // count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
            rebalance(8);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        for (const Layer& L : m->layers) {
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(dalloc((size_t) g.n_head_kv * ctx * g.head_dim));
                ss.vc.push_back(dalloc((size_t) g.n_head_kv * ctx * g.head_dim));
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
            ss.kc.push_back(dalloc((size_t) g.n_head_kv * ctx * g.head_dim));
            ss.vc.push_back(dalloc((size_t) g.n_head_kv * ctx * g.head_dim));
        }
        ss.hlast = dalloc((size_t) g.n_embd);
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
    void set_delta(int32_t d) try {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        /*
        DPCT1124:76: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_delta, hpin + 60, 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw())); // (the pinned word is reused)
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    void init(const Model &model, int ctx_, int tmax_, int n_sessions = 1,
              bool vision_ = false) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        /*
        DPCT1025:77: The SYCL queue is created ignoring the flag and priority
        options.
        */
        CK(DPCT_CHECK_ERROR(st = dev_ct1.create_queue(true)));
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
        CK(DPCT_CHECK_ERROR(q8 = (void *)checked_device_alloc(
                                K::native_q8_1_bytes(max_in, 1) * T, q_ct1)));
        CK(DPCT_CHECK_ERROR(xq = (void *)checked_device_alloc(
                                K::native_q8_1_bytes((int)N, 1) * T, q_ct1)));
        x16 = dalloc_t<sycl::half>(T * (size_t)max_in);
        if (DPCT_CHECK_ERROR(blas = new dpct::blas::descriptor()) != 0 ||
            DPCT_CHECK_ERROR((blas)->set_queue(st)) != 0) {
            std::fprintf(stderr, "strata-q36: oneMKL handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        att_p16 = dalloc_t<sycl::half>(kAttScoreFloats);
        k16 = dalloc_t<sycl::half>((size_t)g.n_head_kv * ctx * g.head_dim);
        v16 = dalloc_t<sycl::half>((size_t)g.n_head_kv * ctx * g.head_dim);
        q16 = dalloc_t<sycl::half>(T * g.n_head * g.head_dim);
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
            CK(DPCT_CHECK_ERROR(
                dev_ct1.queues_wait_and_throw())); // ident's cudaMemset (legacy
                                                   // stream) before the iota on
                                                   // the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(DPCT_CHECK_ERROR(mq_x = (void *)checked_device_alloc(
                                    MMQ::q8_bytes(ne, g.n_embd), q_ct1)));
            CK(DPCT_CHECK_ERROR(mq_hq = (void *)checked_device_alloc(
                                    MMQ::q8_bytes(ne, g.n_ff), q_ct1)));
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
        CK(DPCT_CHECK_ERROR(
            escr = (void *)checked_device_alloc(
                K::native_expert_scratch_bytes((int64_t)(T * Kp), (int64_t)FF),
                q_ct1)));
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
        /*
        DPCT1048:0: The original value cudaHostAllocDefault is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        CK(DPCT_CHECK_ERROR(hpin = checked_host_alloc<int32_t>(64, q_ct1)));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, 64));   // 64 query tokens per attention batch
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
        }
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess);
        {   // the sessions' caches: said before they fail (a slot's K / V at the context, fp32, for the attention layers and
            // the MTP block, and its recurrent state)
            size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
            for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
            const size_t per = (size_t) (g.n_head_kv * g.head_dim * 2 * 4) * (size_t) ctx * n_attn +
                               (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn;
            size_t fr0 = 0, tot0 = 0;
            /*
            DPCT1106:78: 'cudaMemGetInfo' was migrated with the Intel extensions
            for device information which may not be supported by all compilers
            or runtimes. You may need to adjust the code.
            */
            CK(DPCT_CHECK_ERROR(
                dpct::get_current_device().get_memory_info(fr0, tot0)));
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
                CK(DPCT_CHECK_ERROR(*tb.first = checked_device_alloc<float *>(
                                        tb.second->size(), q_ct1)));
                CK(DPCT_CHECK_ERROR(
                    q_ct1
                        .memcpy((void *)*tb.first, tb.second->data(),
                                tb.second->size() * sizeof(float *))
                        .wait_and_throw()));
            }
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            /*
            DPCT1048:1: The original value cudaHostAllocDefault is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            CK(DPCT_CHECK_ERROR(hrow = checked_host_alloc<int32_t>(128, q_ct1)));
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            d_map = dalloc_t<int32_t>(T);
            /*
            DPCT1048:2: The original value cudaHostAllocDefault is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            CK(DPCT_CHECK_ERROR(h_map = checked_host_alloc<int32_t>(T, q_ct1)));
        }
        d_delta = dalloc_t<int32_t>(1);
        cur = 0;
        load_cur();
        CK(DPCT_CHECK_ERROR(
            dev_ct1.queues_wait_and_throw())); // every legacy-stream cudaMemset
                                               // of the buffers above is done
        size_t fr = 0, tot = 0;
        /*
        DPCT1106:79: 'cudaMemGetInfo' was migrated with the Intel extensions for
        device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        CK(DPCT_CHECK_ERROR(
            dpct::get_current_device().get_memory_info(fr, tot)));
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() try {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(DPCT_CHECK_ERROR(st->memset(
                conv[l], 0,
                (size_t)g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4)));
            if (S[l]) CK(DPCT_CHECK_ERROR(
                st->memset(S[l], 0,
                           (size_t)g.ssm_v_heads * g.ssm_state_size *
                               g.ssm_state_size * 4)));
        }
        CK(DPCT_CHECK_ERROR(st->memset(hlast, 0, (size_t)g.n_embd * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float *W, const float *xin, float *out, int n_in,
               int n_out, int T) try {
        if (T <= kGemmT) {
            ((sycl::queue *)(st))
                ->parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, T, rows8(n_out)) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            k_gemv_f32(W, xin, out, n_in, n_out);
                        });
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
                if (tm == 4)((sycl::queue *)(st))
                    ->submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:157: 'BK' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[16][68], 0>
                            As_acc_ct1(cgh);
                        /*
                        DPCT1101:158: 'BK' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        /*
                        DPCT1101:159: 'BN + 4' expression was replaced with a
                        value. Modify the code to use the original expression,
                        provided in comments, if it is correct.
                        */
                        sycl::local_accessor<float[16 /*BK*/][68 /*BN + 4*/], 0>
                            Bs_acc_ct1(cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(splits, gy, gx) *
                                                  sycl::range<3>(1, 1, 256),
                                              sycl::range<3>(1, 1, 256)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_sgemm_nt<4>(W, xin, dst, n_in, n_out, T,
                                              n_out, kz, As_acc_ct1,
                                              Bs_acc_ct1);
                            });
                    });
                else((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1101:160: 'BK' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float[16][20], 0>
                        As_acc_ct1(cgh);
                    /*
                    DPCT1101:161: 'BK' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    /*
                    DPCT1101:162: 'BN + 4' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float[16 /*BK*/][68 /*BN + 4*/], 0>
                        Bs_acc_ct1(cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(splits, gy, gx) *
                                              sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_sgemm_nt<1>(W, xin, dst, n_in, n_out, T, n_out,
                                          kz, As_acc_ct1, Bs_acc_ct1);
                        });
                });
            if (splits > 1)
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    const float *sk_part_ct0 = sk_part;
                    auto int64_t_T_n_out_ct2 = (int64_t)T * n_out;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, grid1((int64_t)T * n_out)) *
                                sycl::range<3>(1, 1, kThreads),
                            sycl::range<3>(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_sum_splits(sk_part_ct0, out, int64_t_T_n_out_ct2,
                                         splits);
                        });
                });
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
                (blas)->get_queue(), oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, n_out, T, n_in, one, W, n_in,
                xin, n_in, zero, out, n_out)) != 0) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) {
            dpct::has_capability_or_fail(((sycl::queue *)(st))->get_device(),
                                         {sycl::aspect::fp16});

            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                auto x16_ct1 = x16;
                auto int64_t_n_T_ct2 = (int64_t)n * T;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, grid1((int64_t)n * T)) *
                            sycl::range<3>(1, 1, kThreads),
                        sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_to_f16(xin, x16_ct1, int64_t_n_T_ct2);
                    });
            });
            } else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (STRATA_Q36_VERIFY=1 compares)
    bool fast_k = [] {
        const char* wide = std::getenv("STRATA_MMVQ_WIDE_32");
        return std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr && (!wide || std::atoi(wide) != 0);
    }();
    bool verify_k = std::getenv("STRATA_Q36_VERIFY") != nullptr;
    float* vbuf = nullptr;                       // verify scratch
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    void verify_cmp(const char *what, const float *a, const float *b, size_t n,
                    const std::string &tag = "") try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        std::vector<float> ha(n), hb(n);
        CK(DPCT_CHECK_ERROR(q_ct1.memcpy(ha.data(), a, n * 4).wait_and_throw()));
        CK(DPCT_CHECK_ERROR(q_ct1.memcpy(hb.data(), b, n * 4).wait_and_throw()));
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048
    /// rows) for T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the 4-warp block of
    /// native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat &w, const void *q, float *out, int T) try {
        if (!fast_k || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const sycl::half *dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(wp, dpl, q, out, nb, w.n_out, st);                  \
        else q36k::launch_q8_rows<NCV, 2, false>(wp, dpl, q, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            if (!vbuf) CK(DPCT_CHECK_ERROR(
                vbuf = (float *)checked_device_alloc(
                    (size_t)8 * 16384 * 4, dpct::get_in_order_queue())));
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char *what, const float *hsrc, int T,
                   const std::string &tag) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        if (!vq) CK(
            DPCT_CHECK_ERROR(vq = (void *)checked_device_alloc(qb * 8, q_ct1)));
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        const long long before = v_bad;
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
        if (v_bad != before && std::getenv("STRATA_Q36_VDEBUG") && v_by[what] <= 2) {   // a mismatching block, on the host
            std::vector<uint8_t> a(qb * (size_t) T), b(qb * (size_t) T);
            std::vector<float> hv((size_t) N * T);
            CK(DPCT_CHECK_ERROR(q_ct1.memcpy(a.data(), xq, a.size()).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(q_ct1.memcpy(b.data(), vq, b.size()).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(hv.data(), hsrc, hv.size() * 4).wait_and_throw()));
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
                    const sycl::half t0h =
                        sycl::vec<float, 1>(t[0])
                            .convert<sycl::half,
                                     sycl::rounding_mode::automatic>()
                                [0]; // host side: __half_as_ushort is
                                     // __device__ only
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer &L, int T) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                if (!vx_) {
                    CK(DPCT_CHECK_ERROR(vx_ = (float *)checked_device_alloc(
                                            (size_t)8 * 2048 * 4, q_ct1)));
                    CK(DPCT_CHECK_ERROR(vh_ = (float *)checked_device_alloc(
                                            (size_t)8 * 2048 * 4, q_ct1)));
                }
                /*
                DPCT1124:80: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                // Finish the verification snapshot before live buffers can be updated.
                CK(DPCT_CHECK_ERROR(st->memcpy(vx_, x, (size_t)T * N * 4).wait_and_throw()));
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1101:163: 'kThreads' expression was replaced with a
                    value. Modify the code to use the original expression,
                    provided in comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> sh_acc_ct1(
                        sycl::range<1>(256 /*kThreads*/), cgh);

                    auto vx__ct0 = vx_;
                    const float *y_ct1 = y;
                    auto vh__ct3 = vh_;
                    auto m_eps_ct5 = m->eps;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                              sycl::range<3>(1, 1, kThreads),
                                          sycl::range<3>(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_add_rms(vx__ct0, y_ct1, L.post_norm, vh__ct3, N,
                                      m_eps_ct5,
                                      sh_acc_ct1
                                          .get_multi_ptr<
                                              sycl::access::decorated::no>()
                                          .get());
                        });
                });
            }
            {
                dpct::has_capability_or_fail(
                    ((sycl::queue *)(st))->get_device(), {sycl::aspect::fp16});

                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<float, 1> sh_acc_ct1(
                        sycl::range<1>(256), cgh);

                    auto x_ct0 = x;
                    const float *y_ct1 = y;
                    auto h_ct3 = h;
                    auto xq_ct4 = (q36k::QB *)xq;
                    auto m_eps_ct6 = m->eps;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                              sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                q36k::k_add_rms_q(
                                    x_ct0, y_ct1, L.post_norm, h_ct3, xq_ct4, N,
                                    m_eps_ct6,
                                    sh_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                            });
                });
            }
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:164: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> sh_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            auto x_ct0 = x;
            const float *y_ct1 = y;
            auto h_ct3 = h;
            auto m_eps_ct5 = m->eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_add_rms(
                        x_ct0, y_ct1, L.post_norm, h_ct3, N, m_eps_ct5,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer &L, int T) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fast_k || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0 ||
            (K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled()))
            return false;
        const int nb = a.n_in / 32;
        const dpct::dim3 grid((unsigned)(FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const sycl::half *da = nullptr;
        const sycl::half *db = nullptr;
/*
DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
/*
DPCT1049:32: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define Q36_SH(NCV)                                                            \
    case NCV:                                                                  \
        if (pkd)                                                               \
        {                                                                      \
            dpct::has_capability_or_fail(((sycl::queue *)(st))->get_device(),  \
                                         {sycl::aspect::fp16});                \
                                                                               \
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {            \
                sycl::local_accessor<float[NCV][32], 0> g_acc_ct1(cgh);        \
                sycl::local_accessor<float[NCV][32], 0> u_acc_ct1(cgh);        \
                                                                               \
                auto wa_ct0 = wa;                                              \
                auto da_ct1 = da;                                              \
                auto wb_ct2 = wb;                                              \
                auto db_ct3 = db;                                              \
                auto xq_ct4 = (const q36k::QB *)xq;                            \
                auto q8_ct5 = (q36k::QB *)q8;                                  \
                auto nb_ct6 = nb;                                              \
                auto FS_ct7 = FS;                                              \
                                                                               \
                cgh.parallel_for(                                              \
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 1024),       \
                                      sycl::range<3>(1, 1, 1024)),             \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            q36k::k_sh_gu<NCV, 2, true>(                       \
                                wa_ct0, da_ct1, wb_ct2, db_ct3, xq_ct4,        \
                                q8_ct5, nb_ct6, FS_ct7, g_acc_ct1, u_acc_ct1); \
                        });                                                    \
            });                                                                \
        } else {                                                               \
            dpct::has_capability_or_fail(((sycl::queue *)(st))->get_device(),  \
                                         {sycl::aspect::fp16});                \
                                                                               \
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {            \
                sycl::local_accessor<float[NCV][32], 0> g_acc_ct1(cgh);        \
                sycl::local_accessor<float[NCV][32], 0> u_acc_ct1(cgh);        \
                                                                               \
                auto wa_ct0 = wa;                                              \
                const sycl::half *nullptr_ct1 = nullptr;                       \
                auto wb_ct2 = wb;                                              \
                const sycl::half *nullptr_ct3 = nullptr;                       \
                auto xq_ct4 = (const q36k::QB *)xq;                            \
                auto q8_ct5 = (q36k::QB *)q8;                                  \
                auto nb_ct6 = nb;                                              \
                auto FS_ct7 = FS;                                              \
                                                                               \
                cgh.parallel_for(                                              \
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 1024),       \
                                      sycl::range<3>(1, 1, 1024)),             \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            q36k::k_sh_gu<NCV, 2, false>(                      \
                                wa_ct0, nullptr_ct1, wb_ct2, nullptr_ct3,      \
                                xq_ct4, q8_ct5, nb_ct6, FS_ct7, g_acc_ct1,     \
                                u_acc_ct1);                                    \
                        });                                                    \
            });                                                                \
        }                                                                      \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        if (verify_k) {
            if (!vbuf) CK(DPCT_CHECK_ERROR(vbuf = (float *)checked_device_alloc(
                                               (size_t)8 * 16384 * 4, q_ct1)));
            if (!vq) CK(DPCT_CHECK_ERROR(
                vq = (void *)checked_device_alloc(
                    K::native_q8_1_bytes(m->g.n_embd, 1) * 8, q_ct1)));
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> cnt_acc_ct1(sycl::range<1>(512),
                                                         cgh);
                sycl::local_accessor<int, 1> off_acc_ct1(sycl::range<1>(512),
                                                         cgh);

                const int32_t *ids_ct0 = ids;
                auto m_g_n_expert_ct3 = (int)m->g.n_expert;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_group_hy(
                            ids_ct0, n_ent, Kp, m_g_n_expert_ct3, slot_of,
                            L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                            o_tok,
                            cnt_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            off_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        else
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> cnt_acc_ct1(sycl::range<1>(512),
                                                         cgh);
                sycl::local_accessor<int, 1> off_acc_ct1(sycl::range<1>(512),
                                                         cgh);

                const int32_t *ids_ct0 = ids;
                auto m_g_n_expert_ct3 = (int)m->g.n_expert;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_group(
                            ids_ct0, n_ent, Kp, m_g_n_expert_ct3, L.blobs,
                            L.stride, o_ptr, o_start, o_ng, o_dst, o_tok,
                            cnt_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            off_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer &L, const int32_t *slot_of, int T) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        if (!vgrp) CK(
            DPCT_CHECK_ERROR(vgrp = (void *)checked_device_alloc(2048, q_ct1)));
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        int nn = 0, no = 0;
        CK(DPCT_CHECK_ERROR(q_ct1.memcpy(&nn, n_groups, 4).wait_and_throw()));
        CK(DPCT_CHECK_ERROR(q_ct1.memcpy(&no, o_ng, 4).wait_and_throw()));
        ++v_calls;
        bool bad = nn != no;
        if (!bad) {
            std::vector<unsigned long long> a((size_t) nn + 1), b((size_t) nn + 1);
            std::vector<int32_t> sa((size_t) nn + 1), sb((size_t) nn + 1), da(64), db(64), ta(64), tb(64);
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(a.data(), grp_ptr, (size_t)nn * 8).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(b.data(), o_ptr, (size_t)nn * 8).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(sa.data(), grp_start, ((size_t)nn + 1) * 4)
                    .wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(sb.data(), o_start, ((size_t)nn + 1) * 4).wait_and_throw()));
            const int tot = sa[(size_t) nn];
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(da.data(), ent_dst, (size_t)tot * 4).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(db.data(), o_dst, (size_t)tot * 4).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(ta.data(), ent_tok, (size_t)tot * 4).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(tb.data(), o_tok, (size_t)tot * 4).wait_and_throw()));
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer &L, const int32_t *slot_of, int T,
                     int ld) try {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_k || T > 8 || m->g.n_expert != 256 || Kp != 8) return false;
        if (slot_of)
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> sids_acc_ct1(sycl::range<1>(64),
                                                          cgh);
                sycl::local_accessor<int, 1> sid_acc_ct1(sycl::range<1>(64),
                                                         cgh);

                const float *logits_r_ct0 = logits_r;
                auto ids_ct1 = ids;
                auto ewts_ct2 = ewts;
                auto grp_ptr_ct8 = grp_ptr;
                auto grp_start_ct9 = grp_start;
                auto n_groups_ct10 = n_groups;
                auto ent_dst_ct11 = ent_dst;
                auto ent_tok_ct12 = ent_tok;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 8, 32),
                                      sycl::range<3>(1, 8, 32)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        q36k::k_route_group<true>(
                            logits_r_ct0, ids_ct1, ewts_ct2, T, ld, slot_of,
                            L.blobs, L.stride, grp_ptr_ct8, grp_start_ct9,
                            n_groups_ct10, ent_dst_ct11, ent_tok_ct12,
                            sids_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sid_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        else
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> sids_acc_ct1(sycl::range<1>(64),
                                                          cgh);
                sycl::local_accessor<int, 1> sid_acc_ct1(sycl::range<1>(64),
                                                         cgh);

                const float *logits_r_ct0 = logits_r;
                auto ids_ct1 = ids;
                auto ewts_ct2 = ewts;
                auto grp_ptr_ct8 = grp_ptr;
                auto grp_start_ct9 = grp_start;
                auto n_groups_ct10 = n_groups;
                auto ent_dst_ct11 = ent_dst;
                auto ent_tok_ct12 = ent_tok;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 8, 32),
                                      sycl::range<3>(1, 8, 32)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        q36k::k_route_group<false>(
                            logits_r_ct0, ids_ct1, ewts_ct2, T, ld, nullptr,
                            L.blobs, L.stride, grp_ptr_ct8, grp_start_ct9,
                            n_groups_ct10, ent_dst_ct11, ent_tok_ct12,
                            sids_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sid_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        grouped_by_route = true;
        if (verify_k) {
            // the old top-k into scratch, the ids / weights compared, then the old grouping of the new ids
            if (!vbuf) CK(DPCT_CHECK_ERROR(
                vbuf = (float *)checked_device_alloc(
                    (size_t)8 * 16384 * 4, dpct::get_in_order_queue())));
            int32_t* vid = (int32_t*) (vbuf + 8 * 8192);
            float* vw = vbuf + 8 * 8192 + 64;
            K::native_router_topk(logits_r, vid, vw, 256, Kp, T, st);
            verify_cmp("route ids", (const float*) ids, (const float*) vid, (size_t) T * Kp, " T " + std::to_string(T));
            verify_cmp("route weights", ewts, vw, (size_t) T * Kp, " T " + std::to_string(T));
            group_verify(L, slot_of, T);
        }
        return true;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<int, 1> sid_acc_ct1(sycl::range<1>(64),
                                                             cgh);

                    const int32_t *ids_ct0 = ids;
                    auto grp_ptr_ct6 = grp_ptr;
                    auto grp_start_ct7 = grp_start;
                    auto n_groups_ct8 = n_groups;
                    auto ent_dst_ct9 = ent_dst;
                    auto ent_tok_ct10 = ent_tok;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, 64),
                                          sycl::range<3>(1, 1, 64)),
                        [=](sycl::nd_item<3> item_ct1) {
                            q36k::k_group_small<true>(
                                ids_ct0, n_ent, Kp, slot_of, L.blobs, L.stride,
                                grp_ptr_ct6, grp_start_ct7, n_groups_ct8,
                                ent_dst_ct9, ent_tok_ct10,
                                sid_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                        });
                });
            else
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<int, 1> sid_acc_ct1(sycl::range<1>(64),
                                                             cgh);

                    const int32_t *ids_ct0 = ids;
                    auto grp_ptr_ct6 = grp_ptr;
                    auto grp_start_ct7 = grp_start;
                    auto n_groups_ct8 = n_groups;
                    auto ent_dst_ct9 = ent_dst;
                    auto ent_tok_ct10 = ent_tok;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, 64),
                                          sycl::range<3>(1, 1, 64)),
                        [=](sycl::nd_item<3> item_ct1) {
                            q36k::k_group_small<false>(
                                ids_ct0, n_ent, Kp, nullptr, L.blobs, L.stride,
                                grp_ptr_ct6, grp_start_ct7, n_groups_ct8,
                                ent_dst_ct9, ent_tok_ct10,
                                sid_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                        });
                });
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t *prow, const float *shared,
                 const float *sgate, const float *wn, float *hout,
                 int sgs = 1) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            if (!vx_) {
                CK(DPCT_CHECK_ERROR(vx_ = (float *)checked_device_alloc(
                                        (size_t)8 * 2048 * 4, q_ct1)));
                CK(DPCT_CHECK_ERROR(vh_ = (float *)checked_device_alloc(
                                        (size_t)8 * 2048 * 4, q_ct1)));
            }
            /*
            DPCT1124:81: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            // Finish the verification snapshot before live buffers can be updated.
                CK(DPCT_CHECK_ERROR(st->memcpy(vx_, x, (size_t)T * N * 4).wait_and_throw()));
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:165: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> sh_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);
                sycl::local_accessor<const float *, 1> pk_acc_ct1(
                    sycl::range<1>(16), cgh);

                auto vx__ct0 = vx_;
                const float *eparts_ct1 = eparts;
                const float *ewts_ct3 = ewts;
                auto vh__ct9 = vh_;
                auto m_eps_ct11 = m->eps;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_moe_out(
                            vx__ct0, eparts_ct1, prow, ewts_ct3, Kp, shared,
                            sgate, sgs, wn, vh__ct9, N, m_eps_ct11,
                            sh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            pk_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            /*
            DPCT1049:33: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                dpct::has_capability_or_fail(
                    ((sycl::queue *)(st))->get_device(), {sycl::aspect::fp16});

                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<float, 1> vv_acc_ct1(
                        sycl::range<1>(2048), cgh);
                    sycl::local_accessor<float, 1> sh_acc_ct1(
                        sycl::range<1>(256), cgh);
                    sycl::local_accessor<const float *, 1> pk_acc_ct1(
                        sycl::range<1>(16), cgh);

                    auto x_ct0 = x;
                    const float *eparts_ct1 = eparts;
                    const float *ewts_ct3 = ewts;
                    auto m_eps_ct11 = m->eps;
                    auto mkq_q36k_QB_xq_nullptr_ct12 =
                        mkq ? (q36k::QB *)xq : nullptr;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                              sycl::range<3>(1, 1, 1024),
                                          sycl::range<3>(1, 1, 1024)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                q36k::k_moe_out_small(
                                    x_ct0, eparts_ct1, prow, ewts_ct3, Kp,
                                    shared, sgate, sgs, wn, hout, N, m_eps_ct11,
                                    vv_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    sh_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    pk_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    mkq_q36k_QB_xq_nullptr_ct12);
                            });
                });
            }
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:166: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> sh_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);
            sycl::local_accessor<const float *, 1> pk_acc_ct1(
                sycl::range<1>(16), cgh);

            auto x_ct0 = x;
            const float *eparts_ct1 = eparts;
            const float *ewts_ct3 = ewts;
            auto m_eps_ct11 = m->eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_moe_out(
                        x_ct0, eparts_ct1, prow, ewts_ct3, Kp, shared, sgate,
                        sgs, wn, hout, N, m_eps_ct11,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        pk_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
    void layer(const Layer &L, size_t li, int T, int64_t p0,
               const int32_t *dpos, bool graph_mode, bool verify,
               const float *wn, float *hout) try {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    const float *h_ct2 = h;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, T, rows8(2 * HV)) *
                                              sycl::range<3>(1, 1, kThreads),
                                          sycl::range<3>(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_gdn_ab(L.alpha, L.beta, h_ct2, L.dt, L.ssm_a,
                                         al_b, be_b, N, HV);
                            });
                });
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                ((sycl::queue *)(st))
                    ->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, grid1((int64_t)HV * T)) *
                                sycl::range<3>(1, 1, kThreads),
                            sycl::range<3>(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_gdn_gb(al_b, L.dt, L.ssm_a, be_b, HV, T);
                        });
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<float, 1> part_acc_ct1(
                        sycl::range<1>(4), cgh);

                    auto HK_ct5 = 2 * HK;
                    auto m_eps_ct6 = m->eps;
                    auto verify_ct8 = !verify;
                    float *const *d_conv_tab_li_n_sess_ct9 =
                        d_conv_tab + li * n_sess;
                    const int32_t *d_gslot_ct10 = d_gslot;
                    const int32_t *d_grow0_ct11 = d_grow0;
                    const int32_t *d_gcnt_ct12 = d_gcnt;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, rows_G, C / 128) *
                                              sycl::range<3>(1, 1, 128),
                                          sycl::range<3>(1, 1, 128)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_conv_seq(
                                    nullptr, qkv_b, L.conv, hc_b, C, HK_ct5,
                                    m_eps_ct6, 1, verify_ct8,
                                    part_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    d_conv_tab_li_n_sess_ct9, d_gslot_ct10,
                                    d_grow0_ct11, d_gcnt_ct12);
                            });
                });
                /*
                DPCT1049:34: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1101:167: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> sk_acc_ct1(
                        sycl::range<1>(128 /*S*/), cgh);
                    /*
                    DPCT1101:168: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> sq_acc_ct1(
                        sycl::range<1>(128 /*S*/), cgh);
                    /*
                    DPCT1101:169: 'RG' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    /*
                    DPCT1101:170: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                        red_kv_acc_ct1(cgh);
                    /*
                    DPCT1101:171: 'RG' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    /*
                    DPCT1101:172: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                        red_o_acc_ct1(cgh);
                    /*
                    DPCT1101:173: 'S * RG / 32' expression was replaced with a
                    value. Modify the code to use the original expression,
                    provided in comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> wsum_acc_ct1(
                        sycl::range<1>(16 /*S * RG / 32*/), cgh);

                    const float *z_ct5 = z;
                    auto m_eps_ct7 = m->eps;
                    auto gdn_y_ct8 = gdn_y;
                    auto verify_ct12 = !verify;
                    float *const *d_S_tab_li_n_sess_ct13 =
                        d_S_tab + li * n_sess;
                    const int32_t *d_gslot_ct14 = d_gslot;
                    const int32_t *d_grow0_ct15 = d_grow0;
                    const int32_t *d_gcnt_ct16 = d_gcnt;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, rows_G, HV) *
                                              sycl::range<3>(1, 4, 128),
                                          sycl::range<3>(1, 4, 128)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_gdn_seq(nullptr, hc_b, C, al_b, be_b, z_ct5,
                                          L.ssm_norm, m_eps_ct7, gdn_y_ct8, HK,
                                          HV, 1, verify_ct12,
                                          sk_acc_ct1
                                              .get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sq_acc_ct1
                                              .get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          red_kv_acc_ct1, red_o_acc_ct1,
                                          wsum_acc_ct1
                                              .get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          d_S_tab_li_n_sess_ct13, d_gslot_ct14,
                                          d_grow0_ct15, d_gcnt_ct16);
                            });
                });
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<float, 1> part_acc_ct1(
                        sycl::range<1>(4), cgh);

                    auto conv_li_ct0 = conv[li];
                    auto HK_ct5 = 2 * HK;
                    auto m_eps_ct6 = m->eps;
                    auto verify_ct8 = !verify;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, tiles, C / 128) *
                                              sycl::range<3>(1, 1, 128),
                                          sycl::range<3>(1, 1, 128)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_conv_seq(
                                    conv_li_ct0, qkv_b, L.conv, hc_b, C, HK_ct5,
                                    m_eps_ct6, T, verify_ct8,
                                    part_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    nullptr, nullptr, nullptr, nullptr);
                            });
                });
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    /*
                    DPCT1049:35: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:174: 'S' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[2][128 /*S*/], 0> sk_acc_ct1(
                            cgh);
                        /*
                        DPCT1101:175: 'S' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[2][128 /*S*/], 0> sq_acc_ct1(
                            cgh);
                        /*
                        DPCT1101:176: 'RG' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        /*
                        DPCT1101:177: 'NC' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[16 /*RG*/][32 /*NC*/], 0>
                            red_kv_acc_ct1(cgh);
                        /*
                        DPCT1101:178: 'RG' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        /*
                        DPCT1101:179: 'NC' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[16 /*RG*/][32 /*NC*/], 0>
                            red_o_acc_ct1(cgh);

                        auto S_li_ct0 = S[li];
                        auto gdn_y_ct5 = gdn_y;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, kGdnColGroups, HV) *
                                    sycl::range<3>(1, 16, 128 / kGdnColGroups),
                                sycl::range<3>(1, 16, 128 / kGdnColGroups)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_gdn_seq_cols(S_li_ct0, hc_b, C, al_b, be_b,
                                               gdn_y_ct5, HK, HV, T, sk_acc_ct1,
                                               sq_acc_ct1, red_kv_acc_ct1,
                                               red_o_acc_ct1);
                            });
                    });
                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<float, 1> part_acc_ct1(
                            sycl::range<1>(4), cgh);

                        const float *gdn_y_ct0 = gdn_y;
                        const float *z_ct1 = z;
                        auto m_eps_ct3 = m->eps;
                        auto gdn_y_ct4 = gdn_y;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, T, HV) *
                                                  sycl::range<3>(1, 1, 128),
                                              sycl::range<3>(1, 1, 128)),
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    k_gdn_norm(
                                        gdn_y_ct0, z_ct1, L.ssm_norm, m_eps_ct3,
                                        gdn_y_ct4, HV,
                                        part_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                                });
                    });
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    /*
                    DPCT1049:36: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:180: 'S' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float, 1> sk_acc_ct1(
                            sycl::range<1>(128 /*S*/), cgh);
                        /*
                        DPCT1101:181: 'S' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float, 1> sq_acc_ct1(
                            sycl::range<1>(128 /*S*/), cgh);
                        /*
                        DPCT1101:182: 'RG' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        /*
                        DPCT1101:183: 'S' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                            red_kv_acc_ct1(cgh);
                        /*
                        DPCT1101:184: 'RG' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        /*
                        DPCT1101:185: 'S' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                            red_o_acc_ct1(cgh);
                        /*
                        DPCT1101:186: 'S * RG / 32' expression was replaced with
                        a value. Modify the code to use the original expression,
                        provided in comments, if it is correct.
                        */
                        sycl::local_accessor<float, 1> wsum_acc_ct1(
                            sycl::range<1>(16 /*S * RG / 32*/), cgh);

                        auto S_li_ct0 = S[li];
                        const float *z_ct5 = z;
                        auto m_eps_ct7 = m->eps;
                        auto gdn_y_ct8 = gdn_y;
                        auto verify_ct12 = !verify;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, HV) *
                                                  sycl::range<3>(1, 4, 128),
                                              sycl::range<3>(1, 4, 128)),
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    k_gdn_seq(
                                        S_li_ct0, hc_b, C, al_b, be_b, z_ct5,
                                        L.ssm_norm, m_eps_ct7, gdn_y_ct8, HK,
                                        HV, T, verify_ct12,
                                        sk_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get(),
                                        sq_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get(),
                                        red_kv_acc_ct1, red_o_acc_ct1,
                                        wsum_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get(),
                                        nullptr, nullptr, nullptr, nullptr);
                                });
                    });
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
            }
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:187: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> sh_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);
                sycl::local_accessor<float, 1> buf_acc_ct1(sycl::range<1>(256),
                                                           cgh);

                const float *qg_ct0 = qg;
                const float *kx_ct1 = kx;
                const float *vx_ct2 = vx;
                auto q_ct5 = q;
                auto gate_ct6 = gate;
                auto rows_mode_nullptr_kc_li_ct7 = rows_mode ? nullptr : kc[li];
                auto rows_mode_nullptr_vc_li_ct8 = rows_mode ? nullptr : vc[li];
                auto m_n_rot_ct12 = m->n_rot;
                auto ctx_ct14 = ctx;
                auto m_rope_base_ct15 = m->rope_base;
                auto m_eps_ct16 = m->eps;
                const int32_t *d_delta_ct17 = d_delta;
                auto mtab_now_ct18 = mtab_now;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, T, H + HKV) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_attn_prep(
                            qg_ct0, kx_ct1, vx_ct2, L.q_norm, L.k_norm, q_ct5,
                            gate_ct6, rows_mode_nullptr_kc_li_ct7,
                            rows_mode_nullptr_vc_li_ct8, H, HKV, HD,
                            m_n_rot_ct12, dpos, ctx_ct14, m_rope_base_ct15,
                            m_eps_ct16, d_delta_ct17, mtab_now_ct18, rs,
                            sh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            buf_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
            if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                {
                    dpct::has_capability_or_fail(
                        ((sycl::queue *)(st))->get_device(),
                        {sycl::aspect::fp16});

                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        const float *kc_li_ct0 = kc[li];
                        const float *vc_li_ct1 = vc[li];
                        auto k16_ct2 = k16;
                        auto v16_ct3 = v16;
                        auto ctx_ct4 = ctx;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1,
                                               grid1((int64_t)HKV * P * HD)) *
                                    sycl::range<3>(1, 1, kThreads),
                                sycl::range<3>(1, 1, kThreads)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_kv_f16(kc_li_ct0, vc_li_ct1, k16_ct2, v16_ct3,
                                         ctx_ct4, P, HD, HKV);
                            });
                    });
                }
                {
                    dpct::has_capability_or_fail(
                        ((sycl::queue *)(st))->get_device(),
                        {sycl::aspect::fp16});

                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        const float *q_ct0 = q;
                        auto q16_ct1 = q16;
                        auto int64_t_T_H_HD_ct2 = (int64_t)T * H * HD;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1,
                                               grid1((int64_t)T * H * HD)) *
                                    sycl::range<3>(1, 1, kThreads),
                                sycl::range<3>(1, 1, kThreads)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_to_f16(q_ct0, q16_ct1, int64_t_T_H_HD_ct2);
                            });
                    });
                }
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const sycl::half *Kh = k16 + (size_t)kvh * P * HD;
                        const sycl::half *Qh =
                            q16 + ((size_t)t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (DPCT_CHECK_ERROR(dpct::blas::gemm_batch(
                                blas, oneapi::mkl::transpose::trans,
                                oneapi::mkl::transpose::nontrans, P, nt, HD,
                                &one, Kh, dpct::library_data_t::real_half, HD,
                                0, Qh, dpct::library_data_t::real_half, H * HD,
                                HD, &zero, Sh, dpct::library_data_t::real_float,
                                ldp, (long long)nt * ldp, G,
                                dpct::compute_type::f32)) != 0) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    {
                        dpct::has_capability_or_fail(
                            ((sycl::queue *)(st))->get_device(),
                            {sycl::aspect::fp16});

                        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                            /*
                            DPCT1101:188: 'kThreads' expression was replaced
                            with a value. Modify the code to use the original
                            expression, provided in comments, if it is correct.
                            */
                            sycl::local_accessor<float, 1> sh_acc_ct1(
                                sycl::range<1>(256 /*kThreads*/), cgh);

                            const float *att_s_ct0 = att_s;
                            auto att_p16_ct1 = att_p16;
                            auto int_p0_t0_ct5 = (int)p0 + t0;
                            auto sqrtf_float_HD_ct6 = 1.0f / sqrtf((float)HD);

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(1, H, nt) *
                                        sycl::range<3>(1, 1, kThreads),
                                    sycl::range<3>(1, 1, kThreads)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    k_attn_softmax16(
                                        att_s_ct0, att_p16_ct1, ldp, nt, P,
                                        int_p0_t0_ct5, sqrtf_float_HD_ct6,
                                        sh_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                                });
                        });
                    }
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const sycl::half *Vh = v16 + (size_t)kvh * P * HD;
                        const sycl::half *Sh =
                            att_p16 + (size_t)kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        q36_attention_values_gemm(blas, Vh, Sh, Oh, HD, nt, P, ldp, H, G);
                    }
                }
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    auto att_ct0 = att;
                    const float *gate_ct1 = gate;
                    auto int64_t_T_H_HD_ct2 = (int64_t)T * H * HD;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, grid1((int64_t)T * H * HD)) *
                                sycl::range<3>(1, 1, kThreads),
                            sycl::range<3>(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_gate_mul(att_ct0, gate_ct1, int64_t_T_H_HD_ct2);
                        });
                });
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:189: 'HD' expression was replaced with a value.
                        Modify the code to use the original expression, provided
                        in comments, if it is correct.
                        */
                        sycl::local_accessor<float, 1> qs_acc_ct1(
                            sycl::range<1>(256 /*HD*/), cgh);
                        /*
                        DPCT1101:190: 'kChunkCells' expression was replaced with
                        a value. Modify the code to use the original expression,
                        provided in comments, if it is correct.
                        */
                        sycl::local_accessor<float, 1> sc_acc_ct1(
                            sycl::range<1>(256 /*kChunkCells*/), cgh);
                        /*
                        DPCT1101:191: 'kThreads' expression was replaced with a
                        value. Modify the code to use the original expression,
                        provided in comments, if it is correct.
                        */
                        sycl::local_accessor<float, 1> sh_acc_ct1(
                            sycl::range<1>(256 /*kThreads*/), cgh);

                        const float *q_ct0 = q;
                        const float *rows_mode_nullptr_kc_li_ct1 =
                            rows_mode ? nullptr : kc[li];
                        const float *rows_mode_nullptr_vc_li_ct2 =
                            rows_mode ? nullptr : vc[li];
                        auto ctx_ct7 = ctx;
                        auto sqrtf_float_HD_ct8 = 1.0f / sqrtf((float)HD);
                        auto n_chunk_ct9 = n_chunk;
                        auto pacc_ct10 = pacc;
                        auto pm_ct11 = pm;
                        auto pl_ct12 = pl;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, nt * H, nc) *
                                    sycl::range<3>(1, 1, kChunkCells),
                                sycl::range<3>(1, 1, kChunkCells)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_attn_part(
                                    q_ct0, rows_mode_nullptr_kc_li_ct1,
                                    rows_mode_nullptr_vc_li_ct2, dpos, t0, H,
                                    HKV, ctx_ct7, sqrtf_float_HD_ct8,
                                    n_chunk_ct9, pacc_ct10, pm_ct11, pl_ct12,
                                    rs,
                                    qs_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    sc_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    sh_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                            });
                    });
                    /*
                    DPCT1049:37: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                    ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                        const float *pacc_ct0 = pacc;
                        const float *pm_ct1 = pm;
                        const float *pl_ct2 = pl;
                        const float *gate_ct3 = gate;
                        auto n_chunk_ct7 = n_chunk;
                        auto att_ct8 = att;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, nt * H) *
                                                  sycl::range<3>(1, 1, HD),
                                              sycl::range<3>(1, 1, HD)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_attn_merge(pacc_ct0, pm_ct1, pl_ct2, gate_ct3,
                                             dpos, t0, H, n_chunk_ct7, att_ct8,
                                             rs);
                            });
                    });
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
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(DPCT_CHECK_ERROR(st->memset(rcnt, 0, (size_t)ne * 4)));
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                const int32_t *ids_ct0 = ids;
                auto rcnt_ct2 = rcnt;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid1(n_ent)) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_route_count(ids_ct0, (int)n_ent, rcnt_ct2);
                    });
            });
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                const int32_t *rcnt_ct0 = rcnt;
                auto rbounds_ct2 = rbounds;
                auto rcur_ct3 = rcur;
                auto rmax_ct4 = rmax;

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 32),
                                                   sycl::range<3>(1, 1, 32)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_route_scan(rcnt_ct0, ne, rbounds_ct2,
                                                  rcur_ct3, rmax_ct4);
                                 });
            });
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                const int32_t *ids_ct0 = ids;
                auto rcur_ct3 = rcur;
                auto ent_dst_ct4 = ent_dst;
                auto ent_tok_ct5 = ent_tok;
                auto rinv_ct6 = rinv;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid1(n_ent)) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_route_scatter(ids_ct0, (int)n_ent, Kp, rcur_ct3,
                                        ent_dst_ct4, ent_tok_ct5, rinv_ct6);
                    });
            });
            /*
            DPCT1124:82: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(hpin + 10, rmax, 4)));
            /*
            DPCT1124:83: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            if (lhy(li)) CK(DPCT_CHECK_ERROR(
                st->memcpy(h_cnt + li * g.n_expert, rcnt, (size_t)ne * 4)));
            CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
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
        // the shared expert, scaled by sigmoid(w . h)
        if (T > kGemmT) {
            dpct::has_capability_or_fail(((sycl::queue *)(st))->get_device(),
                                         {sycl::aspect::fp16});

            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                const float *h_ct0 = h;
                auto x16_ct1 = x16;
                auto int64_t_N_T_ct2 = (int64_t)N * T;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, grid1((int64_t)N * T)) *
                            sycl::range<3>(1, 1, kThreads),
                        sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_to_f16(h_ct0, x16_ct1, int64_t_N_T_ct2);
                    });
            });
        }
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
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                const float *sg_ct0 = sg;
                const float *su_ct1 = su;
                auto shh_ct2 = shh;
                auto int64_t_T_FS_ct3 = (int64_t)T * FS;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, grid1((int64_t)T * FS)) *
                            sycl::range<3>(1, 1, kThreads),
                        sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_swiglu(sg_ct0, su_ct1, shh_ct2, int64_t_T_FS_ct3);
                    });
            });
            mm(L.sh_down, shh, sy, T);
        }
        if (!ext) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
            if (map_on)((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                auto x_ct0 = x;
                const float *d_img_ct1 = d_img;
                const int32_t *d_map_ct2 = d_map;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_put_rows(x_ct0, d_img_ct1, d_map_ct2, N);
                    });
            }); // a picture's cells take its rows
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:192: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> sh_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            const float *x_ct0 = x;
            auto m_layers_attn_norm_ct1 = m->layers[0].attn_norm;
            auto h_ct2 = h;
            auto m_eps_ct4 = m->eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_rms(
                        x_ct0, m_layers_attn_norm_ct1, h_ct2, N, m_eps_ct4,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        }); // later layers: from k_moe_out
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
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:193: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> bv_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);
            /*
            DPCT1101:194: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<int, 1> bi_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            auto amv_ct3 = amv;
            auto ami_ct4 = ami;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, n, kArgBlocks) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argmax(
                        lg, (int)g.n_vocab, pk, amv_ct3, ami_ct4, false,
                        bv_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        bi_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:195: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> bv_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);
            /*
            DPCT1101:196: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<int, 1> bi_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            auto kArgBlocks_ct1 = kArgBlocks;
            auto amv_ct3 = amv;
            auto ami_ct4 = ami;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, n, 1) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argmax(
                        lg, kArgBlocks_ct1, pk, amv_ct3, ami_ct4, true,
                        bv_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        bi_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (!fast_k || draft_n <= 0 || draft_n >= g.n_vocab || draft_n % 8 != 0 || hd.type != 8 || hd.n_in != N || N / 32 != 64) {
            head_rows(hn, 1, logits_m, pick_m);
            return;
        }
        quant(hn, N, 1, q8);
        const uint8_t* pk = nullptr;
        for (const auto& p : m->packed)
            if (std::get<0>(p) == hd.d) pk = std::get<1>(p);
        if (pk) {
            const sycl::half *dpl = reinterpret_cast<const sycl::half *>(
                pk + (size_t)hd.n_in * hd.n_out);
            q36k::launch_q8_rows<1, 2, true>(pk, dpl, q8, logits_m, 64, draft_n, st);
        } else {
            q36k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        }
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:197: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> bv_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);
            /*
            DPCT1101:198: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<int, 1> bi_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            const float *logits_m_ct0 = logits_m;
            auto draft_n_ct1 = draft_n;
            auto pick_m_ct2 = pick_m;
            auto amv_ct3 = amv;
            auto ami_ct4 = ami;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, kArgBlocks) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argmax(
                        logits_m_ct0, draft_n_ct1, pick_m_ct2, amv_ct3, ami_ct4,
                        false,
                        bv_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        bi_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:199: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> bv_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);
            /*
            DPCT1101:200: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<int, 1> bi_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            const float *logits_m_ct0 = logits_m;
            auto kArgBlocks_ct1 = kArgBlocks;
            auto pick_m_ct2 = pick_m;
            auto amv_ct3 = amv;
            auto ami_ct4 = ami;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argmax(
                        logits_m_ct0, kArgBlocks_ct1, pick_m_ct2, amv_ct3,
                        ami_ct4, true,
                        bv_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        bi_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            if (map_on)((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                auto y_ct0 = y;
                const float *d_img_ct1 = d_img;
                const int32_t *d_map_ct2 = d_map;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_put_rows(y_ct0, d_img_ct1, d_map_ct2, N);
                    });
            });
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:201: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> sh_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            const float *y_ct0 = y;
            auto m_mtp_enorm_ct2 = m->mtp_enorm;
            auto mtp_cat_ct3 = mtp_cat;
            auto N_ct4 = 2 * N;
            auto m_eps_ct6 = m->eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_rms_ld(
                        y_ct0, N, m_mtp_enorm_ct2, mtp_cat_ct3, N_ct4, N,
                        m_eps_ct6,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        }); // [e | ...]
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:202: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> sh_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            auto m_mtp_hnorm_ct2 = m->mtp_hnorm;
            auto mtp_cat_N_ct3 = mtp_cat + N;
            auto N_ct4 = 2 * N;
            auto m_eps_ct6 = m->eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_rms_ld(
                        seeds, N, m_mtp_hnorm_ct2, mtp_cat_N_ct3, N_ct4, N,
                        m_eps_ct6,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        }); // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:203: 'kThreads' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<float, 1> sh_acc_ct1(
                sycl::range<1>(256 /*kThreads*/), cgh);

            const float *x_ct0 = x;
            auto m_mtp_attn_norm_ct1 = m->mtp.attn_norm;
            auto h_ct2 = h;
            auto m_eps_ct4 = m->eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_rms(
                        x_ct0, m_mtp_attn_norm_ct1, h_ct2, N, m_eps_ct4,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
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
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 1> part_acc_ct1(sycl::range<1>(4),
                                                            cgh);

                auto conv_l_ct0 = conv[l];
                const float *v_qkv_l_ct1 = v_qkv[l];
                auto hc_ct3 = hc;
                auto HK_ct5 = 2 * HK;
                auto m_eps_ct6 = m->eps;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, C / 128) *
                                          sycl::range<3>(1, 1, 128),
                                      sycl::range<3>(1, 1, 128)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        k_conv_seq(
                            conv_l_ct0, v_qkv_l_ct1, L.conv, hc_ct3, C, HK_ct5,
                            m_eps_ct6, n_keep, true,
                            part_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            nullptr, nullptr, nullptr, nullptr);
                    });
            });
            /*
            DPCT1049:38: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:204: 'S' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float, 1> sk_acc_ct1(
                    sycl::range<1>(128 /*S*/), cgh);
                /*
                DPCT1101:205: 'S' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float, 1> sq_acc_ct1(
                    sycl::range<1>(128 /*S*/), cgh);
                /*
                DPCT1101:206: 'RG' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                /*
                DPCT1101:207: 'S' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                    red_kv_acc_ct1(cgh);
                /*
                DPCT1101:208: 'RG' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                /*
                DPCT1101:209: 'S' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                    red_o_acc_ct1(cgh);
                /*
                DPCT1101:210: 'S * RG / 32' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                sycl::local_accessor<float, 1> wsum_acc_ct1(
                    sycl::range<1>(16 /*S * RG / 32*/), cgh);

                auto S_l_ct0 = S[l];
                const float *v_hc_l_ct1 = v_hc[l];
                const float *v_alpha_l_ct3 = v_alpha[l];
                const float *v_beta_l_ct4 = v_beta[l];
                const float *z_ct5 = z;
                auto m_eps_ct7 = m->eps;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, HV) *
                                          sycl::range<3>(1, 4, 128),
                                      sycl::range<3>(1, 4, 128)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        k_gdn_seq(
                            S_l_ct0, v_hc_l_ct1, C, v_alpha_l_ct3, v_beta_l_ct4,
                            z_ct5, L.ssm_norm, m_eps_ct7, nullptr, HK, HV,
                            n_keep, true,
                            sk_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            sq_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            red_kv_acc_ct1, red_o_acc_ct1,
                            wsum_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            nullptr, nullptr, nullptr, nullptr);
                    });
            });
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) try { // the i-th layer of the sequence into staging
                                  // buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(DPCT_CHECK_ERROR((cst)->ext_oneapi_submit_barrier({*ev_used[b]})));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (all || 2 * L.n_slots < ne) {
            /*
            DPCT1124:84: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(
                cst->memcpy(stage[b], L.hblobs, L.stride * (size_t)ne)));
        } else {
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    /*
                    DPCT1124:85: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    CK(DPCT_CHECK_ERROR(cst->memcpy(
                        stage[b] + (size_t)x * L.stride,
                        L.blobs + (size_t)H.slot_of[(size_t)x] * L.stride, n)));
                else
                    /*
                    DPCT1124:86: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    CK(DPCT_CHECK_ERROR(
                        cst->memcpy(stage[b] + (size_t)x * L.stride,
                                    L.hblobs + (size_t)x * L.stride, n)));
                x = y;
            }
        }
        /*
        DPCT1012:87: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:88: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        ev_copy_ct1[b] = std::chrono::steady_clock::now();
        CK(DPCT_CHECK_ERROR(*ev_copy[b] = (cst)->ext_oneapi_submit_barrier()));
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    const uint8_t *stream_take(int li) try {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        CK(DPCT_CHECK_ERROR((st)->ext_oneapi_submit_barrier({*ev_copy[b]})));
        return stage[b];
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    void stream_done() try {
        const int b = stream_pos % 2;
        /*
        DPCT1012:89: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:90: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        ev_used_ct1[b] = std::chrono::steady_clock::now();
        CK(DPCT_CHECK_ERROR(*ev_used[b] = (st)->ext_oneapi_submit_barrier()));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer &L, size_t li, int T) try {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        /*
        DPCT1124:91: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(h_ids, ids, (size_t)n_ent * 4)));
        /*
        DPCT1124:92: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(h_h, h, (size_t)T * N * 4)));
        /*
        DPCT1012:93: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:94: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        ev_route_ct1 = std::chrono::steady_clock::now();
        CK(DPCT_CHECK_ERROR(*ev_route = (st)->ext_oneapi_submit_barrier()));
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) try {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        ev_route->wait_and_throw();
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
            if (!defer) H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
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
        if (n_miss == 0) return;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        /*
        DPCT1124:95: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(ecpu, cpu_out, (size_t)n_ent * N * 4)));
        /*
        DPCT1124:96: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_miss, h_miss, (size_t)n_ent * 4)));
        ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
            auto eparts_ct0 = eparts;
            const float *ecpu_ct1 = ecpu;
            const int32_t *d_miss_ct2 = d_miss;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, n_ent) *
                                      sycl::range<3>(1, 1, kThreads),
                                  sycl::range<3>(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_merge_cpu(eparts_ct0, ecpu_ct1, d_miss_ct2, N);
                });
        });
#else
        (void) li;
        (void) T;
#endif
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top S, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int y = H.exp_in_slot[(size_t) s];
                    if (H.score[(size_t) y] >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    H.slot_of[(size_t) y] = -1;
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    /*
                    DPCT1124:97: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    CK(DPCT_CHECK_ERROR(st->memcpy(
                        L.blobs + (size_t)s * L.stride,
                        L.hblobs + (size_t)x * L.stride, L.L.bytes)));
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed)
                /*
                DPCT1124:98: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(
                    st->memcpy(H.slot_of_d, H.slot_of.data(), (size_t)ne * 4)));
            for (double& v : H.score) v *= 0.5;
        }
        CK(DPCT_CHECK_ERROR(
            st->wait_and_throw())); // slot_of's host copy is pageable: the upload is done
                          // before it changes
    }

    void shrink_att(size_t floats) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        CK(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        CK(DPCT_CHECK_ERROR(dpct::dpct_free(att_s, q_ct1)));
        CK(DPCT_CHECK_ERROR(dpct::dpct_free(att_p16, q_ct1)));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<sycl::half>(kAttScoreFloats);
        CK(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    void reset_graphs() try {
        for (GSet& gset : gsets) {
            for (dpct::experimental::command_graph_exec_ptr *g :
                 {&gset.g1, &gset.gm1})
                if (*g) { CK(DPCT_CHECK_ERROR(delete (*g))); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(DPCT_CHECK_ERROR(delete (g))); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(DPCT_CHECK_ERROR(delete (kv.second)));
        row_graphs.clear();
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <class F>
    void graph_run(dpct::experimental::command_graph_exec_ptr &gx, F body) try {
        if (!use_graph) { body(); return; }
        if (!gx) {
            dpct::experimental::command_graph_ptr graph = nullptr;
            CK(DPCT_CHECK_ERROR(dpct::experimental::begin_recording(st)));
            body();
            CK(DPCT_CHECK_ERROR(dpct::experimental::end_recording(st, &graph)));
            CK(DPCT_CHECK_ERROR(
                gx = new sycl::ext::oneapi::experimental::command_graph<
                    sycl::ext::oneapi::experimental::graph_state::executable>(
                    graph->finalize())));
            CK(DPCT_CHECK_ERROR(delete (graph)));
        }
        CK(DPCT_CHECK_ERROR((st)->ext_oneapi_graph(*gx)));
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float> &rows,
                    const std::vector<int32_t> &cell_row,
                    const std::vector<int32_t> &mrope, int64_t n_prompt,
                    int32_t delta) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        const size_t need = rows.size();
        if (need > img_cap) {
            CK(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
            if (d_img) CK(DPCT_CHECK_ERROR(dpct::dpct_free(d_img, q_ct1)));
            CK(DPCT_CHECK_ERROR(
                d_img = (float *)checked_device_alloc(need * 4, q_ct1)));
            img_cap = need;
        }
        CK(DPCT_CHECK_ERROR(q_ct1.memcpy(d_img, rows.data(), need * 4).wait_and_throw()));
        CK(DPCT_CHECK_ERROR(
            q_ct1.memcpy(d_mtab, mrope.data(), mrope.size() * 4).wait_and_throw()));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
    void chunk_pictures(int64_t c0, int T) try {
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
            /*
            DPCT1124:99: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(d_map, h_map, (size_t)T * 4)));
            map_on = true;
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
            CK(DPCT_CHECK_ERROR(
                st->wait_and_throw())); // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            /*
            DPCT1124:100: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(tok, hpin, 4)));
            /*
            DPCT1124:101: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(pos0, hpin + 1, 4)));
            chunk_pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (mtab_now || map_on) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, [&] { forward(1, pos, 1, use_graph, false); });
            mtab_now = nullptr;
            map_on = false;
            rows_deferred = false;
            /*
            DPCT1124:102: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(hlast, hmain, (size_t)N * 4)));
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
            hpin[1] = (int32_t) pos;
            /*
            DPCT1124:103: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(tok, toks + i, (size_t)T * 4)));
            /*
            DPCT1124:104: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(pos0, hpin + 1, 4)));
            chunk_pictures(pos, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                /*
                DPCT1124:105: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(
                    st->memcpy(mtp_seed(), hlast, (size_t)N * 4)));
                if (T > 1)
                    /*
                    DPCT1124:106: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    CK(DPCT_CHECK_ERROR(st->memcpy(mtp_seed() + N, hmain,
                                                   (size_t)(T - 1) * N * 4)));
                /*
                DPCT1124:107: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(
                    st->memcpy(tok_m, toks + i, (size_t)T * 4)));
                /*
                DPCT1124:108: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(st->memcpy(pos0_m, hpin + 1, 4)));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            /*
            DPCT1124:109: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(
                st->memcpy(hlast, hmain + (size_t)(T - 1) * N, (size_t)N * 4)));
            pos += T;
            CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
            mtab_now = nullptr;
            map_on = false;
            if (!on_chunk(i + T)) {
                if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
                return i + T;
            }
        }
        if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
        return n;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, dpct::experimental::command_graph_exec_ptr>
        row_graphs; // by (rows, groups, kind): graphs of the batch windows
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
        CK(DPCT_CHECK_ERROR(st->wait_and_throw())); // the pinned staging is free again
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
        /*
        DPCT1124:110: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(tok, hrow, (size_t)T * 4)));
        /*
        DPCT1124:111: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_rcell, hrow + 8, (size_t)T * 4)));
        /*
        DPCT1124:112: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_rslot, hrow + 16, (size_t)T * 4)));
        /*
        DPCT1124:113: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_rdelta, hrow + 24, (size_t)T * 4)));
        /*
        DPCT1124:114: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_gslot, hrow + 40, (size_t)G * 4)));
        /*
        DPCT1124:115: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_grow0, hrow + 48, (size_t)G * 4)));
        /*
        DPCT1124:116: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_gcnt, hrow + 56, (size_t)G * 4)));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        /*
        DPCT1124:117: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(hrow + 32, pick, (size_t)T * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % kRebalanceEvery == 0) rebalance(8);
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
        /*
        DPCT1124:118: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_gcnt, hrow + 56, (size_t)G * 4)));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<float, 1> part_acc_ct1(
                        sycl::range<1>(4), cgh);

                    const float *v_qkv_l_ct1 = v_qkv[l];
                    auto hc_ct3 = hc;
                    auto HK_ct5 = 2 * HK;
                    auto m_eps_ct6 = m->eps;
                    float *const *d_conv_tab_l_n_sess_ct9 =
                        d_conv_tab + l * n_sess;
                    const int32_t *d_gslot_ct10 = d_gslot;
                    const int32_t *d_grow0_ct11 = d_grow0;
                    const int32_t *d_gcnt_ct12 = d_gcnt;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, G, C / 128) *
                                              sycl::range<3>(1, 1, 128),
                                          sycl::range<3>(1, 1, 128)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_conv_seq(
                                    nullptr, v_qkv_l_ct1, L.conv, hc_ct3, C,
                                    HK_ct5, m_eps_ct6, 1, true,
                                    part_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    d_conv_tab_l_n_sess_ct9, d_gslot_ct10,
                                    d_grow0_ct11, d_gcnt_ct12);
                            });
                });
                /*
                DPCT1049:39: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1101:211: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> sk_acc_ct1(
                        sycl::range<1>(128 /*S*/), cgh);
                    /*
                    DPCT1101:212: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> sq_acc_ct1(
                        sycl::range<1>(128 /*S*/), cgh);
                    /*
                    DPCT1101:213: 'RG' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    /*
                    DPCT1101:214: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                        red_kv_acc_ct1(cgh);
                    /*
                    DPCT1101:215: 'RG' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    /*
                    DPCT1101:216: 'S' expression was replaced with a value.
                    Modify the code to use the original expression, provided in
                    comments, if it is correct.
                    */
                    sycl::local_accessor<float[4 /*RG*/][128 /*S*/], 0>
                        red_o_acc_ct1(cgh);
                    /*
                    DPCT1101:217: 'S * RG / 32' expression was replaced with a
                    value. Modify the code to use the original expression,
                    provided in comments, if it is correct.
                    */
                    sycl::local_accessor<float, 1> wsum_acc_ct1(
                        sycl::range<1>(16 /*S * RG / 32*/), cgh);

                    const float *v_hc_l_ct1 = v_hc[l];
                    const float *v_alpha_l_ct3 = v_alpha[l];
                    const float *v_beta_l_ct4 = v_beta[l];
                    const float *z_ct5 = z;
                    auto m_eps_ct7 = m->eps;
                    float *const *d_S_tab_l_n_sess_ct13 = d_S_tab + l * n_sess;
                    const int32_t *d_gslot_ct14 = d_gslot;
                    const int32_t *d_grow0_ct15 = d_grow0;
                    const int32_t *d_gcnt_ct16 = d_gcnt;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, G, HV) *
                                              sycl::range<3>(1, 4, 128),
                                          sycl::range<3>(1, 4, 128)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_gdn_seq(nullptr, v_hc_l_ct1, C, v_alpha_l_ct3,
                                          v_beta_l_ct4, z_ct5, L.ssm_norm,
                                          m_eps_ct7, nullptr, HK, HV, 1, true,
                                          sk_acc_ct1
                                              .get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          sq_acc_ct1
                                              .get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          red_kv_acc_ct1, red_o_acc_ct1,
                                          wsum_acc_ct1
                                              .get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get(),
                                          d_S_tab_l_n_sess_ct13, d_gslot_ct14,
                                          d_grow0_ct15, d_gcnt_ct16);
                            });
                });
            }
        });
        rows_mode = false;
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
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
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            /*
            DPCT1124:119: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(mtp_seed() + (size_t)r * N,
                                           rows[r].seed, (size_t)N * 4)));
        }
        /*
        DPCT1124:120: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(tok_m, hrow, (size_t)T * 4)));
        /*
        DPCT1124:121: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_rcell, hrow + 8, (size_t)T * 4)));
        /*
        DPCT1124:122: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_rslot, hrow + 16, (size_t)T * 4)));
        /*
        DPCT1124:123: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(d_rdelta, hrow + 24, (size_t)T * 4)));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        run_graph_key(3000 + T, [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:218: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> sh_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);

                const float *y_ct0 = y;
                auto m_mtp_enorm_ct2 = m->mtp_enorm;
                auto mtp_cat_ct3 = mtp_cat;
                auto N_ct4 = 2 * N;
                auto m_eps_ct6 = m->eps;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_rms_ld(
                            y_ct0, N, m_mtp_enorm_ct2, mtp_cat_ct3, N_ct4, N,
                            m_eps_ct6,
                            sh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            }); // [e | ...]
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:219: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> sh_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);

                const float *mtp_seed_ct0 = mtp_seed();
                auto m_mtp_hnorm_ct2 = m->mtp_hnorm;
                auto mtp_cat_N_ct3 = mtp_cat + N;
                auto N_ct4 = 2 * N;
                auto m_eps_ct6 = m->eps;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_rms_ld(
                            mtp_seed_ct0, N, m_mtp_hnorm_ct2, mtp_cat_N_ct3,
                            N_ct4, N, m_eps_ct6,
                            sh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            }); // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:220: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> sh_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);

                const float *x_ct0 = x;
                auto m_mtp_attn_norm_ct1 = m->mtp.attn_norm;
                auto h_ct2 = h;
                auto m_eps_ct4 = m->eps;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_rms(x_ct0, m_mtp_attn_norm_ct1, h_ct2, N, m_eps_ct4,
                              sh_acc_ct1
                                  .get_multi_ptr<sycl::access::decorated::no>()
                                  .get());
                    });
            });
            layer(m->mtp, m->layers.size(), T, pmax, pos0_m, use_graph, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:221: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> bv_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);
                /*
                DPCT1101:222: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<int, 1> bi_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);

                const float *logits_mr_ct0 = logits_mr;
                auto pick_mr_ct2 = pick_mr;
                auto amv_ct3 = amv;
                auto ami_ct4 = ami;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, T, kArgBlocks) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_argmax(
                            logits_mr_ct0, (int)g.n_vocab, pick_mr_ct2, amv_ct3,
                            ami_ct4, false,
                            bv_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            bi_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
            ((sycl::queue *)(st))->submit([&](sycl::handler &cgh) {
                /*
                DPCT1101:223: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float, 1> bv_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);
                /*
                DPCT1101:224: 'kThreads' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<int, 1> bi_acc_ct1(
                    sycl::range<1>(256 /*kThreads*/), cgh);

                const float *logits_mr_ct0 = logits_mr;
                auto kArgBlocks_ct1 = kArgBlocks;
                auto pick_mr_ct2 = pick_mr;
                auto amv_ct3 = amv;
                auto ami_ct4 = ami;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, T, 1) *
                                          sycl::range<3>(1, 1, kThreads),
                                      sycl::range<3>(1, 1, kThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_argmax(
                            logits_mr_ct0, kArgBlocks_ct1, pick_mr_ct2, amv_ct3,
                            ami_ct4, true,
                            bv_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            bi_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        });
        rows_deferred = false;
        rows_mode = false;
        /*
        DPCT1124:124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(hrow + 32, pick_mr, (size_t)T * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float> &out) try {
        out.resize((size_t) m->g.n_vocab);
        /*
        DPCT1124:125: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(
            out.data(), logits + (size_t)r * m->g.n_vocab, out.size() * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) try {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t row = (size_t) ctx * g.head_dim * 4;
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                /*
                DPCT1124:126: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(st->memcpy(b.conv[l], a.conv[l],
                                               C * (g.ssm_d_conv - 1) * 4)));
                /*
                DPCT1124:127: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(
                    st->memcpy(b.S[l], a.S[l],
                               (size_t)g.ssm_v_heads * g.ssm_state_size *
                                   g.ssm_state_size * 4)));
            }
            if (a.kc[l] && a.pos > 0) {
                /*
                DPCT1124:128: cudaMemcpy2DAsync is migrated to asynchronous
                memcpy API. While the origin API might be synchronous, it
                depends on the type of operand memory, so you may need to call
                wait() on event return by memcpy API to ensure synchronization
                behavior.
                */
                CK(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(
                    b.kc[l], row, a.kc[l], row, (size_t)a.pos * g.head_dim * 4,
                    (size_t)g.n_head_kv, dpct::device_to_device, *(st))));
                /*
                DPCT1124:129: cudaMemcpy2DAsync is migrated to asynchronous
                memcpy API. While the origin API might be synchronous, it
                depends on the type of operand memory, so you may need to call
                wait() on event return by memcpy API to ensure synchronization
                behavior.
                */
                CK(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(
                    b.vc[l], row, a.vc[l], row, (size_t)a.pos * g.head_dim * 4,
                    (size_t)g.n_head_kv, dpct::device_to_device, *(st))));
            }
        }
        /*
        DPCT1124:130: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(
            st->memcpy(b.hlast, a.hlast, (size_t)g.n_embd * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        if (dst == cur) load_cur();
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        /*
        DPCT1124:131: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(tok_m, hpin + 16, (size_t)T0 * 4)));
        /*
        DPCT1124:132: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(pos0_m, hpin + 1, 4)));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            /*
            DPCT1124:133: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(
                st->memcpy(seeds, hmain, fix.size() * (size_t)N * 4)));
        /*
        DPCT1124:134: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(
            st->memcpy(seeds + fix.size() * (size_t)N, hlast, (size_t)N * 4)));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            /*
            DPCT1124:135: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(hpin + 2, pick_m, 4)));
            CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
            drafts[0] = hpin[2];
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            /*
            DPCT1124:136: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(tok_m, hpin, 4)));
            /*
            DPCT1124:137: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(pos0_m, hpin + 1, 4)));
            /*
            DPCT1124:138: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(
                st->memcpy(seeds, hmtp + (size_t)last_row * N, (size_t)N * 4)));
            last_row = 0;
            if (mtp_graph) graph_run(gs().gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            else forward_mtp(1, pos + d, seeds, true, false);
            /*
            DPCT1124:139: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(st->memcpy(hpin + 2, pick_m, 4)));
            CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
            drafts[(size_t) d] = hpin[2];
        }
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        /*
        DPCT1124:140: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(tok, hpin + 16, (size_t)T * 4)));
        /*
        DPCT1124:141: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(pos0, hpin + 1, 4)));
        rows_deferred = true;
        graph_run(gs().gv[T], [&] { forward(T, pos, T, use_graph, true); });
        rows_deferred = false;
        /*
        DPCT1124:142: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(hpin + 2, pick, (size_t)T * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        int a = 0;
        while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        if (cancel != nullptr && cancel->load()) a = 0;
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        commit(a + 1);
        /*
        DPCT1124:143: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(
            st->memcpy(hlast, hmain + (size_t)a * N, (size_t)N * 4)));
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
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_cpu_ms = 0;
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() try {
        /*
        DPCT1124:144: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(hpin + 2, pick, 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
        return hpin[2];
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    void host_logits(std::vector<float> &out) try {
        out.resize((size_t) m->g.n_vocab);
        /*
        DPCT1124:145: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CK(DPCT_CHECK_ERROR(st->memcpy(out.data(), logits, out.size() * 4)));
        CK(DPCT_CHECK_ERROR(st->wait_and_throw()));
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.
bool place_experts(Model &m, Engine &E, double vram_gib, int threads,
                   int hybrid_t, int slots, std::string &err) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
    /*
    DPCT1106:146: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    CK(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(fr, tot)));
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
    const size_t slack = ((size_t) 512 << 20) + (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 &&
        (total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack)) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(DPCT_CHECK_ERROR(dpct::dpct_free(std::get<1>(pk), q_ct1)));
        }
        m.packed.clear();
        E.reset_graphs();
        /*
        DPCT1106:147: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        CK(DPCT_CHECK_ERROR(
            dpct::get_current_device().get_memory_info(fr, tot)));
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
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            /*
            DPCT1106:148: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            CK(DPCT_CHECK_ERROR(
                dpct::get_current_device().get_memory_info(fr, tot)));
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
            CK(DPCT_CHECK_ERROR(L->blobs = (uint8_t *)checked_device_alloc(
                                    L->stride * (size_t)ne + kMmqTail, q_ct1)));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memset(L->blobs, 0, L->stride * (size_t)ne + kMmqTail)
                    .wait_and_throw()));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(DPCT_CHECK_ERROR(
                    q_ct1
                        .memcpy(L->blobs + (size_t)e * L->stride, blob.data(),
                                L->L.bytes)
                        .wait_and_throw()));
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
    size_t slot_bytes = budget2 > staging ? budget2 - staging : 0;
    if (slot_bytes + staging + ((size_t) 256 << 20) > fr) slot_bytes = fr > staging + ((size_t) 512 << 20) ? fr - staging - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        const size_t Tn = (size_t) E.tmax, Nn = (size_t) g.n_embd, Kn = (size_t) m.top_k;
        size_t need = staging + Tn * Kn * Nn * 4 + Tn * Kn * 4;
        for (Layer* L : all) {
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);
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
    /*
    DPCT1048:3: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    bool pinned = false;
    try {
        arena = (uint8_t*)sycl::malloc_host(host, q_ct1);
        pinned = arena != nullptr;
    } catch (const sycl::exception&) {
        arena = nullptr;
    }
    if (!pinned) {
        /*
        DPCT1010:149: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        arena = (uint8_t*) std::malloc(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    size_t off = 0;
    E.hy.resize(all.size());
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(DPCT_CHECK_ERROR(L.blobs = (uint8_t *)checked_device_alloc(
                                    L.stride * (size_t)ne + kMmqTail, q_ct1)));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memset(L.blobs, 0, L.stride * (size_t)ne + kMmqTail)
                    .wait_and_throw()));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(DPCT_CHECK_ERROR(q_ct1
                                        .memcpy(L.blobs + (size_t)e * L.stride,
                                                blob.data(), L.L.bytes)
                                        .wait_and_throw()));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
        for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        L.n_slots = S;
        if (S > 0) {
            CK(DPCT_CHECK_ERROR(L.blobs = (uint8_t *)checked_device_alloc(
                                    L.stride * (size_t)S + kMmqTail, q_ct1)));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memset(L.blobs, 0, L.stride * (size_t)S + kMmqTail)
                    .wait_and_throw()));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memcpy(L.blobs, L.hblobs, L.stride * (size_t)S).wait_and_throw()));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) s] = s;
            H.exp_in_slot[(size_t) s] = s;
        }
        H.score.assign((size_t) ne, 0.0);
        CK(DPCT_CHECK_ERROR(H.slot_of_d = (int32_t *)checked_device_alloc(
                                (size_t)ne * 4, q_ct1)));
        CK(DPCT_CHECK_ERROR(
            q_ct1.memcpy(H.slot_of_d, H.slot_of.data(), (size_t)ne * 4)
                .wait_and_throw()));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    /*
    DPCT1048:4: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    CK(DPCT_CHECK_ERROR(E.h_h = (float *)checked_host_alloc(T * N * 4, q_ct1)));
    /*
    DPCT1048:5: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    CK(DPCT_CHECK_ERROR(E.cpu_out =
                            (float *)checked_host_alloc(T * Kp * N * 4, q_ct1)));
    /*
    DPCT1048:6: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    CK(DPCT_CHECK_ERROR(E.h_ids =
                            (int32_t *)checked_host_alloc(T * Kp * 4, q_ct1)));
    /*
    DPCT1048:7: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    CK(DPCT_CHECK_ERROR(E.h_miss =
                            (int32_t *)checked_host_alloc(T * Kp * 4, q_ct1)));
    /*
    DPCT1048:8: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    CK(DPCT_CHECK_ERROR(E.h_cnt = (int32_t *)checked_host_alloc(
                            all.size() * (size_t)ne * 4, q_ct1)));
    CK(DPCT_CHECK_ERROR(
        E.ecpu = (float *)checked_device_alloc(T * Kp * N * 4, q_ct1)));
    CK(DPCT_CHECK_ERROR(E.d_miss =
                            (int32_t *)checked_device_alloc(T * Kp * 4, q_ct1)));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
    CK(DPCT_CHECK_ERROR(E.ev_route = new sycl::event()));
    if (staging) {
        /*
        DPCT1025:150: The SYCL queue is created ignoring the flag and priority
        options.
        */
        CK(DPCT_CHECK_ERROR(E.cst = dev_ct1.create_queue(true)));
        for (int b = 0; b < 2; ++b) {
            CK(DPCT_CHECK_ERROR(E.stage[b] = (uint8_t *)checked_device_alloc(
                                    max_layer + kMmqTail, q_ct1)));
            CK(DPCT_CHECK_ERROR(
                q_ct1.memset(E.stage[b], 0, max_layer + kMmqTail).wait_and_throw()));
            CK(DPCT_CHECK_ERROR(E.ev_copy[b] = new sycl::event()));
            CK(DPCT_CHECK_ERROR(E.ev_used[b] = new sycl::event()));
            /*
            DPCT1012:151: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            /*
            DPCT1024:152: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            E.ev_used_ct1[b] = std::chrono::steady_clock::now();
            CK(DPCT_CHECK_ERROR(*E.ev_used[b] =
                                    (E.st)->ext_oneapi_submit_barrier()));
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    CK(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
    if (mtp_only)
        std::fprintf(stderr, "strata-q36: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "strata-q36: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads; %s (%.2f GiB staging); %.1f s\n",
                 S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 E.use_mmq ? "larger prompt chunks streamed" : "prompt chunks use grouped GPU kernels and the CPU pool",
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ------------------------------------------------------------------------------------------------ sampling
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

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
    void window_mtp() try {
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
            /*
            DPCT1124:153: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            CK(DPCT_CHECK_ERROR(E.st->memcpy(
                ss.hlast, E.hmain + (size_t)(r0 + keep[(size_t)i] - 1) * N,
                (size_t)N * 4)));
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
    void window_plain() try {
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
                /*
                DPCT1124:154: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                CK(DPCT_CHECK_ERROR(E.st->memcpy(
                    ss.hlast, E.hmain + r * (size_t)E.m->g.n_embd, Nb)));
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
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids, imgs);
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
        size_t reuse = match_len(ss, ids, ir.imgs);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            /*
            DPCT1040:40: Use sycl::stream instead of printf if your code is used
            on the device.
            */
            std::printf("PP %lld %lld %.0f %.1f\n", (long long)done,
                        (long long)fresh, ms_since(t0),
                        (double)done / std::max(1e-9, ms_since(t0) / 1000.0));
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
        }
        E.sess[(size_t) E.cur].pos = E.pos;
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
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones, and requests beside others, one token at a time
        const bool spec = E.mtp_d > 0 && !smp.needs_logits() && !adm;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
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
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            auto emit = [&](int t) {   // false: the answer ends here
                /*
                DPCT1040:41: Use sycl::stream instead of printf if your code is
                used on the device.
                */
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
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop);
                offered += E.mtp_d;
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
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
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
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
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
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
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
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        /*
        DPCT1106:155: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        CK(DPCT_CHECK_ERROR(
            dpct::get_current_device().get_memory_info(fr, tot)));
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
            const double kv_one = 45056.0, kv_shared = 5120.0;   // per cell: a session's K / V, and the fp16 copies the prompt attention keeps
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
    struct Report {
        Engine& E;
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "strata-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
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
            CK(DPCT_CHECK_ERROR(E.st->wait_and_throw()));
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
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
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
            offered += E.mtp_d;
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

// Test-only access to the kernels used by the model, without duplicating them.
#include "q36_test_hooks.inc"

#include "q36_addrms_hooks.inc"
