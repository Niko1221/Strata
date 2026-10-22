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
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/device.hpp"   // primary_device(): the peer probe and the executors speak to this CUDA ordinal
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/core/pin_budget.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "q36_kernels.cuh"
// STRATA_Q36_PREFILL_PEERS: a prompt chunk's routed experts on the helper GPUs.  The header is the frozen contract
// of the executor this file calls; with the flag unset nothing of it is constructed (see Engine::peer_prefill_on).
#include "q36_peer_prefill.cuh"
// --suffix-draft N (opt-in): suffix-lookup drafts beside the MTP drafts.  The header is host-only; with the flag
// unset the object is never constructed and spec_step never asks it, so the default path is the loop it was.
#include "q36_suffix.h"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

/// The spin-wait's yield hint, spelled as one instruction so it needs no intrinsics header (this file's only spin is
/// the mapped-pick wait below).  The x86 PAUSE; a no-op elsewhere, where the poll loop's own work is the yield.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

constexpr int kThreads = 256;

/// STRATA_Q36_PREFILL_PEERS: a device scope for a few one-off startup calls (the P2P probe).  The engine's own thread
/// lives on its primary device; every probe step runs on the one it needs and puts the caller's back.
struct DevScope {
    int prev = 0;
    explicit DevScope(int dev) {
        cudaGetDevice(&prev);
        if (prev != dev) cudaSetDevice(dev);
    }
    ~DevScope() {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != prev) cudaSetDevice(prev);
    }
};

/// Whether the direct peer link between `prim` and `peer` really CARRIES DATA - the reason STRATA_Q36_PREFILL_PEERS
/// tests a copy rather than a capability query.  It answers the narrower question "is there a direct link worth
/// preferring": a helper that fails it can still serve prompt chunks over host staging, which peer_host_route()
/// below is the gate for, so a false here is no longer by itself a helper out of the prompt path.
///
/// #2666, measured (src/core/peer_experts.cpp keeps the same probe and the same comment): on a pair of GPUs behind a
/// host bridge ("PHB") `cudaDeviceCanAccessPeer` answered yes, `cudaDeviceEnablePeerAccess` succeeded twice, and
/// `cudaMemcpyPeerAsync` then returned success having moved nothing - the peer read zeros and the rows it sent back
/// never arrived.  A capability query cannot see that.  So: enable the link, put a pattern on one side, zero the
/// other, copy each way and compare the bytes.  `why` names the step that failed, for the one line stderr gets.
bool peer_copies_data(int prim, int peer, std::string& why) {
    int a = 0, b = 0;
    if (cudaDeviceCanAccessPeer(&a, prim, peer) != cudaSuccess) {
        (void) cudaGetLastError();
        why = "CUDA refused to ask whether the two GPUs have a peer path";
        return false;
    }
    if (cudaDeviceCanAccessPeer(&b, peer, prim) != cudaSuccess) {
        (void) cudaGetLastError();
        why = "CUDA refused to ask whether the two GPUs have a peer path";
        return false;
    }
    if (a == 0 || b == 0) {
        why = "the two GPUs report no peer path (one direction is missing), so expert rows could not cross directly";
        return false;
    }
    for (int dir = 0; dir < 2; ++dir) {
        const int from = dir == 0 ? prim : peer, to = dir == 0 ? peer : prim;
        DevScope s(from);
        cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) {
            (void) cudaGetLastError();
            e = cudaSuccess;
        }
        if (e != cudaSuccess) {
            (void) cudaGetLastError();
            why = std::string("enabling peer access (CUDA") + std::to_string(from) + " -> CUDA" + std::to_string(to) +
                  ") failed: " + cudaGetErrorString(e);
            return false;
        }
    }
    // the copy itself: 16 KiB of a non-repeating pattern, device to device through the link, compared byte for byte
    constexpr size_t kN = 1 << 12;   // ints
    std::vector<int> want(kN), back(kN, 0);
    for (size_t i = 0; i < kN; ++i) want[i] = (int) ((i * 2654435761u) % 4093u) + 1;
    int* d[2] = {nullptr, nullptr};
    const int dev[2] = {prim, peer};
    for (int k = 0; k < 2; ++k) {
        DevScope s(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * sizeof(int)) != cudaSuccess) {
            (void) cudaGetLastError();
            why = "the probe could not allocate 16 KiB on both GPUs";
            for (int q = 0; q <= k; ++q) {
                DevScope t(dev[q]);
                cudaFree(d[q]);
            }
            (void) cudaGetLastError();
            return false;
        }
    }
    bool ok = true;
    for (int dir = 0; dir < 2 && ok; ++dir) {
        const int from = dir == 0 ? prim : peer, to = dir == 0 ? peer : prim;
        {
            DevScope s(from);
            if (cudaMemcpy(d[0], want.data(), kN * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
                (void) cudaGetLastError();
                ok = false;
                break;
            }
        }
        {
            DevScope s(to);
            if (cudaMemset(d[1], 0, kN * sizeof(int)) != cudaSuccess) {
                (void) cudaGetLastError();
                ok = false;
                break;
            }
        }
        if (cudaMemcpyPeer(d[1], to, d[0], from, kN * sizeof(int)) != cudaSuccess) {
            (void) cudaGetLastError();
            why = std::string("a test copy CUDA") + std::to_string(from) + " -> CUDA" + std::to_string(to) + " failed";
            ok = false;
            break;
        }
        {
            DevScope s(to);
            if (cudaMemcpy(back.data(), d[1], kN * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void) cudaGetLastError();
                ok = false;
                break;
            }
        }
        if (std::memcmp(back.data(), want.data(), kN * sizeof(int)) != 0) {
            why = std::string("a test copy CUDA") + std::to_string(from) + " -> CUDA" + std::to_string(to) +
                  " reported success but arrived wrong (a host bridge, no real peer path)";
            ok = false;
        }
    }
    for (int k = 0; k < 2; ++k) {
        DevScope s(dev[k]);
        cudaFree(d[k]);
    }
    (void) cudaGetLastError();
    return ok;
}

/// Whether a helper with no usable direct link may still serve prompt chunks, staging its expert rows through pinned
/// host memory (device -> host -> device) instead of across the peer link.  Measured on the box this was built for
/// (Tesla P100 primary, three P104-100 helpers): CanAccessPeer answered 0 for all twelve ordered pairs, enabling
/// peer access was refused, and MemcpyPeer reported success having host-staged everything - so on that machine only a
/// host route can serve a prompt chunk at all.
///
/// **OPT-IN, and now for a MEASURED reason: the route does not survive a decode step yet.**  On 192.168.1.98 with
/// `STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1`, `--bench-pp 160 --bench-tg 0` is clean (96.8 / 103.5
/// / 104.2 tok/s, three passes, rc=0) but `--bench-pp 160 --bench-tg 8` dies with "an illegal memory access was
/// encountered" at the first sync after the prompt - with one helper, three helpers, and under both layer striping
/// and the row split, while the SAME configuration with `..._HOST=0` (peer path off, nothing staged) is clean at
/// 49.3 tok/s pp160.  So the chunk arithmetic lands correctly and something in the staged buffers or the last
/// layer's un-waited copies is wrong across the prompt/decode boundary.  Until that is fixed a default-on host
/// route would turn a working default run into a crash, so the route requires the exact string "1"; unset, empty or
/// any other value keeps today's behaviour (P2P or the clean local fallback).  Read once, like helper_async().
bool peer_host_route() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_Q36_PREFILL_PEERS_HOST");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

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

/// The KV cache's int8 mode (--kv int8 / STRATA_Q36_KV=int8): kKVG values share one FP16 scale (the group's absmax
/// / 127), so a cache cell is 1 byte per value plus 2 bytes per 64 (with FP32 it is 4 bytes per value).  HD needs to
/// be a multiple of kKVG (256 and 128 are; the host falls back to FP32 otherwise).
constexpr int kKVG = 64;

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
/// The K / V base tables follow --kv: with an int8 store the FP32 table entries are null and the kc8 / ks8
/// (likewise vc8 / vs8) tables carry the codes and the scales, so a batch window reads the right session's store.
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    float* const* kc = nullptr;      // device [n_sessions]: this layer's K / V base of each session
    float* const* vc = nullptr;
    int8_t* const* kc8 = nullptr;    // device [n_sessions]: the int8 K / V codes and their FP16 scales (--kv int8)
    int8_t* const* vc8 = nullptr;
    __half* const* ks8 = nullptr;
    __half* const* vs8 = nullptr;
};

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256, a multiple of 32).  KI8 / VI8 (--kv): the K / V cells as kKVG-value
/// groups of int8 codes and one FP16 scale (kc / vc hold the codes, ks / vs the scales); the FP32 store writes the
/// values verbatim and its scales stay null.  The stores are independent: --kv k8v32 runs KI8 with a FP32 V.
template <bool KI8, bool VI8>
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, void* kc, void* vc, __half* ks, __half* vs, int H, int HKV, int HD,
                            int n_rot, const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    __shared__ float red[kThreads / 32];   // the warps' partial maxima (int8 mode)
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    // The row's own session's cache (--kv).  The two stores are independent (--kv k8v32 = an int8 K beside a
    // FP32 V), so each is taken from the table that carries it - only the int8 one has a codes+scales table.
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    if (KI8 && rs.kc8) {
        kc = rs.kc8[rs.slot[t]];
        ks = rs.ks8[rs.slot[t]];
    }
    if (VI8 && rs.vc8) {
        vc = rs.vc8[rs.slot[t]];
        vs = rs.vs8[rs.slot[t]];
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
        if (KI8) {
            const int NG = HD / kKVG;
            int8_t* kp = ((int8_t*) kc) + ((size_t) kh * ctx + pos) * HD;
            __half* ksp = ks + ((size_t) kh * ctx + pos) * NG;
            const int g = i / kKVG, w0 = g * 2;   // kKVG = 2 warps
            float am = fabsf(v);
            for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
            if ((i & 31) == 0) red[i >> 5] = am;
            __syncthreads();
            const float gk = fmaxf(red[w0], red[w0 + 1]);
            const float sk = gk > 0.f ? gk / 127.0f : 1.0f;
            kp[i] = (int8_t) __float2int_rn(fminf(fmaxf(v / sk, -127.0f), 127.0f));
            if ((i & (kKVG - 1)) == 0) ksp[g] = __float2half(sk);
            __syncthreads();   // red read out before the V store's round overwrites it
        } else {
            ((float*) kc)[((size_t) kh * ctx + pos) * HD + i] = v;
        }
        if (VI8) {
            const int NG = HD / kKVG;
            int8_t* vp = ((int8_t*) vc) + ((size_t) kh * ctx + pos) * HD;
            __half* vsp = vs + ((size_t) kh * ctx + pos) * NG;
            const int g = i / kKVG, w0 = g * 2;
            const float vi = vx[((size_t) t * HKV + kh) * HD + i];
            float am = fabsf(vi);
            for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
            if ((i & 31) == 0) red[i >> 5] = am;
            __syncthreads();
            const float gv = fmaxf(red[w0], red[w0 + 1]);
            const float sv = gv > 0.f ? gv / 127.0f : 1.0f;
            vp[i] = (int8_t) __float2int_rn(fminf(fmaxf(vi / sv, -127.0f), 127.0f));
            if ((i & (kKVG - 1)) == 0) vsp[g] = __float2half(sv);
        } else {
            ((float*) vc)[((size_t) kh * ctx + pos) * HD + i] = vx[((size_t) t * HKV + kh) * HD + i];
        }
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.  KI8 / VI8 (--kv): the K / V
/// cells dequantized with their group's FP16 scale on the fly (kc / vc the int8 codes, ks / vs the scales; a FP32
/// store passes its scales as null).  The int8 reads are vectorized (one uint2 per 8 K codes, one scale per kKVG
/// group) and the V weights are pre-multiplied into shared memory once per block: the long-context decode is
/// instruction-bound and this is where it shows.
template <bool KI8, bool VI8>
__global__ void k_attn_part(const float* q, const void* kc, const void* vc, const __half* ks, const __half* vs,
                            const int32_t* pos0, int t0, int H, int HKV, int ctx, float scale, int n_chunk,
                            float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    __shared__ float wv[HD / kKVG][kChunkCells];   // VI8: sc[k] * the V group's scale, once per block
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    if (KI8 && rs.kc8) {
        kc = rs.kc8[rs.slot[t]];
        ks = rs.ks8[rs.slot[t]];
    }
    if (VI8 && rs.vc8) {
        vc = rs.vc8[rs.slot[t]];
        vs = rs.vs8[rs.slot[t]];
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        float a = 0.f;
        if (KI8) {
            const int8_t* kp = ((const int8_t*) kc) + ((size_t) kvh * ctx + p) * HD;
            const __half* ksp = ks + ((size_t) kvh * ctx + p) * (HD / kKVG);
#pragma unroll 4
            for (int i0 = 0; i0 < HD; i0 += 8) {   // one uint2 (8 int8 codes) and one scale per kKVG group
                const uint2 raw = *reinterpret_cast<const uint2*>(kp + i0);
                const int8_t* c8 = reinterpret_cast<const int8_t*>(&raw);
                const float sg = __half2float(ksp[i0 / kKVG]);
#pragma unroll
                for (int x = 0; x < 8; ++x) a = fmaf(qs[i0 + x] * sg, (float) c8[x], a);
            }
        } else {
            const float* kp = ((const float*) kc) + ((size_t) kvh * ctx + p) * HD;
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
    if (VI8) {
        const int8_t* vp = ((const int8_t*) vc) + ((size_t) kvh * ctx + p0) * HD;
        const __half* vsp = vs + ((size_t) kvh * ctx + p0) * (HD / kKVG);
        __syncthreads();   // sc[] is complete; the weight table takes its per-cell values now
        for (int x = j; x < (HD / kKVG) * n; x += kThreads) {
            const int g = x / n, k = x - g * n;
            wv[g][k] = sc[k] * __half2float(vsp[(size_t) k * (HD / kKVG) + g]);
        }
        __syncthreads();
        const float* wr = wv[j / kKVG];
        const int8_t* vj = vp + j;
        for (int k = 0; k < n; ++k) acc = fmaf(wr[k], (float) vj[(size_t) k * HD], acc);
    } else {
        const float* vp = ((const float*) vc) + ((size_t) kvh * ctx + p0) * HD + j;
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
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
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
/// k_group_hy with a skip mask (the helper tier's split, --expert-cache-device1..3): the entries the helper already
/// computed into `eparts` (mask 1) are left out, the rest grouped exactly as k_group_hy would group them had those
/// entries not been in `ids`.  A helper that does not hold every expert of a layer still contributes what it does
/// hold (Qwen3.8 Flash splits one window per entry, expert_source.cpp: kind[i] = 2 for the helper, -1 for the host
/// pool).  One block.
__global__ void k_group_hy_masked(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                                   const uint8_t* skip, const uint8_t* blobs, unsigned long long blob,
                                   unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                                   int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (!skip[e] && slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
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
            if (skip[e] || slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group with a skip mask (the helper tier's split): only the entries the mask keeps are grouped, into the same
/// `eparts` rows the full kernel would have used (see k_group_hy_masked).
__global__ void k_group_masked(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* skip,
                               const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
                               int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (!skip[e]) atomicAdd(&cnt[ids[e]], 1);
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
            if (skip[e]) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// STRATA_Q36_HELPER_ASYNC: a helper's rows merged into the entries' rows where the per-entry mask marks
/// them (1 = the helper computed that entry).  The deferred upload stages the helper's sparse rows and
/// merges exactly those entries, so it never overwrites a row a local path computed.  grid (n_ent), block 256.
__global__ void k_merge_helper(float* parts, const float* helper, const uint8_t* mask, int n) {
    const int e = blockIdx.x;
    if (!mask[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = helper[(size_t) e * n + i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).  KI8 / VI8 (--kv): the codes dequantized with their group's FP16 scale
/// (a FP32 store passes its scales as null).
template <bool KI8, bool VI8>
__global__ void k_kv_f16(const void* kc, const void* vc, const __half* ks, const __half* vs, __half* k16, __half* v16,
                         int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    const int64_t p = rest / HD, d = rest % HD;
    const size_t si = (size_t) kvh * ctx * (HD / kKVG) + (size_t) p * (HD / kKVG) + d / kKVG;
    if (KI8) k16[i] = __float2half(__half2float(ks[si]) * (float) ((const int8_t*) kc)[src]);
    else k16[i] = __float2half(((const float*) kc)[src]);
    if (VI8) v16[i] = __float2half(__half2float(vs[si]) * (float) ((const int8_t*) vc)[src]);
    else v16[i] = __float2half(((const float*) vc)[src]);
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

/// STRATA_Q36_PREFILL_PEERS: the row contract of src/qwen35/q36_peer_prefill.cuh, in the engine's own terms.
///
/// A window is n_ent = T * K entries, entry e = token (e / K), slot (e % K).  Entry e BELONGS to helper j iff j's
/// cache holds (layer, ids[e]).  The executor writes helper j's owned entries, taken in INCREASING e, into the
/// contiguous eparts run [base_j, base_j + rows_j), base_j = n_local + sum_{i < j} rows_i (local entries first, then
/// each helper's run in helper order) - that is the contract's "order = entry index, ascending", and it is a prefix
/// count over e that consults nothing but the ownership mask.
///
/// The engine therefore needs two tables: one 32-bit word per entry naming the eparts row that entry's down product
/// lands in (`prow2`, which moe_end reads in place of `rinv`), and the local MMQ's `ids` table naming the same rows
/// for the entries the primary computes.  Both are built from ONE host walk (Engine::peer_plan), so the row
/// assignment the executors were handed and the row assignment the combine reads cannot drift.
///
/// Why the plan is built on the HOST: the ownership counts (n_local, rows_j) and the assertion that they add up to
/// n_ent are decisions about which GPU computes what, and getting one wrong silently corrupts logits.  The route
/// ids are already coming to the host this layer (the same per-layer D2H that brings rmax), so the plan is an O(n_ent)
/// walk over bytes the host has in hand - no extra synchronisation, and the one place the invariant is checked is a
/// place that can print a reason and stop.

/// The local MMQ's `ids` table: sorted row r (r < n_local) writes eparts row prow2[ent_dst[r]].  The local MMQ reads
/// its activation rows grouped by expert (the sorted order), so this is the map from its output row to the entry's
/// own eparts row - the same renumbering `rinv` does with no helper, reproduced through the plan so peer-owned
/// entries can be left out of the sorted order entirely.
__global__ void k_peer_lid(const int32_t* ent_dst, const int32_t* prow2, int n_local, int32_t* lid) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_local) lid[r] = prow2[ent_dst[r]];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
/// `owned` (STRATA_Q36_PREFILL_PEERS only, null otherwise: the default argument keeps every existing call site
/// byte-for-byte what it was) is this layer's per-EXPERT byte saying a helper GPU holds that expert.  Such an entry
/// is not counted here at all, so `bounds` leaves it no row in the sorted order.  The matching skip in
/// k_route_scatter is what keeps the two consistent: count and scatter must skip the SAME entries, or the scatter's
/// atomicAdd lands in the next expert's sorted interval and corrupts another expert's rows.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt, const uint8_t* owned = nullptr) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) {
        if (owned != nullptr && owned[ids[e]] != 0) return;
        atomicAdd(&cnt[ids[e]], 1);
    }
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
                                int32_t* inv, const uint8_t* owned = nullptr) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    // the peer-owned entries take no sorted row: they leave `cur` (and so every expert's interval) untouched, and
    // their `dst` / `src` / `inv` slots stay as they were - their eparts row is the peer plan's business, not rinv's.
    if (owned != nullptr && owned[ids[e]] != 0) return;
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
        //
        // The stride is also what a HELPER's expert cache has to use, because MMQ addresses the next expert as
        // `w + z * (expert_bytes / type_bytes)`: the cache's step and the engine's step must be the same number, or
        // every expert but the first is read from the wrong offset (NaN rows, not an error).  The helper cache is
        // told this step (RemoteExperts::open) and lays its slots out with it verbatim rather than with its own
        // 256-byte alignment, which would move the step (a Q6_K layer's 2041200 becomes 2041344).
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

/// Expert source for the q36 helper tier (a P100 plus 2xP104): the model's GGUF is mapped in `load` and the three
/// role tensors stay open, so the blob for (layer, expert) is the layer's own `[gate | up | down]` blob `eparts`
/// entries are computed from - byte for byte what this engine already reads.  `blob()` assembles it into one
/// rotating staging buffer, valid until the next call, which is the ExpertSource contract RemoteExperts::open()
/// relies on (it copies each selected blob into the helper's VRAM slot as it goes).
class Q36ExpertSource final : public strata::core::ExpertSource {
public:
    void set(const std::vector<const Layer*>& layers, int64_t n_expert) {
        layers_ = layers;
        n_expert_ = n_expert;
        // The staging buffer is sized from the largest layer, not layers_.front(): this GGUF's 41 blocks do not all
        // carry the same expert size (measured 2026-10-20: 1900544 bytes on block 0, 2301952 on the largest), and
        // blob() serves whichever layer open() asks for.  Sizing from front() under-allocated by 401408 bytes, and
        // expert_blob's third memcpy then ran past the buffer and corrupted the heap.
        size_t mx = 0;
        for (const Layer* L : layers_) if (L && L->L.bytes > mx) mx = (size_t) L->L.bytes;
        staging_.assign(layers_.empty() ? 0 : mx, 0);
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (layer < 0 || layer >= (int64_t) layers_.size() || expert < 0 || expert >= n_expert_ || staging_.empty())
            return nullptr;
        expert_blob(*layers_[(size_t) layer], expert, staging_.data());
        return staging_.data();
    }
private:
    std::vector<const Layer*> layers_;
    int64_t n_expert_ = 0;
    std::vector<uint8_t> staging_;
};

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
    // --kv int8 / STRATA_Q36_KV=int8: the K / V caches as int8 codes with one FP16 scale per kKVG values (kc8 / vc8
    // the codes, ks8 / vs8 the scales; the FP32 store's kc / vc stay empty then).  kv_k8 / kv_v8 are independent
    // (--kv k8v32 stores a FP32 V next to the int8 K); decode and prompt attention dequantize on the fly.  The stores
    // apply to every state index, the MTP block's own K / V (state index n_layers) included.
    std::vector<int8_t*> kc8, vc8;
    std::vector<__half*> ks8, vs8;
    bool kv_k8 = false, kv_v8 = false;
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // STRATA_Q36_MTP_MAPPED: the chain's per-step position words (step d reads pos0_m + d) and the host-mapped slot each
    // step's draft pick is published into.  Allocated only under the flag: with it unset pos0_m is the one word it
    // always was and pick_h stays null.
    static constexpr int kChainMax = 8;   // --mtp takes 0 .. 7 drafts, so steps d = 1 .. 7 (the command line checks it)
    int32_t* pick_h = nullptr;            // host-mapped [kChainMax]: the published picks, -1 = not there yet
    int32_t* pick_h_d = nullptr;          // its device-side alias (cudaHostGetDevicePointer)
    /// STRATA_Q36_MTP_MAPPED=1: the Flash-Next draft chain (src/core/mtp.cpp:1599-1650).  The whole chain is submitted
    /// before the host waits for anything, each draft's pick travels device -> host-mapped memory instead of a
    /// cudaMemcpy behind a stream sync, and the next step's token is produced ON THE DEVICE (k_chain_publish) rather
    /// than brought back through the host.
    ///
    /// It can only win where a draft step has no host block in it: a host block inside a queued step makes the host
    /// wait per step again, behind work it just queued.  So the gate also refuses the configurations that cannot
    /// benefit (the MTP layer hybrid - its CPU pool spin and its synchronous pool call - and the VERIFY build, whose
    /// comparisons sync), and says so in the startup line instead of silently measuring nothing.  With the variable
    /// unset (or not exactly "1") none of the statements below run and spec_step is the loop it always was.
    bool mtp_mapped() const {
        static const bool on = [] {
            const char* v = std::getenv("STRATA_Q36_MTP_MAPPED");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return on;
    }
    /// STRATA_Q36_MTP_MAPPED: whether the mapped chain is in force at all - the flag, the buffer, and neither of the two
    /// diagnostics whose own host reads cannot survive pre-submission.  spec_step() asks this per window; the startup
    /// line below asks it once, after the placement, which is what decides whether a draft step can win anything.
    bool mapped_chain_live() const {
        // Every term is a host block inside the draft step: the top-2 runner-up read and the VERIFY comparisons
        // synchronize per step; a hybrid MTP layer's CPU pool spins on the route event and then runs synchronously -
        // the host would wait inside every queued step and pre-submission would buy nothing.  With all of them
        // absent a draft step is launches only, and that is the only place the mapped chain runs.
        return mtp_mapped() && pick_h != nullptr && !mtp_top2() && !verify_k && !lhy(m->layers.size());
    }
    /// ... and, if it is not live, the reason, so a null measurement is not read as a zero gain.  The placement decides:
    /// the MTP block's layer hybrid (its CPU pool spin and its synchronous pool call are host blocks INSIDE a queued
    /// step, so pre-submission buys nothing), or a VERIFY build (its comparisons sync), or the top-2 diagnostic (its
    /// runner-up is a host read of one logits buffer the steps would overwrite each other over).
    void report_mapped_mtp() {
        if (pick_h == nullptr) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: INERT (the mapped pick buffer was not allocated): "
                                 "the draft loop is the per-step-sync one\n");
            return;
        }
        if (mtp_top2()) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: INERT with STRATA_Q36_MTP_TOP2=1 (the runner-up is a "
                                 "host read of one logits buffer, which a pre-submitted chain would overwrite): the "
                                 "draft loop is the per-step-sync one, and the diagnostic is the whole of what is run\n");
            return;
        }
        if (verify_k) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: INERT under STRATA_Q36_VERIFY (its comparisons "
                                 "synchronize inside the step): the draft loop is the per-step-sync one\n");
            return;
        }
        if (lhy(m->layers.size())) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: INERT - the MTP block's layer is hybrid (its CPU "
                         "expert pool spins on the route event and then runs synchronously), so the host waits inside "
                         "every queued step anyway; the chain is submitted in one go but pays the same per-step wait\n");
            return;
        }
        std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: live (%s draft steps, one graph each; the picks travel "
                     "through mapped host memory and the next token is produced on the device)\n", mtp_graph ? "graph" : "eager");
    }
    // graphs: [0] the decode step, [T] a verify window of T rows, gm1[d] the MTP draft step of chain depth d
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1[kChainMax] = {};
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
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- --mtp-adaptive (opt-in): the learned per-depth draft cost and acceptance (EMA), and the D the next
    // speculative step drafts with.  With the flag unset every member below stays at its zero/0.9 value and no
    // call site runs (d_adapt is false), so the default path is the loop it always was.
    struct Adapt {
        int base = 0;             // the depth pool the command line allowed (ceiling; 0 = plain rounds too)
        int next = 0;             // the depth to ask for next round
        double acc[8] = {}, time[8] = {}, n[8] = {};
        int use[8] = {};         // rounds answered at each depth (for the report)
        int probes[8] = {};      // probe rounds per depth (D = 0 included: the plain step has to be MEASURED too,
        int probed_next = 0;     // or its estimated cost would stand against a draft depth's measured one)
        static constexpr int kProbes = 3;   // as DraftPolicy's kProbes: too few rounds veto a depth on one bad
    } adapt{};
    bool d_adapt = false;        // the flag itself: the call sites below run only when this is set
    /// the value of d: committed tokens per ms (1 + the learned committed drafts over the step's ms).  Measured
    /// depths run their EMAs; the others the geometric estimate from the deepest measured depth (committed drafts
    /// x0.7 per level away, step ms x1.45 per level up).
    double adapt_rate(int d) const {        double a, c;
        if (adapt.n[(size_t) d] > 0) { a = adapt.acc[(size_t) d]; c = adapt.time[(size_t) d]; }
        else {
            int last = 0;   // the deepest depth measured
            for (int e = 0; e <= adapt.base; ++e) if (adapt.n[(size_t) e] > 0) last = e;
            const double a0 = last > 0 ? adapt.acc[(size_t) last] : 1.5;   // the prior: committed drafts at the first
            a = a0;            // depth seen (an untried pool), the geometric shape replacing the measured rounds
            for (int k = 0; k < std::abs(d - last); ++k) a *= 0.7;
            a = d == 0 ? 0.0 : std::clamp(a, 0.05, 7.0);
            const double c0 = last > 0 ? adapt.time[(size_t) last] : 0.0;
            c = c0 > 0 ? c0 * (1.0 + 0.45 * (d - last)) : 20.0;
        }
        return (1.0 + a) / c;
    }
    /// --mtp-adaptive: the depth the next speculative step asks for.  Until every pool depth has been PROBED -
    /// adapt.kProbes rounds at it - a depth is never judged by its estimate: the rotation (probed_next) hands out
    /// the unprobed depths in order, and only when the whole pool is probed does it settle on the best
    /// committed-tokens-per-ms measured.  A bad single round (the rebalance clamp cut the drafts, the pool was
    /// cold) cannot veto a depth - the shape of DraftPolicy's kProbes in src/spec/draft_policy.cpp, where a size
    /// not measured for a stretch is tried again instead of priced out for good.
    int next_adapt_D() {
        if (!d_adapt) return 0;   // unset: the caller never asks (it reads mtp_d / gen_D as it always did)
        for (int d = adapt.probed_next; d <= adapt.base; ++d)   // the exploration pass, from the pointer
            if (adapt.probes[(size_t) d] < Adapt::kProbes) { adapt.probed_next = d; return d; }
        for (int d = 0; d <= adapt.base; ++d)   // the rotation wrapped: probe depths the pointer passed by
            if (adapt.probes[(size_t) d] < Adapt::kProbes) { adapt.probed_next = d; return d; }
        // every pool depth probed: the measured winner, re-measured every round after (the acceptance
        // drifts - the prompt's head and its continuation have different curves), and a tie goes to the
        // cheaper depth.
        int best = 0;
        double best_r = adapt_rate(0);
        for (int d = 1; d <= adapt.base; ++d) {
            const double r = adapt_rate(d);
            if (r > best_r || (r == best_r && d < best)) { best_r = r; best = d; }
        }
        return best;
    }
    /// --mtp-adaptive: fold a round (it asked for D, the rebalance clamp may have run less) into
    /// the EMAs.  `accept` is the drafts the window committed (D = 0 is the plain step: it commits
    /// its one token, so accept = 0 and only the step-time EMA moves).  A round not picked by
    /// next_adapt_D() is not folded - the depth it asked for was not the learned one, so its time
    /// says nothing about that depth's step cost.  The probe counter is the rounds at the depth,
    /// and the settled winner re-measures as the answer goes on (the acceptance drifts: the prompt
    /// is read, then the continuation, and the two have different curves).
    void adapt_observe(int D, int accept, double round_ms) {
        if (!d_adapt || D < 0 || D > 7 || D != adapt.next) return;
        adapt.use[(size_t) D] += 1;
        adapt.probes[(size_t) D] += 1;   // the plain step is probed too (see the field's comment)
        if (round_ms > 0.0) adapt.time[(size_t) D] = adapt.n[(size_t) D] > 0 ? 0.75 * adapt.time[(size_t) D] + 0.25 * round_ms
                                                                             : round_ms;
        if (D > 0) {
            adapt.acc[(size_t) D] = adapt.n[(size_t) D] > 0 ? 0.75 * adapt.acc[(size_t) D] + 0.25 * accept
                                                            : accept;
        }
        adapt.n[(size_t) D] += 1.0;   // a depth's EMA has data (D = 0 is the plain step's cost)
        adapt.next = next_adapt_D();
    }

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S, kc, vc;
        // --kv int8: this session's int8 K / V codes and their FP16 scales, per state index.  A store that is
        // FP32 leaves its kc / vc pointing at the values and these null; an int8 store is the reverse.
        std::vector<int8_t*> kc8, vc8;
        std::vector<__half*> ks8, vs8;
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
    // the same tables for the int8 K / V stores (--kv int8 / k8v8 / k8v32 / k32v8): per state index the session's
    // codes and its FP16 scales, so a batch window reads the right session's store
    int8_t **d_kc8_tab = nullptr, **d_vc8_tab = nullptr;
    __half **d_ks8_tab = nullptr, **d_vs8_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    // --suffix-draft N (opt-in): the suffix drafter and its per-round policy.  Null unless the flag is set, and the
    // spec_step call site checks the pointer before touching it, so an unset flag adds nothing to the default path.
    std::unique_ptr<strata::qwen35::Q36Suffix> suffix;
    long long sfx_win = 0, sfx_acc = 0;                                     // suffix windows taken, drafts accepted
    long long sfx_bad = 0;                                                  // rounds skipped: the history did not match pos
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

    // ---- the helper expert tier (--expert-cache-device1..3): routed experts on other CUDA devices, a P100
    // plus two P104s.  Layers are striped across the helpers (see open_helpers) as a preference; a helper
    // computes the entries it holds in its VRAM cache and the local path computes the rest of that layer, so a
    // helper that does not fit a whole layer still contributes everything it does hold (Qwen3.8 Flash's per-
    // entry split).  Off by default: without the flags, helpers_enabled() is false everywhere and this path
    // costs one branch.
    std::array<strata::core::RemoteExperts, 3> helpers;
    std::vector<int8_t> helper_of_layer;   // [n_layers]: -1 local, else 0..2 the helper's index (empty in row-split mode)
    int helper_count = 0;
    bool helpers_enabled() const { return helper_count > 0; }
    std::array<bool, 3> helper_on = {false, false, false};   // which indices of helpers[] are open
    bool helper_row_split = false;   // STRATA_Q36_HELPER_PARALLEL: every helper holds experts of every layer
    /// STRATA_Q36_HELPER_ASYNC (opt-in): moe_helpers() launches the helper where it always did (so the
    /// helper GPU starts at once), but the host's blocking wait for that GPU's result - the eparts
    /// upload of the rows it computed - is deferred to moe_helpers_wait(), after the layer's own
    /// work (the local grouped kernel, the shared expert, the CPU pool) has been issued, so that work
    /// overlaps the helper's compute instead of following it.  Nothing else moves: the same helper
    /// computes this layer's rows and no other (finish() runs before the next layer's begin()), so
    /// helper_out / h_x_ stay single-slot, and the per-entry mask is still written and read inside
    /// this layer only - the local paths take the entries the mask leaves at 0 exactly as before, so
    /// a deferred row lands where that entry's row always lands, and nothing a local path computed is
    /// overwritten.
    ///
    /// CUDA graph: this does not add a capture hazard, but not for the reason it may look like - helpers
    /// are NOT limited to the hybrid placement, so use_graph is not always false when a helper is on
    /// (place_experts returns early when every expert fits in VRAM, and its mtp_only branch clears
    /// main_hybrid, while open_helpers can still hand a layer to a helper either way).  The reason is
    /// that this path already makes two calls that are illegal under stream capture, before and
    /// independently of this flag: moe_helpers's own cudaStreamSynchronize(st) and RemoteExperts::finish's
    /// sync of the helper's stream.  A captured graph can therefore never contain this path, and the
    /// deferral only moves statements between two points of a host-side sequence that is already
    /// uncapturable.  (That pre-existing hazard - all experts in VRAM or mtp_only, plus a helper, plus a
    /// graph - is untouched by this flag and does not fire on any configuration measured here.)
    ///
    /// Failure policy: a deferred finish() failure ends the run (std::exit) instead of falling back the
    /// way the synchronous path does.  The layer's local launches were already queued against the mask
    /// this helper's rows are missing from, so an all-zero mask could only be honoured by re-running that
    /// layer's local work - not worth building for a CUDA error that dooms the run anyway.
    ///
    /// Off by default: with the variable unset moe_helpers() finishes the helper where it always has and
    /// moe_helpers_wait() returns at once.
    bool helper_async() const {
        static const bool on = std::getenv("STRATA_Q36_HELPER_ASYNC") != nullptr;
        return on;
    }
    /// STRATA_Q36_HELPER_PARALLEL=1 (opt-in): the helpers ROW-SPLIT a layer instead of striping layers.
    ///
    /// The problem it removes: with the layer striping helper_layer_stripes() computes, every layer belongs to
    /// exactly ONE helper, so at any instant one helper GPU works and the other idles - per token the helpers
    /// spend ~3.3 ms each (measured with STRATA_Q36_HELPER_TIMING=1, ms_wait 3.342 / 3.310) and those two
    /// intervals are strictly sequential, ~6.6 ms of helper GPU time per token on the critical path.
    ///
    /// What changes: open_helpers() gives each helper a balanced share of EVERY main layer's experts (in
    /// place of a contiguous layer interval), and moe_helpers() begins that layer's window on every helper
    /// that holds part of it.  Each RemoteExperts owns its own CUDA stream, so the helpers compute the same
    /// layer at the same time; the entry a helper owns follows from its own cache (an entry's expert is in at
    /// most one helper's cache), so the ownerships stay disjoint and the per-entry mask is their union.
    ///
    /// The result of a window is unchanged in place and form: each helper's finish() scatters the rows IT
    /// computed into the shared helper_out at that entry's row index, the two scatters touch disjoint rows of
    /// that one buffer, and the whole-window upload / k_merge_helper that follows writes exactly those entries
    /// (the local paths still rewrite every entry whose mask bit is 0).  So a layer's rows land where they
    /// always did; the only difference is which GPU produced them and that they were produced concurrently.
    ///
    /// Total expert residency is unchanged: helper k now holds experts [k*E/n, (k+1)*E/n) of all L layers
    /// instead of all E experts of L/n layers - L*E/n either way, so the same VRAM budget fills the same
    /// slots, and a helper that cannot hold its whole share still contributes what it does hold (the mask is
    /// per entry, and the local paths absorb the rest).
    ///
    /// Off by default: with the variable unset (or not exactly "1") open_helpers keeps the layer striping,
    /// helper_window() returns the one helper the map names, and every statement below is the one it always
    /// was - the parity gate is unaffected.  With a single helper enabled the row split degenerates to exactly
    /// the one stripe that helper would have taken (all experts of all layers), so the flag only ever has an
    /// effect where two or more helpers are open.
    bool helper_parallel() const {
        static const bool on = [] {
            const char* v = std::getenv("STRATA_Q36_HELPER_PARALLEL");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return on;
    }
    /// STRATA_Q36_HELPER_TIMING=1: the throwaway measurement gate.  One reader of the env var, shared by the
    /// RemoteExperts drain overhead and the two primary-stream host blocks below, so every new timer keys off
    /// the same exact '1' the bench harness does.  A flag-off run reads the env once and does no clock work.
    static bool helper_timing() {
        static const bool on = [] {
            const char* ht = std::getenv("STRATA_Q36_HELPER_TIMING");
            return ht && ht[0] == '1' && ht[1] == '\0';
        }();
        return on;
    }
    /// Whether state index li's layer is offered to a helper (the entries the helper holds leave the local path).
    bool helper_layer(size_t li) const {
        // only main-model layers carry a helper slot: the MTP block (li == m->layers.size()) stays local
        if (li >= (size_t) m->g.n_layers) return false;
        // STRATA_Q36_HELPER_PARALLEL: the geometry is not layer-indexed - every main layer is offered to every
        // helper, because each helper holds a share of every layer's experts.  helper_of_layer is empty then.
        if (helper_row_split) return helpers_enabled();
        return !helper_of_layer.empty() && helper_of_layer[li] >= 0;
    }
    /// The helpers that compute state index li's window, in device order; returns how many (0 = none).
    /// Default path: the single helper helper_of_layer names.  STRATA_Q36_HELPER_PARALLEL=1: every open helper,
    /// each owning the entries whose experts its own cache holds.
    int helper_window(size_t li, int* out) const {
        if (!helpers_enabled() || helper_failed || !helper_layer(li)) return 0;
        if (!helper_row_split) {
            out[0] = helper_of_layer[li];
            return 1;
        }
        int n = 0;
        for (int r = 0; r < 3; ++r)
            if (helper_on[(size_t) r]) out[n++] = r;
        return n;
    }
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }

    /// the four K / V store combinations (--kv): f(kv_ic<K8>{}, kv_ic<V8>{}) runs with the kernel's template arguments.
    /// Every state index takes the same stores, the MTP block at index n_layers included.
    template <bool B> struct kv_ic { static constexpr bool value = B; };
    template <class F> void kv4(F&& f) const {
        if (kv_k8 && kv_v8) f(kv_ic<true>{}, kv_ic<true>{});
        else if (kv_k8) f(kv_ic<true>{}, kv_ic<false>{});
        else if (kv_v8) f(kv_ic<false>{}, kv_ic<true>{});
        else f(kv_ic<false>{}, kv_ic<false>{});
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    // the helper tier's host staging: routing ids and the normed activation, and the result rows it fills.
    // `helper_mask` is the per-entry mark of what the helpers computed in the current window (1 = the helper
    // took that entry); `d_helper_mask` is its device copy, read by the masked grouping kernels.  The local
    // paths complete every entry the mask leaves at 0.
    int32_t* h_helper_ids = nullptr;
    float* h_helper_x = nullptr;
    float* helper_out = nullptr;   // [n_entries][N] the helpers' rows (sparse; finish() scatters by row)
    float* helper_stage = nullptr;   // STRATA_Q36_HELPER_ASYNC: [n_entries][N] the deferred upload's staging
    std::vector<uint8_t> helper_mask;
    uint8_t* d_helper_mask = nullptr;   // current layer mask for local grouping / CPU skip
    int helper_mask_li = -1;   // the state index whose window helper_mask belongs to (-1: none)
    // STRATA_Q36_HELPER_ASYNC: the helper windows whose finish() is deferred to moe_helpers_wait().
    // helper_wait_n is 1 on the default path and 2..3 with STRATA_Q36_HELPER_PARALLEL (one window per helper).
    int helper_wait_n = 0;
    int helper_wait_hi[3] = {-1, -1, -1};
    int64_t helper_wait_ent = 0;
    int helper_wait_N = 0;
    Q36ExpertSource helper_src;
    cudaEvent_t ev_route = nullptr;
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    long long hy_helper = 0;   // batch entries computed on a helper GPU (--expert-cache-device1..3)
    double hy_cpu_ms = 0;
    // STRATA_Q36_HELPER_TIMING=1 only: the two primary-stream host blocks the counters above cannot see.
    // d_prebegin_ms is moe_helpers()' cudaStreamSynchronize(st) wait for the D2H of h_helper_ids/h_helper_x on
    // the PRIMARY stream; d_route_spin_ms is moe_hybrid_cpu()'s busy-wait on ev_route, which the hy_cpu_ms timer
    // starts AFTER and so excludes.
    // Both are zero (and never read) with the flag off.
    double d_prebegin_ms = 0;
    long long d_prebegin_n = 0;
    double d_route_spin_ms = 0;
    long long d_route_spin_n = 0;
    // ---- STRATA_Q36_HELPER_TIMING=1 only: the PRIMARY GPU's own per-layer timeline, from cudaEvent on st.
    //
    // These are the segments the HOST counters cannot see (the ~14 ms/token of prebegin/route_spin/helper_wait
    // above is the host standing still; what the P100 is actually doing for the rest of the step is here).
    // gseg_ms[k] accumulates cudaEventElapsedTime between two adjacent boundaries of a layer() call, gseg_n[k]
    // counts the layers, and gspan_ms/gspan_n_k is the layer tile (E0 .. E_last) the segments are supposed to
    // TILE exactly - so gp_resid_ms = gspan - sum(segments) is the residual, not a free parameter.
    //
    // Everything here is inside the HELPER_TIMING gate: with the flag off not one cudaEventRecord runs, so the
    // default path issues the same statements it always did.  Segments are named by the code they bound, and the
    // boundaries are chosen so no gap and no overlap: see kGsegName above and the ev_mark() calls in layer().
    static constexpr int kGsegN = 8;   // the boundaries E0 .. E_last, one event each (E_last bounds the combine)
    static constexpr const char* kGsegName[kGsegN] = {
        "attn_ssm",        // E0..E1: prep_in, the attention or SSM kernels, mm(wo / ssm_out), add_rms_post
        "router",          // E1..E2: f32mm(router[_ext]) plus route_group / native_router_topk
        "helper_dispatch", // E2..E3: the route D2H (h_ids/h_h) plus moe_helpers' host wait for it (E3 lands where that wait ends)
        "local_grouped",   // E3..E4: moe_hybrid_gpu's group_entries + native_expert_grouped
        "local_shared",    // E4..E5: the shared expert (apply / k_swiglu_quantize / mmq(sh_down)) and its gate
        "cpu_pool",        // E5..E6: moe_hybrid_cpu's CPU-pool work, timed on the GPU queue from behind it
        "helper_merge",    // E6..E7: moe_helpers_wait's deferred merge into eparts
        "combine",         // E7..E_last: moe_end - k_moe_out, the residual add and the next norm
    };
    // Boundaries inside layer().  The order is the order the events are recorded in, so a list of
    // {name, boundary-before, boundary-after} that reads as an unbroken tile of the call is the whole design:
    // no delta here covers a period that another also covers, so the residual above is the only slack.
    static constexpr int kSegAttnEsm = 0, kSegRouter = 1, kSegDisp = 2, kSegGrp = 3,
                        kSegShared = 4, kSegCpu = 5, kSegMerge = 6, kSegComb = 7;
    cudaEvent_t ev_seg_ev[kGsegN] = {};   // NOT ev_seg: that name is the marker below, and a member array
                                          // shadows a member function of the same name in this class.
    bool gpu_timing = false;      // helper_timing() and every event exists (see open_engine)
    bool gpu_timing_open = false; // the events of one layer() call are still being recorded (do not read them yet)
    bool gpu_open_this_forward = false;  // forward() arms exactly one layer (the last) to carry the tile
    bool gpu_timing_hold = false; // a graph is being captured (event records are legal, readings are not)
    struct Gseg { double ms[kGsegN]; };   // one queued layer tile; a named type, not an in-class definition
    std::deque<Gseg> gseg_ring;   // queued layer tiles, oldest first
    long long gseg_dropped = 0;   // tiles the ring had no room for (recycled, never counted twice)
    double gseg_ms[kGsegN] = {};  // the accumulated segments, per token
    double gspan_ms = 0;          // the accumulated layer tile, per token
    long long gseg_n[kGsegN] = {}, gspan_n = 0;
    double gp_resid_ms = 0;       // the residual, per token
    double gp_embed_ms = 0, gp_head_ms = 0;   // outside every layer: the embed + first norm / the output head
    long long gp_other_n = 0;
    double gp_last_total_ms = 0;  // the last drain's total: gspan + embed + head
    /// One boundary on st, only under the gate (so the default path gains no statement).
    void ev_mark(int i) {
        if (gpu_timing && gpu_timing_open) CK(cudaEventRecord(ev_seg_ev[i], st));
    }
    /// cudaEventElapsedTime between two boundaries into the accumulator, and the pair's span into the tile.
    void gp_push(int a, int b, double ms) {
        gseg_ms[b - a] += ms;
        ++gseg_n[b - a];
        gspan_ms += ms;
        ++gspan_n;
    }
    /// The one cudaEventSynchronize per token (the DRAIN), which is also the only place the elapsed times are
    /// read.  Reading one event from the very back while the stream is still busy blocks the host on it - which
    /// is exactly the 142 us/layer prebegin cost this probe exists to attribute, measured again instead of
    /// assumed, so the end of a tile is only ever read here, with everything ahead of it known complete.
    void gp_drain() {
        // Only a T==1 forward records a tile: with 40 layers in one forward the 8 shared events would be
        // overwritten by the last layer before they are read, so n>1 could not be measured with them at all.
        // That is fine -- the number being explained is the single-token decode step.
        const int n = (int) gseg_ring.size();
        if (n != 1) { gseg_ring.clear(); gseg_dropped += n; return; }
        gpu_timing_open = false;
        CK(cudaEventSynchronize(ev_seg_ev[kSegComb]));   // the ONLY synchronization this probe adds
        float ms = 0;
        for (int j = 0; j + 1 < kGsegN; ++j) {
            CK(cudaEventElapsedTime(&ms, ev_seg_ev[j], ev_seg_ev[j + 1]));
            gp_push(j, j + 1, (double) ms);
        }
        gseg_ring.clear();
        {   // the residual of the whole token, from the accumulators: the tile minus the segments that tile it
            double sum = 0;
            for (int j = 0; j + 1 < kGsegN; ++j) sum += gseg_ms[j];
            gp_resid_ms = gspan_ms - sum;
        }
        gp_last_total_ms = gspan_ms + gp_embed_ms + gp_head_ms;
    }
    void gp_reset() {
        gseg_ring.clear();
        gseg_dropped = 0;
        for (int j = 0; j < kGsegN; ++j) { gseg_ms[j] = 0; gseg_n[j] = 0; }
        gspan_ms = 0; gspan_n = 0; gp_resid_ms = 0;
        gp_embed_ms = 0; gp_head_ms = 0; gp_other_n = 0; gp_last_total_ms = 0;
    }
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
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(8);
        }
    }

    /// One session's buffers: its recurrent state per GDN layer and its K / V cache per attention layer (the MTP
    /// block's at the end).  The K / V stores follow --kv (kv_k8 / kv_v8, independent): a store that is int8 holds
    /// codes plus one FP16 scale per kKVG values and the FP32 pointer for it stays null, and the reverse for a
    /// FP32 store.  Every state index, the MTP block's own K / V at n_layers included, takes the same stores.
    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t ncells = (size_t) g.n_head_kv * ctx * g.head_dim;
        const size_t nscales = (size_t) g.n_head_kv * ctx * ((size_t) g.head_dim / kKVG);
        auto push_kv = [&](bool attn) {
            if (!attn) {
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
                ss.kc8.push_back(nullptr);
                ss.vc8.push_back(nullptr);
                ss.ks8.push_back(nullptr);
                ss.vs8.push_back(nullptr);
                return;
            }
            ss.kc.push_back(kv_k8 ? nullptr : dalloc(ncells));
            ss.vc.push_back(kv_v8 ? nullptr : dalloc(ncells));
            ss.kc8.push_back(kv_k8 ? dalloc_t<int8_t>(ncells) : nullptr);
            ss.vc8.push_back(kv_v8 ? dalloc_t<int8_t>(ncells) : nullptr);
            ss.ks8.push_back(kv_k8 ? dalloc_t<__half>(nscales) : nullptr);
            ss.vs8.push_back(kv_v8 ? dalloc_t<__half>(nscales) : nullptr);
        };
        for (const Layer& L : m->layers) {
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                push_kv(true);
            } else {
                ss.conv.push_back(dalloc(C * (g.ssm_d_conv - 1)));
                ss.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                push_kv(false);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            push_kv(true);
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
        kc8 = ss.kc8; vc8 = ss.vc8; ks8 = ss.ks8; vs8 = ss.vs8;
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
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
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
        att_s = dalloc(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        k16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        v16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
        // the KV stores (--kv / STRATA_Q36_KV, resolved by the command line): int8 needs head_dim to be a kKVG multiple
        if ((kv_k8 || kv_v8) && (int) g.head_dim % kKVG != 0) {
            std::fprintf(stderr, "strata-q36: head_dim %d is not a multiple of %d: the KV cache stays FP32\n",
                         (int) g.head_dim, kKVG);
            kv_k8 = kv_v8 = false;
        }
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
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, 64));   // 64 query tokens per attention batch
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        // The K / V caches belong to the sessions (alloc_session): conv / S / kc / vc / kc8 / vc8 / ks8 / vs8 are
        // the CURRENT session's, load_cur() swaps them.  Only the verify windows' GDN inputs are one set here,
        // whichever session runs.
        for (const Layer& L : m->layers) {
            if (L.attn) {
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                v_qkv.push_back(dalloc((size_t) kVerify * C));
                v_hc.push_back(dalloc((size_t) kVerify * C));
                v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
            }
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
        if (m->has_mtp) {   // the MTP block's own buffers (its K / V went with the sessions, in alloc_session)
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            // STRATA_Q36_MTP_MAPPED: one position word per chain step, so a step reads pos0_m + d at EXECUTION time and
            // a pre-submitted step cannot be handed the next step's position.  A single shared word is a
            // use-after-write race here (the host would rewrite it while the queued steps have not read it), and it
            // would replay every chained draft onto one MTP K / V row.
            pos0_m = dalloc_t<int32_t>(mtp_mapped() ? kChainMax : 1);
            pick_m = dalloc_t<int32_t>(1);
            // The published picks: HOST-MAPPED, so the device writes them straight into host memory and the host reads
            // them there - no cudaMemcpy per draft.  q36's only pinned buffer (hpin) is cudaHostAllocDefault, which is
            // not mapped, so this is the engine's first mapped allocation: through the pinned budget, and a refusal
            // turns the flag off with one line rather than ending the load.
            if (mtp_mapped()) {
                void* h = nullptr;
                if (strata::core::host_alloc_guarded(&h, (size_t) kChainMax * sizeof(int32_t),
                                                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                    (void) cudaGetLastError();
                    std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: the %d-byte mapped draft-pick buffer was "
                                         "refused; the flag is off and the draft loop is the one it always was\n",
                                 (int) ((size_t) kChainMax * sizeof(int32_t)));
                } else if (cudaHostGetDevicePointer((void**) &pick_h_d, h, 0) != cudaSuccess) {
                    (void) cudaGetLastError();
                    CK(cudaFreeHost(h));
                    std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_MAPPED=1: the mapped draft-pick buffer is not "
                                         "addressable from the device; the flag is off\n");
                } else {
                    pick_h = (int32_t*) h;
                }
            }
        }
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
            const size_t ns = sess[0].conv.size();
            std::vector<float*> hc(ns * (size_t) n_sess), hs(hc), hk(hc), hv(hc);
            std::vector<int8_t*> h8k(hc.size()), h8v(hc.size());
            std::vector<__half*> h8ks(hc.size()), h8vs(hc.size());
            for (size_t li = 0; li < ns; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hc[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                    h8k[li * n_sess + s_] = sess[(size_t) s_].kc8[li];
                    h8v[li * n_sess + s_] = sess[(size_t) s_].vc8[li];
                    h8ks[li * n_sess + s_] = sess[(size_t) s_].ks8[li];
                    h8vs[li * n_sess + s_] = sess[(size_t) s_].vs8[li];
                }
            const std::pair<void***, std::vector<void*>*> tabs[8] = {
                {(void***) &d_conv_tab, (std::vector<void*>*) &hc}, {(void***) &d_S_tab, (std::vector<void*>*) &hs},
                {(void***) &d_kc_tab, (std::vector<void*>*) &hk},   {(void***) &d_vc_tab, (std::vector<void*>*) &hv},
                {(void***) &d_kc8_tab, (std::vector<void*>*) &h8k}, {(void***) &d_vc8_tab, (std::vector<void*>*) &h8v},
                {(void***) &d_ks8_tab, (std::vector<void*>*) &h8ks},{(void***) &d_vs8_tab, (std::vector<void*>*) &h8vs}};
            for (const auto& tb : tabs) {
                CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(void*)));
                CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(void*),
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
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d, KV cache K %s / V %s%s, %.2f GiB of VRAM free\n",
                     ctx, tmax, kv_k8 ? "int8" : "fp32", kv_v8 ? "int8" : "fp32",
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
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
    void group_old(const Layer& L, const int32_t* slot_of, const uint8_t* skip, int T, unsigned long long* o_ptr, int32_t* o_start,
                   int32_t* o_ng, int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of && skip)
            k_group_hy_masked<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, skip, L.blobs, L.stride, o_ptr,
                                                      o_start, o_ng, o_dst, o_tok);
        else if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok);
        else if (skip)
            k_group_masked<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, skip, L.blobs, L.stride, o_ptr, o_start,
                                                   o_ng, o_dst, o_tok);
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
        group_old(L, slot_of, nullptr, T, o_ptr, o_start, o_ng, o_dst, o_tok);
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
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  `skip`: the helper tier's per-entry
    /// mask (--expert-cache-device1..3, null when no helper took part), the entries it already computed into `eparts`.
    /// Returns false when the old kernels must run.
    bool route_group(const Layer& L, const int32_t* slot_of, const uint8_t* skip, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_k || T > 8 || m->g.n_expert != 256 || Kp != 8) return false;
        if (slot_of && skip)
            q36k::k_route_group<true, true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, skip, L.blobs, L.stride, grp_ptr,
                                                                       grp_start, n_groups, ent_dst, ent_tok);
        else if (slot_of)
            q36k::k_route_group<true, false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, nullptr, L.blobs, L.stride, grp_ptr,
                                                                        grp_start, n_groups, ent_dst, ent_tok);
        else if (skip)
            q36k::k_route_group<false, true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, skip, L.blobs, L.stride, grp_ptr,
                                                                        grp_start, n_groups, ent_dst, ent_tok);
        else
            q36k::k_route_group<false, false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, nullptr, L.blobs, L.stride, grp_ptr,
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
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy; with
    /// `skip` the helper tier's entries left out).
    void group_entries(const Layer& L, const int32_t* slot_of, const uint8_t* skip, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of && skip)
                q36k::k_group_small<true, true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, skip, L.blobs, L.stride, grp_ptr, grp_start,
                                                                 n_groups, ent_dst, ent_tok);
            else if (slot_of)
                q36k::k_group_small<true, false><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                                  n_groups, ent_dst, ent_tok);
            else if (skip)
                q36k::k_group_small<false, true><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, skip, L.blobs, L.stride, grp_ptr, grp_start,
                                                                   n_groups, ent_dst, ent_tok);
            else
                q36k::k_group_small<false, false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                                    n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, skip, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout,
                 int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        // DIAG: the combine's four inputs for row 0.  A NaN in the parts (eparts) row, the weights, the shared
        // expert's output or the gate names the culprit.  Remove after the fix.
        if (peer_prefill_on && T > 8) {
            CK(cudaStreamSynchronize(st));   // DIAG: st is non-blocking, so a plain read would race the engine
            float w4[8] = {}, sh4[4] = {}, sg1[1] = {}, p4[4] = {};
            CK(cudaMemcpy(w4, ewts, 8 * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sh4, shared, 4 * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sg1, sgate, 4, cudaMemcpyDeviceToHost));
            int32_t pr[8] = {};
            CK(cudaMemcpy(pr, prow, 8 * 4, cudaMemcpyDeviceToHost));
            for (int k = 0; k < 4; ++k) {
                if (pr[k] < 0) continue;
                CK(cudaMemcpy(p4 + k, eparts + (size_t) pr[k] * N, 4, cudaMemcpyDeviceToHost));
            }
            std::fprintf(stderr, "DIAG moe_end T=%d prow[0..3]=%d %d %d %d ewts[0..3]=%g %g %g %g parts=", T, pr[0],
                         pr[1], pr[2], pr[3], w4[0], w4[1], w4[2], w4[3]);
            for (int k = 0; k < 4; ++k) std::fprintf(stderr, " %g", p4[k]);
            std::fprintf(stderr, " shared[0..3]=%g %g %g %g sgate=%g\n", sh4[0], sh4[1], sh4[2], sh4[3], sg1[0]);
        }
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
        // STRATA_Q36_HELPER_TIMING=1: E0, the start of this layer's tile.  Only opens a tile when the probe is on
        // and no graph is being captured; every other mark below is a no-op unless this one opened it.
        // Only ONE layer of ONE forward carries a tile: the 8 shared events cannot survive 40 layers of records,
        // and the number being explained is the single-token decode step (T==1).  gp_open_this_forward is set by
        // forward() for the LAST layer only, so exactly one tile per forward is ever queued.
        if (gpu_timing && gpu_open_this_forward && !gpu_timing_hold) { gpu_timing_open = true; }
        ev_mark(kSegAttnEsm);
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
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
                    k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                      be_b, gdn_y, HK, HV, T);
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
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
                rs.kc8 = d_kc8_tab + li * n_sess;
                rs.vc8 = d_vc8_tab + li * n_sess;
                rs.ks8 = d_ks8_tab + li * n_sess;
                rs.vs8 = d_vs8_tab + li * n_sess;
            }
            kv4([&](auto k8c, auto v8c) {
                constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                k_attn_prep<K8, V8><<<dim3(H + HKV, T), kThreads, 0, st>>>(
                    qg, kx, vx, L.q_norm, L.k_norm, q, gate,
                    K8 ? (rows_mode ? nullptr : (void*) kc8[li]) : (rows_mode ? nullptr : (void*) kc[li]),
                    V8 ? (rows_mode ? nullptr : (void*) vc8[li]) : (rows_mode ? nullptr : (void*) vc[li]),
                    K8 ? ks8[li] : nullptr, V8 ? vs8[li] : nullptr, H, HKV, HD, m->n_rot, dpos, ctx, m->rope_base,
                    m->eps, d_delta, mtab_now, rs);
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
                kv4([&](auto k8c, auto v8c) {
                    constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                    k_kv_f16<K8, V8><<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(
                        K8 ? (const void*) kc8[li] : (const void*) kc[li],
                        V8 ? (const void*) vc8[li] : (const void*) vc[li], K8 ? ks8[li] : nullptr,
                        V8 ? vs8[li] : nullptr, k16, v16, ctx, P, HD, HKV);
                });
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = k16 + (size_t) kvh * P * HD;
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
                        const __half* Vh = v16 + (size_t) kvh * P * HD;
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
                k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    kv4([&](auto k8c, auto v8c) {
                        constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                        k_attn_part<K8, V8><<<dim3(nc, nt * H), kChunkCells, 0, st>>>(
                            q, K8 ? (rows_mode ? nullptr : (const void*) kc8[li])
                                  : (rows_mode ? nullptr : (const void*) kc[li]),
                            V8 ? (rows_mode ? nullptr : (const void*) vc8[li])
                                  : (rows_mode ? nullptr : (const void*) vc[li]),
                            K8 ? ks8[li] : nullptr, V8 ? vs8[li] : nullptr, dpos, t0, H, HKV, ctx,
                            1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    });
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs);
                }
            }
            mm(L.wo, att, y, T);
        }
        add_rms_post(L, T);
        ev_mark(kSegRouter);   // E1: attention and SSM are behind us; the router follows
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
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, nullptr, T,
                                      ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        helper_mask_li = -1;   // set below, when this layer's window actually goes to a helper
        ev_mark(kSegDisp);      // E2: routing done; the route D2H and moe_helpers' host wait follow
        if (hy_rows) {
            // ONE copy of `ids` and `h` for this layer, issued ahead of every host block.  moe_helpers' window
            // (h_helper_ids/h_helper_x) and moe_hybrid_cpu's routing (h_ids/h_h) read the same two device buffers
            // and nothing writes `ids` or `h` between them, so the pair moe_hybrid_gpu used to issue after
            // moe_helpers had already issued its own is now issued once here, in front of both.  moe_helpers
            // stages its window from h_ids/h_h (see its staged_x/staged_ids parameters) and moe_hybrid_gpu no
            // longer copies at all.
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
            // A helper GPU takes the entries it holds in its VRAM cache (MAXT windows only); the local hybrid path
            // below takes the primary's own slots and the host pool's, and no entry is computed twice: the three
            // buckets are separated at run time by the per-entry mask moe_helpers() writes (the grouping skips the
            // helper's entries, moe_hybrid_cpu's h_miss skips them too).  With no helper this is the plain slots +
            // host pool split it always was.  Note this is a split of WORK, not of arithmetic: the helper and the
            // CPU pool are two different implementations, so their rows differ (see moe_helpers).
            if (helpers_enabled() && helper_layer(li) && T <= strata::kernels::cpu::MAXT && !helper_failed)
                moe_helpers(li, T, h_h, h_ids);   // the two copies are issued: stage from them, no D2H of its own
            moe_hybrid_gpu(L, li, T, h_ids, h_h);   // the same copies feed the CPU pool: no D2H of its own either
            ev_mark(kSegGrp);    // E3: the local grouped kernel follows
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row

            // a helper GPU owns this layer's experts, but a prompt chunk outgrows the helper's MAXT window: MMQ
            // (int8, the model's own expert blobs read on this card) and the helper's q8_1 cache cannot serve it
            // through RemoteExperts::begin/finish, so the default route runs it locally.  Warn once - the helper
            // must not idle silently; the decode windows (T <= MAXT) still offload through the branches above.
            // STRATA_Q36_PREFILL_PEERS replaces that warning with an actual split (see the block below).
            //
            // Over-cap fallback: the `cap_rows` rule of the Flash-Next prompt path (src/prefill/prefill.cpp:3826-3834,
            // where an expert whose rows would overflow the peer's row budget stays on the primary and is counted),
            // at the ONE granularity q36 can still honour - the whole layer.  It cannot be taken per entry, and that
            // is the reason it is here and not in peer_plan(): the two route kernels below have already been issued
            // against peer_mask(li) and skip per EXPERT, so by plan time a helper's experts are already missing from
            // the primary's sorted order, and handing one of their entries back would leave that eparts row holding
            // whatever the previous layer left there - silently wrong logits, not a slower run.  Nor is it a test
            // local to this branch: stream_kick() already skipped uploading these experts' blobs, and the route
            // kernels are about to skip reading them, so the fallback has to be the SAME answer for peer_mask(),
            // peer_mask_host() and this bool.  That is why the test lives inside peer_layer() - the one predicate all
            // three read - against the window `peer_win_ent` forward() records once.  With `cap_entries == tmax *
            // top_k` (peer_open's window) and every chunk capped at tmax the branch is unreachable today; it is here
            // so a future narrower window, or a bigger chunk, falls back to today's route instead of ending the run
            // at peer_plan()'s own size check.  Inert with the flag off: peer_owned_layer() answers false first and
            // neither peer_window_fits() nor the counter is reached.
            const bool peer_here = peer_layer(li);
            if (peer_owned_layer(li) && !peer_here) ++peer_over_cap;
            if (helpers_enabled() && helper_layer(li) && !mmq_helper_warned && !peer_here)
                std::fprintf(stderr, "strata-q36: a prompt MMQ chunk keeps a helper-owned layer local; its experts stay idle on the helper GPU (MMQ int8 vs the helper q8_1 cache; T=%d > MAXT=%d)\n", T, (int) strata::kernels::cpu::MAXT);
            if (helpers_enabled() && helper_layer(li) && !peer_here) mmq_helper_warned = true;
            const int ne = (int) g.n_expert;
            // STRATA_Q36_PREFILL_PEERS: the entries whose expert a helper holds are NOT in the sorted order this
            // launch builds (both route kernels skip them with the same per-expert byte), so `bounds` covers the
            // primary's own entries only and `eparts` rows [0, n_local) are the ones this MMQ writes.  With the
            // flag off `owned` is null and both kernels run exactly the statements they always ran.
            const uint8_t* owned = peer_mask(li);
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt, owned);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv, owned);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            // The peer plan needs the entries' expert ids on the host, and this layer already brings one word back
            // (rmax) and then blocks: the ids ride in the SAME D2H pump and land before the same sync, so the plan
            // costs ONE copy folded into an existing synchronisation - not a second sync per layer.  Issued only
            // on a served layer, so a default / unserved layer copies exactly the bytes it always did.
            if (peer_here) CK(cudaMemcpyAsync(h_peer_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            // The peer split, when this layer is served: it numbers every entry's eparts row (prow2, moe_end's map),
            // builds the local id table, and launches each helper's share on the helper's own stream - so the
            // helpers' arithmetic overlaps everything from here to the wait before moe_end.  A failure ends the run:
            // the plan is the one place a peer bug can silently corrupt logits, and a wrong answer is worse than a
            // failed run.
            if (peer_here) peer_plan_or_exit(li, T, h, ids, eparts);
            // DIAG: the primary's own copy of this layer's first expert blob (the hybrid mode's host arena) and the
            // stride between experts.  The helper's cached copy of the same expert must be byte-identical, or the
            // MMQ reads a different weight format than the layout says.  Remove after the fix.
            if (peer_prefill_on && peer_here && li == 0) {
                const uint8_t* hb = L.hblobs;
                std::fprintf(stderr, "DIAG primary layer 0 expert 0 bytes:");
                for (int i = 0; i < 16; ++i) std::fprintf(stderr, " %d", hb[i]);
                std::fprintf(stderr, "\nDIAG primary stride=%lld bytes=%lld gu_type=%d d_type=%d\n",
                             (long long) L.stride, (long long) L.L.bytes, L.L.gu_type, L.L.d_type);
            }
            // The two numbers the local MMQ reads, each one exactly what the plain path used when the flag is off
            // (`n_local` is n_ent, `lid` the identity table) - so the six statements below are the six they were.
            const int64_t n_local = peer_here ? peer_n : n_ent;
            const int32_t* lid = peer_here ? peer_lid_map : ident;
            // The activation image the local MMQ reads.  Rows [0, n_local) of mq_x are the primary's own entries in
            // sorted-expert order - with no helper every entry, on a served layer the ones the two route kernels did
            // not skip.  `ent_tok` maps sorted row -> token and is already right for both (the skipped rows' entries
            // are never read: `bounds` stops at n_local), so quantize's arguments do not change, only its row count.
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_local, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            // gu keeps `ident`: it writes mq_gu in SORTED row order, which is the space swiglu and the down
            // product's activation rows live in.  Only the down product writes into `eparts`, so only it renumbers.
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_local; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_local, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_local, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = lid;
            dn.total_rows = n_local; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            // DIAG: after each served layer, check the caller's first token.  A change means this layer's host
            // copies overran into the caller's vector.  Remove with the other DIAG prints.
            if (diag_toks != nullptr && peer_here && diag_toks[0] == 2147483647) {
                std::fprintf(stderr, "DIAG corrupted after layer %zu\n", li);
                diag_toks = nullptr;
            }
            // DIAG: where the NaN appears in this layer's MMQ.  mq_x is the quantized activation (h and ent_tok went
            // into it), mq_h the SwiGLU output, mq_hq its quantized image, and wbase the weight rows the local MMQ
            // reads.  Remove after the fix.
            if (peer_prefill_on) {
                CK(cudaStreamSynchronize(st));   // DIAG: st is non-blocking, so a plain read would race the engine
                float q4[4] = {}, sh4[4] = {}, hq4[4] = {};
                CK(cudaMemcpy(q4, mq_x, 4 * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(sh4, mq_h, 4 * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hq4, mq_hq, 4 * 4, cudaMemcpyDeviceToHost));
                std::fprintf(stderr, "DIAG mmq layer %zu n_local=%lld mq_x[0..3]=%g %g %g %g mq_h[0..3]=%g %g %g %g "
                                     "mq_hq[0..3]=%g %g %g %g\n",
                             li, (long long) n_local, q4[0], q4[1], q4[2], q4[3], sh4[0], sh4[1], sh4[2], sh4[3], hq4[0],
                             hq4[1], hq4[2], hq4[3]);
            }
            if (lhy(li)) stream_done();
            // With no helper `prow` stays `rinv`, the map it has always been; on a served layer peer_plan() set it
            // to the per-entry eparts row map the executor's contract defines.
            prow = peer_here ? peer_prow2 : rinv;
        } else {
            // A helper computes the entries it holds (the window fits begin()'s MAXT); the local grouped kernel
            // below computes every other entry.  Without the flags helpers_enabled() is false, so this is the same
            // grouping + grouped launch as it always was.
            const bool via_helper = helpers_enabled() && helper_layer(li) && T <= strata::kernels::cpu::MAXT &&
                                    !helper_failed && moe_helpers(li, T);
            if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
            // route_group ran before moe_helpers (it makes the `ids` moe_helpers reads), so its grouping covers
            // every entry.  With a helper's share, group again with the mask so this launch takes only the rest.
            if (via_helper) grouped_by_route = false;
            group_entries(L, nullptr, via_helper ? d_helper_mask : nullptr, T);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
        }
        // the shared expert, scaled by sigmoid(w . h)
        ev_mark(kSegShared);    // E4: the local grouped expert GEMM is issued; the shared expert follows
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
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        ev_mark(kSegCpu);       // E5: shared expert issued; the CPU pool's work follows on the same queue
        // STRATA_Q36_HELPER_ASYNC: deferred helper completion into eparts, after this layer's own work
        // (local grouped, shared expert, CPU pool) and before moe_end.  Flag-off / sync path: no-op.
        moe_helpers_wait(li);
        // STRATA_Q36_PREFILL_PEERS: the helper launches peer_plan() issued must have written their eparts rows before
        // the combine reads them.  No-op unless a peer layer was planned for this layer; lands in the same place the
        // existing deferred helper wait does.  Flag-off: nothing here runs at all.
        if (peer_wait_n != 0) peer_prefill_wait();
        ev_mark(kSegMerge);     // E6: the CPU pool's merge is issued; moe_helpers_wait's merge follows
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
        ev_mark(kSegComb);      // E7: this layer's tile is complete
        // DIAG: after every layer, check the caller's first token.  A change means a host copy in this layer
        // overran into the caller's vector.  Remove with the other DIAG prints.
        if (diag_toks != nullptr && diag_toks[0] == 2147483647) {
            std::fprintf(stderr, "DIAG corrupted after layer %zu peer_wait_n=%d\n", li, peer_wait_n);
            diag_toks = nullptr;
        }
        if (gpu_timing_open && !gpu_timing_hold) gseg_ring.push_back(Gseg{});
        gpu_timing_open = false;
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        // STRATA_Q36_PREFILL_PEERS: the window this forward works on, recorded before stream_begin() consults it (see
        // peer_win_ent).  Flag-guarded, so no write happens at all in a default run and nothing reads it there either
        // (peer_layer() answers false on `!peer_prefill_on` alone).  It has to be recorded HERE rather than in the MMQ
        // branch: stream_kick() has already skipped uploading the helpers' experts by then, so the over-cap answer
        // must be the same at both ends of the forward.
        if (peer_prefill_on) peer_win_ent = (int64_t) T * m->top_k;
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
            // STRATA_Q36_HELPER_TIMING=1: arm ONLY the last layer to carry the tile.  The probe's 8 events are
            // shared and reused, so one forward can only ever describe one layer; the last one is chosen because
            // its MoE reads eparts the same way the others do, and the whole point is to know where the
            // decode step's time goes, which is the same story in every layer.
            gpu_open_this_forward = last && T == 1;
            layer(m->layers[l], l, T, p0, pos0, graph_mode, verify, last ? m->out_norm : m->layers[l + 1].attn_norm,
                  last ? hmain : h);
            // DIAG: the hidden state after every layer, to find the first one that goes bad.  Remove after the fix.
            if (peer_prefill_on && T > 8) {
                CK(cudaStreamSynchronize(st));
                float hv[4] = {};
                CK(cudaMemcpy(hv, last ? hmain : h, 4 * 4, cudaMemcpyDeviceToHost));
                const bool bad = !(hv[0] == hv[0]) || !(hv[1] == hv[1]) || !(hv[2] == hv[2]) || !(hv[3] == hv[3]);
                if (bad || l >= 2)
                    std::fprintf(stderr, "DIAG layer %zu h[0..3]=%g %g %g %g\n", l, hv[0], hv[1], hv[2], hv[3]);
            }
        }
        gpu_open_this_forward = false;
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
        // STRATA_Q36_HELPER_TIMING=1: the one armed layer has recorded its tile, so this is the ONE place the probe
        // drains and reads elapsed times.  It syncs the last event once per forward, not per layer.
        if (gpu_timing) gp_drain();
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
    // STRATA_Q36_MTP_TOP2=1 (diagnostic, opt-in): count, per draft position, how often a REJECTED draft's runner-up was
    // the token verify actually picked.  High => the drafter ranks the right token second (a head/quantisation problem);
    // low => the drafter is fed the wrong input (a seed/position problem).  Off by default: with the flag unset none of
    // this runs and no buffer is allocated.
    bool mtp_top2() const {
        static const bool on = [] {
            const char* v = std::getenv("STRATA_Q36_MTP_TOP2");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return on;
    }
    float* h_logits = nullptr;          // host staging for the draft head's logits (diagnostic path only)
    int32_t mtp_top2_tok = -1;          // the runner-up token of the draft just computed (-1 = unknown)
    int32_t mtp_top2_prev[16] = {};     // the runner-up of each draft in the current window
    long long dp_tested[16] = {}, dp_ok[16] = {};   // per position: drafts verified / accepted
    long long t2_rej[16] = {}, t2_hit[16] = {};     // per position: rejections / runner-up was the picked token
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        // Fast-path gate.  With draft_n == 0 (the default) the whole-vocabulary head_rows path runs instead; either
        // way the diagnostics below still run, because they read the SAME logits buffer and the same pick.
        const bool fast = fast_k && draft_n > 0 && draft_n < g.n_vocab && draft_n % 8 == 0 && hd.type == 8 &&
                          hd.n_in == N && N / 32 == 64;
        if (!fast) {
            head_rows(hn, 1, logits_m, pick_m);
        } else {
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
        }
        // STRATA_Q36_MTP_TOP2=1 (diagnostic, opt-in): also keep the RUNNER-UP of this draft, so a rejected draft can be
        // asked whether the second choice would have been the token verify picked.  A head that is merely off by one
        // rank shows up as a high runner-up hit rate; a head fed the wrong input does not.  Same role as upstream's
        // STRATA_MTP_TOP2 (src/core/mtp.cpp:520), which q36 cannot share: this engine has its own drafter.
        if (mtp_top2()) {
            const int nv = draft_n > 0 ? draft_n : (int) g.n_vocab;
            CK(cudaMemcpyAsync(h_logits, logits_m, (size_t) nv * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            int64_t b1 = -1, b2 = -1;
            for (int64_t i = 0; i < nv; ++i) {
                const float v = h_logits[(size_t) i];
                if (b1 < 0 || v > h_logits[(size_t) b1]) { b2 = b1; b1 = i; }
                else if (b2 < 0 || v > h_logits[(size_t) b2]) b2 = i;
            }
            mtp_top2_tok = (int32_t) (b2 < 0 ? -1 : b2);
        }
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *dpos) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`), and - with
    /// `pub_slot` >= 0 - the pick is ALSO written to tok_m[0] and published into host-mapped memory at pub_slot, which
    /// is what lets the next chained step be submitted before this one finishes (STRATA_Q36_MTP_MAPPED).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode, const int32_t* dpos,
                     int pub_slot = -1) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(m->mtp, m->layers.size(), T, p0, dpos, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) {
            head_draft(hmtp + (size_t) (T - 1) * N);
            // STRATA_Q36_MTP_MAPPED: the pick becomes the next step's input token on the DEVICE, and the host's report
            // comes from mapped memory.  One launch, right after the head: no host round trip is on the path between
            // two chained steps any more.
            if (pub_slot >= 0) q36k::k_chain_publish<<<1, 32, 0, st>>>(tok_m, pick_m, pick_h_d, pub_slot);
        }
    }

    /// Chained draft step d of the current window (STRATA_Q36_MTP_MAPPED only): the block's own last-row output as the
    /// seed, its own position word pos0_m + d, and its pick published into mapped slot d.  The graph is per depth d
    /// (gm1[d - 1]): `d` and the position-word offset are baked at capture, so one shared exec would replay every
    /// step onto the same position and the same MTP K / V row.
    void chain_step(int d) {
        float* seeds = mtp_seed();
        if (mtp_graph) graph_run(gs().gm1[d - 1], [&] { forward_mtp(1, pos + d, seeds, true, use_graph, pos0_m + d, d); });
        else forward_mtp(1, pos + d, seeds, true, false, pos0_m + d, d);
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
        // STRATA_Q36_PREFILL_PEERS: the experts a helper GPU holds are computed there, so their blobs must not be
        // uploaded here - that upload IS the cost this feature removes.  The hole it leaves in `stage[b]` is never
        // read: the primary's MMQ on a served layer only computes the entries of the experts nobody else holds (the
        // two route kernels skip the rest), so it addresses only non-owned experts' weights.  The table is static
        // for the run (a helper's cache is filled at open time), so it is read from the host copy - no D2H here.
        const uint8_t* owned = peer_mask_host(stream_seq[(size_t) i]);
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs).  A whole-layer copy cannot skip
        // anything, so on a served layer both shortcuts fall through to the per-run loop (which then skips).
        if ((all || 2 * L.n_slots < ne) && owned == nullptr) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int x = 0; x < ne;) {
                // an expert a helper holds is never covered by a copy: the run stops before it and resumes after it
                if (owned != nullptr && owned[(size_t) x] != 0) {
                    ++x;
                    continue;
                }
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1 &&
                           (owned == nullptr || owned[(size_t) y] == 0)) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0 && (owned == nullptr || owned[(size_t) y] == 0)) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
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

    /// Every routed entry of a helper-owned layer, computed on that helper GPU (T <= MAXT windows only: begin() takes
    /// a window of at most MAXT tokens, so a bigger batch keeps the local path).  The rows land in `eparts` exactly
    /// where the local grouped kernel would have put them - `RemoteExperts::begin` keys its output on the entry
    /// index it was given (tok * k + j), which is `eparts`' own row - so `k_moe_out` combines one way either.
    /// A helper takes only the (layer, expert) pairs it really holds in its VRAM cache, not the whole layer: a P104
    /// holds about 4630 of a stripe's 10240 experts, and the entries it missed must still be computed by the local
    /// slots and the host pool, or the layer's output is wrong.  This is Qwen3.8 Flash's per-entry split (kind[i] =
    /// 2 the helper, 0 the primary's own cache, -1 the host pool); the buckets are kept apart at RUN time, by the
    /// per-entry mask - not at open time.  open_helpers() passes an empty_primary cache (0 slots), so
    /// RemoteExperts::open()'s `primary.slot_of(...) < 0` test is vacuously true and admits any ranked pair; a
    /// helper may therefore hold an expert the primary also holds a slot for, and what keeps the two off the same
    /// entry is the mask: the local grouping skips the helper's entries (moe_hybrid_gpu's `skip`) and the CPU pool
    /// skips them too (moe_hybrid_cpu's h_miss).
    ///
    /// The row placement above is exact, but the VALUES are not the local path's arithmetic.  A helper quantizes the
    /// activation q8_1 and runs native_expert_grouped; config A's primary runs the same grouped kernel for the
    /// experts it holds and ggml-cpu AVX for the rest.  So enabling a helper moves batch entries between two
    /// different (each internally consistent) implementations, and the token stream diverges accordingly - measured
    /// on 192.168.1.98 with a forced 36-token prompt (identical tokens, so every logit difference is arithmetic):
    /// the per-row RMS logit difference is 1.5e-1 to 3.3e-1 (relative ~5e-2) even where argmax still agrees, about
    /// 10 orders of magnitude above the ~1.7e-14 an fp32 reassociation of this sum could produce.  Nothing here
    /// reorders or splits the final combine (k_moe_out accumulates all K rows on the primary in pick order, and
    /// moe_helpers_wait() lands every helper row before moe_end), so this is a change of implementation, not a
    /// change of summation.  Forcing it to match would mean running the CPU pool's arithmetic on the helper GPUs,
    /// i.e. giving up the entire speedup, so it is not done; `--kv int8` is a second, independent instance of the
    /// same thing (an opt-in lossy cache, which flips argmax at row 4 where top1-top2 = 0.001).
    ///
    /// False (with the reason) means a helper errored: `helper_failed` is set and the mask uploaded all-zero, so
    /// the local paths redo the whole window and rewrite every row.
    bool moe_helpers(size_t li, int T, const float* staged_x = nullptr, const int32_t* staged_ids = nullptr) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        // The helpers of this layer's window: one on the default path (the helper helper_of_layer names) and
        // 2..3 with STRATA_Q36_HELPER_PARALLEL, where every helper holds a share of every layer's experts.
        int win[3] = {-1, -1, -1};
        const int nwin = helper_window(li, win);
        helper_mask.assign((size_t) n_ent, 0);   // the helpers take what they hold; the local paths take the rest
        helper_wait_n = 0;
        helper_wait_hi[0] = helper_wait_hi[1] = helper_wait_hi[2] = -1;
        helper_wait_ent = 0;
        helper_wait_N = 0;
        if (nwin <= 0) return true;
        // ONE COPY, TWO CONSUMERS (staged_* set by the hy_rows call site): this caller issues the D2H of `ids`/`h`
        // into `staged_ids`/`staged_x`, so staging the helper's window here is a HOST copy of the bytes that
        // landed - at decode (T=1, n_embd 2048, top_k 8) n_ent*4 + T*N*4 = 32 B + 8192 B = 8.2 KiB, roughly 1-2 us,
        // against a second D2H (two more driver submissions per layer, queued in front of this host block) and
        // 8.2 KiB more PCIe traffic per layer.  RemoteExperts::begin only reads x and ids (it memcpys
        // them into the helper's own pinned staging, remote_experts.cpp:433, and scan ids above it) and never keep
        // the caller's pointers, so the copy is done before begin returns and the staging buffers are free to be
        // overwritten by the next layer.
        const bool staged = staged_x != nullptr && staged_ids != nullptr;
        if (!staged) {
            CK(cudaMemcpyAsync(h_helper_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_helper_x, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        }
        // Wait for these TWO copies, not for the queue to drain.  cudaStreamSynchronize(st) here waits for
        // everything the host had already issued for this layer - the attention/SSM kernels, the router GEMM and
        // native_router_topk - which measured 135 us per layer (5.4 ms/token over 40 layers on 192.168.1.98,
        // binary 20702aa4).  RemoteExperts::begin only reads the two HOST buffers this filled (begin_slot does
        // std::memcpy(w.h_x_, x, ...) and scans ids on the CPU, remote_experts.cpp:433 and :394-405), so it
        // never touches device memory, but the drain is what makes the bytes readable and it is kept as-is: a
        // narrower wait (an event recorded just after the copies) measured 5.4 -> 5.7 ms/token, i.e. slightly
        // worse, so it was deleted rather than kept as an option.
        // The staged path waits for the copies the CALLER issued (same stream, earlier position): the wait covers
        // them plus whatever the host issued between, which for hy_rows is nothing.
        //
        // STRATA_Q36_HELPER_TIMING=1 times that wait and off does no clock work: the clock statements sit on an
        // already-taken helper_timing() branch, so with every flag unset not one statement runs that did not run
        // before.
        if (helper_timing()) {
            const auto t_prebegin = Clock::now();
            CK(cudaStreamSynchronize(st));
            d_prebegin_ms += ms_since(t_prebegin);
            ++d_prebegin_n;
        } else {
            CK(cudaStreamSynchronize(st));
        }
        // the device-to-host copies the wait above covered are now readable on the host
        if (staged) {
            std::memcpy(h_helper_ids, staged_ids, (size_t) n_ent * 4);
            std::memcpy(h_helper_x, staged_x, (size_t) T * N * 4);
        }

        std::string err;
        // BEGIN EVERY HELPER OF THE WINDOW BEFORE FINISHING ANY OF THEM.  Each RemoteExperts owns its own CUDA
        // stream (remote_experts.hpp: stream_), so the helpers' kernels on one layer run at the same time; with
        // STRATA_Q36_HELPER_PARALLEL=1 this is what turns the striped "one helper per layer, the others idle"
        // into the row split Flash-Next already runs (expert_source.cpp calls begin() per helper in a loop).
        // On the default path nwin == 1 and this loop is one iteration - the same single begin() as before.
        for (int w = 0; w < nwin; ++w) {
            const int hi = win[w];
            if (helpers[(size_t) hi].begin((int64_t) li, h_helper_x, h_helper_ids, T, Kp, nullptr, nullptr, err))
                continue;
            std::fprintf(stderr, "strata-q36: helper CUDA%d, layer %zu: %s\n", hi + 1, li, err.c_str());
            helper_failed = true;
            // an all-zero mask on the device, so the local paths redo the whole window (every row is then written
            // by the local kernels and the helper's partial rows are overwritten).  A helper that already began is
            // left in flight: nothing uploads its rows (the mask is all zero), and its own next begin_slot drains
            // the window before reusing it.
            helper_mask_li = (int) li;
            CK(cudaMemcpyAsync(d_helper_mask, helper_mask.data(), (size_t) n_ent, cudaMemcpyHostToDevice, st));
            return false;
        }
        if (helper_async()) {
            // STRATA_Q36_HELPER_ASYNC: the per-entry mask goes up now (the local grouped kernel and the
            // CPU pool read it and must keep the same split they always had), but the eparts upload of
            // the rows the helpers compute is deferred to moe_helpers_wait(), after the layer's own
            // work has been issued.  Each helper already computed the entries it holds, so only its own
            // rows are uploaded (merged over the mask), never overwriting a row a local path owns.
            //
            // The mask is the UNION over the window's helpers.  The ownerships are disjoint - an expert blob is
            // claimed by exactly one helper at open time - so the union is a partition of this window between the
            // helpers and the local paths, and one whole-window upload + one k_merge_helper under it covers every
            // helper's rows whatever the window holds.  With nwin == 1 the union is that helper's own mask, so the
            // statements below are the ones this path always issued.
            for (int64_t i = 0; i < n_ent; ++i) {
                bool own = false;
                for (int w = 0; w < nwin && !own; ++w) own = helpers[(size_t) win[w]].owns(i);
                if (!own) continue;
                helper_mask[(size_t) i] = 1;
                ++hy_helper;
            }
            CK(cudaMemcpyAsync(d_helper_mask, helper_mask.data(), (size_t) n_ent, cudaMemcpyHostToDevice, st));
            helper_mask_li = (int) li;
            helper_wait_n = nwin;
            for (int w = 0; w < nwin; ++w) helper_wait_hi[w] = win[w];
            helper_wait_ent = n_ent;
            helper_wait_N = N;
            return true;
        }
        return helper_finish(win, nwin, li, N, n_ent);
    }
    /// The second half of a helper window, on the default (synchronous) path: block on each helper's GPU in turn,
    /// upload the window, mark the per-entry mask.  Byte-identical to the statement order the path always
    /// had - only STRATA_Q36_HELPER_ASYNC reorders it, and only STRATA_Q36_HELPER_PARALLEL makes the loop run more
    /// than once (the helpers are already computing concurrently by then: begin() was issued for all of them).
    bool helper_finish(const int* win, int nwin, size_t li, int N, int64_t n_ent) {
        std::string err;
        for (int w = 0; w < nwin; ++w) {
            const int hi = win[w];
            if (helpers[(size_t) hi].finish(helper_out, err)) continue;
            std::fprintf(stderr, "strata-q36: helper CUDA%d, layer %zu: %s\n", hi + 1, li, err.c_str());
            helper_failed = true;
            // the mask is all zero here (moe_helpers zeroed it and only fills it once every finish succeeded), so
            // uploading it makes the local paths redo the whole window and rewrite every row
            helper_mask_li = (int) li;
            CK(cudaMemcpyAsync(d_helper_mask, helper_mask.data(), (size_t) n_ent, cudaMemcpyHostToDevice, st));
            return false;
        }
        // One whole-window upload: finish() scattered the rows each helper computed to their own entry index and
        // left the rest of helper_out stale, but every entry no helper computed is rewritten by the local paths
        // below (exactly the rows whose mask bit is 0), so the stale rows are never read.
        CK(cudaMemcpyAsync(eparts, helper_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        helper_mask_li = (int) li;
        for (int64_t i = 0; i < n_ent; ++i) {
            bool own = false;
            for (int w = 0; w < nwin && !own; ++w) own = helpers[(size_t) win[w]].owns(i);
            if (!own) continue;
            helper_mask[(size_t) i] = 1;
            ++hy_helper;
        }
        CK(cudaMemcpyAsync(d_helper_mask, helper_mask.data(), (size_t) n_ent, cudaMemcpyHostToDevice, st));
        return true;
    }

    /// STRATA_Q36_HELPER_ASYNC: deferred helper completion for the layer currently being computed, issued
    /// after that layer's own work (local grouped kernel, shared expert, CPU pool) and before moe_end
    /// combines the entries' rows: finish() + host scatter staging + k_merge_helper into eparts.
    /// With the flag off, or with no window pending, this is a no-op.
    void moe_helpers_wait(size_t li) {
        if (!helper_async() || helper_wait_n == 0) return;
        const int nwin = helper_wait_n;
        int win[3];
        for (int w = 0; w < 3; ++w) { win[w] = helper_wait_hi[w]; helper_wait_hi[w] = -1; }
        const int64_t n_ent = helper_wait_ent;
        const int wN = helper_wait_N;
        helper_wait_n = 0;
        helper_wait_ent = 0;
        helper_wait_N = 0;
        std::string err;
        // One blocking wait per helper of the window, then ONE upload and ONE merge: every helper scattered its
        // rows into helper_out at that entry's own row index, and the device mask is the union of all of them, so
        // a single whole-window merge covers the whole window whatever the window holds (nwin == 1 on the default
        // path, which is the statement sequence this path always had).  The waiting is what the row split buys: by
        // the time the first helper's finish() returns, the others have been computing for that whole interval.
        for (int w = 0; w < nwin; ++w) {
            const int hi = win[w];
            if (helpers[(size_t) hi].finish(helper_out, err)) continue;
            // A deferred failure cannot fall back the way a synchronous one does: this layer's local
            // launches were already queued against the helpers' mask, so the entries the helpers took
            // have nothing computing them, and an all-zero mask could only be honoured by re-running
            // this layer's local work.  Not worth building for a CUDA error that dooms the run - so
            // the run ends here, with the reason.
            std::fprintf(stderr, "strata-q36: helper CUDA%d, layer %zu: %s\n", hi + 1, li, err.c_str());
            std::fprintf(stderr, "strata-q36: STRATA_Q36_HELPER_ASYNC defers the helper's wait past this "
                                 "layer's local launches, so a deferred helper failure ends the run\n");
            std::exit(1);
        }
        // Stage the helpers' sparse rows, then merge them into eparts where the mask marks them.
        CK(cudaMemcpyAsync(helper_stage, helper_out, (size_t) n_ent * wN * 4, cudaMemcpyHostToDevice, st));
        k_merge_helper<<<(unsigned) n_ent, kThreads, 0, st>>>(eparts, helper_stage, d_helper_mask, (int) wN);
    }
    /// Whether a helper computed part of state index li's current window, so the local grouping must leave those
    /// entries out and the CPU pool must keep off them.  False for a layer no helper was offered (moe_helpers was
    /// not called), which is what keeps the mask from the previous layer out of this one.
    bool helper_masked(size_t li) const {
        return helpers_enabled() && helper_layer(li) && !helper_failed && helper_mask_li >= 0 &&
               (size_t) helper_mask_li == li;
    }
    bool helper_failed = false;   // a helper errored: the run is over rather than half offloaded
    bool mmq_helper_warned = false;   // the helper-idle warning fired once (prompt MMQ chunks stay local)

    // ---- STRATA_Q36_PREFILL_PEERS (opt-in): a prompt chunk's routed experts on the helper GPUs.
    //
    // THE PROBLEM it removes: a prompt chunk (T > kGemmT) takes the MMQ branch of layer(), which streams every
    // not-yet-resident expert's blob from host memory over PCIe on the `cst` stream (stream_kick), and the helper
    // GPUs sit idle while it does.  That upload, not the arithmetic, is most of a prompt chunk's expert time.
    //
    // WHAT IT DOES: the entries whose expert a helper holds in its VRAM cache are computed on that helper, at full
    // precision of its own cache, writing their rows into `eparts` - over the peer link when there is a direct one,
    // through pinned host memory when there is not (see peer_host_route()).  The primary's MMQ then computes only
    // the entries nobody else holds, and stream_kick no longer uploads the blobs of the experts
    // the helpers own.  The arithmetic of each entry is one of the two the engine already runs (an int8 MMQ on the
    // primary, `PeerPrefill36`'s int8 MMQ on a helper), never a third thing.
    //
    // OPT-IN, and the default path is untouched: `peer_prefill_on` is false unless the variable is set AND the MMQ
    // path exists AND helpers are open AND every helper it will use got a route - a direct link the copy probe
    // trusts, or host staging if that is allowed.  With it false, no executor is constructed, no buffer is
    // allocated, no kernel takes the extra argument's non-null value, and every host test below is a
    // constant-false branch.
    bool prefill_peers() const {
        static const bool on = std::getenv("STRATA_Q36_PREFILL_PEERS") != nullptr;
        return on;
    }
    /// The number stored in `peer_owner` for an expert no helper holds.
    static constexpr uint8_t kPeerNone = 0xFF;
    std::array<std::unique_ptr<strata::qwen35::PeerPrefill36>, 3> peers;   // one per helper, index = helpers[]'s
    std::array<bool, 3> peer_on = {false, false, false};   // which of peers[] opened AND got a route (direct or host)
    // False: every peer statement in this file is dead.  Set once, in peer_open(), and the ONLY reader that is not
    // behind an `if` is the ternary that picks moe_end's row map - which resolves to `rinv` when this is false.
    bool peer_prefill_on = false;
    /// [n_state * n_expert] host master, [n_state * n_expert] device mirror: 0xFF = the primary computes this
    /// (layer, expert), else the index of the helper whose cache holds it.  ONE table, read by the route kernels'
    /// skip, by stream_kick's run splitting and by the per-layer plan, so the three cannot disagree about who owns
    /// an expert.  Built once at startup: a helper's cache is filled at open time and never changes during a run, so
    /// this is static per (layer, expert) for the whole process.
    std::vector<uint8_t> peer_owner;
    uint8_t* d_peer_owner = nullptr;
    std::vector<uint8_t> peer_any;   // [n_state]: some expert of this layer is a helper's (so the layer is served)
    int32_t* peer_lid = nullptr;     // [n_ent] the local MMQ's id table: sorted row -> eparts row (peer path only)
    int32_t* peer_prow2 = nullptr;   // [n_ent] device: entry -> eparts row (moe_end's `prow`)
    std::vector<int32_t> peer_prow2_h;   // [n_ent] host side of the same plan
    std::vector<uint8_t> peer_row_seen;  // [n_ent] the duplicate-row check's scratch (kept to avoid a per-layer alloc)
    // The two numbers peer_plan() leaves behind for the local MMQ of the layer it just planned: the rows the primary
    // computes, and the sorted-row -> eparts-row table.  Read through a `peer_here ?` ternary at the use sites, so a
    // layer without a plan - and every layer of a run with the flag off - takes the value it always took.
    int64_t peer_n = 0;
    const int32_t* peer_lid_map = nullptr;
    int32_t* h_peer_ids = nullptr;   // pinned [n_ent]: the layer's route ids, folded into the route D2H
    int peer_wait_n = 0;             // the helpers' launches this layer that moe_end must wait for
    int peer_wait_hi[3] = {-1, -1, -1};
    int64_t peer_wait_li = -1;
    /// The widest window the live executors can take: the SMALLEST cap_entries() among the helpers that opened,
    /// i.e. the narrowest helper's row budget.  0 when no executor is live (and in every run with the flag off),
    /// which is what makes peer_window_fits() false there and the over-cap branch unreachable.
    int64_t peer_cap_rows = 0;
    /// How many prompt-chunk layers fell back to the primary because their window was wider than peer_cap_rows.
    /// Only peer_open() and the MMQ branch's one counter bump ever touch it, and only the summary line reads it.
    int64_t peer_over_cap = 0;
    /// The entries (tokens x top_k) of the window being forwarded right now, recorded once at the top of forward().
    /// peer_layer() reads it so the three places that must agree about who owns an expert - the route kernels' skip
    /// (peer_mask), stream_kick's upload skip (peer_mask_host) and the plan (peer_plan) - all answer the same thing
    /// even when the window is too wide for the helpers.  stream_kick() runs at the START of the forward, two layers
    /// ahead of the layer whose MMQ branch consults it, so the number cannot be taken from the layer: it is the
    /// forward's window, and one window per forward is the only thing the chunk loop produces.
    int64_t peer_win_ent = 0;
    const int32_t* diag_toks = nullptr;   // DIAG: points at the caller's token vector while a prompt runs
    /// Whether state index li's prompt chunk COULD be served at all: the flag is live, the layer is a main one (the
    /// MTP block stays local, exactly as helper_layer() keeps it), and some expert of it is a helper's.  Says nothing
    /// about the window's width - peer_layer() adds that, and it is this half the over-cap counter is about.
    bool peer_owned_layer(size_t li) const {
        if (!peer_prefill_on || d_peer_owner == nullptr) return false;
        if (li >= (size_t) m->g.n_layers) return false;   // no helper slot for the MTP block
        return li < peer_any.size() && peer_any[li] != 0;
    }
    /// Whether state index li's prompt chunk is actually split with the helpers: peer_owned_layer() AND
    /// a window of `peer_win_ent` entries that fits the helpers' row budget AND took the MMQ branch
    /// (peer_window_fits()).  ONE predicate, read by the route kernels' skip, by stream_kick's
    /// upload skip and by the plan, so those three cannot disagree - a window past the budget, or a
    /// batch too narrow for the MMQ branch (a decode step), runs the layer exactly as it always ran
    /// (no skip, no upload hole, no plan) and the former is counted in peer_over_cap.
    bool peer_layer(size_t li) const { return peer_owned_layer(li) && peer_window_fits(peer_win_ent); }
    /// Whether a window of `n_ent` entries fits the helpers' row budget (the narrowest live cap_entries()) and
    /// is wide enough to have taken the MMQ branch - which is the ONLY branch that plans and waits - one
    /// comparison against a number fixed at startup, and false with no live executor - which is what keeps the
    /// over-cap fallback and every reader of peer_layer() inert in a run whose helpers did not open.
    bool peer_window_fits(int64_t n_ent) const {
        // lower bound = the MMQ branch's entry condition (T > hybrid_t): a narrower window (decode's T=1)
        // would make peer_layer() true on the hy_rows branch, which has no peer plan nor wait.
        return peer_cap_rows > 0 && n_ent <= peer_cap_rows && n_ent > (int64_t) hybrid_t * m->top_k;
    }
    /// The device ownership byte of li's experts (ne bytes, indexed by expert id), or null when the layer is not
    /// served - which is what the two route kernels' extra argument receives on every layer and every default run.
    const uint8_t* peer_mask(size_t li) const {
        return peer_layer(li) ? d_peer_owner + li * (size_t) m->g.n_expert : nullptr;
    }
    const uint8_t* peer_mask_host(int li) const {
        if (li < 0) return nullptr;
        // peer_layer() itself, not a copy of its conditions: this is the upload skip, the route skip and the plan all
        // asking the same question, so an over-cap layer must be unserved here as well (else stream_kick leaves a hole
        // in `stage[b]` that nothing is computing).
        return peer_layer((size_t) li) ? peer_owner.data() + (size_t) li * (size_t) m->g.n_expert : nullptr;
    }

    /// The per-layer plan, the helpers' launches and the local MMQ's id table, for one served layer.  Called after
    /// the layer's route D2H has landed (h_peer_ids holds the entries' expert ids, the sync that brought rmax is
    /// past) and BEFORE the primary's MMQ is issued, so the helpers' work on their own streams overlaps it.  False
    /// (with `err`) = the plan does not add up, or a helper launch was refused: the caller prints and exits - the
    /// primary's MMQ has not been issued yet at plan time, but by the time the combine runs half the layer would
    /// be dead, so there is nothing to fall back to anyway (the failure policy the deferred helper path documents).
    bool peer_plan(size_t li, int T, const float* hdev, const int32_t* idsdev, float* epartsd, std::string& err) {
        const auto& g = m->g;
        const int Kp = (int) m->top_k, ne = (int) g.n_expert;
        const int64_t n_ent = (int64_t) T * Kp;
        // The plan's host buffers are sized for the largest window this engine can hand a chunk (tmax tokens x top_k
        // entries).  A bigger window is not something the chunk loop produces, but reading past the buffer would be
        // silent, so it is refused rather than assumed away.  peer_window_fits() keeps this from being reached: the
        // MMQ branch only plans a layer whose window fits the helpers' row budget, which is this buffer's size.
        if (n_ent > (int64_t) peer_prow2_h.size()) {
            err = "the window is " + std::to_string((long long) n_ent) + " entries, past the " +
                  std::to_string((long long) peer_prow2_h.size()) + " the peer plan allocated for";
            return false;
        }
        const uint8_t* own = peer_owner.data() + li * (size_t) ne;
        // THE OWNERSHIP CHECK.  `peer_owner` is the master table the route kernels, stream_kick and this plan all
        // read, and it is built ONCE in peer_open(); PeerPrefill36::prepare() instead re-reads the helper's
        // RemoteExperts::slot_ptr() for THIS layer on every call and rebuilds its row plan from that live residency.
        // Nothing in q36 moves a helper slot after startup today (rebalance() only moves the PRIMARY's staging slots,
        // and q36 never attaches a RemoteExpertOpt), but the two answers are independently derived: if they ever
        // disagreed, `base_j` and the executor's ranks would describe different rows and the run would combine rows
        // nobody wrote - silent wrong logits.  So the table is verified against the live caches BEFORE the per-entry
        // row plan is trusted, and a disagreement ends the run with a reason naming both sides (the policy
        // peer_plan_or_exit() documents).  It is not behind an environment variable: it is the check that turns a
        // silent corruption into a named failure.  Cost: 3 x n_expert pure host slot_of() lookups per served layer -
        // no CUDA call, no allocation, no synchronisation, so it stays off the layer's critical path.
        for (int32_t e = 0; e < ne; ++e) {
            for (int r = 0; r < 3; ++r) {
                if (!peer_on[(size_t) r] || peers[(size_t) r] == nullptr) continue;
                const bool cache_says = peers[(size_t) r]->owns((int64_t) li, e);
                const bool table_says = own[(size_t) e] == (uint8_t) r;
                if (cache_says == table_says) continue;
                err = "layer " + std::to_string((long long) li) + ", expert " + std::to_string((long long) e) +
                      ": the ownership table says " +
                      (own[(size_t) e] == kPeerNone ? std::string("the primary")
                                                    : "helper CUDA" + std::to_string((int) own[(size_t) e] + 1)) +
                      ", while helper CUDA" + std::to_string(r + 1) + "'s live cache says " +
                      (cache_says ? "it holds that expert" : "it holds no such expert");
                return false;
            }
        }
        // Pass 1: how many entries each helper owns.  An entry belongs to a helper iff that helper's cache holds
        // (layer, ids[entry]) - the same test PeerPrefill36::owns() makes, read here from the master table so the
        // two cannot drift.
        int64_t rows[3] = {0, 0, 0}, n_local = 0;
        for (int64_t e = 0; e < n_ent; ++e) {
            const uint8_t o = own[(size_t) h_peer_ids[e]];
            if (o == kPeerNone) ++n_local;
            else ++rows[(size_t) o];
        }
        // The row contract (src/qwen35/q36_peer_prefill.cuh): the primary's entries first, in entry order, then each
        // helper's run in helper order, per-helper in ascending entry order.  `base_j = n_local + sum_{i<j} rows_i`.
        int64_t base[3] = {0, 0, 0}, at = n_local;
        for (int r = 0; r < 3; ++r) {
            base[(size_t) r] = at;
            at += rows[(size_t) r];
        }
        if (at != n_ent) {   // cannot happen unless a helper claims an expert the table gave to another
            err = "the peer plan covers " + std::to_string((long long) at) + " of " + std::to_string((long long) n_ent) +
                  " entries of layer " + std::to_string((long long) li);
            return false;
        }
        // Pass 2: entry -> eparts row.  A local entry's row is its rank among local entries in entry order - the
        // rows [0, n_local) the local MMQ writes; a helper j entry's row is base_j plus its rank among j's owned
        // entries - the executor's run.  Every row of the whole window is named exactly once, which is what the
        // duplicate-row check below asserts.
        int64_t lrank = 0, rank[3] = {0, 0, 0};
        std::fill(peer_row_seen.begin(), peer_row_seen.begin() + (size_t) n_ent, (uint8_t) 0);
        for (int64_t e = 0; e < n_ent; ++e) {
            const uint8_t o = own[(size_t) h_peer_ids[e]];
            const int64_t row = o == kPeerNone ? lrank++ : base[(size_t) o] + rank[(size_t) o]++;
            peer_prow2_h[(size_t) e] = (int32_t) row;
            // The second half of the invariant, and the one a real bug would hit first: two entries landing in one
            // eparts row would leave one of them combined twice and the other never.  Silent logit corruption is
            // the failure mode, so it is checked here and ends the run rather than producing a wrong answer.
            if (row < 0 || row >= n_ent || peer_row_seen[(size_t) row] != 0) {
                err = "the peer plan gave layer " + std::to_string((long long) li) + " entry " +
                      std::to_string((long long) e) + " the eparts row " + std::to_string((long long) row) +
                      ", which another entry already holds";
                return false;
            }
            peer_row_seen[(size_t) row] = 1;
        }
        CK(cudaMemcpyAsync(peer_prow2, peer_prow2_h.data(), (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        // The local MMQ's id table: sorted row r's product goes to eparts row prow2[ent_dst[r]].  Exactly what the
        // identity table does with no helper (there prow2[e] == rinv[e]), so nothing about the local launch changes
        // beyond which rows of eparts it writes - and the rows it does write are the same ones.  Guarded against an
        // empty grid: a window whose every entry is a helper's leaves the primary nothing to launch (a 0-block launch
        // is a CUDA error, and the MMQ's own run() already treats 0 rows as nothing to do).
        if (n_local > 0) k_peer_lid<<<grid1(n_local), kThreads, 0, st>>>(ent_dst, peer_prow2, (int) n_local, peer_lid);
        peer_n = n_local;
        peer_lid_map = peer_lid;
        // Every helper that owns an entry of this window: launch its share now, so its arithmetic runs on its own
        // stream while the primary's MMQ runs below.  A helper with no entry here is not touched at all (its stream
        // stays idle, which is cheaper than an empty launch).  The row bases passed are the same bases the plan
        // numbered the entries with, so a helper's run lands exactly where prow2 says its entries' rows are.
        peer_wait_n = 0;
        peer_wait_li = (int64_t) li;
        for (int r = 0; r < 3; ++r) {
            if (rows[(size_t) r] == 0 || !peer_on[(size_t) r] || peers[(size_t) r] == nullptr) continue;
            if (!peers[(size_t) r]->prepare((int64_t) li, hdev, idsdev, (int64_t) T, err)) return false;
            if (!peers[(size_t) r]->launch((int64_t) li, epartsd, base[(size_t) r], err)) return false;
            // Second witness of the ownership invariant, from the executor's side and at one integer's cost: rows()
            // is what the helper's OWN live row plan counted this call (built from its cache's slot_ptr()); rows[r]
            // is what the master table above predicted for the same window.  The expert walk catches a table that
            // names the wrong helper; this catches the same disagreement as it shows up in the run the helper will
            // actually write, which is the number that decides where its rows land.
            if (peers[(size_t) r]->rows() != rows[(size_t) r]) {
                err = "helper CUDA" + std::to_string(r + 1) + " planned " +
                      std::to_string((long long) peers[(size_t) r]->rows()) + " rows of layer " +
                      std::to_string((long long) li) + ", the ownership table predicted " +
                      std::to_string((long long) rows[(size_t) r]);
                return false;
            }
            peer_wait_hi[peer_wait_n] = r;
            ++peer_wait_n;
        }
        return true;
    }
    /// peer_plan()'s failure policy, in one place: print the reason and end the run.  There is nothing to fall back
    /// to once the route kernels have already skipped the peer-owned entries against this plan - the primary's MMQ
    /// would leave those eparts rows holding whatever the previous layer left there, which is a wrong answer, not a
    /// slow one.  (A STARTUP failure is different and falls back cleanly: peer_open() simply does not take the
    /// helper, and every layer then runs the plan-free path.)
    void peer_plan_or_exit(size_t li, int T, const float* hdev, const int32_t* idsdev, float* epartsd) {
        // DIAG: the layer's input row and the eparts row the combine will read.  A NaN in hdev means the
        // corruption is upstream of the peer split; a NaN in epartsd means this layer's combine is reading a row
        // nobody wrote.  Remove after the fix.
        if (peer_prefill_on) {
            CK(cudaStreamSynchronize(st));   // DIAG: st is non-blocking, so a plain read would race the engine
            float h4[4] = {}, e4[4] = {};
            CK(cudaMemcpy(h4, hdev, 4 * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(e4, epartsd, 4 * 4, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "DIAG plan layer %zu h[0..3]=%g %g %g %g eparts[0..3]=%g %g %g %g\n", li, h4[0],
                         h4[1], h4[2], h4[3], e4[0], e4[1], e4[2], e4[3]);
        }
        std::string perr;
        if (peer_plan(li, T, hdev, idsdev, epartsd, perr)) return;
        std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS, layer %zu: %s\n", li, perr.c_str());
        std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: the per-entry row plan does not add up; continuing "
                             "would combine rows no one wrote\n");
        std::exit(1);
    }
    /// STRATA_Q36_PREFILL_PEERS startup, called from open_helpers() once the helpers and the layout table are open.
    /// Builds one executor per helper that actually opened, the two-way P2P copy probe for each of them, and the
    /// per-(layer, expert) ownership table both the kernels and stream_kick read.  Every allocation in here is
    /// behind the feature's own gates, so a default run (no variable, or no helper flag, or no MMQ) allocates
    /// nothing and prints nothing it did not before.
    void peer_open(const Model& m, const strata::kernels::NativeExpertLayout* layouts) {
        const auto& g = m.g;
        const int64_t ne = g.n_expert;
        int live = 0;
        // STRATA_PRIMARY_DEVICE may name another card as the one the model runs on: the probe and every executor copy
        // between THAT card and a helper, so it is asked rather than assumed to be 0.
        const int prim = strata::core::primary_device();
        for (int r = 0; r < 3; ++r) {
            if (!helper_on[(size_t) r]) continue;
            const int device = r + 1;   // helpers[r] was opened on CUDA r+1 (0..2 map to 1..3)
            std::string perr;
            // Two questions, two routes: the copy probe says whether a DIRECT link is worth preferring, and
            // peer_host_route() says whether host staging may stand in for one (opt-in: it does not survive a decode
            // step yet - see that function).  Only a helper that fails both is out of the prompt path, which is
            // today's behaviour whenever the variable is not exactly "1".
            const bool direct = peer_copies_data(prim, device, perr);
            if (!direct && !peer_host_route()) {
                std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: helper CUDA%d is out of the prompt "
                                     "path: %s; its experts stay on today's local route\n", device, perr.c_str());
                continue;
            }
            if (!direct) {
                // one line, and it names the reason the direct link was refused: the executor's open() decides the
                // staging itself, but a reader of stderr must not have to guess why a helper went the slow way.
                std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: helper CUDA%d takes the host route: %s; "
                                     "its expert rows are staged through host memory (STRATA_Q36_PREFILL_PEERS_HOST=1, "
                                     "experimental: a decode step after the prompt is not safe yet)\n",
                             device, perr.c_str());
                perr.clear();   // the probe's words are spent; open() must report its own failure, not this one
            }
            // the largest window the prompt hands the executors: chunks cap at tmax tokens, entries at tmax * top_k
            const int64_t cap = (int64_t) tmax * (int64_t) m.top_k;
            auto px = std::make_unique<strata::qwen35::PeerPrefill36>();
            if (!px->open(helpers[(size_t) r], g.n_layers, g.n_expert, m.top_k, g.n_embd, g.n_ff, cap, layouts, perr)) {
                std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: helper CUDA%d's executor did not open "
                                     "(%s); that helper stays on today's local route\n", device, perr.c_str());
                continue;
            }
            peers[(size_t) r] = std::move(px);
            peer_on[(size_t) r] = true;
            ++live;
        }
        if (live == 0) {
            std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: no helper passed the P2P copy test and opened, "
                                 "so prompt chunks take today's local path\n");
            return;
        }
        peer_prefill_on = true;
        // The row budget the prompt path may hand out: the narrowest cap_entries() of the executors that opened.  A
        // window wider than this cannot be split (one helper's buffers would not take it), so the layer runs on the
        // primary and peer_over_cap records it - the same answer Flash-Next gives per expert, given once per layer.
        for (int r = 0; r < 3; ++r) {
            if (!peer_on[(size_t) r] || peers[(size_t) r] == nullptr) continue;
            const int64_t cap = peers[(size_t) r]->cap_entries();
            peer_cap_rows = peer_cap_rows == 0 ? cap : std::min(peer_cap_rows, cap);
        }
        // The master table: which helper holds each main layer's expert.  0xFF = the primary computes it; a helper's
        // index (0..2) = its entries' rows live in that helper's eparts run.  The MTP block's row, when present, is
        // all 0xFF - the block is not served (peer_layer() keeps it local, as helper_layer() does).
        const int64_t n_state = g.n_layers + (m.has_mtp ? 1 : 0);
        peer_owner.assign((size_t) n_state * (size_t) ne, kPeerNone);
        for (int r = 0; r < 3; ++r) {
            if (!peer_on[(size_t) r]) continue;
            for (int64_t l = 0; l < g.n_layers; ++l)
                for (int64_t e = 0; e < ne; ++e)
                    if (peers[(size_t) r]->owns(l, (int32_t) e)) peer_owner[(size_t) l * ne + e] = (uint8_t) r;
        }
        peer_any.assign((size_t) g.n_layers, 0);
        int64_t n_owned = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            for (int64_t e = 0; e < ne; ++e)
                if (peer_owner[(size_t) l * ne + e] != kPeerNone) {
                    peer_any[(size_t) l] = 1;
                    ++n_owned;
                }
        }
        CK(cudaMalloc((void**) &d_peer_owner, (size_t) n_state * (size_t) ne));
        CK(cudaMemcpy(d_peer_owner, peer_owner.data(), (size_t) n_state * (size_t) ne, cudaMemcpyHostToDevice));
        peer_lid = dalloc_t<int32_t>((size_t) tmax * m.top_k);
        peer_prow2 = dalloc_t<int32_t>((size_t) tmax * m.top_k);
        peer_prow2_h.resize((size_t) tmax * m.top_k);
        peer_row_seen.resize((size_t) tmax * m.top_k);
        CK(cudaHostAlloc((void**) &h_peer_ids, (size_t) tmax * m.top_k * 4, cudaHostAllocDefault));
        int n_served = 0;
        for (auto b : peer_any) n_served += b != 0;
        // the evidence line: say how many helpers the prompt path will use, and that a fallback happened for the
        // rest - a probe or an open that failed is a fallback to today's route, not a silent skip.
        int fell = 0;
        for (int r = 0; r < 3; ++r) if (helper_on[(size_t) r] && !peer_on[(size_t) r]) ++fell;
        // `; host route: on` only when the opt-in variable is exactly "1": a run that asked for host staging is
        // told which of the two routes it is on, and every other run's stderr keeps the bytes it always had.
        std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: %d helper(s) serve prompt chunks (%lld experts "
                     "held by helpers, %d layers served, windows up to %d tokens%s); the primary's MMQ computes the "
                     "rest%s\n", live, (long long) n_owned, n_served, tmax,
                     peer_host_route() ? "; host route: on" : "",
                     fell > 0 ? "; fallback: the un-opened/un-probed helpers stay on today's local route" : "");
    }
    /// Make the primary stream wait for this layer's helper launches, so the combine below reads complete rows.
    /// Called just before moe_end, where the existing helper deferral's wait also lands.  A failure ends the run for
    /// the reason the deferred-helper path documents: the primary's MMQ was already issued against this plan, so the
    /// entries the helper took have nothing computing them and there is nothing safe to fall back to mid-layer.
    void peer_prefill_wait() {
        if (peer_wait_n == 0) return;
        const int n = peer_wait_n;
        int hi[3] = {-1, -1, -1};
        for (int i = 0; i < n; ++i) {
            hi[i] = peer_wait_hi[i];
            peer_wait_hi[i] = -1;
        }
        const int64_t li = peer_wait_li;
        peer_wait_n = 0;
        std::string err;
        // DIAG: the helper's rows have landed in eparts by now.  A NaN here is the helper's own arithmetic.
        // Remove after the fix.
        if (peer_prefill_on) {
            CK(cudaStreamSynchronize(st));   // DIAG: st is non-blocking, so a plain read would race the engine
            float e4[4] = {};
            CK(cudaMemcpy(e4, eparts, 4 * 4, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "DIAG wait layer %lld eparts[0..3]=%g %g %g %g\n", (long long) li, e4[0], e4[1],
                         e4[2], e4[3]);
        }
        for (int i = 0; i < n; ++i) {
            const int r = hi[i];
            if (peers[(size_t) r] != nullptr && peers[(size_t) r]->wait(st, err)) continue;
            std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: helper CUDA%d, layer %lld: %s\n", r + 1,
                         (long long) li, err.c_str());
            std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS hands this layer's entries to the helpers before "
                                 "the primary's MMQ is issued, so a helper failure here ends the run (the primary's "
                                 "rows were launched against a plan that has them missing)\n");
            std::exit(1);
        }
    }

    /// The GPU half of a hybrid MoE: the resident experts' entries grouped and run.  `h_ids`/`h_h` are the ROUTE
    /// copies of `ids`/`h` (the two D2H copies, issued once per layer by the caller, land here): the CPU pool
    /// reads them from ev_route below.  When they are null (no caller-side copy issued) moe_hybrid_gpu issues the
    /// pair itself, in the same place and before the grouping, exactly as this always did.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T, int32_t* h_ids_src = nullptr, const float* h_src = nullptr) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        if (h_ids_src && h_src) {
            CK(cudaEventRecord(ev_route, st));   // the caller's copies: the CPU pool waits for those
        } else {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
        }
        // route_group grouped this layer's picks before moe_helpers ran (it makes the `ids` the helper reads), so
        // with a helper's share the grouping is redone without the rows the helper already wrote into `eparts`.
        const uint8_t* skip = helper_masked(li) ? d_helper_mask : nullptr;
        if (skip) grouped_by_route = false;
        group_entries(L, hy[li].slot_of_d, skip, T);
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
        // moe_hybrid_gpu recorded ev_route for this layer (helper or not): the pool reads `h_ids` / `h_h` from it,
        // so it must wait.  The helper path no longer skips the pool, it only adds entries to skip.
        // The hy_cpu_ms timer below starts AFTER this spin, so time the spin itself (STRATA_Q36_HELPER_TIMING=1).
        if (helper_timing()) {
            const auto t_spin = Clock::now();   // timed only under the gate
            while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
            }
            d_route_spin_ms += ms_since(t_spin);
            ++d_route_spin_n;
        } else {
            while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
            }
        }
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
            // Three buckets, as in Qwen3.8 Flash's expert pool: an entry a helper computed is already in `eparts`
            // and its expert has no primary slot, so the mask keeps the pool off it.
            h_miss[e] = H.slot_of[(size_t) x] < 0 && !(helper_masked(li) && helper_mask[(size_t) e]);
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
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
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
                    CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                                       cudaMemcpyHostToDevice, st));
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
    }

    void shrink_att(size_t floats) {
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
            if (gset.g1) { CK(cudaGraphExecDestroy(gset.g1)); gset.g1 = nullptr; }
            for (auto& g : gset.gm1)   // one draft-step graph per chain depth d (STRATA_Q36_MTP_MAPPED)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
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
            // DIAG: the decode step's first kernel (iq_embed_rows) reads tok[0]; print the value that will be copied
            // in, so a crash there can be told apart from a bad token id.  Remove after the peer-route crash is fixed.
            if (peer_prefill_on) {
                std::fprintf(stderr, "DIAG decode in=%d pos=%lld vocab=%lld\n", toks[0], (long long) pos,
                             (long long) m->g.n_vocab);
                // the first 8 of the caller's buffer: if these are garbage the prompt corrupted the host vector
                std::fprintf(stderr, "DIAG caller[0..7]=");
                for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %d", toks[i]);
                std::fprintf(stderr, "\n");
            }
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
            diag_toks = toks + i;   // DIAG: watch the caller's tokens while this chunk runs
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            diag_toks = nullptr;
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, mtp_seed(), false, false, pos0_m);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
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
        const size_t row = (size_t) ctx * g.head_dim * 4;
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, (size_t) a.pos * g.head_dim * 4, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], row, a.vc[l], row, (size_t) a.pos * g.head_dim * 4, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
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
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr, int* ran_D = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        // --suffix-draft: a lookup window replaces the MTP drafts for THIS round only, and only when the policy says
        // it beats them.  The room is the same clamp the MTP depth just took (a window must not cross a rebalance
        // boundary, or its rows would mix two slot tables) and the verify graph's ceiling (gv[] stops at kVerify).
        // Decided HERE, before the MTP chain is submitted: a chain that is then thrown away has already cost its
        // drafts.  With the pointer null every line below is skipped and D is what the caller asked for.
        strata::qwen35::SuffixChoice sfx;
        sfx.deff = D;
        if (suffix && (size_t) pos == suffix->size()) {
            // A lookup window is still a verify window: it must not cross a rebalance boundary either, or its rows
            // mix two slot tables.  The clamp above already cut D; the room a lookup may ask for is that same cut,
            // not kVerify - 1.
            int room = std::min(std::max(D, kVerify - 1), kVerify - 1);
            if (main_hybrid)
                room = std::min(room, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
            sfx = suffix->choose(x_tok, D, std::max(0, room));
            if (sfx.lookup) D = sfx.deff;
        } else if (suffix) {
            ++sfx_bad;
        }
        // The rebalance clamp above can cut this window short of mtp_d; the caller books `offered`, so it needs the
        // size that actually ran, not the size asked for (otherwise a window near a 32-position boundary is charged
        // for drafts it never made, and every acceptance ratio printed is slightly pessimistic).
        if (ran_D != nullptr) *ran_D = D;
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        // STRATA_Q36_MTP_MAPPED: live only with a draft step that has no host block in it, and never with the top-2
        // diagnostic (its runner-up is a host read of ONE logits buffer that pre-submission would let the steps
        // overwrite each other over).  `mapped_chain` is the whole switch; every other statement in this function is
        // guarded by it, so with the flag unset this is the loop, statement for statement, that it always was.
        const bool mapped_chain = mapped_chain_live();
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        // The chain's positions, staged ONCE for the whole window: element 0 is the T0 rows' base (what the step
        // below reads), element d is chained step d's (pos + d).  The kernels read them at execution, so a step
        // submitted now still sees its own position while later steps wait in the queue.
        if (mapped_chain && D > 0) {
            // element 0 is the T0 rows' base (what the T0 call's *pos0_m reads: pos - fix.size()); element d is
            // chained step d's (pos + d).  Staged once for the whole window; the kernels read their own word at
            // EXECUTION, so a step submitted now still sees its own position while the later ones wait in the queue.
            hpin[32] = (int32_t) (pos - (int64_t) fix.size());
            for (int d = 1; d < D; ++d) hpin[32 + d] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(pos0_m, hpin + 32, (size_t) D * 4, cudaMemcpyHostToDevice, st));
        } else {
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        }
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        // A lookup window carries no MTP drafts, so the block only has to advance its own K / V over the redo rows
        // (draft = false writes none).  An MTP window drafts exactly as before.
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0 && !sfx.lookup, false, pos0_m,
                    mapped_chain && D > 0 && !sfx.lookup ? 0 : -1);
        drafts.assign((size_t) D, 0);
        if (sfx.lookup) suffix->fill(drafts, D);   // after the clear: assign() would have wiped a write made before it
        for (int i = 0; i < 16; ++i) mtp_top2_prev[i] = -1;
        // the chain's published slots start empty, so a spin can tell "not yet" from "the token 0" (a draft CAN be 0)
        if (mapped_chain && D > 0) {
            for (int d = 0; d < D; ++d) ((volatile int32_t*) pick_h)[d] = -1;
            // step 0's publish rides with the step itself (below), so the first token of the chain is on the device
            // before any chained step is queued.
        }
        if (D > 0 && !mapped_chain && !sfx.lookup) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = hpin[2];
            if (mtp_top2() && D <= 16) mtp_top2_prev[0] = mtp_top2_tok;
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        if (mapped_chain && !sfx.lookup) {
            // Flash-Next's shape (src/core/mtp.cpp:1623-1658): SUBMIT every chained step before the host waits for
            // anything, then spin on the mapped picks.  A step's seed is the PREVIOUS step's output - a real data
            // dependency - so its D2D copy rides the stream just before that step's launch (stream order: copy_d ->
            // forward_d -> copy_{d+1} -> forward_{d+1}); and a step's token is written by the previous step's
            // k_chain_publish on the device.  Nothing below reads drafts[d - 1] from the host: the mapped slots are
            // REPORTS only.
            int lr = last_row;
            for (int d = 1; d < D; ++d) {
                CK(cudaMemcpyAsync(seeds, hmtp + (size_t) lr * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                lr = 0;
                chain_step(d);
            }
            (void) cudaStreamQuery(st);   // everything is submitted; the host waits below
            // The spin: the upstream one (mtp.cpp:1639-1645) - volatile reads of the mapped slot, with a stream
            // query every 1024 spins to break out if the chain ended early or faulted.  One synchronise at the end,
            // for the last draft, which no spin needs.
            for (int d = 0; d + 1 < D; ++d) {
                unsigned spins = 0;
                volatile int32_t* slot = (volatile int32_t*) pick_h;
                while (slot[d] < 0) {
                    cpu_relax();
                    if ((++spins & 1023u) == 0 && cudaStreamQuery(st) != cudaErrorNotReady) break;
                }
                drafts[(size_t) d] = slot[d];
            }
            CK(cudaStreamSynchronize(st));
            if (D > 0) drafts[(size_t) D - 1] = ((volatile int32_t*) pick_h)[D - 1];
        } else if (!sfx.lookup)
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            if (mtp_graph) graph_run(gs().gm1[0], [&] { forward_mtp(1, pos + d, seeds, true, use_graph, pos0_m); });
            else forward_mtp(1, pos + d, seeds, true, false, pos0_m);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = hpin[2];
            if (mtp_top2() && d < 16) mtp_top2_prev[d] = mtp_top2_tok;
        }
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
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        if (cancel != nullptr && cancel->load()) a = 0;
        // STRATA_Q36_MTP_TOP2=1: the per-position counts.  dp_* is what the plain accepted/offered pair cannot show
        // (that ratio is diluted by the offered count, which grows with N); t2_* is the diagnostic proper.
        if (mtp_top2()) {
            for (int d = 0; d < D && d < 16; ++d) {
                ++dp_tested[d];
                if (d < a) ++dp_ok[d];
                else {
                    ++t2_rej[d];
                    if (mtp_top2_prev[d] >= 0 && mtp_top2_prev[d] == hpin[2 + d]) ++t2_hit[d];
                }
            }
        } else {
            // Same per-depth counts without the runner-up: cheap (no extra device-to-host copy), so the default path
            // can report them too.  This is the direct measurement of p_d that the aggregate accepted/offered pair
            // cannot give, since that ratio's denominator grows with N.
            for (int d = 0; d < D && d < 16; ++d) {
                ++dp_tested[d];
                if (d < a) ++dp_ok[d];
            }
        }
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        if (sfx.lookup) {
            ++sfx_win;
            sfx_acc += a;
            suffix->observe(true, D + 1, a, sfx.match, 0.0);
        }
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
        // Every throw-away pass runs from cell 0: the passes must not stack their positions, or the last ones write
        // their K / V past the cache (a context as short as the chunk turns that into an out-of-range device write).
        // The chunk is also capped at the context (tmax can exceed it with a small --max-context).
        const int T = (int) std::min<int64_t>(tmax, ctx);
        std::vector<int32_t> toks((size_t) T, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        pos = 0;
        if (T > kGemmT + 1) {
            process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
            pos = 0;
        }
        process(&filler, 1, [](int64_t) {});
        pos = 0;
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_helper = 0;   // the warmup's helper work is not part of the batch counts
        hy_cpu_ms = 0;
        d_prebegin_ms = 0; d_prebegin_n = 0; d_route_spin_ms = 0; d_route_spin_n = 0;
        gp_reset();   // STRATA_Q36_HELPER_TIMING=1: the GPU-segment accumulators reset with the host ones, or bench
                      // pass 1 would leak into pass 2 and into the AGG aggregate.
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    /// --mtp-calibrate (opt-in): forget a throwaway pass's RECURRENT state, so the answer that follows starts from
    /// the prompt again, and nothing else.
    ///
    /// What it deliberately does NOT touch is the hybrid placement: the probe's steps counted routing and moved
    /// slots exactly as the answer's own steps would, and the answer re-routes the same traffic, so its slot table
    /// has to continue from where the probe left it.  Clearing the routing scores instead starves rebalance() - the
    /// next run cannot rank anything, so it places nothing, and a different expert in a slot is a different
    /// arithmetic and therefore a different token stream.  Measured on 192.168.1.98 with --mtp 1 --mtp-calibrate 8:
    /// clearing the scores gave 31 slot replacements where the plain reference makes 1640, and an answer that shared
    /// only its first 9 tokens with the reference.  Leaving them alone reproduces the reference token for token.
    ///
    /// The counters cleared are the ones the report prints for a request (the probe's are not part of one), exactly
    /// as warmup() does for its own throw-away passes.
    void discard_pass() {
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_helper = 0;
        hy_cpu_ms = 0;
        d_prebegin_ms = 0; d_prebegin_n = 0; d_route_spin_ms = 0; d_route_spin_n = 0;
        gp_reset();   // STRATA_Q36_HELPER_TIMING=1: same reason as in warmup()
        row_ids.clear();
        reset();
    }

    /// --mtp-calibrate: put the hybrid routing table back to where it was before a throwaway pass.  The scores a
    /// probe counted are not the answer's, and rebalance() ranks on them, so leaving them in place would move the
    /// answer's slots differently from a plain run's - and a different expert in a slot is a different arithmetic
    /// and a different token stream.
    void clear_route_counts() {
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        row_ids.clear();
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        // DIAG: when the peer route is on, print the first logits and the hidden row that produced them.  A prompt
        // whose logits are all -inf shows up as a pick of 0x7fffffff, which then crashes the next decode step's
        // embedding lookup.  Remove with the other DIAG prints.
        if (peer_prefill_on) {
            CK(cudaStreamSynchronize(st));   // DIAG: st is non-blocking, so a plain read would race the engine
            std::vector<float> lg(8);
            CK(cudaMemcpy(lg.data(), logits, 8 * 4, cudaMemcpyDeviceToHost));
            float hl[4] = {};
            CK(cudaMemcpy(hl, hlast, 4 * 4, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "DIAG greedy=%d logits[0..7]=", hpin[2]);
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", lg[(size_t) i]);
            std::fprintf(stderr, " hlast[0..3]=%g %g %g %g\n", hl[0], hl[1], hl[2], hl[3]);
        }
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
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
    const size_t slack = ((size_t) 512 << 20) + (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
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
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
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
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
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
        if (S > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemcpy(L.blobs, L.hblobs, L.stride * (size_t) S, cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) s] = s;
            H.exp_in_slot[(size_t) s] = s;
        }
        H.score.assign((size_t) ne, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(cudaMalloc(&E.ecpu, T * Kp * N * 4));
    CK(cudaMalloc(&E.d_miss, T * Kp * 4));
    // the helper tier's per-entry mask (--expert-cache-device1..3), read by the masked grouping kernels.  Also
    // needed when every expert fits in VRAM (no CPU pool, this function returns early), where a helper still takes
    // the entries it holds - open_helpers allocates it if this path did not.
    if (E.d_helper_mask == nullptr) CK(cudaMalloc(&E.d_helper_mask, T * Kp));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
    // STRATA_Q36_HELPER_TIMING=1: the per-layer GPU-timeline events.  Created only under that gate, so a flag-off
    // run allocates and records nothing it did not before.  NOTE: no cudaEventDisableTiming here - these are the
    // events whose timestamps cudaEventElapsedTime reads, and a no-timing event has none, so ElapsedTime on a
    // pair of them fails with cudaErrorInvalidResourceHandle.  That is the whole reason these eight are separate
    // from ev_route / ev_copy / ev_used, which only ever synchronize and so do want timing disabled.
    E.gpu_timing = false;
    {
        const char* ht = std::getenv("STRATA_Q36_HELPER_TIMING");
        if (ht && ht[0] == '1' && ht[1] == '\0') {
            for (int k = 0; k < Engine::kGsegN; ++k)
                CK(cudaEventCreate(&E.ev_seg_ev[k]));
            E.gpu_timing = true;
        }
    }
    if (staging) {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&E.stage[b], max_layer + kMmqTail));
            CK(cudaMemset(E.stage[b], 0, max_layer + kMmqTail));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    CK(cudaDeviceSynchronize());
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
            if (E.suffix) E.suffix->reset();
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
            if (E.suffix) E.suffix->reset();
        }
        // The history must end at the same token pos does.  A continued request already holds the reused prefix in
        // the drafter (it was appended as it was committed); only the freshly read tail is new.  Appending the whole
        // of ss.live here would double-count that prefix and every later pos == size() check would fail.
        if (E.suffix && !cancelled) {
            if ((size_t) E.pos < E.suffix->size()) E.suffix->reset();
            if ((size_t) E.pos > E.suffix->size())
                E.suffix->append(ss.live.data() + E.suffix->size(), (size_t) E.pos - E.suffix->size());
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
        // text); sampled ones, and requests beside others, one token at a time.  --suffix-draft uses the same verified
        // window, so it is allowed with --mtp 0: the policy then compares a lookup window against a plain step.
        const bool spec = (E.mtp_d > 0 || E.suffix) && !smp.needs_logits() && !adm;
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
            if (spec && n_active == 0 && E.pos + std::max(E.mtp_d, E.suffix ? 7 : 0) + 1 <= E.ctx) {
                int ran_D = 0;
                // --mtp-adaptive: the depth this round drafts with is the learned one (0 = the plain step, and the
                // branch below is skipped so the loop falls through to it); with the flag unset it is mtp_d.
                const int qd = E.d_adapt ? E.adapt.next : E.mtp_d;
                const auto ts_a = (E.d_adapt || E.suffix) ? Clock::now() : Clock::time_point{};
                if (qd > 0 || E.suffix) {
                const int a = E.spec_step(nx, fix, qd, drafts, picks, &stop, &ran_D);
                if (E.d_adapt) E.adapt_observe(qd, a, ms_since(ts_a));
                offered += ran_D;
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                if (E.suffix) E.suffix->append(nx);
                for (int i = 0; i < a; ++i) {
                    ss.live.push_back(drafts[(size_t) i]);
                    if (E.suffix) E.suffix->append(drafts[(size_t) i]);
                }
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                continue;
                }   // qd > 0
                // qd == 0 (the learner picked the plain step): fall through to it below, untimed here
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            if (E.d_adapt && E.adapt.next == 0) {   // the learner picked the plain step: time it, so D = 0's
                const auto ts_p = Clock::now();   // cost enters the same comparison every round
                E.process(&one, 1, [](int64_t) {});
                E.adapt_observe(0, 0, ms_since(ts_p));
            } else {
                E.process(&one, 1, [](int64_t) {});
            }
            ss.live.push_back(nx);
            if (E.suffix) E.suffix->append(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        if (E.suffix && E.sfx_win > 0)
            std::fprintf(stderr, "strata-q36 SUFFIX: %lld lookup windows, %lld drafts accepted, %lld rounds skipped "
                         "(history did not match the position)\n", E.sfx_win, E.sfx_acc, E.sfx_bad);
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

/// One helper's share of the main layers.
///
/// `r` is the helper's own index in `helpers[]` / `want[]` (0..2, i.e. CUDA r+1) and is what the rest of the
/// engine keys on - `helper_of_layer` stores it and `moe_helpers` indexes `helpers[r]` with it.  `lo`/`hi` are
/// this helper's half-open layer interval.
struct HelperStripe {
    int r = -1;
    int device = 0;
    int64_t lo = 0, hi = 0;
};

/// The stripes for every enabled helper, in device order.
///
/// The split is by each device's POSITION AMONG THE ENABLED DEVICES, not by its own index.  Using the index
/// directly is the bug this replaced: with only --expert-cache-device2 set, the index is 1, so a 40-layer model
/// would hand that single helper [40, 80) - an empty window that RemoteExperts::open() rejects, aborting the run
/// at startup.  Only configurations enabled contiguously from device1 came out right, which is exactly the one
/// (device1 + device2) that had been measured.
///
/// The stripes are contiguous, disjoint, cover every layer exactly once, and none is empty: with more helpers
/// than layers only the first `n_layers` (in device order) are used and the caller reports the rest as idle,
/// which is better than refusing to start because a third flag was passed.  The last stripe takes the
/// remainder, so layers not divisible by the device count still all get an owner.
///
/// ONE table, used by both call sites.  They used to derive the same bounds from two duplicated expressions,
/// and a fix applied to only one of them would leave the cache occupancy disagreeing with the layer -> helper
/// map - layers computed on a helper the map does not point at, or the reverse.
inline std::vector<HelperStripe> helper_layer_stripes(const std::array<int, 3>& want, int64_t n_layers) {
    std::vector<HelperStripe> out;
    std::vector<int> enabled;
    for (int r = 0; r < 3; ++r)
        if (want[(size_t) r] > 0) enabled.push_back(r);
    const int64_t n = std::min<int64_t>((int64_t) enabled.size(), std::max<int64_t>(0, n_layers));
    if (n <= 0) return out;
    const int64_t per = n_layers / n;
    for (int64_t k = 0; k < n; ++k) {
        HelperStripe s;
        s.r = enabled[(size_t) k];
        s.device = s.r + 1;
        s.lo = k * per;
        s.hi = (k + 1 == n) ? n_layers : s.lo + per;
        out.push_back(s);
    }
    return out;
}

/// The helper expert tiers (--expert-cache-device1..3): publish this model's own per-layer expert geometry as the
/// process-wide layout, then give each helper device a striped share of the layers' experts.
///
/// WHY THE LAYOUT IS PUBLISHED FIRST.  `RemoteExperts::open` sizes a helper's VRAM slots from the process-wide
/// `expert_layout()` table (per-layer blob bytes, and whether the table is native at all).  That table is normally
/// written by `expert_layout_load` from a pack's native_experts.txt - and qwen35moe has no pack: this engine reads
/// its experts straight from the GGUF tensors (ffn_gate_exps / ffn_up_exps / ffn_down_exps).  Left alone, a helper
/// tier would be sized against the canonical Q2_0 blob (2560-wide, 1,382,400 B) instead of this model's, and every
/// slot would be indexed wrong.  `expert_layout_set_native` writes the same table from the formats this engine
/// already computed per layer, so the helper reads exactly the bytes `load()` would have read from the file.
///
/// WHY WHOLE LAYERS (the default).  A helper computes a routed window through `RemoteExperts::begin/finish`, which
/// keys its output on the entry index it is given (tok * k + j) and takes a window of at most MAXT tokens.  Handing a
/// helper one whole layer makes "which entries are the helper's" a question with one answer - all of them - so the
/// local grouped kernel is skipped for that layer and no entry can be computed twice or left stale.  Layers are
/// striped across the helpers in order, which keeps a helper's experts of adjacent layers together.
///
/// WHY NOT ALWAYS (STRATA_Q36_HELPER_PARALLEL=1).  The stripe is sequential by construction: a layer belongs to one
/// helper, so the other helpers sit idle for it and their GPU time is spent one after another, never at once.  The
/// measured cost of that is the `ms_wait` the two helpers each report per token with STRATA_Q36_HELPER_TIMING=1
/// (~3.3 ms, back to back).  The row split below gives every helper a share of EVERY layer's experts instead: the
/// same total residency (L*E/n experts either way), the same per-entry mask, and every helper of a layer begun
/// before any of them is finished - which is the arrangement Flash-Next already runs (src/core/expert_source.cpp
/// begins every remote tier in a loop and finishes them after the CPU pool).
bool open_helpers(Model& m, Engine& E, const std::array<int, 3>& want, const std::array<bool, 3>& auto_size,
                 std::string& err) {
    namespace C = strata::kernels::cpu;
    const auto& g = m.g;
    const int devices = (want[0] > 0 ? 1 : 0) + (want[1] > 0 ? 1 : 0) + (want[2] > 0 ? 1 : 0);
    if (devices == 0) return true;
    // RemoteExperts::open() sizes a helper's q8_1 activation buffer as cap * (n_embd / 32) * 36 and hands n_embd
    // to quantize_q8_1_rows, which reads it as a block count - so a width that is not a multiple of 32 would
    // truncate here and under-allocate there, silently.  That module's own range check does not test this (it is
    // shared with Flash-Next, whose 2560 is always exact), so the engine that passes a different geometry checks it.
    if (g.n_embd <= 0 || g.n_embd % 32 != 0) {
        err = "helper experts: n_embd " + std::to_string(g.n_embd) +
              " is not a multiple of 32 (the helper's q8_1 activation blocks are 32 values wide)";
        return false;
    }
    // the per-layer table, from each layer's own layout (never a model-level guess): gu_type / d_type / n_embd /
    // n_ff are read per layer, so a model whose layers differ is sized per layer too.  The helper layout table
    // covers the MAIN layers only: the MTP block's experts stay on the primary (its window is one draft step,
    // see helper_of_layer below), and expert_layout_set_native checks fmt.size() == n_layers - a table that also
    // carried m.mtp would be refused (41 != 40).
    std::vector<C::NativeFmt> fmt;
    std::vector<strata::kernels::NativeExpertLayout> layouts;
    fmt.reserve(m.layers.size());
    layouts.reserve(fmt.capacity());
    uint64_t total = 0;
    std::vector<const Layer*> all;
    for (const Layer& L : m.layers) all.push_back(&L);
    if (m.has_mtp) all.push_back(&m.mtp);
// The layout table covers the MAIN layers only: the MTP block's experts stay on the primary (its window is one
    // draft step), and expert_layout_set_native checks fmt.size() == n_layers - a table that also carried m.mtp
    // would be refused (41 != 40).  `all` still holds the MTP block, for the helper tier's own layer map below.
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer* L = all[i];
        C::NativeFmt f;
        if (!C::native_fmt(L->L.gu_type, L->L.d_type, g.n_embd, g.n_ff, f, err)) return false;
        if (f.up_off != L->L.up_off || f.down_off != L->L.down_off || f.bytes != L->L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
        fmt.push_back(f);
        layouts.push_back(L->L);
        // The step between two consecutive experts' blobs, for a cache that lays them out one after another.
        // The engine's own step (q36_engine.cu's Layer::stride) is the least common multiple of the two parts'
        // block sizes, so MMQ can step over the experts in whole blocks; a cache that used a different step (the
        // 256-byte aligned one the expert cache defaults to) would put every expert but the first at an offset
        // MMQ never reads, which is NaN rows rather than an error.  The helper caches get this so their slot
        // layout is the one MMQ is given.
        layouts.back().stride = L->stride;
        total += (uint64_t) f.bytes * (uint64_t) g.n_expert;
    }
    if (!C::expert_layout_set_native(g.n_layers, g.n_expert, fmt, total, err)) return false;

    // one tier per helper device, holding every layer's experts of its stripe
    E.helper_src.set(all, g.n_expert);
    std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
    // STRATA_Q36_HELPER_PARALLEL=1: every helper takes a balanced share of EVERY main layer's experts, in place of
    // the contiguous layer stripes below.  Same expert count per helper (L*E/n either way, so the same VRAM budget
    // fills the same number of slots), but which layer a helper covers is no longer what decides whether the two
    // helpers can work at once.  The share is expert e of every layer where e % n == k, k being the helper's
    // POSITION among the enabled devices (never its own index - see helper_layer_stripes on the bug that causes).
    //
    // WHY ROUND-ROBIN AND NOT CONTIGUOUS HALVES: the two helpers are a PARALLEL split, so what has to come out
    // balanced is the entries they get per layer, and MoE routing is not uniform over expert ids (a few experts
    // take most of the tokens).  Contiguous halves would put all the hot experts on one helper, which then takes
    // more entries every layer and becomes the serial tail the split was meant to remove; every helper sees the
    // same routing distribution under e % n == k, so the expected per-layer entry counts match.
    //
    // The ranked order walks EXPERTS in the outer loop and layers in the inner one, so an `auto` tier that runs out
    // of room stops with a thin but uniform slice of every layer rather than a full set of the first few layers and
    // none of the rest.  `claimed` is shared by all the opens so the shares stay disjoint by construction (each
    // helper is offered its own experts only, so nothing here depends on that guard); an expert a helper could not
    // afford is left to the local paths, which is the same partial-help behaviour the layer stripes have when a
    // helper cannot hold its whole stripe.
    const bool row_split = E.helper_parallel();
    std::vector<int> enabled;
    for (int r = 0; r < 3; ++r)
        if (want[(size_t) r] > 0) enabled.push_back(r);
    // The layer stripes, built once and used by BOTH loops below: the per-device cache build here and the
    // layer -> helper map further down.  Keyed by each device's POSITION AMONG THE ENABLED DEVICES, not by its
    // own index: with --expert-cache-device2 alone the device index is 1, and using it directly would ask that
    // helper for layers [40, 80) of a 40-layer model - an empty window, which RemoteExperts::open() rejects, so
    // the run would abort at startup.  Only configurations enabled contiguously from device1 happened to work.
    const std::vector<HelperStripe> stripes = row_split ? std::vector<HelperStripe>{}
                                                       : helper_layer_stripes(want, g.n_layers);
    if (!row_split && (int64_t) stripes.size() < devices)   // more helpers than layers: the tail ones stay idle this run
        std::fprintf(stderr, "strata-q36: %lld helper device(s) idle: %lld layers for %d helpers\n",
                     (long long) (devices - (int64_t) stripes.size()), (long long) g.n_layers, devices);
    // a primary cache of zero slots: every (layer, expert) is "not on CUDA0", so each helper takes its own stripe.
    // This is deliberately NOT the primary's real cache, so RemoteExperts::open()'s `primary.slot_of(...) < 0`
    // test admits every ranked pair and a helper may hold an expert the primary also has a slot for.  Nothing
    // breaks - the per-entry mask separates them at run time - but do not read that open() guard as "a helper only
    // ever holds experts the primary has no slot for"; it cannot, with this empty cache.
    strata::core::ExpertCache empty_primary;
    if (!empty_primary.open(1, 1, 1, 4096, err)) { err = "helper: " + err; return false; }
    // One entry per helper this run opens: its index in helpers[] / want[], its CUDA device, and the (layer,
    // expert) pairs it is offered in preference order.
    struct Tier {
        int r = -1, device = 0;
        bool row_split = false;
        int64_t e_mod = 0, e_n = 0;   // row split: this helper takes expert e of every layer where e % e_n == e_mod
        int64_t lo = 0, hi = 0;       // stripe: the layer interval
        std::vector<std::pair<int32_t, int32_t>> ranked;
    };
    std::vector<Tier> tiers;
    if (row_split) {
        const int n = (int) enabled.size();
        for (size_t k = 0; k < enabled.size(); ++k) {
            Tier t;
            t.r = enabled[k];
            t.device = t.r + 1;
            t.row_split = true;
            t.e_mod = (int64_t) k;
            t.e_n = n;
            t.ranked.reserve((size_t) ((g.n_expert + n - 1 - (int64_t) k) / n) * (size_t) g.n_layers);
            for (int64_t e = (int64_t) k; e < g.n_expert; e += n)
                for (int64_t l = 0; l < g.n_layers; ++l) t.ranked.emplace_back((int32_t) l, (int32_t) e);
            tiers.push_back(std::move(t));
        }
    } else {
        for (const HelperStripe& stripe : stripes) {
            Tier t;
            t.r = stripe.r;
            t.device = stripe.device;
            t.lo = stripe.lo;
            t.hi = stripe.hi;
            t.ranked.reserve((size_t) std::max<int64_t>(0, t.hi - t.lo) * (size_t) g.n_expert);
            for (int64_t l = t.lo; l < t.hi; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e) t.ranked.emplace_back((int32_t) l, (int32_t) e);
            tiers.push_back(std::move(t));
        }
    }
    for (const Tier& tier : tiers) {
        const int r = tier.r, device = tier.device;
        double free_gib = 0.0;
        if (!strata::core::RemoteExperts::preflight(device, free_gib, err)) return false;
        // RemoteExpertOpt's reduce() assumes ONE window (it reads win_[0] and reduces into it), so it cannot serve
        // the row split, where a layer's window lives on every helper at once.  q36 never attaches one, so this is
        // a guard against a future caller, not a configuration this engine supports.
        if (tier.row_split && E.helpers[(size_t) r].optimized_decode()) {
            err = "helper CUDA" + std::to_string(device) + ": STRATA_Q36_HELPER_PARALLEL=1 cannot be used with the "
                  "single-window remote_opt reduce (a row-split layer has one window per helper)";
            return false;
        }
        const int slots = want[(size_t) r] == std::numeric_limits<int>::max() ? (1 << 20) : want[(size_t) r];
        if (!E.helpers[(size_t) r].open(device, slots, g.n_layers, g.n_expert, tier.ranked, empty_primary,
                                        E.helper_src, claimed, err, auto_size[(size_t) r], g.n_embd, g.n_ff,
                                        m.top_k, &layouts)) {
            err = "helper CUDA" + std::to_string(device) + ": " + err;
            return false;
        }
        E.helper_count += 1;
        E.helper_on[(size_t) r] = true;
        if (tier.row_split)
            std::fprintf(stderr, "strata-q36: helper CUDA%d: %lld experts of every layer (expert %% %lld == %lld), "
                         "%.2f GiB; results return through pinned host rows\n", device,
                         (long long) E.helpers[(size_t) r].resident(), (long long) tier.e_n, (long long) tier.e_mod,
                         E.helpers[(size_t) r].gib());
        else
            std::fprintf(stderr, "strata-q36: helper CUDA%d: %lld experts of layers [%lld, %lld), %.2f GiB; results return "
                         "through pinned host rows\n", device, (long long) E.helpers[(size_t) r].resident(),
                         (long long) tier.lo, (long long) tier.hi, E.helpers[(size_t) r].gib());
    }
    E.helper_row_split = row_split;
    if (row_split)
        std::fprintf(stderr, "strata-q36: STRATA_Q36_HELPER_PARALLEL=1: %lld helper(s) each hold a share of every "
                     "layer's experts and compute one layer at the same time\n", (long long) tiers.size());
    // STRATA_REMOTE_ZEROCOPY: by default RemoteExperts tries to map its pinned staging buffers into the helper's
    // address space (src/core/remote_experts.cpp:236-239) so the helper can read input from, and write compact
    // rows into, host memory without an explicit copy.  Mapping can fail (cudaHostGetDevicePointer), in which
    // case that module already falls back to the copy path - this hint only speaks when the variable has not been
    // set to 0.  The mapped path only pays off when host-mapped access from the helper is cheap; on a box with
    // no usable peer path between the primary and the helpers it is slower than a copy, and
    // STRATA_REMOTE_ZEROCOPY=0 - an existing, supported path - is then the faster one.  Nothing probes that here:
    // cudaDeviceCanAccessPeer can report success on a topology where the real mapped access is still slow, so the
    // hint states the condition and leaves the call to whoever knows the box.  Printed only when at least one
    // helper was opened (no helper flags: byte-identical stderr), once, and not when the variable already says 0 -
    // the same parse that module uses for the env side of the decision, mirrored rather than reaching into its
    // private state.
    {
        const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
        if (!(zc != nullptr && zc[0] == '0'))
            std::fprintf(stderr, "strata-q36: helpers default to mapped host memory when the mapping succeeds: if "
                         "this box has no usable peer path between CUDA0 and the helper GPUs, set "
                         "STRATA_REMOTE_ZEROCOPY=0 - copying to device memory is then faster\n");
    }
    // the layer -> helper map (the MTP block's layer, if any, stays local: its window is one draft step).  Empty
    // under STRATA_Q36_HELPER_PARALLEL=1: there the geometry is not layer-indexed at all (helper_layer() answers
    // from helper_row_split), so a map would be a second, disagreeing answer to the same question.
    E.helper_of_layer.assign((size_t) g.n_layers, -1);
    if (row_split) {
        E.helper_of_layer.clear();
    } else {
        for (const HelperStripe& stripe : stripes)
            for (int64_t l = stripe.lo; l < stripe.hi; ++l) E.helper_of_layer[(size_t) l] = (int8_t) stripe.r;
    }
    // host staging for the offloaded windows (MAXT tokens: the largest window begin() takes)
    const size_t T = (size_t) strata::kernels::cpu::MAXT;
    CK(cudaHostAlloc((void**) &E.h_helper_ids, T * (size_t) m.top_k * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_helper_x, T * (size_t) g.n_embd * 4, cudaHostAllocDefault));
    // the per-entry mask the masked grouping kernels read, for every tier above: the all-in-VRAM placement never
    // reaches place_experts' hybrid branch, where this is otherwise allocated
    if (E.d_helper_mask == nullptr) CK(cudaMalloc(&E.d_helper_mask, E.tmax * (size_t) m.top_k));
    CK(cudaHostAlloc((void**) &E.helper_out, T * (size_t) m.top_k * (size_t) g.n_embd * 4, cudaHostAllocDefault));
    // STRATA_Q36_HELPER_ASYNC: the helpers' sparse rows staged on the device for the deferred merge into
    // `eparts` (only the entries the per-entry mask marks are written from it).  Allocated only when the
    // variable is set, so a default run's allocations are byte-for-byte what they were.
    if (E.helper_async())
        CK(cudaMalloc(&E.helper_stage, T * (size_t) m.top_k * (size_t) g.n_embd * 4));
    // STRATA_Q36_PREFILL_PEERS: the prompt-chunk executors, one per helper that opened, behind the feature's own
    // gates (the variable, the MMQ path this chunk branch needs, and a real two-way P2P copy test per helper).
    // With the variable unset this is one `getenv` and an early return: nothing is allocated, nothing is printed.
    if (E.prefill_peers() && E.use_mmq) E.peer_open(m, layouts.data());
    return true;
}

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, emb_file, kv_opt;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    // --mtp-calibrate N (opt-in): measure the acceptance of the first N speculative steps and, below the
    // measured breakeven, finish the answer as plain decode steps.  0 = off, and off is the whole default path.
    int mtp_calib = 0;
    bool mtp_adapt = false;   // --mtp-adaptive: the learned per-round depth (its number is the pool's ceiling)
    int mtp_adapt_base = 4;   // the ceiling the adaptive pool searches (0 = plain rounds too)
    int suffix_k = 0;         // --suffix-draft N: suffix-lookup drafts, 0 = off
    double expert_vram = -1.0;
    bool warm = true;
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    // --expert-cache-device1..3: 0 = off, else the slot count (INT_MAX for "auto")
    std::array<int, 3> expert_cache_remote = {0, 0, 0};
    std::array<bool, 3> expert_cache_auto = {false, false, false};
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
        else if (a == "--mtp-calibrate") mtp_calib = std::atoi(next().c_str());
        else if (a == "--suffix-draft") suffix_k = std::atoi(next().c_str());
        else if (a == "--mtp-adaptive") {
            // the learned-depth mode: draft a window and verify it every round (the text is the verified greedy
            // text), and let the per-depth cost and acceptance EMAs pick the next round's depth, including 0.
            // The number after it (default 4) is the pool's ceiling - deeper drafts are never offered.  Off by default.
            mtp_adapt = true;
            const std::string av = i + 1 < argc && argv[i + 1][0] != '-' ? argv[++i] : std::string("4");
            mtp_adapt_base = std::atoi(av.c_str());
        }
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--kv") kv_opt = next();
        else if (a == "--expert-cache-device1" || a == "--expert-cache-device2" || a == "--expert-cache-device3") {
            // a P100 plus two P104s as helper expert tiers (the same names and meaning as on the Flash-Next path):
            // sN = N experts on CUDA1..3, "auto" = as many as fit.  Off by default, and without them the q36 MoE
            // runs on the main GPU exactly as before - same kernels, same bytes.
            const int idx = a == "--expert-cache-device1" ? 0 : a == "--expert-cache-device2" ? 1 : 2;
            const std::string v = next();
            expert_cache_remote[idx] = v == "auto" ? std::numeric_limits<int>::max() : std::atoi(v.c_str());
            expert_cache_auto[idx] = v == "auto";
        }
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--kv fp32|int8|k8v32|"
                         "k32v8] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    // --kv (strata's name for the K / V stores) beats STRATA_Q36_KV; the stores are independent: int8 / k8v8 quantize
    // both, k8v32 the K alone, k32v8 the V alone.  q4_0 / k8v4 (PR #21's Hadamard rotation) are not in this engine.
    bool kv_k8 = false, kv_v8 = false;   // the stores the Engine runs with (its members are set from these below)
    {
        std::string kv = kv_opt;
        if (kv.empty()) { const char* e = std::getenv("STRATA_Q36_KV"); kv = e != nullptr ? e : "fp32"; }
        if (kv == "fp32" || kv == "f32") {}   // both stores FP32 (the default)
        else if (kv == "int8" || kv == "k8v8") kv_k8 = kv_v8 = true;
        else if (kv == "k8v32") kv_k8 = true;
        else if (kv == "k32v8") kv_v8 = true;
        else {
            std::fprintf(stderr, "strata-q36: --kv must be fp32, int8, k8v8, k8v32 or k32v8 (q4_0 and k8v4 are PR "
                                 "#21's rotation kernels: not in this engine)\n");
            return 2;
        }
        if (kv_k8 || kv_v8)
            std::fprintf(stderr, "strata-q36: KV cache: K %s, V %s\n", kv_k8 ? "int8" : "fp32", kv_v8 ? "int8" : "fp32");
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
    if (mtp_calib < 0 || mtp_calib > 64) {
        std::fprintf(stderr, "strata-q36: --mtp-calibrate takes 0 .. 64 probe steps\n");
        return 2;
    }
    if (mtp_calib > 0 && mtp_d == 0) {   // it only has meaning once something is being speculated
        std::fprintf(stderr, "strata-q36: --mtp-calibrate needs --mtp N (the drafts it measures): ignored\n");
        mtp_calib = 0;
    }
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
    E.kv_k8 = kv_k8;
    E.kv_v8 = kv_v8;
    E.use_graph = graph;
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
    if (!open_helpers(m, E, expert_cache_remote, expert_cache_auto, err)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    if (E.mtp_mapped()) E.report_mapped_mtp();   // STRATA_Q36_MTP_MAPPED: live, or inert and why
    struct Report {
        Engine& E;
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "strata-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (E.mtp_top2()) {
                // Per draft position: how many were verified / accepted, and (of the rejected) how often the runner-up
                // was the token verify picked.  The last column is the diagnostic: a high rate means the drafter ranks
                // the right token second (head precision), a low rate means its input is wrong (seed / position).
                std::fprintf(stderr, "strata-q36 MTP-TOP2: per draft position (d: verified accepted | rejected runner-up-hit rate)\n");
                for (int d = 0; d < 16; ++d) {
                    if (E.dp_tested[d] == 0) continue;
                    const long long rej = E.t2_rej[d], hit = E.t2_hit[d];
                    std::fprintf(stderr, "strata-q36 MTP-TOP2:   d=%d: %lld %lld (%.1f%%) | %lld %lld (%.1f%%)%s\n", d,
                                 E.dp_tested[d], E.dp_ok[d],
                                 E.dp_tested[d] ? 100.0 * (double) E.dp_ok[d] / (double) E.dp_tested[d] : 0.0,
                                 rej, hit, rej ? 100.0 * (double) hit / (double) rej : 0.0,
                                 d > 0 && E.t2_rej[d] > 0 && E.t2_hit[d] == 0 ? "  <- the second choice never matched" : "");
                }
            } else if (E.dp_tested[0] > 0) {
                // The same per-depth acceptance the aggregate ratio hides, without the runner-up column.  Cheap enough
                // for the default path: it reads counters that were already being incremented.
                std::fprintf(stderr, "strata-q36 MTP-DEPTH: per draft position (verified, accepted, rate)\n");
                for (int d = 0; d < 16; ++d) {
                    if (E.dp_tested[d] == 0) continue;
                    std::fprintf(stderr, "strata-q36 MTP-DEPTH:   d=%d: %lld %lld (%.1f%%)\n", d, E.dp_tested[d],
                                 E.dp_ok[d], 100.0 * (double) E.dp_ok[d] / (double) E.dp_tested[d]);
                }
            }
            if (E.d_adapt) {
                // --mtp-adaptive: where the learned depth spent the rounds, and what it learned per depth.
                std::fprintf(stderr, "strata-q36 MTP-ADAPT: rounds per draft depth and the learned means "
                             "(acceptance / step ms of the depth it ran at)\n");
                for (int d = 0; d <= E.adapt.base; ++d) {
                    if (E.adapt.use[(size_t) d] == 0) continue;
                    if (d == 0)
                        std::fprintf(stderr, "strata-q36 MTP-ADAPT:   D=0: %d rounds (plain steps), step %.1f ms\n",
                                     E.adapt.use[0], E.adapt.time[0]);
                    else
                        std::fprintf(stderr, "strata-q36 MTP-ADAPT:   D=%d: %d rounds, %.1f drafts accepted, step %.1f ms\n",
                                     d, E.adapt.use[(size_t) d],
                                     E.adapt.n[(size_t) d] > 0 ? E.adapt.acc[(size_t) d] : 0.0,
                                     E.adapt.time[(size_t) d]);
                }
                std::fprintf(stderr, "strata-q36 MTP-ADAPT: the pool is D=0..%d; the next round drafts D=%d\n",
                             E.adapt.base, E.adapt.next);
            }
            // STRATA_Q36_PREFILL_PEERS: the layers the helpers' row window could not take.  Printed only when it
            // happened - peer_over_cap is 0 in every run with the flag off, so this is zero stderr bytes there -
            // and ABOVE the hybrid early return below, because a helper tier can be open on a run where every
            // expert fits in VRAM (place_experts returns before m.hybrid is set) and the fallback happened there too.
            if (E.peer_over_cap != 0)
                std::fprintf(stderr, "strata-q36: STRATA_Q36_PREFILL_PEERS: %lld prompt-chunk layer(s) exceeded the "
                             "helpers' %lld-entry row window and ran on the primary\n", (long long) E.peer_over_cap,
                             (long long) E.peer_cap_rows);
            if (!E.m->hybrid) return;
            if (E.helpers_enabled())
                std::fprintf(stderr, "strata-q36: helper experts: %lld batch entries on the helper GPUs\n",
                             E.hy_helper);
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (suffix_k > 0) {
        if (!m.has_mtp) {
            std::fprintf(stderr, "strata-q36: --suffix-draft needs the model's MTP block (its verified window): off\n");
        } else if (suffix_k > 7) {
            std::fprintf(stderr, "strata-q36: --suffix-draft takes 1 .. 7: off\n");
        } else if (mtp_adapt) {
            std::fprintf(stderr, "strata-q36: --suffix-draft and --mtp-adaptive both pick the window: --suffix-draft off\n");
        } else {
            E.suffix = std::make_unique<strata::qwen35::Q36Suffix>(suffix_k);
            std::fprintf(stderr, "strata-q36: --suffix-draft %d (lookup windows replace MTP drafts when they win)\n",
                         suffix_k);
        }
    }
    if (mtp_adapt) {
        if (mtp_adapt_base < 0 || mtp_adapt_base > 7) { std::fprintf(stderr, "strata-q36: --mtp-adaptive takes 0 .. 7\n"); return 2; }
        if (E.mtp_d == 0) {   // with no drafts to offer the learned depth is always 0: the loop the answer ran before
            std::fprintf(stderr, "strata-q36: --mtp-adaptive needs --mtp N with drafts to learn from: off\n");
            mtp_adapt = false;
        } else {
            E.d_adapt = true;
            E.adapt.base = mtp_adapt_base;
            // the D the first speculative round asks for: --mtp-adaptive keeps the command line's depth as the seed
        // (the learner measures every depth in its pool anyway), else the pool's ceiling.
        E.adapt.next = E.d_adapt ? std::min(E.mtp_d, mtp_adapt_base) : E.mtp_d;
        }
        if (E.d_adapt) std::fprintf(stderr, "strata-q36: --mtp-adaptive on: the pool is D=0..%d, the first round drafts D=%d\n",
                                    mtp_adapt_base, E.mtp_d);
    }
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.draft_n = std::max(0, draft_vocab);
    if (E.mtp_top2()) {
        // STRATA_Q36_MTP_TOP2=1: host staging for the draft head's logits.  Allocated only under the flag, so a
        // default run allocates nothing it did not before.
        if (!E.h_logits) CK(cudaHostAlloc((void**) &E.h_logits, (size_t) m.g.n_vocab * 4, cudaHostAllocDefault));
        std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_TOP2=1 (draft runner-up diagnostic on)\n");
    }
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-q36: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        // STRATA_Q36_HELPER_TIMING=1: Phase-3 tg-only RemoteExperts host-time deltas (throwaway measurement).
        const char* ht = std::getenv("STRATA_Q36_HELPER_TIMING");
        const bool helper_timing = ht && ht[0] == '1' && ht[1] == '\0';
        // Per pass (r_begin/r_wait/r_ovh) and the 3-pass aggregate (d_*), so every printed tg tok/s sits next to
        // its OWN per-token numbers: the wall clock quoted from a run is one pass (the steady one), and averaging
        // ms/token over the warm-up passes biases the comparison against it.
        double d_begin[3] = {}, d_wait[3] = {}, d_ovh[3] = {};
        double d_pre = 0, d_spin = 0;
        long long n_pre = 0, n_spin = 0;
        // STRATA_Q36_HELPER_TIMING=1: the primary GPU's own per-token segment totals, same per-pass + aggregate
        // treatment as the host counters above.  gseg_ms[j] is segment (E_j, E_j+1) in ms for the whole pass.
        double d_gseg[Engine::kGsegN] = {}, d_span = 0, d_resid = 0, d_emb = 0, d_head = 0;
        long long d_gn = 0, d_dropped = 0;
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            double begin0[3] = {}, wait0[3] = {}, ovh0[3] = {};
            double pre0 = 0, spin0 = 0;
            long long pn0 = 0, sn0 = 0;
            double g0[Engine::kGsegN] = {}, span0 = 0, resid0 = 0, emb0 = 0, head0 = 0;
            long long gn0 = 0, drop0 = 0;
            if (helper_timing) {
                for (int r = 0; r < 3; ++r) {
                    begin0[r] = E.helpers[(size_t) r].ms_begin();
                    wait0[r] = E.helpers[(size_t) r].ms_wait();
                    ovh0[r] = E.helpers[(size_t) r].ms_sync_overhead();
                }
                pre0 = E.d_prebegin_ms; spin0 = E.d_route_spin_ms;
                pn0 = E.d_prebegin_n; sn0 = E.d_route_spin_n;
                for (int j = 0; j < Engine::kGsegN; ++j) g0[j] = E.gseg_ms[j];
                span0 = E.gspan_ms; resid0 = E.gp_resid_ms;
                emb0 = E.gp_embed_ms; head0 = E.gp_head_ms;
                gn0 = E.gspan_n; drop0 = E.gseg_dropped;
            }
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            const double per_tok = bench_tg > 0 ? 1000.0 * tg_ms / (double) bench_tg : 0.0;   // ms/token, this pass
            double r_begin[3] = {}, r_wait[3] = {}, r_ovh[3] = {};
            double r_pre = 0, r_spin = 0;
            long long r_pn = 0, r_sn = 0;
            double r_gseg[Engine::kGsegN] = {}, r_span = 0, r_resid = 0, r_emb = 0, r_head = 0;
            long long r_gn = 0, r_drop = 0;
            if (helper_timing) {
                for (int r = 0; r < 3; ++r) {
                    r_begin[r] = E.helpers[(size_t) r].ms_begin() - begin0[r];
                    r_wait[r] = E.helpers[(size_t) r].ms_wait() - wait0[r];
                    r_ovh[r] = E.helpers[(size_t) r].ms_sync_overhead() - ovh0[r];
                    d_begin[r] += r_begin[r];
                    d_wait[r] += r_wait[r];
                    d_ovh[r] += r_ovh[r];
                }
                r_pre = E.d_prebegin_ms - pre0;
                r_spin = E.d_route_spin_ms - spin0;
                r_pn = E.d_prebegin_n - pn0;
                r_sn = E.d_route_spin_n - sn0;
                d_pre += r_pre;
                d_spin += r_spin;
                n_pre += r_pn;
                n_spin += r_sn;
                for (int j = 0; j < Engine::kGsegN; ++j) {
                    r_gseg[j] = E.gseg_ms[j] - g0[j];
                    d_gseg[j] += r_gseg[j];
                }
                r_span = E.gspan_ms - span0;   d_span += r_span;
                r_resid = E.gp_resid_ms - resid0; d_resid += r_resid;
                r_emb = E.gp_embed_ms - emb0;  d_emb += r_emb;
                r_head = E.gp_head_ms - head0;  d_head += r_head;
                r_gn = E.gspan_n - gn0;        d_gn += r_gn;
                r_drop = E.gseg_dropped - drop0; d_dropped += r_drop;
            }
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            if (helper_timing && bench_tg > 0)
                for (int r = 0; r < 3; ++r) {
                    if (E.helpers[(size_t) r].resident() <= 0) continue;
                    std::fprintf(stderr,
                                 "strata-q36 PHASE3 bench %d helper CUDA%d: %.3f ms/tok wall | helper ms_begin=%.3f "
                                 "ms_wait=%.3f ms_sync_overhead=%.3f ms/tok\n",
                                 rep, r + 1, per_tok, r_begin[r] / (double) bench_tg,
                                 r_wait[r] / (double) bench_tg, r_ovh[r] / (double) bench_tg);
                }
            if (helper_timing && bench_tg > 0)
                std::fprintf(stderr,
                             "strata-q36 PHASE3 bench %d primary: %.3f ms/tok wall | prebegin_sync=%.3f "
                             "ms/tok (%lld calls) route_spin=%.3f ms/tok (%lld calls)\n",
                             rep, per_tok, r_pre / (double) bench_tg, r_pn,
                             r_spin / (double) bench_tg, r_sn);
            if (helper_timing && bench_tg > 0 && E.gpu_timing) {
                std::fprintf(stderr,
                             "strata-q36 PHASE3 bench %d GPU primary: %.3f ms/tok wall | layers=%lld tiles=%lld |",
                             rep, per_tok, r_gn, (long long) bench_tg);
                double segsum = 0;
                for (int j = 0; j + 1 < Engine::kGsegN; ++j) {
                    segsum += r_gseg[j];
                    std::fprintf(stderr, " %s=%.3f", Engine::kGsegName[j], r_gseg[j] / (double) bench_tg);
                }
                std::fprintf(stderr, " | tile=%.3f segsum=%.3f resid=%.3f (per ms/tok)\n",
                             r_span / (double) bench_tg, segsum / (double) bench_tg, r_resid / (double) bench_tg);
            }
            std::fflush(stdout);
        }
        if (helper_timing && bench_tg > 0) {
            const int tokens = bench_tg * 3;
            for (int r = 0; r < 3; ++r) {
                if (E.helpers[(size_t) r].resident() <= 0) continue;
                std::fprintf(stderr,
                             "strata-q36 PHASE3 AGG benches 0-2 helper CUDA%d: ms_begin=%.1f (%.3f ms/tok) ms_wait=%.1f (%.3f ms/tok) ms_sync_overhead=%.1f (%.3f ms/tok) tokens=%d\n",
                             r + 1, d_begin[r], d_begin[r] / (double) tokens, d_wait[r],
                             d_wait[r] / (double) tokens, d_ovh[r], d_ovh[r] / (double) tokens, tokens);
            }
            std::fprintf(stderr,
                         "strata-q36 PHASE3 AGG benches 0-2 primary: prebegin_sync=%.1f (%.3f ms/tok, %lld calls) route_spin=%.1f (%.3f ms/tok, %lld calls) tokens=%d\n",
                         d_pre, d_pre / (double) tokens, n_pre, d_spin, d_spin / (double) tokens, n_spin,
                         tokens);
            if (E.gpu_timing) {
                std::fprintf(stderr, "strata-q36 PHASE3 AGG GPU primary: layers=%lld dropped=%lld |", d_gn, d_dropped);
                double segsum = 0;
                for (int j = 0; j + 1 < Engine::kGsegN; ++j) {
                    segsum += d_gseg[j];
                    std::fprintf(stderr, " %s=%.3f", Engine::kGsegName[j], d_gseg[j] / (double) tokens);
                }
                std::fprintf(stderr, " | tile=%.3f segsum=%.3f resid=%.3f embed=%.3f head=%.3f (per ms/tok)\n",
                             d_span / (double) tokens, segsum / (double) tokens, d_resid / (double) tokens,
                             d_emb / (double) tokens, d_head / (double) tokens);
            }
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
    // --mtp-calibrate (opt-in).  THE WHOLE PROBE IS OUT OF BAND AND RUNS BEFORE THE PROMPT IS EVER READ, so the
    // answer reads the prompt exactly once and starts from exactly the state a plain run starts from.
    //
    // It reads the prompt, times a few speculative steps and one plain step, picks the faster path, and discards
    // everything it touched.  What "everything" has to cover is not obvious, and each omission changes the answer
    // (all three measured on 192.168.1.98 with --mtp 1 --mtp-calibrate 8, against the no-drafts reference):
    //   * the recurrent (GDN) state and pos - discard_pass(), which is reset()
    //   * the routing counts the probe's steps fed to rebalance() - clear_route_counts(): left in, they make the
    //     answer swap 18 slots where a plain run swaps 1640, and it diverges at token 10
    //   * the slots themselves - `warming` for the length of the probe, because a slot swap is a device copy the
    //     engine cannot undo, and a different expert in a slot is a different arithmetic and a different token
    // The KV cache needs no clearing: the answer's prompt read rewrites those cells.
    //
    // Why the answer cannot simply switch mid-flight instead: a verify window commits its rows through commit(a+1)
    // (spec_step) while a plain step commits through process(1), and those are different kernels for the recurrent
    // state, so the two paths are not bit-comparable and a handover diverges.  Measured: switching mid-answer gave
    // 9 identical tokens and a different 10th; timing a plain step inside the answer gave 13 identical and a
    // different 14th (acceptance 100% -> 96%).  Speculation has to preserve the text, so it does not.
    //
    // The economics, on this run's own timings.  A D-draft step costs S ms and yields (1 + a) tokens, where a is the
    // accepted count DIVIDED BY THE DRAFTS OFFERED (a D=1 step's a is 0 or 1, so it is its own hit rate); a plain step
    // costs P ms and yields 1.  Speculation wins iff (1 + a)/S > 1/P, i.e. iff a > S/P - 1 =: a*(D).
    //
    // Three defects in the first version of this probe were measured on 192.168.1.98 (--mtp 1 --mtp-calibrate 8,
    // --n 48, against the same binary's --mtp 1 --mtp-calibrate 0), and all three pushed toward dropping drafts:
    //
    //  1. It judged only the D the command line asked for, so `--mtp 2` correctly rejected D=2 and then threw the
    //     D=1 gain away with it (P2 +7.3%, P3 +15.9%, measured).  Both are now measured: D=1 first, and a larger D
    //     only if the D below it already paid.  The per-depth acceptance falls off a cliff with depth (P3 --mtp 4:
    //     d=0 81.8%, d=1 33.3%, d=2 5.0%, d=3 0.0%), so a(D) falls and S(D) rises with D: the first D to lose is the
    //     last one tried, and that is what the escalation assumes.  The D the probe picks is the D the answer reads
    //     with, so `--mtp 2 --mtp-calibrate 8` reads like `--mtp 1 --mtp-calibrate 8` when D=1 is the winner.
    //  2. It timed ONE plain step, and a first step of any shape is not the step the answer runs: graph_run
    //     instantiates the shape's graph on first use (so D=2's first step pays gv[3], too), and the CPU pool's
    //     threads and pinned staging are cold.  Measured 16.8 ms against a sustained 21.5-22.1 ms, which put a* far
    //     too low and called a winning configuration a loser.  Both windows now discard kCalibBurnPos positions of
    //     burn-in (run, not timed) and then time mtp_calib steps and take the MEDIAN, which is what the sustained
    //     cost is: no fudge factor, the same number the answer's own steps average.
    //  3. Its acceptance denominator was mtp_calib * E.mtp_d while spec_step narrows D itself near a rebalance
    //     boundary (`D = min(D, kRebalanceEvery - pos % kRebalanceEvery - 1)`), so windows that offered fewer drafts
    //     than asked were charged for drafts they never made.  The denominator is now spec_step's own ran_D, summed
    //     over the timed window exactly as the answer's report line sums it, so the probe's ratio and the answer's
    //     "X of Y accepted" are the same measurement.  The remaining shortfall was the position: 8 steps from the
    //     head of a prompt read 62% on P2 where 48 tokens of sustained decode read 74%, which the burn-in is there to
    //     close.  A short window cannot be made unbiased, only moved off the head of the prompt: kCalibBurnPos
    //     positions of it are run and thrown away, and what is left is 8 steps starting one rebalance period in.
    //
    // `warming` stays on for every timed window.  Off, rebalance() would move slots inside the probe and a moved
    // slot is a different expert and a different token stream in the answer that follows (measured: 9 tokens shared
    // and a different 10th); the burn-in buys the pool's warmth without buying a placement change, since the timed
    // windows are consecutive steps over positions the engine was already going to decode.
    // ----
    // The median of the times a window measured.  Sorted in place, so the caller does not keep the order.
    auto calib_median = [](std::vector<double>& v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    bool speculating = true;
    double plain_ms = 0.0;
    // Positions of burn-in before each timed window: one kRebalanceEvery period, so the window sits on a rebalance
    // boundary the answer's own decode crosses too, and is far enough past the prompt for the CPU pool's threads to
    // be warm.  A D-draft step commits at most D+1 positions, so the burn is measured in positions, not steps.
    const int kCalibBurnPos = 32;
    int calib_D = E.mtp_d;   // the D the answer reads with: E.mtp_d unless the probe found a better one
    if (mtp_calib > 0) {
        std::vector<int32_t> pfix, pdrafts;
        int32_t ppicks[16];
        // One D's measurement: a fresh prompt read, the burn-in, then mtp_calib timed steps.  The timing flag is
        // taken BEFORE the step, so a step that crosses burn_to is the first timed one (its predecessor was the last
        // that started short of it) and no step is both run and timed twice.
        auto calib_spec = [&](int D, double& out_spec_ms, double& out_a, long long& out_acc, long long& out_offered,
                              int64_t& out_burn) {
            E.discard_pass();
            E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
            pfix.clear();   // a fresh read has no redo rows; the previous D's must not seed this one's first window
            // the probe writes KV cells like any other pass: never past the cache, so the burn gives way when the
            // measured window would not fit
            const int64_t room = (int64_t) E.ctx - E.pos;
            const int64_t want = (int64_t) mtp_calib * (D + 1);
            const int64_t burn = std::max<int64_t>(0, std::min<int64_t>(kCalibBurnPos, room - want));
            out_burn = burn;
            int pnx = E.greedy();
            E.warming = true;   // the probe is not a request: no slot swaps, as in warmup()'s throw-away passes
            const int64_t burn_to = E.pos + burn;
            long long acc = 0, offered = 0;
            std::vector<double> step_ms;
            while (E.pos < burn_to || (int) step_ms.size() < mtp_calib) {
                const bool timed = E.pos >= burn_to;
                int ran_D = 0;
                const auto ts = Clock::now();
                const int a = E.spec_step(pnx, pfix, D, pdrafts, ppicks, nullptr, &ran_D);
                const double dt = ms_since(ts);
                pfix.assign(pdrafts.begin(), pdrafts.begin() + a);
                pnx = ppicks[a];
                if (timed) {   // the burn-in positions count as neither tokens nor drafts
                    step_ms.push_back(dt);
                    acc += a;
                    offered += ran_D;
                }
            }
            E.warming = false;
            out_spec_ms = calib_median(step_ms);
            out_a = offered > 0 ? (double) acc / (double) offered : 0.0;
            out_acc = acc;
            out_offered = offered;
        };
        // The plain cost, on this run's own machine, and FIRST: every D is judged against it, and the D=1 measurement
        // is the one the plain window must not be contaminated by.  Same burn-in as the speculative windows, then
        // mtp_calib steps along the same greedy chain the plain answer loop walks, so the median is the per-token cost
        // the answer would really pay - not one cold step out of a fresh prompt read, which measured 16.8 ms against a
        // sustained 21.5-22.1 ms and put every breakeven far too low.
        E.discard_pass();
        E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
        {
            const int64_t room = (int64_t) E.ctx - E.pos;
            const int64_t burn = std::max<int64_t>(0, std::min<int64_t>(kCalibBurnPos, room - mtp_calib));
            int32_t pnx = E.greedy();
            E.warming = true;
            const int64_t burn_to = E.pos + burn;
            std::vector<double> step_ms;
            while (E.pos < burn_to || (int) step_ms.size() < mtp_calib) {
                const auto tp = Clock::now();
                E.process(&pnx, 1, [](int64_t) {});
                step_ms.push_back(ms_since(tp));   // greedy()'s own device-to-host copy is outside the bracket
                pnx = E.greedy();
            }
            E.warming = false;
            plain_ms = calib_median(step_ms);
        }
        // D = 1 first, then each larger D only while the one below it paid.  `escalate` is what makes `--mtp 2` able
        // to find that D=1 wins; the margin, not the raw a, is what is compared, because each D has its own a*(D).
        int best_D = 0;
        double best_margin = -1.0, best_a = 0.0, best_spec_ms = 0.0, best_a_star = 0.0;
        long long best_acc = 0, best_offered = 0, acc = 0, offered = 0;
        bool escalate = true;
        for (int D = 1; D <= E.mtp_d && escalate; ++D) {
            double s_ms = 0.0, a_D = 0.0, star_D = 0.0;
            int64_t burn = 0;
            calib_spec(D, s_ms, a_D, acc, offered, burn);
            star_D = plain_ms > 0.0 ? s_ms / plain_ms - 1.0 : 1.0;
            const double margin = a_D - star_D;
            std::fprintf(stderr, "strata-q36: --mtp-calibrate D=%d over %d step%s after %lld burn-in position%s: "
                         "acceptance %.0f%% (%lld of %lld offered), speculative step %.1f ms (median), "
                         "breakeven %.0f%% -> %s\n",
                         D, mtp_calib, mtp_calib == 1 ? "" : "s", (long long) burn, burn == 1 ? "" : "s",
                         100.0 * a_D, acc, offered, s_ms, 100.0 * star_D,
                         margin > 0.0 ? "drafts are faster at this depth" : "drafts are slower at this depth");
            if (margin > best_margin) {
                best_margin = margin;
                best_D = D;
                best_a = a_D;
                best_spec_ms = s_ms;
                best_a_star = star_D;
                best_acc = acc;
                best_offered = offered;
            }
            // a deeper window is only tried while this one won: the acceptance curve measured on P3 (81.8 / 33.3 /
            // 5.0 / 0.0 per depth) and the growing step cost put every deeper D below this one
            escalate = margin > 0.0;
        }
        speculating = best_D > 0 && best_margin > 0.0;
        calib_D = speculating ? best_D : 0;
        std::fprintf(stderr, "strata-q36: --mtp-calibrate %d probe step%s: plain step %.1f ms (median), %s\n",
                     mtp_calib, mtp_calib == 1 ? "" : "s", plain_ms,
                     speculating ? "drafts are faster, this answer reads with them"
                                : "drafts are slower, this answer reads as plain decode");
        if (speculating)
            std::fprintf(stderr, "strata-q36: --mtp-calibrate chose D=%d: acceptance %.0f%% (%lld of %lld offered), "
                         "speculative step %.1f ms, breakeven %.0f%% (asked for --mtp %d)\n",
                         calib_D, 100.0 * best_a, best_acc, best_offered, best_spec_ms, 100.0 * best_a_star, E.mtp_d);
        E.discard_pass();   // nothing of the probe remains; the answer's own prompt read follows
        E.clear_route_counts();
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if ((E.mtp_d > 0 || E.suffix) && forced.empty() && !ff) {
        // the D the answer drafts with: what was asked for, unless --mtp-calibrate found a deeper-or-shallower D that
        // pays better (it measured D=1 whether or not 1 was the --mtp value, which is the whole point of fix 1).  With
        // the flag unset this is E.mtp_d and the loop below is the one that was always there.
        const int gen_D = calib_D;
        if (E.d_adapt) E.adapt.next = gen_D;   // the learner continues from the calibrate result when both are on
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        if (E.suffix) {
            E.suffix->append(prompt.data(), prompt.size());
            // the spec_step below needs the history length to equal pos; after the prompt read, pos == prompt.size()
        }
        while ((int) gen.size() < n_gen) {
            // the depth this round: with --mtp-adaptive the learned one (0 = a plain step), else the asked-for
            // gen_D - and with --mtp-calibrate the probe's verdict stands: it dropped the drafts and speculating
            // is false, so the rounds are plain steps, as before.
            const bool drafting = E.d_adapt ? true : (speculating || E.suffix);
            const int qd = E.d_adapt ? E.adapt.next : gen_D;
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            if (drafting && (qd > 0 || E.suffix)) {
                int ran_D = 0;
                const auto ts_a = E.d_adapt ? Clock::now() : Clock::time_point{};
                const int a = E.spec_step(nx, fix, qd, drafts, picks, nullptr, &ran_D);
                if (E.d_adapt) E.adapt_observe(qd, a, ms_since(ts_a));
                offered += ran_D;
                accepted += a;
                // spec_step committed x and the a accepted drafts, and advanced pos by a + 1.  The history has to
                // follow the same tokens, or the next round's pos != size() check skips the lookup for good.
                if (E.suffix) E.suffix->append(nx);
                fix.assign(drafts.begin(), drafts.begin() + a);
                for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) {
                    gen.push_back(drafts[(size_t) i]);
                    if (E.suffix) E.suffix->append(drafts[(size_t) i]);
                }
                nx = picks[a];
            } else {
                // one token per step.  With the learner on, the plain step's time feeds D = 0's EMA (its committed
                // token against the step's cost), and its rounds are the only evidence a draft depth worth picking
                // again can win back from;
                if (E.d_adapt) {
                    const auto ts_p = Clock::now();
                    E.process(&nx, 1, [](int64_t) {});
                    E.adapt_observe(0, 0, ms_since(ts_p));
                } else
                    E.process(&nx, 1, [](int64_t) {});
                if (E.suffix) E.suffix->append(nx);
                fix.clear();   // a plain step committed its own token: no old drafts seed the next window
                nx = E.greedy();
            }
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)%s\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), gen_D, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0,
                     mtp_calib > 0 ? (speculating ? ", drafts kept by the probe" : ", drafts dropped by the probe")
                                   : (E.d_adapt ? ", the learned depth" : ""));
        return 0;
    }    for (int i = 0; i < n_gen; ++i) {
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
