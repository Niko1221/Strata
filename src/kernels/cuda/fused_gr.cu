// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/core/emulate.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "s26_tsum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
// the latency-hidden norm/up (gr_norm_fast_kernel / gr_up_fast_kernel, STRATA_GR_FAST): gfx906's, built for CUDA too
#if defined(STRATA_HIP_GFX906) || !defined(__HIPCC__)
#define STRATA_GR_FAST_BUILD 1
#else
#define STRATA_GR_FAST_BUILD 0
#endif
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
        float acc = dot8(__ldg(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};


// S26 STRATA_QFUSE=1: the q8_1 image of `mixed`, written by the up kernels below. A q8_1 block is 32 columns and an
// up block owns UPM_COLS = 16, so the second of the two blocks that own a 32-column group (a per-group counter,
// incremented after the block's writes are fenced, reset by that block for the next launch) reads the 32 values back
// and quantizes them with native_quantize_q8_1_kernel's quantizer: one warp per token, the same XOR-tree max and sum,
// d = amax / 127, roundf(x / d), ds = (d, sum) - the bytes the separate launch writes.
struct GrQ81 { half2 ds; int8_t qs[32]; };
static_assert(2 * 16 == 32, "two up blocks per q8_1 group");
__device__ __forceinline__ void gr_q8_tail(const GrMulti& m, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[warp].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = make_half2(d, sum);
}
// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane); cards whose
// opt-in below 8 * 2560 * 4 B slices the tokens - sm_75 (64 KiB) carries 6 tokens of it - run TILEV 1280 instead,
// which fits all eight tokens in one 40 KiB launch and stages 160 chunks of 8 (5 per lane, half the registers held
// for the weight prefetch); smaller tiles raise how many blocks share an SM (the 41-block grid), e.g. three blocks
// of four tokens instead of one on sm_75.
template <int TILEV, int MAX_T = kFusedGrMaxT>
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    constexpr int TQ = TILEV / 8 / 32;      // uint4 weight chunks per lane per tile
    extern __shared__ __align__(16) float tile[];   // [T][TILEV]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILEV) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// ================================ hc read v3 (opt-in: STRATA_GR_V3=1) - two kernels, stream-split ================
// The norm kernel runs on only T blocks (~16 us of pure latency per call) and `down` on 41 blocks (~280 GB/s).
// v3: `down` is split over (row group, stream[, column half]) = 164 or 328 blocks; each block stages its slice of
// R' * w_norm for the T tokens, reduces that slice's sum of squares itself, and writes UNSCALED partial dots.  The
// rms scale is per stream, so  w_down . xn = sum_c rs[c] * (w_down[:, c] . (R'[c] * w_norm[c]))  - `up` applies it
// in its prologue (lo, inject, rs).  Same maths, ANOTHER SUMMATION ORDER: not bitwise the default kernels, hence
// opt-in.  Dynamic shared memory is T * (N / S) floats: S (1 or 2 column halves) is the smallest that fits the
// card's opt-in limit at kFusedGrMaxT tokens (Ampere 99 KB: S = 1; Turing / HIP 64 KB: S = 2); a card where
// neither fits keeps the default kernels.
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject
constexpr int TQ3 = N / 8 / 32;                    // uint4 weight chunks per lane in one stream's slice (10)

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_v3_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    extern __shared__ __align__(16) float xs[];    // [T][N / S]
    constexpr int R2 = 1;                          // down rows per warp
    __shared__ float red[WARPS][MAX_T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = blockIdx.x, c = blockIdx.y / S, h = blockIdx.y - (blockIdx.y / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    float ssp[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
        if (!EXACT_T && k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N + (size_t) h * SL);
        const float4* G4 = reinterpret_cast<const float4*>(a.w_norm + (size_t) c * N + (size_t) h * SL);
        const float4* B4 = reinterpret_cast<const float4*>(a.bo_prev + (size_t) h * SL);
        float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            float4 r = R4[i];
            if (a.apply) {
                const float4 b = B4[i];
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = G4[i];
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            X4[i] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N + (size_t) h * SL);
        float acc[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            const Bf16x8 wvq = unpack8(__ldg(w4 + j));
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (EXACT_T || k < T) acc[k] += dot8u_ptr(wvq, xs + (size_t) k * SL + j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
    }
}

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_v3_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float rsS[MAX_T][HC];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][r] = x / (1.0f + __expf(-x));
    }
    if (blockIdx.x == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = fmaf(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}
// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; then it recomputes its R' * w_norm and writes it times rs (the plain read stored the product and
//    scaled it in place: the same two roundings).  On an RTX 4080 SUPER, the plain read's norm took 0.58 ms per
//    4-token round (96 launches of 1-4 blocks), this one 0.23 ms.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.  Before sm_80
//    (and on HIP) the staging is a plain copy: the same bits, only not asynchronous.
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

__global__ void __launch_bounds__(THREADS) gr_norm_split_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        const float px = r.x * g.x, py = r.y * g.y, pz = r.z * g.z, pw = r.w * g.w;
        *reinterpret_cast<float4*>(xn + i) = make_float4(px * rs, py * rs, pz * rs, pw * rs);
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order
__device__ __forceinline__ float dot8v(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__device__ __forceinline__ void stage_htile(const GrMulti& m, int T, int h, float4* buf, int t) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_staged_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];      // 2 buffers x [T][2][160] float4
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// the current device is Volta (sm_70): the fast norm/up is its default
bool cur_dev_volta() {
#if defined(__HIPCC__)
    return false;
#else
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = major == 7 && minor == 0 ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (1280 on sm_75, 2560 elsewhere), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device (a per-DEVICE setting: a layer split runs these kernels on two cards).
int down_chunk(bool staged, int* tile_out) {
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the plain down kernel may carry in one launch on this card
    static int chunk_staged[64] = {};  // the same for staged
    static int tile[64] = {};    // the plain down kernel's TILEV on this card (1280 on sm_75, 2560 elsewhere)
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        *tile_out = 2560;
        return kFusedGrMaxT;
    }
    if (!attr[dev]) {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        optin = strata::smem_optin_of(optin);   // STRATA_EMULATE_CC (tests only)
        // the down kernel stages n_tok*TILEV floats of dynamic shared memory - 80 KB at the full 8 tokens of the
        // CUDA tile.  sm_75 gets the smaller tile: all eight tokens fit one 40 KiB launch there (no more slicing),
        // the TQ-5 prefetch holds half the registers, and the smaller blocks raise how many of the 41-block grid
        // share an SM.  Cards whose opt-in is still below that (or that report no opt-in at all) slice the tokens;
        // the down kernel's outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and
        // the up kernel below still sees every token of the batch in one launch.
#if defined(__HIPCC__)
#if defined(STRATA_HIP_GFX906)
        const bool small_tile = false;  // gfx906: 64 KiB LDS, the 2560 tile as before (8 tokens slice into launches)
#else
        const bool small_tile = true;   // all eight tokens fit gfx1100's 64 KiB LDS at this tile
#endif
#else
        int cc_maj = 0, cc_min = 0;
        cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, dev);
        const bool small_tile = strata::cc_major_of(cc_maj) * 10 + strata::cc_minor_of(cc_min) == 75;
#endif
        const int tv = small_tile ? 1280 : 2560;
        tile[dev] = tv;
        int want = (int) (kFusedGrMaxT * tv * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        if (small_tile) {
            cudaFuncSetAttribute(gr_down_multi_kernel<1280>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        } else {
            cudaFuncSetAttribute(gr_down_multi_kernel<2560>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        }
        auto set_staged_attr = [&](auto kernel, int max_t) {
            int ws = (int) (2 * max_t * H_TILE * sizeof(float));
            if (optin > 0 && ws > optin) ws = optin;
            if (ws > 48 * 1024)
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ws);
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        };
        set_staged_attr(gr_down_staged_kernel<1, true>, 1);
        set_staged_attr(gr_down_staged_kernel<2, true>, 2);
        set_staged_attr(gr_down_staged_kernel<3, true>, 3);
        set_staged_attr(gr_down_staged_kernel<4, true>, 4);
        set_staged_attr(gr_down_staged_kernel<5, true>, 5);
        set_staged_attr(gr_down_staged_kernel<6, true>, 6);
        set_staged_attr(gr_down_staged_kernel<4, false>, 4);
        set_staged_attr(gr_down_staged_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        cudaGetLastError();      // drop any error the attempt left behind
        // The opt-in is a promise a pre-Volta card does not keep: an sm_60 answers 65536 and accepts the
        // cudaFuncSetAttribute for 61440 B, then fails the LAUNCH with "invalid argument".  What such a card
        // will launch is its per-block limit, so the capacity comes from that below sm_70 - the same tokens
        // the "no opt-in" branch assumes, but taken from the attribute that is actually enforced.
#if defined(__HIPCC__)
        const int usable = optin > 0 ? optin : 48 * 1024;
#else
        int cc = 0, per_block = 0;
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cc = strata::cc_major_of(cc);
        const int usable = (cc >= 7 && optin > 0) ? optin : per_block;
#endif
        const int capacity = usable / (int) (tv * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        const int capacity_staged = usable / (int) (2 * H_TILE * sizeof(float));
        chunk_staged[dev] = capacity_staged < 1 ? 1 : (capacity_staged > kFusedGrMaxT ? kFusedGrMaxT : capacity_staged);
        attr[dev] = true;
    }
    *tile_out = tile[dev] ? tile[dev] : 2560;
    return staged ? chunk_staged[dev] : chunk[dev];
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path, which fused_gr_read_multi
// takes first).
#if STRATA_GR_FAST_BUILD
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m);   // below, with the AMD fast path
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m);
}  // namespace
static bool gr_fast();
namespace {
#endif
// STRATA_GR_DOWN_MAX4=1 (opt-in): launches of up to 4 tokens hold 4 tokens' sums per thread instead of kFusedGrMaxT (the
// same bits; checked bitwise on the 5070 for 0.1.40). Default off: a 10-pair interleaved decode A/B on the RTX 5070 (Q2_0,
// default config) did not show the +2.8-3.5% of bench #832 (medians 0.97-0.99 of the off arm).
bool gr_down_max4() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GR_DOWN_MAX4");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

void launch_multi(const GrMulti& m, int variant, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
#if STRATA_GR_FAST_BUILD
    // gfx906: the latency-hidden norm/up (STRATA_GR_FAST=0: off) - the same sums in the same order as the kernels
    // they replace (gr_parity checks the multi-token read against single-token calls bitwise)
    const bool fast = gr_fast();
    if (variant >= kHcSplit) gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    else if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
    else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#else
    if (variant >= kHcSplit) gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#endif
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    const bool staged = variant >= kHcStaged;
    int tv = 2560;
    const int chunk_tok = down_chunk(staged, &tv);
    const size_t per_tok = (staged ? (size_t) 2 * H_TILE : (size_t) tv) * sizeof(float);
    static const bool no_multi_gr = [] {
        const char* no = std::getenv("STRATA_NO_MULTI_GR");
        return no != nullptr && no[0] != '\0' && no[0] != '0';
    }();
    const bool exact_t = !no_multi_gr;
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t) ct * per_tok;
        // #443, STRATA_GR_DOWN_MAX4: launches of up to 4 tokens hold 4 tokens' sums per thread instead of
        // kFusedGrMaxT - the same tile, block size, accumulation order and plain/split/staged path, so the same bits
        // (decided once per device: this runs per layer when decode is not captured)
        const bool max4 = ct <= 4 && gr_down_max4();
        if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
            // are compile-time (the same sums in the same order); STRATA_NO_MULTI_GR=1 keeps the generic kernels
            if (exact_t && ct == 1) gr_down_staged_kernel<1, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_staged_kernel<2, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_staged_kernel<3, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_staged_kernel<4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_staged_kernel<5, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_staged_kernel<6, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (max4) gr_down_staged_kernel<4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_staged_kernel<><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else if (tv == 1280) {
            if (max4) gr_down_multi_kernel<1280, 4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_multi_kernel<1280><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else {
            if (max4) gr_down_multi_kernel<2560, 4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_multi_kernel<2560><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
#if STRATA_GR_FAST_BUILD
    if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    else
#endif
    switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
        case 1: gr_up_multi_kernel<1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 2: gr_up_multi_kernel<2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 3: gr_up_multi_kernel<3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 4: gr_up_multi_kernel<4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 5: gr_up_multi_kernel<5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 6: gr_up_multi_kernel<6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        default: gr_up_multi_kernel<kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
    }
}

/// STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;
}

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];


#if STRATA_GR_FAST_BUILD
// ---- AMD fast path (STRATA_GR_FAST, default 1; 0 = the old kernels): the same arithmetic per value in the same order, latency
// hidden.  gr_norm: one block per token (as before) but a thread's 10 float4 of R / bo / w_norm are loaded
// before any is used, and xn stays in registers until rs is known (was: store, then re-read the 40 KB).
// gr_up: a warp's 8 rows of w_up and their epilogue inputs are loaded before the dots (was: one row's load chain
// after another).
constexpr int NQ = D / (THREADS * 4);   // 10 float4 per thread
static_assert(D % (THREADS * 4) == 0, "whole float4 per thread");
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float4 r[NQ], g[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        r[k] = *reinterpret_cast<const float4*>(a.R + i);
        g[k] = *reinterpret_cast<const float4*>(a.w_norm + i);
    }
    if (a.apply) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = t * 4 + k * THREADS * 4, c = i / N, d = i - c * N;
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r[k].x = fmaf(b.x, gw[c], r[k].x); r[k].y = fmaf(b.y, gw[c], r[k].y);
            r[k].z = fmaf(b.z, gw[c], r[k].z); r[k].w = fmaf(b.w, gw[c], r[k].w);
        }
    }
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4, c = i / N;
        const float sq = r[k].x * r[k].x + r[k].y * r[k].y + r[k].z * r[k].z + r[k].w * r[k].w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        r[k] = make_float4(r[k].x * g[k].x, r[k].y * g[k].y, r[k].z * g[k].z, r[k].w * g[k].w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const float sr = s_rs[i / N];
        *reinterpret_cast<float4*>(xn + i) = make_float4(r[k].x * sr, r[k].y * sr, r[k].z * sr, r[k].w * sr);
    }
}
// gr_up: 8 lanes per row instead of a 32-lane warp.  Lane j holds the old lanes j, j+8, j+16, j+24 (and chunk
// 32+j, the old lane j's second chunk): the xor tree's stages 16 and 8 are its own adds (a + b == b + a), stages
// 4, 2, 1 are shuffles inside the 8 - the same tree, bitwise the same sum, 3 shuffles a token instead of 5 and no
// idle lanes on the tail chunks.  A block's 64 rows are 2 passes of 32 groups, their weights loaded up front.
__device__ __forceinline__ float xor8(float v) {
#pragma unroll
#if defined(__HIPCC__)
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
#else
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
    return v;
}
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = j < T && m.a[j < T ? j : 0].apply;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = a.R[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float* l = lo[k];
            const float p0 = dot8(w[p][0], l + j * 8) + dot8(w[p][4], l + (32 + j) * 8);   // old lane j
            const float p1 = dot8(w[p][1], l + (j + 8) * 8);                               // old lane j + 8
            const float p2 = dot8(w[p][2], l + (j + 16) * 8);                              // old lane j + 16
            const float p3 = dot8(w[p][3], l + (j + 24) * 8);                              // old lane j + 24
            const float s = xor8((p0 + p2) + (p1 + p3));   // stage 16: (j, j+16), (j+8, j+24); stage 8; then 4, 2, 1
            if (j == k) mine = s;
        }
        if (j < T) {
            float x0 = rv[p];
            if (apply) {
                x0 = fmaf(bo[p], 2.0f * sigmoidf_(ip[p] / (float) HC), x0);
                m.a[j].R_out[i] = x0;
            }
            const float x = x0 * wn[p] * rsc[p];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE, as gr_up_multi_kernel
}
#endif

#if defined(STRATA_HIP_GFX906)
// ---- AMD: `gr_down_multi` split along K.  The CUDA kernel is one warp per row of w_down (320 + 4 rows), 41
// blocks: on a 60-CU gfx906 that left most of the card idle and read the 6.5 MB matrix at ~100 GB/s.  Here a block
// is 8 wavefronts = 8 rows of one K-slice of 2048 (the slice of every token's xn staged in LDS once), 41 x 5
// blocks, and a second kernel adds the 5 slice sums in a fixed order and runs the epilogue.  A token's result does
// not depend on how many tokens share the window (each column is reduced on its own).
constexpr int GS_SL = 2048;                   // K per slice
constexpr int GS_S = D / GS_SL;               // 5 slices
constexpr int GS_ROWS = LR + HC;              // 324: the down rows, then the inject rows
constexpr int GS_WAVES = 8;
static_assert(D % GS_SL == 0 && GS_SL == 64 * 8 * 4, "a lane takes 4 chunks of 8 per slice");
// the slice sums: a module-scope device array is per device by construction, and needs no allocation during a
// graph capture
__device__ float g_gr_part[GS_S * kFusedGrMaxT * GS_ROWS];
__global__ void __launch_bounds__(GS_WAVES * 64) gr_down_split_kernel(GrMulti m) {
    float* part = g_gr_part;
    extern __shared__ __align__(16) float tile[];   // [T][GS_SL]
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int T = m.T, sl = blockIdx.y;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? wave : blockIdx.x * GS_WAVES + wave;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || wave >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + sl * (GS_SL / 8);
    uint4 wv[4];
    if (active) {
#pragma unroll
        for (int q = 0; q < 4; ++q) wv[q] = __ldg(w4 + lane + 64 * q);
    }
    const float4* src4 = reinterpret_cast<const float4*>(m.xn);
    float4* tile4 = reinterpret_cast<float4*>(tile);
    for (int i = t; i < T * (GS_SL / 4); i += GS_WAVES * 64) {
        const int k = i / (GS_SL / 4), off = i - k * (GS_SL / 4);
        tile4[i] = src4[((size_t) k * D + (size_t) sl * GS_SL) / 4 + off];
    }
    __syncthreads();
    if (!active) return;
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int j = lane + 64 * q;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            if (k < T) acc[k] += dot8(wv[q], tile + k * GS_SL + j * 8);
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        float v = acc[k];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (lane == 0) part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + (inject_block ? LR + row : row)] = v;
    }
}
__global__ void gr_down_finish_kernel(GrMulti m) {
    const float* part = g_gr_part;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = i / GS_ROWS, r = i - k * GS_ROWS;
    if (k >= m.T) return;
    const bool inject = r >= LR;
    if (inject && m.a[0].w_inject == nullptr) return;
    float s = 0.0f;
#pragma unroll
    for (int sl = 0; sl < GS_S; ++sl) s += part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + r];
    if (inject) {
        m.a[k].inject_out[r - LR] = s;
    } else {
        const float x = s / (float) HC;
        m.a[k].lo[r] = x / (1.0f + __expf(-x));
    }
}

#endif

// ================================ S23 experiment (STRATA_HC_Q8=1): the read with the GGUF's Q8_0 projections ======
// The pack holds the hyper-connection projections as BF16 (iq_pack.py --compat-bf16 rounds the GGUF's Q8_0 values to
// BF16: 1.20 GiB per verify step); this read takes the Q8_0 bytes as stored (0.65 GiB - and the GGUF's own values,
// not a rounding of them).  The layout is the stream split of the v3 read above, at 4 chunks per stream: `down` runs
// on (10 groups of 32 rows + the inject rows) x 16 chunks of 640 columns = 176 blocks, each staging its chunk of
// R' * w_norm (unscaled) for the T tokens and writing the chunk's partial dots and sums of squares; `up` sums them
// per stream in a fixed order, applies rs, SwiGLU-free silu, and runs the default epilogue.  Another summation order
// and other weights than the default read: an output-changing step (measured, quality-checked).
constexpr int Q8B = 34;                             // Q8_0 block: fp16 d + 32 int8
constexpr int Q8_RPW = 4;                           // down rows per warp
constexpr int Q8_RG = LR / (WARPS * Q8_RPW);        // 10 row groups (+1: the inject rows)
constexpr int Q8_KC = 640, Q8_NKC = D / Q8_KC;      // 16 chunks, 4 per stream
constexpr int Q8_CPS = N / Q8_KC;                   // chunks per stream
constexpr int Q8_SPB = Q8_KC / 128;                 // 5 steps of 4 Q8_0 blocks (8 lanes x 4 values each)
__device__ __forceinline__ uint32_t q8_ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float q8_half(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }
__device__ __forceinline__ float4 q8_four(uint32_t q, float d) {
    return make_float4(d * (float) (int8_t) (q & 255u), d * (float) (int8_t) ((q >> 8) & 255u),
                       d * (float) (int8_t) ((q >> 16) & 255u), d * (float) (int8_t) (q >> 24));
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_down_q8_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    // S25: the inject rows are F32 in the GGUF with BF16-exact values, so by default they are read as the pack's BF16
    // (exact); a Q8_0 copy (STRATA_HC_Q8_INJECT=1) is the other option
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;       // lanes 8b..8b+7: one Q8_0 block, 4 values each
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows) break;
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
        }
    }
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_up_q8_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// S26 STRATA_TSUM=1: gr_down_q8_kernel / gr_up_q8_kernel with each row's T per-token warp sums (and the sums of
// squares) as one transposed butterfly (s26_tsum.cuh): bitwise the same values (S26 ws4 harness, every output equal,
// T 1-4: 1.00 / 1.03 / 1.03 / 1.15x per read). PF (all rows' weights in registers before the part / lo phase) was
// slower (0.92-1.03x) and is off.
using s26ts::tsum;
using s26ts::tsum_token;
using s26ts::tsum_lane;
using s26ts::pow2_ceil;
// gr_down_q8_kernel / gr_up_q8_kernel with every per-token warp sum done as one transposed butterfly per row (tsum)
template <int T, int RPW = Q8_RPW>
__global__ void __launch_bounds__(THREADS) gr_down_q8_fast_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RG_ = LR / (WARPS * RPW);
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == RG_;
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    uint32_t q[RPW][Q8_SPB];
    float dq[RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[P];
#pragma unroll
    for (int k = 0; k < P; ++k) ssp[k] = 0.0f;
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    {
        const float v = tsum<P>(ssp, lane);
        if (owner) red[warp][tk] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows) break;
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
        const float v = tsum<P>(acc, lane);
        if (owner) part[((size_t) tk * Q8_NKC + kc) * PR + prow] = v;
    }
}

template <int T, bool PF = false>
__global__ void __launch_bounds__(THREADS) gr_up_q8_fast_kernel(GrMulti m, const float* __restrict__ part,
                                                                const float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RPW8 = HC * UPM_COLS / WARPS;
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    // PF: every row's weights in registers before the part / lo phase (its latency hides the weight read)
    uint32_t pq[PF ? RPW8 : 1][3];
    float pd[PF ? RPW8 : 1][3];
    if constexpr (PF) {
#pragma unroll
        for (int qq = 0; qq < RPW8; ++qq) {
            const int r = warp + qq * WARPS;
            const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
            const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const uint8_t* blk = row + (v >> 5) * Q8B;
                    pq[qq][s] = q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16;
                    pd[qq][s] = q8_half(q8_ld16(blk));
                } else {
                    pq[qq][s] = 0u; pd[qq][s] = 0.0f;
                }
            }
        }
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
#pragma unroll(PF ? RPW8 : 2)
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int v = 4 * lane + 128 * s;
            if (PF) {
                w[s] = v < LR ? q8_four(pq[PF ? qq : 0][s], pd[PF ? qq : 0][s]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (owner) {
            const FusedGrArgs& a = m.a[tk];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc[k] = fmaf(w[s].x, x.x, acc[k]); acc[k] = fmaf(w[s].y, x.y, acc[k]);
                    acc[k] = fmaf(w[s].z, x.z, acc[k]); acc[k] = fmaf(w[s].w, x.w, acc[k]);
                }
            }
        }
        const float mine = tsum<P>(acc, lane);
        if (owner) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[tk].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[tk][c];
            g[tk][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

template <int T> void launch_q8_t(const GrMulti& m, float* part, float* ssg, cudaStream_t st, unsigned long long* stamp_buf,
                                  int stamp_i0) {
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    if (ts) gr_down_q8_fast_kernel<T, 4><<<dim3(LR / (WARPS * 4) + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    else gr_down_q8_kernel<T><<<dim3(Q8_RG + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
    if (ts) gr_up_q8_fast_kernel<T, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
    else gr_up_q8_kernel<T><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
}
void launch_q8(const GrMulti& m, float* scratch, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    float* part = scratch;
    float* ssg = scratch + (size_t) m.T * Q8_NKC * PR;
    switch (m.T) {
        case 1: launch_q8_t<1>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 2: launch_q8_t<2>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 3: launch_q8_t<3>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 4: launch_q8_t<4>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 5: launch_q8_t<5>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 6: launch_q8_t<6>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 7: launch_q8_t<7>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        default: launch_q8_t<8>(m, part, ssg, st, stamp_buf, stamp_i0); break;
    }
}

}  // namespace

#if STRATA_GR_FAST_BUILD
int g_gr_fast = -1;   // fused_gr_set_fast (the bench); -1 = STRATA_GR_FAST, else the card's default
static bool gr_fast() {
    static const int env = [] { const char* v = std::getenv("STRATA_GR_FAST"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (g_gr_fast >= 0) return g_gr_fast != 0;
    if (env >= 0) return env != 0;
#if defined(__HIPCC__)
    return true;
#else
    return cur_dev_volta();   // CUDA: on Volta (V100-SXM2: 50.9 -> 46.7 us a read at T 1, bitwise); elsewhere opt-in
#endif
}
#else
int g_gr_fast = -1;
#endif
void fused_gr_set_fast(int on) { g_gr_fast = on; }

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm || x.q8_down != a[0].q8_down ||
            x.q8_up != a[0].q8_up || x.q8_inject != a[0].q8_inject) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    // S26 STRATA_QFUSE: every token needs its q8_1 destination and token 0 the counters; otherwise none is written
    bool q8 = a[0].q8_cnt != nullptr;
    for (int t = 0; t < n_tok; ++t) q8 = q8 && a[t].q8_mixed != nullptr;
    if (!q8)
        for (int t = 0; t < n_tok; ++t) m.a[t].q8_mixed = nullptr;
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    if (a[0].q8_down != nullptr && a[0].q8_up != nullptr) {   // inject: Q8_0 copy, or the BF16 rows
        launch_q8(m, xn_scratch, st, stamp_buf, stamp_i0);   // S23 experiment: STRATA_HC_Q8=1
        const cudaError_t eq = cudaGetLastError();
        if (eq != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi q8: %s\n", cudaGetErrorString(eq));
            std::exit(1);
        }
        return q8;
    }
    // STRATA_GR_V3=1: the two-kernel read above (another summation order - opt-in)
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    static int split3[64] = {};   // per device: 0 = not decided yet, 1 / 2 = column halves S, -1 = does not fit
    int dev3 = 0;
    if (v3) {
        cudaGetDevice(&dev3);
        if (dev3 >= 0 && dev3 < 64 && split3[dev3] == 0) {
            int optin = 0;
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev3);
            const int limit = optin > 0 ? optin : 48 * 1024;
            const int need1 = (int) (kFusedGrMaxT * N * sizeof(float)), need6 = (int) (6 * N * sizeof(float)),
                      need5 = (int) (5 * N * sizeof(float)), need2 = need1 / 2;
            int split = -1;
            if (need1 <= limit &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, kFusedGrMaxT, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, need1) == cudaSuccess &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 6, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, need6) == cudaSuccess &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 5, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, need5) == cudaSuccess) {
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 1, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 2, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 3, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 5, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 6, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, kFusedGrMaxT, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                split = 1;
            } else if (need2 <= limit)
                // #375 (kenh0u): the S = 2 split (a 64 KB opt-in card: Turing) disagrees with itself in gr_parity
                // (graph replay vs direct call) - such a card keeps the default read until that split is fixed
                std::fprintf(stderr, "strata: STRATA_GR_V3=1 needs the two-half split on this card, which fails its "
                                     "checks (#375): the default read is used\n");
            cudaGetLastError();   // drop any error the attempts left behind
            split3[dev3] = split;
        }
    }
    const int split = v3 && dev3 >= 0 && dev3 < 64 ? split3[dev3] : -1;
    if (split > 0) {   // 2 kernels; scratch = partials + sums of squares
        float* part = xn_scratch;
        float* ssg = xn_scratch + (size_t) n_tok * HC * 2 * PR;   // room for S = 2
        const size_t sm = (size_t) n_tok * (N / split) * sizeof(float);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        if (split == 2) gr_down_v3_kernel<2><<<dim3(LR / WARPS + 1, HC * 2), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 1) gr_down_v3_kernel<1, 1, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 2) gr_down_v3_kernel<1, 2, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 3) gr_down_v3_kernel<1, 3, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 4) gr_down_v3_kernel<1, 4, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 5) gr_down_v3_kernel<1, 5, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 6) gr_down_v3_kernel<1, 6, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else gr_down_v3_kernel<1, kFusedGrMaxT, false><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (split == 2) gr_up_v3_kernel<2><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 1) gr_up_v3_kernel<1, 1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 2) gr_up_v3_kernel<1, 2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 3) gr_up_v3_kernel<1, 3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 4) gr_up_v3_kernel<1, 4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 5) gr_up_v3_kernel<1, 5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 6) gr_up_v3_kernel<1, 6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else gr_up_v3_kernel<1, kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        const cudaError_t e3 = cudaGetLastError();
        if (e3 != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi v3: %s\n", cudaGetErrorString(e3));
            std::exit(1);
        }
        return false;   // the v3 read: no q8_1 (the caller quantizes)
    }
#if defined(STRATA_HIP_GFX906)
    // gfx906, opt-in STRATA_GR_SPLIT=1: the latency-hidden norm/up (STRATA_GR_FAST) with the K-split down (~3% faster
    // per verify window).  Its fixed-order finish sums the K slices in another order than the single-token
    // fused_gr_read, so the multi-token read is then no longer bitwise equal to T single-token calls (gr_parity's
    // contract); off by default for that reason.
    static const bool split_on = std::getenv("STRATA_GR_SPLIT") && std::string(std::getenv("STRATA_GR_SPLIT")) == "1";
    if (split_on) {
        const bool fast = gr_fast();
        if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
        else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        gr_down_split_kernel<<<dim3(DOWN_BLOCKS + 1, GS_S), GS_WAVES * 64, (size_t) n_tok * GS_SL * sizeof(float), st>>>(
            m);
        gr_down_finish_kernel<<<(unsigned) ((n_tok * GS_ROWS + 255) / 256), 256, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        else gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return false;   // the gfx906 STRATA_GR_SPLIT read: no q8_1 (the caller quantizes), like the v3 path above
    }
#endif
    // the default read (STRATA_GR_V3 unset): v1, or the bitwise-equal v2 / v3 this card's check accepted
    launch_multi(m, fused_gr_variant(), st, stamp_buf, stamp_i0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return q8;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}


namespace {

/// The plain read, split, staged and (for one token) the single-token read on random bf16 weights and random inputs,
/// 1..8 tokens, with and without the pending write; every output of split and staged compared with the plain read's
/// bit for bit (and the plain read's with the single-token read's).  `why[v]` gets the first difference of variant
/// v; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[4], std::string why[4]) {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 4;                             // sets: 0 = plain, 1 = split, 2 = staged, 3 = single-token
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t) LR * D), h_up((size_t) D * LR), h_inj((size_t) HC * D);
    std::vector<float> h_norm(D), h_R((size_t) TM * D), h_bo((size_t) TM * N), h_ip((size_t) TM * HC);
    for (auto& w : h_down) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up) w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm) w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R) x = nd(rng);
    for (auto& x : h_bo) x = 0.5f * nd(rng);
    for (auto& x : h_ip) x = 2.0f * nd(rng);
    for (int v = 0; v < 4; ++v) { ok_variant[v] = v == 1; why[v].clear(); }

    const size_t n_set = (size_t) TM * (D + D + LR + HC + HC + N);   // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = nullptr;
    if (cudaMalloc((void**) &base, bytes) != cudaSuccess) {
        cudaGetLastError();
        why[0] = "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) { void* p = base + off; off += (b + 255) / 256 * 256; return p; };
    uint16_t* d_down = (uint16_t*) take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*) take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*) take(h_inj.size() * 2);
    float* d_norm = (float*) take(h_norm.size() * 4);
    float* d_R = (float*) take(h_R.size() * 4);
    float* d_bo = (float*) take(h_bo.size() * 4);
    float* d_ip = (float*) take(h_ip.size() * 4);
    struct Set { float *R_out, *xn, *lo, *rs, *inj, *mixed; } set[NV];
    for (Set& x : set) {
        x.R_out = (float*) take((size_t) TM * D * 4); x.xn = (float*) take((size_t) TM * D * 4);
        x.lo = (float*) take((size_t) TM * LR * 4); x.rs = (float*) take((size_t) TM * HC * 4);
        x.inj = (float*) take((size_t) TM * HC * 4); x.mixed = (float*) take((size_t) TM * N * 4);
    }
    cudaStream_t st = nullptr;
    bool ok = off <= bytes && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
              cudaMemcpyAsync(d_down, h_down.data(), h_down.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_up, h_up.data(), h_up.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_inj, h_inj.data(), h_inj.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_norm, h_norm.data(), h_norm.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_R, h_R.data(), h_R.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_bo, h_bo.data(), h_bo.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_ip, h_ip.data(), h_ip.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w` (or a read-back failure in `ok`)
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, std::string& w) {
        h1.resize(n);
        h2.resize(n);
        if (cudaMemcpyAsync(h1.data(), d1, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaMemcpyAsync(h2.data(), d2, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess) {
            why[0] = "reading back the check failed";
            ok = false;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", i, u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, std::string& w) {
        return same(s1.lo, s2.lo, (size_t) T * LR, "lo", T, apply, w) &&
               same(s1.rs, s2.rs, (size_t) T * HC, "rs", T, apply, w) &&
               same(s1.inj, s2.inj, (size_t) T * HC, "inject", T, apply, w) &&
               same(s1.mixed, s2.mixed, (size_t) T * N, "mixed", T, apply, w) &&
               same(s1.R_out, s2.R_out, (size_t) T * D, "R", T, apply, w);
    };
    ok_variant[2] = ok_variant[3] = ok;
    bool single_ok = ok;
    for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t) ((uint8_t*) (set[v].mixed + (size_t) TM * N) - (uint8_t*) set[v].R_out);
                ok = ok && cudaMemsetAsync(set[v].R_out, 0xFF, span, st) == cudaSuccess;   // the whole set
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t) k * D; x.R_out = set[v].R_out + (size_t) k * D; x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t) k * N; x.inj_prev = d_ip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj; x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t) k * LR; x.rs = set[v].rs + (size_t) k * HC;
                    x.inject_out = set[v].inj + (size_t) k * HC; x.mixed = set[v].mixed + (size_t) k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k) m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess) {
                why[0] = "a kernel of the check failed";
                ok = false;
                break;
            }
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, why[3])) ok_variant[3] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, why[1])) single_ok = false;
        }
    }
    if (!single_ok && ok) {                            // the plain read itself disagrees with the single-token read
        why[2] = why[3] = "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = false;
    }
    if (st != nullptr) {
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    cudaFree(base);
    cudaGetLastError();
    if (!ok) ok_variant[2] = ok_variant[3] = false;
    return ok;
}

}  // namespace

int fused_gr_variant() {
    int dev = 0;
    cudaGetDevice(&dev);
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    const char* e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2') ? env_variant() : kHcPlain;
}

void fused_gr_check() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    const int want = env_variant();
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    bool okv[4];
    std::string why[4];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    if (want >= kHcStaged && okv[kHcStaged]) use = kHcStaged;
    else if (okv[kHcSplit]) use = kHcSplit;
    g_variant[dev].store(use);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[4] = {"", "plain", "split", "staged"};
    for (int v = kHcStaged; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran)
            std::fprintf(stderr, "strata hc: CUDA%d: the %s read differs from the plain read on this card - not used: "
                                 "%s\n", dev, name[v], why[v].c_str());
    static const char* const what[4] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain read's down projection)",
        "staged (the norm per token and stream, the down projection's activations staged ahead by cp.async)"};
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)" : "");
}


}  // namespace strata::kernels
