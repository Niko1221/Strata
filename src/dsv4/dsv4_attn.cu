// src/dsv4/dsv4_attn.cu - DeepSeek V4-Flash dense MLA attention block, GPU kernels (dsv4-mvp S3).
//
// Semantics follow src/dsv4/dsv4ref.cpp attention_forward() line by line for the hash layers
// (compress ratio 0, plain rope theta=1e4):
//   - Q8_0 GEMM: out[t][r] = dot(x[t], dequant(W_row r)); W row = (cols/32) blocks of
//     34 B (fp16 scale + 32 int8).  wo_a is grouped: row r consumes x[t] + (r/1024)*4096.
//   - rmsnorm rows with double-precision sum-of-squares (matches the host reference), optional
//     gamma; the q per-head renorm is gamma-free.
//   - rope on the last 2*half dims as interleaved pairs (re, im) = (x[2i], x[2i+1]); the
//     attention OUTPUT gets the inverse rotation (trap: output-side inverse rotary).
//   - MQA-512 attention: 64 query heads share one 512-dim latent kv row per token (K == V).
//     scores over a topk index list (-1 = masked), softmax with per-head attn_sink added to the
//     DENOMINATOR only, scale = 512^-0.5, no mscale.  head_dim 512 exceeds native_flash_attn's
//     256 limit, hence this kernel.
//
// Kernel shapes (correctness-first; 13-token golden prefill + single-token decode):
//   gemm:   one block per output row, 128 threads, token loop, warp+shared reduction.
//   rmsnorm/rope: one block per row / one thread per (row, pair).
//   mqa:    one block per (token, head); 4 warps split the topk list for scores, all threads
//           cooperate on the weighted-value accumulation.

#include "strata/dsv4/attn.hpp"

#include "dsv4_attn_time.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::dsv4 {
namespace {

#define DSV4_CK(x)                                                            \
    do {                                                                      \
        cudaError_t err_ = (x);                                               \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "dsv4_attn: CUDA error %s at %s:%d\n",       \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// [S7d] cp.async (LDGSTS) 16-byte staging helpers: the weight row goes global->shared without a
// register round-trip, and every thread can keep many loads in flight (the previous register-
// batched form stopped at 4).  Byte content, per-thread accumulation order and the block
// reduction are untouched -> bit-identical output.
__device__ __forceinline__ void dsv4_cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void dsv4_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void dsv4_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

constexpr int kNEmbd = 4096;
constexpr int kQRank = 1024;
constexpr int kNHead = 64;
constexpr int kHeadDim = 512;
constexpr int kORows = 8192;   // 8 groups x 1024
constexpr int kOGroup = 1024;  // wo_a rows per group
constexpr double kRmsEps = 1e-6;

// ---------------------------------------------------------------------------
// Q8_0 GEMM: out[t][r] = dot(x[t] (+ group offset), dequant(W_row r)).
// R output rows per block; thread strides the row's 34-byte blocks; deterministic
// per-token block reduction (no atomics).
// ---------------------------------------------------------------------------
// [S7b] Weight rows are staged in shared memory (one cooperative uint4 pass instead of 32 byte
// LDGs per thread per token) and reused across the m-token loop.
// [S7d] R output rows per block: the x float4s a thread reads depend on its block index only, so
// they are loaded once and applied to R rows' codes (x traffic and block count drop by R; the
// per-thread q8 code loads stay the same). Each row keeps its own accumulator, its own
// `part += acc*d` order per block, its own 5-step warp reduction and the same ascending
// thread-0 warp sum as the R=1 kernel -> bit-identical output.
template <int R>
__global__ void dsv4_gemm_q8_tiled_kernel(const uint8_t* __restrict__ w, int rows, int cols,
                                          int group_rows, const float* __restrict__ x, int m,
                                          int x_row_stride, float* __restrict__ out,
                                          int out_xpose) {
    extern __shared__ uint8_t sw[];  // R*nb*34 + 32 bytes, 16B aligned by the runtime
    const int r0 = blockIdx.x * R;
    if (r0 >= rows) return;
    const int nr = min(R, rows - r0);
    const int nb = cols / 32;
    const int total = nb * 34;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    __shared__ float red[32 * R];
    const uint8_t* wbase = w + (size_t) r0 * (size_t) total;
    // All R rows of a block share one x slice (guaranteed by the launcher's R divisibility rule).
    const size_t g_off = group_rows > 0 ? (size_t) (r0 / group_rows) * (size_t) cols : 0;
    const uint8_t* astart = (const uint8_t*) ((uintptr_t) wbase & ~(uintptr_t) 15);
    const int aoff = (int) (wbase - astart);
    const int len = nr * total;
    const int nfull = (aoff + len) >> 4;
    {
        // [S7d] cooperative cp.async stage (all 16B loads issued back to back, no register use).
        const uint4* srcv = reinterpret_cast<const uint4*>(astart);
        uint4* dstv = reinterpret_cast<uint4*>(sw);
        for (int v = threadIdx.x; v < nfull; v += blockDim.x) dsv4_cp_async16(&dstv[v], &srcv[v]);
        dsv4_cp_async_commit();
        // trailing bytes that do not fill a 16B stride (aoff + len may not be 16B aligned)
        for (int i = (nfull << 4) + threadIdx.x; i < aoff + len; i += blockDim.x)
            sw[i] = astart[i];
        dsv4_cp_async_wait_all();
    }
    __syncthreads();
    const uint8_t* swrow = sw + aoff;
    for (int t = 0; t < m; ++t) {
        const float* xt = x + (size_t) t * (size_t) x_row_stride + g_off;
        float part[R];
#pragma unroll
        for (int i = 0; i < R; ++i) part[i] = 0.f;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) {
            // per-row rolling q8 unpack state for this 34-byte block
            const uint32_t* qw[R];
            int qsh[R];
            uint32_t q0[R];
            float d[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if (i < nr) {
                    const uint8_t* p = swrow + (size_t) i * (size_t) total + (size_t) b * 34;
                    unsigned short bits;
                    memcpy(&bits, p, 2);
                    d[i] = __half2float(__ushort_as_half(bits));
                    const uint8_t* qs = p + 2;
                    qw[i] = reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<uintptr_t>(qs) & ~(uintptr_t) 3);
                    qsh[i] = (int) (reinterpret_cast<uintptr_t>(qs) & 3) * 8;
                    q0[i] = qw[i][0];
                }
            }
            // [S7c] x arrives in the warp-slice-transposed layout (dsv4_xpose1024_kernel): block
            // b's float4 j lives at (b>>5)*1024 + j*128 + (b&31)*4, so this LDG.128 is fully
            // coalesced across the warp. Same values reach the same FMAs in the same order.
            const float* xb = xt + ((size_t) (b >> 5) << 10) + ((b & 31) << 2);
            const float4* x4 = reinterpret_cast<const float4*>(xb);
            float acc[R];
#pragma unroll
            for (int i = 0; i < R; ++i) acc[i] = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 xx = x4[j << 5];  // row-invariant: one LDG.128 feeds all R rows
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    if (i < nr) {
                        // [S7b v2] unpack the 32 int8 codes from 4B-aligned shared words with
                        // funnel shifts (9 LDS.32 instead of 32 LDS.U8); byte sequence and FMA
                        // order unchanged.
                        const uint32_t qw1 = qw[i][j + 1];
                        const uint32_t f = __funnelshift_r(q0[i], qw1, qsh[i]);
                        acc[i] += static_cast<float>(static_cast<int8_t>(f & 0xffu)) * xx.x;
                        acc[i] += static_cast<float>(static_cast<int8_t>((f >> 8) & 0xffu)) * xx.y;
                        acc[i] += static_cast<float>(static_cast<int8_t>((f >> 16) & 0xffu)) * xx.z;
                        acc[i] += static_cast<float>(static_cast<int8_t>(f >> 24)) * xx.w;
                        q0[i] = qw1;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if (i < nr) part[i] += acc[i] * d[i];
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float p = part[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
            if (lane == 0 && i < nr) red[i * 32 + wid] = p;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < nr; ++i) {
                float v = 0.f;
                for (int wi = 0; wi < nwarp; ++wi) v += red[i * 32 + wi];
                const int r = r0 + i;
                size_t oi = (size_t) t * rows + r;
                if (out_xpose) {
                    // [S7c] write r at its transposed position (wo_a -> wo_b link; the consumer's
                    // transposed reads then see the plain-layout values).
                    const int q = r & 1023;
                    oi = (size_t) t * rows + ((size_t) (r >> 10) << 10) + (((q >> 2) & 7) << 7) +
                         ((q >> 5) << 2) + (q & 3);
                }
                out[oi] = v;
            }
        }
        __syncthreads();
    }
}

// [S7d] Pre-S7d wide kernel, kept verbatim as the reference arm of the in-model bit-level A/B
// (STRATA_DSV4_Q8BITCMP=1: both arms run on the same weights/x in the same process and their
// outputs are compared word for word).  Never used in the normal path.
// One block per row; thread strides the row's 34-byte blocks; deterministic
// per-token block reduction (no atomics).
// ---------------------------------------------------------------------------
// [S7b] Weight row staged in shared memory (one cooperative uint4 pass instead of 32 byte
// LDGs per thread per token) and reused across the m-token loop. Byte values read, per-thread
// accumulation order, and the block reduction are unchanged -> bit-identical output.
__global__ void dsv4_gemm_q8_ref_kernel(const uint8_t* __restrict__ w, int rows, int cols,
                                    int group_rows, const float* __restrict__ x, int m,
                                    int x_row_stride, float* __restrict__ out, int out_xpose) {
    extern __shared__ uint8_t sw[];  // nb*34 + 32 bytes, 16B aligned by the runtime
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int nb = cols / 32;
    const uint8_t* wrow = w + (size_t)r * (size_t)nb * 34;
    const size_t g_off = group_rows > 0 ? (size_t)(r / group_rows) * (size_t)cols : 0;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    __shared__ float red[32];
    const int total = nb * 34;
    const uint8_t* astart = (const uint8_t*) ((uintptr_t) wrow & ~(uintptr_t) 15);
    const int aoff = (int) (wrow - astart);
    const int nfull = (aoff + total) >> 4;
    {
        // [S7b] register-batched cooperative stage: 4 independent uint4 loads in flight per
        // thread (memory-level parallelism; byte content identical).
        const uint4* srcv = reinterpret_cast<const uint4*>(astart);
        uint4* dstv = reinterpret_cast<uint4*>(sw);
        int v = threadIdx.x;
        for (; v + 3 * (int) blockDim.x < nfull; v += 4 * blockDim.x) {
            uint4 b0 = srcv[v], b1 = srcv[v + blockDim.x], b2 = srcv[v + 2 * blockDim.x],
                  b3 = srcv[v + 3 * blockDim.x];
            dstv[v] = b0;
            dstv[v + blockDim.x] = b1;
            dstv[v + 2 * blockDim.x] = b2;
            dstv[v + 3 * blockDim.x] = b3;
        }
        for (; v < nfull; v += blockDim.x) dstv[v] = srcv[v];
    }
    for (int i = (nfull << 4) + threadIdx.x; i < aoff + total; i += blockDim.x) sw[i] = astart[i];
    __syncthreads();
    const uint8_t* swrow = sw + aoff;
    for (int t = 0; t < m; ++t) {
        const float* xt = x + (size_t)t * (size_t)x_row_stride + g_off;
        float part = 0.f;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) {
            const uint8_t* blk = swrow + (size_t)b * 34;
            unsigned short bits;
            memcpy(&bits, blk, 2);
            const float d = __half2float(__ushort_as_half(bits));
            const uint8_t* qs = blk + 2;
            // [S7c] x arrives in the warp-slice-transposed layout (dsv4_xpose1024_kernel): block
            // b's float4 j lives at (b>>5)*1024 + j*128 + (b&31)*4, so this LDG.128 is fully
            // coalesced across the warp (was a 128B-strided scatter = 32 L1 wavefronts/instr).
            // Same values reach the same FMAs in the same order -> bit-identical.
            const float* xb = xt + ((size_t) (b >> 5) << 10) + ((b & 31) << 2);
            float acc = 0.f;
            // [S7b v2] unpack the 32 int8 codes from 4B-aligned shared words with funnel
            // shifts (9 LDS.32 instead of 32 LDS.U8); byte sequence and FMA order unchanged.
            const uint32_t* qw = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<uintptr_t>(qs) & ~(uintptr_t) 3);
            const int qsh = (int) (reinterpret_cast<uintptr_t>(qs) & 3) * 8;
            uint32_t qw0 = qw[0];
            const float4* x4 = reinterpret_cast<const float4*>(xb);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 xx = x4[j << 5];
                const uint32_t qw1 = qw[j + 1];
                const uint32_t f = __funnelshift_r(qw0, qw1, qsh);
                acc += static_cast<float>(static_cast<int8_t>(f & 0xffu)) * xx.x;
                acc += static_cast<float>(static_cast<int8_t>((f >> 8) & 0xffu)) * xx.y;
                acc += static_cast<float>(static_cast<int8_t>((f >> 16) & 0xffu)) * xx.z;
                acc += static_cast<float>(static_cast<int8_t>(f >> 24)) * xx.w;
                qw0 = qw1;
            }
            part += acc * d;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        if (lane == 0) red[wid] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = 0.f;
            for (int i = 0; i < nwarp; ++i) v += red[i];
            size_t oi = (size_t) t * rows + r;
            if (out_xpose) {
                // [S7c] write r at its transposed position (wo_a -> wo_b link; the consumer's
                // transposed reads then see the plain-layout values).
                const int q = r & 1023;
                oi = (size_t) t * rows + ((size_t) (r >> 10) << 10) + (((q >> 2) & 7) << 7) +
                     ((q >> 5) << 2) + (q & 3);
            }
            out[oi] = v;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// RMS norm rows (double sum-of-squares like dsv4ref); w == nullptr -> plain renorm.
// ---------------------------------------------------------------------------
__global__ void dsv4_rmsnorm_kernel(float* x, int rows, int cols, const float* __restrict__ w,
                                    double eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + (size_t)r * cols;
    double sq = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const double v = static_cast<double>(row[i]);
        sq += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, o);
    __shared__ double red[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        double v = 0.0;
        const int nwarp = blockDim.x >> 5;
        for (int i = 0; i < nwarp; ++i) v += red[i];
        red[0] = 1.0 / std::sqrt(v / cols + eps);
    }
    __syncthreads();
    const float inv = static_cast<float>(red[0]);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float g = w ? w[i] : 1.0f;
        row[i] = g * (row[i] * inv);
    }
}

// ---------------------------------------------------------------------------
// Rope on the last 2*half dims of each row, interleaved pairs.  One thread per
// (row, pair); position = start_pos + row / rows_per_token.
// ---------------------------------------------------------------------------
__global__ void dsv4_rope_kernel(float* x, int rows, int width, int half, int rows_per_token,
                                 int pos_mul, const float* __restrict__ cosv,
                                 const float* __restrict__ sinv, int64_t start_pos,
                                 int inverse) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * half) return;
    const int r = idx / half;
    const int i = idx % half;
    const int64_t pos = start_pos + (int64_t) (r / rows_per_token) * pos_mul;
    const float c = cosv[(size_t)pos * half + i];
    const float s = sinv[(size_t)pos * half + i];
    float* xr = x + (size_t)r * width + (width - 2 * half);
    const float re = xr[2 * i], im = xr[2 * i + 1];
    if (inverse) {
        xr[2 * i] = re * c + im * s;
        xr[2 * i + 1] = -re * s + im * c;
    } else {
        xr[2 * i] = re * c - im * s;
        xr[2 * i + 1] = re * s + im * c;
    }
}

// ---------------------------------------------------------------------------
// [S7e] rmsnorm(row) + rope(row) in one block: the rope pass runs on the row the norm has just
// finished (one __syncthreads in between), so this is exactly the pair of kernels it replaces, one
// launch cheaper per attention layer.  width == cols: the rope acts inside the row, at
// (cols - 2*half).
// ---------------------------------------------------------------------------
__global__ void dsv4_rmsnorm_rope_kernel(float* x, int rows, int cols, int half, int rows_per_token,
                                         int pos_mul, const float* __restrict__ cosv,
                                         const float* __restrict__ sinv, int64_t start_pos,
                                         int inverse, const float* __restrict__ w, double eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* const row = x + (size_t) r * cols;
    double sq = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const double v = static_cast<double>(row[i]);
        sq += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, o);
    __shared__ double red[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        double v = 0.0;
        const int nwarp = blockDim.x >> 5;
        for (int i = 0; i < nwarp; ++i) v += red[i];
        red[0] = 1.0 / std::sqrt(v / cols + eps);
    }
    __syncthreads();
    const float inv = static_cast<float>(red[0]);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float g = w ? w[i] : 1.0f;
        row[i] = g * (row[i] * inv);
    }
    __syncthreads();   // the rope pair may straddle two threads' lanes
    const int64_t pos = start_pos + (int64_t) (r / rows_per_token) * pos_mul;
    float* const xr = row + (cols - 2 * half);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = cosv[(size_t) pos * half + i];
        const float sx = sinv[(size_t) pos * half + i];
        const float re = xr[2 * i], im = xr[2 * i + 1];
        // Pinned to the rounding dsv4_rope_kernel compiles to (FMUL, FMUL, then FADD / FFMA): plain
        // C here lets the compiler contract the real part into one FFMA, which rounds differently and
        // shows up as a 1-ulp difference against the two-kernel reference.
        const float rc = __fmul_rn(re, c), ims = __fmul_rn(im, sx), cim = __fmul_rn(c, im);
        if (inverse) {
            xr[2 * i] = __fadd_rn(rc, ims);
            xr[2 * i + 1] = __fmaf_rn(c, im, -__fmul_rn(re, sx));
        } else {
            xr[2 * i] = __fadd_rn(rc, -ims);
            xr[2 * i + 1] = __fmaf_rn(re, sx, cim);
        }
    }
}

// ---------------------------------------------------------------------------
// Dense MQA-512 attention over a topk list.  One block per (token j, head h).
// scores[k] = dot(q[j][h], kv[topk[j][k]]) * scale (topk = -1 -> -inf);
// m = max valid score; denom = exp(sink[h] - m) + sum_k exp(score_k - m);
// out = sum_k e_k * kv[topk[j][k]] / denom.  The sink never enters the max or
// the numerator (denominator-only trap).
// ---------------------------------------------------------------------------
// [R8] slots is the number of rows readable from kv for this launch (window rows, prefill rows, or
// window + compressed rows).  A table entry outside [0, slots) is treated exactly like -1 (masked), so a
// stale or corrupted index can never read past the end of the KV arena.
// [M2] source of one table entry.  kRing == 0: the table indexes `kv` directly (prefill rows in kv_new
// or in the staged kv_all).  kRing == 1: the table is the pass-relative ring list of
// dsv4_window_topk's start_pos > 0 branch, so a window slot (< win) is this pass's kv_new row when that
// row sits at or before the attending row's position, and an older ring slot otherwise; the compressed
// slots (>= win) live behind the window and always read the ring.  Only the pointer differs - the
// accumulate order inside the score and value loops is untouched.
template <int kRing>
__device__ __forceinline__ const float* dsv4_mqa512_src(const float* kv, const float* kv_new,
                                                        int32_t tidx, int j, int hd, int win,
                                                        int sp0) {
    if (kRing != 0 && tidx < win) {
        int d = tidx - sp0;
        if (d < 0) d += win;
        if (d <= j) return kv_new + (size_t) d * hd;  // row d of this pass; d <= j, never a future row
    }
    return kv + (size_t) tidx * hd;
}

template <int kRing>
__global__ void dsv4_mqa512_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                   const float* __restrict__ kv_new,
                                   const int32_t* __restrict__ topk,
                                   const float* __restrict__ sinks, int nh, int hd, int K, int slots,
                                   int win, int sp0, float scale, float* __restrict__ out) {
    extern __shared__ float smem[];  // q_sh[hd] then scores[K]
    float* q_sh = smem;
    float* scores = smem + hd;
    __shared__ float red[32];
    __shared__ float s_m, s_denom;
    const int j = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int nwarp = blockDim.x >> 5;
    const float* qh = q + ((size_t)j * nh + h) * hd;
    for (int i = tid; i < hd; i += blockDim.x) q_sh[i] = qh[i];
    __syncthreads();
    // scores: warps split the topk list; one warp per entry, lanes split the head dim.
    for (int k = wid; k < K; k += nwarp) {
        const int32_t tidx = topk[(size_t)j * K + k];
        float sc = -INFINITY;
        if (tidx >= 0 && (unsigned) tidx < (unsigned) slots) {   // [R8] upper bound, not just >= 0
            const float* kvr = dsv4_mqa512_src<kRing>(kv, kv_new, tidx, j, hd, win, sp0);
            float acc = 0.f;
            for (int i = lane; i < hd; i += 32) acc += q_sh[i] * kvr[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            if (lane == 0) sc = acc * scale;
        }
        if (lane == 0) scores[k] = sc;
    }
    __syncthreads();
    // max over the list (-inf entries lose naturally)
    float m = -INFINITY;
    for (int k = tid; k < K; k += blockDim.x) m = fmaxf(m, scores[k]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    if (tid == 0) {
        float v = red[0];
        for (int i = 1; i < nwarp; ++i) v = fmaxf(v, red[i]);
        s_m = v;
    }
    __syncthreads();
    m = s_m;
    // e_k in place + denom partials
    float dpart = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        const float e = expf(scores[k] - m);  // expf(-inf) = 0 for masked entries
        scores[k] = e;
        dpart += e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) dpart += __shfl_down_sync(0xffffffffu, dpart, o);
    if (lane == 0) red[wid] = dpart;
    __syncthreads();
    if (tid == 0) {
        float v = expf(sinks[h] - m);  // sink: denominator only
        for (int i = 0; i < nwarp; ++i) v += red[i];
        s_denom = v;
    }
    __syncthreads();
    const float inv_denom = 1.0f / s_denom;
    // weighted value accumulation; thread strides the head dim
    float* oh = out + ((size_t)j * nh + h) * hd;
    for (int i = tid; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            const float e = scores[k];
            if (e > 0.f) {
                const int32_t tidx = topk[(size_t)j * K + k];
                if ((unsigned) tidx < (unsigned) slots)   // [R8] same bound as the score pass
                    acc += e * dsv4_mqa512_src<kRing>(kv, kv_new, tidx, j, hd, win, sp0)[i];
            }
        }
        oh[i] = acc * inv_denom;
    }
}

// [S7b] narrow rows (nb <= 32): one warp per row, 8 rows per block; x and the 8 raw rows
// are staged in shared cooperatively. Per-row math is the old bd=32 path verbatim (lane owns
// block b=lane, warp-shuffle reduction) -> bit-identical.
__global__ void dsv4_gemm_q8_narrow_kernel(const uint8_t* __restrict__ w, int rows, int cols,
                                           int group_rows, const float* __restrict__ x, int m,
                                           int x_row_stride, float* __restrict__ out) {
    extern __shared__ float sxn[];  // [cols floats x] [8*(nb*34)+16 raw bytes]
    const int nb = cols / 32;
    const int r0 = blockIdx.x * 8;
    const int nrow = min(8, rows - r0);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    uint8_t* sraw = reinterpret_cast<uint8_t*>(sxn + cols);
    const int total = nrow * nb * 34;
    const uint8_t* wrow0 = w + (size_t)r0 * (size_t)nb * 34;
    const uint8_t* astart = (const uint8_t*) ((uintptr_t) wrow0 & ~(uintptr_t) 15);
    const int aoff = (int) (wrow0 - astart);
    {
        const int nfull = (aoff + total) >> 4;
        const uint4* srcv = reinterpret_cast<const uint4*>(astart);
        uint4* dstv = reinterpret_cast<uint4*>(sraw);
        int v = threadIdx.x;
        for (; v + 3 * (int) blockDim.x < nfull; v += 4 * blockDim.x) {
            uint4 b0 = srcv[v], b1 = srcv[v + blockDim.x], b2 = srcv[v + 2 * blockDim.x],
                  b3 = srcv[v + 3 * blockDim.x];
            dstv[v] = b0;
            dstv[v + blockDim.x] = b1;
            dstv[v + 2 * blockDim.x] = b2;
            dstv[v + 3 * blockDim.x] = b3;
        }
        for (; v < nfull; v += blockDim.x) dstv[v] = srcv[v];
        for (int i = (nfull << 4) + threadIdx.x; i < aoff + total; i += blockDim.x)
            sraw[i] = astart[i];
    }
    const uint8_t* swrows = sraw + aoff;
    const int r = r0 + wid;
    const size_t g_off = group_rows > 0 ? (size_t)(r / group_rows) * (size_t)cols : 0;
    for (int t = 0; t < m; ++t) {
        const float* xt = x + (size_t)t * (size_t)x_row_stride + g_off;
        __syncthreads();
        {
            // [S7c] x is already warp-slice-transposed in global memory; the stage into shared is a
            // plain float4 copy (was scalar).
            const float4* xv = reinterpret_cast<const float4*>(xt);
            float4* sxv = reinterpret_cast<float4*>(sxn);
            for (int i = threadIdx.x; i < (cols >> 2); i += blockDim.x) sxv[i] = xv[i];
        }
        __syncthreads();
        if (r < rows) {
            const uint8_t* blk = swrows + (size_t)wid * (nb * 34) + (size_t)lane * 34;
            float part = 0.f;
            if (lane < nb) {
                unsigned short bits;
                memcpy(&bits, blk, 2);
                const float d = __half2float(__ushort_as_half(bits));
                const uint8_t* qs = blk + 2;
                // [S7c] transposed layout in shared: lane's float4 j at j*128 + lane*4 (coalesced
                // LDS.128), plus the wide kernel's funnel unpack for the codes (9 LDS.32 instead
                // of 32 LDS.U8).  Byte sequence and FMA order unchanged -> bit-identical.
                const uint32_t* qw = reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<uintptr_t>(qs) & ~(uintptr_t) 3);
                const int qsh = (int) (reinterpret_cast<uintptr_t>(qs) & 3) * 8;
                uint32_t qw0 = qw[0];
                const float4* x4 = reinterpret_cast<const float4*>(sxn + (lane << 2));
                float acc = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 xx = x4[j << 5];
                    const uint32_t qw1 = qw[j + 1];
                    const uint32_t f = __funnelshift_r(qw0, qw1, qsh);
                    acc += static_cast<float>(static_cast<int8_t>(f & 0xffu)) * xx.x;
                    acc += static_cast<float>(static_cast<int8_t>((f >> 8) & 0xffu)) * xx.y;
                    acc += static_cast<float>(static_cast<int8_t>((f >> 16) & 0xffu)) * xx.z;
                    acc += static_cast<float>(static_cast<int8_t>(f >> 24)) * xx.w;
                    qw0 = qw1;
                }
                part = acc * d;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
            if (lane == 0) out[(size_t)t * rows + r] = part;
        }
    }
}

// ---------------------------------------------------------------------------
// [S7c] x layout transform for the q8 gemms.  Within each 1024-float slice the float with
// in-slice index q = 32L + 4j + c (warp lane L, float4 j, component c) is stored at
// j*128 + L*4 + c, turning the gemm's per-thread block reads (8 x float4 at a 128B lane stride,
// 32 L1 wavefronts per warp instruction) into fully coalesced float4 reads (4 wavefronts).
// Pure permutation of fp32 values; consumers are bit-identical.
// ---------------------------------------------------------------------------
__global__ void dsv4_xpose1024_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                      int nf4) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nf4) return;
    const float4 v = in[g];
    out[(g & ~255) | ((g & 7) << 5) | ((g >> 3) & 31)] = v;
}

void xpose1024(const float* x, float* xt, int64_t nfloats, cudaStream_t stream) {
    if (nfloats <= 0) return;
    if ((nfloats & 1023) != 0) {
        std::fprintf(stderr, "dsv4_attn: xpose1024 length %lld not a multiple of 1024\n",
                     (long long) nfloats);
        std::abort();
    }
    const int nf4 = (int) (nfloats >> 2);
    dsv4_xpose1024_kernel<<<(nf4 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(xt), nf4);
    DSV4_CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// [S7c] x layout transform for the NVFP4 MoE GEMV.  That kernel's lane owns 64 consecutive floats
// (one 36 B scale block) and reads them as 16 float4 at a 256 B lane stride - 32 L1 wavefronts per
// warp instruction, and it reads this row once PER OUTPUT ROW (2048 rows x 4096 floats per layer).
// Within each 2048-float chunk the float with index r = 64L + 16s + 4v + c (lane L of the row's
// 32-lane half, sub-block s, float4 v, component c) is stored at (s*4+v)*128 + L*4 + c, so the
// kernel's float4 (s,v) read by the 32 lanes of a warp is contiguous (4 wavefronts).  Pure fp32
// permutation; the consumer's values and FMA order are unchanged -> bit-identical.
// ---------------------------------------------------------------------------
__global__ void dsv4_xpose2048_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                      int nf4) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nf4) return;
    // float4 index within the chunk: q = 16L + 4s + v  ->  (s*4+v)*32 + L
    out[(g & ~511) | (((((g >> 2) & 3) << 2) | (g & 3)) << 5) | ((g >> 4) & 31)] = in[g];
}

void xpose2048(const float* x, float* xt, int64_t nfloats, cudaStream_t stream) {
    if (nfloats <= 0) return;
    if ((nfloats & 2047) != 0) {
        std::fprintf(stderr, "dsv4_attn: xpose2048 length %lld not a multiple of 2048\n",
                     (long long) nfloats);
        std::abort();
    }
    const int nf4 = (int) (nfloats >> 2);
    dsv4_xpose2048_kernel<<<(nf4 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(xt), nf4);
    DSV4_CK(cudaGetLastError());
}

// [S7d diagnostic, off by default] per-shape stream time of the q8 GEMV launcher
// (STRATA_DSV4_Q8TIME=1).  Serialises each launch to read its event pair -> use for attribution
// only, never as a benchmark.
namespace {
// [S7d] pooled event pairs: recorded around each launch on the stream (no per-call sync, so the
// pipeline is not disturbed) and read once at exit -> true per-shape GPU duration.
struct Q8TimeAcc {
    static constexpr int kMax = 65536;
    static constexpr int kCls = 8;
    double ms[kCls] = {0, 0, 0, 0, 0, 0, 0, 0};
    long n[kCls] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool on = false;
    int used = 0;
    cudaEvent_t* ev = nullptr;
    unsigned char* cls = nullptr;
    Q8TimeAcc() {
        const char* s = std::getenv("STRATA_DSV4_Q8TIME");
        on = s != nullptr && s[0] != '\0' && s[0] != '0';
        if (on) {
            ev = (cudaEvent_t*) std::malloc(sizeof(cudaEvent_t) * 2 * kMax);
            cls = (unsigned char*) std::malloc(kMax);
            for (int i = 0; i < 2 * kMax; ++i) cudaEventCreateWithFlags(&ev[i], cudaEventDefault);
            used = 0;
        }
    }
    ~Q8TimeAcc() {
        if (!on) return;
        const char* tp = std::getenv("STRATA_DSV4_Q8TRACE");
        std::FILE* tf = (tp != nullptr && tp[0] != '\0') ? std::fopen(tp, "w") : nullptr;
        for (int i = 0; i < used; ++i) {
            float el = 0;
            cudaEventSynchronize(ev[2 * i + 1]);
            cudaEventElapsedTime(&el, ev[2 * i], ev[2 * i + 1]);
            ms[cls[i]] += el;
            ++n[cls[i]];
            if (tf != nullptr) std::fprintf(tf, "%d %d %.4f\n", i, (int) cls[i], el);
        }
        if (tf != nullptr) std::fclose(tf);
        const char* nm[8] = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b", "idx", "other", "mqa512"};
        double tot = 0;
        long all = 0;
        for (int i = 0; i < 8; ++i) {
            tot += ms[i];
            all += n[i];
            if (n[i])
                std::fprintf(stderr, "dsv4_q8time: %-5s n=%5ld total %8.3f ms (%.3f ms/call)\n", nm[i],
                             n[i], ms[i], ms[i] / (double) n[i]);
        }
        std::fprintf(stderr, "dsv4_q8time: ALL   n=%5ld total %8.3f ms (%.3f ms/call)\n", all, tot,
                     all ? tot / (double) all : 0.0);
    }
    void begin(int c, cudaStream_t st) {
        if (!on || used >= kMax) return;
        cls[used] = (unsigned char) c;
        cudaEventRecord(ev[2 * used], st);
    }
    void end(cudaStream_t st) {
        if (!on || used >= kMax) return;
        cudaEventRecord(ev[2 * used + 1], st);
        ++used;
    }
};
Q8TimeAcc g_q8time;
}  // namespace

// [S7d diagnostic, off by default] in-model bit-level A/B of the q8 GEMV: STRATA_DSV4_Q8BITCMP=1
// runs the pre-S7d reference kernel and the tiled kernel on the same weights/activations inside the
// same process and compares the two outputs word for word (immune to the run-to-run jitter of the
// MoE path, which is not bit-reproducible across processes).
namespace {
__global__ void dsv4_bitcmp_kernel(const float* __restrict__ a, const float* __restrict__ b, long n,
                                   unsigned long long* __restrict__ ndiff) {
    unsigned long long c = 0;
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long) gridDim.x * blockDim.x) {
        uint32_t x, y;
        memcpy(&x, a + i, 4);
        memcpy(&y, b + i, 4);
        if (x != y) ++c;
    }
    if (c) atomicAdd(ndiff, c);
}

struct Q8CmpAcc {
    bool on = false;
    long calls = 0;
    unsigned long long diff = 0;
    long dcalls[7] = {0, 0, 0, 0, 0, 0, 0};
    unsigned long long ddiff[7] = {0, 0, 0, 0, 0, 0, 0};
    float* buf = nullptr;
    size_t cap = 0;
    unsigned long long* dev = nullptr;
    Q8CmpAcc() {
        const char* s = std::getenv("STRATA_DSV4_Q8BITCMP");
        on = s != nullptr && s[0] != '\0' && s[0] != '0';
        if (on) DSV4_CK(cudaMalloc(&dev, sizeof(unsigned long long)));
    }
    ~Q8CmpAcc() {
        if (!on) return;
        const char* nm[7] = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b", "idx", "other"};
        for (int i = 0; i < 7; ++i)
            if (dcalls[i])
                std::fprintf(stderr, "dsv4_q8bitcmp: %-5s calls=%5ld differing_words=%llu\n", nm[i],
                             dcalls[i], ddiff[i]);
        std::fprintf(stderr, "dsv4_q8bitcmp: ALL   calls=%5ld differing_words=%llu (%s)\n", calls,
                     diff, diff == 0 ? "BIT-IDENTICAL" : "MISMATCH");
    }
};
Q8CmpAcc g_q8cmp;
// [S7d diagnostic] STRATA_DSV4_Q8REPEAT=N: on each shape's 3rd call, re-run the same launch N times
// back to back (same weights/x/out, idempotent) and report the sustained rate -> separates
// "kernel is slow" from "something else is using DRAM at the same time".
int g_q8rep = -1;
int g_q8rep_at = 44;  // call index within a shape (43 layers/pass -> 44 = decode1 layer 0)
int g_q8rep_done[8] = {0, 0, 0, 0, 0, 0, 0, 0};
}  // namespace

void gemm_q8(const uint8_t* w, int rows, int cols, int group_rows, const float* x, int m,
             int x_row_stride, float* out, cudaStream_t stream, int out_xpose) {
    // [S7c] x is always the warp-slice-transposed buffer: slice alignment of the per-token stride
    // and of group offsets is a hard requirement (all dsv4 shapes satisfy it).
    if (((x_row_stride & 1023) != 0) || (group_rows > 0 && (cols & 1023) != 0) ||
        ((((uintptr_t) x) & 15) != 0)) {
        std::fprintf(stderr,
                     "dsv4_attn: gemm_q8 x not transpose-compatible (stride %d, cols %d)\n",
                     x_row_stride, cols);
        std::abort();
    }
    // [S7b] narrow rows (nb <= 32) go to the warp-per-row kernel when a block of 8 rows can
    // share one x group slice; wide rows keep block-per-row with narrow blocks by nb.
    const int nb = cols / 32;
    const bool narrow_ok = nb <= 32 && (group_rows == 0 || group_rows % 8 == 0) && !out_xpose;
    const int cls = (rows == 1024 && cols == 4096) ? 0 : (rows == 32768 && cols == 1024) ? 1
                    : (rows == 512 && cols == 4096)               ? 2
                    : (rows == 8192 && cols == 4096)              ? 3
                    : (rows == 4096 && cols == 8192)              ? 4
                    : (rows == 8192 && cols == 1024)              ? 5
                                                                  : 6;
    g_q8time.begin(cls, stream);
    if (narrow_ok) {
        const size_t shmem = (size_t) cols * 4 + (size_t) 8 * (nb * 34) + 16;
        dsv4_gemm_q8_narrow_kernel<<<(rows + 7) / 8, 256, shmem, stream>>>(
            w, rows, cols, group_rows, x, m, x_row_stride, out);
        DSV4_CK(cudaGetLastError());
        g_q8time.end(stream);
        return;
    }
    const int bd = nb >= 128 ? 128 : (nb >= 64 ? 64 : 32);
    // [S7d] R output rows per block.  R > 1 needs every row of a block to see the same x group
    // slice (group_rows % R == 0) and whole blocks (rows % R == 0); STRATA_DSV4_Q8ROWS overrides
    // (1 disables the tiling).
    // Measured in-model (S7d): R>1 is neutral-to-worse here (fewer blocks, bigger staged window),
    // so the default stays R=1 and the tiling is opt-in via STRATA_DSV4_Q8ROWS.
    int R = 1;
    if (nb < 128) R = 1;  // bd < 128 shapes keep their narrow-block shape
    {
        const char* env = std::getenv("STRATA_DSV4_Q8ROWS");
        if (env != nullptr && env[0] != '\0') R = std::atoi(env);
    }
    if (R != 4 && R != 2) R = 1;
    while (R > 1 && ((rows % R) != 0 || (group_rows > 0 && (group_rows % R) != 0))) R >>= 1;
    if ((size_t) R * (size_t) nb * 34 + 32 > 46 * 1024) R = 1;
    const int grid = (rows + R - 1) / R;
    const size_t shmem = (size_t) R * (size_t) nb * 34 + 32;
    if (g_q8cmp.on) {
        const size_t need = (size_t) rows * (size_t) (m > 0 ? m : 1);
        if (need > g_q8cmp.cap) {
            if (g_q8cmp.buf != nullptr) DSV4_CK(cudaFree(g_q8cmp.buf));
            DSV4_CK(cudaMalloc(&g_q8cmp.buf, need * sizeof(float)));
            g_q8cmp.cap = need;
        }
        dsv4_gemm_q8_ref_kernel<<<rows, bd, (size_t) nb * 34 + 32, stream>>>(
            w, rows, cols, group_rows, x, m, x_row_stride, g_q8cmp.buf, out_xpose);
        DSV4_CK(cudaGetLastError());
    }
    switch (R) {
        case 4:
            dsv4_gemm_q8_tiled_kernel<4><<<grid, bd, shmem, stream>>>(
                w, rows, cols, group_rows, x, m, x_row_stride, out, out_xpose);
            break;
        case 2:
            dsv4_gemm_q8_tiled_kernel<2><<<grid, bd, shmem, stream>>>(
                w, rows, cols, group_rows, x, m, x_row_stride, out, out_xpose);
            break;
        default:
            dsv4_gemm_q8_tiled_kernel<1><<<grid, bd, shmem, stream>>>(
                w, rows, cols, group_rows, x, m, x_row_stride, out, out_xpose);
            break;
    }
    DSV4_CK(cudaGetLastError());
    if (g_q8cmp.on) {
        const size_t need = (size_t) rows * (size_t) (m > 0 ? m : 1);
        DSV4_CK(cudaMemsetAsync(g_q8cmp.dev, 0, sizeof(unsigned long long), stream));
        dsv4_bitcmp_kernel<<<64, 256, 0, stream>>>(g_q8cmp.buf, out, (long) need, g_q8cmp.dev);
        unsigned long long h = 0;
        DSV4_CK(cudaMemcpyAsync(&h, g_q8cmp.dev, sizeof(h), cudaMemcpyDeviceToHost, stream));
        DSV4_CK(cudaStreamSynchronize(stream));
        ++g_q8cmp.calls;
        g_q8cmp.diff += h;
        ++g_q8cmp.dcalls[cls];
        g_q8cmp.ddiff[cls] += h;
    }
    DSV4_CK(cudaGetLastError());
    g_q8time.end(stream);
    if (g_q8rep < 0) {
        const char* e = std::getenv("STRATA_DSV4_Q8REPEAT");
        g_q8rep = e != nullptr ? std::atoi(e) : 0;
        const char* a = std::getenv("STRATA_DSV4_Q8REPEAT_AT");
        if (a != nullptr && a[0] != '\0') g_q8rep_at = std::atoi(a);
    }
    if (g_q8rep > 0 && g_q8rep_done[cls] == g_q8rep_at) {
        cudaEvent_t ea, eb;
        cudaEventCreate(&ea);
        cudaEventCreate(&eb);
        cudaEventRecord(ea, stream);
        for (int it = 0; it < g_q8rep; ++it) {
            if (R == 4)
                dsv4_gemm_q8_tiled_kernel<4><<<grid, bd, shmem, stream>>>(
                    w, rows, cols, group_rows, x, m, x_row_stride, out, out_xpose);
            else if (R == 2)
                dsv4_gemm_q8_tiled_kernel<2><<<grid, bd, shmem, stream>>>(
                    w, rows, cols, group_rows, x, m, x_row_stride, out, out_xpose);
            else
                dsv4_gemm_q8_tiled_kernel<1><<<grid, bd, shmem, stream>>>(
                    w, rows, cols, group_rows, x, m, x_row_stride, out, out_xpose);
        }
        cudaEventRecord(eb, stream);
        DSV4_CK(cudaEventSynchronize(eb));
        float el = 0;
        cudaEventElapsedTime(&el, ea, eb);
        const double bytes = (double) rows * (double) nb * 34.0;
        const double per = el / (double) g_q8rep;
        std::fprintf(stderr,
                     "dsv4_q8rep: cls=%d rows=%d cols=%d R=%d m=%d n=%d avg %.4f ms -> %.0f GB/s\n",
                     cls, rows, cols, R, m, g_q8rep, per, bytes / (per * 1e-3) / 1e9);
        cudaEventDestroy(ea);
        cudaEventDestroy(eb);
    }
    ++g_q8rep_done[cls];
}

void rmsnorm(float* x, int rows, int cols, const float* w, cudaStream_t stream) {
    dsv4_rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, rows, cols, w, kRmsEps);
    DSV4_CK(cudaGetLastError());
}

void rope_rows(float* x, int rows, int width, int half, int rows_per_token,
               const Dsv4RopeTables& rope, int64_t start_pos, int inverse,
               cudaStream_t stream) {
    dsv4_rope_rows_mul(x, rows, width, half, rows_per_token, 1, rope, start_pos, inverse, stream);
}

void mqa512(const float* q, const float* kv, const int32_t* topk, const float* sinks, int s, int K,
            int slots, float scale, float* out, cudaStream_t stream) {
    const size_t smem = (size_t)(kHeadDim + K) * 4;
    g_q8time.begin(7, stream);
    dsv4_mqa512_kernel<0><<<s * kNHead, 128, smem, stream>>>(q, kv, nullptr, topk, sinks, kNHead,
                                                             kHeadDim, K, slots, 0, 0, scale, out);
    g_q8time.end(stream);
    DSV4_CK(cudaGetLastError());
}

// [M2] batched MQA-512 for start_pos > 0: one grid over (row, head), ring-resolved table entries.
void mqa512_ring(const float* q, const float* kv_new, const float* kv_ring, const int32_t* topk,
                 const float* sinks, int s, int K, int slots, int win, int64_t start_pos,
                 float scale, float* out, cudaStream_t stream) {
    const size_t smem = (size_t)(kHeadDim + K) * 4;
    const int sp0 = (int)(start_pos % win);
    g_q8time.begin(7, stream);
    dsv4_mqa512_kernel<1><<<s * kNHead, 128, smem, stream>>>(q, kv_ring, kv_new, topk, sinks, kNHead,
                                                             kHeadDim, K, slots, win, sp0, scale,
                                                             out);
    g_q8time.end(stream);
    DSV4_CK(cudaGetLastError());
}

// [M2] write a pass's s KV rows into the ring (start_pos > 0).  At most one wrap because s <= win, so
// two copies cover it; same bytes and same slots as the per-row copies of the default path.
void ring_write_rows(float* kv_cache, const float* kv_new, int win, int64_t start_pos, int s,
                     cudaStream_t stream) {
    const int sp0 = (int)(start_pos % win);
    const int first = std::min(s, win - sp0);
    DSV4_CK(cudaMemcpyAsync(kv_cache + (size_t)sp0 * kHeadDim, kv_new, (size_t)first * kHeadDim * 4,
                            cudaMemcpyDeviceToDevice, stream));
    if (s > first)
        DSV4_CK(cudaMemcpyAsync(kv_cache, kv_new + (size_t)first * kHeadDim,
                                (size_t)(s - first) * kHeadDim * 4, cudaMemcpyDeviceToDevice,
                                stream));
}

// [M2] device twins of the host topk builders (dsv4_window_topk / dsv4_window_topk_pad, and the
// start_pos > 0 branch of dsv4_compress_topk).  One block per row, values identical to the host code.
__global__ void dsv4_window_topk_kernel(int win, int64_t start_pos, int nent, int stride,
                                        int32_t* __restrict__ topk) {
    const int j = blockIdx.x;
    int32_t* row = topk + (size_t) j * stride;
    if (start_pos > 0) {
        const int64_t pos = start_pos + j;
        if (pos >= win - 1) {
            const int64_t sp = pos % win;
            for (int i = threadIdx.x; i < nent; i += blockDim.x)
                row[i] = i >= win ? -1
                                  : (int32_t)(i < win - sp - 1 ? sp + 1 + i : i - (win - sp - 1));
        } else {
            for (int i = threadIdx.x; i < nent; i += blockDim.x) row[i] = i <= pos ? i : -1;
        }
    } else {
        for (int i = threadIdx.x; i < nent; i += blockDim.x) {
            const int64_t v = (j - win + 1 > 0 ? j - win + 1 : 0) + i;
            row[i] = (int32_t)(v > j ? -1 : v);
        }
    }
}

__global__ void dsv4_compress_topk_kernel(int64_t ratio, int64_t start_pos, int64_t offset, int kw,
                                          int kc, int stride, int32_t* __restrict__ topk) {
    const int j = blockIdx.x;
    int32_t* row = topk + (size_t) j * stride + kw;
    const int64_t n = (start_pos + j + 1) / ratio;
    for (int t = threadIdx.x; t < kc; t += blockDim.x) row[t] = (int32_t)(t < n ? t + offset : -1);
}

}  // namespace

// [S4] exported launchers (attn.hpp's internal section) - thin forwards onto the kernels above.
void dsv4_gemm_q8_rows(const uint8_t* w, int rows, int cols, int group_rows, const float* x, int m,
                       int x_row_stride, float* out, cudaStream_t stream, int out_xpose) {
    gemm_q8(w, rows, cols, group_rows, x, m, x_row_stride, out, stream, out_xpose);
}

void dsv4_xpose1024_rows(const float* x, float* xt, int64_t nfloats, cudaStream_t stream) {
    xpose1024(x, xt, nfloats, stream);
}

void dsv4_xpose2048_rows(const float* x, float* xt, int64_t nfloats, cudaStream_t stream) {
    xpose2048(x, xt, nfloats, stream);
}

void dsv4_rmsnorm_rows(float* x, int rows, int cols, const float* w, cudaStream_t stream) {
    rmsnorm(x, rows, cols, w, stream);
}

// [S7e] the rmsnorm+rope pair as ONE launch (bit-identical values, see dsv4_rmsnorm_rope_kernel).
void dsv4_rmsnorm_rope_rows(float* x, int rows, int cols, int half, int rows_per_token, int pos_mul,
                            const Dsv4RopeTables& rope, int64_t start_pos, int inverse,
                            const float* w, cudaStream_t stream) {
    dsv4_rmsnorm_rope_kernel<<<rows, 256, 0, stream>>>(x, rows, cols, half, rows_per_token, pos_mul,
                                                       rope.cos, rope.sin, start_pos, inverse, w,
                                                       kRmsEps);
    DSV4_CK(cudaGetLastError());
}

void dsv4_rope_rows_mul(float* x, int rows, int width, int half, int rows_per_token, int pos_mul,
                        const Dsv4RopeTables& rope, int64_t start_pos, int inverse,
                        cudaStream_t stream) {
    const int total = rows * half;
    const int blocks = (total + 255) / 256;
    dsv4_rope_kernel<<<blocks, 256, 0, stream>>>(x, rows, width, half, rows_per_token, pos_mul,
                                                 rope.cos, rope.sin, start_pos, inverse);
    DSV4_CK(cudaGetLastError());
}

void dsv4_mqa512_rows(const float* q, const float* kv, const int32_t* topk, const float* sinks, int s,
                      int K, int slots, float scale, float* out, cudaStream_t stream) {
    mqa512(q, kv, topk, sinks, s, K, slots, scale, out, stream);
}

// [M2] batched prefill attention switches (see attn.hpp); both default off / opt-in only.
bool dsv4_attn_batch_enabled() {
    static const bool on = [] {
        // Default ON: the batched path is bit-identical to the per-row path on the 176-file prefill gate
        // (and on the 4k chunked gate), and it is the difference between 98 s and 158 s for a 4k prompt
        // on 2xL20.  STRATA_DSV4_ATTN_BATCH=0 still selects the older one-launch-per-row path for A/B.
        const char* v = std::getenv("STRATA_DSV4_ATTN_BATCH");
        return v == nullptr || v[0] == '\0' || std::strcmp(v, "0") != 0;
    }();
    return on;
}

void dsv4_window_topk_dev(int win, int s, int64_t start_pos, int stride, int nent, int32_t* topk,
                          cudaStream_t stream) {
    dsv4_window_topk_kernel<<<s, 128, 0, stream>>>(win, start_pos, nent, stride, topk);
    DSV4_CK(cudaGetLastError());
}

void dsv4_compress_topk_dev(int64_t ratio, int s, int64_t start_pos, int64_t offset, int kw, int kc,
                            int stride, int32_t* topk, cudaStream_t stream) {
    dsv4_compress_topk_kernel<<<s, 128, 0, stream>>>(ratio, start_pos, offset, kw, kc, stride, topk);
    DSV4_CK(cudaGetLastError());
}

void dsv4_mqa512_rows_ring(const float* q, const float* kv_new, const float* kv_ring,
                           const int32_t* topk, const float* sinks, int s, int K, int slots, int win,
                           int64_t start_pos, float scale, float* out, cudaStream_t stream) {
    mqa512_ring(q, kv_new, kv_ring, topk, sinks, s, K, slots, win, start_pos, scale, out, stream);
}

bool dsv4_attn_workspace_init(Dsv4AttnWorkspace& ws, int max_s, int max_k) {
    ws.max_s = max_s;
    ws.max_k = max_k;
    const size_t qr = (size_t)max_s * kQRank, q = (size_t)max_s * kNHead * kHeadDim,
                 kv = (size_t)max_s * kHeadDim, o8k = (size_t)max_s * kORows,
                 topk = (size_t)max_s * max_k;
    bool ok = cudaMalloc(&ws.qr, qr * 4) == cudaSuccess &&
              cudaMalloc(&ws.q, q * 4) == cudaSuccess &&
              cudaMalloc(&ws.kv_new, kv * 4) == cudaSuccess &&
              cudaMalloc(&ws.attn, q * 4) == cudaSuccess &&
              cudaMalloc(&ws.o8k, o8k * 4) == cudaSuccess &&
              cudaMalloc(&ws.cache_tmp, kv * 4) == cudaSuccess &&
              cudaMalloc(&ws.topk, topk * 4) == cudaSuccess &&
              cudaMalloc(&ws.xt, (size_t) max_s * kNEmbd * 4) == cudaSuccess &&
              cudaMalloc(&ws.qr_t, qr * 4) == cudaSuccess &&
              cudaMalloc(&ws.attn_t, q * 4) == cudaSuccess;
    if (!ok) {
        std::fprintf(stderr, "dsv4_attn: workspace allocation failed\n");
        dsv4_attn_workspace_free(ws);
        return false;
    }
    return true;
}

void dsv4_attn_workspace_free(Dsv4AttnWorkspace& ws) {
    cudaFree(ws.qr);
    cudaFree(ws.q);
    cudaFree(ws.kv_new);
    cudaFree(ws.attn);
    cudaFree(ws.o8k);
    cudaFree(ws.cache_tmp);
    cudaFree(ws.topk);
    cudaFree(ws.xt);
    cudaFree(ws.qr_t);
    cudaFree(ws.attn_t);
    ws = Dsv4AttnWorkspace{};
}

int dsv4_window_topk(int win, int s, int64_t start_pos, int32_t* topk) {
    if (start_pos > 0) {
        // [G1] multi-row decode: row j is position start_pos + j, so each row gets the ring permutation
        // of ITS position (or the masked short window before the ring is full).  s == 1 reproduces the
        // single-row table verbatim.
        for (int j = 0; j < s; ++j) {
            int32_t* row = topk + (size_t)j * win;
            int64_t pos = start_pos + j;
            if (pos >= win - 1) {
                const int64_t sp = pos % win;
                for (int i = 0; i < win; ++i)
                    row[i] = (int32_t)(i < win - sp - 1 ? sp + 1 + i : i - (win - sp - 1));
            } else {
                for (int i = 0; i < win; ++i) row[i] = i <= pos ? (int32_t)i : -1;
            }
        }
        return win;
    }
    const int k = std::min(s, win);
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < k; ++i) {
            const int64_t v = std::max<int64_t>(j - win + 1, 0) + i;
            topk[(size_t)j * k + i] = (int32_t)(v > j ? -1 : v);
        }
    return k;
}

// [chunk] Same table as the start_pos == 0 branch of dsv4_window_topk, but always `win` entries per
// row.  A prefilling pass narrower than the window would otherwise lay its compressed section out at
// column `s` instead of `win`, which moves the compressed terms into different softmax denominator
// lanes and changes the rounding of a row whose values are otherwise identical.  The -1 padding is
// inert: exp(-inf) is 0 and the value loop skips e <= 0.
int dsv4_window_topk_pad(int win, int s, int32_t* topk) {
    for (int j = 0; j < s; ++j)
        for (int i = 0; i < win; ++i) {
            const int64_t v = std::max<int64_t>(j - win + 1, 0) + i;
            topk[(size_t)j * win + i] = (int32_t)(v > j ? -1 : v);
        }
    return win;
}

bool dsv4_attn_forward(const Dsv4AttnWeights& w, const Dsv4RopeTables& rope,
                       Dsv4AttnWorkspace& ws, const float* x, float* out, float* kv_cache,
                       int s, int64_t start_pos, int sliding_window, cudaStream_t stream) {
    // [R-02/R-04] input bounds, checked here as well as at run_pass: the workspace holds ws.max_s rows,
    // and a prefill wider than the window would make the ring-shift below copy (size_t)(win - s) rows.
    if (s <= 0 || s > ws.max_s) {
        std::fprintf(stderr, "dsv4_attn: %d rows exceed the attention workspace width %d\n", s,
                     ws.max_s);
        return false;
    }
    if (start_pos < 0) {
        std::fprintf(stderr, "dsv4_attn: negative start_pos %lld\n", (long long) start_pos);
        return false;
    }
    if (start_pos == 0 && s > sliding_window) {
        std::fprintf(stderr,
                     "dsv4_attn: prefill of %d rows exceeds the sliding window %d (ring shift is "
                     "undefined for s > win)\n",
                     s, sliding_window);
        return false;
    }
    // q path: gemm -> rmsnorm(q_norm) -> gemm -> per-head renorm -> rope
    // [S7c] gemms consume the warp-slice-transposed copies (bit-identical values)
    {
        AttnTimeScope t(kAtXpose, stream);
        xpose1024(x, ws.xt, (int64_t) s * kNEmbd, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        gemm_q8(w.wq_a, kQRank, kNEmbd, 0, ws.xt, s, kNEmbd, ws.qr, stream, 0);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        rmsnorm(ws.qr, s, kQRank, w.q_norm, stream);
    }
    {
        AttnTimeScope t(kAtXpose, stream);
        xpose1024(ws.qr, ws.qr_t, (int64_t) s * kQRank, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        gemm_q8(w.wq_b, kNHead * kHeadDim, kQRank, 0, ws.qr_t, s, kQRank, ws.q, stream, 0);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        rmsnorm(ws.q, s * kNHead, kHeadDim, nullptr, stream);
    }
    {
        AttnTimeScope t(kAtRope, stream);
        rope_rows(ws.q, s * kNHead, kHeadDim, rope.half, kNHead, rope, start_pos, 0, stream);
    }
    // kv path: gemm -> rmsnorm(kv_norm) -> rope
    {
        AttnTimeScope t(kAtProj, stream);
        gemm_q8(w.wkv, kHeadDim, kNEmbd, 0, ws.xt, s, kNEmbd, ws.kv_new, stream, 0);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        rmsnorm(ws.kv_new, s, kHeadDim, w.kv_norm, stream);
    }
    {
        AttnTimeScope t(kAtRope, stream);
        rope_rows(ws.kv_new, s, kHeadDim, rope.half, 1, rope, start_pos, 0, stream);
    }
    // cache maintenance + kv source selection (dsv4ref: prefill attends over kv_new directly)
    const float* kv_ptr = ws.kv_new;
    const int win = sliding_window;
    {
        AttnTimeScope t(kAtMem, stream);
        if (start_pos == 0) {
            if (s <= win) {
                DSV4_CK(cudaMemcpyAsync(kv_cache, ws.kv_new, (size_t)s * kHeadDim * 4,
                                        cudaMemcpyDeviceToDevice, stream));
            } else {
                // ring shift: cache <- cache[s:win] ++ kv_new (dsv4ref memmove+memcpy)
                DSV4_CK(cudaMemcpyAsync(ws.cache_tmp, kv_cache + (size_t)s * kHeadDim,
                                        (size_t)(win - s) * kHeadDim * 4, cudaMemcpyDeviceToDevice,
                                        stream));
                DSV4_CK(cudaMemcpyAsync(kv_cache, ws.cache_tmp, (size_t)(win - s) * kHeadDim * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(kv_cache + (size_t)(win - s) * kHeadDim, ws.kv_new,
                                        (size_t)s * kHeadDim * 4, cudaMemcpyDeviceToDevice, stream));
            }
        } else {
            // [G1] multi-row decode: each row writes its own ring slot and attends with its own table, so
            // row r sees only slots up to its own position.  s == 1 is the single-row path unchanged.
            if (s == 1) {
                DSV4_CK(cudaMemcpyAsync(kv_cache + (start_pos % win) * kHeadDim, ws.kv_new,
                                        (size_t)kHeadDim * 4, cudaMemcpyDeviceToDevice, stream));
            }
            kv_ptr = kv_cache;
        }
    }
    // topk (hash layers: window part only).  [M2] batched: built on the device into the same row layout
    // the host builder produces (pitch K), so no host vector and no pageable upload.
    const bool batch = dsv4_attn_batch_enabled() && start_pos > 0 && s > 1;
    std::vector<int32_t> h_topk;
    int K = win;
    if (batch) {
        AttnTimeScope t(kAtTableH2D, stream);
        dsv4_window_topk_dev(win, s, start_pos, K, K, ws.topk, stream);
    } else {
        h_topk.resize((size_t)s * win);
        const auto t_tab0 = std::chrono::steady_clock::now();
        K = dsv4_window_topk(win, s, start_pos, h_topk.data());
        attn_time_host_table(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t_tab0).count());
        AttnTimeScope t(kAtTableH2D, stream);
        DSV4_CK(cudaMemcpyAsync(ws.topk, h_topk.data(), (size_t)s * K * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream));
    }
    // dense MQA-512 attention, softmax scale 512^-0.5, no mscale
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const size_t smem = (size_t)(kHeadDim + K) * 4;
    // [R8] readable KV rows for this launch: the whole kv_new on prefill, the ring on decode
    const int kv_slots = start_pos == 0 ? s : win;
    if (batch) {
        // [M2] one launch for all s rows; the pass's KV rows go into the ring afterwards, because a
        // window entry only ever resolves to a row at or before the attending row ([R-01]).
        {
            AttnTimeScope t(kAtMqa, stream);
            dsv4_mqa512_rows_ring(ws.q, ws.kv_new, kv_cache, ws.topk, w.sinks, s, K, kv_slots, win,
                                  start_pos, scale, ws.attn, stream);
        }
        AttnTimeScope t(kAtMem, stream);
        ring_write_rows(kv_cache, ws.kv_new, win, start_pos, s, stream);
    } else if (start_pos > 0 && s > 1) {
        // [G1] one launch per row (grid kNHead => the kernel's row index is 0 for all of them), so the
        // per-row table is the only thing that changes between rows.
        AttnTimeScope t(kAtMqa, stream);
        for (int j = 0; j < s; ++j) {
            DSV4_CK(cudaMemcpyAsync(kv_cache + ((start_pos + j) % win) * kHeadDim,
                                    ws.kv_new + (size_t)j * kHeadDim, (size_t)kHeadDim * 4,
                                    cudaMemcpyDeviceToDevice, stream));
            dsv4_mqa512_kernel<0><<<kNHead, 128, smem, stream>>>(
                ws.q + (size_t)j * kNHead * kHeadDim, kv_ptr, nullptr, ws.topk + (size_t)j * K,
                w.sinks, kNHead, kHeadDim, K, kv_slots, 0, 0, scale,
                ws.attn + (size_t)j * kNHead * kHeadDim);
            DSV4_CK(cudaGetLastError());
        }
    } else {
        AttnTimeScope t(kAtMqa, stream);
        dsv4_mqa512_kernel<0><<<s * kNHead, 128, smem, stream>>>(
            ws.q, kv_ptr, nullptr, ws.topk, w.sinks, kNHead, kHeadDim, K, kv_slots, 0, 0, scale,
            ws.attn);
        DSV4_CK(cudaGetLastError());
    }
    // inverse rope on the attention output (output-side inverse rotary trap)
    {
        AttnTimeScope t(kAtRope, stream);
        rope_rows(ws.attn, s * kNHead, kHeadDim, rope.half, kNHead, rope, start_pos, 1, stream);
    }
    // grouped O projection: [s][8][4096] einsum wo_a[8][1024][4096] -> [s][8192] -> wo_b
    // [S7c] wo_a reads the transposed attention output and writes o8k directly in the transposed
    // layout for wo_b (free transpose on the producer's scattered single-value writes).
    {
        AttnTimeScope t(kAtXpose, stream);
        xpose1024(ws.attn, ws.attn_t, (int64_t) s * kNHead * kHeadDim, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        gemm_q8(w.wo_a, kORows, kNEmbd, kOGroup, ws.attn_t, s, kNHead * kHeadDim, ws.o8k, stream, 1);
        gemm_q8(w.wo_b, kNEmbd, kORows, 0, ws.o8k, s, kORows, out, stream, 0);
    }
    return true;
}

}  // namespace strata::dsv4
