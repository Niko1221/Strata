// src/inkling2/ink2_engine.cu - strata-ink2: Inkling-2-Mini's engine path (NEW-MODELS-PLAN.md T2), step 1: the whole
// forward pass on the GPU, one token at a time (the prompt too), checked against strata-ink2ref (T1) and the fp32
// transformers oracle (terminal/tools/inkling2/).
//
//   * weights resident in their checkpoint form, as in strata-ink2ref: BF16 dense matrices, NVFP4 experts (layers 0-19:
//     e2m1 pairs, an e4m3 scale per 16, an fp32 factor amax/denominator per expert), MXFP8 experts (layers 20-23: e4m3,
//     a ue8m0 exponent per 32).  Every matrix-vector product reads them as they are, fp32 activations, fp32 sums
//     ("W exact, A16"): nothing is requantized;
//   * the embedding table stays in host RAM (one row per token);
//   * attention: split over 256-position chunks per head with a log-sum-exp merge that adds the per-token sink logit;
//     the relative bias table [24][1024] built per token; KV in fp32 (CBA: fp32 KV, --ctx 8192 by default - int8 KV is
//     T2 step 3); sliding layers keep a 1024-position ring, full layers the whole context;
//   * convolutions with a 3-row history per layer application (48 applications: each loop pass has its own state).
//
//   * one CUDA graph per token (--graph 1, the default): positions on the device, built at the first token.  Build with
//     --default-stream per-thread (the graph is captured on cudaStreamPerThread).
//
//   * the prompt in chunks of --chunk tokens (default 512; 0 = one token at a time through the decode graph): cuBLAS
//     SGEMM in full FP32 for the dense products (fp32 copy of each BF16 matrix made per use), a grouped expert product
//     decoding NVFP4 / MXFP8 on the fly, tiled fp32 attention.  Link -lcublas.
//
// Modes: --tokens IDS | --tokens-file F [--n N] [--dump-logits F] [--ctx N] [--graph 0|1] [--chunk N]; prints prompt /
// decode speed.
#include "../image/vae/safetensors_lite.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using vec = std::vector<float>;
namespace wmma = nvcuda::wmma;
using strata::image::StFile;
using strata::image::StTensor;

constexpr int H = 2304, NL = 24, NQ = 24, NKV = 8, HD = 128, NREP = NQ / NKV, DREL = 16, EXTENT = 1024, WINDOW = 1024;
constexpr int QD = NQ * HD, KVD = NKV * HD;   // 3072, 1024
constexpr int NE = 128, TOPK = 8, FF = 1152, KC = 4, GATE_RANK = 64, VOCAB = 200058, VOCAB_ROWS = 201024;
constexpr int LOOP_BEGIN = 6, LOOP_END = 18, NLOOP = LOOP_END - LOOP_BEGIN, PASSES = 3, BLOCK = 6;
constexpr int NCALLS = NL + (PASSES - 1) * NLOOP;
constexpr int MXFP8_FIRST = 20;
constexpr float EPS = 1e-6f, QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f, LOG_ALPHA = 0.1f, LOG_FLOOR = 8192.0f;
constexpr float GATE_CAP = 4.0f, UP_CAP = 25.0f, ROUTE_SCALE = 8.0f, LOGIT_CAP = 30.0f;
constexpr int CHUNK = 256;   // attention positions per block
int PF_RING_EXTRA = 512;     // = the prefill chunk: a chunk's keys must not evict the window of its first query
constexpr int OFF_GATE = QD, OFF_REL = 2 * QD, OFF_SINK = OFF_REL + NQ * DREL, OFF_K = OFF_SINK + NQ;   // in proj

bool full_layer(int L) { return L % BLOCK == BLOCK - 1; }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// Kernels

__device__ __forceinline__ float bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ float e4m3_decode(uint32_t v) {
    const int e = (v >> 3) & 15, m = v & 7;
    float r = e == 0 ? (float) m * 0.001953125f : ldexpf(1.0f + (float) m * 0.125f, e - 7);
    if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
    return (v & 0x80) ? -r : r;
}

__device__ __forceinline__ float warp_sum(float a) {
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    return a;
}
__device__ __forceinline__ float warp_max(float a) {
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

// sum over the block (blockDim.x a multiple of 32, <= 1024); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x & 31, nw = blockDim.x / 32;
    __syncthreads();
    if (!l) red[w] = v;
    __syncthreads();
    float t = l < nw ? red[l] : 0.0f;
    return warp_sum(t);
}
__device__ float block_max(float v) {
    __shared__ float red[32];
    v = warp_max(v);
    const int w = threadIdx.x / 32, l = threadIdx.x & 31, nw = blockDim.x / 32;
    __syncthreads();
    if (!l) red[w] = v;
    __syncthreads();
    float t = l < nw ? red[l] : -INFINITY;
    return warp_max(t);
}

// y[row] = W[row] . x  (BF16 W [N, K], K % 8 == 0), one warp per row, x staged in shared memory and read as float4
// (lane stride 32 B: 2-way bank conflicts at most)
// R rows per warp: the staged x serves 8R rows per block, R independent loads in flight per lane.  Each row's sum is
// the same sequence of operations whatever R (bit-identical results).
template <int R>
__device__ __forceinline__ void gemv_bf16_dot(const uint16_t* __restrict__ W, const float* xs, float* __restrict__ y,
                                              int N, int K, int row0) {
    const int lane = threadIdx.x & 31;
    const uint4* w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) w[r] = reinterpret_cast<const uint4*>(W + (size_t) min(row0 + r, N - 1) * K);
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint4 v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = __ldg(w[r] + i);
        const float4 a = x4[2 * i], b = x4[2 * i + 1];
#pragma unroll
        for (int r = 0; r < R; ++r)
            acc[r] += bf16lo(v[r].x) * a.x + bf16hi(v[r].x) * a.y + bf16lo(v[r].y) * a.z + bf16hi(v[r].y) * a.w +
                      bf16lo(v[r].z) * b.x + bf16hi(v[r].z) * b.y + bf16lo(v[r].w) * b.z + bf16hi(v[r].w) * b.w;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[row0 + r] = t;
    }
}
template <int R>
__device__ __forceinline__ void gemv_bf16_rows(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                               float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    gemv_bf16_dot<R>(W, xs, y, N, K, row0);
}
template <int R>
__global__ void k_gemv_bf16(const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N,
                            int K) {
    gemv_bf16_rows<R>(W, x, y, N, K);
}
// q / k / v projections with three inputs in one launch: rows [0, b1) . xq, [b1, b2) . xk, [b2, N) . xv (R divides b1 and
// b2: a warp's rows share an input).  A block stages only the inputs its rows touch (smem 3K).
template <int R>
__global__ void k_gemv_bf16_qkv(const uint16_t* __restrict__ W, const float* __restrict__ xq, const float* __restrict__ xk,
                                const float* __restrict__ xv, int b1, int b2, float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    const int rb = blockIdx.x * (blockDim.x / 32) * R, re = min(rb + (int) (blockDim.x / 32) * R, N);
    if (rb < b1) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xq[i];
    if (rb < b2 && re > b1) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[K + i] = xk[i];
    if (re > b2) for (int i = threadIdx.x; i < K; i += blockDim.x) xs[2 * K + i] = xv[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    if (row0 >= N) return;
    gemv_bf16_dot<R>(W, xs + (row0 < b1 ? 0 : row0 < b2 ? K : 2 * K), y, N, K, row0);
}

// Up to 3 independent message gates per launch (gate j = blockIdx.y, or blockIdx.x for the one-block kernels): the
// q / k / v gates of a layer application are independent, 4 launches instead of 12; same arithmetic per gate.
struct GateArgs {
    const uint16_t* down[3];
    const uint16_t* fg[3];
    const float* x[3];
    const float* m[3];
    float* y[3];
    const float* nw[3];
    float* yn[3];
};
// fg GEMV of gate j: gfg[j] [2H] = fg_j [2H, GATE_RANK] . gdown[j]
template <int R>
__global__ void k_gemv_bf16_gate(GateArgs a, const float* __restrict__ gdown, float* __restrict__ gfg, int N, int K) {
    const int j = blockIdx.y;
    gemv_bf16_rows<R>(a.fg[j], gdown + (size_t) j * GATE_RANK, gfg + (size_t) j * N, N, K);
}

// e2m1 pair (one byte: low nibble first) -> fp32 values * 2^-14: the nibble's sign/exponent/mantissa bits placed
// in an fp16 (exact, subnormal 0.5 included); the 2^14 goes into the block scale
__device__ __forceinline__ float2 e2m1x2(uint32_t byte) {
    const uint32_t a = (byte & 0xFu) | ((byte & 0xF0u) << 12);
    const uint32_t h = ((a & 0x00080008u) << 12) | ((a & 0x00070007u) << 9);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
// two e4m3 bytes (in bits 0-7 and 16-23) -> fp32 values * 2^-8 (fp16 placement, exact incl. subnormals)
__device__ __forceinline__ float2 e4m3x2(uint32_t a) {
    const uint32_t h = ((a & 0x00800080u) << 8) | ((a & 0x007F007Fu) << 7);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// The TOPK selected experts at once (blockIdx.y = k): y[k][row] = W_e[row] . x_k, e = ids[k].
// NVFP4: lane reads 4 bytes = 8 columns (half a 16-column block, its e4m3 scale s[i/2]); the sum times factors[e].
// SoftcappedSwiGLU of one (gate, up) pair (row 2j gate j, 2j+1 up j of the gu buffer)
__device__ __forceinline__ float swiglu1(float g, float u) {
    return (tanhf(g / GATE_CAP) * GATE_CAP) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / UP_CAP) * UP_CAP);
}
// SWI: x holds [gate, up] pairs (gu rows, stride x_stride = 2K): the GEMV input is their SwiGLU, computed while staging
template <int R, bool SWI = false>
__global__ void k_gemv_nvfp4_sel(const uint8_t* const* __restrict__ tab,
                                 const float* __restrict__ factors, const int* __restrict__ ids,
                                 const float* __restrict__ x, int x_stride, float* __restrict__ y, int N, int K) {
    extern __shared__ float sm[];
    float* xs = sm;           // K
    float* e4 = sm + K;       // 256 (scale table, * 2^14)
    const float* xk = x + (size_t) blockIdx.y * x_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = SWI ? swiglu1(xk[2 * i], xk[2 * i + 1]) : xk[i];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4[i] = e4m3_decode(i) * 16384.0f;
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const int e = ids[blockIdx.y];
    const uint8_t* W = tab[e];
    const uint8_t* S = W + (size_t) N * (K / 2);
    const uint32_t* w[R];
    const uint8_t* s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const size_t row = min(row0 + r, N - 1);
        w[r] = reinterpret_cast<const uint32_t*>(W + row * (K / 2));
        s[r] = S + row * (K / 16);
    }
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 8; i += 32) {
        uint32_t v[R];
        uint8_t sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); sc[r] = __ldg(s[r] + (i >> 1)); }
        const float4 a = x4[2 * i], b = x4[2 * i + 1];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 p0 = e2m1x2(v[r] & 0xff), p1 = e2m1x2((v[r] >> 8) & 0xff), p2 = e2m1x2((v[r] >> 16) & 0xff),
                         p3 = e2m1x2(v[r] >> 24);
            const float part = p0.x * a.x + p0.y * a.y + p1.x * a.z + p1.y * a.w + p2.x * b.x + p2.y * b.y +
                               p3.x * b.z + p3.y * b.w;
            acc[r] += part * e4[sc[r]];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[(size_t) blockIdx.y * N + row0 + r] = t * factors[e];
    }
}

// MXFP8: lane reads 4 bytes = 4 columns; the ue8m0 exponent per 32 columns (8 words): 2^(s - 127) * 2^8
template <int R, bool SWI = false>
__global__ void k_gemv_mxfp8_sel(const uint8_t* const* __restrict__ tab, const int* __restrict__ ids, const float* __restrict__ x, int x_stride,
                                 float* __restrict__ y, int N, int K) {
    extern __shared__ float sm[];
    float* xs = sm;
    const float* xk = x + (size_t) blockIdx.y * x_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = SWI ? swiglu1(xk[2 * i], xk[2 * i + 1]) : xk[i];
    __syncthreads();
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const int e = ids[blockIdx.y];
    const uint8_t* W = tab[e];
    const uint8_t* S = W + (size_t) N * K;
    const uint32_t* w[R];
    const uint8_t* s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const size_t row = min(row0 + r, N - 1);
        w[r] = reinterpret_cast<const uint32_t*>(W + row * K);
        s[r] = S + row * (K / 32);
    }
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll 2
    for (int i = lane; i < K / 4; i += 32) {
        uint32_t v[R];
        uint8_t sc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { v[r] = __ldg(w[r] + i); sc[r] = __ldg(s[r] + (i >> 3)); }
        const float4 a = x4[i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 p01 = e4m3x2((v[r] & 0xffu) | ((v[r] & 0xff00u) << 8));
            const float2 p23 = e4m3x2(((v[r] >> 16) & 0xffu) | ((v[r] & 0xff000000u) >> 8));
            const float part = p01.x * a.x + p01.y * a.y + p23.x * a.z + p23.y * a.w;
            acc[r] += part * exp2f((float) sc[r] - 119.0f);   // 2^(s-127) * 2^8
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = warp_sum(acc[r]);
        if (!lane && row0 + r < N) y[(size_t) blockIdx.y * N + row0 + r] = t;
    }
}

// y = rms(x) (* w if w): one block
__global__ void k_rms(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n) {
    x += (size_t) blockIdx.x * n;
    y += (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = w ? w[i] * (x[i] * r) : x[i] * r;
}

// attention residual: stack [S][H] (contiguous), NQ queries [NQ][H] -> out [NQ][H] = sum_j softmax_j(q . n_j / sqrt H)
// n_j, n_j = rms(stack_j) (unweighted); q . n_j = rinv_j * (q . stack_j).  One block of 256 threads (9 columns each),
// S and NQ compile-time (registers only).  With `outn`, also outn[k] = rms(out[k]) * nw[k] (nw[k] null: unweighted);
// `out` may then be null.
constexpr int AR_T = 256, AR_C = H / AR_T;
template <int S, int NQ>
__global__ void k_attnres(const float* __restrict__ stack, size_t sstride, const float* __restrict__ queries,
                          float* __restrict__ out, const float* nw0, const float* nw1, const float* nw2,
                          float* __restrict__ outn) {
    stack += (size_t) blockIdx.x * H;   // a grid of T tokens: stack entry j of token t at stack + j * sstride + t * H
    if (out) out += (size_t) blockIdx.x * NQ * H;
    if (outn) outn += (size_t) blockIdx.x * NQ * H;
    constexpr int NV = S + NQ * S;
    __shared__ float red[AR_T / 32][NV + NQ];
    __shared__ float p[NQ][S], rinv[S], onorm[NQ];
    const int t = threadIdx.x, w = t / 32, l = t & 31;
    float v[S][AR_C];
    float acc[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int c = 0; c < AR_C; ++c) {
        const int i = c * AR_T + t;
        float q[NQ];
#pragma unroll
        for (int k = 0; k < NQ; ++k) q[k] = queries[(size_t) k * H + i];
#pragma unroll
        for (int j = 0; j < S; ++j) {
            v[j][c] = stack[j * sstride + i];
            acc[j] += v[j][c] * v[j][c];
#pragma unroll
            for (int k = 0; k < NQ; ++k) acc[S + k * S + j] += q[k] * v[j][c];
        }
    }
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const float r = warp_sum(acc[i]);
        if (!l) red[w][i] = r;
    }
    __syncthreads();
    if (t < NV) {
        float r = 0.0f;
#pragma unroll
        for (int k = 0; k < AR_T / 32; ++k) r += red[k][t];
        red[0][t] = r;
    }
    __syncthreads();
    if (t < S) rinv[t] = rsqrtf(red[0][t] / H + EPS);
    __syncthreads();
    if (t < NQ) {
        float lg[S], mx = -INFINITY, z = 0.0f;
#pragma unroll
        for (int j = 0; j < S; ++j) { lg[j] = red[0][S + t * S + j] * rinv[j] * rsqrtf((float) H); mx = fmaxf(mx, lg[j]); }
#pragma unroll
        for (int j = 0; j < S; ++j) { lg[j] = expf(lg[j] - mx); z += lg[j]; }
#pragma unroll
        for (int j = 0; j < S; ++j) p[t][j] = lg[j] / z * rinv[j];
    }
    __syncthreads();
    float o[NQ][AR_C], ss[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        ss[k] = 0.0f;
#pragma unroll
        for (int c = 0; c < AR_C; ++c) {
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < S; ++j) a += p[k][j] * v[j][c];
            o[k][c] = a;
            ss[k] += a * a;
            if (out) out[(size_t) k * H + c * AR_T + t] = a;
        }
    }
    if (!outn) return;
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const float r = warp_sum(ss[k]);
        if (!l) red[w][NV + k] = r;
    }
    __syncthreads();
    if (t < NQ) {
        float r = 0.0f;
#pragma unroll
        for (int k = 0; k < AR_T / 32; ++k) r += red[k][NV + t];
        onorm[t] = rsqrtf(r / H + EPS);
    }
    __syncthreads();
    const float* nw[3] = {nw0, nw1, nw2};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const float* wk = k < 3 ? nw[k] : nullptr;
#pragma unroll
        for (int c = 0; c < AR_C; ++c) {
            const int i = c * AR_T + t;
            outn[(size_t) k * H + i] = wk ? wk[i] * (o[k][c] * onorm[k]) : o[k][c] * onorm[k];
        }
    }
}

// rms of two vectors at once (blockIdx.x 0: x, 1: m), unweighted: gate j's [rms(x), rms(m)] into cat[j] [2H]
__global__ void k_rms2(GateArgs g, float* __restrict__ cat) {
    const int j = blockIdx.y;
    const float* a = blockIdx.x ? g.m[j] : g.x[j];
    float* b = cat + (size_t) (2 * j + blockIdx.x) * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += a[i] * a[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) b[i] = a[i] * r;
}

// small-N matrix-vector product: one block per row (BF16 W [N, K]), for N too small to fill the GPU warp-per-row
__device__ __forceinline__ float rowblock_dot(const uint16_t* __restrict__ W, const float* __restrict__ x, int K) {
    const uint4* w = reinterpret_cast<const uint4*>(W + (size_t) blockIdx.x * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < K / 8; i += blockDim.x) {
        const uint4 v = __ldg(w + i);
        const float4 a = __ldg(x4 + 2 * i), b = __ldg(x4 + 2 * i + 1);
        acc += bf16lo(v.x) * a.x + bf16hi(v.x) * a.y + bf16lo(v.y) * a.z + bf16hi(v.y) * a.w +
               bf16lo(v.z) * b.x + bf16hi(v.z) * b.y + bf16lo(v.w) * b.z + bf16hi(v.w) * b.w;
    }
    return block_sum(acc);
}
__global__ void k_gemv_bf16_rowblock(const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                     int K) {
    const float acc = rowblock_dot(W, x, K);
    if (!threadIdx.x) y[blockIdx.x] = acc;
}
// gate j's down projection: gdown[j] [GATE_RANK] = down_j . cat[j]; grid (GATE_RANK, gates)
__global__ void k_gate_down(GateArgs g, const float* __restrict__ cat, float* __restrict__ gdown, int K) {
    const int j = blockIdx.y;
    const float acc = rowblock_dot(g.down[j], cat + (size_t) j * 2 * H, K);
    if (!threadIdx.x) gdown[(size_t) j * GATE_RANK + blockIdx.x] = acc;
}

// message gate tail + the following norm: y = (1 - tanh f) * x + tanh g * m; yn = rms(y) * w.  One block per gate.
__global__ void k_gate_mix_rms(GateArgs ga, const float* gfg) {
    const int j = blockIdx.x;
    const float *x = ga.x[j], *m = ga.m[j], *f = gfg + (size_t) j * 2 * H, *g = f + H, *w = ga.nw[j];
    float *y = ga.y[j], *yn = ga.yn[j];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
        y[i] = a;
        ss += a * a;
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) yn[i] = w[i] * (y[i] * r);
}


__device__ __forceinline__ float log_scale_at(int pos) {
    return 1.0f + LOG_ALPHA * logf(fmaxf((float) (pos + 1) / LOG_FLOOR, 1.0f));
}

// head hd of v: v *= scale; L2 clip to thr; (log scale on full layers).  block = 128 threads, dim <= 128
__device__ __forceinline__ void head_clip1(float* v, const float* scale, int dim, float thr, int log_scaled, int pos,
                                           int hd) {
    float* x = v + (size_t) hd * dim;
    const int i = threadIdx.x;
    float a = i < dim ? x[i] * (scale ? scale[(size_t) hd * dim + i] : 1.0f) : 0.0f;
    const float norm = sqrtf(block_sum(a * a));
    a *= fminf(thr / (norm + 1e-6f), 1.0f);
    if (log_scaled) a *= log_scale_at(pos);
    if (i < dim) x[i] = a;
}

// Inkling2LoopShortConvolution: hist = 3 rows (row p % 3 holds position p's input)
__device__ __forceinline__ float short_conv1(float xc, float* __restrict__ hist, const float* __restrict__ w, int C, int c,
                                             int p) {
    float acc = w[(size_t) c * KC + KC - 1] * xc;
    for (int j = 1; j < KC; ++j)
        if (p - j >= 0) acc += w[(size_t) c * KC + KC - 1 - j] * hist[(size_t) ((p - j) % 3) * C + c];
    hist[(size_t) (p % 3) * C + c] = xc;   // after the reads: lag 3 is the row p % 3
    return __fmul_rn(acc + xc, rsqrtf((float) (KC + 1)));   // _rn: no FMA contraction with the caller's residual add
}
// attention tail: conv of the attention output, then either (block boundary) r = y also stored as the next block state,
// or r = h + y also stored over the layer's stack entry
__global__ void k_attn_tail(const float* __restrict__ x, float* __restrict__ hist, const float* __restrict__ w,
                            const float* __restrict__ h, float* __restrict__ r, float* __restrict__ r_copy, int boundary,
                            const int* d_pos) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    const float y = short_conv1(x[c], hist, w, H, c, *d_pos);
    const float v = boundary ? y : h[c] + y;
    r[c] = v;
    r_copy[c] = v;
}
// MoE tail: combine the TOPK expert outputs, conv, residual: out = r + conv(sum_k y_k w_k)
__global__ void k_moe_tail(const float* __restrict__ ey, const float* __restrict__ wts, float* __restrict__ hist,
                           const float* __restrict__ w, const float* __restrict__ r, float* __restrict__ out,
                           const int* d_pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float a = 0.0f;
    for (int k = 0; k < TOPK; ++k) a += ey[(size_t) k * H + i] * wts[k];
    out[i] = r[i] + short_conv1(a, hist, w, H, i, *d_pos);
}

// Inkling2LoopDynamicShortConvolution: taps static_w[lag][c] + dw[(c / 128) * 4 + lag]
__device__ __forceinline__ float dyn_conv1(float xc, const float* __restrict__ dw, float* __restrict__ hist,
                                           const float* __restrict__ static_w, int C, int c, int p) {
    const int head = c / (C / NKV);
    float acc = 0.0f;
    for (int lag = 0; lag < KC; ++lag) {
        const float tap = static_w[(size_t) lag * C + c] + dw[head * KC + lag];
        if (lag == 0) acc += tap * xc;
        else if (p - lag >= 0) acc += tap * hist[(size_t) ((p - lag) % 3) * C + c];
    }
    hist[(size_t) (p % 3) * C + c] = xc;
    return __fmul_rn(acc + xc, rsqrtf((float) (2 * KC + 1)));
}

// KV cache cells [cap][KVD], format c_kvm (--kv): 0 fp32; 1 fp16 (round to nearest); 2 Strata's int8 layout
// (kv_q8.hpp): codes + one fp16 scale per 64 values [cap][KVD / 64] (ksc / vsc); scale = fp16(max|x| / 127),
// code = clamp(rint(x / scale), -127, 127).  K is L2-clipped and V RMS-normalized per head before the store.
constexpr int KVG = 64;
__constant__ int c_kvm;   // 3 = int8 with Walsh-Hadamard rotation (Strata's kv_int8_rotate): K, V, q rotated per head,
                          // the attention output rotated back (H orthonormal and symmetric: scores unchanged)
// orthonormal Walsh-Hadamard transform of one 128-vector held one value per thread by a 128-thread block
__device__ __forceinline__ float wht128(float x) {
    __shared__ float r[128];
    const int t = threadIdx.x;
    __syncthreads();   // r reused across calls
    r[t] = x;
#pragma unroll
    for (int st = 1; st < 128; st <<= 1) {
        __syncthreads();
        const float a = r[t], b = r[t ^ st];
        __syncthreads();
        r[t] = (t & st) ? b - a : a + b;
    }
    __syncthreads();
    return r[t] * 0.08838834764831845f;   // 1 / sqrt(128)
}
__device__ __forceinline__ float4 kv_ld4(const void* B, const __half* sc, size_t row, int e) {
    if (c_kvm == 0) return *reinterpret_cast<const float4*>(static_cast<const float*>(B) + row * KVD + e);
    if (c_kvm == 1) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __half*>(B) + row * KVD + e);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
    const char4 c = *reinterpret_cast<const char4*>(static_cast<const int8_t*>(B) + row * KVD + e);
    const float f = __half2float(sc[row * (KVD / KVG) + e / KVG]);
    return make_float4(c.x * f, c.y * f, c.z * f, c.w * f);
}
__device__ __forceinline__ float kv_ld1(const void* B, const __half* sc, size_t row, int e) {
    if (c_kvm == 0) return static_cast<const float*>(B)[row * KVD + e];
    if (c_kvm == 1) return __half2float(static_cast<const __half*>(B)[row * KVD + e]);
    return static_cast<const int8_t*>(B)[row * KVD + e] * __half2float(sc[row * (KVD / KVG) + e / KVG]);
}
// one cell value, called by every thread of a 128-thread block, thread i holding element e = g * HD + i (warps 0-1:
// first group of 64, warps 2-3: second)
__device__ __forceinline__ void kv_st(void* B, __half* sc, size_t row, int e, float x) {
    if (c_kvm == 0) { static_cast<float*>(B)[row * KVD + e] = x; return; }
    if (c_kvm == 1) { static_cast<__half*>(B)[row * KVD + e] = __float2half_rn(x); return; }
    if (c_kvm == 3) x = wht128(x);
    __shared__ float gm[4];
    const float m = warp_max(fabsf(x));
    __syncthreads();   // gm is reused by the next call
    if (!(threadIdx.x & 31)) gm[threadIdx.x >> 5] = m;
    __syncthreads();
    const int w = threadIdx.x >> 5;
    const __half sh = __float2half_rn(fmaxf(gm[w & ~1], gm[w | 1]) / 127.0f);
    const float f = __half2float(sh);
    const int q = f > 0.0f ? max(-127, min(127, __float2int_rn(x / f))) : 0;
    static_cast<int8_t*>(B)[row * KVD + e] = (int8_t) q;
    if (!(threadIdx.x & (KVG - 1))) sc[row * (KVD / KVG) + e / KVG] = sh;
}

// keys: L2 clip per KV head; values: unweighted RMS per head; stored at position pos (ring: pos % cap)
// k / v convolutions (static or dynamic taps) fused with the store; grid NKV, block HD
template <bool DYN>
__device__ __forceinline__ void conv_kv_store1(const float* k, const float* v, float* khist, float* vhist, const float* kw,
                                               const float* vw, void* K, void* V, __half* ksc, __half* vsc, int cap,
                                               const int* d_pos, int g) {
    const int i = threadIdx.x, c = g * HD + i, p = *d_pos;
    const float kv = DYN ? dyn_conv1(k[c], k + KVD, khist, kw, KVD, c, p) : short_conv1(k[c], khist, kw, KVD, c, p);
    const float vv = DYN ? dyn_conv1(v[c], v + KVD, vhist, vw, KVD, c, p) : short_conv1(v[c], vhist, vw, KVD, c, p);
    const float kn = sqrtf(block_sum(kv * kv));
    const float vr = rsqrtf(block_sum(vv * vv) / HD + EPS);
    const size_t row = (size_t) ((*d_pos) % cap);
    kv_st(K, ksc, row, c, kv * fminf(QK_CLIP / (kn + 1e-6f), 1.0f));
    kv_st(V, vsc, row, c, vv * vr);
}

// Attention prep, one launch (block = 128): blocks [0, NQ) clip q head b; [NQ, 2 NQ) clip rs head h then
// rel[h][d] = sum_r rs[h][r] proj[r][d]; [2 NQ, 2 NQ + NKV) conv + clip/rms + store of k / v group g.  Same arithmetic as
// the former four kernels (k_head_clip x2, k_rel_logits, k_conv_kv_store).
template <bool DYN>
__global__ void k_attn_prep(float* q, const float* q_scale, float* rs, const float* rel_scale, const float* rel_proj,
                            float* rel, int log_scaled, const float* k, const float* v, float* khist, float* vhist,
                            const float* kw, const float* vw, void* K, void* V, __half* ksc, __half* vsc, int cap,
                            const int* d_pos) {
    const int b = blockIdx.x;
    if (b < NQ) {
        head_clip1(q, q_scale, HD, QK_CLIP, log_scaled, *d_pos, b);
        if (c_kvm == 3) q[b * HD + threadIdx.x] = wht128(q[b * HD + threadIdx.x]);
    } else if (b < 2 * NQ) {
        const int h = b - NQ;
        head_clip1(rs, rel_scale, DREL, REL_CLIP, log_scaled, *d_pos, h);
        __syncthreads();   // the clipped rs of head h, written by this block
        for (int d = threadIdx.x; d < EXTENT; d += blockDim.x) {
            float a = 0.0f;
#pragma unroll
            for (int r = 0; r < DREL; ++r) a += rs[h * DREL + r] * rel_proj[r * EXTENT + d];
            rel[h * EXTENT + d] = a;
        }
    } else {
        conv_kv_store1<DYN>(k, v, khist, vhist, kw, vw, K, V, ksc, vsc, cap, d_pos, b - 2 * NQ);
    }
}

// partial attention: block (head h, chunk c) covers distances [c*CHUNK, c*CHUNK + CHUNK) back from pos; window 0 = full.
// writes m, l (sum of exp) and o[HD] (unnormalized) for the merge.  block = 128 threads (4 warps).
__global__ void k_attn_partial(const float* __restrict__ q, const void* __restrict__ K, const void* __restrict__ V,
                               const __half* __restrict__ ksc, const __half* __restrict__ vsc, const float* __restrict__ rel, int cap, int window, const int* d_pos,
                               float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ po) {
    __shared__ __align__(16) float qs[HD];
    __shared__ float lg[CHUNK];
    const int h = blockIdx.x, c = blockIdx.y, g = h / NREP, tid = threadIdx.x, lane = tid & 31, wp = tid / 32;
    const int pos = *d_pos;
    const int span = window > 0 ? min(pos + 1, window) : pos + 1;   // distances 0 .. span-1
    const int d0 = c * CHUNK, n = min(CHUNK, span - d0);
    const size_t part = (size_t) h * gridDim.y + c;
    if (n <= 0) {
        if (!tid) { pm[part] = -INFINITY; pl[part] = 0.0f; }
        po[part * HD + tid] = 0.0f;
        return;
    }
    qs[tid] = q[h * HD + tid];
    __syncthreads();
    const float scale = rsqrtf((float) HD);
    // ring slot of distance d0 + k: r0 - k, wrapped once (k < CHUNK <= cap)
    const int r0 = (pos - d0) % cap;
    // q.k: 8 lanes per position (4 positions per warp, 16 per block step), each lane 4 float4 of the row
    {
        const int sub = lane & 7;
        const float4* q4 = reinterpret_cast<const float4*>(qs);
        for (int k0 = wp * 4; k0 < n; k0 += 16) {
            const int k = k0 + (lane >> 3);
            float a = 0.0f;
            if (k < n) {
                const int r = r0 - k < 0 ? r0 - k + cap : r0 - k;
                float4 kv[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) kv[u] = kv_ld4(K, ksc, (size_t) r, g * HD + 4 * (sub + 8 * u));
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float4 qv = q4[sub + 8 * u];
                    a += qv.x * kv[u].x + qv.y * kv[u].y + qv.z * kv[u].z + qv.w * kv[u].w;
                }
            }
            a += __shfl_xor_sync(0xffffffffu, a, 4);
            a += __shfl_xor_sync(0xffffffffu, a, 2);
            a += __shfl_xor_sync(0xffffffffu, a, 1);
            const int dist = d0 + k;
            if (!sub && k < n) lg[k] = a * scale + (dist < EXTENT ? rel[h * EXTENT + dist] : 0.0f);
        }
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int k = tid; k < n; k += blockDim.x) mx = fmaxf(mx, lg[k]);
    mx = block_max(mx);
    float s = 0.0f;
    for (int k = tid; k < n; k += blockDim.x) { lg[k] = expf(lg[k] - mx); s += lg[k]; }
    s = block_sum(s);   // block_sum syncs, lg visible after
    __syncthreads();
    // p.V: warp wp takes positions wp, wp + 4, ...; lane holds dims 4 lane .. 4 lane + 3; the 4 partial rows summed
    // through smem (positions spread over the warps: 4x more loads in flight than one position per block step)
    __shared__ float4 red[4][HD / 4];
    float4 o4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int ve = g * HD + 4 * lane;
#pragma unroll 4
    for (int k = wp; k < n; k += 4) {
        const int r = r0 - k < 0 ? r0 - k + cap : r0 - k;
        const float4 v = kv_ld4(V, vsc, (size_t) r, ve);
        const float pk = lg[k];
        o4.x += pk * v.x; o4.y += pk * v.y; o4.z += pk * v.z; o4.w += pk * v.w;
    }
    red[wp][lane] = o4;
    __syncthreads();
    const float* rf = reinterpret_cast<const float*>(red);
    const float o = rf[tid] + rf[HD + tid] + rf[2 * HD + tid] + rf[3 * HD + tid];
    if (!tid) { pm[part] = mx; pl[part] = s; }
    po[part * HD + tid] = o;
}

// merge the chunks of each head with its sink logit (zero value); out[h][HD]
__global__ void k_attn_merge(const float* pm, const float* pl, const float* po, int nchunk, const float* sink,
                             const float* gate, float* out) {
    const int h = blockIdx.x, tid = threadIdx.x;
    float M = sink[h];
    for (int c = 0; c < nchunk; ++c) M = fmaxf(M, pm[h * nchunk + c]);
    float L = expf(sink[h] - M), o = 0.0f;
    for (int c = 0; c < nchunk; ++c) {
        const float m = pm[h * nchunk + c];
        if (m == -INFINITY) continue;
        const float f = expf(m - M);
        L += f * pl[h * nchunk + c];
        o += f * po[((size_t) h * nchunk + c) * HD + tid];
    }
    float a = o / L;   // then the output gate: a * sigmoid(g)
    if (c_kvm == 3) a = wht128(a);   // rotated V: the output comes back with the same transform
    out[h * HD + tid] = a * (1.0f / (1.0f + expf(-gate[h * HD + tid])));
}


// rotate head blockIdx.x of row blockIdx.y in place (--kv i8r): prefill q before attention, output after; block 128
__global__ void kb_wht(float* x, int stride) {
    float* v = x + (size_t) blockIdx.y * stride + blockIdx.x * HD;
    v[threadIdx.x] = wht128(v[threadIdx.x]);
}

// router: sigmoid scores, top-8 on score + bias, weights score / sum * 8.  one block of NE threads
__global__ void k_router(const float* logits, const float* bias, int* ids, float* wts) {
    logits += (size_t) blockIdx.x * NE;
    ids += blockIdx.x * TOPK;
    wts += blockIdx.x * TOPK;
    __shared__ float sel[NE], sc[NE];
    __shared__ int pick[TOPK];
    const int e = threadIdx.x;
    sc[e] = 1.0f / (1.0f + expf(-logits[e]));
    sel[e] = sc[e] + bias[e];
    if (e < TOPK) pick[e] = e;   // only NaN scores leave a rank unfilled: keep the ids in range then
    __syncthreads();
    // rank of e = number of experts ahead of it (higher score, ties: lower index); ranks 0..TOPK-1 are the picks in
    // the order successive argmaxes (ties to the lowest index) would give them
    const float me = sel[e];
    int rank = 0;
    for (int i = 0; i < NE; ++i) {
        const float o = sel[i];
        rank += (o > me) || (o == me && i < e);
    }
    if (rank < TOPK) pick[rank] = e;
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < TOPK; ++k) sum += sc[pick[k]];
        for (int k = 0; k < TOPK; ++k) { ids[k] = pick[k]; wts[k] = sc[pick[k]] / (sum + 1e-20f) * ROUTE_SCALE; }
    }
}



__global__ void k_softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill kernels: T tokens at once (positions pos0 .. pos0 + T - 1), activations row-major [T][dim] with a row stride.
// Same fp32 arithmetic as the one-token kernels; only the summation order of the matrix products differs (cuBLAS SGEMM,
// full FP32, no TF32).

__global__ void kb_bf16_to_f32(const uint16_t* __restrict__ src, float* __restrict__ dst, size_t n8) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(src) + i);
    float4* d = reinterpret_cast<float4*>(dst) + 2 * i;
    d[0] = make_float4(bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y));
    d[1] = make_float4(bf16lo(v.z), bf16hi(v.z), bf16lo(v.w), bf16hi(v.w));
}

// [rms(x_t), rms(m_t)] -> cat[t][2H]; grid (2, T)
__global__ void kb_rms2(const float* __restrict__ x, int sx, const float* __restrict__ m, int sm, float* __restrict__ cat) {
    const int t = blockIdx.y;
    const float* a = blockIdx.x ? m + (size_t) t * sm : x + (size_t) t * sx;
    float* b = cat + (size_t) t * 2 * H + (size_t) blockIdx.x * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += a[i] * a[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) b[i] = a[i] * r;
}

// yn_t = rms((1 - tanh f) * x + tanh g * m) * w; fg [T][2H]; grid T
__global__ void kb_gate_mix_rms(const float* x, int sx, const float* m, int sm, const float* fg, const float* w,
                                float* yn) {
    const int t = blockIdx.x;
    x += (size_t) t * sx;
    m += (size_t) t * sm;
    const float* f = fg + (size_t) t * 2 * H;
    const float* g = f + H;
    yn += (size_t) t * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
        ss += a * a;
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + EPS);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float a = (1.0f - tanhf(f[i])) * x[i] + tanhf(g[i]) * m[i];
        yn[i] = w[i] * (a * r);
    }
}

// per head and token: v *= scale; L2 clip; (log scale).  grid (heads, T), block 128
__global__ void kb_head_clip(float* v, int sv, const float* scale, int dim, float thr, int log_scaled, int pos0) {
    float* x = v + (size_t) blockIdx.y * sv + (size_t) blockIdx.x * dim;
    const int i = threadIdx.x;
    float a = i < dim ? x[i] * (scale ? scale[(size_t) blockIdx.x * dim + i] : 1.0f) : 0.0f;
    const float norm = sqrtf(block_sum(a * a));
    a *= fminf(thr / (norm + 1e-6f), 1.0f);
    if (log_scaled) a *= log_scale_at(pos0 + blockIdx.y);
    if (i < dim) x[i] = a;
}

// causal depthwise conv over the chunk; inputs before the chunk come from hist (row p % 3).  grid (C / 256, T).
// dw (dynamic taps, full layers): per token at dw + t * sx, [NKV][KC]; static_w then [KC][C]; else w [C][KC]
__global__ void kb_conv(const float* __restrict__ x, int sx, const float* __restrict__ hist, const float* __restrict__ w,
                        const float* __restrict__ dw, float* __restrict__ y, int sy, int C, int pos0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, p = pos0 + t;
    if (c >= C) return;
    const float xc = x[(size_t) t * sx + c];
    float acc;
    if (dw) {
        const float* d = dw + (size_t) t * sx + (c / (C / NKV)) * KC;
        acc = (w[c] + d[0]) * xc;
        for (int lag = 1; lag < KC; ++lag) {
            if (p - lag < 0) break;
            const float v = t - lag >= 0 ? x[(size_t) (t - lag) * sx + c] : hist[(size_t) ((p - lag) % 3) * C + c];
            acc += (w[(size_t) lag * C + c] + d[lag]) * v;
        }
        y[(size_t) t * sy + c] = (acc + xc) * rsqrtf((float) (2 * KC + 1));
    } else {
        acc = w[(size_t) c * KC + KC - 1] * xc;
        for (int j = 1; j < KC; ++j) {
            if (p - j < 0) break;
            const float v = t - j >= 0 ? x[(size_t) (t - j) * sx + c] : hist[(size_t) ((p - j) % 3) * C + c];
            acc += w[(size_t) c * KC + KC - 1 - j] * v;
        }
        y[(size_t) t * sy + c] = (acc + xc) * rsqrtf((float) (KC + 1));
    }
}
// after kb_conv: the chunk's last (up to 3) inputs into hist.  grid (C / 256, min(3, T))
__global__ void kb_hist_store(const float* __restrict__ x, int sx, float* __restrict__ hist, int C, int T, int pos0) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = T - 1 - blockIdx.y;
    if (c < C) hist[(size_t) ((pos0 + t) % 3) * C + c] = x[(size_t) t * sx + c];
}

// grid (NKV, T), block HD
__global__ void kb_kv_store(const float* kc, const float* vc, void* K, void* V, __half* ksc, __half* vsc, int cap,
                            int pos0) {
    const int g = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const float kv = kc[(size_t) t * KVD + g * HD + i], vv = vc[(size_t) t * KVD + g * HD + i];
    const float kn = sqrtf(block_sum(kv * kv));
    const float vr = rsqrtf(block_sum(vv * vv) / HD + EPS);
    const size_t row = (size_t) ((pos0 + t) % cap);
    kv_st(K, ksc, row, g * HD + i, kv * fminf(QK_CLIP / (kn + 1e-6f), 1.0f));
    kv_st(V, vsc, row, g * HD + i, vv * vr);
}

// rel[t][h][d] = sum_r rs_t[h][r] proj[r][d]; grid (NQ, T)
__global__ void kb_rel_logits(const float* rs, int sr, const float* proj, float* rel) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* r0 = rs + (size_t) t * sr + h * DREL;
    float* o = rel + ((size_t) t * NQ + h) * EXTENT;
    for (int d = threadIdx.x; d < EXTENT; d += blockDim.x) {
        float a = 0.0f;
#pragma unroll
        for (int r = 0; r < DREL; ++r) a += r0[r] * proj[r * EXTENT + d];
        o[d] = a;
    }
}

// Tiled causal attention, fp32: block = (64 queries, one q head), 256 threads; key tiles of 64 from the cache (ring
// index j % cap).  Online softmax started from the sink (m = sink, l = 1, o = 0: its value is zero).
// Thread (ty = tid / 16, tx = tid % 16): queries ty*4 .. +3; scores for keys tx + 16 m; outputs dims tx + 16 m.
constexpr int FQ = 64, FK = 64, FQ_LD = HD + 4, FK_LD = HD + 1, FP_LD = FK + 1;
constexpr size_t FA_SMEM = (size_t) (FQ * FQ_LD + FK * FK_LD + FQ * FP_LD) * 4;
__global__ void __launch_bounds__(256) kb_attn(const float* __restrict__ P, int sp, const void* __restrict__ K,
                                               const void* __restrict__ V, const __half* __restrict__ ksc,
                                               const __half* __restrict__ vsc, const float* __restrict__ rel, int cap,
                                               int window, int pos0, int T, float* __restrict__ out) {
    extern __shared__ float fsm[];
    float* Qs = fsm;                    // [FQ][FQ_LD]
    float* KVs = Qs + FQ * FQ_LD;       // keys [FK][FK_LD], then values [FK][HD]
    float* Ps = KVs + FK * FK_LD;       // [FQ][FP_LD]
    const int h = blockIdx.y, g = h / NREP, t0 = blockIdx.x * FQ, tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    const int nq = min(FQ, T - t0);
    for (int i = tid; i < FQ * HD; i += 256) {
        const int r = i / HD, d = i % HD;
        Qs[r * FQ_LD + d] = r < nq ? P[(size_t) (t0 + r) * sp + h * HD + d] : 0.0f;
    }
    float m[4], l[4], o[4][8];
    const float scale = rsqrtf((float) HD);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty * 4 + i;
        m[i] = r < nq ? P[(size_t) (t0 + r) * sp + OFF_SINK + h] : 0.0f;
        l[i] = 1.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[i][j] = 0.0f;
    }
    const int pa = pos0 + t0, pb = pos0 + t0 + nq - 1;   // first / last query position
    const int lo = window > 0 ? max(0, pa - window + 1) : 0;
    for (int k0 = lo; k0 <= pb; k0 += FK) {
        __syncthreads();
        for (int i = tid; i < FK * HD; i += 256) {
            const int r = i / HD, d = i % HD, j = k0 + r;
            KVs[r * FK_LD + d] = j <= pb ? kv_ld1(K, ksc, (size_t) (j % cap), g * HD + d) : 0.0f;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) s[i][c] = 0.0f;
#pragma unroll 4
        for (int d = 0; d < HD; ++d) {
            float q[4], k[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) q[i] = Qs[(ty * 4 + i) * FQ_LD + d];
#pragma unroll
            for (int c = 0; c < 4; ++c) k[c] = KVs[(tx + 16 * c) * FK_LD + d];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) s[i][c] += q[i] * k[c];
        }
        // bias, mask, online softmax (a query row lives in the 16 threads of a half warp)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = ty * 4 + i, p = pos0 + t0 + r;
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int j = k0 + tx + 16 * c, dist = p - j;
                const bool ok = r < nq && j <= p && (window == 0 || dist < window);
                s[i][c] = ok ? s[i][c] * scale + (dist < EXTENT ? rel[((size_t) (t0 + r) * NQ + h) * EXTENT + dist] : 0.0f)
                             : -INFINITY;
                mx = fmaxf(mx, s[i][c]);
            }
            for (int w = 8; w; w >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, w));
            const float mn = fmaxf(m[i], mx), f = expf(m[i] - mn);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float e = expf(s[i][c] - mn);
                Ps[r * FP_LD + tx + 16 * c] = e;
                sum += e;
            }
            for (int w = 8; w; w >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, w);
            l[i] = l[i] * f + sum;
            m[i] = mn;
#pragma unroll
            for (int j = 0; j < 8; ++j) o[i][j] *= f;
        }
        __syncthreads();
        for (int i = tid; i < FK * HD; i += 256) {
            const int r = i / HD, d = i % HD, j = k0 + r;
            KVs[r * HD + d] = j <= pb ? kv_ld1(V, vsc, (size_t) (j % cap), g * HD + d) : 0.0f;
        }
        __syncthreads();
        const int kn = min(FK, pb - k0 + 1);
        for (int k = 0; k < kn; ++k) {
            float pv[4], v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) pv[i] = Ps[(ty * 4 + i) * FP_LD + k];
#pragma unroll
            for (int j = 0; j < 8; ++j) v[j] = KVs[k * HD + tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) o[i][j] += pv[i] * v[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty * 4 + i;
        if (r >= nq) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) out[(size_t) (t0 + r) * QD + h * HD + tx + 16 * j] = o[i][j] / l[i];
    }
}

// att[t][i] *= sigmoid(gate_t[i]); grid (QD / 256, T)
__global__ void kb_mul_sigmoid(float* att, const float* gate, int sg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    att[(size_t) t * QD + i] *= 1.0f / (1.0f + expf(-gate[(size_t) t * sg + i]));
}

// MoE plan (one block, thread = expert): rows of the grouped products sorted by expert then (token, k);
// rows[r] = token, slot[t * TOPK + k] = r; tiles of FM rows within one expert.
constexpr int FM = 64, FN = 64, FKK = 32;
// (one block of 1024 threads; the order of the rows within an expert is not deterministic and does not matter: each
// output row depends only on its own input row)
__global__ void kb_moe_plan(const int* __restrict__ ids, int T, int* rows, int* slot, int* tile_e, int* tile_r0,
                            int* ntiles, int* eoff, int* ecnt) {
    __shared__ int cnt[NE], off[NE], cur[NE];
    const int n = T * TOPK;
    for (int e = threadIdx.x; e < NE; e += blockDim.x) { cnt[e] = 0; cur[e] = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int j = 0; j < NE; ++j) {
            off[j] = o;
            for (int r = 0; r < cnt[j]; r += FM) { tile_e[nt] = j; tile_r0[nt] = o + r; ++nt; }
            o += cnt[j];
        }
        *ntiles = nt;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NE; e += blockDim.x) { eoff[e] = off[e]; ecnt[e] = cnt[e]; }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int e = ids[i], r = off[e] + atomicAdd(&cur[e], 1);
        rows[r] = i / TOPK;
        slot[i] = r;
    }
}

// Grouped expert product with the weights decoded on the fly (exact: e2m1 * e4m3 and e4m3 * 2^k are exact in fp32):
// C[r][n] = (factor_e) * sum_k A[arow(r)][k] W_e[n][k] for the rows r of a tile (expert e).  grid (max tiles, N / FN),
// 256 threads, 4x4 outputs per thread.  arow = rows[r] (gather) or r.
template <bool NV>
__global__ void __launch_bounds__(256) kb_moe_gemm(const float* __restrict__ A, int lda, const int* __restrict__ rows,
                                                   const uint8_t* const* __restrict__ tab,
                                                   const float* __restrict__ factors, const int* __restrict__ tile_e,
                                                   const int* __restrict__ tile_r0, const int* __restrict__ ntiles,
                                                   const int* __restrict__ eoff, const int* __restrict__ ecnt,
                                                   float* __restrict__ C, int N, int K) {
    __shared__ __align__(16) float As[FKK][FM + 4], Ws[FKK][FN + 4];
    if ((int) blockIdx.x >= *ntiles) return;
    const int e = tile_e[blockIdx.x], r0 = tile_r0[blockIdx.x], rend = eoff[e] + ecnt[e], n0 = blockIdx.y * FN;
    const uint8_t* W = tab[e];
    if (!W) return;
    const uint8_t* S = W + (size_t) N * (NV ? K / 2 : K);
    const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    // A loads: 2 float4 per thread (row idx / 8, float4 idx % 8)
    const float* asrc[2];
    int arow_l[2], akq[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int idx = tid + 256 * u, row = idx / 8, r = r0 + row;
        arow_l[u] = row;
        akq[u] = idx % 8;
        asrc[u] = r < rend ? A + (size_t) (rows ? rows[r] : r) * lda + akq[u] * 4 : nullptr;
    }
    const int wn = tid / 4, wq = tid % 4;   // W loads: row n0 + wn, 8 columns from wq * 8
    const size_t wrow = (size_t) n0 + wn;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += FKK) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const float4 v = asrc[u] ? *reinterpret_cast<const float4*>(asrc[u] + k0) : make_float4(0, 0, 0, 0);
            const int kk = akq[u] * 4, row = arow_l[u];
            As[kk][row] = v.x; As[kk + 1][row] = v.y; As[kk + 2][row] = v.z; As[kk + 3][row] = v.w;
        }
        const int kc = k0 + wq * 8;
        if (NV) {
            const uint32_t v = __ldg(reinterpret_cast<const uint32_t*>(W + wrow * (K / 2) + kc / 2));
            const float sc = e4m3_decode(__ldg(S + wrow * (K / 16) + kc / 16)) * 16384.0f;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float2 p = e2m1x2((v >> (8 * b)) & 0xff);
                Ws[wq * 8 + 2 * b][wn] = p.x * sc;
                Ws[wq * 8 + 2 * b + 1][wn] = p.y * sc;
            }
        } else {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(W + wrow * K + kc));
            const float sc = exp2f((float) __ldg(S + wrow * (K / 32) + k0 / 32) - 119.0f);
            const uint32_t wd[2] = {v.x, v.y};
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const float2 p01 = e4m3x2((wd[b] & 0xffu) | ((wd[b] & 0xff00u) << 8));
                const float2 p23 = e4m3x2(((wd[b] >> 16) & 0xffu) | ((wd[b] & 0xff000000u) >> 8));
                Ws[wq * 8 + 4 * b][wn] = p01.x * sc;
                Ws[wq * 8 + 4 * b + 1][wn] = p01.y * sc;
                Ws[wq * 8 + 4 * b + 2][wn] = p23.x * sc;
                Ws[wq * 8 + 4 * b + 3][wn] = p23.y * sc;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < FKK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Ws[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    const float f = NV ? factors[e] : 1.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + ty * 4 + i;
        if (r < rend)
            *reinterpret_cast<float4*>(C + (size_t) r * N + n0 + tx * 4) =
                make_float4(acc[i][0] * f, acc[i][1] * f, acc[i][2] * f, acc[i][3] * f);
    }
}

// --- tensor-core variant (--tc 1): FP16 inputs, FP32 accumulation.  NVFP4 expert weights are exact in fp16
// (e2m1 * e4m3 <= 2688, >= 2^-10, <= 6 significant bits; the per-expert factor stays an fp32 multiplier); the
// activations are rounded to fp16 (2^-11 relative).  Residual stream, norms, router, gates, attention stay fp32.
// MXFP8 (layers 20-23): e4m3 * 2^(s-127) with s in [113, 118] in this checkpoint (measured, mx_range.py): multiples of
// 2^-23 below 1, exact in fp16; stored * 2^10 (clear of the subnormals), the 2^-10 back as the output factor.
constexpr int TM = 64, TN = 128, TK = 32, TLD = TK + 8, TCLD = TN + 4;
template <bool NV>
__global__ void __launch_bounds__(256) kb_moe_gemm_tc(const half* __restrict__ A, int lda,
                                                      const uint8_t* const* __restrict__ tab,
                                                      const float* __restrict__ factors, const int* __restrict__ tile_e,
                                                      const int* __restrict__ tile_r0, const int* __restrict__ ntiles,
                                                      const int* __restrict__ eoff, const int* __restrict__ ecnt,
                                                      float* __restrict__ C, int N, int K) {
    __shared__ __align__(32) unsigned char smraw[TM * TCLD * 4];   // As + Bs during the loop, C staging after
    half* As = reinterpret_cast<half*>(smraw);
    half* Bs = As + TM * TLD;
    float* Cs = reinterpret_cast<float*>(smraw);
    if ((int) blockIdx.x >= *ntiles) return;
    const int e = tile_e[blockIdx.x], r0 = tile_r0[blockIdx.x], rend = eoff[e] + ecnt[e], n0 = blockIdx.y * TN;
    const uint8_t* W = tab[e];
    if (!W) return;
    const uint8_t* S = W + (size_t) N * (NV ? K / 2 : K);
    const int tid = threadIdx.x, warp = tid / 32, wm = warp / 4, wn = warp % 4;
    // A: fp16 rows in expert order (gathered / converted once per layer, not once per N block): 16 B per thread
    const int arow = tid / 4, akq = tid % 4;
    const half* asrc = r0 + arow < rend ? A + (size_t) (r0 + arow) * lda + akq * 8 : nullptr;
    int wn_[2], wq_[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int idx = tid + 256 * u;
        wn_[u] = idx / 4;
        wq_[u] = idx % 4;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(c[i][j], 0.0f);
    // register prefetch: tile k0 + TK is loaded from global while tile k0 (in smem) goes through the MMAs.  The
    // conversions and the MMA order are unchanged (bit-identical to the single-buffered loop).
    uint4 av;
    uint2 wv[2];
    uint8_t sv[2];
    auto gload = [&](int k0) {
        av = asrc ? *reinterpret_cast<const uint4*>(asrc + k0) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const size_t wrow = (size_t) n0 + wn_[u];
            const int kc = k0 + wq_[u] * 8;
            if (NV) {
                wv[u].x = __ldg(reinterpret_cast<const uint32_t*>(W + wrow * (K / 2) + kc / 2));
                sv[u] = __ldg(S + wrow * (K / 16) + kc / 16);
            } else {
                wv[u] = __ldg(reinterpret_cast<const uint2*>(W + wrow * K + kc));
                sv[u] = __ldg(S + wrow * (K / 32) + k0 / 32);
            }
        }
    };
    gload(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        *reinterpret_cast<uint4*>(As + arow * TLD + akq * 8) = av;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            half2 h[4];
            if (NV) {
                const uint32_t v = wv[u].x;
                const float sc = e4m3_decode(sv[u]) * 16384.0f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float2 p = e2m1x2((v >> (8 * b)) & 0xff);
                    h[b] = __floats2half2_rn(p.x * sc, p.y * sc);
                }
            } else {
                const float sc = exp2f((float) sv[u] - 109.0f);   // 2^(s-127) 2^8 2^10
                const uint32_t wd[2] = {wv[u].x, wv[u].y};
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const float2 p01 = e4m3x2((wd[b] & 0xffu) | ((wd[b] & 0xff00u) << 8));
                    const float2 p23 = e4m3x2(((wd[b] >> 16) & 0xffu) | ((wd[b] & 0xff000000u) >> 8));
                    h[2 * b] = __floats2half2_rn(p01.x * sc, p01.y * sc);
                    h[2 * b + 1] = __floats2half2_rn(p23.x * sc, p23.y * sc);
                }
            }
            *reinterpret_cast<uint4*>(Bs + wn_[u] * TLD + wq_[u] * 8) = *reinterpret_cast<uint4*>(h);
        }
        __syncthreads();
        if (k0 + TK < K) gload(k0 + TK);
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * TLD + kk, TLD);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bs + (wn * 32 + j * 16) * TLD + kk, TLD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * TCLD + wn * 32 + j * 16, c[i][j], TCLD, wmma::mem_row_major);
    __syncthreads();
    const float f = NV ? factors[e] : 1.0f / 1024.0f;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int idx = tid + 256 * u, row = idx / 32, c4 = idx % 32, r = r0 + row;
        if (r >= rend) continue;
        const float4 v = *reinterpret_cast<const float4*>(Cs + row * TCLD + c4 * 4);
        *reinterpret_cast<float4*>(C + (size_t) r * N + n0 + c4 * 4) = make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
    }
}

// fp32 [T][K] (row stride ld) -> fp16 [T][K]; bf16 -> fp16 (dense weights: exact for |w| >= 2^-14, the rare smaller
// ones lose low bits as fp16 subnormals)
__global__ void kb_f32_to_f16(const float* __restrict__ x, int ld, int K, half* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / K, k = i % K;
    y[i] = __float2half_rn(x[t * ld + k]);
}
__global__ void kb_bf16_to_f16(const uint16_t* __restrict__ src, half* __restrict__ dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(__uint_as_float((uint32_t) src[i] << 16));
}

// Tensor-core attention (--tc 1): block = 64 queries x one q head, 4 warps (16 query rows each), key tiles of 32.
// Q (fp16, prepared in qh [T][QD]) stays in matrix_a fragments; K / V tiles converted fp32 -> fp16 into smem;
// S = Q K^T (fp32 accumulation) goes through smem for scale + bias + mask + online softmax (fp32), P in fp16;
// O stays in accumulator fragments, rescaled per row with a row map of the fragment elements found at start
// (store_matrix_sync of known values: the element -> row mapping is not specified by the API, so it is measured).
constexpr int AQ = 64, AK = 32, ALD = HD + 8, SLD = AK + 4, PLD = AK + 8, AW = 4 * NREP;
static_assert(16 * PLD * 2 <= 16 * SLD * 4, "P aliases S");
__global__ void __launch_bounds__(128 * NREP, 1) kb_attn_tc(const half* __restrict__ qh, const float* __restrict__ P, int sp,
                                                  const void* __restrict__ K, const void* __restrict__ V,
                                                  const __half* __restrict__ ksc, const __half* __restrict__ vsc,
                                                  const float* __restrict__ rel, int cap, int window, int pos0, int T,
                                                  float* __restrict__ out) {
    // GQA group per block: the NREP q heads of kv head g share each K / V tile (loaded and converted once);
    // warp w: head g * NREP + w / 4, query rows t0 + (w % 4) * 16.  Per warp the computation is unchanged.
    // P (fp16) overwrites the warp's own S rows once read; the row map lives in the K / V tile before the loop.
    __shared__ __align__(32) half KVs[2 * AK * ALD];
    __shared__ __align__(32) float Ss[AW][16 * SLD];
    __shared__ float ms[AW][16], ls[AW][16], fs[AW][16];
    static_assert(AW * 256 * 4 <= 2 * AK * ALD * 2, "row map fits in the K / V tile");
    half* Ks = KVs;
    half* Vs = KVs + AK * ALD;
    const int g = blockIdx.y, t0 = blockIdx.x * AQ, tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int h = g * NREP + w / 4;
    const int tw = t0 + (w % 4) * 16;   // this warp's first query row
    half* Pw = reinterpret_cast<half*>(Ss[w]);
    int* rmw = reinterpret_cast<int*>(KVs) + w * 256;
    using FA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FBc = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
    using FBr = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    // row map of accumulator elements
    FC probe;
    const int ne = probe.num_elements;
    for (int i = 0; i < ne; ++i) probe.x[i] = (float) (lane * ne + i);
    wmma::store_matrix_sync(Ss[w], probe, SLD, wmma::mem_row_major);
    __syncwarp();
    for (int cell = lane; cell < 256; cell += 32) {
        const int v = (int) Ss[w][(cell / 16) * SLD + cell % 16];
        rmw[v] = cell / 16;
    }
    __syncwarp();
    int rowof[8];
    for (int i = 0; i < ne && i < 8; ++i) rowof[i] = rmw[lane * ne + i];
    FA qa[HD / 16];
#pragma unroll
    for (int kd = 0; kd < HD / 16; ++kd) wmma::load_matrix_sync(qa[kd], qh + (size_t) tw * QD + h * HD + kd * 16, QD);
    FC o[HD / 16];
#pragma unroll
    for (int n = 0; n < HD / 16; ++n) wmma::fill_fragment(o[n], 0.0f);
    if (lane < 16) {
        const int r = tw + lane;
        ms[w][lane] = r < T ? P[(size_t) r * sp + OFF_SINK + h] : 0.0f;
        ls[w][lane] = 1.0f;
    }
    const float scale = rsqrtf((float) HD);
    const int nq = min(AQ, T - t0);
    const int pa = pos0 + t0, pb = pos0 + t0 + nq - 1;
    const int lo = window > 0 ? max(0, pa - window + 1) : 0;
    for (int k0 = lo; k0 <= pb; k0 += AK) {
        __syncthreads();
        for (int i = tid; i < AK * HD / 4; i += 128 * NREP) {   // float4 units
            const int r = i / (HD / 4), d = (i % (HD / 4)) * 4, j = k0 + r;
            float4 kv = make_float4(0, 0, 0, 0), vv = kv;
            if (j <= pb) {
                kv = kv_ld4(K, ksc, (size_t) (j % cap), g * HD + d);
                vv = kv_ld4(V, vsc, (size_t) (j % cap), g * HD + d);
            }
            half2 a[2] = {__floats2half2_rn(kv.x, kv.y), __floats2half2_rn(kv.z, kv.w)};
            half2 b[2] = {__floats2half2_rn(vv.x, vv.y), __floats2half2_rn(vv.z, vv.w)};
            *reinterpret_cast<uint2*>(Ks + r * ALD + d) = *reinterpret_cast<uint2*>(a);
            *reinterpret_cast<uint2*>(Vs + r * ALD + d) = *reinterpret_cast<uint2*>(b);
        }
        __syncthreads();
        if (tw >= T) continue;   // whole warp past the chunk (still joins the block barriers)
        FC s[AK / 16];
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) {
            wmma::fill_fragment(s[n], 0.0f);
#pragma unroll
            for (int kd = 0; kd < HD / 16; ++kd) {
                FBc kb;
                wmma::load_matrix_sync(kb, Ks + n * 16 * ALD + kd * 16, ALD);
                wmma::mma_sync(s[n], qa[kd], kb, s[n]);
            }
            wmma::store_matrix_sync(Ss[w] + n * 16, s[n], SLD, wmma::mem_row_major);
        }
        __syncwarp();
        {
            const int r = lane / 2, c0 = (lane % 2) * 16, t = tw + r, p = pos0 + t;
            float sv[16], mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int j = k0 + c0 + c, dist = p - j;
                const bool ok = t < T && j <= p && (window == 0 || dist < window);
                sv[c] = ok ? Ss[w][r * SLD + c0 + c] * scale +
                                 (dist < EXTENT ? rel[((size_t) t * NQ + h) * EXTENT + dist] : 0.0f)
                           : -INFINITY;
                mx = fmaxf(mx, sv[c]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            const float mo = ms[w][r], mn = fmaxf(mo, mx), f = expf(mo - mn);
            float sum = 0.0f;
            __syncwarp();   // every S value of the warp read before P overwrites them
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float e = expf(sv[c] - mn);
                sum += e;
                Pw[r * PLD + c0 + c] = __float2half_rn(e);
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            __syncwarp();
            if (!(lane % 2)) { ms[w][r] = mn; ls[w][r] = ls[w][r] * f + sum; fs[w][r] = f; }
        }
        __syncwarp();
        float fr[8];
        for (int i = 0; i < ne && i < 8; ++i) fr[i] = fs[w][rowof[i]];
#pragma unroll
        for (int n = 0; n < HD / 16; ++n)
            for (int i = 0; i < ne && i < 8; ++i) o[n].x[i] *= fr[i];
#pragma unroll
        for (int ks = 0; ks < AK / 16; ++ks) {
            FA pa_;
            wmma::load_matrix_sync(pa_, Pw + ks * 16, PLD);
#pragma unroll
            for (int n = 0; n < HD / 16; ++n) {
                FBr vb;
                wmma::load_matrix_sync(vb, Vs + ks * 16 * ALD + n * 16, ALD);
                wmma::mma_sync(o[n], pa_, vb, o[n]);
            }
        }
    }
    if (tw >= T) return;
    __syncwarp();
    float il[8];
    for (int i = 0; i < ne && i < 8; ++i) il[i] = 1.0f / ls[w][rowof[i]];
#pragma unroll
    for (int n = 0; n < HD / 16; ++n) {
        for (int i = 0; i < ne && i < 8; ++i) o[n].x[i] *= il[i];
        wmma::store_matrix_sync(out + (size_t) tw * QD + h * HD + n * 16, o[n], QD, wmma::mem_row_major);
    }
}

// SoftcappedSwiGLU on nr rows: gu [nr][2FF] -> act [nr][FF]
__global__ void kb_swiglu(const float* gu, float* act, int nr) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nr * FF) return;
    const size_t k = i / FF, j = i % FF;
    const float g = gu[k * 2 * FF + 2 * j], u = gu[k * 2 * FF + 2 * j + 1];
    act[i] = (tanhf(g / GATE_CAP) * GATE_CAP) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / UP_CAP) * UP_CAP);
}

// expert-ordered fp16 copy of the MoE input: ah[r][k] = fp16(x[rows[r]][k]) (the rounding the TC GEMM applied per tile)
__global__ void kb_gather_h(const float* __restrict__ x, const int* __restrict__ rows, half* __restrict__ ah) {
    const int r = blockIdx.x, src = rows[r];
    for (int k = threadIdx.x * 2; k < H; k += blockDim.x * 2)
        *reinterpret_cast<half2*>(ah + (size_t) r * H + k) =
            __floats2half2_rn(x[(size_t) src * H + k], x[(size_t) src * H + k + 1]);
}
// kb_swiglu with an fp16 output (the down GEMM's A, same values as converting the fp32 act per tile)
__global__ void kb_swiglu_h(const float* gu, half* act, int nr) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) nr * FF) return;
    const size_t k = i / FF, j = i % FF;
    const float g = gu[k * 2 * FF + 2 * j], u = gu[k * 2 * FF + 2 * j + 1];
    act[i] = __float2half_rn((tanhf(g / GATE_CAP) * GATE_CAP) * (1.0f / (1.0f + expf(-g))) * (tanhf(u / UP_CAP) * UP_CAP));
}

// out[t][i] = sum_k wts[t][k] ey[slot[t][k]][i] (k in routing order, as k_moe_tail); grid (H / 256, T)
__global__ void kb_combine(const float* ey, const int* slot, const float* wts, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    float a = 0.0f;
    for (int k = 0; k < TOPK; ++k) a += ey[(size_t) slot[t * TOPK + k] * H + i] * wts[t * TOPK + k];
    out[(size_t) t * H + i] = a;
}

// routing record (one-token path): rec[pos][k] = ids[k]
__global__ void k_rec_ids(const int* ids, int* rec, const int* d_pos) {
    rec[(size_t) (*d_pos) * TOPK + threadIdx.x] = ids[threadIdx.x];
}
// teacher-forced routing (verification): the recorded experts, weights from this pass' own scores; grid T, block 32
__global__ void kb_router_forced(const float* logits, const int* forced, int* ids, float* wts) {
    const int t = blockIdx.x, k = threadIdx.x;
    if (k) return;
    float sc[TOPK], sum = 0.0f;
    for (int j = 0; j < TOPK; ++j) {
        sc[j] = 1.0f / (1.0f + expf(-logits[(size_t) t * NE + forced[t * TOPK + j]]));
        sum += sc[j];
    }
    for (int j = 0; j < TOPK; ++j) { ids[t * TOPK + j] = forced[t * TOPK + j]; wts[t * TOPK + j] = sc[j] / (sum + 1e-20f) * ROUTE_SCALE; }
}

__global__ void kb_add(const float* a, const float* b, float* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// Checkpoint and weights

struct Ckpt {
    std::string path;
    StFile st;
    const StTensor& t(const std::string& name) const {
        auto it = st.tensors.find(name);
        if (it == st.tensors.end()) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
        return it->second;
    }
    std::vector<unsigned char> bytes(const std::string& name) const {
        std::vector<unsigned char> b;
        const std::string e = strata::image::st_read_tensor(path, st, t(name), b);
        if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); }
        return b;
    }
    vec f32(const std::string& name) const {
        const StTensor& d = t(name);
        auto b = bytes(name);
        vec out;
        if (d.dtype == "F32") {
            out.resize(b.size() / 4);
            std::memcpy(out.data(), b.data(), b.size());
        } else if (d.dtype == "BF16") {
            out.resize(b.size() / 2);
            for (size_t i = 0; i < out.size(); ++i) {
                uint32_t u = (uint32_t) (b[2 * i] | (b[2 * i + 1] << 8)) << 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else { std::fprintf(stderr, "%s: dtype %s\n", name.c_str(), d.dtype.c_str()); std::exit(1); }
        return out;
    }
};

size_t g_vram = 0;

void* dalloc(size_t bytes) {
    void* d = nullptr;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemset(d, 0, bytes));
    g_vram += bytes;
    return d;
}
float* dfloats(size_t n) { return (float*) dalloc(n * 4); }
void* upload(const void* p, size_t bytes) {
    void* d = dalloc(bytes);
    CK(cudaMemcpy(d, p, bytes, cudaMemcpyHostToDevice));
    return d;
}
float* dvec(const vec& v) { return (float*) upload(v.data(), v.size() * 4); }

struct DMat { const uint16_t* d = nullptr; int n = 0, k = 0; };

// Experts live in fixed-size slots of a device pool (one pool per format), each slot one expert as
// [w13 | w13 scales | w2 | w2 scales] (part A = w13 + scales, part B = w2 + scales).  The kernels address an expert
// through tA / tB (device tables, NE part pointers per layer, slices of E.tabA / E.tabB).
struct Experts {
    bool nvfp4 = true;
    const uint8_t* const* tA = nullptr;
    const uint8_t* const* tB = nullptr;
    const float *f13 = nullptr, *f2 = nullptr;
};
constexpr size_t NV_A = (size_t) 2 * FF * H / 2 + (size_t) 2 * FF * H / 16, NV_B = (size_t) H * FF / 2 + (size_t) H * FF / 16;
constexpr size_t MX_A = (size_t) 2 * FF * H + (size_t) 2 * FF * H / 32, MX_B = (size_t) H * FF + (size_t) H * FF / 32;
constexpr size_t NV_BYTES = NV_A + NV_B, MX_BYTES = MX_A + MX_B;   // 4478976 / 8211456
static_assert(NV_A % 256 == 0 && NV_BYTES % 256 == 0 && MX_A % 256 == 0 && MX_BYTES % 256 == 0, "slot alignment");

struct Gate { DMat down, fg; };   // fg: [forget; gain] stacked [2H, GATE_RANK]

struct Layer {
    bool full = false, loop = false;
    // proj: [wq 3072; gate_attn 3072; wr 384; sink 24; wk 1024; (k_wproj 32); wv 1024; (v_wproj 32)] stacked, K = H:
    // one launch when the query, key and value inputs are the same vector, three otherwise
    DMat proj, wo, router;
    int off_k = 0, off_v = 0, n_k = 0, n_v = 0;
    const float *q_scale, *rel_scale, *rel_proj, *k_conv = nullptr, *v_conv = nullptr, *k_static = nullptr,
                *v_static = nullptr, *attn_conv, *mlp_conv, *attn_norm = nullptr, *q_norm = nullptr,
                *k_norm = nullptr, *v_norm = nullptr, *mlp_norm, *attn_query = nullptr, *mlp_query, *router_bias;
    Experts ex;
    Gate gates[4];
};

struct Slot {
    void *K, *V;    // [cap][KVD] fp32 / fp16 / int8 codes with ksc / vsc (--kv, see kv_st)
    __half *ksc = nullptr, *vsc = nullptr;   // [cap][KVD / KVG]
    int cap = 0;
    bool full = false;
    float *khist, *vhist, *ahist, *mhist;   // [3][C]
};

struct Engine {
    const Ckpt* ck = nullptr;
    int ctx = 8192, nchunk_full = 0, nchunk_slide = WINDOW / CHUNK;
    std::vector<uint16_t> embed;   // host BF16
    const float *embed_norm, *norm, *final_query, *msg_queries;
    DMat unembed;
    Layer L[NL];
    Slot S[NCALLS];
    int* d_pos;
    // activations
    float *h, *out, *Bbuf, *preB, *pre_h, *prev_stack, *msgs;
    float *a, *t[4], *xq, *xk, *xv, *q, *k, *v, *kc, *vc, *dw, *rs, *rel, *sink, *gate, *att, *attn_out, *att2, *r;
    float *m, *mn, *cat, *gdown, *gf, *gg, *rl, *wts, *gu, *act, *ey, *moe, *y2, *logits;
    float *pm, *pl, *po, *emb_in, *pout, *gfg;
    int* ids;
    bool graph = true;
    cudaGraphExec_t exec = nullptr;
    int* rec = nullptr;   // routing record / forcing [NCALLS][ctx][TOPK] (verification only)
    int rec_mode = 0;     // 1 record (one-token path), 2 force (prefill)
    int kvm = 1;          // --kv f32 (0) | f16 (1, default) | i8 (2: int8 cells + fp16 scale per 64, Strata's kv_q8
                          // layout) | i8r (3: i8 + Walsh-Hadamard rotation)
    // experts: every expert in xhost layout order (NVFP4 layers 0-19 then MXFP8 layers 20-23, NE per layer, one slot
    // each); resident = the device pool holds them all, in that order
    uint8_t *pool_nv = nullptr, *pool_mx = nullptr;
    const uint8_t **tabA = nullptr, **tabB = nullptr;   // device [NL * NE]: part A / part B of expert (l, e)
    std::vector<const uint8_t*> htabA, htabB;            // host mirrors
};
size_t ex_bytes(int l) { return l < MXFP8_FIRST ? NV_BYTES : MX_BYTES; }
size_t ex_partA(int l) { return l < MXFP8_FIRST ? NV_A : MX_A; }
size_t ex_layer_off(int l) {   // offset of layer l's first expert in the xhost / resident layout
    return l < MXFP8_FIRST ? (size_t) l * NE * NV_BYTES
                           : (size_t) MXFP8_FIRST * NE * NV_BYTES + (size_t) (l - MXFP8_FIRST) * NE * MX_BYTES;
}
int g_slot = 0;   // the layer application being built (host side, baked into the launches)
int g_rpw = 4;    // decode GEMV rows per warp, upper bound (INK2_RPW: 1, 2, 4, 8)
int g_sel_rpw = 2;   // expert (selected) GEMV rows per warp (INK2_SEL_RPW: 1, 2, 4, 8)
int g_gemv_min_blocks = 164;   // bf16 GEMV: R halved until the grid has this many blocks (2 per SM; INK2_GEMV_MINB)

Engine E;

DMat dmat(const std::string& name) {
    const StTensor& t = E.ck->t(name);
    DMat m;
    m.n = (int) t.shape[0];
    m.k = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) m.k *= (int) t.shape[i];
    auto b = E.ck->bytes(name);
    m.d = (const uint16_t*) upload(b.data(), b.size());
    return m;
}
const float* fvec(const std::string& name) { return dvec(E.ck->f32(name)); }

DMat dmat_stack(const std::vector<std::string>& names) {   // BF16 matrices with the same K, rows concatenated
    std::vector<unsigned char> all;
    DMat m;
    for (const auto& n : names) {
        const StTensor& t = E.ck->t(n);
        int k = 1;
        for (size_t i = 1; i < t.shape.size(); ++i) k *= (int) t.shape[i];
        if (m.k && k != m.k) { std::fprintf(stderr, "%s: K %d != %d\n", n.c_str(), k, m.k); std::exit(1); }
        m.k = k;
        m.n += (int) t.shape[0];
        auto b = E.ck->bytes(n);
        all.insert(all.end(), b.begin(), b.end());
    }
    m.d = (const uint16_t*) upload(all.data(), all.size());
    return m;
}

void load() {
    const Ckpt& ck = *E.ck;
    const std::string p = "model.llm.";
    E.embed_norm = fvec(p + "embed_norm.weight");
    E.norm = fvec(p + "norm.weight");
    E.final_query = fvec(p + "attn_res.output_query");
    {
        auto b = ck.bytes(p + "embed.weight");
        E.embed.resize(b.size() / 2);
        std::memcpy(E.embed.data(), b.data(), b.size());
    }
    E.unembed = dmat(p + "unembed.weight");
    {
        const vec q = ck.f32(p + "middle_cycle_cross_pass_queries"), kv = ck.f32(p + "middle_cycle_cross_pass_kv_queries");
        vec mq((size_t) NLOOP * 4 * H);
        for (int i = 0; i < NLOOP; ++i) {
            const float* src[4] = {&q[(size_t) (2 * i) * H], &kv[(size_t) (2 * i) * H], &kv[(size_t) (2 * i + 1) * H],
                                   &q[(size_t) (2 * i + 1) * H]};
            for (int j = 0; j < 4; ++j) std::memcpy(&mq[((size_t) i * 4 + j) * H], src[j], H * 4);
        }
        E.msg_queries = dvec(mq);
    }
    E.pool_nv = (uint8_t*) dalloc((size_t) MXFP8_FIRST * NE * NV_BYTES);
    E.pool_mx = (uint8_t*) dalloc((size_t) (NL - MXFP8_FIRST) * NE * MX_BYTES);
    E.tabA = (const uint8_t**) dalloc((size_t) NL * NE * sizeof(void*));
    E.tabB = (const uint8_t**) dalloc((size_t) NL * NE * sizeof(void*));
    E.htabA.assign((size_t) NL * NE, nullptr);
    E.htabB.assign((size_t) NL * NE, nullptr);
    for (int l = 0; l < NL; ++l) {
        Layer& Y = E.L[l];
        const std::string q = p + "layers." + std::to_string(l) + ".";
        Y.full = full_layer(l);
        Y.loop = l >= LOOP_BEGIN && l < LOOP_END;
        {
            std::vector<std::string> parts = {q + "attn.wq_du.weight", q + "attn.gate_attn.weight", q + "attn.wr_du.weight",
                                              q + "attn.attn_sink_proj.weight", q + "attn.wk_dv.weight"};
            if (Y.full) parts.push_back(q + "attn.k_sconv.weight_proj");
            parts.push_back(q + "attn.wv_dv.weight");
            if (Y.full) parts.push_back(q + "attn.v_sconv.weight_proj");
            Y.proj = dmat_stack(parts);
            Y.off_k = OFF_K;
            Y.n_k = KVD + (Y.full ? NKV * KC : 0);
            Y.off_v = Y.off_k + Y.n_k;
            Y.n_v = Y.n_k;
            if (Y.proj.n != Y.off_v + Y.n_v) { std::fprintf(stderr, "layer %d: stacked rows %d\n", l, Y.proj.n); std::exit(1); }
        }
        Y.wo = dmat(q + "attn.wo_ud.weight");
        Y.router = dmat(q + "mlp.gate.weight");
        Y.router_bias = fvec(q + "mlp.gate.bias");
        Y.q_scale = fvec(q + "attn.q_l2_clip.scale");
        Y.rel_scale = fvec(q + "attn.rel_l2_clip.scale");
        Y.rel_proj = fvec(q + "attn.rel_logits_proj.proj");
        Y.attn_conv = fvec(q + "attn_sconv.weight");
        Y.mlp_conv = fvec(q + "mlp_sconv.weight");
        Y.mlp_norm = fvec(q + "mlp_norm.weight");
        if (Y.full) {
            Y.k_static = fvec(q + "attn.k_sconv.static_w");
            Y.v_static = fvec(q + "attn.v_sconv.static_w");
            Y.q_norm = fvec(q + "q_input_norm.weight");
            Y.k_norm = fvec(q + "k_input_norm.weight");
            Y.v_norm = fvec(q + "v_input_norm.weight");
        } else {
            Y.k_conv = fvec(q + "attn.k_sconv.weight");
            Y.v_conv = fvec(q + "attn.v_sconv.weight");
            Y.attn_norm = fvec(q + "attn_norm.weight");
        }
        if (l > 0) Y.attn_query = fvec(p + "attn_res.attn_queries." + std::to_string(l - 1));
        Y.mlp_query = fvec(p + "attn_res.mlp_queries." + std::to_string(l));
        if (Y.loop) {
            const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int g = 0; g < 4; ++g) {
                const std::string c = q + "controllers." + names[g] + ".";
                Y.gates[g] = {dmat(c + "down_weight"), dmat_stack({c + "forget_weight", c + "gain_weight"})};
            }
        }
        Experts& X = Y.ex;
        X.nvfp4 = l < MXFP8_FIRST;
        {   // the four expert tensors interleaved into one slot per expert, then the layer into its pool range
            const auto w13 = ck.bytes(q + "mlp.experts.w13_weight"), s13 = ck.bytes(q + "mlp.experts.w13_weight_scale");
            const auto w2 = ck.bytes(q + "mlp.experts.w2_weight"), s2 = ck.bytes(q + "mlp.experts.w2_weight_scale");
            const size_t a = w13.size() / NE, b = s13.size() / NE, c = w2.size() / NE, d = s2.size() / NE, eb = ex_bytes(l);
            if (a + b + c + d != eb || a + b != ex_partA(l)) { std::fprintf(stderr, "layer %d: expert bytes %zu\n", l, a + b + c + d); std::exit(1); }
            std::vector<uint8_t> lay((size_t) NE * eb);
            for (int e = 0; e < NE; ++e) {
                uint8_t* o = lay.data() + (size_t) e * eb;
                std::memcpy(o, w13.data() + e * a, a);
                std::memcpy(o + a, s13.data() + e * b, b);
                std::memcpy(o + a + b, w2.data() + e * c, c);
                std::memcpy(o + a + b + c, s2.data() + e * d, d);
            }
            uint8_t* dst = (X.nvfp4 ? E.pool_nv : E.pool_mx) + ex_layer_off(l) - (X.nvfp4 ? 0 : ex_layer_off(MXFP8_FIRST));
            CK(cudaMemcpy(dst, lay.data(), lay.size(), cudaMemcpyHostToDevice));
            for (int e = 0; e < NE; ++e) {
                E.htabA[l * NE + e] = dst + (size_t) e * eb;
                E.htabB[l * NE + e] = dst + (size_t) e * eb + ex_partA(l);
            }
            X.tA = E.tabA + (size_t) l * NE;
            X.tB = E.tabB + (size_t) l * NE;
        }
        if (X.nvfp4) {
            const vec a13 = ck.f32(q + "mlp.experts.w13_weight_amax"), d13 = ck.f32(q + "mlp.experts.w13_weight_outer_scale_denominator");
            const vec a2 = ck.f32(q + "mlp.experts.w2_weight_amax"), d2 = ck.f32(q + "mlp.experts.w2_weight_outer_scale_denominator");
            vec f13(NE), f2(NE);
            for (int e = 0; e < NE; ++e) { f13[e] = a13[e] / d13[0]; f2[e] = a2[e] / d2[0]; }
            X.f13 = dvec(f13);
            X.f2 = dvec(f2);
        }
        std::fprintf(stderr, "\rloaded layer %d/%d, %.1f GiB in VRAM", l + 1, NL, g_vram / 1073741824.0);
    }
    std::fprintf(stderr, "\n");
    CK(cudaMemcpy(E.tabA, E.htabA.data(), E.htabA.size() * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(E.tabB, E.htabB.data(), E.htabB.size() * sizeof(void*), cudaMemcpyHostToDevice));
}

void alloc_state() {
    CK(cudaMemcpyToSymbol(c_kvm, &E.kvm, sizeof(int)));
    E.nchunk_full = (E.ctx + CHUNK - 1) / CHUNK;
    for (int s = 0; s < NCALLS; ++s) {
        const int l = s < NL ? s : LOOP_BEGIN + (s - NL) % NLOOP;
        Slot& S = E.S[s];
        S.full = full_layer(l);
        S.cap = S.full ? E.ctx : WINDOW + PF_RING_EXTRA;
        if (E.kvm) {
            S.K = dalloc((size_t) S.cap * KVD * (E.kvm == 1 ? 2 : 1));
            S.V = dalloc((size_t) S.cap * KVD * (E.kvm == 1 ? 2 : 1));
        }
        if (E.kvm >= 2) {
            S.ksc = (__half*) dalloc((size_t) S.cap * (KVD / KVG) * 2);
            S.vsc = (__half*) dalloc((size_t) S.cap * (KVD / KVG) * 2);
        } else if (!E.kvm) {
            S.K = dfloats((size_t) S.cap * KVD);
            S.V = dfloats((size_t) S.cap * KVD);
        }
        S.khist = dfloats(3 * KVD);
        S.vhist = dfloats(3 * KVD);
        S.ahist = dfloats(3 * H);
        S.mhist = dfloats(3 * H);
    }
    E.d_pos = (int*) dalloc(4);
    E.h = dfloats(H); E.out = dfloats(H); E.Bbuf = dfloats(8 * H); E.preB = dfloats(8 * H); E.pre_h = dfloats(H);
    E.prev_stack = dfloats(8 * H); E.msgs = dfloats(4 * H); E.a = dfloats(3 * H);
    for (auto& t : E.t) t = dfloats(H);
    E.xq = dfloats(H); E.xk = dfloats(H); E.xv = dfloats(H);
    E.q = dfloats(QD); E.k = dfloats(KVD); E.v = dfloats(KVD); E.kc = dfloats(KVD); E.vc = dfloats(KVD); E.dw = dfloats(64);
    E.rs = dfloats(NQ * DREL); E.rel = dfloats(NQ * EXTENT); E.sink = dfloats(NQ); E.gate = dfloats(QD);
    E.att = dfloats(QD); E.attn_out = dfloats(H); E.att2 = dfloats(H); E.r = dfloats(H);
    E.m = dfloats(H); E.mn = dfloats(H); E.cat = dfloats(3 * 2 * H); E.gdown = dfloats(3 * GATE_RANK); E.gf = dfloats(H);
    E.gg = dfloats(H); E.rl = dfloats(NE); E.wts = dfloats(TOPK); E.ids = (int*) dalloc(TOPK * 4);
    E.gu = dfloats((size_t) TOPK * 2 * FF); E.act = dfloats((size_t) TOPK * FF); E.ey = dfloats((size_t) TOPK * H);
    E.moe = dfloats(H); E.y2 = dfloats(H); E.logits = dfloats(VOCAB_ROWS); E.emb_in = dfloats(H); E.pout = dfloats(OFF_K + 2 * (KVD + NKV * KC)); E.gfg = dfloats(3 * 2 * H);
    const int nc = std::max(E.nchunk_full, E.nchunk_slide);
    E.pm = dfloats((size_t) NQ * nc); E.pl = dfloats((size_t) NQ * nc); E.po = dfloats((size_t) NQ * nc * HD);
}

// ---------------------------------------------------------------------------------------------------------------------
// The forward pass (device pointers; the block-state count is static per call, kept on the host)

void gemv(const DMat& W, const float* x, float* y, int rows = -1, int first = 0) {   // rows [first, first + rows)
    const int n = rows < 0 ? W.n - first : rows;
    // small N (k / v projections: 1024 rows, wo: 2304) gave 32-72 blocks for 82 SMs at R = 4 (420-670 GB/s): fewer rows
    // per warp there, more blocks; per-row arithmetic is the same whatever R (bit-identical)
    int R = g_rpw;
    while (R > 1 && (n + 8 * R - 1) / (8 * R) < g_gemv_min_blocks) R /= 2;
    const unsigned g = (unsigned) ((n + 8 * R - 1) / (8 * R));
    const size_t sm = (size_t) W.k * 4;
    const uint16_t* w = W.d + (size_t) first * W.k;
    switch (R) {
    case 1: k_gemv_bf16<1><<<g, 256, sm>>>(w, x, y, n, W.k); break;
    case 2: k_gemv_bf16<2><<<g, 256, sm>>>(w, x, y, n, W.k); break;
    case 8: k_gemv_bf16<8><<<g, 256, sm>>>(w, x, y, n, W.k); break;
    default: k_gemv_bf16<4><<<g, 256, sm>>>(w, x, y, n, W.k);
    }
}
void rms(const float* x, const float* w, float* y, int n = H) { k_rms<<<1, 512>>>(x, w, y, n); }
void copy(float* dst, const float* src, size_t n) { CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice)); }

// Inkling2LoopMessageGate, then rms(.) * nw into yn (every gated stream is normalized next)
// n (<= 3) independent gates in one launch per stage
void message_gates(int n, const Gate* const g[], const float* const x[], const float* const m[], float* const y[],
                   const float* const nw[], float* const yn[]) {
    GateArgs a{};
    for (int j = 0; j < n; ++j) {
        a.down[j] = g[j]->down.d; a.fg[j] = g[j]->fg.d; a.x[j] = x[j]; a.m[j] = m[j]; a.y[j] = y[j]; a.nw[j] = nw[j];
        a.yn[j] = yn[j];
    }
    k_rms2<<<dim3(2, n), 512>>>(a, E.cat);
    k_gate_down<<<dim3(GATE_RANK, n), 256>>>(a, E.cat, E.gdown, g[0]->down.k);
    const int N = g[0]->fg.n, K = g[0]->fg.k;   // as gemv(): R halved while the grid is short
    int R = g_rpw;
    while (R > 1 && (N + 8 * R - 1) / (8 * R) < g_gemv_min_blocks) R /= 2;
    const dim3 grid((unsigned) ((N + 8 * R - 1) / (8 * R)), n);
    const size_t sm = (size_t) K * 4;
    switch (R) {
    case 1: k_gemv_bf16_gate<1><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K); break;
    case 2: k_gemv_bf16_gate<2><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K); break;
    case 8: k_gemv_bf16_gate<8><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K); break;
    default: k_gemv_bf16_gate<4><<<grid, 256, sm>>>(a, E.gdown, E.gfg, N, K);
    }
    k_gate_mix_rms<<<n, 512>>>(a, E.gfg);
}

// attnres dispatch on (S, NQ)
void attnres(const float* stack, int S, const float* queries, int nq, float* out, const float* nw0 = nullptr,
             const float* nw1 = nullptr, const float* nw2 = nullptr, float* outn = nullptr, int T = 1) {
    const size_t ss = (size_t) T * H;   // T tokens: entries [S][T][H]
#define AR_CASE(SS, QQ) if (S == SS && nq == QQ) { k_attnres<SS, QQ><<<T, AR_T>>>(stack, ss, queries, out, nw0, nw1, nw2, outn); return; }
    AR_CASE(1, 1) AR_CASE(2, 1) AR_CASE(3, 1) AR_CASE(4, 1) AR_CASE(5, 1) AR_CASE(6, 1)
    AR_CASE(2, 3) AR_CASE(3, 3) AR_CASE(4, 3) AR_CASE(5, 3)
    AR_CASE(2, 4) AR_CASE(3, 4)
#undef AR_CASE
    std::fprintf(stderr, "attnres: no kernel for S=%d nq=%d\n", S, nq);
    std::exit(1);
}

void attention(const Layer& Y, Slot& S, const float* xq, const float* xk, const float* xv) {
    float* P = E.pout;
    if (xq == xk && xk == xv) {
        gemv(Y.proj, xq, P);
    } else {
        const int N = Y.off_v + Y.n_v, K = Y.proj.k;   // one launch, three inputs (rows: q.. | k | v)
        int R = g_rpw;
        while (R > 1 && ((N + 8 * R - 1) / (8 * R) < g_gemv_min_blocks || Y.off_k % R || Y.off_v % R)) R /= 2;
        const unsigned g = (unsigned) ((N + 8 * R - 1) / (8 * R));
        const size_t sm = (size_t) 3 * K * 4;
        switch (R) {
        case 1: k_gemv_bf16_qkv<1><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
        case 2: k_gemv_bf16_qkv<2><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
        case 8: k_gemv_bf16_qkv<8><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K); break;
        default: k_gemv_bf16_qkv<4><<<g, 256, sm>>>(Y.proj.d, xq, xk, xv, Y.off_k, Y.off_v, P, N, K);
        }
    }
    float *q = P, *k = P + Y.off_k, *v = P + Y.off_v;
    float *rs = P + OFF_REL, *sink = P + OFF_SINK, *gate = P + OFF_GATE;
    static_assert(HD == 128, "k_attn_prep: 128-thread blocks for q heads, rs heads and k / v groups");
    if (Y.full)
        k_attn_prep<true><<<2 * NQ + NKV, 128>>>(q, Y.q_scale, rs, Y.rel_scale, Y.rel_proj, E.rel, 1, k, v, S.khist, S.vhist,
                                                Y.k_static, Y.v_static, S.K, S.V, S.ksc, S.vsc, S.cap, E.d_pos);
    else
        k_attn_prep<false><<<2 * NQ + NKV, 128>>>(q, Y.q_scale, rs, Y.rel_scale, Y.rel_proj, E.rel, 0, k, v, S.khist,
                                                 S.vhist, Y.k_conv, Y.v_conv, S.K, S.V, S.ksc, S.vsc, S.cap, E.d_pos);
    const int nch = S.full ? E.nchunk_full : E.nchunk_slide;
    k_attn_partial<<<dim3(NQ, nch), 128>>>(q, S.K, S.V, S.ksc, S.vsc, E.rel, S.cap, S.full ? 0 : WINDOW, E.d_pos, E.pm, E.pl, E.po);
    k_attn_merge<<<NQ, HD>>>(E.pm, E.pl, E.po, nch, sink, gate, E.att);
    gemv(Y.wo, E.att, E.attn_out);
}

void moe(const Layer& Y, const float* x, const float* r, float* mhist, const float* mlp_conv, float* out) {
    k_gemv_bf16_rowblock<<<NE, 256>>>(Y.router.d, x, E.rl, H);
    k_router<<<1, NE>>>(E.rl, Y.router_bias, E.ids, E.wts);
    if (E.rec_mode == 1) k_rec_ids<<<1, TOPK>>>(E.ids, E.rec + (size_t) g_slot * E.ctx * TOPK, E.d_pos);
    const Experts& X = Y.ex;
    if (X.nvfp4) {
#define SEL(R) \
        k_gemv_nvfp4_sel<R><<<dim3(2 * FF / (8 * R), TOPK), 256, (H + 256) * 4>>>(X.tA, X.f13, E.ids, x, 0, E.gu, 2 * FF, H); \
        k_gemv_nvfp4_sel<R, true><<<dim3(H / (8 * R), TOPK), 256, (FF + 256) * 4>>>(X.tB, X.f2, E.ids, E.gu, 2 * FF, E.ey, H, FF);
        switch (g_sel_rpw) { case 1: SEL(1) break; case 2: SEL(2) break; case 8: SEL(8) break; default: SEL(4) }
#undef SEL
    } else {
#define SEL(R) \
        k_gemv_mxfp8_sel<R><<<dim3(2 * FF / (8 * R), TOPK), 256, H * 4>>>(X.tA, E.ids, x, 0, E.gu, 2 * FF, H); \
        k_gemv_mxfp8_sel<R, true><<<dim3(H / (8 * R), TOPK), 256, FF * 4>>>(X.tB, E.ids, E.gu, 2 * FF, E.ey, H, FF);
        switch (g_sel_rpw) { case 1: SEL(1) break; case 2: SEL(2) break; case 8: SEL(8) break; default: SEL(4) }
#undef SEL
    }
    k_moe_tail<<<(H + 255) / 256, 256>>>(E.ey, E.wts, mhist, mlp_conv, r, out, E.d_pos);
}

// one layer application: input E.h, block states Bbuf[0..nB), output E.out; msgs = 4 x H or nullptr
void layer_forward(int l, Slot& S, int nB, const float* msgs) {
    const Layer& Y = E.L[l];
    float* stack_h = E.Bbuf + (size_t) nB * H;
    copy(stack_h, E.h, H);   // stack [B..., h]
    const float *xq, *xk, *xv;
    if (Y.full) {
        if (!msgs) {   // l > 0 always (full layers are 5, 11, 17, 23)
            attnres(E.Bbuf, nB + 1, Y.attn_query, 3, nullptr, Y.q_norm, Y.k_norm, Y.v_norm, E.a);
            xq = E.a; xk = E.a + H; xv = E.a + 2 * H;
        } else {
            attnres(E.Bbuf, nB + 1, Y.attn_query, 3, E.a);
            const Gate* g3[3] = {&Y.gates[0], &Y.gates[1], &Y.gates[2]};
            const float* x3[3] = {E.a, E.a + H, E.a + 2 * H};
            const float* m3[3] = {msgs, msgs + H, msgs + 2 * H};
            float* y3[3] = {E.t[0], E.t[1], E.t[2]};
            const float* nw3[3] = {Y.q_norm, Y.k_norm, Y.v_norm};
            float* yn3[3] = {E.xq, E.xk, E.xv};
            message_gates(3, g3, x3, m3, y3, nw3, yn3);
            xq = E.xq; xk = E.xk; xv = E.xv;
        }
    } else if (!msgs) {
        if (l > 0) attnres(E.Bbuf, nB + 1, Y.attn_query, 1, nullptr, Y.attn_norm, nullptr, nullptr, E.xq);
        else rms(E.h, Y.attn_norm, E.xq);
        xq = xk = xv = E.xq;
    } else {
        attnres(E.Bbuf, nB + 1, Y.attn_query, 1, E.a);
        const Gate* g3[3] = {&Y.gates[0], &Y.gates[1], &Y.gates[2]};
        const float* x3[3] = {E.a, E.a, E.a};
        const float* m3[3] = {msgs, msgs + H, msgs + 2 * H};
        float* y3[3] = {E.t[0], E.t[1], E.t[2]};
        const float* nw3[3] = {Y.attn_norm, Y.attn_norm, Y.attn_norm};
        float* yn3[3] = {E.xq, E.xk, E.xv};
        message_gates(3, g3, x3, m3, y3, nw3, yn3);
        xq = E.xq; xk = E.xk; xv = E.xv;
    }
    attention(Y, S, xq, xk, xv);
    int ns;
    const float* r;
    if (l % BLOCK == 0) {   // block boundary: the residual restarts, B gains h (already at Bbuf[nB])
        k_attn_tail<<<H / 256, 256>>>(E.attn_out, S.ahist, Y.attn_conv, E.h, E.att2, E.Bbuf + (size_t) (nB + 1) * H, 1, E.d_pos);
        r = E.att2;
        ns = nB + 2;
    } else {
        k_attn_tail<<<H / 256, 256>>>(E.attn_out, S.ahist, Y.attn_conv, E.h, E.r, stack_h, 0, E.d_pos);
        r = E.r;
        ns = nB + 1;
    }
    if (!msgs) {
        attnres(E.Bbuf, ns, Y.mlp_query, 1, nullptr, Y.mlp_norm, nullptr, nullptr, E.mn);
    } else {
        attnres(E.Bbuf, ns, Y.mlp_query, 1, E.m);
        const Gate* g1[1] = {&Y.gates[3]};
        const float* x1[1] = {E.m};
        const float* m1[1] = {msgs + 3 * H};
        float* y1[1] = {E.t[3]};
        const float* nw1[1] = {Y.mlp_norm};
        float* yn1[1] = {E.mn};
        message_gates(1, g1, x1, m1, y1, nw1, yn1);
    }
    moe(Y, E.mn, r, S.mhist, Y.mlp_conv, E.out);
    if (l % BLOCK == 0) copy(stack_h, E.h, H);   // B gets the layer's input (the caller counts it)
}

// the token's embedding row (host table) into E.emb_in
void upload_embed(int tok) {
    vec e(H);
    for (int i = 0; i < H; ++i) {
        uint32_t u = (uint32_t) E.embed[(size_t) tok * H + i] << 16;
        std::memcpy(&e[i], &u, 4);
    }
    CK(cudaMemcpyAsync(E.emb_in, e.data(), H * 4, cudaMemcpyHostToDevice));
}

// one token: embedding in E.emb_in, position in E.d_pos; writes E.logits.  Device work only (captured in a graph):
// the host-side pointer swaps are the same on every call (an even number), so a replay sees the same buffers.
void forward_device() {
    rms(E.emb_in, E.embed_norm, E.h);
    int nB = 0;
    auto run = [&](int l, int slot, const float* msgs) {
        g_slot = slot;
        layer_forward(l, E.S[slot], nB, msgs);
        if (l % BLOCK == 0) ++nB;
        std::swap(E.h, E.out);
    };
    for (int l = 0; l < LOOP_BEGIN; ++l) run(l, l, nullptr);
    const int npre = nB;
    copy(E.pre_h, E.h, H);
    copy(E.preB, E.Bbuf, (size_t) npre * H);
    for (int l = LOOP_BEGIN; l < LOOP_END; ++l) run(l, l, nullptr);
    for (int p = 1; p < PASSES; ++p) {
        const int nprev = nB - (npre + 1);   // previous pass' block states after its first one, then its output h
        copy(E.prev_stack, E.Bbuf + (size_t) (npre + 1) * H, (size_t) nprev * H);
        copy(E.prev_stack + (size_t) nprev * H, E.h, H);
        copy(E.h, E.pre_h, H);
        copy(E.Bbuf, E.preB, (size_t) npre * H);
        nB = npre;
        for (int i = 0; i < NLOOP; ++i) {
            attnres(E.prev_stack, nprev + 1, E.msg_queries + (size_t) i * 4 * H, 4, E.msgs);
            run(LOOP_BEGIN + i, NL + (p - 1) * NLOOP + i, E.msgs);
        }
    }
    for (int l = LOOP_END; l < NL; ++l) run(l, l, nullptr);
    copy(E.Bbuf + (size_t) nB * H, E.h, H);
    attnres(E.Bbuf, nB + 1, E.final_query, 1, nullptr, E.norm, nullptr, nullptr, E.mn);
    gemv(E.unembed, E.mn, E.logits, VOCAB);
    k_softcap<<<(VOCAB + 255) / 256, 256>>>(E.logits, VOCAB, LOGIT_CAP);
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill: the prompt in chunks of up to TP tokens.  Same math as forward_device, row-batched: dense products as
// cuBLAS SGEMM (full FP32) on an fp32 copy of the BF16 matrix made per use, experts as kb_moe_gemm (weights decoded on
// the fly), attention tiled (kb_attn).  It fills the same KV / conv state as the one-token path, which then decodes.

struct Prefill {
    int TP = 512;   // chunk capacity
    bool tc = false;   // --tc 1: fp16 tensor-core products (NVFP4 experts, attention proj / wo)
    cublasHandle_t cb = nullptr, cbh = nullptr;   // fp32 (pedantic) / fp16 tensor-op handles
    half *wh = nullptr, *xh = nullptr, *qh = nullptr;   // fp16 weight / activation / query scratch
    float* wf = nullptr;   // fp32 copy of one dense matrix (or a slice of the unembedding)
    size_t wf_cap = 0;
    float *x0, *h, *out, *B, *preB, *pre_h, *prev, *msgs, *a, *xq, *xk, *xv, *P, *kc, *vc, *rel, *att, *attn_out, *att2,
          *r, *m, *mn, *moe, *y2, *cat, *gdown, *gfg, *rl, *wts, *gu, *act, *ey, *lg;
    int *ids, *rows, *slot, *tile_e, *tile_r0, *ntiles, *eoff, *ecnt;
    int np_max = 0;
};
Prefill PF;

void alloc_prefill() {
    const size_t T = PF.TP, TH = T * H;
    CK(cublasCreate(&PF.cb) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown);
    cublasSetStream(PF.cb, cudaStreamPerThread);
    cublasSetMathMode(PF.cb, CUBLAS_PEDANTIC_MATH);   // full FP32, no TF32 / reduced-precision shortcuts
    for (int l = 0; l < NL; ++l) PF.np_max = std::max(PF.np_max, E.L[l].proj.n);
    PF.wf_cap = std::max((size_t) PF.np_max * H, (size_t) 8192 * H);
    PF.wf = dfloats(PF.wf_cap);
    PF.x0 = dfloats(TH); PF.h = dfloats(TH); PF.out = dfloats(TH); PF.B = dfloats(8 * TH); PF.preB = dfloats(8 * TH);
    PF.pre_h = dfloats(TH); PF.prev = dfloats(8 * TH); PF.msgs = dfloats(4 * TH); PF.a = dfloats(3 * TH);
    PF.xq = dfloats(TH); PF.xk = dfloats(TH); PF.xv = dfloats(TH);
    PF.kc = dfloats(T * KVD); PF.vc = dfloats(T * KVD); PF.att = dfloats(((T + 63) / 64 * 64) * QD);
    PF.attn_out = dfloats(TH); PF.att2 = dfloats(TH); PF.r = dfloats(TH); PF.m = dfloats(TH); PF.mn = dfloats(TH);
    PF.moe = dfloats(TH); PF.y2 = dfloats(TH); PF.cat = dfloats(2 * TH); PF.gdown = dfloats(T * GATE_RANK);
    PF.gfg = dfloats(2 * TH); PF.rl = dfloats(T * NE); PF.wts = dfloats(T * TOPK);
    // One region, phases with disjoint lifetimes (saves ~217 KB per chunk token, 445 MB at chunk 2048):
    //   attention: P [T][np_max], rel [T][NQ][EXTENT];  MoE: gu [T*TOPK][2FF] then ey (same rows, H = 2FF) after the
    //   SwiGLU has read gu, act [T*TOPK][FF] after gu;  after the last layer: lg [T][8192] (all-logits dumps only)
    static_assert(2 * FF == H, "ey aliases gu");
    const size_t att_n = T * ((size_t) PF.np_max + NQ * EXTENT), moe_n = T * TOPK * (size_t) 3 * FF;
    float* region = dfloats(std::max(std::max(att_n, moe_n), T * (size_t) 8192));
    PF.P = region; PF.rel = region + T * PF.np_max;
    PF.gu = region; PF.ey = region; PF.act = region + T * TOPK * 2 * FF;
    PF.lg = region;
    auto ints = [](size_t n) { return (int*) dalloc(n * 4); };
    PF.ids = ints(T * TOPK); PF.rows = ints(T * TOPK); PF.slot = ints(T * TOPK);
    PF.tile_e = ints(T * TOPK / FM + NE); PF.tile_r0 = ints(T * TOPK / FM + NE); PF.ntiles = ints(1);
    PF.eoff = ints(NE); PF.ecnt = ints(NE);
    CK(cudaFuncSetAttribute(kb_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) FA_SMEM));
    if (PF.tc) {
        CK(cublasCreate(&PF.cbh) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown);
        cublasSetStream(PF.cbh, cudaStreamPerThread);
        PF.wh = (half*) dalloc((size_t) PF.np_max * H * 2);
        PF.xh = (half*) dalloc(3 * T * QD * 2);
        PF.qh = (half*) dalloc(((T + 63) / 64 * 64) * QD * 2);
    }
}

// tensor-core dense product: Y[t][0..n) = fp16(X[t]) . fp16(W rows [first, first + n))^T, fp32 accumulation.
// wcvt: convert W (whole matrix) into PF.wh first; xslot: which third of PF.xh holds X's fp16 copy
void hgemm(const DMat& W, bool wcvt, int first, int n, const float* X, int ldx, int K, int xslot, float* Y, int ldy, int T) {
    if (wcvt) {
        const size_t nw = (size_t) W.n * W.k;
        kb_bf16_to_f16<<<(unsigned) ((nw + 255) / 256), 256>>>(W.d, PF.wh, nw);
    }
    half* xh = PF.xh + (size_t) xslot * PF.TP * QD;
    const size_t nx = (size_t) T * K;
    kb_f32_to_f16<<<(unsigned) ((nx + 255) / 256), 256>>>(X, ldx, K, xh, nx);
    const float one = 1.0f, zero = 0.0f;
    if (cublasGemmEx(PF.cbh, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &one, PF.wh + (size_t) first * K, CUDA_R_16F, K, xh,
                     CUDA_R_16F, K, &zero, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasGemmEx failed (%d x %d x %d)\n", n, T, K);
        std::exit(1);
    }
}

// rows [first, first + n) of W (BF16) into PF.wf as fp32
const float* wf32(const DMat& W, int first = 0, int n = -1) {
    if (n < 0) n = W.n - first;
    const size_t n8 = (size_t) n * W.k / 8;
    if ((size_t) n * W.k > PF.wf_cap) { std::fprintf(stderr, "wf32: %d x %d over the scratch\n", n, W.k); std::exit(1); }
    kb_bf16_to_f32<<<(unsigned) ((n8 + 255) / 256), 256>>>(W.d + (size_t) first * W.k, PF.wf, n8);
    return PF.wf;
}
// Y[t][0..n) (row stride ldy) = X[t] (row stride ldx) . Wf[0..n)^T, Wf fp32 [n][K]
void sgemm(const float* Wf, int n, int K, const float* X, int ldx, float* Y, int ldy, int T) {
    const float one = 1.0f, zero = 0.0f;
    if (cublasSgemm(PF.cb, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &one, Wf, K, X, ldx, &zero, Y, ldy) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSgemm failed (%d x %d x %d)\n", n, T, K);
        std::exit(1);
    }
}
void gemm(const DMat& W, const float* X, int ldx, float* Y, int ldy, int T) { sgemm(wf32(W), W.n, W.k, X, ldx, Y, ldy, T); }

void message_gate_b(const Gate& g, const float* x, int sx, const float* m, int sm, const float* nw, float* yn, int T) {
    kb_rms2<<<dim3(2, T), 512>>>(x, sx, m, sm, PF.cat);
    gemm(g.down, PF.cat, 2 * H, PF.gdown, GATE_RANK, T);
    gemm(g.fg, PF.gdown, GATE_RANK, PF.gfg, 2 * H, T);
    kb_gate_mix_rms<<<T, 512>>>(x, sx, m, sm, PF.gfg, nw, yn);
}

void conv_b(const float* x, int sx, float* hist, const float* w, const float* dw, float* y, int sy, int C, int T, int pos0) {
    kb_conv<<<dim3(C / 256, T), 256>>>(x, sx, hist, w, dw, y, sy, C, pos0);
    kb_hist_store<<<dim3(C / 256, std::min(3, T)), 256>>>(x, sx, hist, C, T, pos0);
}

void attention_b(const Layer& Y, Slot& S, const float* xq, int sq, const float* xk, int sk, const float* xv, int sv,
                 int T, int pos0) {
    const int np = Y.proj.n;
    float* P = PF.P;
    if (PF.tc) {
        if (xq == xk && xk == xv && sq == sk && sk == sv) {
            hgemm(Y.proj, true, 0, np, xq, sq, H, 0, P, np, T);
        } else {
            hgemm(Y.proj, true, 0, OFF_K, xq, sq, H, 0, P, np, T);
            hgemm(Y.proj, false, Y.off_k, Y.n_k, xk, sk, H, 1, P + Y.off_k, np, T);
            hgemm(Y.proj, false, Y.off_v, Y.n_v, xv, sv, H, 2, P + Y.off_v, np, T);
        }
    }
    const float* W = PF.tc ? nullptr : wf32(Y.proj);
    if (PF.tc) {
    } else if (xq == xk && xk == xv && sq == sk && sk == sv) {
        sgemm(W, np, H, xq, sq, P, np, T);
    } else {
        sgemm(W, OFF_K, H, xq, sq, P, np, T);
        sgemm(W + (size_t) Y.off_k * H, Y.n_k, H, xk, sk, P + Y.off_k, np, T);
        sgemm(W + (size_t) Y.off_v * H, Y.n_v, H, xv, sv, P + Y.off_v, np, T);
    }
    kb_head_clip<<<dim3(NQ, T), 128>>>(P, np, Y.q_scale, HD, QK_CLIP, Y.full, pos0);
    if (E.kvm == 3) kb_wht<<<dim3(NQ, T), 128>>>(P, np);
    float *k = P + Y.off_k, *v = P + Y.off_v;
    if (Y.full) {
        conv_b(k, np, S.khist, Y.k_static, k + KVD, PF.kc, KVD, KVD, T, pos0);
        conv_b(v, np, S.vhist, Y.v_static, v + KVD, PF.vc, KVD, KVD, T, pos0);
    } else {
        conv_b(k, np, S.khist, Y.k_conv, nullptr, PF.kc, KVD, KVD, T, pos0);
        conv_b(v, np, S.vhist, Y.v_conv, nullptr, PF.vc, KVD, KVD, T, pos0);
    }
    kb_kv_store<<<dim3(NKV, T), HD>>>(PF.kc, PF.vc, S.K, S.V, S.ksc, S.vsc, S.cap, pos0);
    kb_head_clip<<<dim3(NQ, T), 128>>>(P + OFF_REL, np, Y.rel_scale, DREL, REL_CLIP, Y.full, pos0);
    kb_rel_logits<<<dim3(NQ, T), 256>>>(P + OFF_REL, np, Y.rel_proj, PF.rel);
    if (PF.tc) {
        const size_t nqh = (size_t) T * QD;
        kb_f32_to_f16<<<(unsigned) ((nqh + 255) / 256), 256>>>(P, np, QD, PF.qh, nqh);
        kb_attn_tc<<<dim3((T + AQ - 1) / AQ, NKV), 128 * NREP>>>(PF.qh, P, np, S.K, S.V, S.ksc, S.vsc, PF.rel, S.cap, S.full ? 0 : WINDOW, pos0,
                                                         T, PF.att);
    } else {
        kb_attn<<<dim3((T + FQ - 1) / FQ, NQ), 256, FA_SMEM>>>(P, np, S.K, S.V, S.ksc, S.vsc, PF.rel, S.cap, S.full ? 0 : WINDOW, pos0,
                                                             T, PF.att);
    }
    if (E.kvm == 3) kb_wht<<<dim3(NQ, T), 128>>>(PF.att, QD);
    kb_mul_sigmoid<<<dim3(QD / 256, T), 256>>>(PF.att, P + OFF_GATE, np);
    if (PF.tc) hgemm(Y.wo, true, 0, H, PF.att, QD, QD, 0, PF.attn_out, H, T);
    else gemm(Y.wo, PF.att, QD, PF.attn_out, H, T);
}

void moe_b(const Layer& Y, const float* x, float* out, int T, int pos0) {
    gemm(Y.router, x, H, PF.rl, NE, T);
    if (E.rec_mode == 2)
        kb_router_forced<<<T, 32>>>(PF.rl, E.rec + ((size_t) g_slot * E.ctx + pos0) * TOPK, PF.ids, PF.wts);
    else
        k_router<<<T, NE>>>(PF.rl, Y.router_bias, PF.ids, PF.wts);
    kb_moe_plan<<<1, 1024>>>(PF.ids, T, PF.rows, PF.slot, PF.tile_e, PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt);
    const Experts& X = Y.ex;
    const int nr = T * TOPK, maxt = nr / FM + NE;
    if (PF.tc) {
        const int maxt_tc = nr / TM + NE;
        auto g = X.nvfp4 ? kb_moe_gemm_tc<true> : kb_moe_gemm_tc<false>;
        // fp16 A copies live in the act region: the gathered input [nr][H] halves (same bytes as act [nr][FF] floats)
        // until the SwiGLU, then the fp16 act [nr][FF]
        half* ah = reinterpret_cast<half*>(PF.act);
        kb_gather_h<<<nr, 256>>>(x, PF.rows, ah);
        g<<<dim3(maxt_tc, 2 * FF / TN), 256>>>(ah, H, X.tA, X.f13, PF.tile_e, PF.tile_r0, PF.ntiles,
                                               PF.eoff, PF.ecnt, PF.gu, 2 * FF, H);
        kb_swiglu_h<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, ah, nr);
        g<<<dim3(maxt_tc, H / TN), 256>>>(ah, FF, X.tB, X.f2, PF.tile_e, PF.tile_r0, PF.ntiles,
                                          PF.eoff, PF.ecnt, PF.ey, H, FF);
    } else if (X.nvfp4) {
        kb_moe_gemm<true><<<dim3(maxt, 2 * FF / FN), 256>>>(x, H, PF.rows, X.tA, X.f13, PF.tile_e, PF.tile_r0,
                                                            PF.ntiles, PF.eoff, PF.ecnt, PF.gu, 2 * FF, H);
        kb_swiglu<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, PF.act, nr);
        kb_moe_gemm<true><<<dim3(maxt, H / FN), 256>>>(PF.act, FF, nullptr, X.tB, X.f2, PF.tile_e, PF.tile_r0,
                                                       PF.ntiles, PF.eoff, PF.ecnt, PF.ey, H, FF);
    } else {
        kb_moe_gemm<false><<<dim3(maxt, 2 * FF / FN), 256>>>(x, H, PF.rows, X.tA, nullptr, PF.tile_e, PF.tile_r0,
                                                             PF.ntiles, PF.eoff, PF.ecnt, PF.gu, 2 * FF, H);
        kb_swiglu<<<(unsigned) (((size_t) nr * FF + 255) / 256), 256>>>(PF.gu, PF.act, nr);
        kb_moe_gemm<false><<<dim3(maxt, H / FN), 256>>>(PF.act, FF, nullptr, X.tB, nullptr, PF.tile_e,
                                                        PF.tile_r0, PF.ntiles, PF.eoff, PF.ecnt, PF.ey, H, FF);
    }
    kb_combine<<<dim3(H / 256, T), 256>>>(PF.ey, PF.slot, PF.wts, out);
}

void copy_b(float* dst, const float* src, size_t n) { copy(dst, src, n); }
void add_b(const float* a, const float* b, float* y, size_t n) { kb_add<<<(unsigned) ((n + 255) / 256), 256>>>(a, b, y, n); }

// layer_forward for T tokens: input PF.h, block states PF.B[0..nB) ([S][T][H]), output PF.out; msgs [T][4][H] or null
void layer_forward_b(int l, Slot& S, int nB, const float* msgs, int T, int pos0) {
    const Layer& Y = E.L[l];
    const size_t ES = (size_t) T * H;
    float* stack_h = PF.B + nB * ES;
    copy_b(stack_h, PF.h, ES);
    const float *xq, *xk, *xv;
    int sq = H, sk = H, sv = H;
    if (Y.full) {
        if (!msgs) {
            attnres(PF.B, nB + 1, Y.attn_query, 3, nullptr, Y.q_norm, Y.k_norm, Y.v_norm, PF.a, T);
            xq = PF.a; xk = PF.a + H; xv = PF.a + 2 * H;
            sq = sk = sv = 3 * H;
        } else {
            attnres(PF.B, nB + 1, Y.attn_query, 3, PF.a, nullptr, nullptr, nullptr, nullptr, T);
            message_gate_b(Y.gates[0], PF.a, 3 * H, msgs, 4 * H, Y.q_norm, PF.xq, T);
            message_gate_b(Y.gates[1], PF.a + H, 3 * H, msgs + H, 4 * H, Y.k_norm, PF.xk, T);
            message_gate_b(Y.gates[2], PF.a + 2 * H, 3 * H, msgs + 2 * H, 4 * H, Y.v_norm, PF.xv, T);
            xq = PF.xq; xk = PF.xk; xv = PF.xv;
        }
    } else if (!msgs) {
        if (l > 0) attnres(PF.B, nB + 1, Y.attn_query, 1, nullptr, Y.attn_norm, nullptr, nullptr, PF.xq, T);
        else k_rms<<<T, 512>>>(PF.h, Y.attn_norm, PF.xq, H);
        xq = xk = xv = PF.xq;
    } else {
        attnres(PF.B, nB + 1, Y.attn_query, 1, PF.a, nullptr, nullptr, nullptr, nullptr, T);
        message_gate_b(Y.gates[0], PF.a, H, msgs, 4 * H, Y.attn_norm, PF.xq, T);
        message_gate_b(Y.gates[1], PF.a, H, msgs + H, 4 * H, Y.attn_norm, PF.xk, T);
        message_gate_b(Y.gates[2], PF.a, H, msgs + 2 * H, 4 * H, Y.attn_norm, PF.xv, T);
        xq = PF.xq; xk = PF.xk; xv = PF.xv;
    }
    attention_b(Y, S, xq, sq, xk, sk, xv, sv, T, pos0);
    conv_b(PF.attn_out, H, S.ahist, Y.attn_conv, nullptr, PF.att2, H, H, T, pos0);
    int ns;
    const float* r;
    if (l % BLOCK == 0) {
        r = PF.att2;
        copy_b(PF.B + (nB + 1) * ES, PF.att2, ES);
        ns = nB + 2;
    } else {
        add_b(PF.h, PF.att2, PF.r, ES);
        r = PF.r;
        copy_b(stack_h, PF.r, ES);
        ns = nB + 1;
    }
    if (!msgs) {
        attnres(PF.B, ns, Y.mlp_query, 1, nullptr, Y.mlp_norm, nullptr, nullptr, PF.mn, T);
    } else {
        attnres(PF.B, ns, Y.mlp_query, 1, PF.m, nullptr, nullptr, nullptr, nullptr, T);
        message_gate_b(Y.gates[3], PF.m, H, msgs + 3 * H, 4 * H, Y.mlp_norm, PF.mn, T);
    }
    moe_b(Y, PF.mn, PF.moe, T, pos0);
    conv_b(PF.moe, H, S.mhist, Y.mlp_conv, nullptr, PF.y2, H, H, T, pos0);
    add_b(r, PF.y2, PF.out, ES);
    if (l % BLOCK == 0) copy_b(stack_h, PF.h, ES);
}

// tokens ids[pos0 .. pos0 + T) -> logits of the last one in E.logits (all of them into `all` [T][VOCAB] if non-null)
void forward_prefill(const int* toks, int T, int pos0, float* all) {
    const size_t ES = (size_t) T * H;
    {
        vec e(ES);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < H; ++i) {
                uint32_t u = (uint32_t) E.embed[(size_t) toks[t] * H + i] << 16;
                std::memcpy(&e[(size_t) t * H + i], &u, 4);
            }
        CK(cudaMemcpyAsync(PF.x0, e.data(), ES * 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CK(cudaStreamSynchronize(cudaStreamPerThread));
    }
    k_rms<<<T, 512>>>(PF.x0, E.embed_norm, PF.h, H);
    int nB = 0;
    auto run = [&](int l, int slot, const float* msgs) {
        g_slot = slot;
        layer_forward_b(l, E.S[slot], nB, msgs, T, pos0);
        if (l % BLOCK == 0) ++nB;
        std::swap(PF.h, PF.out);
    };
    for (int l = 0; l < LOOP_BEGIN; ++l) run(l, l, nullptr);
    const int npre = nB;
    copy_b(PF.pre_h, PF.h, ES);
    copy_b(PF.preB, PF.B, npre * ES);
    for (int l = LOOP_BEGIN; l < LOOP_END; ++l) run(l, l, nullptr);
    for (int p = 1; p < PASSES; ++p) {
        const int nprev = nB - (npre + 1);
        copy_b(PF.prev, PF.B + (npre + 1) * ES, nprev * ES);
        copy_b(PF.prev + nprev * ES, PF.h, ES);
        copy_b(PF.h, PF.pre_h, ES);
        copy_b(PF.B, PF.preB, npre * ES);
        nB = npre;
        for (int i = 0; i < NLOOP; ++i) {
            attnres(PF.prev, nprev + 1, E.msg_queries + (size_t) i * 4 * H, 4, PF.msgs, nullptr, nullptr, nullptr,
                    nullptr, T);
            run(LOOP_BEGIN + i, NL + (p - 1) * NLOOP + i, PF.msgs);
        }
    }
    for (int l = LOOP_END; l < NL; ++l) run(l, l, nullptr);
    copy_b(PF.B + nB * ES, PF.h, ES);
    attnres(PF.B, nB + 1, E.final_query, 1, nullptr, E.norm, nullptr, nullptr, PF.mn, T);
    gemv(E.unembed, PF.mn + (size_t) (T - 1) * H, E.logits, VOCAB);
    k_softcap<<<(VOCAB + 255) / 256, 256>>>(E.logits, VOCAB, LOGIT_CAP);
    if (all) {   // every position (verification): the unembedding in slices of 8192 rows
        for (int r0 = 0; r0 < VOCAB; r0 += 8192) {
            const int n = std::min(8192, VOCAB - r0);
            sgemm(wf32(E.unembed, r0, n), n, H, PF.mn, H, PF.lg, n, T);
            k_softcap<<<(unsigned) (((size_t) T * n + 255) / 256), 256>>>(PF.lg, T * n, LOGIT_CAP);
            CK(cudaMemcpy2DAsync(all + r0, (size_t) VOCAB * 4, PF.lg, (size_t) n * 4, (size_t) n * 4, T,
                                 cudaMemcpyDeviceToHost, cudaStreamPerThread));
        }
    }
    CK(cudaStreamSynchronize(cudaStreamPerThread));
}

std::vector<int> parse_ids(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p) break;   // trailing whitespace / newline: no number, nothing pushed
        v.push_back((int) x);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// sampling on the host (the engine samples; Strata's server only forwards the keys): greedy without a temperature
struct Sampler {
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    std::mt19937_64 rng{std::random_device{}()};
};
int sample(const vec& lg, Sampler& s) {
    const int best = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    if (s.temp <= 0.0f) return best;
    const float mx = lg[best], thr = mx - 20.8f * s.temp;   // exp(-20.8) ~ 1e-9 of the top probability: never matters
    std::vector<std::pair<float, int>> c;
    for (int i = 0; i < VOCAB; ++i)
        if (lg[i] >= thr) c.push_back({lg[i], i});
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if (s.top_k > 0 && (int) c.size() > s.top_k) c.resize(s.top_k);
    std::vector<double> p(c.size());
    double z = 0.0;
    for (size_t i = 0; i < c.size(); ++i) z += p[i] = std::exp((double) (c[i].first - mx) / s.temp);
    size_t keep = c.size();
    if (s.min_p > 0.0f)
        for (size_t i = 0; i < keep; ++i) if (p[i] < s.min_p * p[0]) { keep = i; break; }
    double acc = 0.0, total = 0.0;
    for (size_t i = 0; i < keep; ++i) total += p[i];
    if (s.top_p < 1.0f) {
        for (size_t i = 0; i < keep; ++i) { acc += p[i]; if (acc >= s.top_p * total) { keep = i + 1; break; } }
        total = 0.0;
        for (size_t i = 0; i < keep; ++i) total += p[i];
    }
    double u = std::uniform_real_distribution<double>(0.0, total)(s.rng);
    for (size_t i = 0; i < keep; ++i) { u -= p[i]; if (u <= 0.0) return c[i].second; }
    return c[keep - 1].second;
}

constexpr int EOS_ID = 200006;   // <|content_model_end_sampling|> (generation_config.json)

}  // namespace

int main(int argc, char** argv) {
    if (const char* r = std::getenv("INK2_RPW")) g_rpw = std::atoi(r);
    if (const char* r = std::getenv("INK2_SEL_RPW")) g_sel_rpw = std::atoi(r);
    if (const char* r = std::getenv("INK2_GEMV_MINB")) g_gemv_min_blocks = std::atoi(r);
    std::string dir, dump_logits, rec_path;
    std::vector<int> ids;
    int n_gen = 32;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--serve") { serve = true; continue; }
        if (i + 1 >= argc) { std::fprintf(stderr, "option %s needs a value\n", a.c_str()); return 2; }
        const char* v = argv[++i];
        if (a == "--model") dir = v;
        else if (a == "--tokens") ids = parse_ids(v);
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(v, "r");
            if (!f) { std::fprintf(stderr, "cannot open %s\n", v); return 2; }
            std::string text;
            for (int c; (c = std::fgetc(f)) != EOF;) text += (char) c;
            std::fclose(f);
            ids = parse_ids(text.c_str());
        }
        else if (a == "--n") n_gen = std::atoi(v);
        else if (a == "--ctx" || a == "--max-context") E.ctx = std::atoi(v);
        else if (a == "--dump-logits") dump_logits = v;
        else if (a == "--graph") E.graph = std::atoi(v) != 0;
        else if (a == "--chunk") PF.TP = std::atoi(v);
        else if (a == "--tc") PF.tc = std::atoi(v) != 0;
        else if (a == "--kv") E.kvm = std::string(v) == "i8r" ? 3 : std::string(v) == "i8" ? 2 : std::string(v) == "f32" ? 0 : 1;
        else if (a == "--record-routing") { rec_path = v; E.rec_mode = 1; }
        else if (a == "--force-routing") { rec_path = v; E.rec_mode = 2; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (dir.empty() || (ids.empty() && !serve)) {
        std::fprintf(stderr, "usage: strata-ink2 --model DIR (--serve | --tokens IDS | --tokens-file F) [--n N] [--ctx N] "
                             "[--chunk N] [--tc 0|1] [--kv f16|f32|i8|i8r] [--dump-logits F]\n");
        return 2;
    }
    if (serve && PF.TP <= 0) { std::fprintf(stderr, "--serve needs --chunk > 0\n"); return 2; }
    if (!serve && (int) ids.size() + n_gen > E.ctx) { std::fprintf(stderr, "prompt + n exceeds --ctx %d\n", E.ctx); return 2; }
    Ckpt ck;
    ck.path = dir + "/model-00001-of-00001.safetensors";
    const std::string e = strata::image::st_read_header(ck.path, ck.st);
    if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    E.ck = &ck;
    const double t_load = now_ms();
    load();
    if (PF.TP > 0) PF_RING_EXTRA = PF.TP;
    alloc_state();
    if (PF.TP > 0) alloc_prefill();
    const size_t rec_n = (size_t) NCALLS * E.ctx * TOPK;
    if (E.rec_mode) E.rec = (int*) dalloc(rec_n * 4);
    if (E.rec_mode == 2) {
        std::vector<int> r(rec_n);
        FILE* f = std::fopen(rec_path.c_str(), "rb");
        if (!f || std::fread(r.data(), 4, rec_n, f) != rec_n) { std::fprintf(stderr, "cannot read %s (same --ctx?)\n", rec_path.c_str()); return 1; }
        std::fclose(f);
        CK(cudaMemcpy(E.rec, r.data(), rec_n * 4, cudaMemcpyHostToDevice));
    }

    vec logits(VOCAB);
    auto step = [&](int tok, int pos, bool want_logits) {
        CK(cudaMemcpyAsync(E.d_pos, &pos, 4, cudaMemcpyHostToDevice));
        upload_embed(tok);
        if (!E.graph) {
            forward_device();
        } else {
            if (!E.exec) {   // built once: every launch reads the position from E.d_pos
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
                forward_device();
                CK(cudaStreamEndCapture(cudaStreamPerThread, &g));
                CK(cudaGraphInstantiate(&E.exec, g, 0));
            }
            CK(cudaGraphLaunch(E.exec, cudaStreamPerThread));
        }
        if (want_logits) CK(cudaMemcpy(logits.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
    };
    // warm-up at position 0 (builds the decode graph outside any timing; position 0 is rewritten by the first prompt)
    step(0, 0, false);
    CK(cudaDeviceSynchronize());
    std::fprintf(stderr, "load %.1f s, %.2f GiB in VRAM (ctx %d)\n", (now_ms() - t_load) / 1000, g_vram / 1073741824.0, E.ctx);

    if (serve) {
        // Strata engine protocol (terminal/serve/server.py): READY <ctx> stop; GEN <max_new>[ k=v ...] <ids>;
        // T <id> per token; DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish>; PP <read> <total> <ms> <tok/s>;
        // STOP (read by its own thread; cleared when the next GEN starts); EOF / QUIT ends.
        std::atomic<bool> stop{false};
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> q;
        bool eof = false;
        std::thread reader([&] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line == "STOP") { stop.store(true); continue; }
                { std::lock_guard<std::mutex> lk(mu); q.push_back(line); }
                cv.notify_one();
            }
            { std::lock_guard<std::mutex> lk(mu); eof = true; }
            cv.notify_one();
        });
        reader.detach();
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::vector<int> hist;   // tokens whose KV / conv state is in the slots; last_logits follow hist.back()
        vec last_logits;
        Sampler smp;
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !q.empty(); });
                if (q.empty()) break;
                line = q.front();
                q.pop_front();
            }
            if (line == "QUIT") break;
            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            stop.store(false);
            std::vector<std::string> f;
            for (size_t p = 0; p < line.size();) {
                const size_t s2 = line.find(' ', p);
                f.push_back(line.substr(p, s2 == std::string::npos ? std::string::npos : s2 - p));
                if (s2 == std::string::npos) break;
                p = s2 + 1;
            }
            if (f.size() < 3) { std::printf("ERR malformed GEN\n"); std::fflush(stdout); continue; }
            int max_new = std::atoi(f[1].c_str());
            std::vector<int> pr = parse_ids(f.back().c_str());
            Sampler s;
            s.rng = std::move(smp.rng);
            for (size_t k = 2; k + 1 < f.size(); ++k) {   // unknown keys are ignored
                const size_t eq = f[k].find('=');
                if (eq == std::string::npos) continue;
                const std::string key = f[k].substr(0, eq), val = f[k].substr(eq + 1);
                if (key == "temperature") s.temp = std::strtof(val.c_str(), nullptr);
                else if (key == "top_p") s.top_p = std::strtof(val.c_str(), nullptr);
                else if (key == "top_k") s.top_k = std::atoi(val.c_str());
                else if (key == "min_p") s.min_p = std::strtof(val.c_str(), nullptr);
                else if (key == "seed") s.rng.seed(std::strtoull(val.c_str(), nullptr, 10));
            }
            if (pr.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); smp.rng = std::move(s.rng); continue; }
            if ((int) pr.size() >= E.ctx) {
                std::printf("ERR prompt of %zu tokens exceeds the context %d\n", pr.size(), E.ctx);
                std::fflush(stdout);
                smp.rng = std::move(s.rng);
                continue;
            }
            max_new = std::max(0, std::min(max_new, E.ctx - (int) pr.size()));
            const bool reuse = !hist.empty() && pr.size() >= hist.size() && std::equal(hist.begin(), hist.end(), pr.begin());
            const size_t start = reuse ? hist.size() : 0;
            const double t0 = now_ms();
            if (start == pr.size()) {
                logits = last_logits;
            } else {
                for (size_t p0 = start; p0 < pr.size(); p0 += PF.TP) {
                    const int T = (int) std::min((size_t) PF.TP, pr.size() - p0);
                    forward_prefill(&pr[p0], T, (int) p0, nullptr);
                    const double ms = now_ms() - t0;
                    std::printf("PP %zu %zu %.0f %.1f\n", p0 + T, pr.size(), ms, 1000.0 * (p0 + T - start) / std::max(ms, 1.0));
                    std::fflush(stdout);
                }
                CK(cudaMemcpy(logits.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
            }
            const double t1 = now_ms();
            std::vector<int> gen;
            const char* finish = "length";
            int pos = (int) pr.size();
            for (int k = 0; k < max_new; ++k) {
                if (stop.load()) { finish = "cancel"; break; }
                const int next = sample(logits, s);
                gen.push_back(next);
                std::printf("T %d\n", next);
                std::fflush(stdout);
                if (next == EOS_ID) { finish = "stop"; break; }
                if (k + 1 < max_new) step(next, pos++, true);
            }
            const double t2 = now_ms();
            hist = pr;
            if (!gen.empty()) hist.insert(hist.end(), gen.begin(), gen.end() - 1);   // the last token was never fed
            last_logits = logits;
            smp.rng = std::move(s.rng);
            std::printf("DONE %zu %zu %.0f %.0f %s\n", gen.size(), pr.size(), t1 - t0, t2 - t1, finish);
            std::fflush(stdout);
        }
        return 0;
    }

    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    const double t0 = now_ms();
    if (PF.TP > 0) {   // chunked prefill
        std::vector<float> all(fg ? (size_t) PF.TP * VOCAB : 0);
        for (size_t p0 = 0; p0 < ids.size(); p0 += PF.TP) {
            const int T = (int) std::min((size_t) PF.TP, ids.size() - p0);
            forward_prefill(&ids[p0], T, (int) p0, fg ? all.data() : nullptr);
            if (fg) std::fwrite(all.data(), 4, (size_t) T * VOCAB, fg);
        }
        CK(cudaMemcpy(logits.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
    } else {
        for (size_t t = 0; t < ids.size(); ++t) {
            step(ids[t], (int) t, fg != nullptr || t + 1 == ids.size());
            if (fg) std::fwrite(logits.data(), 4, logits.size(), fg);
        }
    }
    CK(cudaDeviceSynchronize());
    const double t1 = now_ms();
    if (fg) std::fclose(fg);
    std::printf("gen:");
    int pos = (int) ids.size();
    for (int k = 0; k < n_gen; ++k) {
        const int next = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::printf(" %d", next);
        step(next, pos++, true);
    }
    CK(cudaDeviceSynchronize());
    const double t2 = now_ms();
    if (E.rec_mode == 1) {
        std::vector<int> r(rec_n);
        CK(cudaMemcpy(r.data(), E.rec, rec_n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen(rec_path.c_str(), "wb");
        std::fwrite(r.data(), 4, rec_n, f);
        std::fclose(f);
    }
    std::printf("\nprompt %zu tokens %.0f ms (%.1f tok/s), decode %d tokens %.0f ms (%.1f tok/s)\n", ids.size(), t1 - t0,
                1000.0 * ids.size() / (t1 - t0), n_gen, t2 - t1, n_gen > 0 ? 1000.0 * n_gen / (t2 - t1) : 0.0);
    return 0;
}
