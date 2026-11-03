// src/kernels/cuda/rdna_gr.cu - the RDNA gfx1100 hyper-connection read: split-K down + partial-folding up, from
// nicejunjie/Strata-rdna3's rdna_gr.cu (its BF16 half only; the int8 / A8 / fused-barrier halves are not here).
// Same maths as fused_gr_read_multi, another summation order (opt-in, STRATA_GR_RDNA=1):
//   down: block (8-row group, stream c, half h) dots its rows with R'[c,h] * w_norm[c,h] (no rms yet) and, for row
//         group 0, writes the slice's sum of squares;  w_down . xn = sum_c rs[c] * partial[c]  (rs is per stream).
//   up:   every block folds the 8 partials into lo (and block 0 into inject_out / rs) in its prologue, then four lanes
//         per w_up row, two xor steps per token.  No counters, no inter-block order: safe on any stream.
#include "strata/core/geometry.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#ifndef STRATA_GR_WPE
#define STRATA_GR_WPE 8
#endif
namespace strata::kernels {
namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, PR = LR + HC;   // partial rows: 320 down + 4 inject
constexpr int H = N / 2;                  // half-stream slice, 1280
constexpr int TH = 256, WV = TH / 32;     // threads, waves (wave32)
constexpr int HQ = H / 8 / 32;            // uint4 per lane per row slice: 5
constexpr int UC = 16;                    // up: columns per block (x 4 streams = 64 rows = 256 quarter-rows)
constexpr int UB = N / UC;                // 160 blocks
static_assert(N == strata::core::kCompiledGeometry.n_embd && HC == strata::core::kCompiledGeometry.hc &&
                  LR == strata::core::kCompiledGeometry.hc_lr, "rdna_gr is compiled for core/geometry.hpp");

struct RdnaGr { FusedGrArgs a[kFusedGrMaxT]; float* part; float* ss; };

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ntload(const uint4* p) {
    const u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
// wave32 all-reduce with DPP and permlanex16 (no LDS round trip, unlike __shfl_xor's ds_bpermute)
#define STRATA_DPP(v, ctrl) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xf, 0xf, false))
__device__ __forceinline__ float wsum(float v) {
    v += STRATA_DPP(v, 0xB1);    // quad_perm [1,0,3,2]
    v += STRATA_DPP(v, 0x4E);    // quad_perm [2,3,0,1]
    v += STRATA_DPP(v, 0x141);   // row_half_mirror: 8-lane sums
    v += STRATA_DPP(v, 0x140);   // row_mirror: 16-lane sums
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
    return v;
}
__device__ __forceinline__ float dot8(const uint4 w, const float4 x0, const float4 x1, float acc) {
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc); acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc); acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc); acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc); acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

template <int T, int RW>
__device__ __forceinline__ void down_body(const RdnaGr& m, int rg, int slice) {
    __shared__ __align__(16) float xs[T][H];
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int c = slice >> 1, h = slice & 1;   // slice = 2c + h
    uint4 wv4[RW][HQ];
    bool act[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        act[r] = row < LR || (row < PR && m.a[0].w_inject != nullptr);
        const uint16_t* wbase = row < LR ? m.a[0].w_down + (size_t) row * D : m.a[0].w_inject + (size_t) (row - LR) * D;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + c * N + h * H);
        if (act[r]) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv4[r][q] = ntload(w4 + lane + 32 * q);
        }
    }
    const int e0 = c * N + h * H;   // this slice's first element in D
    // stage R'[c,h] * w_norm for every token: one flattened (token, float4) loop, loads of several tokens in flight
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
            *reinterpret_cast<float4*>(&xs[k][i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    if (rg == 0) {   // the slice's sums of squares (only row group 0 publishes them)
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));   // explicit: the same in every instance
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
    // tokens in pairs: the weights stay in registers, the accumulators do not grow with T
#pragma unroll 1
    for (int k0 = 0; k0 < T; k0 += 2) {
        const bool two = k0 + 1 < T;
        float acc0[RW], acc1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int j = (lane + 32 * q) * 8;
            const float4 a0 = *reinterpret_cast<const float4*>(&xs[k0][j]);
            const float4 a1 = *reinterpret_cast<const float4*>(&xs[k0][j + 4]);
            float4 b0 = make_float4(0.f, 0.f, 0.f, 0.f), b1 = b0;
            if (two) { b0 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j]); b1 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j + 4]); }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc0[r] = dot8(wv4[r][q], a0, a1, acc0[r]);
                acc1[r] = dot8(wv4[r][q], b0, b1, acc1[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v0 = wsum(acc0[r]), v1 = wsum(acc1[r]);
            if (act[r] && lane == 0) m.part[((size_t) k0 * 2 * HC + slice) * PR + row] = v0;
            if (act[r] && lane == 1 && two) m.part[((size_t) (k0 + 1) * 2 * HC + slice) * PR + row] = v1;
        }
    }
}

template <int T, int RW>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_down(RdnaGr m) {
    down_body<T, RW>(m, (int) blockIdx.x, (int) blockIdx.y);
}

// ---- STRATA_GR_RDNA_F16=1: the same two passes on the FP16 packed dot (`v_dot2c_f32_f16`, which gfx1100 HAS; the
// BF16 one it rejects).  The weights stay BF16 in memory and are converted in registers once per (row, lane) - bf16 ->
// fp16 is exact for every value in fp16's range - and the staged activations are written as fp16 pairs, so an element
// costs half a dot2 instead of a shift and an fma.  Another summation order, and the activations round to fp16
// (~5e-4 relative; the products still accumulate in fp32).
//
// MEASURED, NOT DEPLOYED: the dot is only part of the kernel.  32K prod switches, stage table: hc0 down 0.66 -> 0.62,
// hc0 up 0.65 -> 0.63, hc1 0.64/0.64 -> 0.61/0.62 (ms per 36 GDN layers), total 22.11 -> 22.04, ms_fwd 25.89 ->
// 25.63 - about 1% for a rounding change that moved the acceptance 161/219 -> 160/220.  The other two thirds of the
// kernel are the R'*w_norm staging (one __expf per float4 for `apply`), the sum-of-squares pass and the loads.
__device__ __forceinline__ unsigned pk_f16(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ unsigned bf2f16(unsigned w) {   // two BF16 in one word -> two FP16 in one word
    return pk_f16(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}
__device__ __forceinline__ uint4 bf4f16(uint4 w) {
    return make_uint4(bf2f16(w.x), bf2f16(w.y), bf2f16(w.z), bf2f16(w.w));
}
__device__ __forceinline__ float d2f16(unsigned w, unsigned x, float acc) {
    asm volatile("v_dot2c_f32_f16 %0, %1, %2" : "=v"(acc) : "v"(w), "v"(x), "0"(acc));
    return acc;
}
__device__ __forceinline__ float dot8h(uint4 w, uint4 x, float acc) {   // 8 elements: 4 packed dots
    return d2f16(w.x, x.x, d2f16(w.y, x.y, d2f16(w.z, x.z, d2f16(w.w, x.w, acc))));
}

template <int T, int RW>
__device__ __forceinline__ void down_body_f16(const RdnaGr& m, int rg, int slice) {
    __shared__ __align__(16) unsigned xs[T][H / 2];   // the staged activations, two fp16 a word
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int c = slice >> 1, h = slice & 1;   // slice = 2c + h
    uint4 wv4[RW][HQ];
    bool act[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        act[r] = row < LR || (row < PR && m.a[0].w_inject != nullptr);
        const uint16_t* wbase = row < LR ? m.a[0].w_down + (size_t) row * D : m.a[0].w_inject + (size_t) (row - LR) * D;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + c * N + h * H);
        if (act[r]) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv4[r][q] = bf4f16(ntload(w4 + lane + 32 * q));
        }
    }
    const int e0 = c * N + h * H;   // this slice's first element in D
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
        *reinterpret_cast<uint2*>(&xs[k][i / 2]) =
            make_uint2(pk_f16(r.x * g.x, r.y * g.y), pk_f16(r.z * g.z, r.w * g.w));
    }
    if (rg == 0) {   // the slice's sums of squares (only row group 0 publishes them)
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
#pragma unroll 1
    for (int k0 = 0; k0 < T; k0 += 2) {
        const bool two = k0 + 1 < T;
        float acc0[RW], acc1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int j = (lane + 32 * q) * 4;   // words: 8 elements
            const uint4 a0 = *reinterpret_cast<const uint4*>(&xs[k0][j]);
            const uint4 b0 = two ? *reinterpret_cast<const uint4*>(&xs[k0 + 1][j]) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc0[r] = dot8h(wv4[r][q], a0, acc0[r]);
                acc1[r] = dot8h(wv4[r][q], b0, acc1[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v0 = wsum(acc0[r]), v1 = wsum(acc1[r]);
            if (act[r] && lane == 0) m.part[((size_t) k0 * 2 * HC + slice) * PR + row] = v0;
            if (act[r] && lane == 1 && two) m.part[((size_t) (k0 + 1) * 2 * HC + slice) * PR + row] = v1;
        }
    }
}

template <int T, int RW>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_down_f16(RdnaGr m) {
    down_body_f16<T, RW>(m, (int) blockIdx.x, (int) blockIdx.y);
}

// STRATA_QFUSE: the consumer's q8_1 image of `mixed`, as fused_gr.cu's gr_q8_tail - the two blocks of a 32-column
// group count in, the last one quantizes the group (one wave per token, the same bytes as native_quantize_q8_1)
struct RdnaQ81 { half2 ds; int8_t qs[32]; };
template <int T>
__device__ __forceinline__ void q8_tail(const RdnaGr& m, int d0) {
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
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (wave >= T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[wave].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    RdnaQ81* y = reinterpret_cast<RdnaQ81*>(m.a[wave].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

template <int T>
__device__ __forceinline__ void up_body(const RdnaGr& m, const int bx) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rs[T][HC];
    __shared__ float g[T][HC][UC];
    const int t = threadIdx.x;
    // four lanes per row: quarter q reads uint4 q, q+4, ..., q+36 (10 of the row's 40) - issued first, so the DRAM
    // latency overlaps the partial folding below
    const int rr = t >> 2, q = t & 3;          // 64 rows: stream c = rr / UC, column dd = rr % UC
    const int c = rr / UC, dd = rr - c * UC, d = bx * UC + dd, i = c * N + d;
    uint4 w[10];
    {
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int u = 0; u < 10; ++u) w[u] = ntload(w4 + q + 4 * u);
    }
    float rv[T], wn = m.a[0].w_norm[i];
#pragma unroll
    for (int k = 0; k < T; ++k) rv[k] = m.a[k].R[i];
    if (t < T * HC) {
        const int k = t / HC, cs = t - k * HC;
        const float s = m.ss[k * 2 * HC + 2 * cs] + m.ss[k * 2 * HC + 2 * cs + 1];
        rs[k][cs] = rsqrtf(s / (float) N + m.a[k].eps);
        if (bx == 0) m.a[k].rs[cs] = rs[k][cs];
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += TH) {
        const int k = i / PR, j = i - k * PR;
        const float* p = m.part + (size_t) k * 2 * HC * PR + j;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s = fmaf(rs[k][cc], p[(2 * cc) * PR] + p[(2 * cc + 1) * PR], s);
        if (j < LR) {
            const float x = s / (float) HC;
            lo[k][j] = x / (1.0f + __expf(-x));
            if (bx == 0) m.a[k].lo[j] = lo[k][j];
        } else if (bx == 0 && m.a[0].w_inject != nullptr) {
            m.a[k].inject_out[j - LR] = s;
        }
    }
    __syncthreads();
    float acc[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        float a = 0.0f;
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const int j = (q + 4 * u) * 8;
            a = dot8(w[u], *reinterpret_cast<const float4*>(&lo[k][j]), *reinterpret_cast<const float4*>(&lo[k][j + 4]), a);
        }
        a += __shfl_xor(a, 1);
        a += __shfl_xor(a, 2);
        acc[k] = a;
    }
    if (q == 0) {
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            float r = rv[k];
            if (a.apply) {
                r = fmaf(a.bo_prev[d], 2.0f * sigm(a.inj_prev[c] / (float) HC), r);
                a.R_out[i] = r;
            }
            g[k][c][dd] = r * wn * rs[k][c] * sigm(acc[k]);
        }
    }
    __syncthreads();
    if (t < T * UC) {
        const int k = t / UC, col = t - k * UC;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s += g[k][cc][col];
        m.a[k].mixed[bx * UC + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) q8_tail<T>(m, bx * UC);   // STRATA_QFUSE
}

template <int T>
__global__ void __launch_bounds__(TH) rdna_gr_up(RdnaGr m) {
    up_body<T>(m, blockIdx.x);
}

// STRATA_GR_RDNA_F16=1: up_body with the fp16 packed dot - `lo` keeps its float copy (it is an output) and gets an
// fp16 image beside it, so the row's dot is 40 packed dots a token instead of 80 unpack+fma pairs.
template <int T>
__device__ __forceinline__ void up_body_f16(const RdnaGr& m, const int bx) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ __align__(16) unsigned loh[T][LR / 2];
    __shared__ float rs[T][HC];
    __shared__ float g[T][HC][UC];
    const int t = threadIdx.x;
    const int rr = t >> 2, q = t & 3;          // 64 rows: stream c = rr / UC, column dd = rr % UC
    const int c = rr / UC, dd = rr - c * UC, d = bx * UC + dd, i = c * N + d;
    uint4 w[10];
    {
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int u = 0; u < 10; ++u) w[u] = bf4f16(ntload(w4 + q + 4 * u));
    }
    float rv[T], wn = m.a[0].w_norm[i];
#pragma unroll
    for (int k = 0; k < T; ++k) rv[k] = m.a[k].R[i];
    if (t < T * HC) {
        const int k = t / HC, cs = t - k * HC;
        const float s = m.ss[k * 2 * HC + 2 * cs] + m.ss[k * 2 * HC + 2 * cs + 1];
        rs[k][cs] = rsqrtf(s / (float) N + m.a[k].eps);
        if (bx == 0) m.a[k].rs[cs] = rs[k][cs];
    }
    __syncthreads();
    for (int i2 = t; i2 < T * PR; i2 += TH) {
        const int k = i2 / PR, j = i2 - k * PR;
        const float* p = m.part + (size_t) k * 2 * HC * PR + j;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s = fmaf(rs[k][cc], p[(2 * cc) * PR] + p[(2 * cc + 1) * PR], s);
        if (j < LR) {
            const float x = s / (float) HC;
            lo[k][j] = x / (1.0f + __expf(-x));
            if (bx == 0) m.a[k].lo[j] = lo[k][j];
        } else if (bx == 0 && m.a[0].w_inject != nullptr) {
            m.a[k].inject_out[j - LR] = s;
        }
    }
    __syncthreads();
    // the fp16 image, pairs at a time (thread t takes the floats t, t + TH, ...; the pairs are rebuilt below so the
    // order does not matter - every element is written once by its own thread's neighbour)
    for (int i2 = t * 2; i2 < T * LR; i2 += TH * 2) {
        const int k = i2 / LR, j = i2 - k * LR;
        loh[k][j / 2] = pk_f16(lo[k][j], lo[k][j + 1]);
    }
    __syncthreads();
    float acc[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        float a = 0.0f;
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const int j = (q + 4 * u) * 8;
            a = dot8h(w[u], *reinterpret_cast<const uint4*>(&loh[k][j / 2]), a);
        }
        a += __shfl_xor(a, 1);
        a += __shfl_xor(a, 2);
        acc[k] = a;
    }
    if (q == 0) {
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            float r = rv[k];
            if (a.apply) {
                r = fmaf(a.bo_prev[d], 2.0f * sigm(a.inj_prev[c] / (float) HC), r);
                a.R_out[i] = r;
            }
            g[k][c][dd] = r * wn * rs[k][c] * sigm(acc[k]);
        }
    }
    __syncthreads();
    if (t < T * UC) {
        const int k = t / UC, col = t - k * UC;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s += g[k][cc][col];
        m.a[k].mixed[bx * UC + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) q8_tail<T>(m, bx * UC);   // STRATA_QFUSE
}

template <int T>
__global__ void __launch_bounds__(TH) rdna_gr_up_f16(RdnaGr m) {
    up_body_f16<T>(m, blockIdx.x);
}

// STRATA_GR_FUSED=1 (opt-in): down and up in ONE launch - every block runs its down slice, a sense-reversing grid
// barrier (the down grid's 168 blocks are co-resident: two blocks a CU at 8 waves, 113 VGPRs), then the first UB blocks
// run the up pass.  The same two bodies, the same arithmetic per element as the two launches; one graph node less per
// read (96 fewer a window), which is worth ~6 us each on gfx1100.  The barrier words are module-static (zeroed at
// load) and self-resetting (a generation counter), so no allocation happens inside a graph capture.
__device__ unsigned g_gr_bar[2];
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* gen = bar + 1;
        const unsigned g = *gen;
        __threadfence();
        if (atomicAdd(bar, 1u) == nblocks - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*gen == g) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    __threadfence();   // every wave: the other blocks' partials and sums
}
template <int T>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_fused(RdnaGr m) {
    constexpr int NBX = (PR + 2 * WV - 1) / (2 * WV);   // the down grid's x extent (row groups)
    const int b = (int) blockIdx.x;
    down_body<T, 2>(m, b % NBX, b / NBX);
    grid_barrier(g_gr_bar, gridDim.x);
    if (b < UB) up_body<T>(m, b);
}


template <int T>
void launch(const RdnaGr& m, cudaStream_t st, unsigned long long* sb, int si) {
    static const int only = [] { const char* v = std::getenv("STRATA_GR_RDNA_ONLY"); return v ? std::atoi(v) : 0; }();   // bench
    static const int rw = [] { const char* v = std::getenv("STRATA_GR_RDNA_RW"); return v ? std::atoi(v) : 2; }();
    // STRATA_GR_FUSED: one launch for both passes, only when every down block is co-resident (a grid barrier deadlocks
    // otherwise) - the occupancy is asked of the driver, and the profiler's two stamps need the two-launch path.
    static const bool g_fused = [] { const char* v = std::getenv("STRATA_GR_FUSED"); return v != nullptr && v[0] == '1'; }();
    static const int g_fused_ok = [] {
        if (!g_fused) return 0;
        int nblk = 0, cus = 0;
#if defined(__HIPCC__)
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, (const void*) &rdna_gr_fused<T>, TH, 0) != hipSuccess) return 0;
#else
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, rdna_gr_fused<T>, TH, 0) != cudaSuccess) return 0;
#endif
        if (cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return 0;
        const int need = ((PR + 2 * WV - 1) / (2 * WV)) * 2 * HC;
        if (nblk * cus >= need) return 1;
        std::fprintf(stderr, "strata gr: STRATA_GR_FUSED off (%d blocks x %d CU < %d co-resident)\n", nblk, cus, need);
        return 0;
    }();
    if (g_fused_ok && only == 0 && sb == nullptr) {
        rdna_gr_fused<T><<<((PR + 2 * WV - 1) / (2 * WV)) * 2 * HC, TH, 0, st>>>(m);
        return;
    }
    // STRATA_GR_RDNA_F16: the fp16 packed-dot pair (same shape, same outputs apart from the fp16 rounding)
    static const bool g_f16 = [] { const char* v = std::getenv("STRATA_GR_RDNA_F16"); return v != nullptr && v[0] == '1'; }();
    if (g_f16 && rw == 2) {
        if (sb) gpu_stamp(sb, si, st);
        if (only != 2) rdna_gr_down_f16<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m);
        if (sb) gpu_stamp(sb, si + 1, st);
        if (only != 1) rdna_gr_up_f16<T><<<UB, TH, 0, st>>>(m);
        return;
    }
    if (sb) gpu_stamp(sb, si, st);
    if (only != 2) {
        if (rw == 1) rdna_gr_down<T, 1><<<dim3((PR + WV - 1) / WV, 2 * HC), TH, 0, st>>>(m);
        else if (rw == 4) rdna_gr_down<T, 4><<<dim3((PR + 4 * WV - 1) / (4 * WV), 2 * HC), TH, 0, st>>>(m);
        else rdna_gr_down<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m);
    }
    if (sb) gpu_stamp(sb, si + 1, st);
    if (only != 1) rdna_gr_up<T><<<UB, TH, 0, st>>>(m);
}
}  // namespace

bool rdna_gr_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GR_RDNA"); return v != nullptr && v[0] == '1'; }();
    return on;
}

// scratch: at least T * 2*HC * (PR + 1) floats (the caller's xn scratch, T * HC * N floats, is far larger)
void rdna_gr_read_multi(const FusedGrArgs* a, int n_tok, float* scratch, void* stream, unsigned long long* sb, int si) {
    RdnaGr m;
    for (int t = 0; t < n_tok; ++t) m.a[t] = a[t];
    m.part = scratch;
    m.ss = scratch + (size_t) n_tok * 2 * HC * PR;
    cudaStream_t st = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: launch<1>(m, st, sb, si); break;
        case 2: launch<2>(m, st, sb, si); break;
        case 3: launch<3>(m, st, sb, si); break;
        case 4: launch<4>(m, st, sb, si); break;
        case 5: launch<5>(m, st, sb, si); break;
        case 6: launch<6>(m, st, sb, si); break;
        case 7: launch<7>(m, st, sb, si); break;
        default: launch<8>(m, st, sb, si); break;
    }
}
}  // namespace strata::kernels
