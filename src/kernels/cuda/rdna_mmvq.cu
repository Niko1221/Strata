// src/kernels/cuda/rdna_mmvq.cu - gfx1100 multi-column Q6_K x q8_1 decode GEMV (STRATA_RDNA_Q6K=1, opt-in).
// One wave per output row; per superblock every lane takes 8 consecutive weights (half n, quadrant qd, 8 l's), decodes
// them once (ql nibble | qh 2 bits, -32) and applies them to every column with two dp4a.  The same products and
// per-16 scales as ggml's vec_dot_q6_K_q8_1, another summation order (not bitwise the reference kernel).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"   // iq3s_grid (IQ3_S)

namespace strata::kernels {
bool rdna_q6k_mmvq2(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                    const void* w2, float* y2, int n_out2);
bool rdna_kq_mmvq2(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                   const void* w2, float* y2, int n_out2);
namespace {
struct Q81 { half2 ds; int8_t qs[32]; };
constexpr int QK = 256, BB = 210;   // Q6_K: ql[128] qh[64] scales[16] d(2)

__device__ __forceinline__ float wsum32(float v) {
#define STRATA_DPP6(v, c) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), c, 0xf, 0xf, false))
    v += STRATA_DPP6(v, 0x161); v += STRATA_DPP6(v, 0x162); v += STRATA_DPP6(v, 0x164); v += STRATA_DPP6(v, 0x168);
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
    return v;
}
__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) {   // gfx11: unaligned dword loads are fine
    return *reinterpret_cast<const uint32_t*>(p);
}

template <int NC, int UNR = (NC <= 2 ? 5 : NC <= 4 ? 3 : 2)>
__global__ void __launch_bounds__(256) rdna_q6k_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out, const uint8_t* __restrict__ w2, float* __restrict__ y2, int n_out2) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w2 != nullptr && row >= n_out) {   // STRATA_RDNA_PAIR: rows past the first matrix are the second's
        row -= n_out; w = w2; y = y2; n_out = n_out2;
    }
    if (row >= n_out) return;
    const int nsb = n_in / QK, xs = n_in / 32;
    const int n = lane >> 4, qd = (lane >> 2) & 3, l = (lane & 3) * 8;
    const int ql_off = n * 64 + (qd & 1) * 32 + l, qh_off = 128 + n * 32 + l, sc_off = 192 + n * 8 + (l >> 4) + 2 * qd;
    const int xb = n * 4 + qd, xi = l >> 2;     // the q8_1 block of these 8 weights inside the superblock, int index
    const int hs = 2 * qd, ns = (qd >= 2) ? 4 : 0;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    const uint8_t* wr = w + (size_t) row * nsb * BB;
#pragma unroll UNR
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = wr + (size_t) sb * BB;
        const uint32_t l0 = ld32u(b + ql_off), l1 = ld32u(b + ql_off + 4);
        const uint32_t h0 = ld32u(b + qh_off), h1 = ld32u(b + qh_off + 4);
        const int sc = (int) (int8_t) b[sc_off];
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        // 8 six-bit weights as two packed int8x4, minus 32 (signed saturating like ggml's __vsubss4)
        const uint32_t lo0 = (l0 >> ns) & 0x0f0f0f0fu, lo1 = (l1 >> ns) & 0x0f0f0f0fu;
        const uint32_t hi0 = ((h0 >> hs) & 0x03030303u) << 4, hi1 = ((h1 >> hs) & 0x03030303u) << 4;
        // per-byte x - 32 for x in 0..63 without a borrow across bytes: set each byte's top bit, subtract, flip it back
        const int w0 = (int) ((((lo0 | hi0) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
        const int w1 = (int) ((((lo1 | hi1) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
        const float ds = d * (float) sc;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Q81* xb8 = x + (size_t) c * xs + sb * 8 + xb;
            const int* q = reinterpret_cast<const int*>(xb8->qs) + xi;
            int s = __builtin_amdgcn_sudot4(true, w0, true, q[0], 0, false);
            s = __builtin_amdgcn_sudot4(true, w1, true, q[1], s, false);
            acc[c] = fmaf(ds * __low2float(xb8->ds), (float) s, acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = wsum32(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}

// ---- Q4_K / Q5_K / IQ4_XS: lane = (sub-block j = lane / 4, 8 weights l0 = 8 (lane % 4)); weights decoded once per
// superblock; K-quants take the min term from the lane's own activation sum (dot with ones) - same products,
// another summation order than ggml's vec_dot.
__device__ __forceinline__ int dot2(int w0, int w1, const int* q, int acc) {
    acc = __builtin_amdgcn_sudot4(true, w0, true, q[0], acc, false);
    return __builtin_amdgcn_sudot4(true, w1, true, q[1], acc, false);
}
__device__ __forceinline__ void kq_scale(const uint8_t* s, int j, int& sc, int& m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}
__device__ __forceinline__ uint32_t iq4_lut4(uint32_t idx) {   // 4 nibble indices (one per byte) -> 4 int8 table values
    const uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;   // -127..-10 | 1..113
    const uint32_t sel = idx & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel), hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t msk = ((idx >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~msk) | (hi & msk);
}
template <int FMT> struct KFmt;   // 12 Q4_K, 13 Q5_K, 23 IQ4_XS
template <> struct KFmt<12> { static constexpr int B = 144; };
template <> struct KFmt<13> { static constexpr int B = 176; };
template <> struct KFmt<23> { static constexpr int B = 136; };
template <> struct KFmt<21> { static constexpr int B = 110; };   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]

template <int FMT, int NC, int UNR = (FMT == 21 ? 2 : (NC <= 2 ? 5 : NC <= 4 ? 3 : 2))>
__global__ void __launch_bounds__(256) rdna_kq_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                      float* __restrict__ y, int n_in, int n_out, const uint8_t* __restrict__ w2, float* __restrict__ y2, int n_out2) {
    constexpr int BB_ = KFmt<FMT>::B;
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    // IQ3_S: the 512-entry grid staged in LDS (2 KB) before any early return
    __shared__ uint32_t s_grid[FMT == 21 ? 512 : 1];
    if constexpr (FMT == 21) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_grid[i] = iq3s_grid[i];
        __syncthreads();
    }
    if (w2 != nullptr && row >= n_out) {   // STRATA_RDNA_PAIR: rows past the first matrix are the second's
        row -= n_out; w = w2; y = y2; n_out = n_out2;
    }
    if (row >= n_out) return;
    const int nsb = n_in / QK, xs = n_in / 32;
    const int j = lane >> 2, l0 = (lane & 3) * 8;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    const uint8_t* wr = w + (size_t) row * nsb * BB_;
#pragma unroll UNR
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = wr + (size_t) sb * BB_;
        int w0, w1;
        float f1, f2 = 0.0f;   // acc += d8 * (f1 * dot(w, q) - f2 * dot(1, q))
        if constexpr (FMT == 21) {
            // IQ3_S: lane (j, l) owns values 8l..8l+7 of 32-block j: two grid entries (high bit from qh[j]), signs
            // byte 4j + l, scale 1 + 2 * nibble j of scales[j / 2] (vec_dot_iq3_s_q8_1's decode, per lane)
            const int l = lane & 3;
            const uint8_t* qs = b + 2 + 8 * j + 2 * l;
            const int qh = b[66 + j];
            const int sg = b[74 + 4 * j + l];
            const int g1 = (int) s_grid[qs[0] | ((qh << (8 - 2 * l)) & 0x100)];
            const int g2 = (int) s_grid[qs[1] | ((qh << (7 - 2 * l)) & 0x100)];
            const int sg0 = __vcmpne4(((sg & 0x03) << 7) | ((sg & 0x0C) << 21), 0x00000000);
            const int sg1 = __vcmpne4(((sg & 0x30) << 3) | ((sg & 0xC0) << 17), 0x00000000);
            w0 = __vsub4(g1 ^ sg0, sg0);
            w1 = __vsub4(g2 ^ sg1, sg1);
            const int sc = (b[106 + j / 2] >> (4 * (j & 1))) & 0x0f;
            f1 = __half2float(*reinterpret_cast<const __half*>(b)) * (float) (1 + 2 * sc);
        } else if constexpr (FMT == 23) {
            const int hl = l0 >= 16 ? 4 : 0;
            const uint32_t q0 = ld32u(b + 8 + 16 * j + (l0 & 15)), q1 = ld32u(b + 8 + 16 * j + (l0 & 15) + 4);
            w0 = (int) iq4_lut4((q0 >> hl) & 0x0f0f0f0fu);
            w1 = (int) iq4_lut4((q1 >> hl) & 0x0f0f0f0fu);
            const uint16_t sh = *reinterpret_cast<const uint16_t*>(b + 2);
            const int ls = ((b[4 + j / 2] >> (4 * (j & 1))) & 0xf) | (((sh >> (2 * j)) & 3) << 4);
            f1 = __half2float(*reinterpret_cast<const __half*>(b)) * (float) (ls - 32);
        } else {
            const int qo = (FMT == 12 ? 16 : 48) + (j >> 1) * 32 + l0, ns = 4 * (j & 1);
            uint32_t v0 = (ld32u(b + qo) >> ns) & 0x0f0f0f0fu, v1 = (ld32u(b + qo + 4) >> ns) & 0x0f0f0f0fu;
            if constexpr (FMT == 13) {
                v0 |= ((ld32u(b + 16 + l0) >> j) & 0x01010101u) << 4;
                v1 |= ((ld32u(b + 16 + l0 + 4) >> j) & 0x01010101u) << 4;
            }
            w0 = (int) v0; w1 = (int) v1;
            // the 12 scale bytes as three dwords (the same for every lane), bytes picked arithmetically, no branches
            const uint32_t s0 = ld32u(b + 4), s1 = ld32u(b + 8), s2 = ld32u(b + 12);
            auto sb8 = [&](int k) -> int {
                const uint32_t w = k < 4 ? s0 : (k < 8 ? s1 : s2);
                return (int) ((w >> (8 * (k & 3))) & 0xffu);
            };
            int sc, m;
            if (j < 4) { sc = sb8(j) & 63; m = sb8(j + 4) & 63; }
            else { sc = (sb8(j + 4) & 0xF) | ((sb8(j - 4) >> 6) << 4); m = (sb8(j + 4) >> 4) | ((sb8(j) >> 6) << 4); }
            const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
            f1 = dm.x * (float) sc; f2 = dm.y * (float) m;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Q81* xb8 = x + (size_t) c * xs + sb * 8 + j;
            const int* q = reinterpret_cast<const int*>(xb8->qs) + (l0 >> 2);
            const float d8 = __low2float(xb8->ds);
            const int s1 = dot2(w0, w1, q, 0);
            if constexpr (FMT == 23 || FMT == 21) acc[c] = fmaf(d8 * f1, (float) s1, acc[c]);
            else {
                const int s2 = dot2(0x01010101, 0x01010101, q, 0);
                acc[c] = fmaf(d8, f1 * (float) s1 - f2 * (float) s2, acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = wsum32(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}
template <int FMT>
void kq_launch(const uint8_t* wb, const Q81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s,
               const uint8_t* w2 = nullptr, float* y2 = nullptr, int n_out2 = 0) {
    const unsigned blocks = (unsigned) ((n_out + 7) / 8 + (w2 ? (n_out2 + 7) / 8 : 0));
    switch (ncols) {
        case 1: rdna_kq_kernel<FMT, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 2: rdna_kq_kernel<FMT, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 3: rdna_kq_kernel<FMT, 3><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 4: rdna_kq_kernel<FMT, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 5: rdna_kq_kernel<FMT, 5><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 6: rdna_kq_kernel<FMT, 6><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 7: rdna_kq_kernel<FMT, 7><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        default: rdna_kq_kernel<FMT, 8><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
    }
}
}  // namespace

bool rdna_kq_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    return rdna_kq_mmvq2(type, w, xq, y, n_in, n_out, ncols, stream, nullptr, nullptr, 0);
}
bool rdna_kq_mmvq2(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                   const void* w2v, float* y2, int n_out2) {
    if (n_in % QK != 0 || ncols < 1 || ncols > 8) return false;
    if (w2v != nullptr && n_out % 8 != 0) return false;
    const auto* w2 = static_cast<const uint8_t*>(w2v);
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* x = static_cast<const Q81*>(xq);
    cudaStream_t s = (cudaStream_t) stream;
    switch (type) {
        // Q4_K / Q5_K: measured slower than the native wave kernels on gfx1100 - not taken (kept for the bench)
        case 12: if (ncols < 4 && std::getenv("STRATA_RDNA_KQ_ALL") == nullptr) return false;   // faster than the wave kernel from 4 columns
            kq_launch<12>(wb, x, y, n_in, n_out, ncols, s, w2, y2, n_out2); return true;
        case 13: if (ncols < 4 && std::getenv("STRATA_RDNA_KQ_ALL") == nullptr) return false;
            kq_launch<13>(wb, x, y, n_in, n_out, ncols, s, w2, y2, n_out2); return true;
        case 23: kq_launch<23>(wb, x, y, n_in, n_out, ncols, s, w2, y2, n_out2); return true;
        case 21: kq_launch<21>(wb, x, y, n_in, n_out, ncols, s, w2, y2, n_out2); return true;
        default: return false;
    }
}
namespace {
}  // namespace

bool rdna_q6k_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_RDNA_Q6K"); return v != nullptr && v[0] == '1'; }();
    return on;
}

bool rdna_q6k_mmvq(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    return rdna_q6k_mmvq2(w, xq, y, n_in, n_out, ncols, stream, nullptr, nullptr, 0);
}
bool rdna_q6k_mmvq2(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream,
                    const void* w2v, float* y2, int n_out2) {
    if (n_in % QK != 0 || ncols < 1 || ncols > 8) return false;
    if (w2v != nullptr && n_out % 8 != 0) return false;   // the second matrix starts on a block boundary
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* w2 = static_cast<const uint8_t*>(w2v);
    const auto* x = static_cast<const Q81*>(xq);
    const unsigned blocks = (unsigned) ((n_out + 7) / 8 + (w2 ? (n_out2 + 7) / 8 : 0));
    cudaStream_t s = (cudaStream_t) stream;
    switch (ncols) {
        case 1: rdna_q6k_kernel<1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 2: rdna_q6k_kernel<2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 3: rdna_q6k_kernel<3><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 4: rdna_q6k_kernel<4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 5: rdna_q6k_kernel<5><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 6: rdna_q6k_kernel<6><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        case 7: rdna_q6k_kernel<7><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
        default: rdna_q6k_kernel<8><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, w2, y2, n_out2); break;
    }
    return true;
}
}  // namespace strata::kernels
