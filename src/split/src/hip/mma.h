// HIP (gfx12) stand-in for CUDA's <mma.h>: nvcuda::wmma's 16x16x16 half / float fragments on RDNA4's
// v_wmma_f32_16x16x16_f16 (wave32). The fragment layout is the hardware's (checked on an R9700 by results/gdn/GDN-A/micro/wmma12.cpp,
// src/prefill/gdn_wmma12.inc): A and B take 8 halves a lane, lane l holding row (A) or column (B) l & 15 and k = 8 * (l >> 4) + 0..7;
// the accumulator lane l holds column l & 15 and rows 8 * (l >> 4) + 0..7. Only what src/split uses is here.
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace nvcuda {
namespace wmma {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
enum layout_t { mem_row_major, mem_col_major };

typedef _Float16 fnx_h8 __attribute__((ext_vector_type(8)));
typedef float fnx_f8 __attribute__((ext_vector_type(8)));

template <typename Use, int M, int N, int K, typename T, typename Layout = void> struct fragment;
template <typename L> struct fragment<matrix_a, 16, 16, 16, __half, L> { fnx_h8 v; };
template <typename L> struct fragment<matrix_b, 16, 16, 16, __half, L> { fnx_h8 v; };
template <> struct fragment<accumulator, 16, 16, 16, float, void> { fnx_f8 v; };

__device__ __forceinline__ int fnx_lane() { return threadIdx.x & 31; }

// A (16 x 16, M x K): row_major ptr[m * ldm + k], col_major ptr[k * ldm + m]
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, row_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), m = l & 15, k0 = 8 * (l >> 4);
    f.v = *reinterpret_cast<const fnx_h8*>(p + (size_t) m * ldm + k0);   // 16 bytes, ldm a multiple of 8 (the sources keep that)
}
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, col_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), m = l & 15, k0 = 8 * (l >> 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) f.v[i] = (_Float16) __half2float(p[(size_t) (k0 + i) * ldm + m]);
}
// B (16 x 16, K x N): col_major ptr[n * ldm + k], row_major ptr[k * ldm + n]
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, col_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), n = l & 15, k0 = 8 * (l >> 4);
    f.v = *reinterpret_cast<const fnx_h8*>(p + (size_t) n * ldm + k0);
}
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, row_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), n = l & 15, k0 = 8 * (l >> 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) f.v[i] = (_Float16) __half2float(p[(size_t) (k0 + i) * ldm + n]);
}
__device__ __forceinline__ void fill_fragment(fragment<accumulator, 16, 16, 16, float>& f, float x) {
#pragma unroll
    for (int i = 0; i < 8; ++i) f.v[i] = x;
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    d.v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a.v, b.v, c.v);
#else
    (void) a; (void) b; (void) c; (void) d;
    __builtin_trap();   // not a gfx12 device pass
#endif
}
__device__ __forceinline__ void store_matrix_sync(float* p, const fragment<accumulator, 16, 16, 16, float>& f, unsigned ldm, layout_t layout) {
    const int l = fnx_lane(), n = l & 15, r0 = 8 * (l >> 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (layout == mem_row_major) p[(size_t) (r0 + i) * ldm + n] = f.v[i];
        else p[(size_t) n * ldm + r0 + i] = f.v[i];
    }
}

}  // namespace wmma
}  // namespace nvcuda
