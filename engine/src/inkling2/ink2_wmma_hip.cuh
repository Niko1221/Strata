#pragma once

#include <cstddef>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define INK2_WMMA_GFX11 1
#define INK2_WMMA_GFX12 0
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define INK2_WMMA_GFX11 0
#define INK2_WMMA_GFX12 1
#else
#define INK2_WMMA_GFX11 0
#define INK2_WMMA_GFX12 0
#endif

namespace ink2_wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

static_assert(sizeof(__half) == sizeof(_Float16), "expected 16-bit __half");
static_assert(sizeof(unsigned short) == sizeof(_Float16), "expected 16-bit _Float16");

using v16h = _Float16 __attribute__((ext_vector_type(16)));
using v8h = _Float16 __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ _Float16 to_f16(__half h) {
    return __builtin_bit_cast(_Float16, __half_as_ushort(h));
}

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

#if INK2_WMMA_GFX11
template <typename Layout>
struct fragment<matrix_a, 16, 16, 16, __half, Layout> {
    v16h v;
};

template <typename Layout>
struct fragment<matrix_b, 16, 16, 16, __half, Layout> {
    v16h v;
};
#elif INK2_WMMA_GFX12
template <typename Layout>
struct fragment<matrix_a, 16, 16, 16, __half, Layout> {
    v8h v;
};

template <typename Layout>
struct fragment<matrix_b, 16, 16, 16, __half, Layout> {
    v8h v;
};
#else
template <typename Layout>
struct fragment<matrix_a, 16, 16, 16, __half, Layout> {
    float v[16];
};

template <typename Layout>
struct fragment<matrix_b, 16, 16, 16, __half, Layout> {
    float v[16];
};
#endif

template <>
struct fragment<accumulator, 16, 16, 16, float, void> {
    static constexpr int num_elements = 8;
    float x[num_elements];
};

__device__ __forceinline__ void fill_fragment(fragment<accumulator, 16, 16, 16, float>& f, float val) {
#pragma unroll
    for (int i = 0; i < 8; ++i) f.x[i] = val;
}

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, row_major>& f,
                                                 const __half* src, unsigned ldm) {
    const int lane = threadIdx.x & 31;
    const int sub = lane & 15;
#if INK2_WMMA_GFX11
    __builtin_memcpy(&f.v, src + (size_t) sub * ldm, sizeof(v16h));
#elif INK2_WMMA_GFX12
    const int hi = lane >> 4;
    __builtin_memcpy(&f.v, src + (size_t) sub * ldm + 8 * hi, sizeof(v8h));
#else
    const __half* row = src + (size_t) sub * ldm;
#pragma unroll
    for (int k = 0; k < 16; ++k) f.v[k] = __half2float(row[k]);
#endif
}

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, col_major>& f,
                                                 const __half* src, unsigned ldm) {
    const int lane = threadIdx.x & 31;
    const int sub = lane & 15;
#if INK2_WMMA_GFX11
    __builtin_memcpy(&f.v, src + (size_t) sub * ldm, sizeof(v16h));
#elif INK2_WMMA_GFX12
    const int hi = lane >> 4;
    __builtin_memcpy(&f.v, src + (size_t) sub * ldm + 8 * hi, sizeof(v8h));
#else
    const __half* col = src + (size_t) sub * ldm;
#pragma unroll
    for (int k = 0; k < 16; ++k) f.v[k] = __half2float(col[k]);
#endif
}

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, row_major>& f,
                                                 const __half* src, unsigned ldm) {
    const int lane = threadIdx.x & 31;
    const int sub = lane & 15;
#if INK2_WMMA_GFX11
#pragma unroll
    for (int k = 0; k < 16; ++k) f.v[k] = to_f16(src[(size_t) k * ldm + sub]);
#elif INK2_WMMA_GFX12
    const int k0 = (lane >> 4) * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) f.v[k] = to_f16(src[(size_t) (k0 + k) * ldm + sub]);
#else
#pragma unroll
    for (int k = 0; k < 16; ++k) f.v[k] = __half2float(src[(size_t) k * ldm + sub]);
#endif
}

template <typename LayoutA, typename LayoutB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __half, LayoutA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LayoutB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c) {
#if INK2_WMMA_GFX11
    v8f cv;
    __builtin_memcpy(&cv, c.x, sizeof(cv));
    cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.v, b.v, cv);
    __builtin_memcpy(d.x, &cv, sizeof(cv));
#elif INK2_WMMA_GFX12
    v8f cv;
    __builtin_memcpy(&cv, c.x, sizeof(cv));
    cv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a.v, b.v, cv);
    __builtin_memcpy(d.x, &cv, sizeof(cv));
#else
    const int hi = (threadIdx.x & 31) >> 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + hi;
        float sum = c.x[i];
#pragma unroll
        for (int k = 0; k < 16; ++k) sum = fmaf(__shfl(a.v[k], row), b.v[k], sum);
        d.x[i] = sum;
    }
#endif
}

__device__ __forceinline__ void store_matrix_sync(float* dst,
                                                  const fragment<accumulator, 16, 16, 16, float>& f,
                                                  unsigned ldm, layout_t layout) {
    const int lane = threadIdx.x & 31;
    const int sub = lane & 15;
    const int hi = lane >> 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#if INK2_WMMA_GFX12
        const int row = 8 * hi + i;
#else
        const int row = 2 * i + hi;
#endif
        if (layout == mem_row_major) dst[(size_t) row * ldm + sub] = f.x[i];
        else dst[(size_t) sub * ldm + row] = f.x[i];
    }
}

}
