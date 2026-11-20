// Test-only (build_dq_test.sh adds it with -include when CUDA_ARCH < 70): lets ds4_iq.cu and ds4_gguf.cu compile for a Pascal
// P100 (compute 6.0) - there is no __dp4a below 6.1 and no nvcuda::wmma below 7.0.  The DeepSeek engine itself does not build
// for Pascal (the packaging script says so); this is only so the dequantizer and GEMV kernels, which do not need either, can be
// run on that architecture.  The tensor-core kernels are stubs that never launch (tc_ok() is false below sm_70).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    return c + (int) (signed char) (a) * (int) (signed char) (b) + (int) (signed char) (a >> 8) * (int) (signed char) (b >> 8) +
           (int) (signed char) (a >> 16) * (int) (signed char) (b >> 16) + (int) (signed char) (a >> 24) * (int) (signed char) (b >> 24);
}
__device__ __forceinline__ unsigned __dp4a(unsigned a, unsigned b, unsigned c) {
    return c + (a & 255u) * (b & 255u) + ((a >> 8) & 255u) * ((b >> 8) & 255u) + ((a >> 16) & 255u) * ((b >> 16) & 255u) + (a >> 24) * (b >> 24);
}
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
namespace nvcuda {
namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <class Use, int M, int N, int K, class T, class L = void>
struct fragment { T x[1]; };
template <class F, class T> __device__ inline void fill_fragment(F&, const T&) {}
template <class F, class P> __device__ inline void load_matrix_sync(F&, const P*, unsigned) {}
template <class F, class P> __device__ inline void load_matrix_sync(F&, const P*, unsigned, layout_t) {}
template <class D, class A, class B, class C> __device__ inline void mma_sync(D&, const A&, const B&, const C&) {}
template <class P, class F> __device__ inline void store_matrix_sync(P*, const F&, unsigned, layout_t) {}
}  // namespace wmma
}  // namespace nvcuda
#endif
