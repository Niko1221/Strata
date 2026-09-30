// src/prefill/wmma_gemm.cu - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill (gfx1100).
//
// Computes Y[t, n] = beta * Y[t, n] + sum_k W[n, k] * X[t, k] using RDNA3 WMMA intrinsics:
//   * X is T x K row-major (leading dim K), fp16 or bf16
//   * W is N x K row-major (leading dim K), fp16 or bf16
//   * Y is T x N row-major with leading dimension ldy >= N, fp32
//
// Hardware: AMD RDNA3 (gfx1100, e.g. RX 7900 XTX)
//   * v_wmma_f32_16x16x16_f16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
//   * v_wmma_f32_16x16x16_bf16_w32 intrinsic (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
//   * Wave32 doubled input fragment: lane t (lane_lo = t & 15) holds row lane_lo of A (16 elements along K)
//     and column lane_lo of B (16 elements along K). Lanes 16..31 duplicate lanes 0..15.
//   * Wave32 C output mapping: lane t holds column lane_lo of 16x16 output tile,
//     with 8 elements alternating rows: row m = 2*i + lane_hi (lane_hi = t >> 4).
//   * Tile variants:
//       1. gemm_wmma_64x64_4w: 4 waves (128 threads), 64T x 64N x 16K tile, double-buffered LDS for W.
//       2. gemm_wmma_16x16_1w: 1 wave (32 threads), 16T x 16N x 16K tile, zero LDS/barriers, for small T/N.

#include "wmma_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// STRATA_WMMA_GFX11 is defined by the BUILD (CMakeLists.txt, from CMAKE_HIP_ARCHITECTURES), not inferred
// from compiler macros: measured on this toolchain the HOST pass of a HIP compile does not define
// __gfx1100__ but does define __HIP_DEVICE_COMPILE__, so a compiler-macro guard here silently selected the
// "return false" stub at the bottom of this file for the very symbol the engine links - the WMMA path then
// never ran, while the same file compiled with hipcc (as the probes do) took the real branch.  One
// build-defined macro is uniform across the host and device passes.
#if defined(STRATA_WMMA_GFX11)

using v8fp32 = float __attribute__((ext_vector_type(8)));

template <typename ElemT>
struct WmmaTraits;

template <>
struct WmmaTraits<_Float16> {
    using vec_t = _Float16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
};

template <>
struct WmmaTraits<__bf16> {
    using vec_t = __bf16 __attribute__((ext_vector_type(16)));
    __device__ static inline v8fp32 mma(vec_t a, vec_t b, v8fp32 c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
};

// ===========================================================================
// Variant 1: 16x16_1w - Single-wave kernel for small T / N.
// Zero LDS, zero barrier synchronization, full register residency.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_16x16_1w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 16;
    const int n_tile = blockIdx.x * 16;
    if (m_tile >= T || n_tile >= N) return;

    const int lane = threadIdx.x;   // 0..31
    const int lane_lo = lane & 15;  // 0..15
    const int lane_hi = lane >> 4;  // 0 or 1

    v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

    const int m_row = m_tile + lane_lo;
    const int n_row = n_tile + lane_lo;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        vec_t a_frag, b_frag;

        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        if (n_row < N) {
            __builtin_memcpy(&b_frag, W + (int64_t)n_row * K + k_tile, sizeof(b_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) b_frag[i] = 0;
        }

        c_acc = WmmaTraits<ElemT>::mma(a_frag, b_frag, c_acc);
    }

    const int out_n = n_tile + lane_lo;
    if (out_n < N) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_m = m_tile + 2 * i + lane_hi;
            if (out_m < T) {
                float* dst = Y + (int64_t)out_m * ldy + out_n;
                if (beta == 0.0f) {
                    *dst = c_acc[i];
                } else {
                    *dst = beta * (*dst) + c_acc[i];
                }
            }
        }
    }
}

// ===========================================================================
// Variant 2: 64x64_4w - 4 waves per block (128 threads), 64T x 64N tile.
// Double-buffered LDS tile for W (4 KB LDS total), cooperative 128-bit global loads.
// ===========================================================================
template <typename ElemT>
__global__ void gemm_wmma_64x64_4w(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    float* __restrict__ Y,
    int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {

    using vec_t = typename WmmaTraits<ElemT>::vec_t;
    const int m_tile = blockIdx.y * 64;
    const int n_tile = blockIdx.x * 64;
    if (m_tile >= T || n_tile >= N) return;

    const int tid = threadIdx.x;   // 0..127
    const int wave_id = tid >> 5;  // 0..3
    const int lane = tid & 31;     // 0..31
    const int lane_lo = lane & 15; // 0..15
    const int lane_hi = lane >> 4; // 0 or 1

    // 4 accumulators per wave covering 4 x 16 N-subtiles
    v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

    // Double-buffered LDS tile: 64 rows of N x 16 elements of K (2 KB per buffer)
    alignas(16) __shared__ ElemT b_lds[2][64][16];

    // Thread mapping for cooperative loading of W into LDS (128 threads load 64x16 elements)
    // Each thread loads 8 halfs (16 bytes = uint4)
    const int row_in_tile = tid >> 1;     // 0..63
    const int k_sub = (tid & 1) << 3;     // 0 or 8
    const int actual_n = n_tile + row_in_tile;

    auto load_w_into_lds = [&](int buf, int k_tile) {
        const int actual_k = k_tile + k_sub;
        uint4 val = {0, 0, 0, 0};
        if (actual_n < N && actual_k < K) {
            const void* ptr = reinterpret_cast<const void*>(W + (int64_t)actual_n * K + actual_k);
            val = *reinterpret_cast<const uint4*>(ptr);
        }
        *reinterpret_cast<uint4*>(&b_lds[buf][row_in_tile][k_sub]) = val;
    };

    // Pre-fill buffer 0
    load_w_into_lds(0, 0);
    __syncthreads();

    const int m_row = m_tile + wave_id * 16 + lane_lo;
    int cur_buf = 0;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        const int next_buf = 1 - cur_buf;
        const int k_next = k_tile + 16;

        // Prefetch next K-tile of W into LDS
        if (k_next < K) {
            load_w_into_lds(next_buf, k_next);
        }

        // Load A fragment from X for current wave's 16 M-rows
        vec_t a_frag;
        if (m_row < T) {
            __builtin_memcpy(&a_frag, X + (int64_t)m_row * K + k_tile, sizeof(a_frag));
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) a_frag[i] = 0;
        }

        // Read B fragments from LDS and compute WMMA
        vec_t b_frag0, b_frag1, b_frag2, b_frag3;
        __builtin_memcpy(&b_frag0, &b_lds[cur_buf][0 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag1, &b_lds[cur_buf][16 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag2, &b_lds[cur_buf][32 + lane_lo][0], sizeof(vec_t));
        __builtin_memcpy(&b_frag3, &b_lds[cur_buf][48 + lane_lo][0], sizeof(vec_t));

        c_acc0 = WmmaTraits<ElemT>::mma(a_frag, b_frag0, c_acc0);
        c_acc1 = WmmaTraits<ElemT>::mma(a_frag, b_frag1, c_acc1);
        c_acc2 = WmmaTraits<ElemT>::mma(a_frag, b_frag2, c_acc2);
        c_acc3 = WmmaTraits<ElemT>::mma(a_frag, b_frag3, c_acc3);

        __syncthreads();
        cur_buf = next_buf;
    }

    // Store C to Y (coalesced 64-byte writes per wave half)
    const int m_tile_wave = m_tile + wave_id * 16;
    auto store_acc = [&](const v8fp32& acc, int n_base) {
        const int out_n = n_base + lane_lo;
        if (out_n < N) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int out_m = m_tile_wave + 2 * i + lane_hi;
                if (out_m < T) {
                    float* dst = Y + (int64_t)out_m * ldy + out_n;
                    if (beta == 0.0f) {
                        *dst = acc[i];
                    } else {
                        *dst = beta * (*dst) + acc[i];
                    }
                }
            }
        }
    };

    store_acc(c_acc0, n_tile + 0);
    store_acc(c_acc1, n_tile + 16);
    store_acc(c_acc2, n_tile + 32);
    store_acc(c_acc3, n_tile + 48);
}

template <typename ElemT>
static inline bool strata_wmma_gemm_dispatch(const uint16_t* X, const uint16_t* W, float* Y,
                                             int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                                             void* stream) {
    if (!X || !W || !Y) return false;
    if (T <= 0 || N <= 0 || K <= 0) return false;
    if (K % 16 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    if (ldy <= 0) ldy = N;
    if (ldy < N) return false;

    hipStream_t s = static_cast<hipStream_t>(stream);

    if (T >= 32 && N >= 32) {
        dim3 block(128);
        dim3 grid((uint32_t)((N + 63) / 64), (uint32_t)((T + 63) / 64), 1);
        gemm_wmma_64x64_4w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    } else {
        dim3 block(32);
        dim3 grid((uint32_t)((N + 15) / 16), (uint32_t)((T + 15) / 16), 1);
        gemm_wmma_16x16_1w<ElemT><<<grid, block, 0, s>>>(X, W, Y, T, N, K, ldy, beta);
    }
    return true;
}

bool strata_wmma_gemm_f16(const uint16_t* X, const uint16_t* W, float* Y,
                          int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                          void* stream) {
    return strata_wmma_gemm_dispatch<_Float16>(X, W, Y, T, N, K, ldy, beta, stream);
}

bool strata_wmma_gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y,
                           int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                           void* stream) {
    return strata_wmma_gemm_dispatch<__bf16>(X, W, Y, T, N, K, ldy, beta, stream);
}

#else
// Non-HIP / Non-GFX11 compilation fallback
bool strata_wmma_gemm_f16(const uint16_t*, const uint16_t*, float*,
                          int64_t, int64_t, int64_t, int64_t, float,
                          void*) {
    return false;
}

bool strata_wmma_gemm_bf16(const uint16_t*, const uint16_t*, float*,
                           int64_t, int64_t, int64_t, int64_t, float,
                           void*) {
    return false;
}
#endif
