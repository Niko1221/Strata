// src/kernels/cuda/xor_scatter.cuh - several warp sums in one xor tree, each bitwise the butterfly's.
//
// `v += __shfl_xor_sync(v, o)` for o = 16, 8, 4, 2, 1 leaves every lane with the same sum.  For NV sums a lane does not
// need all of them in every lane: at distance o, while it holds more than one, it keeps the half its bit o selects and
// adds its partner's copy of them - the butterfly's own additions in the same order - then the last steps as the
// butterfly.  NV + NV/2 + ... shuffles instead of 5 NV.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

// NV (a power of two, at most 32) values a lane; afterwards v[0] is the sum of value xor_scatter_index<NV>(lane)
template <int NV>
__device__ __forceinline__ void xor_scatter(float (&v)[NV], int lane) {
    static_assert(NV >= 1 && NV <= 32 && (NV & (NV - 1)) == 0, "NV must be a power of two <= 32");
#pragma unroll
    for (int s = 0; s < 5; ++s) {
        const int o = 16 >> s, n = NV >> s;   // n: the values left before this step
        if (n > 1) {
            const bool hi = (lane & o) != 0;
#pragma unroll
            for (int e = 0; e < n / 2; ++e) {
                const float mine = hi ? v[n / 2 + e] : v[e];
                const float other = hi ? v[e] : v[n / 2 + e];
                v[e] = mine + __shfl_xor_sync(0xffffffffu, other, o);
            }
        } else {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
        }
    }
}

// the value a lane's v[0] sums after xor_scatter: the halves its bits 16, 8, ... selected
template <int NV>
__device__ __forceinline__ int xor_scatter_index(int lane) {
    int vi = 0;
#pragma unroll
    for (int s = 0; (NV >> s) > 1; ++s) vi += (lane & (16 >> s)) ? NV >> (s + 1) : 0;
    return vi;
}

// the lowest lane holding value vi
template <int NV>
__device__ __forceinline__ int xor_scatter_lane(int vi) {
    int lane = 0;
#pragma unroll
    for (int s = 0; (NV >> s) > 1; ++s) lane += (vi & (NV >> (s + 1))) ? 16 >> s : 0;
    return lane;
}

// the smallest power of two >= n
__host__ __device__ constexpr int pow2_at_least(int n) { return n <= 1 ? 1 : 2 * pow2_at_least((n + 1) / 2); }

}  // namespace
}  // namespace strata::kernels
