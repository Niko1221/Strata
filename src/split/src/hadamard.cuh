// The rotation Agention's APR files (Gyro-S, Gyro-M; prism.hadamard, hadamard_meta.hpp) fold into some routed experts' input columns,
// on the GPU: one 128-column block per 128-thread block.
#pragma once

#include <cuda_runtime.h>

namespace fnx {

// y = H_128 (s * v) / sqrt(128) over the 128 threads of a block (v: this thread's column of the block; scale: 1 / sqrt(128) as the host
// computes it): the butterflies of the main engine's hadamard_rows_kernel in its order, so the same floats. Every thread must call it
__device__ __forceinline__ float had128(float v, float s, float scale, float* hb) {
    const int i = threadIdx.x & 127;
    hb[i] = v * s;
    for (int len = 1; len < 128; len <<= 1) {
        __syncthreads();
        if ((i & len) == 0) {
            const float a = hb[i], c = hb[i + len];
            hb[i] = a + c;
            hb[i + len] = a - c;
        }
    }
    __syncthreads();
    const float y = hb[i] * scale;
    __syncthreads();   // hb is free again for the caller's next call
    return y;
}

}  // namespace fnx
