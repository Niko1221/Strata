// F16 weights for the opt-in qwen35moe path. Activations and sums stay in FP32 during decode;
// prompt products use Gemm::f16. These kernels do not quantize either the weights or the activations.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace q36f16 {

__device__ __forceinline__ float sum32(float v) {
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d, 32);
    return v;
}

// Eight independent rows per block. The reduction order is the same for every activation column.
__global__ void matvec(const __half* w, const float* x, float* y, int ni, int no) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= no) return;
    const __half* wr = w + (size_t) row * ni;
    const float* xr = x + (size_t) blockIdx.y * ni;
    float v = 0.f;
    for (int i = lane; i < ni; i += 32) v = fmaf(__half2float(wr[i]), xr[i], v);
    v = sum32(v);
    if (lane == 0) y[(size_t) blockIdx.y * no + row] = v;
}

__global__ void embed(const __half* w, const int32_t* ids, float* y, int n) {
    const size_t base = (size_t) ids[blockIdx.x] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[(size_t) blockIdx.x * n + i] = __half2float(w[base + i]);
}

// The existing route/group tables name resident GPU experts only. CPU misses are merged by the engine.
__global__ void gate_up(const unsigned long long* ptr, const int32_t* start, const int32_t* groups,
                        const int32_t* token, const float* x, float* hidden, int n, int ff) {
    const int g = blockIdx.y, row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= *groups || row >= ff) return;
    const __half* w = reinterpret_cast<const __half*>(ptr[g]);
    const __half* wg = w + (size_t) row * n;
    const __half* wu = w + (size_t) (ff + row) * n;
    for (int e = start[g]; e < start[g + 1]; ++e) {
        const float* xr = x + (size_t) token[e] * n;
        float a = 0.f, b = 0.f;
        for (int i = lane; i < n; i += 32) {
            a = fmaf(__half2float(wg[i]), xr[i], a);
            b = fmaf(__half2float(wu[i]), xr[i], b);
        }
        a = sum32(a); b = sum32(b);
        if (lane == 0) hidden[(size_t) e * ff + row] = (a / (1.f + expf(-a))) * b;
    }
}

__global__ void down(const unsigned long long* ptr, const int32_t* start, const int32_t* groups,
                     const int32_t* dest, const float* hidden, float* y, int n, int ff) {
    const int g = blockIdx.y, row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= *groups || row >= n) return;
    const __half* w = reinterpret_cast<const __half*>(ptr[g]) + (size_t) 2 * ff * n + (size_t) row * ff;
    for (int e = start[g]; e < start[g + 1]; ++e) {
        const float* h = hidden + (size_t) e * ff;
        float v = 0.f;
        for (int i = lane; i < ff; i += 32) v = fmaf(__half2float(w[i]), h[i], v);
        v = sum32(v);
        if (lane == 0) y[(size_t) dest[e] * n + row] = v;
    }
}

// Compact expert-major rows for the prompt GEMMs, using the same routing permutation as MMQ.
__global__ void gather(const float* x, const int32_t* token, __half* y, int n, int rows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[(size_t) row * n + i] = __float2half(x[(size_t) token[row] * n + i]);
}

__global__ void swiglu(const float* gu, __half* y, int ff, int rows) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * ff) return;
    const size_t row = i / ff, col = i % ff;
    const float g = gu[row * 2 * ff + col], u = gu[row * 2 * ff + ff + col];
    y[i] = __float2half((g / (1.f + expf(-g))) * u);
}

}  // namespace q36f16
