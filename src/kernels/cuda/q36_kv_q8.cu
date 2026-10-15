// src/kernels/cuda/q36_kv_q8.cu - see include/strata/kernels/q36_kv_q8.hpp.
#include "strata/kernels/q36_kv_q8.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "q36_kv_q8: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(int hd, const char* what) {
    if (hd % Q36_KV_Q8_GROUP != 0 || hd <= 0 || hd > 1024) {
        std::fprintf(stderr, "q36_kv_q8: %s: hd %d must be a positive multiple of %d and at most 1024\n", what,
                     hd, Q36_KV_Q8_GROUP);
        std::exit(1);
    }
}

// One warp = one 32-value group.  The warp reduces its own max |x| with __shfl_xor_sync, so lane 0 of each
// group writes the scale and every lane writes its code - the group being one warp is why no shared memory
// or __syncthreads is needed.
__global__ void q36_kv_q8_quantize_kernel(int8_t* __restrict__ q, uint16_t* __restrict__ scale,
                                          const float* __restrict__ x, int groups) {
    const int t = threadIdx.x;
    const int g = t / Q36_KV_Q8_GROUP;
    if (g >= groups) return;
    const float v = x[t];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const uint16_t sb = q36_kv_q8_scale_bits(a);
    if ((t & (Q36_KV_Q8_GROUP - 1)) == 0) scale[g] = sb;
    q[t] = q36_kv_q8_quantize(v, sb);
}

__global__ void q36_kv_q8_dequantize_kernel(float* __restrict__ y, const int8_t* __restrict__ q,
                                            const uint16_t* __restrict__ scale, int hd) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hd) return;
    y[t] = q36_kv_q8_dequantize(q[t], scale[t / Q36_KV_Q8_GROUP]);
}

}  // namespace

void q36_kv_q8_quantize(int8_t* q, uint16_t* scale, const float* x, int hd, void* stream) {
    validate(hd, "q36_kv_q8_quantize");
    q36_kv_q8_quantize_kernel<<<1, hd, 0, (cudaStream_t) stream>>>(q, scale, x, hd / Q36_KV_Q8_GROUP);
    check("q36_kv_q8_quantize launch");
}

void q36_kv_q8_dequantize(float* y, const int8_t* q, const uint16_t* scale, int hd, void* stream) {
    validate(hd, "q36_kv_q8_dequantize");
    const int blocks = (hd + 255) / 256;
    q36_kv_q8_dequantize_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(y, q, scale, hd);
    check("q36_kv_q8_dequantize launch");
}

}  // namespace strata::kernels
