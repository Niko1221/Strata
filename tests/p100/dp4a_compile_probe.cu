#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"

// No host main and no GPU execution. Emit PTX/cubin for SASS/resource comparison.
__device__ __forceinline__ int scalar_dot(int a, int b, int c) {
    const int8_t *av = reinterpret_cast<const int8_t *>(&a);
    const int8_t *bv = reinterpret_cast<const int8_t *>(&b);
    return c + av[0]*bv[0] + av[1]*bv[1] + av[2]*bv[2] + av[3]*bv[3];
}
extern "C" __global__ void p100_scalar_probe(const int *a, const int *b, const int *c, int *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = scalar_dot(a[i], b[i], c[i]);
}
extern "C" __global__ void p100_candidate_probe(const int *a, const int *b, const int *c, int *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = STRATA_DP4A(a[i], b[i], c[i]);
}
