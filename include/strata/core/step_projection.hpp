#pragma once
// Compiled only into strata-step: graph-safe copies into diagnostic-owned memory.
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::core {
struct StepProjection;
inline StepProjection* step_projection = nullptr;
struct StepProjection {
    float* mixed = nullptr;
    float* qkv = nullptr;
    size_t input_count = 0, output_count = 0;
    bool init(size_t n, size_t c) {
        input_count = n; output_count = c;
        if (cudaMalloc((void**) &mixed, (n + c) * sizeof(float)) != cudaSuccess) return false;
        qkv = mixed + n;
        if (cudaMemset(mixed, 0, (n + c) * sizeof(float)) != cudaSuccess) return false;
        step_projection = this;
        return true;
    }
    bool capture(const float* input, const float* output, cudaStream_t stream) {
        return cudaMemcpyAsync(mixed, input, input_count * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
               cudaMemcpyAsync(qkv, output, output_count * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
    }
    ~StepProjection() {
        if (step_projection == this) step_projection = nullptr;
        if (mixed) cudaFree(mixed);
    }
};
}
