#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    int failures = 0;
    // Non-aligned hidden size and padded planes detect row/plane stride mistakes.
    for (int rows = 1; rows <= 8; ++rows) for (int capacity : {rows, rows + 5}) {
        const int taps = 5, hidden = 257;
        const int64_t stride = (int64_t) capacity * hidden + 7;
        const int count = rows * taps * hidden;
        std::vector<float> input(taps * stride);
        std::vector<uint16_t> bits(input.size()), expected(count + 16, 0x5a5a), actual(expected.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = (float)((int)(i % 997) - 498) / 37.0f;
            bits[i] = strata::kernels::bf16_from_f32(input[i]);
        }
        for (int r = 0; r < rows; ++r) for (int t = 0; t < taps; ++t) for (int c = 0; c < hidden; ++c)
            expected[(r * taps + t) * hidden + c] = bits[t * stride + r * hidden + c];
        float* f = nullptr; uint16_t *b = nullptr, *out = nullptr;
        if (cudaMalloc(&f, input.size() * 4) != cudaSuccess ||
            cudaMalloc(&b, bits.size() * 2) != cudaSuccess || cudaMalloc(&out, expected.size() * 2) != cudaSuccess) return 1;
        cudaMemcpy(f, input.data(), input.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(b, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice);
        for (bool fp32 : {false, true}) {
            std::vector<uint16_t> guard(expected.size(), 0x5a5a);
            cudaMemcpy(out, guard.data(), guard.size() * 2, cudaMemcpyHostToDevice);
            if (fp32) strata::kernels::dflash_gather_taps(f, out, taps, hidden, rows, stride, stream);
            else strata::kernels::dflash_gather_taps(b, out, taps, hidden, rows, stride, stream);
            if (cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaMemcpy(actual.data(), out, actual.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess || actual != expected)
                ++failures;
        }
        cudaFree(f); cudaFree(b); cudaFree(out);
    }
    cudaStreamDestroy(stream);
    std::printf("dflash_taps_test: %s (%d failures)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
