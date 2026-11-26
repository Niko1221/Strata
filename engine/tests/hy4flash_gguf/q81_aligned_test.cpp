#include "strata/hy4flash_gguf/cpu_experts.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>

namespace hy4 = strata::hy4_flash;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr int width = 1536;
    hy4::cpu::NativeFmt fmt;
    std::string error;
    if (!hy4::CpuExperts::make_format(12, 14, width, width, 10.f, fmt, error, true)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!fmt.q81_aligned_table) return 77;

    std::array<float, width> input;
    std::array<uint8_t, width / 32 * 36> cpu, gpu;
    float* x;
    uint8_t* q;
    cudaStream_t stream;
    if (cudaMalloc(&x, sizeof(input)) != cudaSuccess || cudaMalloc(&q, sizeof(gpu)) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess) return 1;
    std::mt19937 rng(42);
    for (int exponent = -149; exponent <= 125; ++exponent) {
        const float amplitude = std::ldexp(1.f, exponent);
        for (int i = 0; i < width; ++i) {
            const float half_code = static_cast<float>(int(rng() % 252) - 126) + .5f;
            input[i] = i % 32 == 0 ? amplitude : half_code * (amplitude / 127.f);
        }
        hy4::cpu::native_quant_act(fmt, input.data(), cpu.data());
        if (cudaMemcpyAsync(x, input.data(), sizeof(input), cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return 1;
        strata::kernels::native_quantize_q8_1(x, q, width, 1, stream);
        if (cudaMemcpyAsync(gpu.data(), q, gpu.size(), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) return 1;
        if (cpu != gpu) {
            for (unsigned i = 0; i < gpu.size(); ++i) {
                if (cpu[i] != gpu[i]) {
                    std::fprintf(stderr, "exponent %d byte %u: CPU %u, GPU %u\n", exponent, i, cpu[i], gpu[i]);
                    break;
                }
            }
            return 1;
        }
    }
    cudaStreamDestroy(stream);
    cudaFree(q);
    cudaFree(x);
    return 0;
}
