// Independent exhaustive-byte-pair and randomized mixed-lane parity.
#include <hip/hip_runtime.h>
#include "strata/hip_compat/intrinsics.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(call) do { auto error = (call); if (error != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); return 2; } } while (0)

struct Input { uint32_t a, b, selector; };
__global__ void probe(const Input* input, uint32_t* output, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    auto x = input[i];
    output[i * 4] = __byte_perm(x.a, x.b, x.selector);
    output[i * 4 + 1] = static_cast<uint32_t>(__vsub4(static_cast<int>(x.a), static_cast<int>(x.b)));
    output[i * 4 + 2] = static_cast<uint32_t>(__vsubss4(static_cast<int>(x.a), static_cast<int>(x.b)));
    output[i * 4 + 3] = static_cast<uint32_t>(__vcmpne4(static_cast<int>(x.a), static_cast<int>(x.b)));
}

int main() {
    constexpr size_t count = 262144, exhaustive = 65536, guard = 16;
    constexpr uint32_t sentinel = 0xdeadbeefu;
    std::vector<Input> input(count);
    std::mt19937 random(20260930);
    for (size_t i = 0; i < count; ++i) {
        input[i].a = i < exhaustive ? static_cast<uint32_t>(i >> 8) * 0x01010101u : random();
        input[i].b = i < exhaustive ? static_cast<uint32_t>(i & 255u) * 0x01010101u : random();
        input[i].selector = static_cast<uint32_t>(i) & 0x7777u;
    }
    std::vector<uint32_t> output(count * 4 + guard, sentinel);
    Input* device_input = nullptr;
    uint32_t* device_output = nullptr;
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(Input)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(uint32_t)));
    CHECK(hipMemcpy(device_input, input.data(), input.size() * sizeof(Input), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_output, output.data(), output.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe, dim3((count + 255) / 256), dim3(256), 0, 0, device_input, device_output, count);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(output.data(), device_output, output.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    CHECK(hipFree(device_output));
    CHECK(hipFree(device_input));
    for (size_t i = 0; i < count; ++i) {
        uint32_t expected[4] = {};
        uint64_t pair = (static_cast<uint64_t>(input[i].b) << 32) | input[i].a;
        for (unsigned lane = 0; lane < 4; ++lane) {
            unsigned shift = lane * 8;
            unsigned a = (input[i].a >> shift) & 255u, b = (input[i].b >> shift) & 255u;
            unsigned selected = (input[i].selector >> (lane * 4)) & 7u;
            expected[0] |= static_cast<uint32_t>((pair >> (selected * 8)) & 255u) << shift;
            expected[1] |= ((a - b) & 255u) << shift;
            int signed_a = a < 128 ? static_cast<int>(a) : static_cast<int>(a) - 256;
            int signed_b = b < 128 ? static_cast<int>(b) : static_cast<int>(b) - 256;
            expected[2] |= (static_cast<uint32_t>(std::clamp(signed_a - signed_b, -128, 127)) & 255u) << shift;
            expected[3] |= (a != b ? 255u : 0u) << shift;
        }
        for (unsigned operation = 0; operation < 4; ++operation) {
            if (output[i * 4 + operation] != expected[operation]) {
                std::fprintf(stderr, "Mismatch case=%zu operation=%u a=%08x b=%08x selector=%04x actual=%08x expected=%08x\n",
                             i, operation, input[i].a, input[i].b, input[i].selector,
                             output[i * 4 + operation], expected[operation]);
                return 1;
            }
        }
    }
    for (size_t i = count * 4; i < output.size(); ++i) {
        if (output[i] != sentinel) { std::fprintf(stderr, "Output guard changed\n"); return 1; }
    }
    std::printf("Fast intrinsics parity PASS: %zu cases, %zu operation results, all 65536 byte pairs, 4096 selectors, 16 guards\n", count, count * 4);
    return 0;
}
