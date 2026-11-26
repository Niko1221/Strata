#include "q81_aligned.hpp"

#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hy4_flash::cpu::q81_aligned {
namespace {

// For each possible exp2 result in [0.5, 2), find the first input producing it.
// Float keys are monotonic across the sign change at zero.
__global__ void fill_table(uint32_t* tile, unsigned start) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned index = start + lane;
    if (index >= 2 * mantissa_count) {
        const float x = __uint_as_float(0x3f800000u + index - 2 * mantissa_count);
        float inverse;
        asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inverse) : "f"(x));
        tile[lane] = __float_as_uint(inverse);
        return;
    }
    const float target = __uint_as_float(0x3f000000u + index);
    unsigned lo = ~0xbf800000u, hi = 0xbf800000u;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo) / 2;
        const float x = __uint_as_float(mid >> 31 ? mid ^ 0x80000000u : ~mid);
        float y;
        asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x));
        if (y >= target)
            hi = mid;
        else
            lo = mid + 1;
    }
    tile[lane] = lo;
}

void check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Hy4 Q8_1 lookup table: ") + cudaGetErrorString(result));
}

std::vector<uint32_t> make_tables() {
    constexpr unsigned tile_size = 1u << 18;
    std::vector<uint32_t> host(3 * mantissa_count);
    uint32_t* ptr = nullptr;
    check(cudaMalloc(&ptr, tile_size * sizeof(*ptr)));
    std::unique_ptr<uint32_t, decltype(&cudaFree)> tile(ptr, cudaFree);
    for (unsigned start = 0; start < host.size(); start += tile_size) {
        fill_table<<<tile_size / 256, 256>>>(ptr, start);
        check(cudaGetLastError());
        check(cudaMemcpy(host.data() + start, ptr, tile_size * sizeof(*ptr), cudaMemcpyDeviceToHost));
    }
    std::fprintf(stderr, "HY4_FLASH_CPU_Q81_ALIGNED active=1 host_table_mib=96\n");
    return host;
}
} // namespace

const uint32_t* device_tables() {
    int device, major, minor;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major != 8 || minor != 9)
        return nullptr;
    static const auto host = make_tables();
    return host.data();
}
} // namespace strata::hy4_flash::cpu::q81_aligned
