// Exercise registration failure without exhausting the machine's shared RAM.
// GNU ld wraps only this test's cudaHostRegister references.
#include "strata/core/pinned.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" cudaError_t __real_cudaHostRegister(void*, size_t, unsigned int);
static int refused = 0;
extern "C" cudaError_t __wrap_cudaHostRegister(void*, size_t, unsigned int) {
    ++refused;
    // Seed an actual runtime error; the fallback must consume it before the
    // next engine operation. No large allocation or driver resource is needed.
    (void) __real_cudaHostRegister(nullptr, 4096, cudaHostRegisterMapped);
    return cudaErrorMemoryAllocation;
}

int main() {
    constexpr size_t bytes = 64u << 10;
    int failures = 0;
    for (bool sliced : {false, true}) {
        const std::vector<uint64_t> bounds = sliced ? std::vector<uint64_t>{0, bytes / 2, bytes}
                                                    : std::vector<uint64_t>{};
        strata::core::PinnedArena arena(bytes, bounds);
        if (!arena.valid() || arena.registered_bytes != 0 || arena.registered_slices != 0 ||
            arena.note.find("FAILED") == std::string::npos || cudaGetLastError() != cudaSuccess) {
            std::fprintf(stderr, "fallback state: %s\n", arena.note.c_str()); ++failures; continue;
        }
        std::memset(arena.data(), 0x87, bytes);
        void* device = nullptr;
        std::vector<unsigned char> restored(bytes);
        if (cudaMalloc(&device, bytes) != cudaSuccess) return 2;
        const bool copied = cudaMemcpyAsync(device, arena.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                            cudaDeviceSynchronize() == cudaSuccess &&
                            cudaMemcpy(restored.data(), device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(device);
        if (!copied || std::memcmp(restored.data(), arena.data(), bytes) != 0) ++failures;
    }
    if (refused != 3) ++failures; // whole arena; whole arena then first slice
    std::printf("pinned registration refusal, sliced refusal, error cleanup, pageable copy: %d failures\n", failures);
    return failures ? 1 : 0;
}
