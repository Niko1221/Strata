// Model-free allocation and address-stability checks for the actual expert cache.
#include "strata/core/expert_cache.hpp"
#include "strata/core/device.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    using strata::core::ExpertCache;
    std::string err;
    ExpertCache cache;
    if (argc == 2 && std::strcmp(argv[1], "--refusal") == 0) {
        // An integrated device with headroom above physical capacity must refuse
        // even a tiny cache before allocating. This is a separate process so the
        // cached headroom setting is read afresh.
        if (!strata::core::device_info(0).integrated) return 77;
        if (cache.open(1, 1, 1, 4096, err) || cache.valid() || err.empty()) return 1;
        std::printf("budget refusal: %s\n", err.c_str());
        return 0;
    }
    constexpr int64_t segment = 2ll << 20;
    cache.set_segment_bytes(segment);
    if (!cache.open(4, 1, 4, segment, err)) {
        std::fprintf(stderr, "segmented open: %s\n", err.c_str()); return 1;
    }
    if (!cache.segmented() || cache.slots() != 4) return 1;
    auto* address = cache.device_slot(0);
    const int value = 0x12345678;
    if (cudaMemcpy(address, &value, sizeof(value), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (!cache.shrink(segment, err) || cache.slots() != 1 || cache.mapped_bytes() != segment) return 1;
    if (!cache.grow(4 * segment, err) || cache.slots() != 4 || cache.device_slot(0) != address) return 1;
    int restored = 0;
    if (cudaMemcpy(&restored, address, sizeof(restored), cudaMemcpyDeviceToHost) != cudaSuccess || restored != value)
        return 1;
    cache.close();
    cache.set_segment_bytes(0);
    ExpertCache::set_vmm(false);
    if (!cache.open_sized({4096, 8192}, 1, 2, err) || cache.segmented() || cache.slots() != 2) return 1;
    cache.close();
    std::puts("expert cache VMM shrink/grow, stable address, retained data and cudaMalloc path: PASS");
    return 0;
}
