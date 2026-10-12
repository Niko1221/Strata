// A real CUDA probe; unavailable GPU support is a failure, never a skip.
#include "strata/core/branch_digest.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <vector>

int main() {
    using namespace strata::core;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(std::cin)), {});
    if (cudaFree(nullptr) != cudaSuccess) return 2;
    const auto before = branch_digest_counters();
    {
        BranchDigestWorkspace workspace;
        const auto size = bytes.empty() ? 1 : bytes.size();
        if (!workspace.reserve(size)) return 3;
        uint8_t* device = nullptr;
        if (cudaMalloc(&device, size) != cudaSuccess) return 4;
        if (!bytes.empty() && cudaMemcpy(device, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) != cudaSuccess)
            return 5;
        std::vector<BranchDigest> digests;
        if (!workspace.digest(device, bytes.size(), digests)) return 6;
        if (cudaFree(device) != cudaSuccess) return 7;
        const auto measured = branch_digest_counters();
        std::printf("{\"digests\":[");
        for (size_t i = 0; i < digests.size(); ++i) {
            if (i) std::putchar(',');
            std::putchar('"');
            for (auto byte : digests[i].bytes) std::printf("%02x", byte);
            std::putchar('"');
        }
        std::printf("],\"device_hashed_bytes\":%llu,\"digest_copy_bytes\":%llu,\"kernel_calls\":%llu",
            (unsigned long long)(measured.device_hashed_bytes - before.device_hashed_bytes),
            (unsigned long long)(measured.digest_copy_bytes - before.digest_copy_bytes),
            (unsigned long long)(measured.kernel_calls - before.kernel_calls));
    }
    const auto after = branch_digest_counters();
    std::printf(",\"workspace_allocations\":%llu,\"workspace_releases\":%llu,\"workspace_live_bytes\":%llu}\n",
        (unsigned long long)(after.workspace_allocations - before.workspace_allocations),
        (unsigned long long)(after.workspace_releases - before.workspace_releases),
        (unsigned long long)after.workspace_live_bytes);
    return after.workspace_live_bytes == before.workspace_live_bytes ? 0 : 8;
}
