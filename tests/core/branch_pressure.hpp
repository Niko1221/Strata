#pragma once
// Test-only ownership of actual CUDA physical allocations. Run only while the
// device is reserved for this test; no synthetic allocation failure is injected.
#include "strata/core/vmm.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct BranchPressure {
    std::vector<strata::core::VmmChunk> handles;
    uint64_t allocated_bytes = 0, released_bytes = 0;
    size_t free_before = 0, free_after_fill = 0, free_after_release = 0, total = 0;

    ~BranchPressure() { release(); }
    bool fill() {
        if (!handles.empty() || cudaMemGetInfo(&free_before, &total) != cudaSuccess) return false;
        const auto granularity = strata::core::vmm_granularity();
        if (!granularity) return false;
        // Reserve host bookkeeping first so a host allocator exception cannot
        // be mistaken for the driver's actual device-memory exhaustion.
        handles.reserve(total / granularity + 1);
        const auto before = strata::core::vmm_failure_counters();
        for (;;) {
            const auto handle = strata::core::vmm_chunk_new();
            if (!handle) break;
            handles.push_back(handle);
            allocated_bytes += granularity;
        }
        const auto after = strata::core::vmm_failure_counters();
        return cudaMemGetInfo(&free_after_fill, &total) == cudaSuccess && allocated_bytes > 0 &&
            after.out_of_memory_failures == before.out_of_memory_failures + 1 &&
            after.last_error == CUDA_ERROR_OUT_OF_MEMORY;
    }
    bool release() {
        bool good = true;
        const auto granularity = strata::core::vmm_granularity();
        for (auto& handle : handles) {
            if (!handle) continue;
            if (strata::core::vmm_chunk_release(handle)) {
                handle = 0;
                released_bytes += granularity;
            } else good = false;
        }
        if (good) handles.clear();
        return good && cudaMemGetInfo(&free_after_release, &total) == cudaSuccess;
    }
};
