#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <new>
#include <vector>

namespace strata::prefill::detail {

// The gather future has joined before this is called. The last upload may still be
// using the old allocation; retain it until that upload completes, even on fallback.
inline bool ensure_ple_host(float*& data, std::vector<float>& pageable, size_t& capacity,
                            size_t count, cudaEvent_t copied, bool force_pageable) {
    if (data && capacity >= count) return true;
    if (data) {
        if (cudaEventSynchronize(copied) != cudaSuccess) return false;
        if (pageable.empty()) {
            if (cudaFreeHost(data) != cudaSuccess) return false;
        } else {
            std::vector<float>().swap(pageable);
        }
        data = nullptr;
        capacity = 0;
    }
    if (force_pageable || cudaHostAlloc((void**) &data, count * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        try { pageable.resize(count); }
        catch (const std::bad_alloc&) { return false; }
        data = pageable.data();
    }
    capacity = count;
    return true;
}

}  // namespace strata::prefill::detail
