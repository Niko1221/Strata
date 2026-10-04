#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::core {

// Independent immutable host blobs, disjoint device destinations. The caller keeps
// both alive until the stream completes. Ordering with earlier/later stream work
// (including the verifier's completion callback) is unchanged. No extra VRAM.
inline cudaError_t copy_expert_blobs(uint8_t* dst, const uint8_t* const* src,
                                     int n, size_t bytes, cudaStream_t stream,
                                     int batch_mode) {
    if (n <= 0) return cudaSuccess;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    constexpr int cap = 128;
    if (batch_mode && n > 1 && n <= cap) {
        std::array<void*, cap> dsts{};
        std::array<const void*, cap> srcs{};
        std::array<size_t, cap> sizes{};
        for (int i = 0; i < n; ++i) {
            dsts[i] = dst + (size_t) i * bytes;
            // Registered host VAs may not be device-visible on Windows. Unlike
            // cudaMemcpyAsync(kind=HostToDevice), the batch API needs the alias.
            void* alias = nullptr;
            const auto mapped = cudaHostGetDevicePointer(&alias, (void*) src[i], 0);
            if (mapped != cudaSuccess) return mapped;
            srcs[i] = alias;
            sizes[i] = bytes;
        }
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
#if CUDART_VERSION >= 13040
        if (batch_mode == 2) attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
#endif
        size_t first = 0;
        return cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                                     (size_t) n, &attr, &first, 1, stream);
    }
#else
    (void) batch_mode;
#endif
    for (int i = 0; i < n; ++i) {
        const auto e = cudaMemcpyAsync(dst + (size_t) i * bytes, src[i], bytes,
                                       cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace strata::core
