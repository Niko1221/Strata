#pragma once
#include "strata/qwen35/kv_layout.hpp"
#include <cuda_runtime.h>

namespace strata::qwen35 {
// A view borrows its session allocation. Session/device teardown remains Engine-owned.
struct Q36KvView { void* data; const Q36KvLayout& layout; };
inline cudaError_t q36_copy_kv_prefix(Q36KvView dst, Q36KvView src, size_t valid, cudaStream_t stream) {
    require_kv_identity(src.layout, dst.layout, valid);
    if (!valid) return cudaSuccess;
    return cudaMemcpy2DAsync(dst.data, dst.layout.head_stride, src.data, src.layout.head_stride,
                             src.layout.prefix_bytes(valid), src.layout.kv_heads, cudaMemcpyDeviceToDevice, stream);
}
inline cudaError_t q36_clear_kv(Q36KvView cache, cudaStream_t stream) {
    if (!cache.layout.executable()) throw std::invalid_argument("Q36 KV codec has no clear implementation");
    // Clear capacity, including speculative suffixes beyond the committed prefix.
    return cudaMemsetAsync(cache.data, 0, cache.layout.allocation_bytes, stream);
}
}  // namespace strata::qwen35
