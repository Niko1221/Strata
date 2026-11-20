#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::qwen35 {
enum class Q36KvType { Fp32, Fp16, Q8_0, Nvfp4 };
// Q8_0 and NVFP4 describe proposed storage only. No executable codec is advertised.
inline constexpr const char* q36_kv_capabilities =
    "Q36_KV_CAPABILITIES {\"schema\":1,\"family\":\"qwen35moe\",\"layoutVersion\":1,"
    "\"scaleSchema\":\"none\",\"cacheTypesK\":[\"f32\",\"f16\"],\"cacheTypesV\":[\"f32\",\"f16\"],"
    "\"flags\":[\"--cache-type-k\",\"--cache-type-v\"],\"quality\":\"experimental-unqualified\"}";
inline size_t kv_checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error("Q36 KV size overflow");
    return a + b;
}
inline size_t kv_checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) throw std::overflow_error("Q36 KV size overflow");
    return a * b;
}
inline size_t kv_align(size_t n, size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) throw std::invalid_argument("Q36 KV alignment");
    return kv_checked_add(n, alignment - 1) & ~(alignment - 1);
}
struct Q36KvLayout {
    Q36KvType type = Q36KvType::Fp32;
    uint32_t version = 1, scale_schema = 0;
    size_t context_capacity = 0, kv_heads = 0, head_dim = 0;
    size_t row_stride = 0, head_stride = 0, block_stride = 0;
    size_t block_scale_row_stride = 0, block_scale_head_stride = 0;
    size_t outer_scale_row_stride = 0, outer_scale_head_stride = 0;
    size_t payload_offset = 0, block_scale_offset = 0, outer_scale_offset = 0;
    size_t payload_bytes = 0, block_scale_bytes = 0, outer_scale_bytes = 0;
    size_t allocation_bytes = 0, padding_bytes = 0;
    bool executable() const { return type == Q36KvType::Fp32 || type == Q36KvType::Fp16; }
    size_t prefix_bytes(size_t valid) const {
        if (!executable() || valid > context_capacity) throw std::invalid_argument("Q36 KV prefix/layout");
        return kv_checked_mul(valid, row_stride);
    }
    bool same_identity(const Q36KvLayout& b) const {
        return type == b.type && version == b.version && scale_schema == b.scale_schema &&
            context_capacity == b.context_capacity && kv_heads == b.kv_heads && head_dim == b.head_dim;
    }
};
// Head-major scalar storage is byte-identical to the original F32/F16 allocation.
// Proposed Q8_0 rows use interleaved 2-byte scales + 32 signed bytes per block.
// Proposed NVFP4 uses separate 256-byte-aligned payload/E4M3/FP32 planes, never row padding.
inline Q36KvLayout make_kv_layout(Q36KvType type, size_t context, size_t heads, size_t dim) {
    if (!context || !heads || !dim) throw std::invalid_argument("Q36 KV geometry must be positive");
    Q36KvLayout l;
    l.type = type; l.context_capacity = context; l.kv_heads = heads; l.head_dim = dim;
    const size_t rows = kv_checked_mul(context, heads);
    if (type == Q36KvType::Fp32 || type == Q36KvType::Fp16) {
        l.row_stride = kv_checked_mul(dim, type == Q36KvType::Fp32 ? 4 : 2);
        l.head_stride = kv_checked_mul(context, l.row_stride);
        l.payload_bytes = l.allocation_bytes = kv_checked_mul(heads, l.head_stride);
    } else if (type == Q36KvType::Q8_0) {
        if (dim % 32) throw std::invalid_argument("Q36 proposed Q8_0 requires full 32-value blocks");
        l.scale_schema = 1; l.block_stride = 34; l.payload_offset = 2;
        l.row_stride = kv_checked_mul(dim / 32, 34);
        l.block_scale_row_stride = l.row_stride;
        l.block_scale_head_stride = kv_checked_mul(context, l.row_stride);
        l.head_stride = kv_checked_mul(context, l.row_stride);
        l.payload_bytes = kv_checked_mul(rows, dim);
        l.block_scale_bytes = kv_checked_mul(rows, kv_checked_mul(dim / 32, 2));
        l.allocation_bytes = kv_checked_mul(heads, l.head_stride);
    } else if (type == Q36KvType::Nvfp4) {
        if (dim % 16) throw std::invalid_argument("Q36 proposed NVFP4 requires full 16-value blocks");
        l.scale_schema = 2; l.block_stride = 8; l.row_stride = dim / 2;
        l.head_stride = kv_checked_mul(context, l.row_stride);
        l.payload_bytes = kv_checked_mul(rows, l.row_stride);
        l.block_scale_row_stride = dim / 16;
        l.block_scale_head_stride = kv_checked_mul(context, l.block_scale_row_stride);
        l.outer_scale_row_stride = 4;
        l.outer_scale_head_stride = kv_checked_mul(context, 4);
        l.block_scale_bytes = kv_checked_mul(rows, dim / 16);
        l.outer_scale_bytes = kv_checked_mul(rows, 4);
        l.block_scale_offset = kv_align(l.payload_bytes, 256);
        l.outer_scale_offset = kv_align(kv_checked_add(l.block_scale_offset, l.block_scale_bytes), 256);
        l.allocation_bytes = kv_align(kv_checked_add(l.outer_scale_offset, l.outer_scale_bytes), 256);
        l.padding_bytes = l.allocation_bytes - kv_checked_add(l.payload_bytes, kv_checked_add(l.block_scale_bytes, l.outer_scale_bytes));
    } else throw std::invalid_argument("Q36 unknown KV type");
    return l;
}
inline void require_kv_identity(const Q36KvLayout& a, const Q36KvLayout& b, size_t valid) {
    if (!a.executable() || !b.executable() || !a.same_identity(b)) throw std::invalid_argument("Q36 incompatible KV layout");
    (void) a.prefix_bytes(valid);
}
}  // namespace strata::qwen35
