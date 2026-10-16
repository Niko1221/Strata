#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35 {

// Copy pageable host input through caller-owned bounded staging. The callback
// must finish consuming each chunk before return so registered pages stay bounded.
template <class Copy>
void copy_pageable_in_chunks(uint8_t* dst, const uint8_t* src, size_t bytes, size_t capacity, Copy&& copy) {
    if (bytes == 0) return;
    if (dst == nullptr || src == nullptr || capacity == 0) {
        throw std::invalid_argument("invalid pageable transfer");
    }

    for (size_t offset = 0; offset < bytes;) {
        const size_t count = std::min(capacity, bytes - offset);
        copy(dst + offset, src + offset, count);
        offset += count;
    }
}

}  // namespace strata::qwen35
