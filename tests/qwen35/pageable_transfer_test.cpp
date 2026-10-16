#include "strata/qwen35/pageable_transfer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "pageable transfer test: %s\n", message);
        std::exit(1);
    }
}
}  // namespace

int main() {
    constexpr size_t bytes = 37;
    constexpr size_t capacity = 8;
    std::vector<uint8_t> source(bytes), destination(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) source[i] = static_cast<uint8_t>((i * 73 + 19) & 0xff);

    std::vector<size_t> chunks;
    strata::qwen35::copy_pageable_in_chunks(destination.data(), source.data(), bytes, capacity,
        [&](uint8_t* dst, const uint8_t* src, size_t count) {
            chunks.push_back(count);
            std::memcpy(dst, src, count);
        });
    require(destination == source, "bytes changed or omitted");
    require(chunks == std::vector<size_t>{8, 8, 8, 8, 5}, "wrong chunk boundaries");

    chunks.clear();
    strata::qwen35::copy_pageable_in_chunks(nullptr, nullptr, 0, 0,
        [&](uint8_t*, const uint8_t*, size_t) { chunks.push_back(1); });
    require(chunks.empty(), "empty transfer called the callback");

    bool rejected_capacity = false;
    try {
        strata::qwen35::copy_pageable_in_chunks(destination.data(), source.data(), 1, 0,
            [](uint8_t*, const uint8_t*, size_t) {});
    } catch (const std::invalid_argument&) {
        rejected_capacity = true;
    }
    require(rejected_capacity, "zero capacity accepted");

    bool propagated_failure = false;
    try {
        strata::qwen35::copy_pageable_in_chunks(destination.data(), source.data(), 1, capacity,
            [](uint8_t*, const uint8_t*, size_t) { throw std::runtime_error("copy"); });
    } catch (const std::runtime_error&) {
        propagated_failure = true;
    }
    require(propagated_failure, "copy callback failure did not propagate");
    std::puts("pageable_transfer_test OK");
}
