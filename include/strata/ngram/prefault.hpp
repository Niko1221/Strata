#pragma once

// Concurrent read-only prefaulting, inspired by upstream Strata PR #651.
// Caller owns a readable mapping for [base, base + bytes) until all workers join.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <system_error>
#include <vector>

namespace strata::ngram {
struct PrefaultStats {
    uint64_t pages = 0;
    uint64_t checksum = 0;
    unsigned threads = 0;
};

inline PrefaultStats prefault_pages(const void* base, size_t bytes, size_t page, unsigned requested) {
    PrefaultStats result;
    if (!bytes || !page || !requested) return result;
    const size_t pages = 1 + (bytes - 1) / page;
    const size_t chunk_pages = std::max<size_t>(1, (64ull << 20) / page);
    const size_t chunks = 1 + (pages - 1) / chunk_pages;
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    unsigned workers = std::min({requested, hardware, 64u});
    workers = (unsigned) std::min<size_t>(workers, chunks);
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> checksum{0};
    auto touch = [&] {
        uint64_t sum = 0;
        const auto* data = static_cast<const volatile uint8_t*>(base);
        for (size_t chunk; (chunk = next.fetch_add(1, std::memory_order_relaxed)) < chunks;) {
            const size_t begin = chunk * chunk_pages;
            const size_t end = begin + std::min(chunk_pages, pages - begin);
            for (size_t i = begin; i < end; ++i) sum += data[i * page];
        }
        checksum.fetch_add(sum, std::memory_order_relaxed);
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    // If thread creation fails, existing workers and the caller finish the work.
    try {
        for (unsigned i = 1; i < workers; ++i) pool.emplace_back(touch);
    } catch (const std::system_error&) {}
    touch();
    for (auto& t : pool) t.join();
    result.pages = pages;
    result.checksum = checksum.load(std::memory_order_relaxed);
    result.threads = (unsigned) pool.size() + 1;
    return result;
}
} // namespace strata::ngram
