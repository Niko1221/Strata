#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cache_hit.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace strata::core::detail;

static void check(bool ok) {
    if (!ok) std::abort();
}

static void layout(const std::vector<uint64_t>& sizes) {
    const int64_t n = (int64_t) sizes.size();
    std::vector<uint64_t> forward((size_t) n + 1, 0);
    std::partial_sum(sizes.begin(), sizes.end(), forward.begin() + 1);
    auto reverse = forward;
    reverse_cache_offsets(reverse);
    check(reverse.back() == forward.back());
    check(reverse[(size_t) n - 1] == 0);
    uint64_t tail = 0;
    for (int64_t first = n; first >= 0; --first) {
        if (first < n) {
            tail += sizes[(size_t) first];
            check(reverse[(size_t) first] == forward.back() - forward[(size_t) first + 1]);
            check(reverse[(size_t) first] + sizes[(size_t) first] == tail);
        }
        check(cache_prefix_bytes(n, reverse.data(), true, first) == forward[(size_t) first]);
        check(cache_prefix_bytes(n, forward.data(), false, first) == forward[(size_t) first]);
        check(cache_tail_bytes(n, 0, reverse.data(), true, first) == tail);
        check(cache_tail_bytes(n, 0, forward.data(), false, first) == tail);
        if (tail <= kCacheLoanLimit) {
            check(cache_tail_slots(n, 0, reverse.data(), true, tail) == n - first);
            if (tail) check(cache_tail_slots(n, 0, reverse.data(), true, tail - 1) == n - first);
        } else {
            check(cache_tail_slots(n, 0, reverse.data(), true, tail) == n + 1);
        }
        check(cache_tail_slots(n, 0, forward.data(), false, tail) == n - first);
    }
}

static void fill_loan_refill() {
    const std::vector<uint64_t> sizes{256, 768, 512, 256};
    std::vector<uint64_t> offsets{0, 256, 1024, 1536, 1792};
    reverse_cache_offsets(offsets);
    std::vector<uint8_t> arena((size_t) offsets.back(), 0);
    auto fill = [&](size_t slot) {
        std::fill_n(arena.begin() + offsets[slot], sizes[slot], (uint8_t) (slot + 1));
    };
    for (size_t slot = 0; slot < sizes.size(); ++slot) fill(slot);
    const auto mirror = arena;
    const int64_t n = (int64_t) sizes.size();
    const int64_t first = n - cache_tail_slots(n, 0, offsets.data(), true, 700);
    check(first == 2);
    const auto loan = cache_tail_bytes(n, 0, offsets.data(), true, first);
    std::fill_n(arena.begin(), (size_t) loan, 0);
    for (int64_t slot = 0; slot < first; ++slot)
        check(std::equal(arena.begin() + offsets[slot], arena.begin() + offsets[slot] + sizes[slot],
                         mirror.begin() + offsets[slot]));
    for (int64_t slot = first; slot < n; ++slot) fill((size_t) slot);
    check(arena == mirror);
    for (int64_t slot = 0; slot < n; ++slot) {
        std::fill_n(arena.begin() + offsets[slot], sizes[slot], 0);
        std::copy_n(mirror.begin() + offsets[slot], sizes[slot], arena.begin() + offsets[slot]);
    }
    check(arena == mirror);
}

int main() {
    check(!cache_reverse(kCacheGemmLimit - 1, nullptr));
    check(!cache_reverse(kCacheGemmLimit, nullptr));
    check(cache_reverse(kCacheGemmLimit + 1, nullptr));
    check(cache_reverse(256, "1"));
    check(!cache_reverse(9ull << 30, "0"));
    check(cache_reverse(9ull << 30, "invalid"));
    check(kCacheLoanLimit < kCacheGemmLimit);
    check(cache_tail_slots(0, 0, nullptr, false, 0) == 0);
    check(cache_tail_slots(0, 0, nullptr, false, 1024) > 0);
    for (int64_t first = 0; first <= 8; ++first)
        check(cache_tail_bytes(8, 256, nullptr, false, first) == (uint64_t) (8 - first) * 256);
    for (uint64_t need = 0; need <= 2304; ++need)
        check(cache_tail_slots(8, 256, nullptr, false, need) == (int64_t) ((need + 255) / 256));
    layout({256});
    layout({256, 512, 768, 1024});
    layout(std::vector<uint64_t>(8192, 1ull << 20));
    layout({5ull << 30, 256, 3ull << 30});
    layout({256, kCacheLoanLimit});
    layout({256, kCacheLoanLimit + 1});
    const uint64_t off[]{kCacheLoanLimit, 0, kCacheLoanLimit + 256};
    check(cache_tail_slots(2, 0, off, true, kCacheLoanLimit) == 1);
    check(cache_tail_slots(2, 0, off, true, kCacheLoanLimit + 1) == 3);
    const uint64_t oversized[]{0, kCacheLoanLimit + 1};
    check(cache_tail_slots(1, 0, oversized, true, 1) == 2);
    for (int n = 1; n <= 64; ++n) {
        std::vector<uint64_t> sizes((size_t) n);
        for (int i = 0; i < n; ++i) sizes[(size_t) i] = (uint64_t) (1 + (i * 17 + n) % 31) * 256;
        layout(sizes);
    }
    fill_loan_refill();
    std::puts("expert_cache_reverse_test: passed");
}
