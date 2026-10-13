#include "strata/core/remote_reserve.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

using strata::core::detail::remote_cache_fits;
using strata::core::detail::remote_reserve_bytes;

int main() {
    constexpr uint64_t MiB = 1ull << 20;
    constexpr uint64_t poison = std::numeric_limits<uint64_t>::max();
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        if (!ok) { std::cerr << "FAIL: " << name << '\n'; ++failures; }
    };
    auto accepted = [&](const char* value, uint64_t work, uint64_t expected) {
        uint64_t bytes = poison;
        std::string err;
        check(remote_reserve_bytes(value, work, bytes, err) && bytes == expected && err.empty(),
              value ? value : "unset preserves 512 MiB");
    };
    auto refused = [&](const char* value, uint64_t work) {
        uint64_t bytes = poison;
        std::string err;
        check(!remote_reserve_bytes(value, work, bytes, err) && !err.empty() && bytes == poison,
              value ? value : "default cannot fit work buffers");
    };
    accepted(nullptr, 112 * MiB, 512 * MiB);
    accepted("512", 112 * MiB, 512 * MiB);
    accepted("128", 112 * MiB, 128 * MiB);
    accepted("8192", 112 * MiB, 8192 * MiB);
    accepted("129", 112 * MiB + 1, 129 * MiB);
    refused("128", 112 * MiB + 1);
    refused("127", 112 * MiB);
    refused("0", 0);
    refused(nullptr, 512 * MiB);
    refused("8192", poison);
    for (const char* value : {"", "-1", "+128", " 128", "128 ", "128MiB", "1.5", "8193",
                              "4294967296", "999999999999999999999999999999"})
        refused(value, 0);

    // The unchanged default and an explicit 512 select the same admission limit.
    for (uint64_t free_mib : {128ull, 512ull, 640ull, 1024ull}) {
        for (uint64_t selected_mib : {0ull, 64ull, 512ull}) {
            const uint64_t next = 64 * MiB;
            const bool historical = (selected_mib * MiB + next + 512 * MiB) <= free_mib * MiB;
            check(remote_cache_fits(free_mib * MiB, selected_mib * MiB, next, 512 * MiB) == historical,
                  "default admission matches historical limit");
        }
    }
    check(remote_cache_fits(640 * MiB, 448 * MiB, 64 * MiB, 128 * MiB), "exact budget fits");
    check(!remote_cache_fits(640 * MiB, 448 * MiB, 64 * MiB + 1, 128 * MiB), "one byte over refused");
    check(!remote_cache_fits(640 * MiB, 448 * MiB, 64 * MiB, 512 * MiB), "reserve changes admission");
    check(!remote_cache_fits(127 * MiB, 0, 0, 128 * MiB), "reserve itself must fit");
    check(!remote_cache_fits(poison, poison, 1, 512 * MiB), "selected-byte sum cannot wrap");
    check(!remote_cache_fits(poison, 0, poison, 512 * MiB), "next-byte sum cannot wrap");
    if (!failures) std::cout << "remote_reserve_test: OK (parsing, work-buffer floor, default admission and bounds)\n";
    return failures ? 1 : 0;
}
