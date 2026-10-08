#include "strata/core/conversation_memory.hpp"

#include "strata/core/conversation_buffer.hpp"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
#if defined(__GLIBC__)
    {   // glibc keeps freed 16 MiB segments in its heap once its mmap threshold has risen; releasing hands them back
        auto rss_mib = [] {
            std::ifstream status("/proc/self/status");
            for (std::string line; std::getline(status, line);)
                if (line.rfind("VmRSS:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
            return -1.0;
        };
        std::deque<strata::core::ConversationBuffer> parked;
        std::vector<std::vector<char>> small_allocations;
        for (int round = 0; round < 12; ++round) {   // park a new snapshot, drop the oldest, as a long-running server does
            strata::core::ConversationBuffer snapshot;
            snapshot.resize(64u << 20, 1);
            parked.push_back(std::move(snapshot));
            small_allocations.emplace_back(4096 + round * 37, 'x');
            if (parked.size() > 3) parked.pop_front();
        }
        const double before = rss_mib();
        parked.pop_front();
        conversation_release_freed_memory();
        check(rss_mib() <= before - 48.0, "a dropped 64 MiB snapshot is back with the kernel after the release");
    }
#endif
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = conversation_available_memory();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total physical memory, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX physical{};
    physical.dwLength = sizeof physical;
    check(GlobalMemoryStatusEx(&physical) != 0, "Windows physical-memory API is available");
    check(available.has_value() && *available <= physical.ullTotalPhys,
          "Windows provider returns a known, physically bounded sample");
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
