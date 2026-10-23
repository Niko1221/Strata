#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

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
    // A disk park keeps only the disk-park margin free, not the full --conversation-cache-min-free-mib: a nearly-full
    // box must still park.  The numbers are the field log (need ~355 MiB, ~1.8 GiB free, refused by the 2560 gate).
    const auto MiB = [](uint64_t n) { return n << 20; };
    check(disk_park_floor_mib(2560) == DISK_PARK_MIN_FREE_MIB, "the default 2560 floor caps to the park margin");
    check(DISK_PARK_MIN_FREE_MIB == 512, "the park margin is 512 MiB");
    check(disk_park_floor_mib(256) == 256, "a floor below the margin is honored");
    check(disk_park_floor_mib(0) == 0, "a zero floor parks with no margin");
    // the reported refusal: 1803 MiB free, a 355 MiB snapshot.  The RAM floor refused it; the park margin admits it.
    check(!conversation_memory_admit(MiB(1803), MiB(355), MiB((uint64_t) 2560)), "field case: the 2560 floor refused it");
    check(conversation_memory_admit(MiB(1803), MiB(355), MiB((uint64_t) disk_park_floor_mib(2560))),
          "field case: the park margin admits it");
    // but not below the snapshot itself: there is nowhere to read the running state, so a park still skips.
    check(!conversation_memory_admit(MiB(100), MiB(355), MiB((uint64_t) disk_park_floor_mib(2560))),
          "field case: under the snapshot's own size, park skips");
    // #3493: the budget used when --conversation-cache-mib is not given
    const auto GiB = [](uint64_t n) { return n << 30; };
    check(auto_conversation_cache_mib({}, 2560) == 0, "unknown telemetry: no cache");
    check(auto_conversation_cache_mib(GiB(2), 2560) == 0, "below the parking floor: off");
    check(auto_conversation_cache_mib(GiB(4), 2560) == 0, "4 GiB free: 40% of 1.5 GiB is under 1 GiB: off");
    check(auto_conversation_cache_mib(GiB(8), 2560) == 2048, "8 GiB free: 40% of 5.5 GiB, in steps of 256");
    check(auto_conversation_cache_mib(GiB(20), 2560) == 7168, "20 GiB free: 40% of 17.5 GiB");
    check(auto_conversation_cache_mib(GiB(40), 2560) == 8192, "capped at 8 GiB");
    check(auto_conversation_cache_mib(GiB(1000), 0) == 8192, "a big server: still the cap");
    check(auto_conversation_cache_mib(GiB(40), 1000000) == 0, "a floor above the RAM: off, no underflow");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = available_host_bytes();
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
