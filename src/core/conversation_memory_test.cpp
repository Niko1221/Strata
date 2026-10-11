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
    {
        // The probe is what tells a size the machine cannot commit from one refused by the moment, and it
        // runs in the bad_alloc handler. A size of 0 was never worth asking about: a park that reuses all
        // of its K/V asks for nothing new routinely, and the probe must not imply a cause it did not measure.
        const AllocationProbe none = conversation_memory_probe(0);
        check(!none.probed && !none.reserved && !none.committed && none.error == 0,
              "a zero-byte request is not probed");
        check(std::string(conversation_memory_probe_text(none)).find("nothing new") != std::string::npos,
              "the zero-byte probe says the size is not the cause");

        const size_t modest = 4 << 20;
        const AllocationProbe fits = conversation_memory_probe(modest);
        check(fits.probed && fits.reserved && fits.committed && fits.error == 0,
              "a size this machine has room for reserves and commits");
        check(std::string(conversation_memory_probe_text(fits)).find("allocates now") != std::string::npos,
              "the text for a size that fits says so");

        // The distinction the field needs: reserved-but-not-committed is the commit limit, and only a
        // fabricated probe can show it here - the real machine either has the room or does not.
        AllocationProbe commit_short;
        commit_short.probed = commit_short.reserved = true;
        check(std::string(conversation_memory_probe_text(commit_short)).find("commit is exhausted") != std::string::npos,
              "a size that reserves but will not commit is reported as commit exhaustion");
        AllocationProbe no_space;
        no_space.probed = true;
        check(std::string(conversation_memory_probe_text(no_space)).find("address space") != std::string::npos,
              "a size that cannot even be reserved is reported as address space");

        // The real provider is not assumed to be empty: what it says is either a usable sample or unknown.
        const auto room = available_host_bytes();
        if (room) check(conversation_memory_probe(*room).probed, "the probe reports for the room the guard measured");
    }
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
