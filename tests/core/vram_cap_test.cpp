// CPU only; no CUDA headers/runtime, model files or device queries.
// MSVC: cl /std:c++20 /EHsc /Iinclude tests/core/vram_cap_test.cpp /Febuild-vram/vram_cap_test.exe
#include "strata/program/vram_cap.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace cap = strata::program::vram_cap;
static int checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

int main() {
    double f = 0.5;
    check(cap::parse_fraction(nullptr, nullptr, f) && f == 1.0, "absent: cap off");
    check(cap::parse_fraction(nullptr, "0.8", f) && f == 0.8, "environment fraction");
    check(cap::parse_fraction("0.8", "0.5", f) && f == 0.8, "CLI wins");
    check(cap::parse_fraction("1", "not a fraction", f) && f == 1.0, "explicit off wins even over invalid env");
    check(!cap::parse_fraction("bad", "0.8", f), "bad CLI does not fall back to env");
    check(cap::parse_fraction("8e-1", nullptr, f) && f == 0.8, "scientific notation");
    check(cap::parse_fraction("5e-324", nullptr, f) && f > 0.0, "finite positive subnormal is in range");
    for (const char* bad : {"", "0", "-0", "-0.1", "1.00001", "80", "nan", "NaN", "inf", "-inf",
                            "1e309", "1e-999", "0.8garbage", "0.8%", "0.8 0.7", " "}) {
        f = 0.6;
        check(!cap::parse_fraction(bad, nullptr, f) && f == 0.6, "invalid CLI leaves output alone");
        check(!cap::parse_fraction(nullptr, bad, f), "invalid env rejected");
    }
    constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    check(cap::floor_mib(12 * GiB, 0.8) == 2458, "12 GiB at 80% rounds up to 2458 MiB");
    check(cap::floor_mib(10 * GiB, 0.8) == 2048, "10 GiB at 80% keeps 2048 MiB");
    check(cap::floor_mib(12 * GiB, 1.0) == 0, "off has no floor");
    check(cap::floor_mib(2 * MiB + 1, 0.5) == 2, "non-MiB total rounds conservatively");
    check(cap::floor_mib(1, 0.5) == 1, "a fractional MiB is not rounded down");
    check(cap::floor_mib(12 * GiB, std::numeric_limits<double>::min()) == 12288, "tiny positive fraction");
    check(cap::reserve_mib(700, 12 * GiB, 1.0) == 700, "default reserve unchanged");
    check(cap::reserve_mib(0, 12 * GiB, 1.0) == 0, "explicit zero unchanged with cap off");
    check(cap::reserve_mib(700, 12 * GiB, 0.8) == 2458, "cap beats smaller reserve");
    check(cap::reserve_mib(3000, 12 * GiB, 0.8) == 3000, "explicit larger reserve wins");
    check(cap::reserve_mib(0, 12 * GiB, 0.8) == 2458, "explicit zero cannot bypass cap");
    check(cap::reserve_mib(700, 8 * GiB, 0.8) == 1639, "each GPU uses its own total");
    check(cap::cache_room(7 * GiB, 2048, 800 * MiB) == 4320 * MiB, "late buffers do not spend cap floor");
    check(cap::cache_room(100 * MiB, 700) == 0, "already short: zero room, no underflow");
    check(cap::cache_room(700 * MiB, 700) == 0, "at reserve: zero room");
    check(cap::cache_room(900 * MiB, 700, 300 * MiB) == 0, "late buffers exhaust room");
    check(cap::cache_room(7 * GiB, 700) == 6468 * MiB, "off budget is old free minus reserve");
    const uint64_t blob = 2 * MiB;
    const uint64_t capped_slots = cap::cache_room(8 * GiB, cap::reserve_mib(700, 10 * GiB, 0.8), 700 * MiB) / blob;
    check(capped_slots == 2722, "fixed-cache capacity uses total VRAM floor plus late headroom");
    check(std::min<uint64_t>(6000, capped_slots) == 2722, "oversized fixed cache is clamped");
    check(std::min<uint64_t>(1024, capped_slots) == 1024, "smaller fixed request is unchanged");
    check(cap::cache_room(8 * GiB, 700) / blob == 3746, "cap-off fixed capacity retains old reserve math");
    check(cap::cache_room(8 * GiB, 2048, 700 * MiB + GiB) / blob == 2210, "WDDM pre-touch margin is booked before allocation");
    for (const uint64_t gib : {4ull, 6ull, 8ull, 10ull, 12ull, 16ull, 24ull})
        for (const double frac : {0.5, 0.8, 0.95})
            for (const int64_t reserve : {0ll, 700ll, 3000ll}) {
                const uint64_t total = gib * GiB;
                const uint64_t free = total - GiB;   // fixed engine allocations (or another app) already counted
                const uint64_t late = 512 * MiB;
                const uint64_t room = cap::cache_room(free, cap::reserve_mib(reserve, total, frac), late);
                if (room > 0)
                    check((long double) (GiB + room + late) <= (long double) total * frac,
                          "fixed + cache + later <= fraction of total, not fraction of free");
            }
    std::printf("vram_cap_test: %d CPU checks passed (no GPU calls)\n", checks);
    return 0;
}
