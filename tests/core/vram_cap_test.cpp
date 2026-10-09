// CPU only; no CUDA headers/runtime, model files or device queries.
// Also built as the vram_cap_test CMake target with STRATA_BUILD_TESTS=ON.
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
    cap::Mode mode = cap::Mode::Quality;
    check(cap::parse_mode(nullptr, mode) && mode == cap::Mode::Fast, "absent mode: fast");
    check(cap::parse_mode("fast", mode) && mode == cap::Mode::Fast, "explicit fast");
    check(cap::parse_mode("quality", mode) && mode == cap::Mode::Quality, "explicit quality");
    for (const char* bad : {"", "QUALITY", "slow", "quality ", "1"})
        check(!cap::parse_mode(bad, mode) && mode == cap::Mode::Quality, "bad mode rejected without changing output");
    check(!cap::quality_active(cap::Mode::Fast), "a cap alone keeps the fast split");
    check(cap::quality_active(cap::Mode::Quality), "explicit quality also enables an uncapped A/B reference");
    for (const double requested : {-1.0, 0.0, 0.2, 0.55, 1.0}) {
        check(cap::pcie_fraction(cap::Mode::Fast, requested) == requested, "flag absent preserves CLI/link/request shares");
        check(cap::pcie_fraction(cap::Mode::Quality, requested) == 1.0, "quality overrides CLI/request shares");
    }
    constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    check(cap::startup_haircut_bytes(1.0, true) == 0, "cap off: no startup haircut on WDDM");
    check(cap::startup_haircut_bytes(1.0, false) == 0, "cap off: no startup haircut on Linux");
    check(cap::startup_haircut_bytes(0.8, true) == GiB, "active cap: one fixed WDDM haircut");
    check(cap::startup_haircut_bytes(0.8, false) == 0, "native driver: no WDDM haircut");
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
    // Freeze the pre-touch budget. Changing the post-touch reading can only accept/refuse it, not give it
    // a new slot count. The 1 GiB haircut is charged once before touch, not again to the late-buffer check.
    const uint64_t late = 700 * MiB;
    const uint64_t haircut = cap::startup_haircut_bytes(0.8, true);
    const uint64_t frozen_room = cap::cache_room(8 * GiB, 2048, late + haircut);
    check(frozen_room / blob == 2210, "frozen sizing includes exactly one haircut");
    check(cap::cache_room(8 * GiB, 2048, late + haircut + 32 * MiB) / blob == 2194,
          "extra quality staging is booked before sizing, outside the cap floor");
    check(cap::post_touch_fits(2048 * MiB + late, 2048, late), "post-touch exact fit accepted");
    check(!cap::post_touch_fits(2048 * MiB + late - 1, 2048, late), "post-touch one byte short refused");
    check(cap::post_touch_fits(2048 * MiB, 2048, 0), "floor alone at exact fit accepted");
    for (const uint64_t post : {0ull, 393 * MiB, 422 * MiB, 2048 * MiB, 4096 * MiB}) {
        check(cap::post_touch_fits(post, 2048, late) == (post >= 2048 * MiB + late), "post-touch accept/refuse only");
        check(frozen_room / blob == 2210, "0/393/422 MiB post-touch readings cannot resize the plan");
    }
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
