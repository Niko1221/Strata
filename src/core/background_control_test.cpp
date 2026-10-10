#include "strata/core/background_control.hpp"
#include <cstdio>
#include <limits>

int main() {
    strata::core::BackgroundControl c;
    int failures = 0;
    auto check = [&](bool ok, const char* why) {
        if (!ok) { std::fprintf(stderr, "%s\n", why); ++failures; }
    };
    check(!c.active(0), "initial state must not throttle");
    check(c.ingest("BACKGROUND 5000 20 1", 100), "valid wait");
    check(c.waiting(100) && c.delay_ms(100) == 20 && c.remaining_ms(100) == 5000, "lease begins");
    check(c.waiting(5099) && !c.waiting(5100) && c.delay_ms(5100) == 0, "lost server lease expires");
    check(!c.waiting(99), "backwards clock does not prolong lease");
    check(c.ingest("BACKGROUND 5000 10 0", 1000), "newest control replaces old wait");
    check(!c.waiting(1000) && c.delay_ms(1000) == 10, "throttle is not a wait");
    for (const auto* bad : {"BACKGROUND 5001 0 1", "BACKGROUND -1 0 1", "BACKGROUND 100 101 0",
                           "BACKGROUND 100 0 2", "BACKGROUND 0 1 0", "BACKGROUND 0 0 1",
                           "BACKGROUND 100 0 0 extra", "BACKGROUND 1.5 0 0", "BACKGROUND 100 0",
                           "BACKGROUND 18446744073709551616 0 0", "MEMORY 100 0 0"}) {
        check(!c.ingest(bad, 1100), "malformed control rejected");
        check(c.delay_ms(1100) == 10 && !c.waiting(1100), "malformed control preserves prior lease");
    }
    check(!c.ingest("BACKGROUND 5 0 1", std::numeric_limits<int64_t>::max() - 1), "overflow rejected");
    check(!c.ingest("BACKGROUND 5 0 1", -1), "invalid clock rejected");
    check(c.ingest("BACKGROUND 0 0 0", 1200) && !c.active(1200), "explicit release");
    std::printf("background_control_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
