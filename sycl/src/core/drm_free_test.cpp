// src/core/drm_free_test.cpp - when the DRM fdinfo replaces the driver's free figure (host only, no card needed).
#include "strata/sycl_drm_free.hpp"

#include <cstdio>

namespace {
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }
}  // namespace

int main() {
    using strata::drm_total_replaces_free;
    struct Case { const char* what; uint64_t free_b, total_b, own; bool replaces; };
    const uint64_t mib16 = 16ull << 20;
    const Case cases[] = {
        {"2x Arc Pro B70, CUDA1 after its weights", gib(26.97), gib(31.89), gib(28.08), false},
        {"2x Arc Pro B60, CUDA1 after its weights", gib(18.32), gib(23.91), gib(17.73), false},
        {"2x Arc Pro B60, CUDA0 with both stages' weights", gib(19.6), gib(23.91), gib(9.47), false},
        {"Arc A750, 5.5 GiB held, whole card reported free", gib(8.0), gib(8.0), gib(5.5), true},
        {"Arc A750, 5.5 GiB held, driver 16 MiB short of total", gib(8.0) - mib16, gib(8.0), gib(5.5), true},
        {"Arc A750, 5.5 GiB held, driver 16 MiB + 1 B short", gib(8.0) - mib16 - 1, gib(8.0), gib(5.5), false},
        {"no DRM total (fdinfo unreadable)", gib(8.0), gib(8.0), 0, false},
        {"DRM total larger than the card", gib(8.0), gib(8.0), gib(9.0), false},
        {"DRM total below what the driver says is taken", gib(8.0) - mib16, gib(8.0), mib16 / 2, false},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = drm_total_replaces_free(c.free_b, c.total_b, c.own);
        std::printf("drm_total_replaces_free %-52s -> %s%s\n", c.what, got ? "DRM total" : "driver",
                    got == c.replaces ? "" : "   <-- WRONG");
        fail |= got != c.replaces;
    }
    std::printf("drm_free_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
