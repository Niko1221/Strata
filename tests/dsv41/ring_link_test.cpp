// ring_link_test: the staging ring's slots by the measured link (ds4_ring_link.hpp, Project Maya #609).  No GPU, no model.
#include <cstdio>

#include "ds4_ring_link.hpp"

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main() {
    namespace d = strata::ds4;
    const double GB = 1e9;
    const size_t v100 = 6912000, cmp = 10234000, l40s = 18800640;   // (an expert blob: 256-aligned stride)
    CHECK(d::ring_slots(0, v100, 0, 4) == 4 * (int) (((size_t) 400 << 20) / v100 / 4));   // (not measured: 400 MB)
    CHECK(d::ring_slots(11.5 * GB, v100, 22 * GB, 4) == 60);   // (PCIe 3 x16: the 400 MB ring, as before)
    CHECK(d::ring_slots(1.63 * GB, cmp, 50 * GB, 4) == 4);     // (Gen2 x4: one a batch)
    CHECK(d::ring_slots(6.2 * GB, cmp, 50 * GB, 4) == 4);      // (Gen2 x16: 1.65 ms an expert - one a batch)
    CHECK(d::ring_slots(25 * GB, l40s, 30 * GB, 4) == 20);     // (PCIe 4 x16, big blobs: 400 MB, as before)
    CHECK(d::ring_slots(11.5 * GB, v100, 1 * GB, 4) == 8);     // (1 GB of room: 62 MB, 2 a batch)
    CHECK(d::ring_slots(11.5 * GB, v100, 0.01 * GB, 4) == 4);  // (at least one a batch)
    CHECK(d::ring_slots(11.5 * GB, 0, 0, 4) == 4);
    CHECK(d::ring_slots(8.0 * GB, v100, 50 * GB, 4) == 44);    // (0.86 ms an expert: ~10 ms a batch, 11)
    CHECK(d::ring_slots(6.0 * GB, v100, 50 * GB, 4) == 4);     // (1.15 ms an expert: one a batch)
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
