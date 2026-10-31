// The card whose run counts its own PCIe link bytes: a string rule, no GPU, no environment.
#include <cstdio>

#include "strata/core/link_meter.hpp"

using strata::core::link_meter_arch;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // gfx1100 - the RX 7900 XTX / XT / GRE, the W7900 / W7800 - counts, with its feature suffix included
    CHECK(link_meter_arch("gfx1100", nullptr));
    CHECK(link_meter_arch("gfx1100:sramecc-:xnack-", nullptr));
    // gfx1201 - the Radeon AI PRO R9700 / RX 9070 - counts as well: a discrete card whose experts cross the link
    CHECK(link_meter_arch("gfx1201", nullptr));
    CHECK(link_meter_arch("gfx1201:sramecc-:xnack-", nullptr));
    // other AMD parts do not: no card of ours to read the figure on, and an integrated Radeon would count blobs
    // that never cross a link (they stay in the host's memory)
    CHECK(!link_meter_arch("gfx1101", nullptr));   // RX 7800 XT
    CHECK(!link_meter_arch("gfx1151", nullptr));   // Strix Halo (APU)
    CHECK(!link_meter_arch("gfx1200", nullptr));   // RDNA4 as well, but no card of ours runs it
    CHECK(!link_meter_arch("gfx908", nullptr));
    // a CUDA / SYCL build has no gcnArchName: nothing is counted, and the engine's output is the last release's
    CHECK(!link_meter_arch("", nullptr));
    CHECK(!link_meter_arch(nullptr, nullptr));
    // STRATA_LINK_CHIPS=all is the owner's opt-in for any AMD card - and only for a gfx name
    CHECK(link_meter_arch("gfx1100", "all"));
    CHECK(link_meter_arch("gfx1201", "all"));
    CHECK(!link_meter_arch("", "all"));
    CHECK(!link_meter_arch(nullptr, "all"));
    CHECK(!link_meter_arch("sm_120", "all"));
    // an unset or other value keeps the two-architecture rule (STRATA_LINK_CHIPS=0 is not a way to widen it)
    CHECK(!link_meter_arch("gfx1200", "0"));
    CHECK(link_meter_arch("gfx1100", "0"));
    CHECK(link_meter_arch("gfx1201", "0"));
    if (failed == 0) std::printf("link_meter_test: ok\n");
    return failed ? 1 : 0;
}
