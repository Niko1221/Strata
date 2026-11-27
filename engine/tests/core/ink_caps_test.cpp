// Per-device capability answers (strata/core/ink_caps.hpp, #6796 #6808): on a mixed split the second card must get its own answer, not
// the first card's. No second card needed: two capability sets are fed through the same cache the CUDA code uses.
#include "strata/core/ink_caps.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

struct Cc { int major, minor; };

// what ink_has_int8_tc / ink_has_wmma_f16 / sm120::available do for the card `cc`, with a counter of how often the card is asked
struct Probe {
    PerDeviceFlag int8, wmma, fp4;
    int asked = 0;
    bool has_int8(int dev, Cc cc) { return int8.get(dev, [&] { ++asked; return ink_cap_int8_tc(cc.major); }); }
    bool has_wmma(int dev, Cc cc) { return wmma.get(dev, [&] { ++asked; return ink_cap_wmma_f16(cc.major); }); }
    bool has_fp4(int dev, Cc cc) { return fp4.get(dev, [&] { ++asked; return ink_cap_sm120(cc.major, cc.minor); }); }
};

// the old behaviour: one answer for the whole process
struct OldProbe {
    int cached = -1;
    bool has_fp4(Cc cc) { if (cached < 0) cached = ink_cap_sm120(cc.major, cc.minor) ? 1 : 0; return cached != 0; }
};
}  // namespace

int main() {
    const Cc sm120{12, 0}, sm86{8, 6}, sm60{6, 0}, sm70{7, 0}, sm75{7, 5};

    // rules
    CHECK(ink_cap_int8_tc(8) && ink_cap_int8_tc(12) && !ink_cap_int8_tc(7) && !ink_cap_int8_tc(6));
    CHECK(ink_cap_wmma_f16(7) && !ink_cap_wmma_f16(6) && !ink_cap_wmma_f16(8));
    CHECK(ink_cap_sm120(12, 0) && ink_cap_sm120(12, 1) && !ink_cap_sm120(12, 2) && !ink_cap_sm120(8, 6));

    // #6808: sm_120 first, sm_86 second
    {
        Probe p;
        CHECK(p.has_fp4(0, sm120));
        CHECK(!p.has_fp4(1, sm86));          // the old cache said true here: the sm_86 stage ran the sm_120 FP4 kernels
        CHECK(p.has_fp4(0, sm120));          // stage 0 keeps its answer
        CHECK(p.has_int8(0, sm120) && p.has_int8(1, sm86));
        OldProbe o;
        const bool first = o.has_fp4(sm120), second = o.has_fp4(sm86);
        CHECK(first && second);              // documents the bug: the second card got the first card's answer
    }
    // #6796: sm_86 first, sm_60 second (and the other order)
    {
        Probe p;
        CHECK(p.has_int8(0, sm86));
        CHECK(!p.has_int8(1, sm60));         // the old cache said true: the P100 got the INT8 tensor-core kernels
        CHECK(!p.has_wmma(0, sm86) && !p.has_wmma(1, sm60));
        Probe q;
        CHECK(!q.has_int8(0, sm60));
        CHECK(q.has_int8(1, sm86));          // and the 3080 Ti must not inherit "no" from the P100 either
    }
    // Volta + Turing + Ampere: WMMA for the first two only
    {
        Probe p;
        CHECK(p.has_wmma(0, sm70) && p.has_wmma(1, sm75) && !p.has_wmma(2, sm86));
        CHECK(!p.has_int8(0, sm70) && !p.has_int8(1, sm75) && p.has_int8(2, sm86));
    }
    // each device is asked once; an unreadable device (-1) answers false and is not remembered
    {
        Probe p;
        p.has_int8(0, sm86); p.has_int8(0, sm86); p.has_int8(0, sm86);
        CHECK(p.asked == 1);
        p.has_int8(1, sm60); p.has_int8(1, sm60);
        CHECK(p.asked == 2);
        CHECK(!p.has_int8(-1, sm86));
        CHECK(p.asked == 2);
        CHECK(p.has_int8(kInkCapDevices + 5, sm86));   // an index past the table shares the last slot instead of overflowing
    }
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("ink_caps_test: ok");
    return 0;
}
