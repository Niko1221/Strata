// split_relay_rule.hpp: the switch, the share and the per-chunk rule of the staged walk's relay.  No GPU, no model.
#include <cstdint>
#include <cstdio>

#include "strata/prefill/split_relay_rule.hpp"

using strata::prefill::RelayShare;
using strata::prefill::split_relay_chunk;
using strata::prefill::split_relay_frac;
using strata::prefill::split_relay_on;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: off unless a number other than 0
    CHECK(!split_relay_on(nullptr));
    CHECK(!split_relay_on(""));
    CHECK(!split_relay_on("0"));
    CHECK(!split_relay_on("off"));
    CHECK(split_relay_on("1"));
    CHECK(split_relay_on("2"));
    // the share: 0.5 unless a number, clamped
    CHECK(split_relay_frac(nullptr) == 0.5);
    CHECK(split_relay_frac("") == 0.5);
    CHECK(split_relay_frac("x") == 0.5);
    CHECK(split_relay_frac("0.4") == 0.4);
    CHECK(split_relay_frac("0") == 0.0);
    CHECK(split_relay_frac("-1") == 0.0);
    CHECK(split_relay_frac("3") == 1.0);
    // the chunk: only an armed stage's one-chunk, unhelped, staged read
    CHECK(split_relay_chunk(true, true, false, false));
    CHECK(!split_relay_chunk(false, true, false, false));
    CHECK(!split_relay_chunk(true, false, false, false));   // a chunk of many: the other stage may be reading
    CHECK(!split_relay_chunk(true, true, true, false));     // the help already uses the other card's link
    CHECK(!split_relay_chunk(true, true, false, true));     // the streamed walk
    // byte-balancing: equal blobs at 0.5 alternate, the first direct
    {
        RelayShare s{0.5};
        const bool want[6] = {false, true, false, true, false, true};
        for (bool w : want) CHECK(s.take(100) == w);
        CHECK(s.relayed == 300 && s.total == 600);
    }
    // the ends: 0 relays nothing, 1 relays everything
    {
        RelayShare none{0.0}, all{1.0};
        for (int i = 0; i < 5; ++i) {
            CHECK(!none.take(100));
            CHECK(all.take(100));
        }
        CHECK(none.relayed == 0 && all.relayed == 500);
    }
    // mixed sizes: the relayed bytes never pass the share
    {
        RelayShare s{0.4};
        const uint64_t sizes[8] = {300, 100, 700, 200, 500, 100, 900, 400};
        for (uint64_t b : sizes) {
            s.take(b);
            CHECK((double) s.relayed <= 0.4 * (double) s.total);
        }
        CHECK(s.relayed > 0);
    }
    if (failed) std::fprintf(stderr, "%d check(s) failed\n", failed);
    else std::puts("split_relay_rule_test: OK");
    return failed ? 1 : 0;
}
