#include "strata/spec/dflash_policy.hpp"
#include <cstdio>

using strata::spec::DFlashPolicy;
static int fails;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++fails; } } while (0)

int main() {
    using strata::spec::dflash_attention_bucket;
    CHECK(dflash_attention_bucket(1, 8192) == 64);
    CHECK(dflash_attention_bucket(64, 8192) == 64);
    CHECK(dflash_attention_bucket(65, 8192) == 256);
    CHECK(dflash_attention_bucket(257, 8192) == 1024);
    CHECK(dflash_attention_bucket(1025, 8192) == 4096);
    CHECK(dflash_attention_bucket(4097, 8192) == 8192);
    CHECK(dflash_attention_bucket(90, 98) == 98);
    CHECK(dflash_attention_bucket(2052, 2052) == 2052);
    CHECK(dflash_attention_bucket(INT64_MAX, INT64_MAX) == INT64_MAX);
    CHECK(dflash_attention_bucket(0, 8192) == 0);
    CHECK(dflash_attention_bucket(8193, 8192) == 0);
    using strata::spec::dflash_proposal_fits;
    CHECK(dflash_proposal_fits(90, 7, false, 100, 97));
    CHECK(!dflash_proposal_fits(90, 7, true, 100, 97));
    CHECK(dflash_proposal_fits(90, 7, true, 100, 98));
    CHECK(!dflash_proposal_fits(93, 7, false, 100, 100));
    CHECK(!dflash_proposal_fits(-1, 3, true, 100, 100));
    CHECK(!dflash_proposal_fits(INT64_MAX, 7, true, INT64_MAX, INT64_MAX));
    CHECK(!dflash_proposal_fits(90, 0, true, 100, 100));
    for (int cap = 0; cap <= 7; ++cap) {
        DFlashPolicy p(cap);
        int counts[8] = {};
        for (int i = 0; i < 512; ++i) {
            const int k = p.choose();
            CHECK(k >= 0 && k <= cap);
            if (k < 0 || k > cap) break;
            ++counts[k];
            // Long blocks accept a little more, but K=2 gives the best throughput.
            p.observe(k, k == 0 ? 1 : k == 1 ? 2 : 3, k <= 2 ? 10 : 50);
        }
        const int expected = cap < 2 ? cap : 2;
        CHECK(counts[expected] > 400);
        for (int k = 0; k <= cap; ++k) CHECK(counts[k] >= 2);
    }
    {
        DFlashPolicy p(7);
        int plain = 0, resumed = 0;
        for (int i = 0; i < 512; ++i) {
            const int k = p.choose();
            plain += k == 0;
            p.observe(k, 1, k == 0 ? 10 : 40);   // drafts all rejected
        }
        CHECK(plain > 400);
        for (int i = 0; i < 2048; ++i) {
            const int k = p.choose();
            if (i >= 1536) resumed += k == 2;
            p.observe(k, k == 2 ? 3 : 1, k == 0 || k == 2 ? 10 : 40);
        }
        CHECK(resumed > 400);   // periodic probes discover useful drafting again
    }
    {
        DFlashPolicy p(7);
        int wide = 0;
        for (int i = 0; i < 2048; ++i) {
            int k = p.choose();
            if (i >= 1536) wide += k == 5;
            p.observe(k, k == 5 ? 6 : 1, 10);
        }
        CHECK(wide > 400);   // deferred probes must still discover better wide forwards
        p.reset();
        int plain = 0;
        for (int i = 0; i < 512; ++i) {
            int k = p.choose();
            plain += k == 0;
            p.observe(k, 1, k == 0 ? 10 : 40);
        }
        CHECK(plain > 400);   // a new prompt must not inherit stale acceptance/costs
    }
    std::printf("dflash_policy_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
