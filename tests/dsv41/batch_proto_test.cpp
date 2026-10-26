// Strata. The engine's batch lines (docs/DSV41_BATCHING.md, M4): BGEN/BGENI split, and when an admission continues
// in its slot.  No GPU, no model - the serve loop's own use of these is checked in a model window.
#include "ds4_batch_proto.hpp"

#include <cstdio>

using strata::ds4::batchproto::Admission;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

int main() {
    {   // BGEN: the slot, the max_new, and the GEN 1 line the ordinary path reads
        Admission a;
        CHECK(strata::ds4::batchproto::split("BGEN 0 40 temperature=0.7 top_p=1 pin=2048 1,2,3", a));
        CHECK(a.slot == 0 && a.max_new == 40);
        CHECK(a.request == "GEN 1 temperature=0.7 top_p=1 pin=2048 1,2,3");
    }
    {   // BGENI: the pictures' file stays where the GENI path expects it
        Admission a;
        CHECK(strata::ds4::batchproto::split("BGENI 1 12 temperature=0 pics.dsi1 7,8,9", a));
        CHECK(a.slot == 1 && a.max_new == 12);
        CHECK(a.request == "GENI 1 temperature=0 pics.dsi1 7,8,9");
    }
    {   // a second slot, no keys, one id
        Admission a;
        CHECK(strata::ds4::batchproto::split("BGEN 3 2 5", a));
        CHECK(a.slot == 3 && a.max_new == 2 && a.request == "GEN 1 5");
    }
    {   // not a batch line
        Admission a;
        CHECK(!strata::ds4::batchproto::split("GEN 4 1,2", a));
        CHECK(!strata::ds4::batchproto::split("GENI 4 p.dsi1 1,2", a));
        CHECK(!strata::ds4::batchproto::split("STOP", a));
        CHECK(!strata::ds4::batchproto::split("BSTOP 0", a));
        CHECK(a.slot == -1 && a.max_new == 0);
    }
    {   // malformed: no slot, no max_new
        Admission a;
        CHECK(!strata::ds4::batchproto::split("BGEN ", a));
        CHECK(!strata::ds4::batchproto::split("BGEN 0", a));
        CHECK(!strata::ds4::batchproto::split("BGEN 0 x 1,2", a));
        CHECK(!strata::ds4::batchproto::split("BGENI 2", a));
    }
    {   // the slot decodes the rest: one token admitted, more asked for, and the model stopped at length
        CHECK(strata::ds4::batchproto::continues(1, 40, "length"));
        CHECK(strata::ds4::batchproto::continues(1, 2, "length"));
        // not: the answer ended at its first token, one token was all that was asked, or the read was cancelled
        CHECK(!strata::ds4::batchproto::continues(1, 1, "length"));
        CHECK(!strata::ds4::batchproto::continues(1, 40, "stop"));
        CHECK(!strata::ds4::batchproto::continues(1, 40, "cancel"));
        CHECK(!strata::ds4::batchproto::continues(0, 40, "length"));
        CHECK(!strata::ds4::batchproto::continues(2, 40, "length"));
        CHECK(!strata::ds4::batchproto::continues(1, 40, nullptr));
    }
    if (fails != 0) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("batch_proto_test: ok\n");
    return 0;
}
