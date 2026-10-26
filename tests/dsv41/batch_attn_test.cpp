// Strata. The M2 step table: several sequences' window and compressed-row regions in one buffer, checked without a GPU.
#include "ds4_batch_attn.hpp"

#include <cstdio>

using strata::ds4::batchattn::Layout;
using strata::ds4::batchattn::Step;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

int main() {
    Step s;
    s.n = 3;
    s.win = 128;
    s.ratio = 4;
    s.kcap = 512;
    s.seq[0] = Layout{100, 0, 0, 4096};
    s.seq[1] = Layout{2000, 128, 4096, 4096};
    s.seq[2] = Layout{5, 256, 8192, 4096};

    int win0[8] = {}, nwin[8] = {}, nsel[8] = {};
    CHECK(strata::ds4::batchattn::rows(s, win0, nwin, nsel, 8));
    // each sequence's own ring: its base, its valid rows (the third one has only 5 tokens so far)
    CHECK(win0[0] == 0 && win0[1] == 128 && win0[2] == 256);
    CHECK(nwin[0] == 100 && nwin[1] == 128 && nwin[2] == 5);
    // compressed rows: (pos + 1) / ratio, capped by kcap (none of these reach it)
    CHECK(nsel[0] == 25 && nsel[1] == 500 && nsel[2] == 1);
    CHECK(strata::ds4::batchattn::selectable(s, 0) == 25);

    // a step of three sequences is fine; one with a sequence that has no row yet is not
    CHECK(strata::ds4::batchattn::disjoint(s));
    Step z = s;
    z.seq[2].pos = 0;
    CHECK(!strata::ds4::batchattn::rows(z, win0, nwin, nsel, 8));

    // the indexer's picks land in the sequence's own rows
    std::vector<int> out;
    const int picks[3] = {4, 9, 17};
    CHECK(strata::ds4::batchattn::place(s, 1, picks, 3, out));
    CHECK(out.size() == 3 && out[0] == 4100 && out[2] == 4113);
    const int outside[1] = {4096};
    CHECK(!strata::ds4::batchattn::place(s, 1, outside, 1, out));      // its region is 4096 rows: 4096 is outside
    const int too_many[600] = {};
    CHECK(!strata::ds4::batchattn::place(s, 0, too_many, 600, out));   // 25 selectable, 600 asked

    // overlapping regions are refused: two sequences would read each other
    Step o = s;
    o.seq[1].win_base = 64;
    CHECK(!strata::ds4::batchattn::disjoint(o));
    Step o2 = s;
    o2.seq[2].kv_base = 4096;
    CHECK(!strata::ds4::batchattn::disjoint(o2));

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("batch_attn_test: ok\n");
    return 0;
}
