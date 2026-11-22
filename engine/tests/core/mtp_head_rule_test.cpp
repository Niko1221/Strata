// mtp_draft_head_row_cap (include/strata/core/mtp_head_rule.hpp): the draft head subset's row bytes (#5841).  No GPU.
// STRATA_MTP_HEAD_INDEX: mtp_draft_head_by_index, the one rule MtpDrafter::bind and bind_bytes apply, says yes only where bind
// reads the rows in the head; in every case where bind copies them (the fallbacks) bind_bytes prices the copy, so the expert
// cache is never sized without it (#199).
#include <cstdio>
#include <vector>

#include "strata/core/mtp_head_rule.hpp"

using strata::core::mtp_draft_head_by_index;
using strata::core::mtp_draft_head_row_cap;
using strata::core::mtp_draft_head_rows_bytes;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const int64_t n = 2560;
    const int64_t iq4_xs = n / 256 * 136;   // 1360
    const int64_t q4_0 = n / 32 * 18;       // 1440
    const int64_t q6_k = n / 256 * 210;     // 2100
    const int64_t q8_0 = n / 32 * 34;       // 2720
    CHECK(iq4_xs == 1360 && q4_0 == 1440);
    // #5841: an IQ4_XS head rewritten as Q4_0 needs the Q4_0 row: 106,299 rows x 80 bytes = 8.1 MiB more than the head's own
    CHECK(mtp_draft_head_row_cap(iq4_xs, n, true) == q4_0);
    CHECK(106299 * (mtp_draft_head_row_cap(iq4_xs, n, true) - iq4_xs) == 106299 * 80);
    // --mtp-q4 off: the head's own row, as before
    CHECK(mtp_draft_head_row_cap(iq4_xs, n, false) == iq4_xs);
    // formats at least as wide as Q4_0: unchanged
    CHECK(mtp_draft_head_row_cap(q6_k, n, true) == q6_k);
    CHECK(mtp_draft_head_row_cap(q8_0, n, true) == q8_0);
    CHECK(mtp_draft_head_row_cap(q4_0, n, true) == q4_0);
    // an n_embd that is no multiple of 32 has no Q4_0 row: the head's own
    CHECK(mtp_draft_head_row_cap(iq4_xs, 2564, true) == iq4_xs);

    // STRATA_MTP_HEAD_INDEX: the ids as draft_vocab.bin stores them (int32, little-endian), a vocabulary of 1000
    auto ids_of = [](std::vector<int32_t> v) {
        std::vector<uint8_t> b;
        for (int32_t x : v)
            for (int k = 0; k < 4; ++k) b.push_back((uint8_t) ((uint32_t) x >> (8 * k)));
        return b;
    };
    const std::vector<uint8_t> ids = ids_of({5, 6, 7, 999, 0, 42});
    const int64_t nv = 1000;
    const uint64_t rows = ids.size() / 4, q5_k = n / 256 * 176;
    // the case it is for: a Q5_K head in VRAM on the drafter's card (a recorded device or none), CUDA, no --mtp-q4 head
    CHECK(mtp_draft_head_by_index(true, false, false, 0, 0, true, ids.data(), ids.size(), nv));
    CHECK(mtp_draft_head_by_index(true, false, false, -1, 0, true, ids.data(), ids.size(), nv));
    CHECK(mtp_draft_head_rows_bytes(rows, q5_k, true) == 0);
    // every fallback: bind copies the rows, and the same rule makes bind_bytes count them
    struct Fallback { const char* what; bool env, q4, host; int head_dev; bool supported; std::vector<uint8_t> ids; };
    const Fallback fallbacks[] = {
        {"switch off", false, false, false, 0, true, ids},
        {"--mtp-q4 head|all", true, true, false, 0, true, ids},
        {"the head moved to the host (the spill walk)", true, false, true, 0, true, ids},
        {"the head on another card (mtp_first)", true, false, false, 1, true, ids},
        {"no native_mmvq_rows for the type (not Q5_K; every HIP build)", true, false, false, 0, false, ids},
        {"an id past the vocabulary", true, false, false, 0, true, ids_of({5, 1000})},
        {"a negative id", true, false, false, 0, true, ids_of({-1, 5})},
        {"no ids", true, false, false, 0, true, {}},
        {"a torn id file (not a multiple of 4 bytes)", true, false, false, 0, true, std::vector<uint8_t>{1, 0, 0}},
    };
    for (const Fallback& f : fallbacks) {
        const bool by_index =
            mtp_draft_head_by_index(f.env, f.q4, f.host, f.head_dev, 0, f.supported, f.ids.data(), f.ids.size(), nv);
        if (by_index || mtp_draft_head_rows_bytes(rows, q5_k, by_index) != rows * q5_k) {
            std::fprintf(stderr, "FAIL fallback \"%s\": read by index, or the copy not priced\n", f.what);
            ++failed;
        }
    }
    if (failed) { std::fprintf(stderr, "mtp_head_rule_test: %d failed\n", failed); return 1; }
    std::printf("mtp_head_rule_test: ok\n");
    return 0;
}
