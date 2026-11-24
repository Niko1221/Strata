// Check Engram row hashing, history padding and embedding masks.
#include "strata/hy4flash_gguf/engram.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// hy4-flash's shape, with made-up constants: enough to exercise the hash's structure without a model.
strata::hy4_flash::EngramHashParams synthetic() {
    strata::hy4_flash::EngramHashParams p;
    p.ngram_size = 3;
    p.heads_per_ngram = 8;
    p.columns = 16;
    p.pad_token = 5;
    p.multipliers = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull};
    for (int c = 0; c < 16; ++c) {
        p.primes.push_back(1000003ull + (uint64_t) c * 17ull);
        p.offsets.push_back((uint64_t) c * 2000000ull);
    }
    return p;
}

void structure() {
    strata::hy4_flash::EngramHashParams p = synthetic();
    std::string err;
    check(p.check(err), "a complete parameter set is accepted" + (err.empty() ? "" : ": " + err));

    strata::hy4_flash::EngramHashParams bad = p;
    bad.multipliers.pop_back();
    check(!bad.check(err), "one multiplier short is refused (they come from the file, not from a formula)");
    bad = p;
    bad.primes[3] = 0;
    check(!bad.check(err), "a zero bucket count is refused rather than dividing by zero");
    bad = p;
    bad.columns = 8;
    check(!bad.check(err), "columns that disagree with (ngram - 1) * heads_per_ngram are refused");
    bad = p;
    bad.pad_token = -1;
    check(!bad.check(err), "no pad token is refused: a short history is padded, not dropped");

    // Four tokens, positions 0..3, two predecessors each (oldest first).
    const int32_t toks[4] = {11, 22, 33, 44};
    const int32_t pos[4] = {0, 1, 2, 3};
    const int32_t prev[8] = {0, 0, /*t1*/ 0, 11, /*t2*/ 11, 22, /*t3*/ 22, 33};
    std::vector<int64_t> ids;
    std::vector<float> valid;
    // err still holds the message from the last negative case above; clear it so a success does not report
    // someone else's error text.
    err.clear();
    const bool ran = strata::hy4_flash::engram_row_ids(p, toks, pos, prev, nullptr, 4, ids, valid, err);
    check(ran, ran ? "the hash runs" : "the hash runs: " + err);
    check(ids.size() == 64 && valid.size() == 64, "16 columns for each of 4 tokens");

    // Every id must land inside its own column's bucket range - that is what `offsets` is for, and an id
    // outside it would read another column's rows.
    bool in_range = true;
    for (int t = 0; t < 4; ++t)
        for (int c = 0; c < 16; ++c) {
            const int64_t v = ids[(size_t) (t * 16 + c)];
            in_range &= v >= (int64_t) p.offsets[(size_t) c] &&
                        v < (int64_t) (p.offsets[(size_t) c] + p.primes[(size_t) c]);
        }
    check(in_range, "every id lies in its own column's [offset, offset + buckets) window");

    // The eight heads of one order share a rolling value, so within an order the ids differ ONLY through the
    // prime and the offset. Subtracting the offset must leave a value below that column's prime - already
    // checked - and the two orders must generally differ from each other.
    bool orders_differ = false;
    for (int t = 0; t < 4; ++t)
        for (int h = 0; h < 8; ++h)
            if (ids[(size_t) (t * 16 + h)] - (int64_t) p.offsets[(size_t) h] !=
                ids[(size_t) (t * 16 + 8 + h)] - (int64_t) p.offsets[(size_t) (8 + h)])
                orders_differ = true;
    check(orders_differ, "the 2-gram and 3-gram halves hash differently (the XOR actually accumulated)");

    // A position with no n-gram marks every column invalid.
    const uint8_t embd[4] = {0, 1, 0, 0};
    check(strata::hy4_flash::engram_row_ids(p, toks, pos, prev, embd, 4, ids, valid, err), "with an embd flag");
    bool masked = true;
    for (int c = 0; c < 16; ++c) masked &= valid[(size_t) (1 * 16 + c)] == 0.f;
    for (int c = 0; c < 16; ++c) masked &= valid[(size_t) (0 * 16 + c)] == 1.f;
    check(masked, "an embedding position is invalid in every column, its neighbours are not");

    // The pad path: token 0's two predecessors do not exist (pos 0), so both are the pad. Hashing a token at
    // pos 0 must still produce an id - the row is read, not skipped.
    check(valid[0] == 1.f && ids[0] >= (int64_t) p.offsets[0],
          "a token at position 0 still gets a row: its missing history is padded");

    // Unsigned wrap: a token whose id times the multiplier overflows must not produce a negative row. This is
    // the one place a signed type would silently change the hash.
    strata::hy4_flash::EngramHashParams big = p;
    const int32_t huge[1] = {2000000000};
    const int32_t zero_pos[1] = {0};
    const int32_t no_prev[2] = {0, 0};
    std::vector<int64_t> bids;
    std::vector<float> bvalid;
    check(strata::hy4_flash::engram_row_ids(big, huge, zero_pos, no_prev, nullptr, 1, bids, bvalid, err),
          "a token whose product overflows 64 bits still hashes");
    bool nonneg = true;
    for (int64_t v : bids) nonneg &= v >= 0;
    check(nonneg, "and every id is non-negative (the rolling value is unsigned)");
}

}  // namespace

int main() {
    std::printf("the hash's structure, on synthetic constants\n");
    structure();

    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
