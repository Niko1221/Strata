// batch_read_piece: the pieces of a --batch prompt read (synthetic: no model, no GPU).
#include "strata/program/batch_read.hpp"

#include <cstdio>

using strata::program::batch_read_piece;
using strata::program::piped_decode_budget_ms;
using strata::program::piped_read_lead_in;
using strata::program::piped_read_piece;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // no --batch, or the pipelined groups: one run
    CHECK(batch_read_piece(false, false, false, false, 2048, 8) == 0);
    CHECK(batch_read_piece(true, true, true, false, 2048, 8) == 0);
    // one GPU: a chunk per piece, whether or not a slot decodes (the yield point is the chunk boundary)
    CHECK(batch_read_piece(true, false, false, false, 2048, 8) == 2048);
    CHECK(batch_read_piece(true, false, false, true, 2048, 8) == 2048);
    // layer split with slots decoding: a chunk per piece as before
    CHECK(batch_read_piece(true, false, true, true, 2048, 8) == 2048);
    // layer split, nobody decoding: pieces of 8 chunks - the BYIELD of a shorter request can be taken (it was one run)
    CHECK(batch_read_piece(true, false, true, false, 2048, 8) == 8 * 2048);
    CHECK(batch_read_piece(true, false, true, false, 1024, 4) == 4096);
    // STRATA_BATCH_YIELD_CHUNKS=0 keeps the old single run
    CHECK(batch_read_piece(true, false, true, false, 2048, 0) == 0);
    // a degenerate chunk never gives a zero piece
    CHECK(batch_read_piece(true, false, false, false, 0, 8) == 1);
    // a piece is always a multiple of the chunk
    for (int64_t c : {256, 1000, 2048, 4096})
        for (int64_t k : {1, 3, 8, 16}) CHECK(batch_read_piece(true, false, true, false, c, k) % c == 0);
    // piped_read_piece (#1906): off -> one run; whole chunks, at least one, at most 32
    CHECK(piped_read_piece(0, 8192) == 0);
    CHECK(piped_read_piece(-5, 8192) == 0);
    CHECK(piped_read_piece(32768, 8192) == 32768);
    CHECK(piped_read_piece(20000, 8192) == 16384);      // rounded down to whole chunks
    CHECK(piped_read_piece(4096, 8192) == 8192);        // never below one chunk
    CHECK(piped_read_piece(1 << 30, 8192) == 32 * 8192);
    CHECK(piped_read_piece(5000, 0) == 32);             // chunk 0 guarded as 1 (and the cap applies)
    for (int64_t c : {1024, 2048, 6144, 8192})
        for (int64_t p : {1, 1000, 9000, 50000, 300000}) CHECK(piped_read_piece(p, c) % c == 0);   // the header's rule
    // piped_decode_budget_ms: share x piece, capped, floored; negatives are 0
    CHECK(piped_decode_budget_ms(2000.0, 2.0, 400.0, 0.0) == 400.0);
    CHECK(piped_decode_budget_ms(2000.0, 0.35, 3000.0, 0.0) == 700.0);
    CHECK(piped_decode_budget_ms(2000.0, 0.0, 400.0, 150.0) == 150.0);
    CHECK(piped_decode_budget_ms(2000.0, 2.0, 0.0, 0.0) == 4000.0);   // no cap
    CHECK(piped_decode_budget_ms(-5.0, -1.0, 400.0, -3.0) == 0.0);
    // piped_read_lead_in: on, beside decoding slots, within 1000 ms + the budget of the previous piecewise read's end
    CHECK(piped_read_lead_in(true, true, 0.0, 400.0));
    CHECK(piped_read_lead_in(true, true, 1400.0, 400.0));
    CHECK(!piped_read_lead_in(true, true, 1401.0, 400.0));
    CHECK(!piped_read_lead_in(true, true, -1.0, 400.0));
    CHECK(!piped_read_lead_in(false, true, 0.0, 400.0));
    CHECK(!piped_read_lead_in(true, false, 0.0, 400.0));
    if (fails == 0) std::printf("batch_read_test: ok\n");
    return fails == 0 ? 0 : 1;
}
