// batch_read_piece: the pieces of a --batch prompt read (synthetic: no model, no GPU).
#include "strata/program/batch_read.hpp"

#include <cstdio>

using strata::program::batch_read_piece;

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
    if (fails == 0) std::printf("batch_read_test: ok\n");
    return fails == 0 ? 0 : 1;
}
