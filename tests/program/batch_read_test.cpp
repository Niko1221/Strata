// batch_read.hpp: the pieces of a --batch prompt read, where the concurrent read runs; prompt_uploads.hpp: what its
// pinned slots are grown to before a read (synthetic: no model, no GPU).
#include "strata/prefill/prompt_uploads.hpp"
#include "strata/program/batch_read.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>

using strata::prefill::kPictureRunBytes;
using strata::prefill::picture_run_rows;
using strata::prefill::prompt_upload_bytes;
using strata::program::batch_read_piece;
using strata::program::concurrent_read_off;
using strata::program::ConcurrentReadSetup;
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
    // concurrent_read_off (STRATA_PIPED_CONCURRENT): runs with nothing set; each setting alone keeps it off, named
    CHECK(concurrent_read_off({}) == nullptr);
    bool ConcurrentReadSetup::*const each[] = {
        &ConcurrentReadSetup::prefill_help, &ConcurrentReadSetup::split_mtp_batch, &ConcurrentReadSetup::fs_slots,
        &ConcurrentReadSetup::pcie_dma,     &ConcurrentReadSetup::helper_caches,   &ConcurrentReadSetup::staged_source,
        &ConcurrentReadSetup::other_source};
    const char* named[std::size(each)] = {};
    auto same = [](const char* a, const char* b) { return a != nullptr && b != nullptr && std::strcmp(a, b) == 0; };
    for (size_t i = 0; i < std::size(each); ++i) {
        ConcurrentReadSetup s;
        s.*each[i] = true;
        named[i] = concurrent_read_off(s);
        CHECK(named[i] != nullptr);
        for (size_t j = 0; j < i; ++j) CHECK(!same(named[i], named[j]));   // a reason of its own
    }
    CHECK(same(named[0], "STRATA_PREFILL_HELP"));
    CHECK(same(named[3], "--pcie-mode dma"));
    // with several, the first in the struct's order is named
    for (size_t i = 0; i < std::size(each); ++i) {
        ConcurrentReadSetup s;
        for (size_t j = i; j < std::size(each); ++j) s.*each[j] = true;
        CHECK(same(concurrent_read_off(s), named[i]));
    }
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
    // prompt_upload_bytes: what the concurrent read grows the prompt path's pinned slots to before a read
    // (the model's K = 10 routed ids a token, rows of N = 2560 floats)
    const int64_t K = 10, N = 2560, row = N * 4;
    // text: the routing tables, or the PLE rows when they could not be pinned - the sizes before pictures were read so
    for (int64_t T : {1, 512, 4096, 16384}) {
        CHECK(prompt_upload_bytes(T, K, N, false, false) == T * K * 4);
        CHECK(prompt_upload_bytes(T, K, N, true, false) == T * N * 4);
    }
    // a picture's rows go up in runs of at most 4 MiB: 409 rows of 10,240 bytes
    CHECK(picture_run_rows(row) == 409);
    CHECK(prompt_upload_bytes(16384, K, N, false, true) == 409 * row);
    CHECK(prompt_upload_bytes(100, K, N, false, true) == 100 * row);       // a chunk holds at most T of them
    CHECK(prompt_upload_bytes(16384, K, N, true, true) == 16384 * row);    // the PLE rows are larger still
    CHECK(picture_run_rows(kPictureRunBytes + 1) == 1);                    // a row wider than a run: one at a time
    CHECK(picture_run_rows(0) == kPictureRunBytes);
    // the default path: the gather's run is the one it had before this header (prefill.cpp), for every row width
    // up to 8 MiB
    for (int64_t rb = 1; rb <= (int64_t(8) << 20); rb += rb < 65536 ? 1 : 4093)
        if (picture_run_rows(rb) != std::max<int64_t>(1, (4 << 20) / rb)) { CHECK(false); break; }
    // every upload the picture rows make fits the slots grown for them: the prompt path's gather (prefill.cpp),
    // over chunks with picture rows at different places
    for (int64_t T : {1, 7, 408, 409, 410, 1000, 4096, 16384}) {
        const int64_t reserved = prompt_upload_bytes(T, K, N, false, true), run_max = picture_run_rows(row);
        for (int layout = 0; layout < 4; ++layout) {
            auto pic = [&](int64_t t) {   // all picture rows; every other; a picture from the middle on; one 409-row run in
                return layout == 0 ? true : layout == 1 ? t % 2 == 0 : layout == 2 ? t >= T / 2 : (t >= 3 && t < 412);
            };
            int64_t largest = 0;
            for (int64_t t = 0; t < T;) {
                if (!pic(t)) { ++t; continue; }
                int64_t t1 = t + 1;
                while (t1 < T && t1 - t < run_max && pic(t1)) ++t1;
                largest = std::max(largest, (t1 - t) * row);
                t = t1;
            }
            CHECK(largest <= reserved);
        }
    }
    if (fails == 0) std::printf("batch_read_test: ok\n");
    return fails == 0 ? 0 : 1;
}
