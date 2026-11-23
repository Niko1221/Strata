// The prefill chunk strata-inkling picks without --chunk (strata/core/ink_chunk.hpp): the memory left after the experts decides, the slot count caps it.
#include "strata/core/ink_chunk.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    // measured: RTX 3060 Ti 8 GB, SQ-10, 2955 MiB free after the experts: 512 (chunk 1024 would take 1804 MiB and leave 1151 MiB, below the margin)
    CHECK(ink_auto_chunk(2955.0, 1) == 512);
    // the ledger of an RTX 4090 (SQ-20, all resident, 5255 MiB free after the experts): runner + K/V took 1460 / 1822 / 2710 / 4558 MiB at 512 / 1024 / 2048 / 4096
    CHECK(ink_chunk_need_mib(4096) - ink_chunk_need_mib(512) > (4558.0 - 1460.0) * 0.97);
    CHECK(ink_chunk_need_mib(4096) - ink_chunk_need_mib(512) < (4558.0 - 1460.0) * 1.06);
    CHECK(ink_chunk_need_mib(2048) - ink_chunk_need_mib(512) > (2710.0 - 1460.0) * 0.97);
    CHECK(ink_chunk_need_mib(1024) - ink_chunk_need_mib(512) > (1822.0 - 1460.0));
    CHECK(ink_auto_chunk(5255.0, 1) == 2048);         // that 4090: 2048 leaves 2545 MiB
    // a card that kept room: bigger chunks, one step at a time; never above 2048 by itself (4096 is for --chunk)
    CHECK(ink_auto_chunk(3500.0, 1) == 1024);
    CHECK(ink_auto_chunk(4400.0, 1) == 2048);
    CHECK(ink_auto_chunk(6200.0, 1) == 2048);
    CHECK(ink_auto_chunk(12000.0, 1) == 2048);        // an RTX 5090 with every expert resident
    CHECK(ink_auto_chunk(1.0e9, 1) == 2048);
    // never below the old default, never from an unknown amount
    CHECK(ink_auto_chunk(0.0, 1) == 512);
    CHECK(ink_auto_chunk(-1.0, 1) == 512);
    // slots cap the chunk: a decoding slot waits for the prefill tokens of the others in the same step (8 GB card: 43 s stall at 2048 with 32 slots)
    CHECK(ink_auto_chunk(12000.0, 2) == 2048);
    CHECK(ink_auto_chunk(12000.0, 3) == 2048);
    CHECK(ink_auto_chunk(12000.0, 4) == 1024);
    CHECK(ink_auto_chunk(12000.0, 8) == 1024);
    CHECK(ink_auto_chunk(12000.0, 15) == 1024);
    CHECK(ink_auto_chunk(12000.0, 16) == 512);        // heavy parallel keeps the old chunk
    CHECK(ink_auto_chunk(12000.0, 32) == 512);
    CHECK(ink_auto_chunk(12000.0, 64) == 512);
    CHECK(ink_auto_chunk(3500.0, 8) == 1024);
    CHECK(ink_auto_chunk(2000.0, 8) == 512);
    // the K/V that still has to grow stays free of the chunk (RTX 4090, UD-Q4_K_XL, 5255 MiB free after the experts, 32K context, fp16: 896 MiB per slot to grow):
    // 1 slot keeps 2048 (decode at 30K tokens 87.7 tok/s), 2 slots 1024, 3 slots 512 (with the K/V on the card 87; spilled to host 36 and 14.7 tok/s)
    CHECK(ink_auto_chunk(5255.0, 1, 896.0) == 2048);
    CHECK(ink_auto_chunk(5255.0, 2, 1792.0) == 1024);
    CHECK(ink_auto_chunk(5255.0, 3, 2688.0) == 512);
    CHECK(ink_auto_chunk(12000.0, 3, 2688.0) == 2048);        // a card with room for both
    CHECK(ink_chunk_need_mib(2048, 1000.0) == ink_chunk_need_mib(2048) + 1000.0);
    CHECK(ink_chunk_need_mib(512, -5.0) == ink_chunk_need_mib(512));
    // monotone: more free memory never gives a smaller chunk
    int last = 0;
    for (double f = 0.0; f <= 20000.0; f += 50.0) {
        const int c = ink_auto_chunk(f, 1);
        CHECK(c >= last);
        last = c;
    }
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("ink_chunk_test: ok");
    return 0;
}
