// tests/core/dense_arena_test.cpp - STRATA_DENSE_ARENA=1: where NativeDense::load puts each native matrix in its one
// allocation, and what that saves against one cudaMalloc each (strata/core/dense_arena.hpp, the functions load and
// weight_bytes_for use). No model, no GPU.
//
// 1. Every matrix starts on a 256 B boundary (cudaMalloc's alignment, so the kernels see the alignment they always
//    had), right after the one before it, and none overlaps the next.
// 2. The granule rule: below 1 MiB the payload, from 1 MiB up whole 2 MiB granules.
// 3. Qwen3.8-Flash-Next IQ3_XXS's 300 matrices (sizes from the GGUF's tensor directory, the eligible names and types
//    load takes): 1,781.11 MiB of weights, 1,968.12 MiB one cudaMalloc each, 1,782 MiB as one arena: 186.12 MiB less.
#include "strata/core/dense_arena.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace a = strata::core::dense_arena;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-88s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("1. offsets\n");
    {
        const std::vector<uint64_t> sizes = {1, 255, 256, 257, 1088, 3, 1u << 20, 17};
        uint64_t used = 0, end = 0;
        bool aligned = true, after = true, tight = true;
        for (uint64_t b : sizes) {
            const uint64_t at = a::place(used, b);
            aligned = aligned && at % 256 == 0;
            after = after && at >= end;                       // no overlap with the matrix before
            tight = tight && at - end < 256;                  // and no more than the alignment between them
            end = at + b;
            if (used != end) after = false;
        }
        check(aligned, "every matrix starts on a 256 B boundary");
        check(after, "each starts at or after the end of the one before; used is the last end");
        check(tight, "the gap before each is under 256 B");
        uint64_t u2 = 0;
        const uint64_t o0 = a::place(u2, 256), o1 = a::place(u2, 512), o2 = a::place(u2, 1);
        check(o0 == 0 && o1 == 256 && o2 == 768 && u2 == 769, "multiples of 256 are packed with no gap (0, 256, 768; 769 used)");
    }

    std::printf("2. one cudaMalloc's price\n");
    const uint64_t MiB = 1ull << 20;
    check(a::granule_bytes(0) == 0 && a::granule_bytes(MiB - 1) == MiB - 1, "under 1 MiB: the payload");
    check(a::granule_bytes(MiB) == 2 * MiB && a::granule_bytes(2 * MiB) == 2 * MiB && a::granule_bytes(2 * MiB + 1) == 4 * MiB,
          "1 MiB and up: whole 2 MiB granules");

    std::printf("3. Qwen3.8-Flash-Next IQ3_XXS: 300 matrices\n");
    struct Size { uint64_t bytes; int count; };
    static const Size model[] = {
        {460800, 8}, {563200, 1}, {696320, 4}, {704000, 40}, {737280, 3}, {870400, 31}, {901120, 1}, {921600, 59},
        {1075200, 15}, {1126400, 2}, {1344000, 4}, {6758400, 13}, {8355840, 17}, {8847360, 12}, {10813440, 8},
        {12902400, 34}, {13516800, 1}, {13926400, 11}, {14745600, 8}, {16711680, 4}, {17694720, 3}, {18022400, 3},
        {21504000, 14}, {25804800, 4}};
    uint64_t payload = 0, each = 0, used = 0;
    int n = 0;
    for (const Size& s : model)
        for (int i = 0; i < s.count; ++i, ++n) {
            payload += s.bytes;
            each += a::granule_bytes(s.bytes);
            a::place(used, s.bytes);
        }
    const uint64_t arena = a::granule_bytes(used);
    std::printf("  %d matrices, %.2f MiB of weights; one cudaMalloc each %.2f MiB, one arena %.2f MiB: %.2f MiB less\n", n,
                payload / (double) MiB, each / (double) MiB, arena / (double) MiB, (each - arena) / (double) MiB);
    check(n == 300 && payload == 1867632640ull, "300 matrices, 1,867,632,640 B (the start-up log's 1781.11 MiB)");
    check(each == 2063724544ull, "one cudaMalloc each: 2,063,724,544 B (1,968.12 MiB)");
    check(used == payload, "every size is a multiple of 256 B, so the arena holds the payload with no padding");
    check(arena == 1782 * MiB, "the arena: 1,782 MiB");
    check(each - arena == 195162112ull, "saved: 195,162,112 B (186.12 MiB)");

    std::printf(g_fail ? "dense_arena_test: %d FAILED\n" : "dense_arena_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
