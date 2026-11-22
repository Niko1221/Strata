// tests/core/mmvq_rows_test.cpp - STRATA_MTP_HEAD_INDEX=1: the draft head read by index from the main head takes, for
// every output row and every reduction block, the very bytes the plain read takes from gather_rows' copy of the subset.
// The address is strata::kernels::mmvq_block, the function native_q5_k_mmvq_kernel itself calls; the kernel's arithmetic
// does not depend on the address, so equal operands mean equal logits. No model, no GPU.
//
// A tiny fake head: 97 vocabulary rows of 10 Q5_K blocks (2,560 values, 1,760 B a row, as Qwen3.8-Flash-Next's), random
// bytes; a subset of 40 ids in unsorted runs like draft_vocab.bin's; the copy built as gather_rows builds it (row i of
// the copy = row ids[i] of the head).  A wrong address (ids off by one) must be caught.
#include "strata/kernels/mmvq_rows.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using strata::kernels::mmvq_block;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-88s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int kBlockBytes = 176;     // one Q5_K block (256 values)
constexpr int kBlocksPerRow = 10;    // n_in 2560 / 256
constexpr int kRowBytes = kBlockBytes * kBlocksPerRow;
constexpr int kVocab = 97;

// how many (row, block) reads of the indexed head differ from the plain read of the copy
int mismatches(const std::vector<uint8_t>& head, const std::vector<int32_t>& ids, const std::vector<uint8_t>& copy) {
    int bad = 0;
    for (int r = 0; r < (int) ids.size(); ++r)
        for (int kbx = 0; kbx < kBlocksPerRow; ++kbx) {
            const uint8_t* indexed = head.data() + mmvq_block<true>(ids.data(), r, kBlocksPerRow, kbx) * kBlockBytes;
            const uint8_t* plain = copy.data() + mmvq_block<false>(nullptr, r, kBlocksPerRow, kbx) * kBlockBytes;
            if (std::memcmp(indexed, plain, kBlockBytes) != 0) ++bad;
        }
    return bad;
}
}  // namespace

int main() {
    std::vector<uint8_t> head((size_t) kVocab * kRowBytes);
    uint32_t s = 12345u;
    for (uint8_t& b : head) { s = s * 1664525u + 1013904223u; b = (uint8_t) (s >> 24); }
    std::vector<int32_t> ids;
    for (int32_t i = 50; i < 57; ++i) ids.push_back(i);   // runs, not sorted
    for (int32_t i = 3; i < 5; ++i) ids.push_back(i);
    ids.push_back(96);
    ids.push_back(0);
    for (int32_t i = 10; i < 29; ++i) ids.push_back(i);
    for (int32_t i = 77; i < 86; ++i) ids.push_back(i);
    ids.push_back(5);
    std::vector<uint8_t> copy(ids.size() * (size_t) kRowBytes);   // gather_rows: row i of the copy = row ids[i]
    for (size_t i = 0; i < ids.size(); ++i)
        std::memcpy(copy.data() + i * kRowBytes, head.data() + (size_t) ids[i] * kRowBytes, kRowBytes);

    std::printf("1. the indexed read against the copy (%zu rows x %d blocks)\n", ids.size(), kBlocksPerRow);
    check(ids.size() == 40, "40 ids in 7 runs, not sorted");
    check(mismatches(head, ids, copy) == 0, "every block the indexed read takes is the copy's block for that row");
    std::vector<int32_t> wrong = ids;
    for (int32_t& id : wrong) id = (id + 1) % kVocab;
    check(mismatches(head, wrong, copy) == (int) ids.size() * kBlocksPerRow, "ids off by one: every block differs (the check bites)");

    std::printf("2. the plain address is the kernel's former expression\n");
    bool same = true;
    for (int r : {0, 1, 106298, 248319})
        for (int kbx : {0, 9})
            same = same && mmvq_block<false>(nullptr, r, kBlocksPerRow, kbx) == std::size_t(r) * kBlocksPerRow + kbx;
    check(same, "mmvq_block<false>(r, kbx) == size_t(r) * blocks_per_row + kbx, up to the head's last row");

    std::printf(g_fail ? "mmvq_rows_test: %d FAILED\n" : "mmvq_rows_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
