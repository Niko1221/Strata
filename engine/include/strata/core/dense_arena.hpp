#pragma once
// STRATA_DENSE_ARENA=1 (opt-in): NativeDense::load places the native projection matrices in one device allocation
// instead of one cudaMalloc each.  The bytes are the same and every matrix starts on a 256 B boundary, as its own
// cudaMalloc would, so the kernels read the same values in the same order: the output is bit-identical.  What changes
// is the rounding: the driver backs each allocation with 2 MiB granules, so 300 matrices cost one partial granule each
// (Qwen3.8-Flash-Next IQ3_XXS: 1,781.11 MiB of weights priced at 1,968.12 MiB, one arena 1,782 MiB: 186.1 MiB less).
// Header-only so the CPU test (tests/core/dense_arena_test.cpp) checks the very functions `load` uses.

#include <cstdint>

namespace strata::core::dense_arena {

constexpr uint64_t kAlign = 256;              // cudaMalloc's own alignment
constexpr uint64_t kGranule = 2ull << 20;
constexpr uint64_t kSmall = 1ull << 20;

// What one cudaMalloc of `bytes` costs the card.  Measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and
// cost 2400, so a granule for every allocation of a MiB or more; below that the payload, because the driver packs the
// sub-MiB allocations into the slack of the rounded ones (iq3_s pack: 2255 MiB priced, 2238 taken, +0.8%).
constexpr uint64_t granule_bytes(uint64_t bytes) {
    return bytes < kSmall ? bytes : (bytes + kGranule - 1) / kGranule * kGranule;
}

// The offset of the next matrix of `bytes` in the arena; `used` becomes the arena's size so far.
constexpr uint64_t place(uint64_t& used, uint64_t bytes) {
    const uint64_t at = (used + kAlign - 1) / kAlign * kAlign;
    used = at + bytes;
    return at;
}

}  // namespace strata::core::dense_arena
