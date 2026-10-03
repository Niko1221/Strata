// expert_pack.hpp - which expert blobs a native-pack cache slot holds, and in what order.
//
// **WHY THIS IS A PURE FUNCTION.** A profile ranks every (layer, expert) pair; a cache slot holds one pair's
// blob, and blobs differ per layer, so "N slots" packs more than a uniform N * max_blob budget would - that is
// plan v0.3 P6.  The packer used to run inline in generate and take `min(budget, free VRAM)` as its cap for BOTH
// the auto and an explicit `--expert-cache N`, which made the resident set depend on how much VRAM happened to
// be free at the moment the engine started: two engines launched minutes apart pinned to the same 5479/5531/
// 5549 slots built 5616 vs 5621 vs 5666 (bench/results/prefill-preempt, 2026-10-03), the resident experts rounded
// differently, and an interim request's greedy decode diverged into an early EOS (repeat-3, 5 vs 128 tokens).
// A resident set that a parity comparison depends on must be a pure function of the arguments, so the packing
// lives here where a CPU test can pin it down.
//
// Explicit N now means exactly the first N ranked pairs (a pure function of N, and what the flag's help always
// said: "keep N expert blobs resident").  If their bytes do not fit what VRAM is free the allocation says so
// plainly instead of silently delivering a different resident set.  `auto` keeps its free-VRAM sizing: that is
// the point of auto.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

/// The byte sizes of the cache slots, in the profile's rank order: the ranked pairs in order while their 256-byte
/// rounded blob sizes sum to at most `cap_bytes` and at most `max_count` of them are taken.  `cap_bytes` = UINT64_MAX
/// with a finite `max_count` takes exactly the first `max_count` pairs; a finite cap with SIZE_MAX count is the
/// auto sizing.  Never throws, never reads anything but the vector it is given.
inline std::vector<int64_t> pack_expert_slot_bytes(const std::vector<int64_t>& ranked_pair_bytes,
                                                   uint64_t cap_bytes, size_t max_count) {
    std::vector<int64_t> slots;
    uint64_t used = 0;
    for (size_t i = 0; i < ranked_pair_bytes.size() && slots.size() < max_count; ++i) {
        if (ranked_pair_bytes[i] < 0) break;
        const uint64_t b = ((uint64_t) ranked_pair_bytes[i] + 255) / 256 * 256;
        if (b > cap_bytes - used) break;    // subtraction cannot wrap: b > cap implies b > cap - used too
        used += b;
        slots.push_back(ranked_pair_bytes[i]);
    }
    return slots;
}

}  // namespace strata::core
