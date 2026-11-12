// src/dsv41/ds4_ring_link.hpp - the prompt path's staging ring sized by the GPU's measured host link.
//
// From Project Maya #609 (mw00, 2e0d2f9 and 4b76907; MIT, the notice in third_party/project-maya): the prompt path streams a
// layer's CPU-lane experts through a staging ring in `parts` batches, so a batch is what a layer waits for before it computes on
// it.  A batch of about kRingBatchS of this device's measured link (link: bytes a second, 0 = not measured) - a slow link waits
// less for each, a fast one keeps the batch it had - at least one slot, never more than kRingMaxMB (the fixed ring before) and
// never more than 1/16 of the device's VRAM room for experts (room: bytes, 0 = not known).  A link that takes kRingSlowS or more
// for one expert takes one a batch.  Measured by Maya: 2x CMP 170HX on PCIe Gen2 x4 / x16 read prompts fastest at 1-3 slots a
// batch (one a batch on the x16 card: 2-3.5% faster than 6 of ~10 ms), an L40S at 5 of 18.8 MB (the most tried), a V100 on PCIe 3
// x16 (6.9 MB experts) at 15, the 400 MB ring - smaller rings were slower there.
//
// In Strata this is opt-in (DS4_RING_MB=link): the ring stays what ds4_experts_rule.hpp's staging_ring_mb picks unless asked.
#pragma once

#include <algorithm>
#include <cstddef>

namespace strata::ds4 {

constexpr double kRingBatchS = 0.010;
constexpr double kRingSlowS = 0.001;
constexpr int kRingMaxMB = 400;

// -> the ring's slots (a multiple of parts)
inline int ring_slots(double link, size_t stride, double room, int parts) {
    if (stride == 0 || parts <= 0) return parts;
    const double cap_b = std::min((double) ((size_t) kRingMaxMB << 20), room > 0 ? room / 16.0 : 1e300);
    const int cap = std::max(1, (int) (cap_b / (double) stride) / parts);   // (batches of the cap)
    const int want = link <= 0 ? cap
                   : (double) stride / link >= kRingSlowS ? 1
                   : (int) (link * kRingBatchS / (double) stride);
    return std::max(1, std::min(cap, want)) * parts;
}

}  // namespace strata::ds4
