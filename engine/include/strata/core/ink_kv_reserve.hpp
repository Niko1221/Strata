// strata-inkling CUDA: what the K/V auto-offload keeps free for the workspaces the runner allocates AFTER the global K/V.
//
// The engine sizes the window of the global K/V that stays on the card (3/4 of what fits) from the memory left after the experts, minus this reserve.
// v0.2.20 reserved 640 MiB (1280 from 8 slots) plus 160 MiB when the file carries a drafter. 759ccf6c6 raised that to 768 (+256), more than the loader
// leaves for the same workspaces on a card of up to 13 GB (512 MiB, +224 with a drafter), and on a 12 GB card the window fell from 7168 to 2304 positions
// (SQ-10) and from 5120 to 0 (UD-Q4_K_XL): decode at a 4K prompt 39.5 -> 29.2 and 10.2 -> 8.9 tok/s.
// These are v0.2.20's values again, exactly.
#pragma once

#include <cstddef>

namespace strata::core {

constexpr size_t kInkWsReserveFewSlotsBytes  = 640ull << 20;    // under 8 slots
constexpr size_t kInkWsReserveManySlotsBytes = 1280ull << 20;   // 8 slots and more
constexpr size_t kInkWsReserveDrafterBytes   = 160ull << 20;    // when the file carries a drafter

inline size_t ink_runner_ws_reserve_bytes(int max_slots, bool drafter_in_file) {
    return (max_slots >= 8 ? kInkWsReserveManySlotsBytes : kInkWsReserveFewSlotsBytes) + (drafter_in_file ? kInkWsReserveDrafterBytes : 0);
}

}  // namespace strata::core
