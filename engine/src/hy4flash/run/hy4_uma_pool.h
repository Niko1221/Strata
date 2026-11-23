// One physical pool on a unified-memory GPU (#6648): Strix Halo (gfx1151), Strix Point (890M), Phoenix (780M).
//
// On such a chip the "VRAM" the runner asks for (hipMalloc) and the pinned host arena / the CPU-computed experts are the same RAM. The
// expert placement sized the adaptive VRAM slots from the free GPU pool and then the host-resident experts from the free RAM, each one
// from the whole pool: on a 128 GiB Strix Halo, SQ-128 at 128K took 97.45 GB of slots next to a 105.76 GB host arena and the kernel
// killed the engine (the reporter's log). Every function here is plain arithmetic: no GPU, no HIP header.
#pragma once

#include <algorithm>
#include <cstdint>

namespace hy4 {

// What the VRAM-side expert arena may take: never more than the RAM budget, which already leaves the system its share.
inline uint64_t uma_vram_budget(bool uma, uint64_t vram_budget, uint64_t ram_budget) {
    return uma ? std::min(vram_budget, ram_budget) : vram_budget;
}

// The adaptive VRAM slot cache copies experts that also sit in host RAM. On one pool that is the same bytes twice: off.
inline double uma_dyn_frac(bool uma, double dyn_frac) { return uma ? 0.0 : dyn_frac; }

// What is left for the host-resident experts once the static VRAM-side arena (`arena_bytes`, at most all experts) is taken from the same pool.
inline uint64_t uma_host_room(bool uma, uint64_t ram_budget, uint64_t vram_budget, uint64_t all_experts_bytes) {
    if (!uma) return ram_budget;
    const uint64_t arena = std::min(vram_budget, all_experts_bytes);
    return ram_budget > arena ? ram_budget - arena : 0;
}

}  // namespace hy4
