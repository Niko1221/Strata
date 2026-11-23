// One physical pool on a unified-memory GPU (#6648): Strix Halo (gfx1151), Strix Point (890M), Phoenix (780M).
//
// On such a chip part of the "VRAM" is RAM: the BIOS carve-out (`dedicated`: what hipMemGetInfo reports itself) is the GPU's own, and what
// the runtime adds from system RAM (the GTT / shared aperture) is the same memory the pinned host arena and the CPU-computed experts use.
// The expert placement sized the adaptive VRAM slots from the free GPU pool and then the host-resident experts from the free RAM, each one
// from the whole pool: on a 128 GiB Strix Halo (carve-out 0.5 GiB, GTT 112 GiB), SQ-128 at 128K took 97.45 GB of slots next to a 105.76 GB
// host arena and the kernel killed the engine (the reporter's log). A Strix Halo whose BIOS gives the GPU 112 GB of dedicated memory
// (Windows, 31.6 GB left for the system) has `dedicated` >= the arena and is not changed by any of this. Plain arithmetic: no GPU, no HIP header.
#pragma once

#include <algorithm>
#include <cstdint>

namespace hy4 {

// What the VRAM-side expert arena may take: its dedicated memory plus at most the safe RAM budget (which already leaves the system its share).
inline uint64_t uma_vram_budget(bool uma, uint64_t vram_budget, uint64_t ram_budget, uint64_t dedicated) {
    return uma ? std::min(vram_budget, dedicated + ram_budget) : vram_budget;
}

// The part of the arena that is RAM: what goes beyond the dedicated memory.
inline uint64_t uma_shared_bytes(bool uma, uint64_t arena_bytes, uint64_t dedicated) {
    return (uma && arena_bytes > dedicated) ? arena_bytes - dedicated : 0;
}

// The adaptive VRAM slot cache copies experts that also sit in host RAM. Inside the dedicated memory that is harmless; beyond it the
// copy is the same RAM twice: off.
inline double uma_dyn_frac(bool uma, double dyn_frac, uint64_t vram_budget, uint64_t dedicated) {
    return (uma && vram_budget > dedicated) ? 0.0 : dyn_frac;
}

// What is left for the host-resident experts once the RAM-backed part of the static arena (at most all experts) is taken from the same pool.
inline uint64_t uma_host_room(bool uma, uint64_t ram_budget, uint64_t vram_budget, uint64_t dedicated, uint64_t all_experts_bytes) {
    if (!uma) return ram_budget;
    const uint64_t shared = uma_shared_bytes(true, std::min(vram_budget, all_experts_bytes), dedicated);
    return ram_budget > shared ? ram_budget - shared : 0;
}

}  // namespace hy4
