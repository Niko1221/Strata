// include/strata/core/low_vram_rules.hpp - pure low-VRAM (<= 13 GiB) consumer GPU rules (no GPU needed to test).
//
// Track 6A (8 GB & 12 GB consumer GPUs + 4 GB 0-slot cap, v0.2.20):
//   1. Non-vision default VRAM reserve: on <= 8.5 GiB cards, 700 MiB (+ 256 MiB late alloc) leaves ~750 MiB idle
//      while starving ExpertCache (~1,375 slots). Lowering the automatic non-vision reserve to 480 MiB on <= 8.5 GiB
//      and 550 MiB on <= 12.5 GiB recovers ~120-200 VRAM expert slots without starving verify graphs or DFlash2.
//   2. Tiny-cache (< 25% of routed experts) anti-thrashing: when ExpertCache holds < 25% of total routed experts,
//      default --adapt-swaps 96 exchanges ~2,000+ blobs over PCIe per short session and regresses decode by 14-26%
//      on 8 GB & 12 GB cards (RTX 3070 8 GB, RTX 4070 Ti 12 GB, RTX 5070 12 GB). Dampening default swaps (20 when
//      < 12.5% resident, 32 when < 25%) and pinning the top 25% profile Zipf-head slots prevents PCIe thrashing
//      while letting PCIe miss-sharing + RouterLookahead saturate the link.
//   3. Automatic L3 CPU prefetch (STRATA_L3_PF=1): enabled by default when VRAM <= 13 GiB, > 50% of experts live
//      in host RAM, and the CPU pool has >= 8 workers (measured +2..6% decode on 9-worker EPYC; avoided on 4-6
//      worker desktops where static slicing loses work-stealing balance).
//   4. 4 GB VRAM cap (0-slot ExpertCache): allows auto-sizing to land cleanly on 0 VRAM expert slots (k_floor = 0)
//      when VRAM only fits dense trunk + KV + small owned prefill buffers, routing 100% of experts through host RAM
//      + PCIe share without failing Verifier::init.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

constexpr uint64_t kLowVram4GiB  = (5ull << 30) + (512ull << 20);   // 5.5 GiB ceiling (covers 4 GB cards)
constexpr uint64_t kLowVram8GiB  = (8ull << 30) + (512ull << 20);   // 8.5 GiB (covers 8 GB cards: 7.9-8.0 GiB)
constexpr uint64_t kLowVram10GiB = (9ull << 30) + (512ull << 20);   // 9.5 GiB floor (covers 10/12 GB cards: 9.8-12.0 GiB)
constexpr uint64_t kLowVram12GiB = (12ull << 30) + (512ull << 20);  // 12.5 GiB (covers 10/12 GB cards: 9.8-12.0 GiB)
constexpr uint64_t kLowVram13GiB = 13ull << 30;                     // 13.0 GiB ceiling for low-VRAM fastpaths
constexpr uint64_t kWddm16GiB    = 15ull << 30;                     // 15.0 GiB floor (16 GB cards report ~15.9 GiB)

constexpr int kDefaultReserveMib = 700;
constexpr int kLowVram8ReserveMib = 480;
constexpr int kLowVram12ReserveMib = 550;

// #6128 / #5915: Windows WDDM driver 595+ / 621.01 silently pages GPU memory to host DDR4/DDR5 over PCIe
// whenever free physical VRAM falls below the WDDM headroom floor after all late allocations (cuBLAS,
// MMQ/prompt workspace, MTP/DFlash2, CUDA graph capture). Keep a hard post-load free-VRAM safety floor
// on Windows WDDM discrete GPUs (non-TCC, non-integrated).
constexpr int kWddmFloor16GPlusMib = 1280;
constexpr int kWddmFloor10To12GMib = 960;
constexpr int kWddmFloor6To8GMib   = 640;
constexpr int kWddmFloor4GMib      = 448;
// 0.2.21 (#6836 / #6887 / #6509): 384 -> 192.  The BF16/FP16 GEMM warm-up (cuBLAS workspaces, ~180 MiB) now runs BEFORE the cache is sized, so what is
// still allocated after it is the verifier's buffers and graph captures: measured on an RTX 5070 12 GB (WDDM, IQ3_S, spec 4, MTP) 134 MiB at
// --vram-reserve-mib 300 and 110 MiB at 700 ("strata serve: after the expert cache was written ..." line).  384 plus the shrink-after-write loop
// ended the card with 606-620 MiB free (reserve 300): 0.36 GiB of cache lost against 0.1.65 (2287 -> 2099 slots), 886 -> 562 experts on the 8 GB
// laptop of #6836, whose 0.1.62 ran at 336 MiB free and whose forced 250 MiB free paged (the engine's own line: below ~256 MiB a later allocation
// pages GPU memory).  192 leaves the 50-80 MiB margin of a wider window and ends at about the reserve.
constexpr int64_t kWddmLateAllocMib = 192;

/// True on Windows WDDM discrete GPUs (excludes Linux, integrated APUs/UMA, and Windows TCC compute-only GPUs).
inline bool is_wddm_discrete(bool windows, bool integrated, bool tcc_driver) noexcept {
    return windows && !integrated && !tcc_driver;
}

/// Hard free-VRAM safety floor (MiB) on Windows WDDM discrete GPUs after all late allocations.
inline int wddm_vram_safety_floor_mib(uint64_t total_vram_bytes) noexcept {
    if (total_vram_bytes == 0) return 0;
    if (total_vram_bytes >= kWddm16GiB) return kWddmFloor16GPlusMib;
    if (total_vram_bytes >= kLowVram10GiB) return kWddmFloor10To12GMib;
    if (total_vram_bytes >= kLowVram4GiB) return kWddmFloor6To8GMib;
    return kWddmFloor4GMib;
}

/// Effective VRAM reserve (MiB) on Windows WDDM discrete GPUs for `--expert-cache auto`.
/// Raises the default reserve (and legacy setup.py `--vision --vram-reserve-mib 700` configs per #6128)
/// to at least `wddm_vram_safety_floor_mib(total_vram_bytes)`, unless `STRATA_WDDM_VRAM_FLOOR=0`.
/// Never lowers a larger reserve (e.g. Windows HIP's 2560 MiB floor or user `--vram-reserve-mib 1500`).
inline int wddm_effective_reserve_mib(bool wddm_discrete, bool reserve_given, bool vision,
                                      int cur_reserve_mib, uint64_t total_vram_bytes,
                                      const char* floor_env = nullptr) noexcept {
    if (!wddm_discrete || total_vram_bytes == 0) return cur_reserve_mib;
    if (floor_env != nullptr && floor_env[0] == '0') return cur_reserve_mib;
    const int floor_mib = wddm_vram_safety_floor_mib(total_vram_bytes);
    if (cur_reserve_mib >= floor_mib) return cur_reserve_mib;
    if (!reserve_given || (vision && cur_reserve_mib == kDefaultReserveMib)) return floor_mib;
    return cur_reserve_mib;
}

/// Unpriced late-allocation allowance (MiB) reserved before ExpertCache on auto-sized GPUs.
/// On Windows WDDM discrete GPUs this covers cuBLAS/hipBLAS handles, DEQUANT_SIDE scratch, 12-token verify
/// buffers, and 7 captured CUDA graphs so the post-load free VRAM stays strictly above the WDDM floor.
inline int64_t wddm_late_alloc_mib(bool wddm_discrete, bool defaults026, int64_t base_late_mib = 256) noexcept {
    if (defaults026) return 0;
    return wddm_discrete ? kWddmLateAllocMib : base_late_mib;
}

/// Automatic non-vision VRAM reserve (MiB) for discrete <= 12.5 GiB GPUs.
/// Never overrides an explicit --vram-reserve-mib (vision passes 2100 explicitly) or non-default reserves.
inline int low_vram_auto_reserve_mib(bool discrete, bool reserve_given, int cur_reserve_mib, uint64_t total_vram_bytes) noexcept {
    if (!discrete || reserve_given || cur_reserve_mib != kDefaultReserveMib || total_vram_bytes == 0)
        return cur_reserve_mib;
    if (total_vram_bytes <= kLowVram8GiB) return kLowVram8ReserveMib;
    if (total_vram_bytes <= kLowVram12GiB) return kLowVram12ReserveMib;
    return cur_reserve_mib;
}

/// The tiny-cache rules below (dampened --adapt-swaps, pinning the top quarter of the profile) are opt-in since 0.2.21: STRATA_TINY_CACHE=1.
/// Measured against 0.1.65 on an RTX 5080 (sm_120, 16 GB, Qwen3.8-Flash-Next IQ2_XS, `--expert-cache 3000` = 12.2% of the experts, same
/// binary, one engine start per arm and round, arms alternating, 5 rounds): with the rules on decode was -22% (code), -18% (chat), -13% (8K document);
/// with the old swaps given explicitly (`--adapt-swaps 96`) -10% / -10% / -3%, and without the pinning as well (STRATA_CACHE_PIN_FRAC=0 and 96)
/// -5.5% / -6.4% / -4.4%; with a 7.6% cache (`--expert-cache 1800`, 2 rounds so far) -25% / -18% / -18% with the rules on, -10% / -8% / -5% with 96 swaps.  The claim these rules were written on (a 96-swap default costs 14-26% on 8 and 12 GB cards) did not hold on this card.
/// STRATA_CACHE_PIN_FRAC=<f> still pins by itself, and an explicit --adapt-swaps always wins.
inline bool tiny_cache_rules_enabled(const char* env) noexcept { return env != nullptr && env[0] == '1' && env[1] == '\0'; }

/// Dampened default --adapt-swaps when ExpertCache holds < 25% of total routed experts.
/// Preserves explicit --adapt-swaps (including 0) and leaves >= 25% caches untouched.
inline int low_vram_adapt_swaps(int cur_swaps, bool swaps_given, int64_t resident_slots, int64_t total_experts) noexcept {
    if (swaps_given || cur_swaps <= 0 || total_experts <= 0) return cur_swaps;
    if (resident_slots <= 0) return 0;
    if (resident_slots * 8 < total_experts) return std::min(cur_swaps, 16);   // < 12.5% resident (typical 8-12 GB)
    if (resident_slots * 4 < total_experts) return std::min(cur_swaps, 32);   // 12.5-25% resident (typical 12-16 GB)
    return cur_swaps;
}

/// Number of top-ranked expert-profile.bin slots to pin against eviction on tiny caches (< 25% resident).
/// `pin_env` is STRATA_CACHE_PIN_FRAC (nullptr/empty = default 0.25; "0" = off; otherwise clamped 0.0..0.75).
inline int64_t low_vram_pin_slots(int64_t resident_slots, int64_t total_experts, const char* pin_env) noexcept {
    if (resident_slots <= 0 || total_experts <= 0) return 0;
    double frac = 0.0;
    if (pin_env != nullptr && pin_env[0] != '\0') {
        frac = std::max(0.0, std::min(0.75, std::atof(pin_env)));
    } else if (resident_slots * 4 < total_experts) {
        frac = 0.25;
    }
    if (!(frac > 0.0)) return 0;
    return std::min<int64_t>(resident_slots, (int64_t) ((double) resident_slots * frac));
}

/// Whether STRATA_L3_PF (next-layer CPU L3 expert prefetch) should run.
/// Explicit env value always wins; when unset, auto-enables on <= 13 GiB GPUs with > 50% host experts and >= 8 workers.
inline bool l3_pf_auto_default(const char* env, uint64_t total_vram_bytes, int64_t resident_slots,
                               int64_t total_experts, int pool_workers) noexcept {
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    if (total_vram_bytes == 0 || total_vram_bytes > kLowVram13GiB) return false;
    if (total_experts <= 0 || resident_slots * 2 >= total_experts) return false;
    return pool_workers >= 8;
}

}  // namespace strata::core
