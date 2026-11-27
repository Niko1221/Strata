// engine/src/inkling2/hip/ink_hip_strixhalo_cfg.hpp
// Per-ordinal (0..15) real-card hardware gate & opt-in switch configuration for AMD Strix Halo
// (Ryzen AI Max+ 395 / Radeon 8060S / 8050S, gfx1151, RDNA 3.5, 128 GB UMA, Windows 11 & Linux).
//
// ALL switches are OFF by default. When STRATA_INK_STRIXHALO is unset or "0", every probe returns
// false immediately with ZERO HIP API calls and ZERO default-path behavior changes.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/hip_uma_guard.hpp"

namespace strata::inkling2::hip {

struct StrixHaloCfg {
    bool enabled = false;        // Master switch (STRATA_INK_STRIXHALO=1) AND real-card gate passed
    bool exact_mode = false;     // STRATA_INK_STRIXHALO_EXACT=1 -> forces ALL lossy paths OFF (bit-exact to OFF)
    bool default_set = false;    // STRATA_INK_STRIXHALO unset: the release default for a verified gfx1151 (fused SWA prefill attention only); =1 is the full FAST set, =0 turns everything off

    // Lossless / exact-safe workstreams (enabled in both Exact and Fast modes unless overridden)
    bool slab_arena = false;     // WS1: 256 MiB UmaSlabArena spanning carveout hipMalloc + WDDM mapped slabs
    int  prefill_chunk = 0;      // WS2: effective_chunk() override (8192 in Fast, configurable in Exact)
    int  wddm_flush_stride = 8;  // WS2: non-blocking hipEventRecord + hipEventQuery every N visits (<=180ms/pkt)
    bool swa_fused = false;      // WS2: fixed swa_ring_=1024 + intra-chunk KV + fused rel_logits_proj WMMA
    bool pass0_prune = false;    // WS4: Visit-17 lossless causal subtree compaction M=16 -> M_survive=8 (0.0000 KLD)
    int  pass0_survive = 8;      // WS4: surviving nodes after Visit 17 (8 default, 10 for tool calls)
    bool tree16 = false;         // WS5: M=16 dynamic GPU tree, WinPath16, 115MB INT4 screen head, 1-graph verify
    bool sampled_tree = false;   // WS5: on-GPU stochastic multi-branch verify for temp > 0 (0 full-vocab D2H)
    bool screen_u4 = false;      // WS5: 115 MB symmetric INT4 screen head for DSpark Stage-1 (verify stays exact)
    bool zen5_l3 = false;        // WS6: 16c physical-core high-QoS Zen 5 worker pool & L3 Markov / co-exec
    bool pld = false;            // WS7: Prompt Lookup Drafting (n-gram suffix graft into M=16 tree, 0.0000 KLD)
    bool c64_opt = false;        // WS8: WMMA softcap unembed (8.1x faster) + split-KV C64 decode
    bool c64_spec = false;       // WS8 opt-in: Batched Multi-Slot DSpark (K=2..3) via STRATA_INK_STRIXHALO_C64_SPEC=1
    int  c64_draft_k = 2;        // WS8: draft tokens per slot in C64 batched spec (1..3)
    bool async_ckpt = false;     // WS2: avoid synchronous prompt_len-1 prefill split when supported

    // Lossy / Fast-only workstreams (STRICTLY forced OFF when exact_mode == true!)
    bool pf_iu4 = false;         // WS3A: gfx1151 118.8 TOPS v_wmma_i32_16x16x16_iu4 prompt MoE (opt-in)
    bool pf_i8_64 = false;       // WS3A: gfx1151 59.4 TOPS 2x-rate IU8 WMMA block-64 SF4 prompt MoE (Fast only)
    bool tri_engine = false;     // WS3B: Zero-Copy Tri-Engine (GPU + 16c Zen 5 AVX-512 + XDNA 2 NPU) prefill
    bool trunk_q8 = false;       // WS5: Q8_G32 transcoded dense trunk for decode (-48% trunk BW, Fast only)
    bool sf4_trunk = false;      // Wave-8: SF4 + Outlier-16 trunk + SF4 Coda experts (STRATA_HALO_SF4_TRUNK)
    bool fp8_kv_c64 = false;     // WS8: auto FP8 KV at parallel >= 16 (Fast only)
    bool turbo_tail_skip = false;// WS5 opt-in: Pass 1 & 2 sq.tail_comp <= 6% mass skip (STRATA_INK_STRIXHALO_TURBO=1)
    bool nt_loads = false;       // WS6: non-temporal expert loads (OFF by default on UMA)
};

namespace detail {

inline bool env_truthy(const char* name, bool def_val) noexcept {
    const char* v = std::getenv(name);
    if (!v || !*v) return def_val;
    if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
    return true;
}

inline int env_int(const char* name, int def_val, int lo, int hi) noexcept {
    const char* v = std::getenv(name);
    if (!v || !*v) return def_val;
    int x = std::atoi(v);
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

inline bool str_icontains(const char* hay, const char* needle) noexcept {
    if (!hay || !needle || !*needle) return false;
    const size_t nlen = std::strlen(needle);
    for (const char* p = hay; *p; ++p) {
        size_t i = 0;
        while (i < nlen && p[i] &&
               std::tolower(static_cast<unsigned char>(p[i])) ==
               std::tolower(static_cast<unsigned char>(needle[i]))) {
            ++i;
        }
        if (i == nlen) return true;
    }
    return false;
}

// Release default (11-27): for a verified gfx1151 the fused SWA prefill attention is ON when STRATA_INK_STRIXHALO is unset (KLD at the measured floor, no effect on decode or verify rows, prefill +72% at 4K
// and +96% at 32K on a Radeon 8060S); every other card refuses silently. STRATA_INK_STRIXHALO=0 turns it off, =1 asks for the full FAST set, STRATA_INK_STRIXHALO_EXACT=1 for the exact form.
inline constexpr bool kDefaultOn = true;

inline bool master_explicit() noexcept {
    const char* v = std::getenv("STRATA_INK_STRIXHALO");
    return v && *v;
}

inline bool master_env_requested() noexcept {
    static const bool req = env_truthy("STRATA_INK_STRIXHALO", kDefaultOn);
    return req;
}

inline bool exact_env_requested() noexcept {
    static const bool req = env_truthy("STRATA_INK_STRIXHALO_EXACT", false);
    return req;
}

// Per-ordinal 0..15 cached real-card gate:
//  0 = unchecked, 1 = verified Strix Halo (gfx1151, 8060S/8050S, >=16 WGPs / >=32 CUs, UMA >= 60 GiB), -1 = refused
inline std::atomic<int8_t>& halo_gate_slot(int dev) noexcept {
    static std::atomic<int8_t> slots[16]{};
    return slots[dev & 15];
}

inline std::atomic<int8_t>& halo_warn_slot(int dev) noexcept {
    static std::atomic<int8_t> slots[16]{};
    return slots[dev & 15];
}

inline bool verify_strixhalo_device(int dev) noexcept {
    if (dev < 0 || dev >= 16) return false;
    const int8_t cached = halo_gate_slot(dev).load(std::memory_order_acquire);
    if (cached != 0) return cached > 0;

    hipDeviceProp_t dp{};
    if (hipGetDeviceProperties(&dp, dev) != hipSuccess) {
        halo_gate_slot(dev).store(-1, std::memory_order_release);
        return false;
    }

    const bool arch_ok = strata::hip_uma::arch_is_exact(dp.gcnArchName, "gfx1151");
    const bool is_uma = (dp.integrated != 0) || strata::hip_uma::is_known_amd_igpu_arch(dp.gcnArchName);
    // Radeon 8060S (40 CUs = 20 WGPs) and 8050S (32 CUs = 16 WGPs); ROCm reports WGP count on gfx1151 (16 or 20)
    // or CU count (32 or 40) depending on driver/runtime. Reject small APUs (<=12 CUs / <=8 WGPs) & discrete cards!
    const bool name_ok = str_icontains(dp.name, "8060S") || str_icontains(dp.name, "8050S") ||
                         str_icontains(dp.name, "Strix Halo") || str_icontains(dp.name, "gfx1151");
    const bool cu_ok = (dp.multiProcessorCount >= 16 && dp.multiProcessorCount <= 44);

    // On Windows 11 with a large BIOS carveout (e.g. 96 GB carveout + 32 GB Windows RAM),
    // GlobalMemoryStatusEx reports ~31.6 GiB while hipDeviceProp_t / hipMemGetInfo reports ~107.9 GiB.
    // Check effective combined UMA memory (carveout + host RAM) >= 60 GiB so both 64 GB and 128 GB Strix Halo pass!
    uint64_t eff_free = 0, eff_tot = 0;
    strata::hip_uma::effective_uma_memory(is_uma, dp.totalGlobalMem, dp.totalGlobalMem, eff_free, eff_tot);
    const bool mem_ok = (eff_tot >= (60ULL << 30)) || (dp.totalGlobalMem >= (60ULL << 30));

    const bool pass = arch_ok && is_uma && name_ok && cu_ok && mem_ok;
    halo_gate_slot(dev).store(pass ? 1 : -1, std::memory_order_release);
    if (!pass) {
        int8_t expected = 0;
        if (master_explicit() && halo_warn_slot(dev).compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            std::fprintf(stderr,
                         "[ink-hip] STRATA_INK_STRIXHALO=1 ignored on GPU %d (%s, arch=%s, mp=%d, integrated=%d, mem=%.1f GiB): "
                         "requires AMD Strix Halo gfx1151 (Radeon 8060S/8050S, >=16 WGP, >=60 GiB UMA); running baseline\n",
                         dev, dp.name, dp.gcnArchName, dp.multiProcessorCount, dp.integrated,
                         static_cast<double>(eff_tot) / 1073741824.0);
        }
    }
    return pass;
}

}  // namespace detail

// Query Strix Halo opt-in configuration for a specific HIP ordinal `dev` (0..15).
// Returns an all-false StrixHaloCfg with ZERO HIP calls when STRATA_INK_STRIXHALO is unset/0.
inline StrixHaloCfg query_strixhalo_cfg(int dev, bool ep2_active = false, int peer_dev = -1) noexcept {
    StrixHaloCfg cfg{};
    if (!detail::master_env_requested()) return cfg;
    if (ep2_active) {
        // Multi-GPU EP2: require BOTH ordinals to pass hardware verification independently
        if (!detail::verify_strixhalo_device(dev) || !detail::verify_strixhalo_device(peer_dev)) {
            return cfg;
        }
    } else {
        if (!detail::verify_strixhalo_device(dev)) return cfg;
    }

    cfg.enabled = true;
    cfg.exact_mode = detail::exact_env_requested();

    if (!detail::master_explicit()) {
        // Release default set: the fused SWA prefill attention only (the kernel runs for single-slot batches of 64 rows or more; decode and verify rows keep the shipped kernels).
        cfg.default_set = true;
        cfg.wddm_flush_stride = -1;   // leave the shipped flush stride alone
        cfg.swa_fused = !cfg.exact_mode && detail::env_truthy("STRATA_INK_STRIXHALO_SWA_FUSED", true);
        // chunk 8192 (lossless: KLD 0.000000 at width 2048, 0.0009 / 0.0006 at width 16 for prose / code, floor 0.004): the fused kernel needs the long chunks (prefill 4K: 1071 ppts with the shipped chunk, 1524 with 8192)
        if (cfg.swa_fused) cfg.prefill_chunk = detail::env_int("STRATA_INK_STRIXHALO_CHUNK", 8192, 0, 16384);
        return cfg;
    }

    // Lossless / exact-safe workstreams (all ON when master is ON, individually overridable)
    cfg.slab_arena        = detail::env_truthy("STRATA_INK_STRIXHALO_SLAB", true);
    cfg.wddm_flush_stride = detail::env_int("STRATA_INK_STRIXHALO_FLUSH", 8, 0, 48);
    // In EXACT mode, match baseline's 2048 chunk (100% bit-exact!) unless overridden; Fast uses 8192 + fused SWA.
    cfg.prefill_chunk     = detail::env_int("STRATA_INK_STRIXHALO_CHUNK", cfg.exact_mode ? 2048 : 8192, 0, 16384);
    // SWA_FUSED: wrong logits at prefill width 2048 on gfx1151 (teacher-forced KLD 10.0, top-1 3.5%, PPL 114175 against 0.000000 without it; p620/strix 11-27): opt-in only until fixed.
    cfg.swa_fused         = detail::env_truthy("STRATA_INK_STRIXHALO_SWA_FUSED", !cfg.exact_mode);   // fixed in a583885d5 (it had been skipped by a merge error)
    cfg.pass0_prune       = detail::env_truthy("STRATA_INK_STRIXHALO_PASS0_PRUNE", true);
    cfg.pass0_survive     = detail::env_int("STRATA_INK_STRIXHALO_PASS0_SURVIVE", 8, 4, 16);
    cfg.tree16            = detail::env_truthy("STRATA_INK_STRIXHALO_TREE16", true);
    cfg.sampled_tree      = detail::env_truthy("STRATA_INK_STRIXHALO_SAMPLED_TREE", true);
    cfg.screen_u4         = detail::env_truthy("STRATA_INK_STRIXHALO_SCREEN_U4", true);
    // Core-0 hard affinity contends with Windows 11 WDDM/dxgkrnl DPCs on heavy C64/PPTS; keep opt-in!
    cfg.zen5_l3           = detail::env_truthy("STRATA_INK_STRIXHALO_ZEN5", false);
    cfg.pld               = detail::env_truthy("STRATA_INK_STRIXHALO_PLD", true);
    cfg.c64_opt           = detail::env_truthy("STRATA_INK_STRIXHALO_C64", true);
    cfg.c64_spec          = detail::env_truthy("STRATA_INK_STRIXHALO_C64_SPEC", false);
    cfg.c64_draft_k       = detail::env_int("STRATA_INK_STRIXHALO_C64_K", 2, 1, 3);
    cfg.async_ckpt        = detail::env_truthy("STRATA_INK_STRIXHALO_ASYNC_CKPT", false);
    // Non-temporal loads bypass the 32 MiB Strix Halo MALL and regress bandwidth 2.5x-4x; keep OFF by default.
    cfg.nt_loads          = detail::env_truthy("STRATA_INK_STRIXHALO_NT", false);

    // Lossy / Fast-only workstreams — STRICTLY OFF when exact_mode == true!
    if (!cfg.exact_mode) {
        // FP16 WMMA MoE measured faster than I8/IU4 on gfx1151 (2,375ms vs 2,627ms at 8K); keep I8/IU4 opt-in!
        cfg.pf_iu4          = detail::env_truthy("STRATA_INK_STRIXHALO_PF_IU4", false);
        cfg.pf_i8_64        = detail::env_truthy("STRATA_INK_STRIXHALO_PF_I8", false);
        cfg.tri_engine      = detail::env_truthy("STRATA_INK_STRIXHALO_TRI_ENGINE", false);   // telemetry probe that loads DirectML.dll / xrt_coreutil.dll: no speed effect, off
        cfg.trunk_q8        = detail::env_truthy("STRATA_INK_STRIXHALO_TRUNK_Q8", true);
        cfg.sf4_trunk       = detail::env_truthy("STRATA_HALO_SF4_TRUNK", false);   // verify-width KLD 0.027 prose / 0.018 code, 4-7x the floor (strix 11-27); the Q8 trunk is 0.007 / 0.006: Q8 is the default, SF4 opt-in
        cfg.fp8_kv_c64      = detail::env_truthy("STRATA_INK_STRIXHALO_FP8_KV_C64", false);   // FP8 K/V at C16 and above: lossy, no KL measured: opt-in
        cfg.turbo_tail_skip = detail::env_truthy("STRATA_INK_STRIXHALO_TURBO", false);
    }
    return cfg;
}

}  // namespace strata::inkling2::hip
