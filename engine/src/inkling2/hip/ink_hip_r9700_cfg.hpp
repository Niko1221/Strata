// engine/src/inkling2/hip/ink_hip_r9700_cfg.hpp
// Opt-in configuration & strict per-ordinal hardware gate for AMD Radeon AI PRO R9700 32GB (gfx1201).
// ALL switches are strictly OFF by default (zero HIP calls, zero behavior change when STRATA_INK_R9700 is unset/0).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/hip_uma_guard.hpp"

namespace strata::inkling2::hip::r9700 {

struct Config {
    bool enabled = false;        // Master gate passed (env ON + verified real R9700 32GB on this ordinal)
    bool exact = false;          // STRATA_INK_R9700_EXACT=1 -> 100% lossless bit-identical to OFF at temp=0
    bool fp8_trunk = false;      // FAST only: per-row FP8_E4M3FN dense trunk + 383 TF wmma_fp8_f32 / hipBLASLt FP8
    bool fp8_moe = false;        // FAST only: native gfx1201 wmma_fp8_f32 SF4/MX8 MoE + BM=16 persistent queue
    bool tree_fp8_moe = false;   // FAST only: double-buffered BM=16 FP8 WMMA MoE in forward_tree_verify (B=2..16)
    bool fp8_decode = false;     // FAST only: the FP8 trunk and FP8 MoE also run for B <= 16 (decode / verify rows); off = they run for prompt reading (B > 16) only, decode keeps the shipped kernels
    bool fused_rel_attn = false; // FAST only: fused d_rel=16 wmma_f16 inside FlashAttn (deletes 1.61 GiB sc_rel_bias_)
    bool fixed_swa_ring = false; // Decouple swa_ring_=1024 via intra-chunk KV reads (saves 40 GiB at C64)
    bool nt_mall = false;        // Non-temporal load_u128_nt expert streams (64 MiB MALL bypass; lossless)
    bool spec_opt = false;       // WinPath16 (M=16), INT4 screen head, PLD+tool spine, on-GPU sampled tree, C64 bspec
    bool async_ckpt = false;     // Single-pass prefill + async prompt checkpoint + BCO prewarm + radix prefix
    bool cand_rows = false;      // On-GPU exact top-128 candidate rows (eliminates 51 MB D2H logits at C64)
    int  prefill_chunk = 0;      // Effective prefill chunk (0 = default, up to 16384 on R9700)
    int  bspec_k = 2;            // Batched C64 DSpark lookahead K in {1, 2}
};

namespace detail {

inline bool parse_env_truthy(const char* name, bool default_val) noexcept {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_val;
    return (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
}

inline bool env_set(const char* name) noexcept {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0');
}

inline int parse_env_int(const char* name, int default_val) noexcept {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_val;
    return std::atoi(v);
}

// The FAST set is on by default for a verified R9700 (gfx1201, >= 28 GiB, name R9700; every other card refuses silently); STRATA_INK_R9700=0 turns it off, STRATA_INK_R9700_EXACT=1 asks
// for the exact form. What the default set contains (all measured with DFlash 2 on p620, owner review 11-27): the FP8 trunk and FP8 MoE for prompt rows only (the layers the drafter taps and
// the drafter's window before the prompt's end stay exact), the fused RelPos attention and prefill chunk 8192. Everything else (FIXED_SWA, NT_MALL, ASYNC_CKPT, FP8 decode rows) is opt-in.
inline constexpr bool kDefaultOn = true;

inline bool master_explicit() noexcept {
    return env_set("STRATA_INK_R9700");
}

inline bool master_env_on() noexcept {
    static const bool on = parse_env_truthy("STRATA_INK_R9700", kDefaultOn);
    return on;
}

inline bool exact_env_on() noexcept {
    static const bool on = parse_env_truthy("STRATA_INK_R9700_EXACT", false);
    return on;
}

// Per-ordinal 0..15 hardware verification cache: 0 = unchecked, 1 = verified R9700, -1 = refused
inline std::atomic<int8_t>& ordinal_state(int dev) noexcept {
    static std::atomic<int8_t> st[16]{};
    return st[dev & 15];
}

inline std::atomic<int8_t>& ordinal_warned(int dev) noexcept {
    static std::atomic<int8_t> w[16]{};
    return w[dev & 15];
}

}  // namespace detail

// Fast zero-HIP-call check when master switch is OFF.
inline bool master_requested() noexcept {
    return detail::master_env_on();
}

// Strict per-ordinal real-card check for AMD Radeon AI PRO R9700 32GB (gfx1201, discrete, >= 28 GiB, >= 28 WGPs/CUs, name contains "R9700").
// Explicitly rejects RX 9070 / RX 9070 XT (16 GB gfx1201), RX 9060 XT (gfx1200), RDNA3/3.5, and RDNA2 with a 1-line warning once per ordinal.
inline bool device_is_verified_r9700(int dev) noexcept {
    if (!detail::master_env_on()) return false;
    if (dev < 0 || dev >= 16) return false;
    int8_t s = detail::ordinal_state(dev).load(std::memory_order_acquire);
    if (s != 0) return s > 0;

    hipDeviceProp_t dp{};
    if (hipGetDeviceProperties(&dp, dev) != hipSuccess) {
        (void)hipGetLastError();
        detail::ordinal_state(dev).store(-1, std::memory_order_release);
        if (detail::master_explicit() && detail::ordinal_warned(dev).exchange(1, std::memory_order_acq_rel) == 0) {
            std::fprintf(stderr, "[ink-hip] STRATA_INK_R9700=1 ignored on GPU %d: hipGetDeviceProperties failed; running baseline\n", dev);
        }
        return false;
    }

    const bool arch_ok = strata::hip_uma::arch_is_exact(dp.gcnArchName, "gfx1201");
    const bool discrete_ok = (dp.integrated == 0);
    // ROCm reports WGP count (32 on 64-CU R9700) or CU count (64); require >= 28 to cover both while excluding smaller parts.
    const bool cu_ok = (dp.multiProcessorCount >= 28);
    // Explicitly require >= 28 GiB total VRAM so 16 GB RX 9070 / RX 9070 XT (also gfx1201!) are cleanly excluded.
    const bool mem_ok = (static_cast<uint64_t>(dp.totalGlobalMem) >= (28ULL << 30));
    const bool name_ok = (std::strstr(dp.name, "R9700") != nullptr);

    const bool ok = arch_ok && discrete_ok && cu_ok && mem_ok && name_ok;
    detail::ordinal_state(dev).store(ok ? 1 : -1, std::memory_order_release);
    if (!ok && detail::master_explicit() && detail::ordinal_warned(dev).exchange(1, std::memory_order_acq_rel) == 0) {
        std::fprintf(stderr,
                     "[ink-hip] STRATA_INK_R9700=1 ignored on GPU %d (%s, arch=%s, mp=%d, mem=%.1f GiB): "
                     "requires AMD Radeon AI PRO R9700 (gfx1201, discrete, >=28 WGPs/CUs, >=28 GiB); running baseline\n",
                     dev, dp.name, dp.gcnArchName, dp.multiProcessorCount,
                     static_cast<double>(dp.totalGlobalMem) / 1073741824.0);
    }
    return ok;
}

inline Config resolve_config(int dev, bool ep2_active, int peer_dev) noexcept {
    Config cfg{};
    if (!detail::master_env_on()) return cfg;
    if (!device_is_verified_r9700(dev)) return cfg;
    if (ep2_active && !device_is_verified_r9700(peer_dev)) {
        if (detail::master_explicit() && detail::ordinal_warned(dev).exchange(1, std::memory_order_acq_rel) == 0) {
            std::fprintf(stderr, "[ink-hip] STRATA_INK_R9700=1 ignored under heterogeneous EP2 (GPU %d + GPU %d); running baseline\n",
                         dev, peer_dev);
        }
        return cfg;
    }

    cfg.enabled = true;
    cfg.exact = detail::exact_env_on();

    // Lossless features (active in BOTH EXACT and FAST unless individually turned off via sub-switch=0):
    // Note: nt_mall defaults OFF in EXACT to guarantee 4/4 100% greedy bit-identity with OFF (verified on p620).
    // NT_MALL: decode-only, changes prose tokens and lowers prose decode by 5% with DFlash 2 (p620 only-X matrix): off in both modes now.
    cfg.nt_mall        = detail::parse_env_truthy("STRATA_INK_R9700_NT_MALL", false);
    cfg.spec_opt       = detail::parse_env_truthy("STRATA_INK_R9700_SPEC", true);
    // Single-pass prefill computes the last prompt token inside the big batch instead of alone after the checkpoint; on HIP a row of a batch is not bit-identical to the same row alone,
    // so it changes tokens. Off in EXACT (measured on p620: EXACT with it differs from off on 3 of 4 prompts, without it on 0 of 4); STRATA_INK_R9700_ASYNC_CKPT=1 forces it.
    // With DFlash 2 it also changed code/prose tokens and lowered code decode by 12% (only-X matrix): off in both modes, STRATA_INK_R9700_ASYNC_CKPT=1 still enables it.
    cfg.async_ckpt     = detail::parse_env_truthy("STRATA_INK_R9700_ASYNC_CKPT", false);
    cfg.cand_rows      = detail::parse_env_truthy("STRATA_INK_R9700_CAND", true);

    // Lossy features (STRICTLY OFF in EXACT mode, ON by default in FAST mode unless sub-switch=0):
    if (!cfg.exact) {
        // FIXED_SWA keeps swa_ring_ at the window (1024) with no margin: a later row of a small batch (DFlash verify rows, prompt chunks under 64) overwrites a key an earlier row still reads.
        // Measured on p620 (2047 real tokens, KLD vs the one-token path): width 16 = 0.0144 prose / 0.0090 code with it, 0.000000 without; PPL 6.80 against 7.22. Opt-in only now.
        cfg.fixed_swa_ring = detail::parse_env_truthy("STRATA_INK_R9700_FIXED_SWA", false);
        cfg.fp8_trunk      = detail::parse_env_truthy("STRATA_INK_R9700_FP8_TRUNK", true);
        cfg.fp8_moe        = detail::parse_env_truthy("STRATA_INK_R9700_FP8_MOE", true);
        cfg.tree_fp8_moe   = detail::parse_env_truthy("STRATA_INK_R9700_TREE_FP8_MOE", false);
        cfg.fp8_decode     = detail::parse_env_truthy("STRATA_INK_R9700_FP8_DECODE", false);   // measured with DFlash 2: FP8 rows for B <= 16 lower decode (prose -14%, chat -16..-21%, tool -20% for the trunk), so off
        cfg.fused_rel_attn = detail::parse_env_truthy("STRATA_INK_R9700_FUSED_ATTN", true);
        cfg.prefill_chunk  = detail::parse_env_int("STRATA_INK_R9700_CHUNK", 8192);
        if (cfg.prefill_chunk < 512) cfg.prefill_chunk = 512;
        if (cfg.prefill_chunk > 16384) cfg.prefill_chunk = 16384;
    } else {
        // In EXACT mode, chunk defaults to 0 (baseline 512) unless explicitly overridden;
        // fixed_swa_ring activates automatically for large EXACT chunks (>512) or high concurrency (parallel>=16)
        // so single-slot EXACT at default chunk matches OFF 100% bit-identically on every prompt!
        cfg.prefill_chunk = detail::env_set("STRATA_INK_R9700_EXACT_CHUNK")
            ? std::max(512, std::min(16384, detail::parse_env_int("STRATA_INK_R9700_EXACT_CHUNK", 512)))
            : 0;
        cfg.fixed_swa_ring = detail::env_set("STRATA_INK_R9700_FIXED_SWA")
            ? detail::parse_env_truthy("STRATA_INK_R9700_FIXED_SWA", false)
            : (cfg.prefill_chunk > 512);
    }

    cfg.bspec_k = std::max(1, std::min(3, detail::parse_env_int("STRATA_INK_R9700_BSPEC_K", 2)));
    return cfg;
}

}  // namespace strata::inkling2::hip::r9700
