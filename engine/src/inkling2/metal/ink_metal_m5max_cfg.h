// engine/src/inkling2/metal/ink_metal_m5max_cfg.h - Strict Opt-In Apple M5 Max Speed Path Configuration
// Default OFF (STRATA_INK_M5MAX unset or != "1"): zero drift, unmodified efe32d018 baseline path.
// STRATA_INK_M5MAX=1: hardware-gated to Apple M5 Max (Apple10 + Metal 4 + >=32 GPU cores + >=64 GiB UMA).
//   - STRATA_INK_M5MAX_EXACT=1 -> EXACT mode (lossless speedups only; 100% greedy temp=0 bit-identical to OFF)
//   - STRATA_INK_M5MAX_EXACT=0/unset -> FAST mode (adds INT8 Q8_ROW NAX trunk, FP8_E4M3 KV, TM=8/16 NAX MoE)
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mtl_rt.h"

namespace ink_metal {

struct M5MaxCfg {
    bool requested = false;       // STRATA_INK_M5MAX=1 was set
    bool active = false;          // hardware matched & enabled
    bool exact = false;           // STRATA_INK_M5MAX_EXACT=1 (zero lossy opts)
    bool refused = false;         // STRATA_INK_M5MAX=1 on a non-matching GPU

    // Detected hardware properties
    std::string device_name;
    bool is_apple10 = false;
    bool has_metal4 = false;
    int gpu_cores = 0;
    uint64_t phys_mem_bytes = 0;

    // Lossless speedups (enabled in BOTH EXACT and FAST when active)
    bool use_residency_set = false;
    bool use_untracked_scratch = false;
    bool use_flat_tape = false;
    bool use_madv_willneed = false;
    bool use_native_rco_gguf = false;
    bool use_dflash2 = false;
    bool use_fast_sf4_shfl = false;
    bool use_dbuf_moe = false;
    bool use_reg_relpos_pf = false;
    bool use_swa_intrachunk = false;
    bool use_candidates = false;
    bool use_post_step_ckpt = false;
    int  prefill_chunk = 512;
    int  prefill_cap_with_decoders = 0;
    bool exact_verify_m1_match = false;

    // Lossy speedups (enabled ONLY in FAST when active && !exact; every item documented in HANDOFF.md)
    bool use_i8_trunk = false;
    bool use_fp8_kv = false;
    int  mpp_moe_min_m = 10;
    bool use_moe_tm8 = false;

    static bool env_on(const char* name) {
        const char* v = std::getenv(name);
        return v && v[0] == '1' && v[1] == '\0';
    }

    static bool hardware_matches(const std::string& name, bool apple10, bool metal4, int cores, uint64_t phys_bytes) {
        if (name.find("Apple M5 Max") == std::string::npos) return false;
        if (!apple10 || !metal4) return false;
        if (cores > 0 && cores < 32) return false;
        if (phys_bytes < (64ull << 30)) return false;
        return true;
    }

    static M5MaxCfg probe(const std::string& name, bool apple10, bool metal4, int cores, uint64_t phys_bytes,
                          bool req_override = false, bool exact_override = false, bool use_overrides = false) {
        M5MaxCfg c;
        c.device_name = name;
        c.is_apple10 = apple10;
        c.has_metal4 = metal4;
        c.gpu_cores = cores;
        c.phys_mem_bytes = phys_bytes;
        c.requested = use_overrides ? req_override : env_on("STRATA_INK_M5MAX");
        if (!c.requested) return c;

        if (!hardware_matches(name, apple10, metal4, cores, phys_bytes)) {
            c.refused = true;
            std::fprintf(stderr,
                         "[ink_metal] STRATA_INK_M5MAX=1 refused: device '%s' (apple10=%d, metal4=%d, cores=%d, mem=%.0f GiB) is not Apple M5 Max (>=64 GiB); running baseline path\n",
                         name.c_str(), (int)apple10, (int)metal4, cores, phys_bytes / (1024.0 * 1024.0 * 1024.0));
            return c;
        }

        c.active = true;
        c.exact = use_overrides ? exact_override : env_on("STRATA_INK_M5MAX_EXACT");

        // Lossless optimizations (EXACT + FAST)
        c.use_residency_set = true;
        c.use_untracked_scratch = true;
        c.use_flat_tape = true;
        c.use_madv_willneed = true;
        c.use_native_rco_gguf = true;
        c.use_dflash2 = true;
        c.use_fast_sf4_shfl = false;  // Measured on M5 Max: constant-register-file LUT beats 16x simd_shuffle by +3.3 ms/tok
        c.use_dbuf_moe = !c.exact;    // Use 8 KB SLM dir kernels for SF4/MX8 (2x occupancy); dbuf_gen unlocks NAX MoE for FMT_GGUF_RAW in FAST
        c.use_reg_relpos_pf = !c.exact;
        c.use_swa_intrachunk = !c.exact;
        c.use_candidates = true;
        c.use_post_step_ckpt = !c.exact;  // EXACT matches OFF's [P-1]+[1] prompt-end split for 100% greedy parity
        c.prefill_chunk = c.exact ? 512 : 8192;
        if (!c.exact) {
            if (const char* ce = std::getenv("STRATA_INK_M5MAX_CHUNK"); ce && std::atoi(ce) >= 64) {
                c.prefill_chunk = std::min(8192, std::atoi(ce));
            }
        }
        c.prefill_cap_with_decoders = c.exact ? 0 : 1024;
        c.exact_verify_m1_match = c.exact;

        // Lossy optimizations (FAST only)
        if (!c.exact) {
            c.use_i8_trunk = !(std::getenv("STRATA_INK_M5MAX_I8_TRUNK") && std::getenv("STRATA_INK_M5MAX_I8_TRUNK")[0] == '0');
            c.use_fp8_kv = !(std::getenv("STRATA_INK_M5MAX_FP8_KV") && std::getenv("STRATA_INK_M5MAX_FP8_KV")[0] == '0');
            c.mpp_moe_min_m = 10;
            c.use_moe_tm8 = false;
        } else {
            c.use_i8_trunk = false;
            c.use_fp8_kv = false;
            c.mpp_moe_min_m = 10;
            c.use_moe_tm8 = false;
        }
        return c;
    }

    static M5MaxCfg detect(const mtl::Context& cx) {
        return probe(cx.device_name(), cx.has_family_apple(10), cx.supports_metal4(),
                     cx.gpu_core_count(), cx.physical_memory_bytes());
    }
};

}  // namespace ink_metal
