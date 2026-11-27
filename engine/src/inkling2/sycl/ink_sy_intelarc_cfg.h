// ink_sy_intelarc_cfg.h - Opt-in Intel Arc Battlemage/Alchemist/Core Ultra specialization gate (STRATA_INK_INTELARC)
// Default OFF (STRATA_INK_INTELARC=0): zero default-path drift on any device!
// Per-ordinal 0..15 real-card gate matching Intel Arc Battlemage (B70/B65/B60/B580/B570), Alchemist (A770/A750/A580),
// and Core Ultra UMA (Arc B390/140V/130V). Non-matching SYCL devices log a 1-line refusal warning and stay on default path.
// STRATA_INK_INTELARC_EXACT=1 forces 100% lossless exact math (greedy temp=0 bit-identical to OFF, Mean KLD = 0.000000).
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace ink_sycl {

enum class IntelArcSku : int {
    None = 0,
    B70,      // Arc Pro B70 32GB (bmg-g31, 608 GB/s)
    B65,      // Arc Pro B65 32GB (bmg-g31)
    B60,      // Arc Pro B60 24GB (bmg-g21, 456 GB/s)
    B580,     // Arc B580 12GB (bmg-g21, 456 GB/s)
    B570,     // Arc B570 10GB (bmg-g21, 380 GB/s)
    A770,     // Arc A770 16GB / 8GB (acm-g10, 512/560 GB/s)
    A750,     // Arc A750 8GB (acm-g10, 512 GB/s)
    A580,     // Arc A580 8GB (acm-g10)
    CoreUltraUma, // Core Ultra Arc B390 / 140V / 130V / integrated Arc UMA
    OtherArc  // Other real Intel Arc Battlemage/Alchemist GPU
};

struct IntelArcCfg {
    bool requested = false;       // STRATA_INK_INTELARC=1
    bool active = false;          // requested && matched real Intel Arc GPU
    bool exact = false;           // active && STRATA_INK_INTELARC_EXACT=1 (100% greedy bit-identical to OFF)
    bool fast = false;            // active && !exact (allows itemized lossy optimizations)
    IntelArcSku sku = IntelArcSku::None;
    std::string sku_name = "none";
    std::string dev_name;
    bool is_bmg = false;          // Battlemage (bmg-g21 / bmg-g31 / B390)
    bool is_uma = false;          // host_unified_memory (zero-copy UMA)
    bool has_xmx = false;         // ext_intel_matrix (DPAS joint_matrix)
    uint64_t vram_bytes = 0;
    uint32_t eu_count = 0;

    // Tuned knobs (all zero/default when !active)
    int effective_chunk = 0;          // 8192 (>=28 GiB), 4096 (21..27 GiB), 2048 (<=16 GiB)
    int dflash2_max_m = 8;            // 16 all-VRAM (B70/B65/B60), 8 Hybrid / <=16 GiB (calibrated via ceiling.py)
    size_t headroom_reserve_mib = 0;  // 580 MiB clean reserve when active
    int hybrid_max_m = 2;             // 16 when active (Iso-Time Hybrid), 2 when OFF
    bool use_l0_v2 = false;           // UR_LOADER_USE_LEVEL_ZERO_V2=1 + immediate command lists
    bool use_graph = false;           // multi-bucket SYCL command_graph
    bool use_gpu_moe_plan = false;    // GPU-side active-expert bucket planner (0 CPU d2h stalls)
    bool use_l0_ep2_barrier = false;  // ext_oneapi_submit_barrier for EP2
    bool use_q8_trunk = false;        // FAST only: fused 1-byte Q8_ROW trunk compaction (~0.69 GiB saved)
    bool use_fp8_kv_auto = false;     // FAST only: auto FP8_E4M3 64-token paged KV when kv_dtype_auto
    bool use_bf16_sconv = false;      // FAST only: BF16 ShortConv history
    bool use_joint_matrix_xmx = false;// FAST only: native SG16 joint_matrix XMX FP16/INT8 DPAS (zero oneMKL)
    bool use_vec_lsc_decode = false;  // vectorized 16B uint4 LSC + 1-cycle select_from_group LUT
    bool use_dflash2 = false;         // native SYCL SG16 DFlash 2 drafter
};

inline bool env_truthy_arc(const char* name) noexcept {
    const char* e = std::getenv(name);
    if (!e || !e[0]) return false;
    return e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T';
}

inline bool intelarc_env_requested() noexcept {
    return env_truthy_arc("STRATA_INK_INTELARC");
}

inline bool intelarc_exact_env_requested() noexcept {
    return env_truthy_arc("STRATA_INK_INTELARC_EXACT");
}

// Is this device a Battlemage-generation (Xe2) Intel GPU? By the device's own architecture id (the oneAPI architecture extension), not by its name. -1: the runtime cannot say.
inline int intelarc_arch_is_battlemage(const sycl::device& dev) {
#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
    namespace exp = sycl::ext::oneapi::experimental;
    try {
        const auto a = dev.get_info<exp::info::device::architecture>();
        if (a == exp::architecture::intel_gpu_bmg_g21 || a == exp::architecture::intel_gpu_bmg_g31 || a == exp::architecture::intel_gpu_lnl_m) return 1;
        if (a == exp::architecture::unknown) return -1;
        return 0;
    } catch (...) {
        return -1;
    }
#else
    (void)dev;
    return -1;
#endif
}

// The fast Intel Arc set (STRATA_INK_INTELARC=1 without _EXACT) is refused on Battlemage in 0.2.23: on a real Arc Pro B70 it lost the GPU (UR_RESULT_ERROR_DEVICE_LOST) on the GGUF files and in the
// 32-slot run and returned garbage text from the server; only the exact set is verified there. Alchemist keeps the fast set as measured.
inline bool intelarc_fast_refused(bool battlemage, bool exact_requested) noexcept {
    return battlemage && !exact_requested;
}

inline std::string to_lower_ascii(const std::string& s) {
    std::string out = s;
    for (char& ch : out) ch = (char)std::tolower((unsigned char)ch);
    return out;
}

inline IntelArcCfg probe_intelarc_card(const sycl::device& dev, int ordinal) {
    IntelArcCfg cfg;
    cfg.requested = intelarc_env_requested();
    cfg.dev_name = dev.get_info<sycl::info::device::name>();
    cfg.vram_bytes = dev.get_info<sycl::info::device::global_mem_size>();
    cfg.eu_count = dev.get_info<sycl::info::device::max_compute_units>();
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    cfg.is_uma = dev.get_info<sycl::info::device::host_unified_memory>();
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(SYCL_EXT_INTEL_MATRIX)
    cfg.has_xmx = dev.has(sycl::aspect::ext_intel_matrix);
#else
    cfg.has_xmx = false;
#endif
    if (!cfg.requested) return cfg;

    const std::string nl = to_lower_ascii(cfg.dev_name);
    const std::string vl = to_lower_ascii(dev.get_info<sycl::info::device::vendor>());
    const bool is_intel = (vl.find("intel") != std::string::npos) || (nl.find("intel") != std::string::npos);
    const bool is_arc_or_bmg = (nl.find("arc") != std::string::npos) || (nl.find("bmg") != std::string::npos) ||
                               (nl.find("battlemage") != std::string::npos) || (nl.find("dg2") != std::string::npos) ||
                               (nl.find("alchemist") != std::string::npos) || (nl.find("140v") != std::string::npos) ||
                               (nl.find("130v") != std::string::npos) || (nl.find("b390") != std::string::npos);

    if (!is_intel || !is_arc_or_bmg) {
        static std::array<bool, 16> warned{};
        const int idx = std::clamp(ordinal, 0, 15);
        if (!warned[idx]) {
            warned[idx] = true;
            std::fprintf(stderr,
                         "[ink_sycl] STRATA_INK_INTELARC=1 refused on ordinal %d (%s, vendor=%s): not a supported Intel Arc Battlemage/Alchemist/Core Ultra GPU; using default path\n",
                         ordinal, cfg.dev_name.c_str(), dev.get_info<sycl::info::device::vendor>().c_str());
        }
        return cfg;
    }

    const double gib = (double)cfg.vram_bytes / (1024.0 * 1024.0 * 1024.0);
    if (nl.find("b70") != std::string::npos || (nl.find("bmg") != std::string::npos && gib >= 28.0)) {
        cfg.sku = IntelArcSku::B70; cfg.sku_name = "Arc Pro B70"; cfg.is_bmg = true;
    } else if (nl.find("b65") != std::string::npos) {
        cfg.sku = IntelArcSku::B65; cfg.sku_name = "Arc Pro B65"; cfg.is_bmg = true;
    } else if (nl.find("b60") != std::string::npos || (nl.find("bmg") != std::string::npos && gib >= 20.0)) {
        cfg.sku = IntelArcSku::B60; cfg.sku_name = "Arc Pro B60"; cfg.is_bmg = true;
    } else if (nl.find("b580") != std::string::npos) {
        cfg.sku = IntelArcSku::B580; cfg.sku_name = "Arc B580"; cfg.is_bmg = true;
    } else if (nl.find("b570") != std::string::npos) {
        cfg.sku = IntelArcSku::B570; cfg.sku_name = "Arc B570"; cfg.is_bmg = true;
    } else if (nl.find("b390") != std::string::npos || nl.find("140v") != std::string::npos ||
               nl.find("130v") != std::string::npos || cfg.is_uma) {
        cfg.sku = IntelArcSku::CoreUltraUma; cfg.sku_name = "Core Ultra Arc UMA";
        cfg.is_bmg = (nl.find("b390") != std::string::npos || nl.find("140v") != std::string::npos || nl.find("130v") != std::string::npos);
    } else if (nl.find("a770") != std::string::npos) {
        cfg.sku = IntelArcSku::A770; cfg.sku_name = "Arc A770";
    } else if (nl.find("a750") != std::string::npos) {
        cfg.sku = IntelArcSku::A750; cfg.sku_name = "Arc A750";
    } else if (nl.find("a580") != std::string::npos) {
        cfg.sku = IntelArcSku::A580; cfg.sku_name = "Arc A580";
    } else {
        cfg.sku = IntelArcSku::OtherArc; cfg.sku_name = "Intel Arc";
        cfg.is_bmg = (nl.find("bmg") != std::string::npos || nl.find("battlemage") != std::string::npos);
    }

    cfg.active = true;
    cfg.exact = intelarc_exact_env_requested();
    {   // Battlemage by the device's architecture (the name list above is only the fallback when the runtime cannot say)
        const int arch_bmg = intelarc_arch_is_battlemage(dev);
        const bool battlemage = (arch_bmg >= 0) ? (arch_bmg == 1) : cfg.is_bmg;
        if (intelarc_fast_refused(battlemage, cfg.exact)) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "[ink_sycl] the fast Intel Arc set is not available on Battlemage in this release; running the exact set\n");
            }
            cfg.exact = true;
        }
    }
    cfg.fast = !cfg.exact;

    // Effective prefill chunk sized by real VRAM (with intra-chunk SWA pos&1023 fix!)
    // Kept at 0 (baseline 512) in EXACT mode to guarantee 100% greedy bit-identity vs OFF on 8K+ prompts!
    if (cfg.fast) {
        if (gib >= 28.0) cfg.effective_chunk = 8192;
        else if (gib >= 20.5) cfg.effective_chunk = 4096;
        else cfg.effective_chunk = 2048;
    } else {
        cfg.effective_chunk = 0;
    }

    // ceiling.py MoE expert-union calibrated verify width: 16 all-VRAM (>=21 GiB), 8 Hybrid (<=16 GiB)
    cfg.dflash2_max_m = (gib >= 20.5) ? 16 : 8;
    cfg.headroom_reserve_mib = 580;
    cfg.hybrid_max_m = 16;

    // Lossless / exact-safe speedups active in BOTH EXACT and FAST modes:
    cfg.use_l0_v2 = true;
    cfg.use_graph = true;
    cfg.use_gpu_moe_plan = true;
    cfg.use_l0_ep2_barrier = true;
    cfg.use_vec_lsc_decode = true;
    cfg.use_dflash2 = true;

    // Lossy optimizations strictly gated to FAST mode (!exact), itemized 1-by-1 in HANDOFF.md:
    //   1. Native SG16 joint_matrix XMX FP16/INT8 DPAS (Battlemage Xe2 only; zero oneMKL; FP16/INT8 input rounding)
    //   2. Fused 1-byte Q8_ROW dense trunk compaction (quantizes BF16 attn projections to Q8_ROW)
    //   3. Auto FP8_E4M3 64-token paged KV cache (when --kv auto)
    //   4. BF16 ShortConv history storage (halves ShortConv footprint)
    //   5. Larger effective prefill chunk (2048/4096/8192 vs 512) with intra-chunk SWA
    if (cfg.fast) {
        cfg.use_joint_matrix_xmx = cfg.is_bmg && cfg.has_xmx && !env_truthy_arc("STRATA_SYCL_NO_XMX") && !env_truthy_arc("INK_SYCL_NO_XMX");
        cfg.use_q8_trunk = !env_truthy_arc("STRATA_INK_INTELARC_NO_Q8_TRUNK");
        cfg.use_fp8_kv_auto = !env_truthy_arc("STRATA_INK_INTELARC_NO_FP8_KV");
        cfg.use_bf16_sconv = !env_truthy_arc("STRATA_INK_INTELARC_NO_BF16_SCONV");
    }

    static std::array<bool, 16> logged{};
    const int idx = std::clamp(ordinal, 0, 15);
    if (!logged[idx]) {
        logged[idx] = true;
        std::fprintf(stderr,
                     "[ink_sycl_intelarc] ACTIVE ordinal=%d sku=%s (%s, %.1f GiB, %u EU, bmg=%d, uma=%d, xmx=%d) mode=%s chunk=%d df2_M*=%d q8_trunk=%d fp8_kv=%d bf16_sc=%d jm_xmx=%d\n",
                     ordinal, cfg.sku_name.c_str(), cfg.dev_name.c_str(), gib, cfg.eu_count,
                     (int)cfg.is_bmg, (int)cfg.is_uma, (int)cfg.has_xmx,
                     cfg.exact ? "EXACT" : "FAST",
                     cfg.effective_chunk, cfg.dflash2_max_m,
                     (int)cfg.use_q8_trunk, (int)cfg.use_fp8_kv_auto, (int)cfg.use_bf16_sconv,
                     (int)cfg.use_joint_matrix_xmx);
    }
    return cfg;
}

// Cached per-ordinal (0..15) IntelArcCfg so hot paths pay zero SYCL info queries.
inline const IntelArcCfg& intelarc_cfg(const sycl::device& dev, int ordinal = 0) {
    static std::array<IntelArcCfg, 16> cache{};
    static std::array<bool, 16> ready{};
    const int idx = std::clamp(ordinal, 0, 15);
    if (!ready[idx]) {
        cache[idx] = probe_intelarc_card(dev, idx);
        ready[idx] = true;
    }
    return cache[idx];
}

}  // namespace ink_sycl
