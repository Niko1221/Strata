// engine/src/inkling2/cuda/ink_sm120_cfg.cuh - the one per-device switch set of the RTX 50 (sm_120) fast mode
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// The decision itself (switch names, aliases, the card table, "refuse cleanly") lives in strata/core/ink_sm120_table.hpp, which has no CUDA
// dependency and a unit test. This header reads the device once (name, cc, SMs, VRAM), asks it, caches the answer per device ordinal, and
// offers the answer to the CUDA code in two shapes the code already uses:
//   * strata::ink::rtx5090::*_on()     the 5090 hand-over's part switches  (kept as thin wrappers so the call sites read as before)
//   * strata::ink::ink_rtx5060ti_cfg() the 5060 Ti hand-over's config struct (kept the same way)
// Neither reads the environment itself any more. With every switch unset (and kSm120FastDefault false) every flag below is false and every
// number is the shipped one.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ink_cuda.cuh"
#include "ink_moe_fp4_sm120.cuh"
#include "strata/core/ink_caps.hpp"
#include "strata/core/ink_sm120_table.hpp"

namespace strata::ink {

struct Sm120Cfg {
    strata::core::Sm120Decision d;                   // what was decided and why
    const strata::core::Sm120Row* row = nullptr;     // the row whose numbers apply (nullptr: none)
    bool ready = false;

    bool fast = false;     // mode Fast
    bool exact = false;    // mode Exact (the form that must not change a bit)
    bool any = false;      // at least one part is on

    // parts
    bool spec_adapt = false;       // the draft window follows the measured step cost per row and acceptance (fast only)
    bool tc_attn = false;          // tensor-core split-K decode attention of the global visits (fast only)
    bool pf_attn = false, fp8_cache = false, fp8_inplace = false, moe_opt = false, dense_opt = false, dspark_opt = false;
    bool c64_bspec = false, c64_k8v4 = false;
    bool compact_xcode = false, compact_kv = false;   // the two lossy halves of compact, each on only while compact is on
    bool compact_hybrid = false;   // compact trunk + the experts that fit (8 / 12 GB cards), instead of refusing compact
    bool compact = false, tc_smallm = false, pld = false, pack = false, prefill = false, core_graph = false, l2pf = false;
    bool sf4_trunk = false;        // experimental SF4 dense trunk (never on by default)
    bool deep_tree = false;        // the 32 x 32 ancestor table and the deep verify kernel: dspark_opt, pld or core_graph need it

    // numbers (the row's, with the environment's overrides)
    float pcie_share = 0.28f;
    int   c64_draft_k = 2;
    int   c1_beam = 1, c1_max_nodes = 12, c1_pld_max_m = 24;
    float c1_spec_cost = 2.0f, c1_spec_plain = 1.02f;
    int   spec_beam = 4;
    float spec_cost = 12.0f, spec_plain = 1.02f;
    int   c1_chunk_big = 0, c1_chunk_big_headroom_mib = 0, c1_chunk = 0;
    int   compact_max_vram_gib = 0, fp8_cache_headroom_mib = 0, ws_extra_fp8_mib = 0, ws_extra_dspark_mib = 0;
    int   experts_in_vram_policy = 0;
};

namespace sm120_cfg_detail {

inline const char* env_get(const char* n) { return std::getenv(n); }
/// new name first, then the old (5060 Ti hand-over) alias
inline const char* env2(const char* n, const char* old_name) { const char* e = std::getenv(n); return (e && *e) ? e : std::getenv(old_name); }

inline int env_int(const char* name, const char* alias, int def, int lo, int hi) {
    const char* e = std::getenv(name);
    if ((!e || !*e) && alias) e = std::getenv(alias);
    if (!e || !*e) return def;
    return std::max(lo, std::min(hi, std::atoi(e)));
}
inline float env_f32(const char* name, const char* alias, float def, float lo, float hi) {
    const char* e = std::getenv(name);
    if ((!e || !*e) && alias) e = std::getenv(alias);
    if (!e || !*e) return def;
    return std::max(lo, std::min(hi, (float)std::atof(e)));
}

inline void fill(Sm120Cfg& c, const strata::core::Sm120Decision& d) {
    using namespace strata::core;
    c.d = d;
    c.row = d.row;
    c.fast = (d.mode == Sm120Decision::Fast);
    c.exact = (d.mode == Sm120Decision::Exact);
    c.any = (d.parts != 0);
    const uint32_t p = d.parts;
    const Sm120Row* r = d.row;
    c.pf_attn = (p & P_PF_ATTN) != 0;
    c.tc_attn = (p & P_TC_ATTN) != 0;
    c.spec_adapt = (p & P_SPEC_ADAPT) != 0;
    c.moe_opt = (p & P_MOE_OPT) != 0;
    c.dense_opt = (p & P_DENSE_OPT) != 0;
    c.dspark_opt = (p & P_DSPARK_OPT) != 0;
    // one FP8 trunk: the 5090's cache. The in-place one runs only when FP8_INPLACE is asked for by name (and then instead of the cache).
    c.fp8_inplace = (p & P_FP8_INPLACE) != 0 && r != nullptr;
    c.fp8_cache = (p & P_FP8_TRUNK) != 0 && r && r->fp8_mode == Sm120Fp8::Cache && !c.fp8_inplace;
    c.c64_bspec = (p & P_C64_SPEC) != 0 && r && r->c64_mode == Sm120C64::Bspec;
    c.c64_k8v4 = (p & P_C64_SPEC) != 0 && r && r->c64_mode == Sm120C64::K8v4;
    c.compact = (p & P_COMPACT) != 0;
    c.compact_xcode = c.compact && (p & P_COMPACT_XCODE) != 0;
    c.compact_kv = c.compact && (p & P_COMPACT_KV) != 0;
    c.compact_hybrid = c.compact && (p & P_COMPACT_HYBRID) != 0;
    c.tc_smallm = (p & P_TC_SMALLM) != 0;
    c.pld = (p & P_PLD) != 0;
    c.pack = (p & P_PACK) != 0;
    c.prefill = (p & P_PREFILL) != 0;
    c.core_graph = (p & P_CORE_GRAPH) != 0;
    c.l2pf = (p & P_L2PF) != 0;
    c.sf4_trunk = c.any && env_int("STRATA_INK_SM120_SF4_TRUNK", "STRATA_INK_5060TI_SF4_TRUNK", 0, 0, 1) != 0;
    c.deep_tree = c.dspark_opt || c.pld || c.core_graph;
    if (r) {
        c.pcie_share = env_f32("STRATA_INK_SM120_PCIE_SHARE", "STRATA_INK_5060TI_PCIE_SHARE", r->pcie_share, 0.0f, 1.0f);
        c.c64_draft_k = env_int("STRATA_INK_SM120_C64_DRAFT_K", "STRATA_INK_5060TI_C64_DRAFT_K", r->c64_draft_k, 1, 7);
        c.c1_beam = env_int("STRATA_INK_SM120_BEAM", "STRATA_INK_5060TI_BEAM", 1, 1, 4);
        c.c1_max_nodes = env_int("STRATA_INK_SM120_MAX_NODES", "STRATA_INK_5060TI_MAX_NODES", r->c1_max_nodes, 4, kInkMaxTree);
        c.c1_pld_max_m = std::max(c.c1_max_nodes,
                                  env_int("STRATA_INK_SM120_PLD_MAX_M", "STRATA_INK_5060TI_PLD_MAX_M", r->c1_pld_max_m, 4, kInkMaxTree));
        c.c1_spec_cost = std::max(0.2f, env_f32("STRATA_INK_SM120_SPEC_COST", "STRATA_INK_5060TI_SPEC_COST", r->spec_cost, 0.0f, 1000.0f));
        c.c1_spec_plain = std::max(0.0f, env_f32("STRATA_INK_SM120_SPEC_PLAIN", "STRATA_INK_5060TI_SPEC_PLAIN", r->spec_plain, 0.0f, 1000.0f));
        c.spec_beam = r->spec_beam;
        c.spec_cost = r->spec_cost;
        c.spec_plain = r->spec_plain;
        c.c1_chunk_big = r->c1_chunk_big;
        c.c1_chunk_big_headroom_mib = r->c1_chunk_big_headroom_mib;
        c.c1_chunk = r->c1_chunk;
        c.compact_max_vram_gib = r->compact_max_vram_gib;
        c.fp8_cache_headroom_mib = r->fp8_cache_headroom_mib;
        c.ws_extra_fp8_mib = r->ws_extra_fp8_mib;
        c.ws_extra_dspark_mib = r->ws_extra_dspark_mib;
        c.experts_in_vram_policy = r->experts_in_vram_policy;
        if (c.c64_k8v4) {}  // (the 5060 Ti numbers above are the row's)
    }
}

}  // namespace sm120_cfg_detail

/// The MoE tile size and kernel variants for a batch of M tokens, from the card's table row. `active` is false when no MoE part is on (the shipped
/// dispatch then applies and every other field is the shipped value). STRATA_INK_SM120_MOE_FORCE=<sf4>[,<mx8>] forces a variant name for the A/B runs
/// (base | bk128 | direct | tr128 | tr128s | tr128bk) on every range where it exists.

namespace sm120_cfg_detail {
inline int variant_from_name(const char* s, strata::core::Sm120Tc* out) {
    using strata::core::Sm120Tc;
    if (!s || !*s) return 0;
    if (!std::strcmp(s, "base")) { *out = Sm120Tc::Base; return 1; }
    if (!std::strcmp(s, "bk128")) { *out = Sm120Tc::Bk128; return 1; }
    if (!std::strcmp(s, "direct")) { *out = Sm120Tc::Direct; return 1; }
    if (!std::strcmp(s, "tr128")) { *out = Sm120Tc::Tr128; return 1; }
    if (!std::strcmp(s, "tr128s")) { *out = Sm120Tc::Tr128Small; return 1; }
    if (!std::strcmp(s, "tr128bk")) { *out = Sm120Tc::Tr128Bk128; return 1; }
    return 0;
}
}  // namespace sm120_cfg_detail

inline Sm120MoeSel sm120_moe_sel(const Sm120Cfg& c, int M) noexcept {
    Sm120MoeSel s;
    if (!c.row || !(c.moe_opt || c.tc_smallm)) return s;
    const auto& r = *c.row;
    const auto& g = r.moe[strata::core::ink_sm120_moe_range(r, M)];
    s.active = true;
    s.tile_rows = g.tile_rows;
    s.sf4 = g.sf4;
    s.mx8 = g.mx8;
    s.pf_gu = g.pf_bytes;
    s.pf_dn = (c.l2pf || c.moe_opt) ? g.pf_bytes : 0;
    s.row_quant = sm120::sf4_variant_row_quant((int)g.sf4);
    static const char* force = std::getenv("STRATA_INK_SM120_MOE_FORCE");
    if (force && *force) {
        char a[24] = {}, b[24] = {};
        const char* comma = std::strchr(force, ',');
        std::snprintf(a, sizeof(a), "%.*s", comma ? (int)(comma - force) : 23, force);
        if (comma) std::snprintf(b, sizeof(b), "%s", comma + 1);
        strata::core::Sm120Tc v;
        const bool t128 = (g.tile_rows == 128);
        if (sm120_cfg_detail::variant_from_name(a, &v)) {
            const bool is128 = (v == strata::core::Sm120Tc::Tr128 || v == strata::core::Sm120Tc::Tr128Small || v == strata::core::Sm120Tc::Tr128Bk128);
            if (is128 == t128) { s.sf4 = v; s.row_quant = sm120::sf4_variant_row_quant((int)v); }
        }
        if (sm120_cfg_detail::variant_from_name(b, &v)) {
            const bool is128 = (v == strata::core::Sm120Tc::Tr128);
            if (is128 == t128 && (v == strata::core::Sm120Tc::Base || v == strata::core::Sm120Tc::Bk128 || v == strata::core::Sm120Tc::Tr128)) s.mx8 = v;
        }
    }
    return s;
}

/// The decision for one device ordinal, read once. Safe to call from any thread.
inline const Sm120Cfg& sm120_cfg(int dev) noexcept {
    static Sm120Cfg cfgs[16];
    static std::atomic<int> state[16];   // 0 = not started, 1 = being filled, 2 = ready
    const int d = (dev >= 0 && dev < 16) ? dev : 0;
    if (state[d].load(std::memory_order_acquire) == 2) return cfgs[d];
    int expect = 0;
    if (!state[d].compare_exchange_strong(expect, 1, std::memory_order_acq_rel)) {
        while (state[d].load(std::memory_order_acquire) != 2) {}
        return cfgs[d];
    }
    strata::core::Sm120Dev sd;
    sd.ordinal = d;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, d) == cudaSuccess) {
        sd.name = p.name;
        sd.cc_major = p.major;
        sd.cc_minor = p.minor;
        sd.sm_count = p.multiProcessorCount;
        sd.vram_bytes = (uint64_t)p.totalGlobalMem;
    } else {
        (void)cudaGetLastError();
    }
    const strata::core::Sm120Decision dec = strata::core::ink_sm120_decide(sd, sm120_cfg_detail::env_get);
    sm120_cfg_detail::fill(cfgs[d], dec);
    cfgs[d].ready = true;
    if (!dec.note.empty()) std::fprintf(stderr, "%s\n", dec.note.c_str());
    state[d].store(2, std::memory_order_release);
    return cfgs[d];
}

inline const Sm120Cfg& sm120_cfg_current() noexcept {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void)cudaGetLastError(); dev = 0; }
    return sm120_cfg(dev);
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// 5090 hand-over API (strata::ink::rtx5090::*): thin wrappers, no environment reads.
// ------------------------------------------------------------------------------------------------------------------------------------------
namespace rtx5090 {
inline bool enabled() { return sm120_cfg_current().fast; }
inline bool exact_mode_on() { return sm120_cfg_current().exact; }
inline bool pf_attn_on() { return sm120_cfg_current().pf_attn; }
inline bool moe_opt_on() { return sm120_cfg_current().moe_opt; }
inline bool dense_opt_on() { return sm120_cfg_current().dense_opt; }
inline bool dspark_opt_on() { return sm120_cfg_current().dspark_opt; }
inline bool c64_spec_on() { return sm120_cfg_current().c64_bspec; }
inline bool fp8_trunk_on() { return sm120_cfg_current().fp8_cache; }
inline bool deep_tree_on() { return sm120_cfg_current().deep_tree; }
}  // namespace rtx5090

// ------------------------------------------------------------------------------------------------------------------------------------------
// 5060 Ti hand-over API (InkRtx5060TiCfg / ink_rtx5060ti_cfg): the same struct, filled from the one decision.
// ------------------------------------------------------------------------------------------------------------------------------------------
struct InkRtx5060TiCfg {
    bool enabled         = false;  // any 5060 Ti-family part is on
    bool compact_vram    = false;
    bool compact_xcode   = false;  // compact: MX8 -> SF4 transcode of layers 20-23 (SQ-20 needs it to fit)
    bool compact_kv      = false;  // compact: FP8 KV cache on a one-slot run
    bool compact_hybrid  = false;  // compact trunk with the experts that fit (the rest in RAM) when all of them do not fit
    bool fp8_trunk       = false;
    bool sf4_trunk       = false;
    bool sm120_tc_smallm = false;
    bool pld             = false;
    bool pack            = false;
    bool prefill_lm      = false;
    bool core_graph      = false;
    bool c64_spec        = false;
    bool sm120_pf        = false;
    float pcie_share     = 0.28f;
    int   c64_draft_k    = 4;
    int   c1_beam        = 2;
    int   c1_max_nodes   = 20;
    int   c1_pld_max_m   = 24;
    float c1_spec_cost   = 0.75f;
    float c1_spec_plain  = 0.0f;
};

inline const InkRtx5060TiCfg& ink_rtx5060ti_cfg(int dev = 0) noexcept {
    static InkRtx5060TiCfg cfgs[16];
    static std::atomic<int> state[16];
    const int d = (dev >= 0 && dev < 16) ? dev : 0;
    if (state[d].load(std::memory_order_acquire) == 2) return cfgs[d];
    int expect = 0;
    if (!state[d].compare_exchange_strong(expect, 1, std::memory_order_acq_rel)) {
        while (state[d].load(std::memory_order_acquire) != 2) {}
        return cfgs[d];
    }
    const Sm120Cfg& s = sm120_cfg(d);
    InkRtx5060TiCfg c{};
    c.compact_vram = s.compact;
    c.compact_xcode = s.compact_xcode;
    c.compact_kv = s.compact_kv;
    c.compact_hybrid = s.compact_hybrid;
    c.fp8_trunk = s.fp8_inplace;
    c.sf4_trunk = s.sf4_trunk;
    c.sm120_tc_smallm = s.tc_smallm;
    c.pld = s.pld;
    c.pack = s.pack;
    c.prefill_lm = s.prefill;
    c.core_graph = s.core_graph;
    c.c64_spec = s.c64_k8v4;
    c.sm120_pf = s.l2pf;
    c.enabled = s.compact || s.fp8_inplace || s.tc_smallm || s.pld || s.pack || s.prefill || s.core_graph || s.c64_k8v4 || s.l2pf;
    if (c.enabled) {
        c.pcie_share = s.pcie_share;
        c.c64_draft_k = s.c64_draft_k;
        c.c1_beam = s.c1_beam;
        c.c1_max_nodes = s.c1_max_nodes;
        c.c1_pld_max_m = s.c1_pld_max_m;
        c.c1_spec_cost = s.c1_spec_cost;
        c.c1_spec_plain = s.c1_spec_plain;
    }
    cfgs[d] = c;
    state[d].store(2, std::memory_order_release);
    return cfgs[d];
}

/// The loader found that the card cannot hold COMPACT right now (free VRAM, a model with other expert formats). The rest of the small-card fast set (PACK, PREFILL, ...) was
/// measured only together with COMPACT and is slower than the old path without it, so the whole device goes back to the old path: every flag off, exactly as with
/// STRATA_INK_SM120=0. Called once, single-threaded, at the start of the model load, before any consumer of the flags runs.
/// The slot count of the run that is being loaded (set by the backend before the model load; the loader's COMPACT check reads it). COMPACT leaves about 1 GiB free on a
/// 16 GB card: that holds one slot (or the 16-64 slot host-KV mode) but not 2..15 slots (rows lane 11-26: --parallel 4 on a 5080 died with "not enough free GPU memory for the
/// working buffers"; the engine must then take the old path instead).
inline std::atomic<int>& sm120_plan_slots_ref() noexcept {
    static std::atomic<int> v{1};
    return v;
}

inline void sm120_compact_refuse(int dev) noexcept {
    Sm120Cfg& s = const_cast<Sm120Cfg&>(sm120_cfg(dev));   // static, non-const storage underneath
    s = Sm120Cfg{};
    s.ready = true;
    InkRtx5060TiCfg& c = const_cast<InkRtx5060TiCfg&>(ink_rtx5060ti_cfg(dev));
    c = InkRtx5060TiCfg{};
}

inline bool ink_5060ti_env_flag(const char* name, bool def_val) noexcept {
    const char* e = std::getenv(name);
    if (!e || !*e) return def_val;
    return (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}
inline float ink_5060ti_env_f32(const char* name, float def_val) noexcept {
    const char* e = std::getenv(name);
    if (!e || !*e) return def_val;
    return static_cast<float>(std::atof(e));
}
inline int ink_5060ti_env_i32(const char* name, int def_val) noexcept {
    const char* e = std::getenv(name);
    if (!e || !*e) return def_val;
    return std::atoi(e);
}

}  // namespace strata::ink
