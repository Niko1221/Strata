// engine/src/inkling2/cuda/ink_runner.cuh - Complete 48-Visit Inkling-2-Mini CUDA Runner (B=1..64, Tree-16, Full CUDA Graph, sm_70..sm_121a)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include "ink_attn.cuh"
#include "ink_attn_shared.cuh"   // opt-in STRATA_INK_VERIFY_SHARED_KV: one KV read per verify step
#include "ink_attn_tc.cuh"       // fast-only STRATA_INK_SM120_TC_ATTN: tensor-core split-K decode attention of the global visits
#include "ink_attn_probe.cuh"   // debug only (STRATA_INK_ATTN_PROBE), no effect unless that env var is set
#include "ink_cuda.cuh"
#include "ink_dflash2_path.cuh"
#include "ink_hybrid.cuh"
#include "ink_elastic_kv_ssd.cuh"
#include "ink_rtx5060ti_opt.cuh"
#include "ink_5060ti_fp8.cuh"
#include "ink_kernels.cuh"
#include "ink_manifest.hpp"
#include "ink_moe_cuda.cuh"
#include "ink_rtx5090_kernels.cuh"
#include "ink_rtx5090_fp8.cuh"
#include "ink_rtx5090_bspec_kernels.cuh"
#include "ink_pdl_kernels.cuh"   // PERF-STEP: programmatic dependent launch of the decode graphs
#include "ink_head_screen.cuh"   // PERF-STEP: INT8-screened exact top-128 of the head (STRATA_INK_HEAD_SCREEN)

#include "strata/artifact/strata_reader.hpp"
#include "strata/core/ink_kv_grow.hpp"
#include "strata/core/ink_kv_reserve.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::ink {

// ---- Tree-32 Speculative Verification Descriptor (M <= 32 nodes rooted at anchor 0, topological order parent[i] < i) ----
struct InkTreeSpec {
    int M = 0;
    int32_t tok[kInkMaxTree] = {};
    int8_t  parent[kInkMaxTree] = {};
    uint8_t depth[kInkMaxTree] = {};
    uint8_t anc[kInkMaxTree][kInkMaxTreeDepth] = {};
    uint8_t anc32[kInkMaxTree][kInkMaxTree] = {};
    uint8_t max_depth = 0;
    int8_t  sconv_lag[kInkMaxTree * 3] = {};  // for each node m: lag-1, lag-2, lag-3 indices (>=0 node, -1/-2/-3 history[2/1/0])

    void build() {
        max_depth = 0;
        for (int i = 0; i < M; ++i) {
            if (depth[i] > max_depth) max_depth = depth[i];
            int cur = i;
            for (int d = (int)depth[i]; d >= 0; --d) {
                if (d < kInkMaxTreeDepth) anc[i][d] = (uint8_t)cur;
                if (d < kInkMaxTree)      anc32[i][d] = (uint8_t)cur;
                cur = (cur > 0) ? (int)parent[cur] : -1;
            }
            // ShortConv 3 lags: t-1, t-2, t-3
            int p = (i > 0) ? (int)parent[i] : -1;
            for (int l = 0; l < 3; ++l) {
                if (p >= 0) {
                    sconv_lag[i * 3 + l] = (int8_t)p;
                    p = (p > 0) ? (int)parent[p] : -1;
                } else {
                    // Remaining lags come from committed history: first miss = -1 (hist[2]), next = -2 (hist[1]), next = -3 (hist[0])
                    int miss = l - (int)depth[i];  // 0 -> -1, 1 -> -2, 2 -> -3
                    sconv_lag[i * 3 + l] = (int8_t)(-(miss + 1));
                }
            }
        }
    }
};

// (Exact 2-stage screened argmax kernels live in ink_kernels.cuh)

// ============================================================================
// Host GGUF_RAW -> canonical SF4 (E2M1 + E4M3 block-16) converter
// Used for exotic 1-2 bit GGUF quants (IQ1_S, IQ2_XXS, IQ3_XXS, Q2_0, etc.) or host-spilled GGUF experts!
// ============================================================================
inline uint8_t host_f32_to_e4m3_pos(float v) noexcept {
    if (v <= 0.0f) return 0u;
    if (v >= 448.0f) return 0x7Eu;
    // Exact round-to-nearest positive FP8 E4M3 (exp bias=7, 3 mantissa bits, subnormals exp=0 -> m * 2^-9)
    if (v < (1.0f / 64.0f)) {
        int m = (int)std::lround(v * 512.0f);
        if (m < 0) m = 0;
        if (m > 7) m = 7;
        return (uint8_t)m;
    }
    int exp = 0;
    float frac = std::frexp(v, &exp);  // v = frac * 2^exp, frac in [0.5, 1.0)
    int e_unb = exp - 1;               // v = (frac * 2) * 2^e_unb, mant in [1.0, 2.0)
    int e_bits = e_unb + 7;
    if (e_bits <= 0) {
        int m = (int)std::lround(v * 512.0f);
        return (uint8_t)std::max(0, std::min(7, m));
    }
    int m = (int)std::lround((frac * 2.0f - 1.0f) * 8.0f);
    if (m >= 8) { m = 0; ++e_bits; }
    if (e_bits >= 15) return (e_bits == 15 && m <= 6) ? (uint8_t)((15 << 3) | m) : 0x7Eu;
    return (uint8_t)((e_bits << 3) | (m & 7));
}

inline float host_e4m3_pos_to_f32(uint8_t b) noexcept {
    b &= 0x7Fu;
    int e = (b >> 3) & 0xF;
    int m = b & 7;
    if (e == 0) return (float)m * (1.0f / 512.0f);
    return std::ldexp(1.0f + (float)m * 0.125f, e - 7);
}

inline uint8_t host_f32_to_e2m1_nib(float v) noexcept {
    uint8_t sign = (v < 0.0f) ? 8u : 0u;
    float a = std::fabs(v);
    // Midpoints between {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    uint8_t mag = 0u;
    if      (a < 0.25f) mag = 0u;
    else if (a < 0.75f) mag = 1u;
    else if (a < 1.25f) mag = 2u;
    else if (a < 1.75f) mag = 3u;
    else if (a < 2.50f) mag = 4u;
    else if (a < 3.50f) mag = 5u;
    else if (a < 5.00f) mag = 6u;
    else                mag = 7u;
    return (uint8_t)(sign | mag);
}

inline void host_quant_row_sf4(const float* row, int K, float outer_scale, uint8_t* dst_p, uint8_t* dst_s) noexcept {
    const float inv_outer = (outer_scale > 0.0f) ? (1.0f / outer_scale) : 0.0f;
    const int nblk = K / 16;
    for (int b = 0; b < nblk; ++b) {
        const float* x = row + b * 16;
        float am = 0.0f;
        for (int i = 0; i < 16; ++i) am = std::max(am, std::fabs(x[i]));
        uint8_t sb = host_f32_to_e4m3_pos((am / 6.0f) * inv_outer);
        dst_s[b] = sb;
        const float sc = host_e4m3_pos_to_f32(sb) * outer_scale;
        const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        uint8_t* p = dst_p + b * 8;
        for (int i = 0; i < 8; ++i) {
            uint8_t lo = host_f32_to_e2m1_nib(x[2 * i + 0] * inv);
            uint8_t hi = host_f32_to_e2m1_nib(x[2 * i + 1] * inv);
            p[i] = (uint8_t)((lo & 0xFu) | ((hi & 0xFu) << 4));
        }
    }
}

template <typename DequantFn, typename RowBytesFn>
inline bool repack_gguf_raw_body_to_sf4(
    const uint8_t* raw_body, uint32_t raw_dn_off,
    int D, int F,
    DequantFn&& dequant_row, RowBytesFn&& row_bytes,
    std::vector<uint8_t>& out_sf4, uint32_t& out_dn_off, std::string& err) {
    InkGgufExpertHeader hdr{};
    std::memcpy(&hdr, raw_body, sizeof(hdr));
    const size_t rb_gu = row_bytes(hdr.gu_type, D);
    const size_t rb_dn = row_bytes(hdr.dn_type, F);
    if (rb_gu == 0 || rb_dn == 0) {
        err = "SF4 repack: unsupported GGUF quant type (" + std::to_string(hdr.gu_type) + "/" + std::to_string(hdr.dn_type) + ")";
        return false;
    }

    // Dequant gate [F, D], up [F, D], down [D, F] once into FP32
    std::vector<float> g_buf((size_t)F * D), u_buf((size_t)F * D), d_buf((size_t)D * F);
    const uint8_t* gu_src = raw_body + sizeof(hdr);
    const uint8_t* dn_src = raw_body + raw_dn_off;
    float max_gu = 0.0f, max_dn = 0.0f;
    for (int f = 0; f < F; ++f) {
        float* gr = g_buf.data() + (size_t)f * D;
        float* ur = u_buf.data() + (size_t)f * D;
        if (!dequant_row(hdr.gu_type, gu_src + (size_t)f * rb_gu, gr, D) ||
            !dequant_row(hdr.gu_type, gu_src + (size_t)(F + f) * rb_gu, ur, D)) {
            err = "SF4 repack: gate/up dequant failed";
            return false;
        }
        for (int c = 0; c < D; ++c) {
            gr[c] *= hdr.gate_scale;
            ur[c] *= hdr.up_scale;
            max_gu = std::max(max_gu, std::max(std::fabs(gr[c]), std::fabs(ur[c])));
        }
    }
    for (int r = 0; r < D; ++r) {
        float* dr = d_buf.data() + (size_t)r * F;
        if (!dequant_row(hdr.dn_type, dn_src + (size_t)r * rb_dn, dr, F)) {
            err = "SF4 repack: down dequant failed";
            return false;
        }
        for (int c = 0; c < F; ++c) {
            dr[c] *= hdr.dn_scale;
            max_dn = std::max(max_dn, std::fabs(dr[c]));
        }
    }

    const float outer_gu = std::max(max_gu / (6.0f * 240.0f), 1e-6f);
    const float outer_dn = std::max(max_dn / (6.0f * 240.0f), 1e-6f);
    const size_t w13_p_bytes = (size_t)(2 * F) * (D / 2);
    const size_t w13_s_bytes = (size_t)(2 * F) * (D / 16);
    const size_t w2_p_bytes  = (size_t)D * (F / 2);
    const size_t w2_s_bytes  = (size_t)D * (F / 16);
    out_dn_off = (uint32_t)(16u + w13_p_bytes + w13_s_bytes);
    out_sf4.assign(out_dn_off + w2_p_bytes + w2_s_bytes, 0);

    float sf4_hdr[4] = {outer_gu, 1.0f, outer_dn, 1.0f};
    std::memcpy(out_sf4.data(), sf4_hdr, 16);
    uint8_t* w13_p = out_sf4.data() + 16;
    uint8_t* w13_s = w13_p + w13_p_bytes;
    for (int f = 0; f < F; ++f) {
        host_quant_row_sf4(g_buf.data() + (size_t)f * D, D, outer_gu,
                           w13_p + (size_t)(2 * f + 0) * (D / 2), w13_s + (size_t)(2 * f + 0) * (D / 16));
        host_quant_row_sf4(u_buf.data() + (size_t)f * D, D, outer_gu,
                           w13_p + (size_t)(2 * f + 1) * (D / 2), w13_s + (size_t)(2 * f + 1) * (D / 16));
    }
    uint8_t* w2_p = out_sf4.data() + out_dn_off;
    uint8_t* w2_s = w2_p + w2_p_bytes;
    for (int r = 0; r < D; ++r) {
        host_quant_row_sf4(d_buf.data() + (size_t)r * F, F, outer_dn,
                           w2_p + (size_t)r * (F / 2), w2_s + (size_t)r * (F / 16));
    }
    return true;
}

// ============================================================================
// InkCudaModel: GPU-resident .strata v1.1 / .gguf / HF Trunk + Experts (Single-GPU & Dual-GPU EP!)
// ============================================================================
/// What DFlash 2 really takes on a 16 GB card (RTX 5060 Ti and RTX 5080, 1 slot, the mix): 331 MiB of weights (112 MiB of bf16 tensors + 219 MiB of block weights), 15 MiB of scratch, a 41 MiB context ring
/// for each of the first 8 slots and the driver's share of the kernels' first use (45 MiB): 432 MiB measured ("GPU memory taken 432 MiB"). df2_total_mib() is the planner's rounder number (492).
inline size_t df2_real_bytes(int slots) {
    return (size_t)(112.0 + df2_big_mib(df2_variant(true)) + 42.0 * (double)std::min(std::max(slots, 1), 8) + 60.0) << 20;
}

/// What the runner allocates after the K/V in the COMPACT layout (workspaces: max_batch x 520 KiB + 237 MiB; 367 MiB at max_batch 256 on an RTX 5060 Ti and an RTX 5080, measured) and what it takes between the end of
/// the load and the K/V sizing (234 MiB: 1425 -> 1191 MiB free on the 5060 Ti, 1282 -> 1050 MiB on the 5080).
constexpr size_t kCompactWsConstBytes = 237ull << 20;
constexpr size_t kCompactSizingOverheadBytes = 234ull << 20;
/// Free memory COMPACT must still have at the end of the start with no device K/V (the engine's own "compact keeps 128 MiB free" cut moves K/V to mapped host memory when it has to):
/// the margin of the crash-safety rule (an RTX 5080 ended at 144 MiB free and died; an RTX 5070 Ti at 149 MiB and ran).
constexpr size_t kCompactCrashMarginBytes = 256ull << 20;

struct InkCudaModel {
    InkManifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    int primary_gpu = 0;
    int peer_gpu = -1;
    bool ep2_active = false;
    bool ep2_p2p = false;
    float ep_split = 0.5f;
    // Multi-GPU Pipeline Layer-Split (--layer-split auto | K1,K2,...) across 2..4 GPUs (0 duplicated KV/experts!)
    static constexpr int kMaxStages = 4;
    bool ls_active = false;
    int  n_stages = 1;
    int  stage_gpu[kMaxStages] = {0, -1, -1, -1};
    int  stage_cc[kMaxStages]  = {};
    int  stage_l0[kMaxStages]  = {0, 24, 24, 24};
    int  stage_l1[kMaxStages]  = {24, 24, 24, 24};
    int  layer_owner[24]       = {};
    bool stage_p2p[kMaxStages][kMaxStages] = {};
    bool stage_trunk_attn_fp16[kMaxStages] = {};
    uint8_t* stage_d_trunk[kMaxStages] = {};
    uint8_t* stage_d_sec4[kMaxStages]  = {};
    size_t   stage_sec4_bytes[kMaxStages] = {};
    size_t   stage_n_experts[kMaxStages]  = {};
    std::unordered_map<std::string, size_t> stage_trunk_off[kMaxStages];
    int sm_cc = 0;  // major * 10 + minor (e.g. 60, 61, 70, 75, 86, 89, 120, 121)
    bool trunk_attn_fp16 = false;  // true on sm_70/75: .attn.* BF16 weights converted in-place to FP16 at load!

    uint8_t* d_trunk = nullptr;
    void*    h_embed_raw = nullptr;      // opt-in mapped pinned host RAM for model.llm.embed.weight (0 MiB VRAM!)
    uint8_t* d_embed_mapped = nullptr;   // GPU device pointer to h_embed_raw
    bool     all_sf4_vram = false;       // true when opt-in 100%-VRAM compact mode placed all 3,072 experts as SF4 in GDDR7
    opt5060::Ink5060TiFp8Trunk fp8_trunk; // opt-in in-place FP8_E4M3 attention & controller descriptors inside d_trunk (0 MiB dup!)
    uint8_t* d_sec4  = nullptr;
    uint8_t* d_sec4_peer = nullptr;
    uint8_t* h_sec4_pinned = nullptr;  // optional pinned overflow for 8/12/16 GB GPUs
    size_t trunk_bytes = 0, sec4_bytes = 0, sec4_peer_bytes = 0;
    size_t n_vram_experts = 0, n_peer_experts = 0, n_host_experts = 0;

    // Preprocessed per-layer F32 buffers (tiny: < 40 MB total)
    float* d_msg_queries[12] = {};      // [4, 2304] F32 for physical layers 6..17
    float* d_sconv_k[24] = {};          // [1024, 4] F32 (lag 3..0)
    float* d_sconv_v[24] = {};          // [1024, 4] F32 (lag 3..0)
    float* d_sconv_attn[24] = {};       // [2304, 4] F32
    float* d_sconv_mlp[24] = {};        // [2304, 4] F32

    // Per-physical-layer expert directory on GPU 0 + optional GPU 1 (Dual-GPU EP) + Hybrid Engine
    InkExpertDev* d_experts[24] = {};
    InkExpertDev* d_experts_peer[24] = {};
    InkLayerMoeInfo moe_info[24]{};
    InkLayerMoeInfo moe_info_peer[24]{};
    bool layer_has_host[24] = {};
    bool layer_has_peer[24] = {};
    InkHybridEngine hybrid;

    // Exact 2-stage unembed screening index (for BF16 unembed in SQ-20)
    UnembedScreenIndex unembed_screen;

    const uint8_t* trunk_ptr(const std::string& name) const {
        return d_trunk + man.get(name).off;
    }
    const uint8_t* embed_ptr() const {
        return d_embed_mapped ? d_embed_mapped : (d_trunk + man.get("model.llm.embed.weight").off);
    }
    const uint8_t* trunk_ptr_stage(int s, const std::string& name) const {
        if (!ls_active || s <= 0) return d_trunk + man.get(name).off;
        auto it = stage_trunk_off[s].find(name);
        if (it != stage_trunk_off[s].end()) return stage_d_trunk[s] + it->second;
        return d_trunk + man.get(name).off;
    }
    const uint8_t* sec4_stage(int s) const {
        return (ls_active && s > 0) ? stage_d_sec4[s] : d_sec4;
    }
    const TensorInfo& tinfo(const std::string& name) const {
        return man.get(name);
    }

    static size_t wddm_safety_floor(size_t vtot) noexcept {
#ifdef _WIN32
        if (vtot <= (6ull << 30))  return 640ull << 20;
        if (vtot <= (10ull << 30)) return 896ull << 20;
        return 1152ull << 20;
#else
        return (vtot <= (10ull << 30)) ? (384ull << 20) : (512ull << 20);
#endif
    }

    static size_t safe_host_ram_cap(double ram_experts_gb, bool ssd_experts) noexcept {
        if (ink_elastic_1m_enabled() || ink_elastic_ram_cap_bytes() > 0) {
            return ink_elastic_expert_ram_cap(ram_experts_gb, ssd_experts, ink_ram_available_bytes());
        }
        if (ssd_experts || ram_experts_gb == 0.0) return 0u;
        if (ram_experts_gb > 0.0) return static_cast<size_t>(ram_experts_gb * 1073741824.0);
        const uint64_t avail = ink_ram_available_bytes();
        if (avail == 0) return static_cast<size_t>(-1);
        const uint64_t os_res = std::max<uint64_t>(2ull << 30, std::min<uint64_t>(avail / 4, 4ull << 30));
        return (avail > os_res) ? static_cast<size_t>(avail - os_res) : 0u;
    }

    bool load(const std::string& strata_path, int gpu_id, bool build_screen, std::string& err,
              size_t extra_reserve_bytes = 0, double vram_experts_gb = -1.0,
              double ram_experts_gb = -1.0, bool ssd_experts = false, int cpu_threads = 0) {
        path = strata_path;
        primary_gpu = gpu_id;
        INK_CK(cudaSetDevice(primary_gpu));
        {
            cudaDeviceProp prop{};
            INK_CK(cudaGetDeviceProperties(&prop, primary_gpu));
            sm_cc = prop.major * 10 + prop.minor;
        }
        if (!rd.open(strata_path, err)) return false;
        parse_ink_manifest(rd.manifest_json(), man, true);

        const auto& hdr = rd.header();
        trunk_bytes = (size_t)hdr.sections[1].byte_length;
        sec4_bytes  = (size_t)hdr.sections[4].byte_length;

        double t0 = now_s();
        const auto& opt5060_ld = ink_rtx5060ti_cfg(primary_gpu);
        size_t vfree_pre = 0, vtot_pre = 0;
        INK_CK(ink_cuda_mem_info(&vfree_pre, &vtot_pre));
        bool do_compact_vram = opt5060_ld.compact_vram && !ssd_experts && vram_experts_gb <= 0.0 &&
                                     ram_experts_gb != 0.0 && sm_cc >= 80 &&
                                     ink_elastic_ram_cap_bytes() == 0 && vtot_pre <= ((size_t)std::max(1, sm120_cfg(primary_gpu).compact_max_vram_gib) << 30);   // the table row says how big a card "compact" is tried on
        if (do_compact_vram) {
            // COMPACT is the default on the 16 GB 5060 Ti, so it must refuse BEFORE it converts anything when the card cannot hold it (a desktop or another program on
            // the card, the Windows WDDM reserve, a smaller free size): the plain path then runs exactly as with COMPACT off. Size of the compact trunk and of the
            // all-SF4 expert section are computed here from the manifest and the directory, without touching the card.
            size_t est_trunk = 0, est_sec4 = 0;
            bool all_sf4_or_mx8_pre = true;
            const size_t sf4_dn_off_pre = 16u + (size_t)(2 * kInkInter) * (kInkHidden / 2) + (size_t)(2 * kInkInter) * (kInkHidden / 16);
            const size_t sf4_body_pre   = sf4_dn_off_pre + (size_t)kInkHidden * (kInkInter / 2) + (size_t)kInkHidden * (kInkInter / 16);
            for (const auto& e : rd.expert_directory()) {
                if (e.gu_code_id == FMT_SF4 && e.dn_code_id == FMT_SF4) est_sec4 += e.body_bytes;
                else if (e.gu_code_id == FMT_MX8 && e.dn_code_id == FMT_MX8) est_sec4 += sf4_body_pre;
                else { all_sf4_or_mx8_pre = false; break; }
            }
            auto est_map = [&](const std::unordered_map<std::string, TensorInfo>& tmap) {
                for (const auto& kv : tmap) {
                    const std::string& nm = kv.first;
                    const TensorInfo& ti = kv.second;
                    if (nm == "model.llm.embed.weight") continue;   // mapped host memory
                    const bool is_unemb = (nm == "model.llm.unembed.weight" && ti.fmt == FMT_BF16);
                    const bool is_attn_bf16 = (ti.fmt == FMT_BF16 && ti.shape.size() == 2 && nm.find(".attn.") != std::string::npos && ti.bytes > 100000);
                    const bool is_dspark_bf16 = (ti.fmt == FMT_BF16 && ti.shape.size() == 2 && nm.rfind("dspark.layers.", 0) == 0 && ti.bytes > 100000);
                    const bool is_mw1_bf16 = (nm == "dspark.markov_head.markov_w1.weight" && ti.fmt == FMT_BF16 && ti.shape.size() == 2 && (int)ti.rows() == kInkVocabPad);
                    const bool is_ctrl_f32 = (ti.fmt == FMT_F32 && ti.shape.size() == 2 && nm.find(".controllers.") != std::string::npos);
                    est_trunk = (est_trunk + 255ull) & ~255ull;
                    if (is_unemb || is_attn_bf16 || is_dspark_bf16 || is_mw1_bf16 || is_ctrl_f32) {
                        const int K = (int)ti.cols();
                        const int N_src = is_unemb ? std::min((int)ti.rows(), man.c.vocab_unpadded) : (int)ti.rows();
                        const int N_pad = is_unemb ? ((man.c.vocab_unpadded + 3) & ~3) : N_src;
                        est_trunk += (size_t)N_pad * 4u + (size_t)N_pad * K;
                    } else {
                        est_trunk += ti.bytes;
                    }
                }
            };
            est_map(man.trunk);
            if (df2_dspark_selected()) est_map(man.dspark);   // DSpark's weights are only loaded when STRATA_INK_DRAFTER=dspark asks for them (add_map above): DFlash 2 is the drafter
            const int plan_slots_pre = sm120_plan_slots_ref().load();
            const bool df2_pre = df2_replaces_dspark(strata_path);
            const size_t df2_pre_bytes = df2_pre ? df2_real_bytes(plan_slots_pre) : 0u;
            // the loader holds a 64 MiB staging buffer and the amax word next to the trunk while it converts, then the all-VRAM branch below asks for 768 MiB after the experts (ballast tests: with 713 MiB left the engine died at its first top-k, with 39-43 MiB left at the end it was on the edge: lazy module loads need free VRAM)
            {
                const int plan_slots = sm120_plan_slots_ref().load();
                if (plan_slots >= 2 && plan_slots < 16 && !opt5060_ld.compact_hybrid) {
                    std::fprintf(stderr, "[Ink5060Ti] COMPACT refused: this run asks for %d slots and compact leaves room for one slot (or 16 and more); the old path (every RTX 50 part off) runs\n", plan_slots);
                    do_compact_vram = false;
                    sm120_compact_refuse(primary_gpu);
                }
            }
            const size_t compact_margin = (768ull << 20) + (64ull << 20) + (plan_slots_pre == 1 ? df2_pre_bytes : 0u);   // + DFlash 2's real total for 1 slot (the precheck did not count it); 16 and more slots: the rings that fit
            std::fprintf(stderr, "[Ink5060Ti] COMPACT check: free %zu MiB, trunk %zu MiB + experts %zu MiB + headroom %zu MiB = %zu MiB needed\n",
                         vfree_pre >> 20, est_trunk >> 20, est_sec4 >> 20, compact_margin >> 20, (est_trunk + est_sec4 + compact_margin) >> 20);
            const bool hybrid_going = do_compact_vram && opt5060_ld.compact_hybrid && (!all_sf4_or_mx8_pre || vfree_pre < est_trunk + est_sec4 + compact_margin);
            if (hybrid_going) {
                // COMPACT_HYBRID: the experts do not all fit (8 / 12 GB card): build the compact trunk anyway (it is 2.3 GiB smaller than the BF16 trunk) and let the normal hybrid
                // placement (adaptive expert tier, hottest experts first, looped layers weighted by their visits) fill what is left of the card; the rest stay in RAM.
                std::fprintf(stderr, "[Ink5060Ti] COMPACT_HYBRID: not all experts fit (free %zu MiB, all-VRAM needs about %zu MiB): compact trunk %zu MiB, the experts that fit go to the card, the rest stay in RAM\n",
                             vfree_pre >> 20, (est_trunk + est_sec4 + compact_margin) >> 20, est_trunk >> 20);
            } else if (do_compact_vram && (!all_sf4_or_mx8_pre || vfree_pre < est_trunk + est_sec4 + compact_margin)) {
                std::fprintf(stderr, "[Ink5060Ti] COMPACT refused: %s (free %zu MiB, needs about %zu MiB for the compact trunk %zu MiB + all experts in VRAM %zu MiB + %zu MiB headroom); the old path (every RTX 50 part off) runs\n",
                             all_sf4_or_mx8_pre ? "not enough free VRAM" : "the model has experts that are not SF4 / MX8",
                             vfree_pre >> 20, (est_trunk + est_sec4 + compact_margin) >> 20, est_trunk >> 20, est_sec4 >> 20, compact_margin >> 20);
                do_compact_vram = false;
                sm120_compact_refuse(primary_gpu);
            }
            // The crash margin (1 slot; 16 and more slots keep the SWA ring and the K/V on the host): what the runner and DFlash 2 really take after the load must leave at least 256 MiB free at the end
            // of the start, with no device K/V (the K/V is the elastic part: the sizing moves it to mapped host memory). RTX 5080 (15,558 MiB free): 147 MiB predicted, 144 measured, refused;
            // RTX 5060 Ti (15,698 MiB free): 290 predicted, 285 measured, runs.
            if (do_compact_vram && plan_slots_pre == 1 && !hybrid_going) {   // COMPACT_HYBRID has no all-VRAM end state: the hybrid loader keeps its own reserve (the estimate would refuse every 8 / 12 GB card)
                const size_t per_visit = (size_t)kInkKVDim * 2u + (size_t)(2 * kInkKVHeads * 4);   // the FP8 K/V of COMPACT (InkCudaRunner::kv_bytes_per_visit_pos)
                int n_swa = 0, n_glob = 0;
                for (int v = 0; v < kInkVisits; ++v) {
                    if (man.c.is_swa[ink_visit_phys(v)]) ++n_swa; else ++n_glob;
                }
                const int mb = 256;   // the compact prefill chunk (max_batch)
                const size_t swa_b = (size_t)n_swa * (kInkSwaWin + mb) * per_visit;
                const size_t sconv_b = (size_t)(n_swa + n_glob) * 3 * kInkSconvCh * 4;
                const size_t ws_b = (size_t)mb * (520ull << 10) + kCompactWsConstBytes;
                const size_t used = kCompactSizingOverheadBytes + swa_b + sconv_b + ws_b + df2_pre_bytes;
                const size_t after_load = vfree_pre > est_trunk + est_sec4 ? vfree_pre - est_trunk - est_sec4 : 0u;
                const size_t end_free = after_load > used ? after_load - used : 0u;
                std::fprintf(stderr, "[Ink5060Ti] COMPACT end-of-start estimate: free after the load about %zu MiB - runner %zu MiB (swa %.0f, sconv %.0f, workspaces %.0f, sizing %zu) - DFlash 2 %zu MiB = %zu MiB free with no device K/V, crash margin %zu MiB\n",
                             after_load >> 20, used >> 20, (double)swa_b / 1048576.0, (double)sconv_b / 1048576.0, (double)ws_b / 1048576.0, kCompactSizingOverheadBytes >> 20, df2_pre_bytes >> 20, end_free >> 20,
                             kCompactCrashMarginBytes >> 20);
                if (end_free < kCompactCrashMarginBytes) {
                    std::fprintf(stderr, "[Ink5060Ti] COMPACT refused: it would end the start with about %zu MiB free (crash margin %zu MiB; free now %zu MiB, DFlash 2 %zu MiB); the old path (every RTX 50 part off) runs\n",
                                 end_free >> 20, kCompactCrashMarginBytes >> 20, vfree_pre >> 20, df2_pre_bytes >> 20);
                    do_compact_vram = false;
                    sm120_compact_refuse(primary_gpu);
                }
            }
        }
        if (do_compact_vram) {
            // Opt-in RTX 5060 Ti 16 GB 100%-VRAM Compact Trunk Loader:
            //   - embed.weight (883 MiB BF16)       -> pinned mapped host RAM (0 MiB VRAM!)
            //   - unembed.weight (883 MiB BF16)     -> single Q8_ROW padded to V_4 (442 MiB, shared w/ unembed_screen!)
            //   - .attn.* BF16 weights (1228 MiB)   -> Q8_ROW (615 MiB)
            //   - dspark markov_w1 (107 MiB BF16)   -> Q8_ROW (53 MiB)
            // Total compact trunk = ~1.47 GiB (saves ~2.35 GiB -> fits ALL 3,072/3,072 experts in 16 GB GDDR7!)
            std::vector<uint8_t> h_sec1(trunk_bytes);
            {
                const size_t CH = 64ull << 20;
                size_t done = 0;
                while (done < trunk_bytes) {
                    const size_t want = std::min(CH, trunk_bytes - done);
                    if (!rd.read_at(hdr.sections[1].file_offset + done, h_sec1.data() + done, want, err)) return false;
                    done += want;
                }
            }
            struct PackJob {
                TensorInfo* ti;
                size_t src_off;
                size_t src_bytes;
                bool to_q8;
                bool to_sf4;
                bool src_is_f32;
                int n_src;
                int n_pad;
                int cols;
                bool is_embed;
            };
            std::vector<PackJob> jobs;
            jobs.reserve(man.trunk.size() + man.dspark.size());
            size_t new_trunk_bytes = 0;
            const bool want_fp8_trunk = ink_rtx5060ti_cfg(primary_gpu).fp8_trunk && sm_cc >= 120;
            const bool want_sf4_trunk = !want_fp8_trunk && ink_rtx5060ti_cfg(primary_gpu).sf4_trunk && sm_cc >= 120;
            static constexpr const char* kAttnFp8Names[6] = {
                "wq_du.weight", "wr_du.weight", "gate_attn.weight", "wk_dv.weight", "wv_dv.weight", "wo_ud.weight"
            };
            static constexpr const char* kCtrlFp8Kinds[4] = { "attn", "k", "v", "mlp" };
            struct Fp8AttnSrc { size_t src_off; size_t dst_off; size_t elems; };
            struct Fp8CtrlSrc { size_t dn_src_off; size_t g_src_off; size_t f_src_off; size_t dn_dst_off; size_t gf_dst_off; };
            Fp8AttnSrc fp8_attn_src[24][6] = {};
            Fp8CtrlSrc fp8_ctrl_src[12][4] = {};
            bool fp8_layout_ok = want_fp8_trunk;
            std::unordered_map<std::string, bool> fp8_claimed;
            if (want_fp8_trunk) {
                for (int ip = 0; ip < man.c.n_phys && fp8_layout_ok; ++ip) {
                    const std::string lp = "model.llm.layers." + std::to_string(ip) + ".attn.";
                    for (int k = 0; k < 6; ++k) {
                        const std::string nm = lp + kAttnFp8Names[k];
                        auto it = man.trunk.find(nm);
                        if (it == man.trunk.end() || it->second.fmt != FMT_BF16 || it->second.shape.size() != 2) {
                            fp8_layout_ok = false; break;
                        }
                        fp8_claimed[nm] = true;
                        fp8_attn_src[ip][k].src_off = it->second.off;
                        fp8_attn_src[ip][k].elems   = (size_t)it->second.rows() * (size_t)it->second.cols();
                    }
                }
                for (int ci = 0; ci < 12 && fp8_layout_ok; ++ci) {
                    const int ip = man.c.core_start + ci;
                    for (int ck = 0; ck < 4; ++ck) {
                        const std::string cp = "model.llm.layers." + std::to_string(ip) + ".controllers." + kCtrlFp8Kinds[ck] + ".";
                        auto it_d = man.trunk.find(cp + "down_weight");
                        auto it_g = man.trunk.find(cp + "gain_weight");
                        auto it_f = man.trunk.find(cp + "forget_weight");
                        if (it_d == man.trunk.end() || it_g == man.trunk.end() || it_f == man.trunk.end() ||
                            it_d->second.fmt != FMT_F32 || it_g->second.fmt != FMT_F32 || it_f->second.fmt != FMT_F32) {
                            fp8_layout_ok = false; break;
                        }
                        fp8_claimed[cp + "down_weight"] = true;
                        fp8_claimed[cp + "gain_weight"] = true;
                        fp8_claimed[cp + "forget_weight"] = true;
                        fp8_ctrl_src[ci][ck].dn_src_off = it_d->second.off;
                        fp8_ctrl_src[ci][ck].g_src_off  = it_g->second.off;
                        fp8_ctrl_src[ci][ck].f_src_off  = it_f->second.off;
                    }
                }
                if (!fp8_layout_ok) fp8_claimed.clear();
            }
            auto add_map = [&](std::unordered_map<std::string, TensorInfo>& tmap) {
                for (auto& kv : tmap) {
                    const std::string& nm = kv.first;
                    if (fp8_layout_ok && fp8_claimed.count(nm)) continue;
                    TensorInfo& ti = kv.second;
                    PackJob j{};
                    j.ti = &ti;
                    j.src_off = ti.off;
                    j.src_bytes = ti.bytes;
                    if (nm == "model.llm.embed.weight") {
                        j.is_embed = true;
                        jobs.push_back(j);
                        continue;
                    }
                    const bool is_unemb = (nm == "model.llm.unembed.weight" && ti.fmt == FMT_BF16);
                    const bool is_attn_bf16 = (ti.fmt == FMT_BF16 && ti.shape.size() == 2 &&
                                               nm.find(".attn.") != std::string::npos && ti.bytes > 100000);
                    const bool is_dspark_bf16 = (ti.fmt == FMT_BF16 && ti.shape.size() == 2 &&
                                                 nm.rfind("dspark.layers.", 0) == 0 && ti.bytes > 100000);
                    const bool is_mw1_bf16 = (nm == "dspark.markov_head.markov_w1.weight" && ti.fmt == FMT_BF16 &&
                                              ti.shape.size() == 2 && (int)ti.rows() == kInkVocabPad);
                    const bool is_ctrl_f32 = (ti.fmt == FMT_F32 && ti.shape.size() == 2 &&
                                              nm.find(".controllers.") != std::string::npos);
                    const bool is_attn_sf4 = want_sf4_trunk && is_attn_bf16 &&
                                             ((ti.rows() % 128) == 0) && ((ti.cols() % 64) == 0);
                    new_trunk_bytes = (new_trunk_bytes + 255ull) & ~255ull;
                    if (is_attn_sf4) {
                        const int N = (int)ti.rows(), K = (int)ti.cols();
                        const size_t sf4_b = 16u + (size_t)N * (K >> 1) + (size_t)N * (K >> 4);
                        j.to_sf4 = true; j.n_src = N; j.n_pad = N; j.cols = K;
                        ti.off = new_trunk_bytes;
                        ti.fmt = FMT_SF4;
                        ti.bytes = sf4_b;
                        new_trunk_bytes += sf4_b;
                    } else if (is_unemb || is_attn_bf16 || is_dspark_bf16 || is_mw1_bf16 || is_ctrl_f32) {
                        const int K = (int)ti.cols();
                        const int N_src = is_unemb ? std::min((int)ti.rows(), man.c.vocab_unpadded) : (int)ti.rows();
                        const int N_pad = is_unemb ? ((man.c.vocab_unpadded + 3) & ~3) : N_src;
                        const size_t q8_b = (size_t)N_pad * 4u + (size_t)N_pad * K;
                        j.to_q8 = true; j.src_is_f32 = is_ctrl_f32; j.n_src = N_src; j.n_pad = N_pad; j.cols = K;
                        ti.off = new_trunk_bytes;
                        ti.fmt = FMT_Q8ROW;
                        ti.bytes = q8_b;
                        if (is_unemb && !ti.shape.empty()) ti.shape[0] = (int64_t)N_pad;
                        new_trunk_bytes += q8_b;
                    } else {
                        ti.off = new_trunk_bytes;
                        new_trunk_bytes += ti.bytes;
                    }
                    jobs.push_back(j);
                }
            };
            add_map(man.trunk);
            if (df2_dspark_selected()) add_map(man.dspark);   // DFlash 2 is the only drafter (owner, 2026-11-26): DSpark's weights stay off the card unless STRATA_INK_DRAFTER=dspark
            if (fp8_layout_ok) {
                new_trunk_bytes = (new_trunk_bytes + 255ull) & ~255ull;
                for (int ip = 0; ip < man.c.n_phys; ++ip) {
                    // Strictly contiguous back-to-back layout [wq, wr, wgate, wk, wv, wo] (each is multiple of 256 B!)
                    for (int k = 0; k < 6; ++k) {
                        fp8_attn_src[ip][k].dst_off = new_trunk_bytes;
                        new_trunk_bytes += fp8_attn_src[ip][k].elems;
                    }
                }
                constexpr size_t kCtrlDnBytes = (size_t)kInkGateRank * (2 * kInkHidden);
                constexpr size_t kCtrlGfBytes = (size_t)(2 * kInkHidden) * kInkGateRank;
                for (int ci = 0; ci < 12; ++ci) {
                    for (int ck = 0; ck < 4; ++ck) {
                        new_trunk_bytes = (new_trunk_bytes + 255ull) & ~255ull;
                        fp8_ctrl_src[ci][ck].dn_dst_off = new_trunk_bytes;
                        new_trunk_bytes += kCtrlDnBytes;
                        new_trunk_bytes = (new_trunk_bytes + 255ull) & ~255ull;
                        fp8_ctrl_src[ci][ck].gf_dst_off = new_trunk_bytes;
                        new_trunk_bytes += kCtrlGfBytes;
                    }
                }
            }
            new_trunk_bytes = (new_trunk_bytes + 255ull) & ~255ull;
            trunk_bytes = new_trunk_bytes;
            INK_CK(cudaMalloc(&d_trunk, trunk_bytes));
            constexpr size_t kQ8StageBytes = 64ull << 20;
            uint16_t* d_bf16_tmp = nullptr;
            uint32_t* d_sf4_amax = nullptr;
            INK_CK(cudaMalloc(&d_bf16_tmp, kQ8StageBytes));
            INK_CK(cudaMalloc(&d_sf4_amax, 4));
            if (fp8_layout_ok) {
                fp8_trunk.d_sc_pool = dalloc<float>(24 * 6 + 12 * 8);
                int sc_idx = 0;
                float* d_am_f32 = reinterpret_cast<float*>(d_sf4_amax);
                for (int ip = 0; ip < man.c.n_phys; ++ip) {
                    for (int k = 0; k < 6; ++k) {
                        const auto& s = fp8_attn_src[ip][k];
                        uint8_t* dst = d_trunk + s.dst_off;
                        float* sc_p  = fp8_trunk.d_sc_pool + sc_idx++;
                        INK_CK(cudaMemcpy(d_bf16_tmp, h_sec1.data() + s.src_off, s.elems * 2u, cudaMemcpyHostToDevice));
                        INK_CK(cudaMemset(d_am_f32, 0, 4));
                        const unsigned g = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (s.elems / 2 + 255) / 256));
                        opt5060::k_fp8_absmax_bf16<<<g, 256>>>(reinterpret_cast<const __nv_bfloat16*>(d_bf16_tmp), s.elems, d_am_f32);
                        opt5060::k_fp8_finalize_scale<<<1, 1>>>(d_am_f32, sc_p);
                        opt5060::k_fp8_quant_bf16<<<g, 256>>>(reinterpret_cast<const __nv_bfloat16*>(d_bf16_tmp), s.elems, sc_p, dst);
                        fp8_trunk.w_attn[ip][k]  = dst;
                        fp8_trunk.sc_attn[ip][k] = sc_p;
                    }
                }
                constexpr size_t kCtrlDnN = (size_t)kInkGateRank * (2 * kInkHidden);
                constexpr size_t kCtrlHalfN = (size_t)kInkHidden * kInkGateRank;
                constexpr size_t kCtrlGfN = 2 * kCtrlHalfN;
                for (int ci = 0; ci < 12; ++ci) {
                    for (int ck = 0; ck < 4; ++ck) {
                        const auto& cs = fp8_ctrl_src[ci][ck];
                        // down_weight [64, 4608] F32 -> FP8
                        uint8_t* dst_dn = d_trunk + cs.dn_dst_off;
                        float* sc_dn    = fp8_trunk.d_sc_pool + sc_idx++;
                        INK_CK(cudaMemcpy(d_bf16_tmp, h_sec1.data() + cs.dn_src_off, kCtrlDnN * 4u, cudaMemcpyHostToDevice));
                        INK_CK(cudaMemset(d_am_f32, 0, 4));
                        const unsigned gd = (unsigned)std::min<size_t>(512, (kCtrlDnN / 4 + 255) / 256);
                        opt5060::k_fp8_absmax_f32<<<gd, 256>>>(reinterpret_cast<const float*>(d_bf16_tmp), kCtrlDnN, d_am_f32);
                        opt5060::k_fp8_quant_f32_padded<<<gd, 256>>>(reinterpret_cast<const float*>(d_bf16_tmp), kCtrlDnN, kCtrlDnN, d_am_f32, sc_dn, dst_dn);
                        fp8_trunk.w_ctrl_dn[ci][ck]  = dst_dn;
                        fp8_trunk.sc_ctrl_dn[ci][ck] = sc_dn;
                        // fused [gain_weight; forget_weight] [4608, 64] F32 -> FP8
                        uint8_t* dst_gf = d_trunk + cs.gf_dst_off;
                        float* sc_gf    = fp8_trunk.d_sc_pool + sc_idx++;
                        INK_CK(cudaMemcpy(d_bf16_tmp, h_sec1.data() + cs.g_src_off, kCtrlHalfN * 4u, cudaMemcpyHostToDevice));
                        INK_CK(cudaMemcpy(reinterpret_cast<float*>(d_bf16_tmp) + kCtrlHalfN, h_sec1.data() + cs.f_src_off, kCtrlHalfN * 4u, cudaMemcpyHostToDevice));
                        INK_CK(cudaMemset(d_am_f32, 0, 4));
                        const unsigned ggf = (unsigned)std::min<size_t>(512, (kCtrlGfN / 4 + 255) / 256);
                        opt5060::k_fp8_absmax_f32<<<ggf, 256>>>(reinterpret_cast<const float*>(d_bf16_tmp), kCtrlGfN, d_am_f32);
                        opt5060::k_fp8_quant_f32_padded<<<ggf, 256>>>(reinterpret_cast<const float*>(d_bf16_tmp), kCtrlGfN, kCtrlGfN, d_am_f32, sc_gf, dst_gf);
                        fp8_trunk.w_ctrl_gf[ci][ck]  = dst_gf;
                        fp8_trunk.sc_ctrl_gf[ci][ck] = sc_gf;
                    }
                }
                fp8_trunk.ready = true;
            }
            for (const auto& j : jobs) {
                if (j.is_embed) {
                    INK_CK(cudaHostAlloc(&h_embed_raw, j.src_bytes, cudaHostAllocMapped));
                    std::memcpy(h_embed_raw, h_sec1.data() + j.src_off, j.src_bytes);
                    void* dp = nullptr;
                    INK_CK(cudaHostGetDevicePointer(&dp, h_embed_raw, 0));
                    d_embed_mapped = static_cast<uint8_t*>(dp);
                    j.ti->off = 0;
                    continue;
                }
                if (j.to_sf4) {
                    INK_CK(cudaMemcpy(d_bf16_tmp, h_sec1.data() + j.src_off, j.src_bytes, cudaMemcpyHostToDevice));
                    INK_CK(cudaMemset(d_sf4_amax, 0, 4));
                    k_ink_bf16_amax<<<256, 256>>>(d_bf16_tmp, (int64_t)j.n_src * j.cols, d_sf4_amax);
                    k_ink_bf16_to_sf4<<<j.n_src, 256>>>(d_bf16_tmp, d_sf4_amax, d_trunk + j.ti->off, j.n_src, j.cols);
                    continue;
                }
                if (!j.to_q8) {
                    INK_CK(cudaMemcpy(d_trunk + j.ti->off, h_sec1.data() + j.src_off, j.src_bytes, cudaMemcpyHostToDevice));
                    continue;
                }
                float*  dst_sc = reinterpret_cast<float*>(d_trunk + j.ti->off);
                int8_t* dst_q  = reinterpret_cast<int8_t*>(d_trunk + j.ti->off + (size_t)j.n_pad * 4u);
                const size_t elem_b = j.src_is_f32 ? 4u : 2u;
                const int max_stripe = std::max(1, (int)(kQ8StageBytes / ((size_t)j.cols * elem_b)));
                for (int r0 = 0; r0 < j.n_pad; r0 += max_stripe) {
                    const int rc_pad = std::min(max_stripe, j.n_pad - r0);
                    const int rc_src = std::max(0, std::min(rc_pad, j.n_src - r0));
                    if (rc_src > 0) {
                        INK_CK(cudaMemcpy(d_bf16_tmp,
                                          h_sec1.data() + j.src_off + (size_t)r0 * j.cols * elem_b,
                                          (size_t)rc_src * j.cols * elem_b, cudaMemcpyHostToDevice));
                    }
                    if (j.src_is_f32) {
                        k_ink_f32_to_q8row<<<rc_pad, 256>>>(
                            reinterpret_cast<const float*>(d_bf16_tmp),
                            dst_sc + r0, dst_q + (size_t)r0 * j.cols, rc_src, rc_pad, j.cols);
                    } else {
                        k_ink_bf16_to_q8row<<<rc_pad, 256>>>(
                            d_bf16_tmp, dst_sc + r0, dst_q + (size_t)r0 * j.cols, rc_src, rc_pad, j.cols);
                    }
                }
            }
            INK_CK(cudaDeviceSynchronize());
            INK_CK(cudaFree(d_sf4_amax));
            INK_CK(cudaFree(d_bf16_tmp));
            h_sec1.clear();
            h_sec1.shrink_to_fit();
            // Share Q8_ROW unembed directly as unembed_screen (0 MiB duplicate allocation!)
            const auto& ui_c = man.get("model.llm.unembed.weight");
            if (ui_c.fmt == FMT_Q8ROW) {
                unembed_screen.V        = man.c.vocab_unpadded;
                unembed_screen.V_pad    = (int)ui_c.rows();
                unembed_screen.sc       = reinterpret_cast<float*>(d_trunk + ui_c.off);
                unembed_screen.q8       = reinterpret_cast<int8_t*>(d_trunk + ui_c.off + (size_t)ui_c.rows() * 4u);
                unembed_screen.eps      = nullptr;
                unembed_screen.borrowed = true;
                unembed_screen.ready    = true;
            }
        } else {
            // 1. Load Section 1 (Contiguous GPU Trunk + DSpark) in one DMA transfer
            INK_CK(cudaMalloc(&d_trunk, trunk_bytes));
            {
                const size_t CH = 64ull << 20;
                uint8_t* hbuf = halloc_pinned<uint8_t>(std::min(CH, trunk_bytes));
                size_t done = 0;
                while (done < trunk_bytes) {
                    size_t want = std::min(CH, trunk_bytes - done);
                    if (!rd.read_at(hdr.sections[1].file_offset + done, hbuf, want, err)) {
                        cudaFreeHost(hbuf);
                        return false;
                    }
                    INK_CK(cudaMemcpy(d_trunk + done, hbuf, want, cudaMemcpyHostToDevice));
                    done += want;
                }
                INK_CK(cudaFreeHost(hbuf));
            }
        }
        double t_trunk = now_s() - t0;

        // 1B. On Volta sm_70 (V100) & Turing sm_75 (2080 Ti), convert .attn.* BF16 weights IN-PLACE to IEEE FP16 once!
        //     Eliminates 336+ per-visit conversion launches & unlocks 125 TFLOPS direct FP16 Tensor Core GEMMs!
        trunk_attn_fp16 = (sm_cc >= 70 && sm_cc < 80);
        if (trunk_attn_fp16) {
            for (const auto& kv : man.trunk) {
                if (kv.second.fmt == FMT_BF16 && kv.first.find(".attn.") != std::string::npos) {
                    const size_t nw = (size_t)kv.second.rows() * kv.second.cols();
                    __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(d_trunk + kv.second.off);
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                }
            }
            INK_CK(cudaDeviceSynchronize());
        }

        // 2. Check free VRAM -> load Section 4 experts (100% VRAM when fits, else 3x loop-weighted vram_load_order!)
        size_t vfree = 0, vtot = 0;
        INK_CK(ink_cuda_mem_info(&vfree, &vtot));
        const auto& dir = rd.expert_directory();
        const int n_total = (int)dir.size();  // 24 * 128 = 3072
        std::vector<InkExpertDev> h_exp(n_total);
        for (int il = 0; il < man.c.n_phys; ++il) {
            moe_info[il] = {};
            for (int r = 0; r < man.c.n_experts; ++r) {
                const auto& e = dir[(size_t)il * man.c.n_experts + r];
                InkExpertDev dev{};
                dev.body_off = e.body_offset;
                dev.dn_off   = e.dn_offset;
                dev.gu_code  = e.gu_code_id;
                dev.dn_code  = e.dn_code_id;
                dev.tier     = 0;
                h_exp[(size_t)il * man.c.n_experts + e.expert_id] = dev;
                if ((e.gu_code_id == FMT_SF4 || e.gu_code_id == FMT_MX8) && e.dn_code_id != e.gu_code_id) moe_info[il].mixed_gu_dn = true;
                if (e.gu_code_id == FMT_SF4) moe_info[il].has_sf4 = true;
                else if (e.gu_code_id == FMT_MX8) moe_info[il].has_mx8 = true;
                else {
                    moe_info[il].has_sk = true;
                    if (e.gu_code_id < 32) moe_info[il].sk_gu_mask |= (1u << e.gu_code_id);
                }
                if (e.dn_code_id != FMT_SF4 && e.dn_code_id != FMT_MX8 && e.dn_code_id < 32) {
                    moe_info[il].has_sk = true;
                    moe_info[il].sk_dn_mask |= (1u << e.dn_code_id);
                }
            }
        }

        double t1 = now_s();
        {   // which codecs the experts use (the CPU path's speed depends on it): gate_up code / down code -> count, body bytes
            std::map<int, std::pair<int, size_t>> hist;
            for (const auto& e : dir) { auto& h = hist[(int)e.gu_code_id * 100 + (int)e.dn_code_id]; ++h.first; h.second += e.body_bytes; }
            std::fprintf(stderr, "[InkCUDA] expert codecs (gate_up/down code: count, mean MB):");
            for (const auto& kv : hist) std::fprintf(stderr, " %d/%d: %d, %.2f;", kv.first / 100, kv.first % 100, kv.second.first, (double)kv.second.second / kv.second.first / 1e6);
            std::fprintf(stderr, "\n");
        }
        // Guaranteed headroom BEFORE sizing VRAM experts: KV init + O(1) prefill ws (144 MiB) + DSpark + runner ws + OS/WDDM floor!
        const size_t wddm_floor = wddm_safety_floor(vtot);
        const size_t base_reserve = ((vtot >= (20ull << 30)) ? (1600ull << 20)
                                   : (vtot <= (13ull << 30)) ? (960ull << 20)
                                                             : (1280ull << 20)) + (wddm_floor > (512ull << 20) ? (wddm_floor - (512ull << 20)) : 0ull);
        const size_t reserve = std::max(base_reserve, extra_reserve_bytes + wddm_floor);
        const size_t user_vram_cap = (vram_experts_gb >= 0.0)
            ? static_cast<size_t>(vram_experts_gb * 1073741824.0)
            : static_cast<size_t>(-1);
        for (int il = 0; il < 24; ++il) layer_has_host[il] = false;
        const size_t sf4_dn_off_c = 16u + (size_t)(2 * kInkInter) * (kInkHidden / 2) + (size_t)(2 * kInkInter) * (kInkHidden / 16);
        const size_t sf4_body_c   = sf4_dn_off_c + (size_t)kInkHidden * (kInkInter / 2) + (size_t)kInkHidden * (kInkInter / 16);
        bool all_sf4_or_mx8 = true;
        size_t compact_sec4_bytes = 0;
        if (do_compact_vram) {
            for (const auto& e : dir) {
                if (e.gu_code_id == FMT_SF4 && e.dn_code_id == FMT_SF4) {
                    compact_sec4_bytes += e.body_bytes;
                } else if (e.gu_code_id == FMT_MX8 && e.dn_code_id == FMT_MX8) {
                    compact_sec4_bytes += opt5060_ld.compact_xcode ? sf4_body_c : e.body_bytes;   // COMPACT_XCODE=0: MX8 experts stay MX8 (bigger)
                } else {
                    all_sf4_or_mx8 = false;
                    break;
                }
            }
        }
        if (do_compact_vram && all_sf4_or_mx8 && vfree > compact_sec4_bytes + (768ull << 20)) {
            // 100.0% ALL-3,072-EXPERT GDDR7 RESIDENCY on 16 GB RTX 5060 Ti (0 host misses, GPU MX8->SF4 transcode < 20 ms)!
            sec4_bytes = compact_sec4_bytes;
            INK_CK(cudaMalloc(&d_sec4, sec4_bytes));
            uint8_t* d_mx8_tmp = nullptr;
            uint32_t* d_amax_u32 = nullptr;
            INK_CK(cudaMalloc(&d_mx8_tmp, 16ull << 20));
            INK_CK(cudaMalloc(&d_amax_u32, 4));
            uint8_t* h_stg = halloc_pinned<uint8_t>(16ull << 20);
            size_t v_cur = 0;
            int n_transcoded = 0;
            for (int il = 0; il < man.c.n_phys; ++il) {
                bool any_tr_layer = false;
                for (int r = 0; r < man.c.n_experts; ++r) {
                    const int di = il * man.c.n_experts + r;
                    const auto& e = dir[di];
                    const int key = il * man.c.n_experts + e.expert_id;
                    auto& dev = h_exp[key];
                    if (!rd.read_at(hdr.sections[4].file_offset + e.body_offset, h_stg, e.body_bytes, err)) {
                        cudaFreeHost(h_stg); cudaFree(d_mx8_tmp); cudaFree(d_amax_u32);
                        return false;
                    }
                    if (opt5060_ld.compact_xcode && e.gu_code_id == FMT_MX8 && e.dn_code_id == FMT_MX8) {
                        INK_CK(cudaMemcpy(d_mx8_tmp, h_stg, e.body_bytes, cudaMemcpyHostToDevice));
                        uint8_t* dst = d_sec4 + v_cur;
                        float* dst_hdr = reinterpret_cast<float*>(dst);
                        // Gate+Up: [2304, 2304]
                        const uint8_t* gu_sp = d_mx8_tmp;
                        const uint8_t* gu_ss = gu_sp + (size_t)(2 * kInkInter) * kInkHidden;
                        uint8_t* gu_dp = dst + 16;
                        uint8_t* gu_ds = gu_dp + (size_t)(2 * kInkInter) * (kInkHidden / 2);
                        INK_CK(cudaMemset(d_amax_u32, 0, 4));
                        k_ink_mx8_amax<<<256, 256>>>(gu_sp, gu_ss, 2 * kInkInter, kInkHidden, d_amax_u32);
                        k_ink_mx8_to_sf4<<<2 * kInkInter, 256>>>(gu_sp, gu_ss, d_amax_u32, gu_dp, gu_ds, dst_hdr, 2 * kInkInter, kInkHidden);
                        // Down: [2304, 1152]
                        const uint8_t* dn_sp = d_mx8_tmp + e.dn_offset;
                        const uint8_t* dn_ss = dn_sp + (size_t)kInkHidden * kInkInter;
                        uint8_t* dn_dp = dst + sf4_dn_off_c;
                        uint8_t* dn_ds = dn_dp + (size_t)kInkHidden * (kInkInter / 2);
                        INK_CK(cudaMemset(d_amax_u32, 0, 4));
                        k_ink_mx8_amax<<<256, 256>>>(dn_sp, dn_ss, kInkHidden, kInkInter, d_amax_u32);
                        k_ink_mx8_to_sf4<<<kInkHidden, 256>>>(dn_sp, dn_ss, d_amax_u32, dn_dp, dn_ds, dst_hdr + 2, kInkHidden, kInkInter);
                        dev.body_off = v_cur;
                        dev.dn_off   = (uint32_t)sf4_dn_off_c;
                        dev.gu_code  = FMT_SF4;
                        dev.dn_code  = FMT_SF4;
                        dev.tier     = 0;
                        v_cur += sf4_body_c;
                        ++n_transcoded;
                        any_tr_layer = true;
                    } else {
                        INK_CK(cudaMemcpy(d_sec4 + v_cur, h_stg, e.body_bytes, cudaMemcpyHostToDevice));
                        dev.body_off = v_cur;
                        dev.tier     = 0;
                        v_cur += e.body_bytes;
                    }
                }
                if (any_tr_layer) {
                    moe_info[il].has_mx8 = false;
                    moe_info[il].has_sf4 = true;
                    moe_info[il].mixed_gu_dn = false;
                }
            }
            INK_CK(cudaDeviceSynchronize());
            INK_CK(cudaFreeHost(h_stg));
            INK_CK(cudaFree(d_mx8_tmp));
            INK_CK(cudaFree(d_amax_u32));
            n_vram_experts = n_total;
            n_host_experts = 0;
            all_sf4_vram = true;
            size_t vf_after = 0, vt_after = 0;
            INK_CK(ink_cuda_mem_info(&vf_after, &vt_after));
            std::fprintf(stderr, "[Ink5060Ti] 100%%-VRAM Compact Mode: trunk=%.2f GiB (embed=host, attn=%s, unembed=Q8_ROW), experts=%zu/%zu ALL SF4 in GDDR7 (%.2f GiB, %d MX8->SF4 transcoded), %.0f MiB free\n",
                         (double)trunk_bytes / 1073741824.0, fp8_trunk.ready ? "FP8_E4M3" : "Q8_ROW",
                         n_vram_experts, (size_t)n_total,
                         (double)sec4_bytes / 1073741824.0, n_transcoded, (double)vf_after / 1048576.0);
        } else if (!ssd_experts && vfree > sec4_bytes + reserve && sec4_bytes <= user_vram_cap) {
            // 100% VRAM resident! Single sequential streaming DMA!
            INK_CK(cudaMalloc(&d_sec4, sec4_bytes));
            const size_t CH = 64ull << 20;
            uint8_t* hbuf = halloc_pinned<uint8_t>(std::min(CH, sec4_bytes));
            size_t done = 0;
            while (done < sec4_bytes) {
                size_t want = std::min(CH, sec4_bytes - done);
                if (!rd.read_at(hdr.sections[4].file_offset + done, hbuf, want, err)) {
                    cudaFreeHost(hbuf);
                    return false;
                }
                INK_CK(cudaMemcpy(d_sec4 + done, hbuf, want, cudaMemcpyHostToDevice));
                done += want;
            }
            INK_CK(cudaFreeHost(hbuf));
            n_vram_experts = n_total;
            n_host_experts = 0;
        } else {
            // 3x Loop-Weighted Tiered Residency + Strata Predictive CPU/RAM Hybrid Engine (Doorbell + Ping-Pong DMA) + Auto-SSD!
            const auto& vorder = rd.vram_load_order();
            std::vector<uint16_t> order;
            if ((int)vorder.size() == n_total) {
                order = vorder;
            } else {
                // Guarantee 3x looped core layers (core_start..core_end-1) are ALWAYS prioritized before 1x prelude/coda!
                order.reserve(n_total);
                for (int il = man.c.core_start; il < man.c.core_end; ++il)
                    for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
                for (int il = 0; il < man.c.core_start; ++il)
                    for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
                for (int il = man.c.core_end; il < man.c.n_phys; ++il)
                    for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
            }
            const auto& opt5060 = ink_rtx5060ti_cfg(primary_gpu);
            if (opt5060.core_graph) {
                hybrid.rtx5060ti_opt = true;
                hybrid.rtx5060ti_pcie_share = opt5060.pcie_share;
            }
            // Reserve full headroom (runner + KV init + O(1) prefill + DSpark + WDDM floor + 2x pf_stage)!
            size_t max_body_all = 0;
            for (const auto& e : dir) max_body_all = std::max<size_t>(max_body_all, e.body_bytes);
            const size_t pf_bytes = 2 * InkHybridEngine::pf_stage_size(max_body_all);
            const size_t hyb_post_res = opt5060.pack
                ? std::max(reserve, (size_t)(960ull << 20) + wddm_floor / 2)
                : std::max(reserve, (size_t)(1408ull << 20) + wddm_floor / 2);
            const size_t hyb_reserve = hyb_post_res + pf_bytes + (64ull << 20);
            size_t vram_budget = (vfree > hyb_reserve) ? (vfree - hyb_reserve) : 0;
            if (user_vram_cap < vram_budget) vram_budget = user_vram_cap;
            const size_t ram_cap = safe_host_ram_cap(ram_experts_gb, ssd_experts);
            size_t vram_used = 0, host_used = 0, max_exp_bytes = 0, n_ssd_experts = 0;
            std::vector<uint8_t> in_vram(n_total, 0);
            std::vector<uint8_t> in_ram(n_total, 0);
            const bool force_ssd_only = ssd_experts || (ram_experts_gb == 0.0);
            if (opt5060.pack) {
                // Static Full-Layer VRAM Packing (RTX 5060 Ti 16 GB):
                // Pin 100% of 3x-looped L6..17 first, then full Prelude L0..5, then full Coda L18..23.
                // Whole-layer boundaries ensure layer_has_host[il] == false on every pinned layer (zero host syncs, zero core evictions!).
                const int E = man.c.n_experts;
                std::vector<int> layer_order;
                layer_order.reserve(man.c.n_phys);
                for (int il = man.c.core_start; il < man.c.core_end; ++il) layer_order.push_back(il);
                for (int il = 0; il < man.c.core_start; ++il) layer_order.push_back(il);
                for (int il = man.c.core_end; il < man.c.n_phys; ++il) layer_order.push_back(il);
                int n_full_vram_layers = 0;
                for (int il : layer_order) {
                    size_t layer_b = 0;
                    for (int r = 0; r < E; ++r) {
                        const auto& e = dir[il * E + r];
                        max_exp_bytes = std::max<size_t>(max_exp_bytes, e.body_bytes);
                        layer_b += e.body_bytes;
                    }
                    if (vram_used + layer_b <= vram_budget) {
                        for (int r = 0; r < E; ++r) {
                            in_vram[il * E + r] = 1;
                            ++n_vram_experts;
                        }
                        vram_used += layer_b;
                        ++n_full_vram_layers;
                    } else {
                        for (int r = 0; r < E; ++r) {
                            const int di = il * E + r;
                            const size_t al = (size_t)((dir[di].body_bytes + 63ull) & ~63ull);
                            if (!force_ssd_only && host_used + al <= ram_cap) {
                                in_ram[di] = 1;
                                host_used += al;
                                ++n_host_experts;
                            } else {
                                ++n_ssd_experts;
                                ++n_host_experts;
                            }
                        }
                    }
                }
                std::fprintf(stderr, "[Ink5060Ti] Static full-layer VRAM packing: %d/%d full layers in VRAM (%zu experts, %.2f GiB / budget %.2f GiB), %zu host experts\n",
                             n_full_vram_layers, man.c.n_phys, n_vram_experts,
                             (double)vram_used / 1073741824.0, (double)vram_budget / 1073741824.0, n_host_experts);
            } else {
                for (uint16_t di : order) {
                    const auto& e = dir[di];
                    max_exp_bytes = std::max<size_t>(max_exp_bytes, e.body_bytes);
                    const size_t al = (size_t)((e.body_bytes + 63ull) & ~63ull);
                    if (vram_used + e.body_bytes <= vram_budget) {
                        in_vram[di] = 1;
                        vram_used += e.body_bytes;
                        ++n_vram_experts;
                    } else if (!force_ssd_only && host_used + al <= ram_cap) {
                        in_ram[di] = 1;
                        host_used += al;
                        ++n_host_experts;
                    } else {
                        ++n_ssd_experts;
                        ++n_host_experts;
                    }
                }
            }
            // Adaptive hot-expert tier (see InkHybridEngine::tier_plan): disabled under opt5060.pack so pinned L6..17 core layers stay 100% VRAM!
            bool tier_mode = false;
            if (!opt5060.pack && n_host_experts > 0 && n_ssd_experts == 0 && vram_used > 0 && !ssd_experts && InkHybridEngine::tier_enabled_env()) {
                size_t add = 0;
                for (int di = 0; di < n_total; ++di) if (in_vram[di]) add += (size_t)((dir[di].body_bytes + 63ull) & ~63ull);
                const uint64_t avail_ram = ink_ram_available_bytes();
                if (ink_elastic_allow_host_tier_dup(avail_ram, host_used, add, primary_gpu)) tier_mode = true;
                else std::fprintf(stderr, "[InkTier] off: the host copy of the card's experts (%.2f GiB more) does not fit the available RAM (%.2f GiB, %.2f GiB already planned)\n",
                                  (double)add / 1073741824.0, (double)avail_ram / 1073741824.0, (double)host_used / 1073741824.0);
            }
            if (tier_mode) {
                const int E = man.c.n_experts;
                std::vector<size_t> kb(n_total, 0);
                std::vector<int> ord_keys;
                ord_keys.reserve(order.size());
                for (uint16_t di : order) {
                    const int key = (di / E) * E + dir[di].expert_id;
                    kb[(size_t)key] = dir[di].body_bytes;
                    ord_keys.push_back(key);
                }
                std::vector<double> kw(n_total, 1.0);
                for (int key = 0; key < n_total; ++key) { const int il = key / E; if (il >= man.c.core_start && il < man.c.core_end) kw[(size_t)key] = 3.0; }
                hybrid.tier_plan(kb, ord_keys, vram_used, kw);
            }
            // #6617: Allocate ping-pong prefill staging buffers BEFORE d_sec4 and the pinned host arena so
            // pinned-RAM VRAM page tables can never starve pf_buf to 0 MB and silently skip host experts!
            if (n_host_experts > 0) {
                hybrid.start_workers(cpu_threads);
                if (!hybrid.alloc_pf_stage(max_exp_bytes)) {
                    err = "not enough free VRAM for the 2x prefill staging buffers required by host experts; close other GPU apps or use a smaller pack";
                    return false;
                }
            }
            if (tier_mode) {
                if (hybrid.tier_alloc()) {
                    const int E = man.c.n_experts;
                    size_t n_start = 0;
                    for (int di = 0; di < n_total; ++di) {
                        if (in_vram[di]) {
                            in_vram[di] = 0;
                            in_ram[di] = 1;
                            host_used += (size_t)((dir[di].body_bytes + 63ull) & ~63ull);
                        }
                        if (hybrid.tier_slot_ptr((di / E) * E + dir[di].expert_id)) ++n_start;
                    }
                    vram_used = 0;
                    n_vram_experts = n_start;
                    n_host_experts = (size_t)n_total - n_start;
                } else {
                    tier_mode = false;
                    std::fprintf(stderr, "[InkTier] off: the card has no room for the slot pool (%.2f GiB)\n", (double)hybrid.tier_pool_bytes / 1073741824.0);
                }
            }
            INK_CK(cudaMalloc(&d_sec4, std::max<size_t>(256, vram_used)));
            uint8_t* h_arena_base = nullptr;
            if (n_host_experts > 0) {
                if (host_used > 0) {
                    h_arena_base = hybrid.alloc_host_arena(host_used);
                    if (!h_arena_base) { err = "cannot allocate host expert arena"; return false; }
                }
                hybrid.host_meta.assign((size_t)man.c.n_phys * man.c.n_experts, {});
                hybrid.orig_host_off.assign((size_t)man.c.n_phys * man.c.n_experts, 0);
            }
            if (n_ssd_experts > 0) {
                hybrid.ssd_only = true;
                std::vector<ink2::InkColdExpertDesc> cdescs((size_t)man.c.n_phys * man.c.n_experts);
                for (int di = 0; di < n_total; ++di) {
                    const auto& e = dir[di];
                    const int il = di / man.c.n_experts;
                    const int key = il * man.c.n_experts + e.expert_id;
                    cdescs[key].file_off   = hdr.sections[4].file_offset + e.body_offset;
                    cdescs[key].body_bytes = e.body_bytes;
                    cdescs[key].dn_off     = e.dn_offset;
                    cdescs[key].gu_code    = e.gu_code_id;
                    cdescs[key].dn_code    = e.dn_code_id;
                }
                // Bounded 1 GiB (or user ram_cap up to 2 GiB) page-locked DMA bounce/prefetch ring (0 OS swap)!
                const size_t ring_bytes = (ram_experts_gb > 0.0)
                    ? std::min<size_t>(2ull << 30, std::max<size_t>(256ull << 20, ram_cap))
                    : (512ull << 20);
                hybrid.cold_store = std::make_unique<ink2::InkColdStore>();
                auto pin_alloc = [](size_t b) -> uint8_t* {
                    uint8_t* p = nullptr;
                    if (cudaHostAlloc(&p, b, cudaHostAllocDefault) == cudaSuccess) return p;
                    return nullptr;
                };
                auto pin_free = [](uint8_t* p) { if (p) cudaFreeHost(p); };
                if (!hybrid.cold_store->open(strata_path, std::move(cdescs), max_exp_bytes, ring_bytes, 4, err, pin_alloc, pin_free)) {
                    return false;
                }
                INK_CK(cudaHostAlloc(&hybrid.h_copyback_buf, (max_exp_bytes + 255ull) & ~255ull, cudaHostAllocDefault));
                hybrid.h_copyback_bytes = (max_exp_bytes + 255ull) & ~255ull;
                std::fprintf(stderr, "[inkling-cuda] GPU+SSD streaming active: %zu VRAM (%.2f GiB), %zu RAM, %zu SSD-only "
                                     "(%d-slot pinned DMA ring = %.0f MiB, direct=%d)\n",
                             n_vram_experts, (double)vram_used / 1073741824.0,
                             n_host_experts - n_ssd_experts, n_ssd_experts,
                             hybrid.cold_store->num_slots(), (double)hybrid.cold_store->pool_bytes() / 1048576.0,
                             (int)hybrid.cold_store->is_direct());
            }

            size_t v_cur = 0, h_cur = 0;
            std::vector<uint8_t> tmp_body;
            for (int di = 0; di < n_total; ++di) {
                const auto& e = dir[di];
                const int il = di / man.c.n_experts;
                const int key = il * man.c.n_experts + e.expert_id;
                auto& dev = h_exp[key];
                if (in_vram[di]) {
                    tmp_body.resize(e.body_bytes);
                    if (!rd.read_at(hdr.sections[4].file_offset + e.body_offset, tmp_body.data(), e.body_bytes, err)) return false;
                    INK_CK(cudaMemcpy(d_sec4 + v_cur, tmp_body.data(), e.body_bytes, cudaMemcpyHostToDevice));
                    dev.body_off = v_cur;
                    dev.tier = 0;
                    v_cur += e.body_bytes;
                } else if (in_ram[di]) {
                    tmp_body.resize(e.body_bytes);
                    if (!rd.read_at(hdr.sections[4].file_offset + e.body_offset, tmp_body.data(), e.body_bytes, err)) return false;
                    uint8_t* hdst = h_arena_base + h_cur;
                    std::memcpy(hdst, tmp_body.data(), e.body_bytes);
                    hybrid.host_meta[key].body       = hdst;
                    hybrid.host_meta[key].body_bytes = e.body_bytes;
                    hybrid.host_meta[key].dn_off     = e.dn_offset;
                    hybrid.host_meta[key].gu_code    = e.gu_code_id;
                    hybrid.host_meta[key].dn_code    = e.dn_code_id;
                    hybrid.host_meta[key].cold_ssd   = false;
                    hybrid.orig_host_off[key]        = h_cur;
                    dev.body_off = h_cur;
                    dev.tier = 2;  // CPU Doorbell (M=1) + Ping-Pong DMA Streaming (M>=2)!
                    layer_has_host[il] = true;
                    h_cur += (size_t)((e.body_bytes + 63ull) & ~63ull);
                    if (tier_mode) {   // starts in a slot of the hot tier: bytes into the slot, table entry on the card (the host copy stays)
                        if (uint8_t* sp = hybrid.tier_slot_ptr(key)) {
                            INK_CK(cudaMemcpy(sp, tmp_body.data(), e.body_bytes, cudaMemcpyHostToDevice));
                            dev.body_off = (uint64_t)((intptr_t)sp - (intptr_t)d_sec4);
                            dev.tier = 0;
                        }
                    }
                } else {
                    // Cold SSD expert: ZERO host RAM allocated at startup! Streamed via pinned InkColdStore DMA ring!
                    hybrid.host_meta[key].body       = nullptr;
                    hybrid.host_meta[key].body_bytes = e.body_bytes;
                    hybrid.host_meta[key].dn_off     = e.dn_offset;
                    hybrid.host_meta[key].gu_code    = e.gu_code_id;
                    hybrid.host_meta[key].dn_code    = e.dn_code_id;
                    hybrid.host_meta[key].cold_ssd   = true;
                    hybrid.orig_host_off[key]        = 0;
                    dev.body_off = 0;
                    dev.tier = 2;
                    layer_has_host[il] = true;
                }
            }
            if (n_host_experts > 0) {
                hybrid.start_workers(cpu_threads);
                size_t vf_now = 0, vt_now = 0;
                INK_CK(ink_cuda_mem_info(&vf_now, &vt_now));
                const size_t avail = (vf_now > hyb_post_res) ? (vf_now - hyb_post_res) : 0;
                hybrid.alloc_staging(max_exp_bytes, avail);
                hybrid.exact_prepare();
                if (hybrid.cold_store) {
                    std::vector<int> idle_keys;
                    for (uint16_t di : order) {
                        if (!in_vram[di] && !in_ram[di]) {
                            const int il = di / man.c.n_experts;
                            idle_keys.push_back(il * man.c.n_experts + dir[di].expert_id);
                        }
                    }
                    hybrid.cold_store->start_idle_preloader(std::move(idle_keys), 500);
                }
            }
        }
        double t_exp = now_s() - t1;

        for (int il = 0; il < man.c.n_phys; ++il) {
            d_experts[il] = dalloc<InkExpertDev>(man.c.n_experts);
            INK_CK(cudaMemcpy(d_experts[il], h_exp.data() + (size_t)il * man.c.n_experts,
                              (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
        }

        // 3. Preprocess msg_queries[6..17] and static_sconv[0..23] into exact contiguous F32 GPU arrays
        {
            const auto& ti_q  = man.get("model.llm.middle_cycle_cross_pass_queries");
            const auto& ti_kv = man.get("model.llm.middle_cycle_cross_pass_kv_queries");
            std::vector<float> h_q(24 * kInkHidden), h_kv(24 * kInkHidden);
            INK_CK(cudaMemcpy(h_q.data(), d_trunk + ti_q.off, h_q.size() * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(h_kv.data(), d_trunk + ti_kv.off, h_kv.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 12; ++i) {
                std::vector<float> mq(4 * kInkHidden);
                std::memcpy(mq.data() + 0 * kInkHidden, h_q.data()  + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 1 * kInkHidden, h_kv.data() + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 2 * kInkHidden, h_kv.data() + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 3 * kInkHidden, h_q.data()  + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                d_msg_queries[i] = dalloc<float>(4 * kInkHidden);
                INK_CK(cudaMemcpy(d_msg_queries[i], mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
            }
        }
        for (int il = 0; il < man.c.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            d_sconv_attn[il] = (float*)(d_trunk + man.get(p + ".attn_sconv.weight").off);
            d_sconv_mlp[il]  = (float*)(d_trunk + man.get(p + ".mlp_sconv.weight").off);
            if (man.c.is_swa[il]) {
                d_sconv_k[il] = (float*)(d_trunk + man.get(p + ".attn.k_sconv.weight").off);
                d_sconv_v[il] = (float*)(d_trunk + man.get(p + ".attn.v_sconv.weight").off);
            } else {
                // Global static_w is [4, 1024] with lag 0 first -> flip(0).T -> [1024, 4] with lag 3..0!
                auto flip_t = [&](const std::string& tname) -> float* {
                    std::vector<float> src(4 * kInkKVDim), dst(kInkKVDim * 4);
                    INK_CK(cudaMemcpy(src.data(), d_trunk + man.get(tname).off, src.size() * 4, cudaMemcpyDeviceToHost));
                    for (int c = 0; c < kInkKVDim; ++c) {
                        for (int l = 0; l < 4; ++l) dst[c * 4 + l] = src[(3 - l) * kInkKVDim + c];
                    }
                    float* dptr = dalloc<float>(dst.size());
                    INK_CK(cudaMemcpy(dptr, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
                    return dptr;
                };
                d_sconv_k[il] = flip_t(p + ".attn.k_sconv.static_w");
                d_sconv_v[il] = flip_t(p + ".attn.v_sconv.static_w");
            }
        }

        // 4. Build exact 2-stage Cauchy-Schwarz screening index for BF16 or S6_K unembed (SQ-20, SQ-15, SQ-10)!
        const auto& ui = man.get("model.llm.unembed.weight");
        if (build_screen && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K)) {
            INK_CK(ink_cuda_mem_info(&vfree, &vtot));
            const size_t screen_min_vfree = ink_rtx5060ti_cfg(primary_gpu).pack ? (512ull << 20) : (2048ull << 20);
            if (vfree > screen_min_vfree) {
                unembed_screen.V = man.c.vocab_unpadded;
                unembed_screen.q8  = dalloc<int8_t>((size_t)unembed_screen.V * kInkHidden);
                unembed_screen.sc  = dalloc<float>(unembed_screen.V);
                unembed_screen.eps = dalloc<float>(unembed_screen.V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_build_unembed_screen_bf16<<<(unembed_screen.V + 7) / 8, 256>>>(
                        reinterpret_cast<const uint16_t*>(d_trunk + ui.off),
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                } else {
                    k_ink_build_unembed_screen_s6k<<<(unembed_screen.V + 7) / 8, 256>>>(
                        d_trunk + ui.off,
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                }
                INK_CK(cudaDeviceSynchronize());
                unembed_screen.ready = true;
            }
        }

        std::fprintf(stderr, "[InkCUDA] Loaded %s (%s): trunk=%.2f GB (%.2fs), experts=%zu VRAM / %zu host (%.2fs, hyb=%d, promo=%d, pf_buf=%.0fMB), sm120_tc=%d, screen=%d\n",
                     strata_path.c_str(), man.build.c_str(), (double)trunk_bytes / 1e9, t_trunk,
                     n_vram_experts, n_host_experts, t_exp, (int)hybrid.active, hybrid.n_promo_slots,
                     (double)hybrid.pf_stage_bytes / (1024.0 * 1024.0), (int)sm120::available(), (int)unembed_screen.ready);
        return true;
    }

    // Hardware P2P roundtrip + latency probe between g0 and g1.
    // On PCIe root-complex systems without NVLink/PCIe-switch, cudaDeviceCanAccessPeer can report true
    // while peer DMA stalls for tens of milliseconds; benchmark P2P vs portable pinned-host bounce and
    // auto-select whichever is genuinely faster (honoring STRATA_CUDA_PEER_PINNED=1 / STRATA_CUDA_PEER_P2P=1).
    static bool probe_p2p(int g0, int g1) {
        if (const char* env = std::getenv("STRATA_CUDA_PEER_PINNED"); env && std::atoi(env) != 0) return false;
        const bool force_p2p = []() {
            const char* env = std::getenv("STRATA_CUDA_PEER_P2P");
            return env && std::atoi(env) != 0;
        }();
        int can01 = 0, can10 = 0;
        if (cudaDeviceCanAccessPeer(&can01, g0, g1) != cudaSuccess ||
            cudaDeviceCanAccessPeer(&can10, g1, g0) != cudaSuccess || !can01 || !can10) {
            return false;
        }
        cudaSetDevice(g0);
        cudaError_t e0 = cudaDeviceEnablePeerAccess(g1, 0);
        if (e0 != cudaSuccess && e0 != cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return false; }
        cudaSetDevice(g1);
        cudaError_t e1 = cudaDeviceEnablePeerAccess(g0, 0);
        if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled) {
            cudaSetDevice(g0); cudaDeviceDisablePeerAccess(g1); cudaGetLastError();
            return false;
        }
        cudaGetLastError();

        // 9,216-byte (1x kInkHidden float) roundtrip verification + latency comparison
        constexpr size_t kN = kInkHidden;
        constexpr size_t kB = kN * sizeof(float);
        std::vector<uint32_t> h_tx(kN), h_rx01(kN, 0), h_rx10(kN, 0);
        for (size_t i = 0; i < kN; ++i) h_tx[i] = (uint32_t)(0x12345678u ^ (i * 2654435761u));
        uint32_t *d0 = nullptr, *d1 = nullptr, *hp = nullptr;
        cudaStream_t s0 = nullptr, s1 = nullptr;
        cudaSetDevice(g0);
        cudaMalloc(&d0, kB);
        cudaHostAlloc(&hp, kB, cudaHostAllocPortable);
        cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        cudaMemcpy(d0, h_tx.data(), kB, cudaMemcpyHostToDevice);
        cudaSetDevice(g1);
        cudaMalloc(&d1, kB);
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaMemset(d1, 0, kB);

        bool ok = (cudaMemcpyPeer(d1, g1, d0, g0, kB) == cudaSuccess) &&
                  (cudaMemcpy(h_rx01.data(), d1, kB, cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaSetDevice(g0);
        ok = ok && (cudaMemcpyPeer(d0, g0, d1, g1, kB) == cudaSuccess) &&
             (cudaMemcpy(h_rx10.data(), d0, kB, cudaMemcpyDeviceToHost) == cudaSuccess);
        ok = ok && (std::memcmp(h_tx.data(), h_rx01.data(), kB) == 0) &&
             (std::memcmp(h_tx.data(), h_rx10.data(), kB) == 0);

        double t_p2p_us = 1e9, t_pin_us = 1e9;
        if (ok && !force_p2p) {
            // Warmup + up to 8 timed roundtrips via P2P (early-exit if a single roundtrip stalls > 5 ms!)
            cudaSetDevice(g1);
            const double tw0 = now_s();
            cudaMemcpyPeerAsync(d1, g1, d0, g0, kB, s1);
            cudaMemcpyPeerAsync(d0, g0, d1, g1, kB, s1);
            cudaStreamSynchronize(s1);
            const double t_warm_us = (now_s() - tw0) * 1e6;
            if (t_warm_us > 5000.0) {
                t_p2p_us = t_warm_us;
            } else {
                const double tp0 = now_s();
                for (int it = 0; it < 8; ++it) {
                    cudaMemcpyPeerAsync(d1, g1, d0, g0, kB, s1);
                    cudaMemcpyPeerAsync(d0, g0, d1, g1, kB, s1);
                    cudaStreamSynchronize(s1);
                }
                t_p2p_us = (now_s() - tp0) * 1e6 / 8.0;
            }

            // Warmup + 8 timed roundtrips via portable pinned-host bounce
            cudaEvent_t ev0 = nullptr, ev1 = nullptr;
            cudaSetDevice(g0); cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming);
            cudaSetDevice(g1); cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming);
            auto do_pin = [&]() {
                cudaSetDevice(g0);
                cudaMemcpyAsync(hp, d0, kB, cudaMemcpyDeviceToHost, s0);
                cudaEventRecord(ev0, s0);
                cudaSetDevice(g1);
                cudaStreamWaitEvent(s1, ev0, 0);
                cudaMemcpyAsync(d1, hp, kB, cudaMemcpyHostToDevice, s1);
                cudaMemcpyAsync(hp, d1, kB, cudaMemcpyDeviceToHost, s1);
                cudaEventRecord(ev1, s1);
                cudaSetDevice(g0);
                cudaStreamWaitEvent(s0, ev1, 0);
                cudaMemcpyAsync(d0, hp, kB, cudaMemcpyHostToDevice, s0);
                cudaStreamSynchronize(s0);
            };
            do_pin();
            const double th0 = now_s();
            for (int it = 0; it < 8; ++it) do_pin();
            t_pin_us = (now_s() - th0) * 1e6 / 8.0;
            cudaSetDevice(g0); cudaEventDestroy(ev0);
            cudaSetDevice(g1); cudaEventDestroy(ev1);
        }

        cudaSetDevice(g0); cudaStreamDestroy(s0); cudaFree(d0); cudaFreeHost(hp);
        cudaSetDevice(g1); cudaStreamDestroy(s1); cudaFree(d1);
        cudaSetDevice(g0);
        const bool use_p2p = ok && (force_p2p || (t_p2p_us < t_pin_us * 0.90 && t_p2p_us < 500.0));
        if (!use_p2p) {
            cudaSetDevice(g0); cudaDeviceDisablePeerAccess(g1); cudaGetLastError();
            cudaSetDevice(g1); cudaDeviceDisablePeerAccess(g0); cudaGetLastError();
            cudaSetDevice(g0);
        }
        if (ok && !force_p2p) {
            std::fprintf(stderr, "[InkCUDA] EP2 GPU%d<->GPU%d transport: %s (p2p=%.1f us vs pinned=%.1f us)\n",
                         g0, g1, use_p2p ? "direct-p2p" : "pinned-host", t_p2p_us, t_pin_us);
        }
        return use_p2p;
    }

    // Unified fast GPU loader for .strata v1.1, .gguf (Unsloth + ggml-org), HF safetensors,
    // Dual-GPU EP (`cuda-ep2`), AND Multi-GPU Pipeline Layer-Split (`cuda-ls`, 2..4 GPUs)!
    template <typename ModelFileT, typename DequantFn, typename RowBytesFn>
    bool load_from_model_file(
        const ModelFileT& mf,
        int gpu0,
        int gpu1,
        float ep_split_in,
        bool build_screen,
        std::string& err,
        size_t extra_reserve_bytes,
        DequantFn&& dequant_row,
        RowBytesFn&& row_bytes,
        double vram_experts_gb = -1.0,
        double ram_experts_gb = -1.0,
        bool ssd_experts = false,
        const std::vector<int>& all_gpus = {},
        const std::string& layer_split_spec = "",
        int cpu_threads = 0) {
        // Fast zero-overhead path for single-GPU .strata v1.1!
        // The file's real kind decides, not the manifest's format string: a GGUF / safetensors file must never reach load() (the .strata reader).
        const bool is_strata_fmt = mf.is_native_strata();
        const bool single_gpu_req = (gpu1 < 0) && (all_gpus.size() <= 1) && layer_split_spec.empty();
        if (is_strata_fmt && single_gpu_req) {
            return load(mf.path, gpu0, build_screen, err, extra_reserve_bytes, vram_experts_gb, ram_experts_gb, ssd_experts, cpu_threads);
        }

        // Collect deduplicated active GPU list (1..kMaxStages)
        std::vector<int> gpus;
        if (!all_gpus.empty()) {
            for (int g : all_gpus) {
                if (g >= 0 && std::find(gpus.begin(), gpus.end(), g) == gpus.end() && (int)gpus.size() < kMaxStages) {
                    gpus.push_back(g);
                }
            }
        }
        if (gpus.empty()) {
            gpus.push_back(gpu0);
            if (gpu1 >= 0 && gpu1 != gpu0) gpus.push_back(gpu1);
        }

        path = mf.path;
        primary_gpu = gpus[0];
        peer_gpu = (gpus.size() == 2) ? gpus[1] : -1;
        ep2_active = false;
        ep2_p2p = false;
        ls_active = false;
        n_stages = 1;
        stage_gpu[0] = primary_gpu;
        for (int il = 0; il < 24; ++il) layer_owner[il] = 0;
        ep_split = std::max(0.1f, std::min(0.9f, ep_split_in));

        std::vector<int> gpu_cc(gpus.size(), 0);
        std::vector<size_t> gpu_vf(gpus.size(), 0), gpu_vt(gpus.size(), 0);
        for (size_t gi = 0; gi < gpus.size(); ++gi) {
            INK_CK(cudaSetDevice(gpus[gi]));
            cudaDeviceProp prop{};
            INK_CK(cudaGetDeviceProperties(&prop, gpus[gi]));
            gpu_cc[gi] = prop.major * 10 + prop.minor;
            INK_CK(ink_cuda_mem_info(&gpu_vf[gi], &gpu_vt[gi], gpus[gi]));
        }
        INK_CK(cudaSetDevice(primary_gpu));
        sm_cc = gpu_cc[0];
        stage_cc[0] = sm_cc;
        stage_trunk_attn_fp16[0] = (sm_cc >= 70 && sm_cc < 80);

        // Choose Multi-GPU mode: Pipeline Layer-Split (`cuda-ls`) vs Dual-GPU EP (`cuda-ep2`)!
        // Auto-select Layer-Split on: explicit --layer-split, 3-4 GPUs, mixed arch, unequal VRAM (>15%),
        // Windows WDDM (avoids 48x/tok cross-GPU EP sync stalls & #6592), or non-P2P Linux rigs (Report C 2x 3090 EPYC!).
        if (gpus.size() >= 2) {
            const bool force_ep2 = (layer_split_spec == "off" || layer_split_spec == "none" || layer_split_spec == "ep2");
            bool mixed_cc = false;
            size_t min_vt = gpu_vt[0], max_vt = gpu_vt[0];
            for (size_t gi = 1; gi < gpus.size(); ++gi) {
                if (gpu_cc[gi] != gpu_cc[0]) mixed_cc = true;
                min_vt = std::min(min_vt, gpu_vt[gi]);
                max_vt = std::max(max_vt, gpu_vt[gi]);
            }
            const bool unequal_vram = (min_vt > 0) && ((double)max_vt > (double)min_vt * 1.15);
#ifdef _WIN32
            const bool prefer_ls_os = true;
#else
            const bool prefer_ls_os = false;
#endif
            const bool explicit_ls = !layer_split_spec.empty() && !force_ep2;
            bool p2p01 = false;
            if (!force_ep2 && (explicit_ls || gpus.size() > 2 || mixed_cc || unequal_vram || prefer_ls_os)) {
                ls_active = true;
            } else if (gpus.size() == 2) {
                p2p01 = probe_p2p(gpus[0], gpus[1]);
                INK_CK(cudaSetDevice(primary_gpu));
                if (!force_ep2 && !p2p01) {
                    ls_active = true;
                    std::fprintf(stderr, "[InkCUDA] Multi-GPU non-P2P detected: auto-selecting Pipeline Layer-Split (--layer-split auto, 6 handoffs/tok vs 48x EP sync)\n");
                } else {
                    ep2_active = true;
                    ep2_p2p = p2p01;
                    peer_gpu = gpus[1];
                }
            } else {
                ls_active = true;
            }
        }
        if (ls_active) {
            n_stages = (int)gpus.size();
            for (int s = 0; s < n_stages; ++s) {
                stage_gpu[s] = gpus[s];
                stage_cc[s] = gpu_cc[s];
                stage_trunk_attn_fp16[s] = (stage_cc[s] >= 70 && stage_cc[s] < 80);
                for (int t = 0; t < n_stages; ++t) stage_p2p[s][t] = (s == t);
            }
            for (int a = 0; a < n_stages; ++a) {
                for (int b = a + 1; b < n_stages; ++b) {
                    const bool ok = probe_p2p(stage_gpu[a], stage_gpu[b]);
                    stage_p2p[a][b] = ok;
                    stage_p2p[b][a] = ok;
                }
            }
            INK_CK(cudaSetDevice(primary_gpu));
        }

        // 1. Populate InkManifest directly from unified InkModelFile
        man = InkManifest{};
        man.build = mf.m.build;
        man.c.n_phys         = mf.m.c.n_phys;
        man.c.n_visits       = mf.m.c.n_visits;
        man.c.hidden         = mf.m.c.hidden;
        man.c.inter          = mf.m.c.inter;
        man.c.vocab_padded   = mf.m.c.vocab_padded;
        man.c.vocab_unpadded = mf.m.c.vocab_unpadded;
        man.c.n_head         = mf.m.c.n_head;
        man.c.n_kv           = mf.m.c.n_kv;
        man.c.head_dim       = mf.m.c.head_dim;
        man.c.d_rel          = mf.m.c.d_rel;
        man.c.rel_extent     = mf.m.c.rel_extent;
        man.c.swa_win        = mf.m.c.swa_win;
        man.c.block_size     = mf.m.c.block_size;
        man.c.n_experts      = mf.m.c.n_experts;
        man.c.top_k          = mf.m.c.top_k;
        man.c.core_start     = mf.m.c.core_start;
        man.c.core_end       = mf.m.c.core_end;
        man.c.num_passes     = mf.m.c.num_passes;
        man.c.gate_rank      = mf.m.c.gate_rank;
        man.c.eos_id         = mf.m.c.eos_token_id;
        man.c.rms_eps        = mf.m.c.rms_eps;
        man.c.qk_clip        = mf.m.c.qk_l2_clip;
        man.c.rel_clip       = mf.m.c.rel_l2_clip;
        man.c.log_alpha      = mf.m.c.log_scale_alpha;
        man.c.log_floor      = mf.m.c.log_scale_floor;
        man.c.route_scale    = mf.m.c.route_scale;
        man.c.situ_beta      = mf.m.c.situ_beta;
        man.c.situ_lb        = mf.m.c.situ_lb;
        man.c.logit_softcap  = mf.m.c.logit_softcap;
        man.c.is_swa.assign(mf.m.c.is_swa.begin(), mf.m.c.is_swa.end());
        man.has_dspark       = (mf.m.dspark.present && df2_dspark_selected()) || df2_replaces_dspark(mf.path);   // "has a drafter": DSpark of the pack only when STRATA_INK_DRAFTER=dspark asks for it (a pack that carries it but runs without a drafter keeps no reserve for it), or DFlash 2 sidecar beside the model
        for (const auto& kv : mf.m.trunk) {
            TensorInfo ti;
            ti.name = kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.trunk[kv.first] = std::move(ti);
        }
        for (const auto& kv : mf.m.dspark_tensors) {
            TensorInfo ti;
            ti.name = kv.second.name;
            ti.fmt = kv.second.fmt;
            ti.shape = kv.second.shape;
            ti.off = kv.second.off;
            ti.bytes = kv.second.bytes;
            man.dspark[kv.first] = std::move(ti);
        }

        // 2. Read Section 1 trunk + DSpark & upload to GPU 0
        double t0 = now_s();
        std::vector<uint8_t> sec1;
        if (!mf.read_section1(sec1, err)) return false;
        trunk_bytes = sec1.size();
        if (mf.m.embed_host_tail && !ls_active && !ep2_active) {
            // GGUF compact trunk (GSQ-RCO files, see ink_gguf_loader.h): the embedding is the last tensor; it stays in pinned mapped host RAM (0 MiB of VRAM)
            const auto& ei_h = man.get("model.llm.embed.weight");
            if ((size_t)ei_h.off + (size_t)ei_h.bytes == trunk_bytes) {
                INK_CK(cudaHostAlloc(&h_embed_raw, (size_t)ei_h.bytes, cudaHostAllocMapped));
                std::memcpy(h_embed_raw, sec1.data() + ei_h.off, (size_t)ei_h.bytes);
                void* dp = nullptr;
                INK_CK(cudaHostGetDevicePointer(&dp, h_embed_raw, 0));
                d_embed_mapped = static_cast<uint8_t*>(dp);
                trunk_bytes = (size_t)ei_h.off;
            }
        }
        INK_CK(cudaMalloc(&d_trunk, std::max<size_t>(256, trunk_bytes)));
        if (trunk_bytes > 0) INK_CK(cudaMemcpy(d_trunk, sec1.data(), trunk_bytes, cudaMemcpyHostToDevice));
        double t_trunk = now_s() - t0;

        trunk_attn_fp16 = stage_trunk_attn_fp16[0];
        if (trunk_attn_fp16) {
            for (const auto& kv : man.trunk) {
                if (kv.second.fmt == FMT_BF16 && kv.first.find(".attn.") != std::string::npos) {
                    const size_t nw = (size_t)kv.second.rows() * kv.second.cols();
                    __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(d_trunk + kv.second.off);
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                }
            }
            INK_CK(cudaDeviceSynchronize());
        }

        // 3. Prepare all 3072 experts (native SF4/MX8/S_K/GGUF_RAW, auto-repacking exotic GGUF quants -> SF4 in parallel!)
        double t1 = now_s();
        const auto& dir = mf.expert_directory();
        const int n_total = (int)dir.size();
        const int D = man.c.hidden, F = man.c.inter;
        const size_t sf4_dn_off_const = 16u + (size_t)(2 * F) * (D / 2) + (size_t)(2 * F) * (D / 16);
        const size_t sf4_body_const   = sf4_dn_off_const + (size_t)D * (F / 2) + (size_t)D * (F / 16);

        // Estimate expert bytes & check which GGUF_RAW experts need SF4 repack (exotic 1-2 bit quants)
        std::vector<uint8_t> need_sf4_repack(n_total, 0);
        std::vector<size_t> eff_bytes(n_total, 0);
        for (int di = 0; di < n_total; ++di) {
            const auto& e = dir[di];
            if (e.gu_code_id == FMT_GGUF_RAW) {
                std::vector<uint8_t> tmp;
                if (di % man.c.n_experts == 0) {
                    if (!mf.read_expert_body(di, tmp, err)) return false;
                    InkGgufExpertHeader gh{};
                    std::memcpy(&gh, tmp.data(), sizeof(gh));
                    const bool gpu_ok = ink_gguf_has_gpu_kernel(gh.gu_type) && ink_gguf_has_gpu_kernel(gh.dn_type);
                    for (int ee = 0; ee < man.c.n_experts && (di + ee) < n_total; ++ee) {
                        need_sf4_repack[di + ee] = gpu_ok ? 0 : 1;
                    }
                }
            }
            eff_bytes[di] = need_sf4_repack[di] ? sf4_body_const : (size_t)e.body_bytes;
        }

        // If Layer-Split is active, partition 24 physical layers across stages & upload compact per-stage trunks!
        if (ls_active) {
            std::vector<size_t> layer_exp_b(man.c.n_phys, 0);
            for (int il = 0; il < man.c.n_phys; ++il) {
                for (int r = 0; r < man.c.n_experts; ++r) layer_exp_b[il] += eff_bytes[(size_t)il * man.c.n_experts + r];
            }
            // #7093 (from #7093 by the reporter): a peer stage (s > 0) also holds the packed trunk tensors of its layers, so the
            // "every expert of a peer stage fits in its VRAM" clamp has to count them as well.
            std::vector<size_t> layer_trunk_b(man.c.n_phys, 0);
            for (const auto& kv : man.trunk) {
                const std::string& nm = kv.first;
                for (int il = 0; il < man.c.n_phys; ++il) {
                    const std::string lp = "model.llm.layers." + std::to_string(il) + ".";
                    if (nm.rfind(lp, 0) == 0 ||
                        nm == ("model.llm.attn_res.mlp_queries." + std::to_string(il)) ||
                        (il > 0 && nm == ("model.llm.attn_res.attn_queries." + std::to_string(il - 1)))) {
                        layer_trunk_b[il] += (kv.second.bytes + 255ull) & ~255ull;
                        break;
                    }
                }
            }
            const size_t ls_user_cap = (vram_experts_gb >= 0.0)
                ? static_cast<size_t>(vram_experts_gb * 1073741824.0)
                : static_cast<size_t>(-1);
            // The VRAM a peer stage keeps back at the tier step below (same formula as there).
            auto peer_stage_reserve = [&](size_t vtot) -> size_t {
                return std::max<size_t>((size_t)(896ull << 20), wddm_safety_floor(vtot) + (size_t)(512ull << 20));
            };
            // Host experts are computed ONLY on stage 0 (the hybrid branch of the MoE step needs cur_stage == 0), so a peer stage
            // must keep every one of its experts in its own VRAM. If the plan misses (the free VRAM measured at the partition is not
            // what is left after the peer has its trunk), the partition is done again with the measured shortfall as extra margin.
            const int ls_n_stages0 = n_stages;
            int ls_gpu0[kMaxStages], ls_cc0[kMaxStages];
            bool ls_fp16_0[kMaxStages];
            for (int s = 0; s < kMaxStages; ++s) { ls_gpu0[s] = stage_gpu[s]; ls_cc0[s] = stage_cc[s]; ls_fp16_0[s] = stage_trunk_attn_fp16[s]; }
            std::vector<size_t> ls_margin(kMaxStages, 0);   // by stage index at the first try (= by position in the --gpus list)
            // The loop body below is not re-indented, to keep the #7093 diff small.
            for (int ls_try = 0;; ++ls_try) {
            if (ls_try > 0) {
                n_stages = ls_n_stages0;
                for (int s = 0; s < kMaxStages; ++s) { stage_gpu[s] = ls_gpu0[s]; stage_cc[s] = ls_cc0[s]; stage_trunk_attn_fp16[s] = ls_fp16_0[s]; }
                for (int s = 1; s < n_stages; ++s) INK_CK(ink_cuda_mem_info(&gpu_vf[s], &gpu_vt[s], stage_gpu[s]));
                INK_CK(cudaSetDevice(primary_gpu));
            }
            // Refresh free VRAM on GPU0 after trunk upload
            INK_CK(ink_cuda_mem_info(&gpu_vf[0], &gpu_vt[0], stage_gpu[0]));

            std::vector<int> counts;
            // VRAM a peer stage s > 0 can give to (experts + trunk + 48 MiB per layer): the same overhead the auto weights below use.
            auto peer_room = [&](int s) -> size_t {
                const size_t oh = wddm_safety_floor(gpu_vt[s]) + (size_t)(1024ull << 20) + ls_margin[s];
                return gpu_vf[s] > oh ? gpu_vf[s] - oh : 0;
            };
            // How many layers, counted from the end of layers [0, rem), peer stage s can take with ALL their experts in its VRAM (at most 'want').
            auto fit_tail = [&](int s, int rem, int want) -> int {
                const size_t max_b = peer_room(s);
                size_t acc_b = 0, acc_e = 0;
                int fit = 0;
                for (int k = 0; k < want && rem - 1 - k >= 0; ++k) {
                    const int il = rem - 1 - k;
                    const size_t cost = layer_exp_b[il] + layer_trunk_b[il] + (48ull << 20);
                    if (acc_b + cost <= max_b && acc_e + layer_exp_b[il] <= ls_user_cap) {
                        acc_b += cost; acc_e += layer_exp_b[il]; ++fit;
                    } else break;
                }
                return fit;
            };
            // A split that was asked for (--layer-split K1,K2,.. or the even fallback) is cut back the same way: a peer stage
            // that cannot keep all its experts in VRAM hands its first layers to the stage before it (stage 0 may use host experts).
            auto clamp_peers = [&]() {
                const std::vector<int> before = counts;
                int rem = man.c.n_phys;
                for (int s = n_stages - 1; s >= 1; --s) {
                    const int want = counts[s];
                    const int fit = fit_tail(s, rem, want);
                    counts[s] = fit;
                    counts[s - 1] += want - fit;
                    rem -= fit;
                }
                if (counts != before) {
                    std::fprintf(stderr, "[InkCUDA] layer split: cut back to fit every expert of a peer stage in its VRAM (host experts are computed only on the first stage, #7093):");
                    for (int s = 0; s < n_stages; ++s) std::fprintf(stderr, " %d->%d", before[s], counts[s]);
                    std::fprintf(stderr, " layers per stage\n");
                }
            };
            if (!layer_split_spec.empty() && layer_split_spec != "auto") {
                size_t ppos = 0;
                while (ppos < layer_split_spec.size()) {
                    size_t comma = layer_split_spec.find(',', ppos);
                    std::string tok = (comma == std::string::npos) ? layer_split_spec.substr(ppos) : layer_split_spec.substr(ppos, comma - ppos);
                    int v = std::atoi(tok.c_str());
                    if (v > 0) counts.push_back(v);
                    if (comma == std::string::npos) break;
                    ppos = comma + 1;
                }
            }
            if ((int)counts.size() == n_stages) {
                int sum = 0;
                for (int v : counts) sum += v;
                if (sum != man.c.n_phys) {
                    // Scale proportionally to n_phys
                    int acc = 0;
                    for (int s = 0; s < n_stages - 1; ++s) {
                        counts[s] = std::max(1, (int)std::lround((double)counts[s] * man.c.n_phys / std::max(1, sum)));
                        acc += counts[s];
                    }
                    counts[n_stages - 1] = std::max(1, man.c.n_phys - acc);
                }
                clamp_peers();
            } else {
                // Automatic VRAM-weighted partition (ensuring peer stages s > 0 fit 100% in VRAM!)
                counts.assign(n_stages, 0);
                std::vector<double> usable(n_stages, 0.0);
                double tot_u = 0.0;
                for (int s = 0; s < n_stages; ++s) {
                    const size_t wf = wddm_safety_floor(gpu_vt[s]);
                    const size_t stage_oh = wf + ((s == 0) ? std::max(extra_reserve_bytes, (size_t)(1280ull << 20)) : (size_t)(1024ull << 20));
                    usable[s] = (gpu_vf[s] > stage_oh) ? (double)(gpu_vf[s] - stage_oh) : 0.0;
                    tot_u += usable[s];
                }
                if (tot_u <= 1.0) {
                    const int base = man.c.n_phys / n_stages;
                    for (int s = 0; s < n_stages; ++s) counts[s] = base;
                    counts[0] += man.c.n_phys - base * n_stages;
                    clamp_peers();
                } else {
                    int rem = man.c.n_phys;
                    for (int s = n_stages - 1; s >= 1; --s) {
                        int want = std::max(1, (int)std::lround((double)man.c.n_phys * (usable[s] / tot_u)));
                        want = std::min(want, rem - s);
                        // Clamp stage s > 0 to layers that fit 100% in stage_gpu[s] VRAM (leaving stage 0 to handle any hybrid overflow!)
                        // (#7093: counts the packed trunk bytes of those layers and the --vram-experts-gb cap too, and keeps the
                        // shortfall found by an earlier try as extra margin)
                        const int fit = fit_tail(s, rem, want);
                        counts[s] = fit;
                        rem -= fit;
                    }
                    counts[0] = rem;
                }
            }
            // Drop any empty peer stages if a secondary GPU had zero usable VRAM
            std::vector<int> active_stages;
            std::vector<int> active_counts;
            for (int s = 0; s < n_stages; ++s) {
                if (s == 0 || counts[s] > 0) {
                    active_stages.push_back(stage_gpu[s]);
                    active_counts.push_back(counts[s]);
                }
            }
            if (active_stages.size() < 2) {
                ls_active = false;
                n_stages = 1;
                for (int il = 0; il < 24; ++il) layer_owner[il] = 0;
            } else {
                n_stages = (int)active_stages.size();
                int cur_l = 0;
                for (int s = 0; s < n_stages; ++s) {
                    stage_gpu[s] = active_stages[s];
                    cudaDeviceProp sp{};
                    INK_CK(cudaGetDeviceProperties(&sp, stage_gpu[s]));
                    stage_cc[s] = sp.major * 10 + sp.minor;
                    stage_trunk_attn_fp16[s] = (stage_cc[s] >= 70 && stage_cc[s] < 80);
                    stage_l0[s] = cur_l;
                    stage_l1[s] = cur_l + active_counts[s];
                    for (int il = stage_l0[s]; il < stage_l1[s]; ++il) layer_owner[il] = s;
                    cur_l = stage_l1[s];
                }
                // Pack ONLY stage-owned trunk tensors onto each peer GPU s > 0 (saves ~1.5-2 GiB per peer GPU!)
                for (int s = 1; s < n_stages; ++s) {
                    stage_trunk_off[s].clear();
                    std::vector<std::pair<std::string, TensorInfo>> owned;
                    size_t pack_bytes = 0;
                    for (const auto& kv : man.trunk) {
                        const std::string& nm = kv.first;
                        bool own = false;
                        for (int il = stage_l0[s]; il < stage_l1[s]; ++il) {
                            const std::string lp = "model.llm.layers." + std::to_string(il) + ".";
                            if (nm.rfind(lp, 0) == 0 ||
                                nm == ("model.llm.attn_res.mlp_queries." + std::to_string(il)) ||
                                (il > 0 && nm == ("model.llm.attn_res.attn_queries." + std::to_string(il - 1)))) {
                                own = true;
                                break;
                            }
                        }
                        if (own) {
                            pack_bytes = (pack_bytes + 255ull) & ~255ull;
                            stage_trunk_off[s][nm] = pack_bytes;
                            owned.push_back({nm, kv.second});
                            pack_bytes += kv.second.bytes;
                        }
                    }
                    std::vector<uint8_t> hpack(std::max<size_t>(256, pack_bytes), 0);
                    for (const auto& item : owned) {
                        const size_t dst_o = stage_trunk_off[s][item.first];
                        if (item.second.off + item.second.bytes <= sec1.size()) {
                            std::memcpy(hpack.data() + dst_o, sec1.data() + item.second.off, item.second.bytes);
                        }
                    }
                    INK_CK(cudaSetDevice(stage_gpu[s]));
                    INK_CK(cudaMalloc(&stage_d_trunk[s], hpack.size()));
                    INK_CK(cudaMemcpy(stage_d_trunk[s], hpack.data(), hpack.size(), cudaMemcpyHostToDevice));
                    if (stage_trunk_attn_fp16[s]) {
                        for (const auto& item : owned) {
                            if (item.second.fmt == FMT_BF16 && item.first.find(".attn.") != std::string::npos) {
                                const size_t nw = (size_t)item.second.rows() * item.second.cols();
                                __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(stage_d_trunk[s] + stage_trunk_off[s][item.first]);
                                k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256>>>(ptr, reinterpret_cast<__half*>(ptr), nw);
                            }
                        }
                        INK_CK(cudaDeviceSynchronize());
                    }
                }
                INK_CK(cudaSetDevice(primary_gpu));
            }
            // Check the plan against what each peer card really has now that its trunk is there (the tier step below uses these numbers).
            bool ls_redo = false;
            if (ls_active) {
                for (int s = 1; s < n_stages; ++s) {
                    size_t sf = 0, st = 0;
                    INK_CK(ink_cuda_mem_info(&sf, &st, stage_gpu[s]));
                    const size_t sres = peer_stage_reserve(st);
                    size_t bud = (sf > sres) ? (sf - sres) : 0;
                    if (ls_user_cap < bud) bud = ls_user_cap;
                    size_t need = 0;
                    for (int il = stage_l0[s]; il < stage_l1[s]; ++il) need += layer_exp_b[il];
                    if (need > bud) {
                        int orig = s;
                        for (int t = 0; t < ls_n_stages0; ++t) if (ls_gpu0[t] == stage_gpu[s]) orig = t;
                        size_t packed = 0;
                        for (int il = stage_l0[s]; il < stage_l1[s]; ++il) packed += layer_trunk_b[il];
                        const size_t used = (gpu_vf[orig] > sf) ? (gpu_vf[orig] - sf) : 0;
                        // memory the card lost beyond the packed trunk itself (context, handles, ...): the plan did not know it
                        const size_t extra = (used > packed) ? (used - packed) : 0;
                        const size_t lack = need - bud;
                        std::fprintf(stderr, "[InkCUDA] layer split: GPU%d: %.0f MiB free at the partition, %.0f MiB packed trunk, %.0f MiB free now (%.0f MiB lost beyond the trunk); "
                                     "%.0f MiB left for experts but its %d layer(s) need %.0f MiB (try %d)%s\n",
                                     stage_gpu[s], (double)gpu_vf[orig] / 1048576.0, (double)packed / 1048576.0, (double)sf / 1048576.0, (double)extra / 1048576.0,
                                     (double)bud / 1048576.0, stage_l1[s] - stage_l0[s], (double)need / 1048576.0, ls_try,
                                     (ls_try < 3) ? "; planning again with that margin" : "");
                        // first miss: the measured loss is the margin (removing layers removes their trunk too, so the shortfall itself would over-correct);
                        // a later miss: add what is still missing
                        ls_margin[orig] = (ls_try == 0) ? std::max<size_t>(ls_margin[orig], extra + (64ull << 20)) : ls_margin[orig] + lack + (64ull << 20);
                        ls_redo = true;
                    }
                }
            }
            if (ls_redo && ls_try < 3) {
                for (int s = 1; s < n_stages; ++s) {
                    INK_CK(cudaSetDevice(stage_gpu[s]));
                    if (stage_d_trunk[s]) { cudaFree(stage_d_trunk[s]); stage_d_trunk[s] = nullptr; }
                    stage_trunk_off[s].clear();
                }
                INK_CK(cudaSetDevice(primary_gpu));
                continue;
            }
            break;
            }   // the #7093 retry loop
        }
        sec1.clear();
        sec1.shrink_to_fit();

        // Determine per-expert target tier: 0=stage/GPU0 VRAM, 1=GPU1 (ep2), 2=Host RAM hybrid, 3=SSD-only (.strata)
        size_t vf0 = 0, vt0 = 0, vf1 = 0, vt1 = 0;
        INK_CK(ink_cuda_mem_info(&vf0, &vt0, primary_gpu));
        const size_t wddm_floor0 = wddm_safety_floor(vt0);
        const size_t base_res0 = ((vt0 >= (20ull << 30)) ? (1600ull << 20)
                                : (vt0 <= (13ull << 30)) ? (960ull << 20) : (1280ull << 20))
                               + (wddm_floor0 > (512ull << 20) ? (wddm_floor0 - (512ull << 20)) : 0ull);
        const size_t res0 = std::max(base_res0, extra_reserve_bytes + wddm_floor0);
        const size_t bud0 = (vf0 > res0) ? (vf0 - res0) : 0;
        size_t bud1 = 0;
        if (ep2_active) {
            INK_CK(cudaSetDevice(peer_gpu));
            INK_CK(ink_cuda_mem_info(&vf1, &vt1, peer_gpu));
            const size_t res1 = std::max((size_t)(768ull << 20), wddm_safety_floor(vt1));
            bud1 = (vf1 > res1) ? (vf1 - res1) : 0;
            INK_CK(cudaSetDevice(primary_gpu));
        }

        // Build 3x loop-weighted load order
        const auto& vorder = mf.vram_load_order;
        std::vector<uint16_t> order;
        if ((int)vorder.size() == n_total) {
            order = vorder;
        } else {
            order.reserve(n_total);
            for (int il = man.c.core_start; il < man.c.core_end; ++il)
                for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
            for (int il = 0; il < man.c.core_start; ++il)
                for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
            for (int il = man.c.core_end; il < man.c.n_phys; ++il)
                for (int r = 0; r < man.c.n_experts; ++r) order.push_back((uint16_t)(il * man.c.n_experts + r));
        }

        float eff_split = ep_split;
        if (ep2_active && std::fabs(ep_split_in - 0.5f) < 1e-3f && vf1 > vf0 + (512ull << 20)) {
            const double ratio = (double)vf1 / std::max(1.0, (double)vf0 + (double)vf1);
            eff_split = (float)std::max(0.50, std::min(0.625, ratio));
        }
        std::vector<uint8_t> target_tier(n_total, 2);
        size_t u0 = 0, u1 = 0, uh = 0, max_exp_b = 0;
        size_t stage_u[kMaxStages] = {};
        size_t stage_bud[kMaxStages] = {};
        const size_t user_vram_cap = (vram_experts_gb >= 0.0)
            ? static_cast<size_t>(vram_experts_gb * 1073741824.0)
            : static_cast<size_t>(-1);
        size_t max_body_all = 0;
        for (int di = 0; di < n_total; ++di) max_body_all = std::max<size_t>(max_body_all, eff_bytes[di]);
        const size_t pf_bytes = 2 * InkHybridEngine::pf_stage_size(max_body_all);
        const size_t hyb_post_res = std::max(res0, (size_t)(1664ull << 20));
        const size_t hyb_bud0 = (vf0 > hyb_post_res + pf_bytes + (64ull << 20)) ? (vf0 - hyb_post_res - pf_bytes - (64ull << 20)) : 0;

        n_vram_experts = 0; n_peer_experts = 0; n_host_experts = 0;
        for (int s = 0; s < kMaxStages; ++s) { stage_sec4_bytes[s] = 0; stage_n_experts[s] = 0; }

        size_t ls_peer_host_overflow = 0;
        if (ls_active) {
            for (int s = 0; s < n_stages; ++s) {
                size_t sf = 0, st = 0;
                INK_CK(ink_cuda_mem_info(&sf, &st, stage_gpu[s]));
                const size_t sres = (s == 0) ? res0 : std::max<size_t>((size_t)(896ull << 20), wddm_safety_floor(st) + (size_t)(512ull << 20));
                stage_bud[s] = (sf > sres) ? (sf - sres) : 0;
                if (user_vram_cap < stage_bud[s]) stage_bud[s] = user_vram_cap;
            }
            for (uint16_t di : order) {
                const int il = (int)di / man.c.n_experts;
                const int s = layer_owner[il];
                const size_t eb = eff_bytes[di];
                if (stage_u[s] + eb <= stage_bud[s]) {
                    target_tier[di] = 0;
                    stage_u[s] += eb;
                    stage_n_experts[s]++;
                    if (s == 0) ++n_vram_experts; else ++n_peer_experts;
                } else {
                    // Preserve lossless native FMT_GGUF_RAW blocks in host RAM (zero SF4 quality loss & zero repack startup delay!)
                    const size_t hb = (eff_bytes[di] + 63ull) & ~63ull;
                    target_tier[di] = 2; uh += hb; ++n_host_experts;
                    max_exp_b = std::max(max_exp_b, eff_bytes[di]);
                    if (s > 0) ++ls_peer_host_overflow;
                }
            }
            u0 = stage_u[0];
            // #7093: host experts are computed ONLY on stage 0 (the hybrid branch of the MoE step needs cur_stage == 0). A host expert
            // on a peer stage would be read as if it sat in that card's VRAM and the output would be garbage, with no error.
            if (ls_peer_host_overflow > 0) {
                err = "layer split: " + std::to_string(ls_peer_host_overflow) +
                      " experts of a peer card do not fit in its VRAM, and host experts are computed only on the first card (#7093); "
                      "give the other card(s) fewer layers (--layer-split), raise --vram-experts-gb, or use one card (--gpus 1)";
                return false;
            }
        } else {
            const int n_peer_per_layer = ep2_active ? std::max(1, std::min(man.c.n_experts - 1, (int)std::lround(man.c.n_experts * eff_split))) : 0;
            auto want_peer = [&](int exp_id) -> bool {
                if (!ep2_active) return false;
                if (n_peer_per_layer == man.c.n_experts / 2) return (exp_id & 1) == 1;
                return ((exp_id * n_peer_per_layer) % man.c.n_experts) < n_peer_per_layer;
            };
            size_t need0_ideal = 0, need1_ideal = 0;
            for (int di = 0; di < n_total; ++di) {
                if (want_peer(dir[di].expert_id)) need1_ideal += eff_bytes[di];
                else                              need0_ideal += eff_bytes[di];
            }
            const bool all_fit_gpu = !ssd_experts && (need0_ideal <= std::min(bud0, user_vram_cap) && need1_ideal <= std::min(bud1, user_vram_cap));
            const size_t cap0 = std::min(all_fit_gpu ? bud0 : hyb_bud0, user_vram_cap);
            if (user_vram_cap < bud1) bud1 = user_vram_cap;

            for (uint16_t di : order) {
                const size_t eb = eff_bytes[di];
                const bool pref_p = want_peer(dir[di].expert_id);
                if (pref_p && u1 + eb <= bud1) {
                    target_tier[di] = 1; u1 += eb; ++n_peer_experts;
                } else if (!pref_p && u0 + eb <= cap0) {
                    target_tier[di] = 0; u0 += eb; ++n_vram_experts;
                } else if (ep2_active && u1 + eb <= bud1) {
                    target_tier[di] = 1; u1 += eb; ++n_peer_experts;
                } else if (u0 + eb <= cap0) {
                    target_tier[di] = 0; u0 += eb; ++n_vram_experts;
                } else {
                    // Preserve lossless native FMT_GGUF_RAW blocks in host RAM (evaluated via strata::quant SIMD & GPU DP4A staging!)
                    const size_t hb = (eff_bytes[di] + 63ull) & ~63ull;
                    target_tier[di] = 2; uh += hb; ++n_host_experts;
                    max_exp_b = std::max(max_exp_b, eff_bytes[di]);
                }
            }
            if (ep2_active && n_host_experts > 0) {
                const int E = man.c.n_experts;
                for (int il = 0; il < man.c.n_phys; ++il) {
                    bool has1 = false, has2 = false;
                    for (int r = 0; r < E; ++r) {
                        const uint8_t t = target_tier[il * E + r];
                        if (t == 1) has1 = true;
                        else if (t == 2) has2 = true;
                    }
                    if (has1 && has2) {
                        for (int r = 0; r < E; ++r) {
                            const int di = il * E + r;
                            if (target_tier[di] == 1) {
                                u1 -= eff_bytes[di]; --n_peer_experts;
                                const size_t hb = (eff_bytes[di] + 63ull) & ~63ull;
                                target_tier[di] = 2; uh += hb; ++n_host_experts;
                                max_exp_b = std::max(max_exp_b, eff_bytes[di]);
                            }
                        }
                    }
                }
            }
        }

        // Adaptive hot-expert tier (one GPU, experts partly in RAM): the card's expert bytes become a pool of slots that follows the router;
        // every expert then has a host copy. Needs the RAM for the card's share on top of the host share (checked against what is available).
        bool tier_mode = false;
        std::vector<int> tier_key_of_di;
        if (!ls_active && !ep2_active && n_host_experts > 0 && u0 > 0 && !ssd_experts && InkHybridEngine::tier_enabled_env()) {
            size_t add = 0;
            for (int di = 0; di < n_total; ++di) if (target_tier[di] == 0) add += (eff_bytes[di] + 63ull) & ~63ull;
            const uint64_t avail_ram = ink_ram_available_bytes();
            if (avail_ram == 0 || avail_ram > (uint64_t)(uh + add) + (5ull << 30)) tier_mode = true;
            else std::fprintf(stderr, "[InkTier] off: the host copy of the card's experts (%.2f GiB more) does not fit the available RAM (%.2f GiB, %.2f GiB already planned)\n",
                              (double)add / 1073741824.0, (double)avail_ram / 1073741824.0, (double)uh / 1073741824.0);
        }
        if (tier_mode) {
            const int E = man.c.n_experts;
            tier_key_of_di.assign(n_total, 0);
            std::vector<size_t> kb(n_total, 0);
            for (int di = 0; di < n_total; ++di) {
                const int key = (di / E) * E + dir[di].expert_id;
                tier_key_of_di[di] = key;
                kb[(size_t)key] = eff_bytes[di];
            }
            std::vector<int> ord_keys;
            ord_keys.reserve(order.size());
            for (uint16_t di : order) ord_keys.push_back(tier_key_of_di[di]);
            // pool = the bytes the static fill would have used
            std::vector<double> kw(n_total, 1.0);
            for (int key = 0; key < n_total; ++key) { const int il = key / E; if (il >= man.c.core_start && il < man.c.core_end) kw[(size_t)key] = 3.0; }
            hybrid.tier_plan(kb, ord_keys, u0, kw);
            max_body_all = 0;
            for (int di = 0; di < n_total; ++di) max_body_all = std::max<size_t>(max_body_all, eff_bytes[di]);
        }

        // #6617: Allocate ping-pong prefill staging buffers BEFORE d_sec4 and host pinned arena!
        INK_CK(cudaSetDevice(primary_gpu));
        if (n_host_experts > 0) {
            hybrid.start_workers(cpu_threads);
            if (!hybrid.alloc_pf_stage(tier_mode ? max_body_all : max_exp_b)) {
                err = "not enough free VRAM for the 2x prefill staging buffers required by host experts; close other GPU apps or use a smaller quant";
                return false;
            }
        }
        if (tier_mode) {
            if (hybrid.tier_alloc()) {
                // every expert is now a host expert; the ones that start in a slot are copied into it below
                int n_start = 0;
                size_t uh2 = 0;
                for (int di = 0; di < n_total; ++di) {
                    if (target_tier[di] == 0) { target_tier[di] = 2; }
                    uh2 += (eff_bytes[di] + 63ull) & ~63ull;
                    if (hybrid.tier_slot_ptr(tier_key_of_di[di])) ++n_start;
                }
                uh = uh2;
                u0 = 0;
                n_vram_experts = (size_t)n_start;
                n_host_experts = (size_t)(n_total - n_start);
                max_exp_b = max_body_all;
            } else {
                tier_mode = false;
                std::fprintf(stderr, "[InkTier] off: the card has no room for the slot pool (%.2f GiB)\n", (double)hybrid.tier_pool_bytes / 1073741824.0);
            }
        }
        sec4_bytes = u0;
        INK_CK(cudaMalloc(&d_sec4, std::max<size_t>(256, sec4_bytes)));
        stage_d_sec4[0] = d_sec4;
        stage_sec4_bytes[0] = sec4_bytes;
        if (ls_active) {
            for (int s = 1; s < n_stages; ++s) {
                INK_CK(cudaSetDevice(stage_gpu[s]));
                stage_sec4_bytes[s] = stage_u[s];
                INK_CK(cudaMalloc(&stage_d_sec4[s], std::max<size_t>(256, stage_sec4_bytes[s])));
            }
            INK_CK(cudaSetDevice(primary_gpu));
        } else if (ep2_active && u1 > 0) {
            INK_CK(cudaSetDevice(peer_gpu));
            sec4_peer_bytes = u1;
            INK_CK(cudaMalloc(&d_sec4_peer, std::max<size_t>(256, sec4_peer_bytes)));
            INK_CK(cudaSetDevice(primary_gpu));
        }
        uint8_t* h_arena_base = nullptr;
        if (uh > 0) {
            const size_t ram_cap = safe_host_ram_cap(ram_experts_gb, ssd_experts);
            if (ram_cap < uh && !is_strata_fmt) {
                std::fprintf(stderr, "[InkCUDA] Note: GGUF/HF host spill %.2f GiB exceeds soft RAM cap %.2f GiB (allocating pageable host arena)\n",
                             (double)uh / 1073741824.0, (double)ram_cap / 1073741824.0);
            }
            h_arena_base = hybrid.alloc_host_arena(uh);
            if (!h_arena_base) { err = "cannot allocate host expert arena"; return false; }
            hybrid.host_meta.assign((size_t)man.c.n_phys * man.c.n_experts, {});
            hybrid.orig_host_off.assign((size_t)man.c.n_phys * man.c.n_experts, 0);
        }

        // Precompute exact destination offsets so we can upload/repack in parallel chunks!
        std::vector<size_t> dst_off(n_total, 0);
        {
            size_t c0 = 0, c1 = 0, ch = 0;
            size_t cs[kMaxStages] = {};
            for (int di = 0; di < n_total; ++di) {
                if (ls_active && target_tier[di] == 0) {
                    const int s = layer_owner[di / man.c.n_experts];
                    dst_off[di] = cs[s];
                    cs[s] += eff_bytes[di];
                } else if (target_tier[di] == 0) { dst_off[di] = c0; c0 += eff_bytes[di]; }
                else if (target_tier[di] == 1)   { dst_off[di] = c1; c1 += eff_bytes[di]; }
                else                             { dst_off[di] = ch; ch += (eff_bytes[di] + 63ull) & ~63ull; }
            }
        }

        std::vector<InkExpertDev> h_exp0(n_total), h_exp1(n_total);
        for (int il = 0; il < 24; ++il) {
            moe_info[il] = {}; moe_info_peer[il] = {};
            layer_has_host[il] = false; layer_has_peer[il] = false;
        }

        // Process layer-by-layer: sequential file read, then multi-threaded CPU repack for any exotic GGUF->SF4 experts!
        for (int il = 0; il < man.c.n_phys; ++il) {
            const int E = man.c.n_experts;
            std::vector<std::vector<uint8_t>> layer_bodies(E);
            std::vector<uint32_t> layer_dn_off(E, 0);
            std::vector<uint8_t> layer_gu_code(E, 0), layer_dn_code(E, 0);
            bool any_repack = false;
            for (int r = 0; r < E; ++r) {
                const int di = il * E + r;
                if (need_sf4_repack[di]) any_repack = true;
            }
            if (any_repack) {
                std::vector<std::vector<uint8_t>> raw_bodies(E);
                for (int r = 0; r < E; ++r) {
                    const int di = il * E + r;
                    if (!mf.read_expert_body(di, raw_bodies[r], err)) return false;
                }
                std::atomic<int> next_r{0};
                std::atomic<bool> par_ok{true};
                std::mutex err_mu;
                std::string par_err;
                const unsigned n_thr = std::max(1u, std::min<unsigned>(16u, std::thread::hardware_concurrency()));
                std::vector<std::thread> workers;
                workers.reserve(n_thr);
                for (unsigned t = 0; t < n_thr; ++t) {
                    workers.emplace_back([&]() {
                        while (true) {
                            const int r = next_r.fetch_add(1);
                            if (r >= E || !par_ok.load(std::memory_order_relaxed)) break;
                            const int di = il * E + r;
                            const auto& e = dir[di];
                            if (need_sf4_repack[di]) {
                                uint32_t new_dn = 0;
                                std::string lerr;
                                if (!repack_gguf_raw_body_to_sf4(raw_bodies[r].data(), e.dn_offset, D, F, dequant_row, row_bytes, layer_bodies[r], new_dn, lerr)) {
                                    std::lock_guard<std::mutex> lk(err_mu);
                                    par_ok.store(false);
                                    if (par_err.empty()) par_err = lerr;
                                    break;
                                }
                                raw_bodies[r].clear();
                                raw_bodies[r].shrink_to_fit();
                                layer_dn_off[r] = new_dn;
                                layer_gu_code[r] = FMT_SF4;
                                layer_dn_code[r] = FMT_SF4;
                            } else {
                                layer_bodies[r] = std::move(raw_bodies[r]);
                                layer_dn_off[r] = e.dn_offset;
                                layer_gu_code[r] = e.gu_code_id;
                                layer_dn_code[r] = e.dn_code_id;
                            }
                        }
                    });
                }
                for (auto& w : workers) w.join();
                if (!par_ok.load()) { err = par_err; return false; }
            } else {
                for (int r = 0; r < E; ++r) {
                    const int di = il * E + r;
                    const auto& e = dir[di];
                    if (!mf.read_expert_body(di, layer_bodies[r], err)) return false;
                    layer_dn_off[r] = e.dn_offset;
                    layer_gu_code[r] = e.gu_code_id;
                    layer_dn_code[r] = e.dn_code_id;
                }
            }

            // Upload layer's experts to owning GPU stage / EP2 / Host hybrid (updating PR#6593 sk_gu_mask / sk_dn_mask!)
            auto mark_info = [](InkLayerMoeInfo& mi, uint8_t gu_code, uint8_t dn_code) {
                if ((gu_code == FMT_SF4 || gu_code == FMT_MX8) && dn_code != gu_code) mi.mixed_gu_dn = true;
                if (gu_code == FMT_SF4)           mi.has_sf4  = true;
                else if (gu_code == FMT_MX8)      mi.has_mx8  = true;
                else if (gu_code == FMT_GGUF_RAW) mi.has_gguf = true;
                else {
                    mi.has_sk = true;
                    if (gu_code < 32) mi.sk_gu_mask |= (1u << gu_code);
                }
                if (dn_code != FMT_SF4 && dn_code != FMT_MX8 && dn_code != FMT_GGUF_RAW && dn_code < 32) {
                    mi.has_sk = true;
                    mi.sk_dn_mask |= (1u << dn_code);
                }
            };
            const int own_s = ls_active ? layer_owner[il] : 0;
            const int own_gpu = ls_active ? stage_gpu[own_s] : primary_gpu;
            uint8_t* own_sec4 = ls_active ? stage_d_sec4[own_s] : d_sec4;
            for (int r = 0; r < E; ++r) {
                const int di = il * E + r;
                const auto& e = dir[di];
                const int key = il * E + e.expert_id;
                const auto& body = layer_bodies[r];
                InkExpertDev d0{}, d1{};
                d0.dn_off = d1.dn_off = layer_dn_off[r];
                d0.gu_code = d1.gu_code = layer_gu_code[r];
                d0.dn_code = d1.dn_code = layer_dn_code[r];
                if (target_tier[di] == 0) {
                    INK_CK(cudaSetDevice(own_gpu));
                    INK_CK(cudaMemcpy(own_sec4 + dst_off[di], body.data(), body.size(), cudaMemcpyHostToDevice));
                    d0.body_off = dst_off[di]; d0.tier = 0;
                    d1.body_off = 0;           d1.tier = 3;
                    mark_info(moe_info[il], layer_gu_code[r], layer_dn_code[r]);
                } else if (target_tier[di] == 1) {
                    INK_CK(cudaSetDevice(peer_gpu));
                    INK_CK(cudaMemcpy(d_sec4_peer + dst_off[di], body.data(), body.size(), cudaMemcpyHostToDevice));
                    d0.body_off = 0;           d0.tier = 3;
                    d1.body_off = dst_off[di]; d1.tier = 0;
                    layer_has_peer[il] = true;
                    mark_info(moe_info_peer[il], layer_gu_code[r], layer_dn_code[r]);
                } else {
                    uint8_t* hdst = h_arena_base + dst_off[di];
                    std::memcpy(hdst, body.data(), body.size());
                    hybrid.host_meta[key].body       = hdst;
                    hybrid.host_meta[key].body_bytes = (uint32_t)body.size();
                    hybrid.host_meta[key].dn_off     = layer_dn_off[r];
                    hybrid.host_meta[key].gu_code    = layer_gu_code[r];
                    hybrid.host_meta[key].dn_code    = layer_dn_code[r];
                    hybrid.orig_host_off[key]        = dst_off[di];
                    d0.body_off = dst_off[di]; d0.tier = 2;
                    d1.body_off = 0;           d1.tier = 3;
                    layer_has_host[il] = true;
                    mark_info(moe_info[il], layer_gu_code[r], layer_dn_code[r]);
                    if (tier_mode) {
                        // starts in a slot of the hot tier: bytes into the slot, table entry on the card (the host copy stays: it is what a swap demotes to)
                        if (uint8_t* sp = hybrid.tier_slot_ptr(key)) {
                            INK_CK(cudaMemcpy(sp, body.data(), body.size(), cudaMemcpyHostToDevice));
                            d0.body_off = (uint64_t)((intptr_t)sp - (intptr_t)d_sec4);
                            d0.tier = 0;
                        }
                    }
                }
                h_exp0[key] = d0;
                h_exp1[key] = d1;
            }
            if (moe_info[il].has_gguf) {   // one ggml type pair for the whole layer -> the type-specialised decode kernels
                uint32_t gt = 0, dt = 0; bool uni = true, first = true;
                for (int r = 0; r < E; ++r) {
                    if (layer_gu_code[r] != FMT_GGUF_RAW || layer_bodies[r].size() < sizeof(InkGgufExpertHeader)) { uni = false; break; }
                    InkGgufExpertHeader gh{};
                    std::memcpy(&gh, layer_bodies[r].data(), sizeof(gh));
                    if (first) { gt = gh.gu_type; dt = gh.dn_type; first = false; }
                    else if (gh.gu_type != gt || gh.dn_type != dt) { uni = false; break; }
                }
                if (uni && !first) { moe_info[il].gguf_gu_t = gt; moe_info[il].gguf_dn_t = dt; }
            }
        }
        INK_CK(cudaSetDevice(primary_gpu));
        if (n_host_experts > 0) {
            hybrid.start_workers(cpu_threads);
            size_t vf_now = 0, vt_now = 0;
            INK_CK(ink_cuda_mem_info(&vf_now, &vt_now, primary_gpu));
            const size_t avail = (vf_now > hyb_post_res) ? (vf_now - hyb_post_res) : 0;
            hybrid.alloc_staging(max_exp_b, avail);
            hybrid.exact_prepare();
        }
        double t_exp = now_s() - t1;

        for (int il = 0; il < man.c.n_phys; ++il) {
            const int own_gpu = ls_active ? stage_gpu[layer_owner[il]] : primary_gpu;
            INK_CK(cudaSetDevice(own_gpu));
            d_experts[il] = dalloc<InkExpertDev>(man.c.n_experts);
            INK_CK(cudaMemcpy(d_experts[il], h_exp0.data() + (size_t)il * man.c.n_experts,
                              (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
            if (ep2_active) {
                INK_CK(cudaSetDevice(peer_gpu));
                d_experts_peer[il] = dalloc<InkExpertDev>(man.c.n_experts);
                INK_CK(cudaMemcpy(d_experts_peer[il], h_exp1.data() + (size_t)il * man.c.n_experts,
                                  (size_t)man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
            }
        }
        INK_CK(cudaSetDevice(primary_gpu));

        // 4. Preprocess msg_queries[6..17] and static_sconv[0..23] (placed on each layer's owning stage GPU!)
        {
            const auto& ti_q  = man.get("model.llm.middle_cycle_cross_pass_queries");
            const auto& ti_kv = man.get("model.llm.middle_cycle_cross_pass_kv_queries");
            std::vector<float> h_q(24 * kInkHidden), h_kv(24 * kInkHidden);
            INK_CK(cudaMemcpy(h_q.data(), d_trunk + ti_q.off, h_q.size() * 4, cudaMemcpyDeviceToHost));
            INK_CK(cudaMemcpy(h_kv.data(), d_trunk + ti_kv.off, h_kv.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 12; ++i) {
                std::vector<float> mq(4 * kInkHidden);
                std::memcpy(mq.data() + 0 * kInkHidden, h_q.data()  + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 1 * kInkHidden, h_kv.data() + (2 * i)     * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 2 * kInkHidden, h_kv.data() + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                std::memcpy(mq.data() + 3 * kInkHidden, h_q.data()  + (2 * i + 1) * kInkHidden, kInkHidden * 4);
                const int il = man.c.core_start + i;
                const int own_gpu = ls_active ? stage_gpu[layer_owner[il]] : primary_gpu;
                INK_CK(cudaSetDevice(own_gpu));
                d_msg_queries[i] = dalloc<float>(4 * kInkHidden);
                INK_CK(cudaMemcpy(d_msg_queries[i], mq.data(), mq.size() * 4, cudaMemcpyHostToDevice));
            }
            INK_CK(cudaSetDevice(primary_gpu));
        }
        for (int il = 0; il < man.c.n_phys; ++il) {
            const std::string p = "model.llm.layers." + std::to_string(il);
            const int own_s = ls_active ? layer_owner[il] : 0;
            const int own_gpu = ls_active ? stage_gpu[own_s] : primary_gpu;
            d_sconv_attn[il] = (float*)trunk_ptr_stage(own_s, p + ".attn_sconv.weight");
            d_sconv_mlp[il]  = (float*)trunk_ptr_stage(own_s, p + ".mlp_sconv.weight");
            if (man.c.is_swa[il]) {
                d_sconv_k[il] = (float*)trunk_ptr_stage(own_s, p + ".attn.k_sconv.weight");
                d_sconv_v[il] = (float*)trunk_ptr_stage(own_s, p + ".attn.v_sconv.weight");
            } else {
                auto flip_t = [&](const std::string& tname) -> float* {
                    std::vector<float> src(4 * kInkKVDim), dst(kInkKVDim * 4);
                    INK_CK(cudaSetDevice(primary_gpu));
                    INK_CK(cudaMemcpy(src.data(), d_trunk + man.get(tname).off, src.size() * 4, cudaMemcpyDeviceToHost));
                    for (int c = 0; c < kInkKVDim; ++c)
                        for (int l = 0; l < 4; ++l) dst[c * 4 + l] = src[(3 - l) * kInkKVDim + c];
                    INK_CK(cudaSetDevice(own_gpu));
                    float* dptr = dalloc<float>(dst.size());
                    INK_CK(cudaMemcpy(dptr, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
                    return dptr;
                };
                d_sconv_k[il] = flip_t(p + ".attn.k_sconv.static_w");
                d_sconv_v[il] = flip_t(p + ".attn.v_sconv.static_w");
            }
        }
        INK_CK(cudaSetDevice(primary_gpu));

        // 5. Optional unembed screening index
        const auto& ui = man.get("model.llm.unembed.weight");
        if (build_screen && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K)) {
            INK_CK(ink_cuda_mem_info(&vf0, &vt0, primary_gpu));
            if (vf0 > (2048ull << 20)) {
                unembed_screen.V = man.c.vocab_unpadded;
                unembed_screen.q8  = dalloc<int8_t>((size_t)unembed_screen.V * kInkHidden);
                unembed_screen.sc  = dalloc<float>(unembed_screen.V);
                unembed_screen.eps = dalloc<float>(unembed_screen.V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_build_unembed_screen_bf16<<<(unembed_screen.V + 7) / 8, 256>>>(
                        reinterpret_cast<const uint16_t*>(d_trunk + ui.off),
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                } else {
                    k_ink_build_unembed_screen_s6k<<<(unembed_screen.V + 7) / 8, 256>>>(
                        d_trunk + ui.off,
                        unembed_screen.q8, unembed_screen.sc, unembed_screen.eps, unembed_screen.V);
                }
                INK_CK(cudaDeviceSynchronize());
                unembed_screen.ready = true;
            }
        } else if (ui.fmt == FMT_Q8ROW) {
            // the loader's compact trunk (GGUF, GSQ-RCO): Q8_ROW unembed shared as the screen, as the .strata compact trunk does (0 MiB duplicate)
            unembed_screen.V        = man.c.vocab_unpadded;
            unembed_screen.V_pad    = (int)ui.rows();
            unembed_screen.sc       = reinterpret_cast<float*>(d_trunk + ui.off);
            unembed_screen.q8       = reinterpret_cast<int8_t*>(d_trunk + ui.off + (size_t)ui.rows() * 4u);
            unembed_screen.eps      = nullptr;
            unembed_screen.borrowed = true;
            unembed_screen.ready    = true;
        }

        if (ls_active) {
            std::fprintf(stderr, "[InkCUDA] Loaded %s (%s, fmt=%s) [cuda-ls %d stages]: trunk=%.2f GB (%.2fs), experts=%zu+%zu GPU / %zu host (%.2fs, hyb=%d):",
                         path.c_str(), man.build.c_str(), mf.m.format.c_str(), n_stages, (double)trunk_bytes / 1e9, t_trunk,
                         n_vram_experts, n_peer_experts, n_host_experts, t_exp, (int)hybrid.active);
            for (int s = 0; s < n_stages; ++s) {
                std::fprintf(stderr, " GPU%d(L%d..%d,%zuexp)", stage_gpu[s], stage_l0[s], stage_l1[s] - 1, stage_n_experts[s]);
            }
            std::fprintf(stderr, "\n");
        } else {
            std::fprintf(stderr, "[InkCUDA] Loaded %s (%s, fmt=%s): trunk=%.2f GB (%.2fs), experts=%zu GPU%d + %zu GPU%d / %zu host (%.2fs, ep2=%d, p2p=%d, hyb=%d), sm120_tc=%d\n",
                         path.c_str(), man.build.c_str(), mf.m.format.c_str(), (double)trunk_bytes / 1e9, t_trunk,
                         n_vram_experts, primary_gpu, n_peer_experts, peer_gpu, n_host_experts, t_exp,
                         (int)ep2_active, (int)ep2_p2p, (int)hybrid.active, (int)sm120::available());
        }
        return true;
    }

    void free_all() {
        if (ep2_active && peer_gpu >= 0) {
            cudaSetDevice(peer_gpu);
            for (int il = 0; il < 24; ++il) {
                if (d_experts_peer[il]) { cudaFree(d_experts_peer[il]); d_experts_peer[il] = nullptr; }
            }
            if (d_sec4_peer) { cudaFree(d_sec4_peer); d_sec4_peer = nullptr; }
            cudaSetDevice(primary_gpu);
        }
        ep2_active = false; ep2_p2p = false; peer_gpu = -1;
        n_peer_experts = 0; sec4_peer_bytes = 0;
        hybrid.free_all();
        unembed_screen.free_all();
        for (int i = 0; i < 12; ++i) {
            if (d_msg_queries[i]) {
                const int il = man.c.core_start + i;
                const int dev = (ls_active && il >= 0 && il < 24) ? stage_gpu[layer_owner[il]] : primary_gpu;
                cudaSetDevice(dev);
                cudaFree(d_msg_queries[i]);
                d_msg_queries[i] = nullptr;
            }
        }
        for (int il = 0; il < 24; ++il) {
            const int dev = ls_active ? stage_gpu[layer_owner[il]] : primary_gpu;
            cudaSetDevice(dev);
            if (d_experts[il]) { cudaFree(d_experts[il]); d_experts[il] = nullptr; }
            if (!man.c.is_swa.empty() && !man.c.is_swa[il]) {
                if (d_sconv_k[il]) { cudaFree(d_sconv_k[il]); d_sconv_k[il] = nullptr; }
                if (d_sconv_v[il]) { cudaFree(d_sconv_v[il]); d_sconv_v[il] = nullptr; }
            }
        }
        if (ls_active) {
            for (int s = 1; s < n_stages; ++s) {
                cudaSetDevice(stage_gpu[s]);
                if (stage_d_sec4[s])  { cudaFree(stage_d_sec4[s]); stage_d_sec4[s] = nullptr; }
                if (stage_d_trunk[s]) { cudaFree(stage_d_trunk[s]); stage_d_trunk[s] = nullptr; }
                stage_trunk_off[s].clear();
            }
        }
        cudaSetDevice(primary_gpu);
        stage_d_sec4[0] = nullptr;
        ls_active = false; n_stages = 1;
        if (d_sec4) { cudaFree(d_sec4); d_sec4 = nullptr; }
        if (h_sec4_pinned) { cudaFreeHost(h_sec4_pinned); h_sec4_pinned = nullptr; }
        if (h_embed_raw) { cudaFreeHost(h_embed_raw); h_embed_raw = nullptr; d_embed_mapped = nullptr; }
        fp8_trunk.free_all();
        all_sf4_vram = false;
        if (d_trunk) { cudaFree(d_trunk); d_trunk = nullptr; }
    }
};

// ============================================================================
// InkCudaRunner: Full 48-Visit Forward, Continuous Batching (B=1..64), Tree-16, CUDA Graphs!
// ============================================================================
struct InkCudaRunner {
    InkCudaModel& m;
    cudaStream_t st = nullptr;
    cublasHandle_t cublas = nullptr;

    int max_slots = 1;
    int max_ctx = 4096;
    int kv_alloc_ctx = 0;         // Elastic global KV current allocated capacity (dev_ctx + host_ctx <= max_ctx!)
    int kv_dev_ctx = 0;           // Global KV positions resident in VRAM
    int kv_host_ctx = 0;          // Global KV positions offloaded to mapped pinned host RAM
    int kv_offload_mode = 0;      // 0 = auto, 1 = on (100% host global KV), -1 = off (VRAM-only)
    int max_batch = 512;
    int swa_ring = 1024 + 512;
    int tree_cap = 16;
    int max_passes_override = 0;  // 0 = full 3 passes (48 visits); 1 or 2 = early-pass exit probe!
    InkKvDtype kv_dtype = InkKvDtype::FP16;

    // KV Cache (48 visits: 40 SWA in VRAM, 8 Global split across VRAM + zero-copy mapped pinned host RAM!)
    InkKvVisitDesc kv_desc[kInkVisits]{};
    void* host_kv_raw_k[kInkVisits]   = {};
    void* host_kv_raw_v[kInkVisits]   = {};
    void* host_kv_raw_ksc[kInkVisits] = {};
    void* host_kv_raw_vsc[kInkVisits] = {};
    // ShortConv state pool: [48 visits] x 4 states (k [1024], v [1024], attn [2304], mlp [2304]) -> 1 contiguous buffer!
    // Layout: visit -> [max_slots, 3, 6656] F32 (or 256 MB for 64 slots)
    float* d_sconv_pool[kInkVisits] = {};

    // Activation workspaces
    float *d_cur = nullptr, *d_res = nullptr, *d_prelude = nullptr;
    float *d_blocks[8] = {};          // attention-residual block bank (max 6 needed)
    float *d_prelude_blocks[2] = {};  // saved prelude block snapshots
    float *d_bank[4] = {};            // saved previous-pass core blocks
    float *d_bank_out = nullptr;
    float *d_mixed = nullptr;         // [4, max_batch, 2304]
    float *d_msgs = nullptr;          // [4, max_batch, 2304]
    float *d_inp0 = nullptr, *d_inp1 = nullptr, *d_inp2 = nullptr;
    float *d_q = nullptr, *d_k_raw = nullptr, *d_v_raw = nullptr, *d_k = nullptr, *d_v = nullptr, *d_r = nullptr;
    float *d_k_dyn = nullptr, *d_v_dyn = nullptr;
    float *d_sinks = nullptr, *d_gate_attn = nullptr, *d_rel_table = nullptr;
    float *d_attn_out = nullptr, *d_wo_raw = nullptr, *d_wo_conv = nullptr;
    float *d_moe_in = nullptr, *d_moe_raw = nullptr, *d_moe_conv = nullptr;
    float *d_h_norm = nullptr;        // [max_batch, 2304] final output_norm hidden
    // #6671 diagnostics (STRATA_INK_HASH=1): per row hashes of cur before the MoE (slot 4v), the MoE input (4v+1), the router logits (4v+3), cur at
    // the end of the visit (4v+2) and the final norm (192); visit 0 also q, k, v after the clip (193..195), the attention output (196), the sinks (198)
    bool dbg_hash_on = false;
    int dbg_tap_visit = 0;   // STRATA_INK_HASH_VISIT: the visit whose inner stages (projections, attention, MoE) are hashed
    unsigned long long* d_dbg_hash = nullptr;
    static constexpr int kDbgSlots = 212;
    void dbg_hash(int slot, const float* ptr, int rows, int dim) {
        if (!dbg_hash_on || rows > 16 || rows < 1) return;
        k_ink_dbg_hash_rows<<<rows, 256, 0, st>>>(ptr, dim, d_dbg_hash, slot, kDbgSlots);
    }
    void dbg_hash_print(const char* tag, int M, const int32_t* pos, const int32_t* dep, const int32_t* tok, const int32_t* par, int anchor) {
        if (!dbg_hash_on || M > 16 || M < 1) return;
        INK_CK(cudaStreamSynchronize(st));
        std::vector<unsigned long long> h((size_t)M * kDbgSlots);
        INK_CK(cudaMemcpy(h.data(), d_dbg_hash, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int b = 0; b < M; ++b) {
            std::fprintf(stderr, "[HASH] %s B=%d row=%d pos=%d dep=%d anchor=%d tok=%d par=%d", tag, M, b, pos[b], dep ? dep[b] : 0, anchor, tok[b], par ? par[b] : b - 1);
            for (int q = 0; q <= 211; ++q) std::fprintf(stderr, " %llx", h[(size_t)b * kDbgSlots + q]);
            std::fputc('\n', stderr);
        }
    }
    float *d_taps[4] = {};            // [max_batch, 2304] taps at visits [5, 23, 41, 48]
    float *d_taps_compact[4] = {};    // [tree_cap, 2304] compacted accepted-path taps for DSpark
    float *d_ctrl_cat = nullptr, *d_ctrl_gate = nullptr, *d_ctrl_gain = nullptr, *d_ctrl_forget = nullptr;
    float *d_attn_pm = nullptr, *d_attn_pl = nullptr, *d_attn_po = nullptr;  // Split-K FlashDecoding buffers
    static constexpr int kInkPfKvTile = 2048;  // Fixed O(1)-VRAM KV tile (~114 MiB forever, fixes #6609 & #6605!)
    __nv_bfloat16 *d_pf_q = nullptr, *d_pf_k = nullptr, *d_pf_v = nullptr, *d_pf_p = nullptr;  // TC prefill attention
    float *d_pf_s = nullptr, *d_pf_o = nullptr, *d_pf_ot = nullptr, *d_pf_m = nullptr, *d_pf_l = nullptr;
    int pf_kv_cap = 0;
    float *d_stage_k = nullptr, *d_stage_v = nullptr, *d_stage_sconv = nullptr;  // Tree-16 verify staging
    InkKvVisitDesc* d_kv_desc_arr = nullptr;
    float** d_sconv_pool_arr = nullptr;
    int32_t* d_acc_path = nullptr;
    __nv_bfloat16 *d_x_bf16 = nullptr;
    __nv_bfloat16 *d_w_scratch = nullptr;  // [max_N, max_K] BF16 scratch for Q8_ROW / S_K cuBLAS prefill
    // Opt-in RTX 5090 (STRATA_INK_RTX5090=1) pre-packed fused BF16 QKV+R+Gate & BF16 Loop Controller pools
    __nv_bfloat16 *d_rtx_qkv_pool = nullptr;
    __nv_bfloat16 *d_rtx_qkv[24] = {};
    float *d_rtx_ctrl_pool = nullptr;
    float *d_rtx_ctrl_gf[12][4] = {};          // [core_layer][0=attn,1=k,2=v,3=mlp]: [4608, 64] F32 (gain;forget) for exact TF32 GEMM!
    __half2 *d_rtx_rel_proj_pool = nullptr;    // [24, 1024, 8] __half2 transposed rel_proj (768 KB total!)
    __half2 *d_rtx_rel_proj_T[24] = {};
    rtx5090::Rtx5090Fp8Engine fp8_eng;         // Opt-in RTX 5090 runtime FP8_E4M3 trunk & controller engine (672 MB)!
    int8_t* d_xq8 = nullptr;               // activations of a Q8_0 dense GEMM, quantized into the mma B-fragment order (k_ink_quant_x_q8f)
    float*  d_xs8 = nullptr;               // and their scales, one per (token, 32-wide block)

    // Logits & screened argmax buffers
    float   *d_logits = nullptr;      // [logits_cap, 200058] (allocated lazily if full logits needed)
    int      logits_cap = 0;
    int      screen_cap = 8;          // 8 baseline, 24 when STRATA_INK_RTX5090=1 (single-pass M<=24!)
    float   *d_dot_hat = nullptr;     // [screen_cap, 200058] for screened argmax
    float   *d_h_norm2 = nullptr;     // [screen_cap]
    int     *d_max_lb = nullptr;      // [screen_cap]
    int32_t *d_cand_ids = nullptr;    // [screen_cap, 4096]
    int32_t *d_cand_cnt = nullptr;    // [screen_cap]
    int32_t *d_argmax = nullptr;      // [64]

    // RTX 5060 Ti opt-in vocab-tiled INT8 TC exact argmax scratch for Q8_ROW unembed
    int32_t *d_q8_tile_i32 = nullptr;
    float   *d_q8_tile_val = nullptr;
    int32_t *d_q8_tile_id = nullptr;
    int      q8_argmax_cap_M = 0;
    int      q8_argmax_vt_nom = 0;
    uint8_t *d_kv_slab = nullptr;
    uint8_t *d_act_slab = nullptr;
    void    *d_cublas_ws = nullptr;
    InkC64KvPrefetch c64_kv_pf;
    bool     mc_kv_pf_active = false;
    bool     swa_pinned[kInkVisits] = {};
    int      n_swa_pinned = 0;
    int      mc_pf_slot[kInkVisits] = {};
    int      mc_pf_next[kInkVisits] = {};

    // Step metadata in device memory (CUDA-Graph capturable!) + 64-slot pinned host ring (zero async H2D overwrite!)
    int32_t *d_tok_ids = nullptr;     // [max_batch]
    int32_t *d_slots = nullptr;       // [max_batch]
    int32_t *d_pos = nullptr;         // [max_batch]
    uint8_t *d_is_img = nullptr;      // [max_batch] optional vision mask
    float   *d_img_emb = nullptr;     // [max_batch, 2304] optional vision embeddings
    bool     has_img_step = false;
    int8_t  *d_sconv_lag = nullptr;   // [32 * 3]
    uint8_t *d_tree_anc = nullptr;    // [32 * 8]
    uint8_t *d_tree_anc32 = nullptr;  // [32 * 32] opt-in deep tree (depth <= 31)
    uint8_t *d_tree_dep = nullptr;    // [32]
    int32_t *h_meta_pin = nullptr;    // 64-slot ring buffer
    size_t   meta_ring_stride = 0;
    size_t   meta_ring_idx = 0;

    InkMoeWorkspace moe_ws;
    InkMoeWorkspace* cur_moe_ws = &moe_ws;

    // Multi-GPU Pipeline Layer-Split (`cuda-ls`) per-stage execution context (stages[0] mirrors primary pointers!)
    struct StageExec {
        cudaStream_t st = nullptr;
        cublasHandle_t cublas = nullptr;
        cudaEvent_t ev_done = nullptr;
        float *d_cur = nullptr, *d_res = nullptr, *d_bank_out = nullptr;
        float *d_blocks[8] = {};
        float *d_bank[4] = {};
        float *d_mixed = nullptr, *d_msgs = nullptr, *d_inp0 = nullptr, *d_inp1 = nullptr, *d_inp2 = nullptr;
        float *d_q = nullptr, *d_k_raw = nullptr, *d_v_raw = nullptr, *d_k = nullptr, *d_v = nullptr, *d_r = nullptr;
        float *d_k_dyn = nullptr, *d_v_dyn = nullptr;
        float *d_sinks = nullptr, *d_gate_attn = nullptr, *d_rel_table = nullptr;
        float *d_attn_out = nullptr, *d_wo_raw = nullptr, *d_wo_conv = nullptr;
        float *d_moe_in = nullptr, *d_moe_raw = nullptr, *d_moe_conv = nullptr;
        float *d_ctrl_cat = nullptr, *d_ctrl_gate = nullptr, *d_ctrl_gain = nullptr, *d_ctrl_forget = nullptr;
        float *d_attn_pm = nullptr, *d_attn_pl = nullptr, *d_attn_po = nullptr;
        __nv_bfloat16 *d_pf_q = nullptr, *d_pf_k = nullptr, *d_pf_v = nullptr, *d_pf_p = nullptr;
        float *d_pf_s = nullptr, *d_pf_o = nullptr, *d_pf_ot = nullptr, *d_pf_m = nullptr, *d_pf_l = nullptr;
        int pf_kv_cap = 0;
        __nv_bfloat16 *d_x_bf16 = nullptr, *d_w_scratch = nullptr;
        int32_t *d_slots = nullptr, *d_pos = nullptr;
        InkMoeWorkspace ws;
    };
    StageExec ls_stages[InkCudaModel::kMaxStages]{};
    int cur_stage = 0;
    float* h_ls_bounce = nullptr;
    int32_t* h_ls_meta_bounce = nullptr;

    // Dual-GPU Expert Parallelism (cuda-ep2) peer workspace & async sync (dynamically sized to max_batch -> zero B=64 cap!)
    cudaStream_t peer_st = nullptr;
    cudaEvent_t ev_moe_ready = nullptr, ev_peer_done = nullptr;
    InkMoeWorkspace peer_ws;
    float *d_peer_x = nullptr, *d_peer_y = nullptr, *d_peer_y_rx = nullptr;
    float *h_ep_x = nullptr, *h_ep_w = nullptr, *h_ep_y = nullptr;
    int32_t *h_ep_idx = nullptr;

    // 48-Visit CUDA Graph cache (keyed by M, is_tree, capture_taps)
    struct GraphKey {
        int M;
        int is_tree;
        int want_taps;
        int kv_fp8;
        bool operator==(const GraphKey& o) const {
            return M == o.M && is_tree == o.is_tree && want_taps == o.want_taps && kv_fp8 == o.kv_fp8;
        }
    };
    struct GraphEntry {
        GraphKey key;
        cudaGraphExec_t exec = nullptr;
    };
    std::vector<GraphEntry> graph_cache;
    std::vector<GraphEntry> prefix_core_graph_cache;
    // Every captured decode graph holds the K/V descriptor pointers and the lane it was captured with: when the K/V layout changes (elastic tier-1 growth, a new device cap)
    // or the cold-expert lane flips, ALL of them go, the 5060 Ti prefix+core graphs included. (Dropping only graph_cache left stale prefix+core graphs that read the freed
    // host K/V buffer: CUDA error 700 on the first prompt over 16K with PACK + CORE_GRAPH + elastic KV, found 11-26.)
    void drop_all_graphs() {
        for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        graph_cache.clear();
        for (auto& e : prefix_core_graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        prefix_core_graph_cache.clear();
    }
    bool use_cuda_graph = true;
    // Mode-3 batched multi-slot speculative staging pointers (owned by Rtx5090BatchedSpec when active)
    __half* d_bspec_k = nullptr;
    __half* d_bspec_v = nullptr;
    float*  d_bspec_sconv = nullptr;
    int     bspec_m_cap = 0;
    int     bspec_k_slot = 0;
    InkElasticKvSsd elastic_kv;
    int cur_max_pos = 0;

    // Opt-in RTX 5060 Ti (STRATA_INK_RTX5060TI=1) Layer-Major / All-VRAM Macro Prefill + FP8 Engine + C64 Mapped SWA + Mode-3 Batched Speculative Verify state
    opt5060::Ink5060TiFp8Engine fp8_sm;
    Ink5060TiPrefillBufs opt5060_pf;
    Ink5060TiMacroBufs   opt5060_macro;
    uint8_t*             d_pf_slab = nullptr;
    size_t               d_pf_slab_bytes = 0;
    bool swa_in_host = false;
    void*     h_mc_sconv_raw     = nullptr;
    uint16_t* d_mc_sconv_stage   = nullptr;  // [48, max_slots, mc_stage_cap_k, kInkSconvCh] BF16 (VRAM preferred, else mapped host!)
    bool      d_mc_sconv_is_vram = false;
    int       mc_stage_cap_k     = 0;
    int32_t* d_mc_slot_ids    = nullptr;  // [max_slots]
    int32_t* d_mc_base_pos    = nullptr;  // [max_slots]
    int32_t* d_mc_n_acc       = nullptr;  // [max_slots]
    int      mc_cur_B         = 0;
    int      mc_cur_K         = 0;

    // Optional per-stage profiler (active only when prof_enabled=true and graph disabled)
    bool prof_enabled = false;
    bool prof_prefill = false;   // STRATA_INK_PROF=2: also print the phases of every prefill chunk (M >= 64)
    long prof_steps = 0;
    double prof_ms[16] = {};  // 0=res/ctrl/norm, 1=dense_qkv, 2=sconv_clip_kv, 3=attn, 4=dense_wo, 5=moe, 6=other, 7=unembed
    cudaEvent_t ev_p0 = nullptr, ev_p1 = nullptr;
    // The marks only record events (no sync per phase), so the GPU stays fed and the phase times are the stream's own;
    // prof_flush() adds them up once the step is done.
    struct ProfPair { cudaEvent_t a, b; int idx; };
    std::vector<cudaEvent_t> prof_pool;
    std::vector<ProfPair> prof_pairs;
    size_t prof_used = 0;
    cudaEvent_t prof_prev = nullptr;
    cudaEvent_t prof_next_ev() {
        if (prof_used == prof_pool.size()) { cudaEvent_t e; INK_CK(cudaEventCreate(&e)); prof_pool.push_back(e); }
        return prof_pool[prof_used++];
    }
    void prof_mark_start() {
        if (!prof_enabled) return;
        cudaEvent_t e = prof_next_ev();
        INK_CK(cudaEventRecord(e, st));
        prof_prev = e;
    }
    void prof_mark_end(int idx) {
        if (!prof_enabled) return;
        cudaEvent_t e = prof_next_ev();
        INK_CK(cudaEventRecord(e, st));
        if (prof_prev) prof_pairs.push_back({prof_prev, e, idx});
        prof_prev = e;
    }
    void prof_flush() {
        if (!prof_enabled || prof_pairs.empty()) { prof_used = 0; return; }
        INK_CK(cudaStreamSynchronize(st));
        for (const auto& p : prof_pairs) {
            float ms = 0.0f;
            INK_CK(cudaEventElapsedTime(&ms, p.a, p.b));
            prof_ms[p.idx] += ms;
        }
        prof_pairs.clear();
        prof_used = 0;
        prof_prev = nullptr;
    }
    void prof_reset() { for (double& x : prof_ms) x = 0.0; }
    void prof_print(const char* tag, int iters) const {
        double tot = 0; for (double x : prof_ms) tot += x;
        std::fprintf(stderr, "[PROF %s / %d iters] total=%.2f ms/step | res_ctrl=%.2f dense_qkv=%.2f sconv_kv=%.2f attn=%.2f dense_wo=%.2f moe=%.2f unembed=%.2f\n",
                    tag, iters, tot / iters,
                    prof_ms[0] / iters, prof_ms[1] / iters, prof_ms[2] / iters, prof_ms[3] / iters,
                    prof_ms[4] / iters, prof_ms[5] / iters, prof_ms[7] / iters);
        std::fprintf(stderr, "[PROF moe parts / %d iters] (the moe= above is what is left after these) router+topk=%.2f bin=%.2f gu(sf4/mx8/sk)=%.2f dn(sf4/mx8/sk)=%.2f gu(gguf)=%.2f dn(gguf)=%.2f ms/step\n",
                    iters, prof_ms[8] / iters, prof_ms[9] / iters, prof_ms[10] / iters, prof_ms[11] / iters, prof_ms[12] / iters, prof_ms[13] / iters);
    }

    static size_t kv_bytes_per_visit_pos(InkKvDtype dt) noexcept {
        if (dt == InkKvDtype::FP16) return (size_t)kInkKVDim * 4u;
        if (dt == InkKvDtype::K8V4) return (size_t)kInkKVDim + ((size_t)kInkKVDim / 2u) + (size_t)(2 * kInkKVHeads * 4);
        return (size_t)kInkKVDim * 2u + (size_t)(2 * kInkKVHeads * 4);  // FP8 or K8V8
    }

    InkCudaRunner(InkCudaModel& model) : m(model) {}

    void bind_stage(int s) {
        if (!m.ls_active || s == cur_stage) return;
        ls_stages[cur_stage].d_pf_q = d_pf_q; ls_stages[cur_stage].d_pf_k = d_pf_k;
        ls_stages[cur_stage].d_pf_v = d_pf_v; ls_stages[cur_stage].d_pf_p = d_pf_p;
        ls_stages[cur_stage].d_pf_s = d_pf_s; ls_stages[cur_stage].d_pf_o = d_pf_o;
        ls_stages[cur_stage].d_pf_ot = d_pf_ot; ls_stages[cur_stage].d_pf_m = d_pf_m;
        ls_stages[cur_stage].d_pf_l = d_pf_l; ls_stages[cur_stage].pf_kv_cap = pf_kv_cap;
        cur_stage = s;
        INK_CK(cudaSetDevice(m.stage_gpu[s]));
        StageExec& se = ls_stages[s];
        st = se.st; cublas = se.cublas;
        d_cur = se.d_cur; d_res = se.d_res; d_bank_out = se.d_bank_out;
        for (int i = 0; i < 8; ++i) d_blocks[i] = se.d_blocks[i];
        for (int i = 0; i < 4; ++i) d_bank[i] = se.d_bank[i];
        d_mixed = se.d_mixed; d_msgs = se.d_msgs; d_inp0 = se.d_inp0; d_inp1 = se.d_inp1; d_inp2 = se.d_inp2;
        d_q = se.d_q; d_k_raw = se.d_k_raw; d_v_raw = se.d_v_raw; d_k = se.d_k; d_v = se.d_v; d_r = se.d_r;
        d_k_dyn = se.d_k_dyn; d_v_dyn = se.d_v_dyn;
        d_sinks = se.d_sinks; d_gate_attn = se.d_gate_attn; d_rel_table = se.d_rel_table;
        d_attn_out = se.d_attn_out; d_wo_raw = se.d_wo_raw; d_wo_conv = se.d_wo_conv;
        d_moe_in = se.d_moe_in; d_moe_raw = se.d_moe_raw; d_moe_conv = se.d_moe_conv;
        d_ctrl_cat = se.d_ctrl_cat; d_ctrl_gate = se.d_ctrl_gate; d_ctrl_gain = se.d_ctrl_gain; d_ctrl_forget = se.d_ctrl_forget;
        d_attn_pm = se.d_attn_pm; d_attn_pl = se.d_attn_pl; d_attn_po = se.d_attn_po;
        d_pf_q = se.d_pf_q; d_pf_k = se.d_pf_k; d_pf_v = se.d_pf_v; d_pf_p = se.d_pf_p;
        d_pf_s = se.d_pf_s; d_pf_o = se.d_pf_o; d_pf_ot = se.d_pf_ot; d_pf_m = se.d_pf_m; d_pf_l = se.d_pf_l;
        pf_kv_cap = se.pf_kv_cap;
        d_x_bf16 = se.d_x_bf16; d_w_scratch = se.d_w_scratch;
        d_slots = se.d_slots; d_pos = se.d_pos;
        cur_moe_ws = (s == 0) ? &moe_ws : &se.ws;
    }

    void copy_between_stages(int from_s, int to_s, const void* src, void* dst, size_t bytes) {
        if (from_s == to_s || bytes == 0) return;
        const int g_from = m.stage_gpu[from_s];
        const int g_to   = m.stage_gpu[to_s];
        if (m.stage_p2p[from_s][to_s]) {
            INK_CK(cudaMemcpyPeerAsync(dst, g_to, src, g_from, bytes, ls_stages[to_s].st));
        } else {
            INK_CK(cudaSetDevice(g_from));
            INK_CK(cudaMemcpyAsync(h_ls_bounce, src, bytes, cudaMemcpyDeviceToHost, ls_stages[from_s].st));
            INK_CK(cudaEventRecord(ls_stages[from_s].ev_done, ls_stages[from_s].st));
            INK_CK(cudaSetDevice(g_to));
            INK_CK(cudaStreamWaitEvent(ls_stages[to_s].st, ls_stages[from_s].ev_done, 0));
            INK_CK(cudaMemcpyAsync(dst, h_ls_bounce, bytes, cudaMemcpyHostToDevice, ls_stages[to_s].st));
            INK_CK(cudaEventRecord(ls_stages[to_s].ev_done, ls_stages[to_s].st));
            INK_CK(cudaSetDevice(g_from));
            INK_CK(cudaStreamWaitEvent(ls_stages[from_s].st, ls_stages[to_s].ev_done, 0));
            INK_CK(cudaSetDevice(g_to));
        }
    }

    void switch_stage_handoff(int to_s, int M, int mode, int n_blocks, const float* (&block_ptrs)[8],
                              DepthStatePtrs* bank_stp, int n_bank) {
        if (!m.ls_active || to_s == cur_stage) return;
        const int from_s = cur_stage;
        const size_t hbytes = (size_t)M * kInkHidden * sizeof(float);
        StageExec& sf = ls_stages[from_s];
        StageExec& st_to = ls_stages[to_s];
        if (m.stage_p2p[from_s][to_s]) {
            INK_CK(cudaSetDevice(m.stage_gpu[from_s]));
            INK_CK(cudaEventRecord(sf.ev_done, sf.st));
            INK_CK(cudaSetDevice(m.stage_gpu[to_s]));
            INK_CK(cudaStreamWaitEvent(st_to.st, sf.ev_done, 0));
        }
        copy_between_stages(from_s, to_s, sf.d_cur, st_to.d_cur, hbytes);
        for (int b = 0; b < n_blocks; ++b) {
            copy_between_stages(from_s, to_s, sf.d_blocks[b], st_to.d_blocks[b], hbytes);
            block_ptrs[b] = st_to.d_blocks[b];
        }
        if (bank_stp && bank_stp->count > 0) {
            for (int b = 0; b < n_bank; ++b) {
                copy_between_stages(from_s, to_s, sf.d_bank[b], st_to.d_bank[b], hbytes);
                bank_stp->p[b] = st_to.d_bank[b];
            }
            copy_between_stages(from_s, to_s, sf.d_bank_out, st_to.d_bank_out, hbytes);
            bank_stp->p[n_bank] = st_to.d_bank_out;
        }
        copy_between_stages(from_s, to_s, sf.d_pos, st_to.d_pos, (size_t)M * sizeof(int32_t));
        if (mode == 1) {
            copy_between_stages(from_s, to_s, sf.d_slots, st_to.d_slots, (size_t)M * sizeof(int32_t));
        }
        bind_stage(to_s);
    }

    template <typename T>
    static bool alloc_mapped_host(size_t count, void*& raw_out, T*& dev_out) noexcept {
        raw_out = nullptr;
        dev_out = nullptr;
        if (count == 0) return true;
        const size_t bytes = count * sizeof(T);
        if (cudaHostAlloc(&raw_out, bytes, cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            raw_out = nullptr;
            return false;
        }
        void* dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, raw_out, 0) != cudaSuccess) {
            cudaGetLastError();
            cudaFreeHost(raw_out);
            raw_out = nullptr;
            return false;
        }
        dev_out = reinterpret_cast<T*>(dp);
        return true;
    }

    void free_host_kv_visit(int v) noexcept {
        if (host_kv_raw_k[v])   { cudaFreeHost(host_kv_raw_k[v]);   host_kv_raw_k[v] = nullptr; }
        if (host_kv_raw_v[v])   { cudaFreeHost(host_kv_raw_v[v]);   host_kv_raw_v[v] = nullptr; }
        if (host_kv_raw_ksc[v]) { cudaFreeHost(host_kv_raw_ksc[v]); host_kv_raw_ksc[v] = nullptr; }
        if (host_kv_raw_vsc[v]) { cudaFreeHost(host_kv_raw_vsc[v]); host_kv_raw_vsc[v] = nullptr; }
        kv_desc[v].h_k_f16 = nullptr; kv_desc[v].h_v_f16 = nullptr;
        kv_desc[v].h_k_f8  = nullptr; kv_desc[v].h_v_f8  = nullptr;
        kv_desc[v].h_k_sc  = nullptr; kv_desc[v].h_v_sc  = nullptr;
        kv_desc[v].host_cap = 0;
    }

    void init(int slots, int ctx, int chunk, InkKvDtype kvd = InkKvDtype::FP16, int kv_offload = 0) {
        max_slots = std::max(1, slots);
        max_ctx = std::max(64, ctx);
        kv_offload_mode = kv_offload;
        max_batch = std::max(chunk, std::max(16, max_slots));
        swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + std::max(max_batch, 64));
        kv_dtype = kvd;
        cur_stage = 0;
        cur_moe_ws = &moe_ws;

        INK_CK(cudaSetDevice(m.primary_gpu));
        (void)ink_verify_shared_cfg();   // STRATA_INK_VERIFY_SHARED_KV: read the switch and run its start-up check here, outside any graph capture
        {
            const char* hv = std::getenv("STRATA_INK_HASH");
            if (hv && hv[0] == '1') {
                dbg_hash_on = true;
                if (const char* tv = std::getenv("STRATA_INK_HASH_VISIT")) dbg_tap_visit = std::atoi(tv);
                INK_CK(cudaMalloc(&d_dbg_hash, (size_t)16 * kDbgSlots * 8));
                INK_CK(cudaMemset(d_dbg_hash, 0, (size_t)16 * kDbgSlots * 8));
            }
        }
        INK_CK(cudaStreamCreate(&st));
        INK_CK(cudaEventCreate(&ev_p0));
        INK_CK(cudaEventCreate(&ev_p1));
        INK_CUBLAS_CK(cublasCreate(&cublas));
        INK_CUBLAS_CK(cublasSetStream(cublas, st));
        INK_CUBLAS_CK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
        if (m.all_sf4_vram) {
            constexpr size_t kWsBytes = 4ull << 20;
            d_cublas_ws = dalloc<uint8_t>(kWsBytes);
            INK_CUBLAS_CK(cublasSetWorkspace(cublas, d_cublas_ws, kWsBytes));
        }

        // Auto-fit max_ctx & split Global KV across VRAM + zero-copy mapped pinned host RAM (--kv-offload auto|on|off)!
        {
            size_t vfree = 0, vtot = 0;
            INK_CK(ink_cuda_mem_info(&vfree, &vtot, m.primary_gpu));
            tree_cap = (max_slots <= 4 && vfree > (700ull << 20)) ? kInkMaxTree : 16;
            max_batch = std::max(max_batch, tree_cap);
            const auto& opt5060_r = ink_rtx5060ti_cfg(m.primary_gpu);
            if (!m.all_sf4_vram) {
                if (vfree < (2200ull << 20) && max_batch > 512) {
                    max_batch = 512;
                    swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + max_batch);
                }
                if (vfree < (1200ull << 20) && max_batch > 256) {
                    max_batch = 256;
                    swa_ring = (max_slots >= 16) ? (kInkSwaWin + 32) : (kInkSwaWin + max_batch);
                }
            }
            const size_t per_visit_pos = kv_bytes_per_visit_pos(kv_dtype);
            int n_swa_g0 = 0, n_glob_g0 = 0;
            for (int v = 0; v < kInkVisits; ++v) {
                const int ip = ink_visit_phys(v);
                if (!m.ls_active || m.layer_owner[ip] == 0) {
                    if (m.man.c.is_swa[ip]) ++n_swa_g0; else ++n_glob_g0;
                }
            }
            const size_t swa_kv_bytes = (size_t)n_swa_g0 * max_slots * swa_ring * per_visit_pos;
            const size_t sconv_bytes  = (size_t)(n_swa_g0 + n_glob_g0) * max_slots * 3 * kInkSconvCh * 4;
            const size_t wddm_fl = InkCudaModel::wddm_safety_floor(vtot);
            // the workspaces the runner still allocates after the K/V (strata/core/ink_kv_reserve.hpp: v0.2.20's values)
            const Sm120Cfg& s120_r = sm120_cfg(m.primary_gpu);   // the card's table row supplies the reserve numbers (5090: 768 and 1536 MiB)
            const size_t rtx_extra_res =
                ((s120_r.fp8_cache && !m.ls_active) ? ((size_t)s120_r.ws_extra_fp8_mib << 20) : 0ull) +
                ((s120_r.dspark_opt && !s120_r.exact && max_slots >= 2 && m.man.has_dspark) ? ((size_t)s120_r.ws_extra_dspark_mib << 20) : 0ull) +
                (s120_r.tc_attn ? (size_t)tree_cap * (size_t)std::max(0, tcattn::splits_dev(m.primary_gpu) + 1 - kInkAttnMaxSplits) * kInkHeads * (kInkHeadDim + 2) * 4 : 0ull);   // TC_ATTN: the larger split-K partial buffers (a 5090: 32 x 54 x 24 x 130 x 4 = 13 MiB; a 5060 Ti: none)
            // RTX 5060 Ti row ("compact"): the all-VRAM expert set sizes the workspaces after the K/V itself, so its own, smaller reserve is used.
            // Every other case, with every switch off included, keeps the shipped reserve (the 5060 Ti hand-over applied the smaller one to every
            // all-VRAM load, which moved the KV split from 4352 to 4096 device tokens on an A4000: review finding B1).
            // the small compact reserve was sized with the FP8 KV: COMPACT_KV=0 (FP16 KV) left 43 MiB and crashed at the first top-k, so it takes the shipped reserve
            // DFlash 2 as the drafter (STRATA_INK_DRAFTER=dflash2): its weights, rings and scratch come out of what the K/V may take
            const size_t df2_extra = df2_replaces_dspark(m.path) ? ((opt5060_r.compact_vram && m.all_sf4_vram) ? df2_real_bytes(max_slots)
                                                                                                               : ((df2_total_mib(df2_variant(vtot < (24ull << 30)), max_slots) + 180) << 20)) : 0ull;
            const size_t ws_reserve = df2_extra + ((opt5060_r.compact_vram && opt5060_r.compact_kv && m.all_sf4_vram)
                ? ((size_t)max_batch * (size_t)(520ull << 10) + kCompactWsConstBytes)
                : (strata::core::ink_runner_ws_reserve_bytes(max_slots, m.man.has_dspark) + (wddm_fl / 2) + rtx_extra_res));
            swa_in_host = (opt5060_r.compact_vram && max_slots >= 16 && !m.ls_active);
            std::memset(swa_pinned, 0, sizeof(swa_pinned));
            n_swa_pinned = 0;
            const size_t eff_swa_vram = swa_in_host ? 0u : swa_kv_bytes;
            const size_t avail_glob = (vfree > eff_swa_vram + sconv_bytes + ws_reserve)
                ? (vfree - eff_swa_vram - sconv_bytes - ws_reserve) : 0u;
            const size_t per_pos_glob = (size_t)std::max(1, n_glob_g0) * max_slots * per_visit_pos
                                      + (m.man.has_dspark ? (size_t)4096u : 0u);
            int vram_fit_ctx = (n_glob_g0 > 0)
                ? (int)std::min<size_t>((size_t)max_ctx, (avail_glob / std::max<size_t>(1, per_pos_glob)) & ~(size_t)63)
                : max_ctx;
            if (m.all_sf4_vram) {
                std::fprintf(stderr, "[Ink5060Ti] compact sizing: free %.0f MiB, swa %.1f MiB, sconv %.1f MiB, ws reserve %.1f MiB, %zu B per global position, device KV that fits %d tokens\n",
                             (double)vfree / 1048576.0, (double)eff_swa_vram / 1048576.0, (double)sconv_bytes / 1048576.0, (double)ws_reserve / 1048576.0, per_pos_glob, vram_fit_ctx);
            }
            const int c1_init_dev_cap = m.all_sf4_vram ? 10240 : 8192;
            if (m.all_sf4_vram && max_slots == 1 && kv_dtype != InkKvDtype::FP16 && vram_fit_ctx < std::min(max_ctx, c1_init_dev_cap)) {
                vram_fit_ctx = std::min(max_ctx, c1_init_dev_cap);
            }

            // Check mapped host RAM budget for Host KV Offloading (8 global visits total!)
            const uint64_t host_avail = ink_ram_available_bytes();
            const uint64_t host_os_res = 2ull << 30;
            const size_t per_pos_all_glob = (size_t)8 * max_slots * per_visit_pos;
            const int host_fit_ctx = (host_avail > host_os_res)
                ? (int)std::min<size_t>((size_t)max_ctx, ((size_t)(host_avail - host_os_res) / std::max<size_t>(1, per_pos_all_glob)) & ~(size_t)63)
                : max_ctx;

            // COMPACT never ends with less than ~128 MiB free (rows lane 11-26): the device K/V is the elastic part of what compact allocates, so when the workspaces, the SWA ring and the
            // K/V of the floor above would leave less, the K/V that does not fit goes to mapped host memory (the Auto Host KV Offload below). An RTX 5080 (driver 595, CUDA 13.0) ended with
            // 104 MiB free after the floor's 7680 device tokens and its engine died at the first top-k (ink_cuda_backend.cu:995, "Tree CUDA Graph capture unavailable (err=2)"); an RTX 5070 Ti
            // with 149 MiB free and the 5060 Ti (140 MiB) ran. 128 MiB gives a margin over the card that failed by 24 MiB and keeps all of the K/V on cards that have the room.
            if (m.all_sf4_vram && max_slots == 1 && kv_offload_mode == 0 && host_fit_ctx >= 256) {
                const double keep = 128.0 * 1048576.0;
                const double room = (double)vfree - (double)eff_swa_vram - (double)sconv_bytes - (double)ws_reserve - keep;
                const int safe_dev = room > 0.0 ? ((int)std::min<double>((double)(1 << 20), room / (double)std::max<size_t>(1, per_pos_glob)) & ~63) : 0;
                const int dev_target = (vram_fit_ctx >= max_ctx) ? std::min(max_ctx, c1_init_dev_cap)
                                                                 : (vram_fit_ctx >= 1024 ? ((vram_fit_ctx * 3 / 4) & ~255) : 0);
                if (dev_target > safe_dev) {
                    std::fprintf(stderr, "[Ink5060Ti] compact keeps 128 MiB free: device K/V %d -> %d tokens, the rest in mapped host memory\n", dev_target, safe_dev >= 768 ? safe_dev : 0);
                    vram_fit_ctx = (safe_dev >= 768) ? std::min(max_ctx - 1, safe_dev * 4 / 3) : 0;
                }
            }
            const int eff_kv_offload_mode = swa_in_host ? 1 : kv_offload_mode;
            if (ink_elastic_1m_enabled(m.primary_gpu) && !m.ls_active) {
                elastic_kv.plan(m.primary_gpu, max_slots, max_ctx, kv_dtype, vram_fit_ctx, eff_kv_offload_mode,
                                host_avail, kv_dev_ctx, kv_host_ctx, max_ctx);
                kv_alloc_ctx = kv_dev_ctx + kv_host_ctx;
            } else if (eff_kv_offload_mode == 1 && host_fit_ctx >= 256) {
                if (host_fit_ctx < max_ctx) max_ctx = std::max(256, host_fit_ctx);
                kv_dev_ctx = 0;
                kv_host_ctx = max_ctx;
                kv_alloc_ctx = max_ctx;
                std::fprintf(stderr, "[InkCUDA] Host KV Offload ON (kv_fmt=%d): 8 global visits = 0 VRAM + %d mapped host tokens (%.1f MiB host RAM)\n",
                             (int)kv_dtype, kv_host_ctx, (double)((size_t)kv_host_ctx * per_pos_all_glob) / (1024.0 * 1024.0));
            } else if (vram_fit_ctx >= max_ctx) {
                kv_dev_ctx = std::max(256, std::min(max_ctx, (max_slots == 1) ? c1_init_dev_cap : 4096));
                kv_host_ctx = 0;
                kv_alloc_ctx = kv_dev_ctx;
            } else if (kv_offload_mode == 0 && host_fit_ctx >= 256) {
                int dev_part = (vram_fit_ctx >= 1024) ? ((vram_fit_ctx * 3 / 4) & ~255) : 0;
                int host_part = std::min(max_ctx - dev_part, host_fit_ctx);
                if (dev_part + host_part < max_ctx) {
                    max_ctx = std::max(256, dev_part + host_part);
                    host_part = std::max(0, max_ctx - dev_part);
                }
                kv_dev_ctx = dev_part;
                kv_host_ctx = host_part;
                kv_alloc_ctx = kv_dev_ctx + kv_host_ctx;
                std::fprintf(stderr, "[InkCUDA] Auto Host KV Offload (kv_fmt=%d, %.2f GiB free VRAM): global KV = %d VRAM + %d mapped host tokens (%.1f MiB host RAM, max_ctx=%d)\n",
                             (int)kv_dtype, (double)vfree / (1024.0 * 1024.0 * 1024.0),
                             kv_dev_ctx, kv_host_ctx,
                             (double)((size_t)kv_host_ctx * per_pos_all_glob) / (1024.0 * 1024.0), max_ctx);
            } else {
                const int clamped = std::max(256, vram_fit_ctx);
                if (clamped < max_ctx) {
                    std::fprintf(stderr, "[InkCUDA] Auto-sizing max_ctx %d -> %d for parallel=%d (%.2f GiB free VRAM)\n",
                                 max_ctx, clamped, max_slots, (double)vfree / (1024.0 * 1024.0 * 1024.0));
                    max_ctx = clamped;
                }
                kv_dev_ctx = std::max(256, std::min(max_ctx, (max_slots == 1) ? c1_init_dev_cap : 4096));
                kv_host_ctx = 0;
                kv_alloc_ctx = kv_dev_ctx;
            }
        }

        // Allocate KV caches & ShortConv state pools on each visit's owning GPU (+ mapped pinned host tail for global visits)!
        // Under m.all_sf4_vram (opt-in STRATA_INK_RTX5060TI=1), sub-allocate all device KV + ShortConv buffers from ONE single
        // 256-byte-aligned slab to eliminate ~240 separate 2 MiB cudaMalloc rounding slabs (~240+ MiB reclaimed)!
        const bool use_slab = m.all_sf4_vram && !m.ls_active;
        size_t kv_slab_off = 0;
        auto kv_bump = [&](size_t bytes) -> size_t {
            const size_t o = kv_slab_off;
            kv_slab_off += (bytes + 255ull) & ~255ull;
            return o;
        };
        if (use_slab) {
            for (int v = 0; v < kInkVisits; ++v) {
                const int ip = ink_visit_phys(v);
                const bool swa = m.man.c.is_swa[ip] != 0;
                const int dcap = swa ? swa_ring : kv_dev_ctx;
                if (dcap > 0 && !(swa && swa_in_host && !swa_pinned[v])) {
                    const size_t elems = (size_t)max_slots * dcap * kInkKVHeads * kInkHeadDim;
                    if (kv_dtype == InkKvDtype::FP16) {
                        kv_bump(elems * sizeof(__half));
                        kv_bump(elems * sizeof(__half));
                    } else {
                        const size_t v_b = (kv_dtype == InkKvDtype::K8V4) ? (elems / 2u) : elems;
                        const size_t sc_b = (size_t)max_slots * dcap * kInkKVHeads * sizeof(float);
                        kv_bump(elems);
                        kv_bump(v_b);
                        kv_bump(sc_b);
                        kv_bump(sc_b);
                    }
                }
                kv_bump((size_t)max_slots * 3 * kInkSconvCh * sizeof(float));
            }
            if (kv_slab_off > 0) {
                INK_CK(cudaSetDevice(m.primary_gpu));
                d_kv_slab = dalloc<uint8_t>(kv_slab_off);
            }
            kv_slab_off = 0;
        }
        auto kv_slice = [&](size_t bytes) -> uint8_t* {
            const size_t o = kv_bump(bytes);
            return d_kv_slab + o;
        };
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const int dev = m.ls_active ? m.stage_gpu[m.layer_owner[ip]] : m.primary_gpu;
            INK_CK(cudaSetDevice(dev));
            const bool swa = m.man.c.is_swa[ip] != 0;
            const int dcap = swa ? swa_ring : kv_dev_ctx;
            const int hcap = swa ? 0 : kv_host_ctx;
            kv_desc[v] = {};
            kv_desc[v].cap_pos  = dcap + hcap;
            kv_desc[v].dev_cap  = dcap;
            kv_desc[v].host_cap = hcap;
            kv_desc[v].is_swa   = swa;
            kv_desc[v].kv_fmt   = (uint8_t)kv_dtype;
            if (dcap > 0) {
                const size_t elems = (size_t)max_slots * dcap * kInkKVHeads * kInkHeadDim;
                if (swa && swa_in_host && !swa_pinned[v]) {
                    bool ok_sw = false;
                    if (kv_dtype == InkKvDtype::FP16) {
                        ok_sw = alloc_mapped_host<__half>(elems, host_kv_raw_k[v], kv_desc[v].k_f16)
                             && alloc_mapped_host<__half>(elems, host_kv_raw_v[v], kv_desc[v].v_f16);
                    } else {
                        const size_t v_b  = (kv_dtype == InkKvDtype::K8V4) ? (elems / 2u) : elems;
                        const size_t sc_n = (size_t)max_slots * dcap * kInkKVHeads;
                        ok_sw = alloc_mapped_host<uint8_t>(elems, host_kv_raw_k[v], kv_desc[v].k_f8)
                             && alloc_mapped_host<uint8_t>(v_b,   host_kv_raw_v[v], kv_desc[v].v_f8)
                             && alloc_mapped_host<float>(sc_n,    host_kv_raw_ksc[v], kv_desc[v].k_sc)
                             && alloc_mapped_host<float>(sc_n,    host_kv_raw_vsc[v], kv_desc[v].v_sc);
                    }
                    if (!ok_sw) {
                        ink_alloc_failed(cudaErrorMemoryAllocation, elems, __FILE__, __LINE__);
                    }
                } else if (use_slab) {
                    if (kv_dtype == InkKvDtype::FP16) {
                        kv_desc[v].k_f16 = reinterpret_cast<__half*>(kv_slice(elems * sizeof(__half)));
                        kv_desc[v].v_f16 = reinterpret_cast<__half*>(kv_slice(elems * sizeof(__half)));
                    } else {
                        const size_t v_b = (kv_dtype == InkKvDtype::K8V4) ? (elems / 2u) : elems;
                        const size_t sc_b = (size_t)max_slots * dcap * kInkKVHeads * sizeof(float);
                        kv_desc[v].k_f8 = kv_slice(elems);
                        kv_desc[v].v_f8 = kv_slice(v_b);
                        kv_desc[v].k_sc = reinterpret_cast<float*>(kv_slice(sc_b));
                        kv_desc[v].v_sc = reinterpret_cast<float*>(kv_slice(sc_b));
                    }
                } else if (kv_dtype == InkKvDtype::FP16) {
                    kv_desc[v].k_f16 = dalloc<__half>(elems);
                    kv_desc[v].v_f16 = dalloc<__half>(elems);
                } else {
                    kv_desc[v].k_f8 = dalloc<uint8_t>(elems);
                    kv_desc[v].v_f8 = dalloc<uint8_t>((kv_dtype == InkKvDtype::K8V4) ? (elems / 2u) : elems);
                    kv_desc[v].k_sc = dalloc<float>((size_t)max_slots * dcap * kInkKVHeads);
                    kv_desc[v].v_sc = dalloc<float>((size_t)max_slots * dcap * kInkKVHeads);
                }
            }
            if (hcap > 0) {
                const size_t helems = (size_t)max_slots * hcap * kInkKVHeads * kInkHeadDim;
                bool hok = true;
                if (kv_dtype == InkKvDtype::FP16) {
                    hok = alloc_mapped_host<__half>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f16)
                       && alloc_mapped_host<__half>(helems, host_kv_raw_v[v], kv_desc[v].h_v_f16);
                } else {
                    const size_t v_b = (kv_dtype == InkKvDtype::K8V4) ? (helems / 2u) : helems;
                    const size_t sc_n = (size_t)max_slots * hcap * kInkKVHeads;
                    hok = alloc_mapped_host<uint8_t>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f8)
                       && alloc_mapped_host<uint8_t>(v_b, host_kv_raw_v[v], kv_desc[v].h_v_f8)
                       && alloc_mapped_host<float>(sc_n, host_kv_raw_ksc[v], kv_desc[v].h_k_sc)
                       && alloc_mapped_host<float>(sc_n, host_kv_raw_vsc[v], kv_desc[v].h_v_sc);
                }
                if (!hok) {
                    free_host_kv_visit(v);
                    kv_desc[v].cap_pos = dcap;
                    kv_host_ctx = 0;
                    kv_alloc_ctx = dcap;
                    max_ctx = std::max(256, dcap);
                    std::fprintf(stderr, "[InkCUDA] Warning: mapped host KV alloc failed at visit %d; falling back to dev_cap=%d\n", v, dcap);
                }
            }
            const size_t sc_bytes = (size_t)max_slots * 3 * kInkSconvCh * sizeof(float);
            d_sconv_pool[v] = use_slab ? reinterpret_cast<float*>(kv_slice(sc_bytes))
                                       : dalloc<float>((size_t)max_slots * 3 * kInkSconvCh);
            INK_CK(cudaMemset(d_sconv_pool[v], 0, sc_bytes));
        }
        INK_CK(cudaSetDevice(m.primary_gpu));

        const size_t MB = (size_t)max_batch;
        const size_t sp_cap = std::max<size_t>(std::max<size_t>((size_t)kInkAttnMaxSplits, (size_t)tree_cap * kInkAttnMaxSplits),   // #6671: a tree uses the M == 1 split count
                                               sm120_cfg(m.primary_gpu).tc_attn ? (size_t)tree_cap * (size_t)(tcattn::splits_dev(m.primary_gpu) + 1) : (size_t)0);   // TC_ATTN: the card's slices + the verify tail per head (85 on a 5090, 18 on a 5060 Ti)
        const size_t x_bf16_elems = m.all_sf4_vram
            ? std::max<size_t>(MB * (size_t)(3 * kInkHidden), 256ull * kInkDsparkInter)
            : (MB * (size_t)std::max(kInkDsparkInter, std::max(kInkQDim, kInkHidden)));
        const size_t w_scr_elems = (m.all_sf4_vram && ink_rtx5060ti_cfg(m.primary_gpu).sm120_tc_smallm)
            ? (64ull * 64ull)
            : (m.all_sf4_vram
                ? ((size_t)kInkQDim * kInkHidden)
                : ((size_t)kInkDsparkInter * kInkHidden));
        const size_t mpad = (size_t)((max_batch + 7) / 8) * 8;
        const bool skip_bf16_screen = m.all_sf4_vram && (m.tinfo("model.llm.unembed.weight").fmt == FMT_Q8ROW);

        screen_cap = (rtx5090::dspark_opt_on() && max_slots <= 4) ? 24 : 8;   // 5090 drafting: 24-row screened argmax; shipped: 8
        size_t act_slab_off = 0;
        auto bind_acts = [&](int pass) {
            act_slab_off = 0;
            auto al = [&](auto*& ptr, size_t n) {
                using T = typename std::remove_pointer<typename std::remove_reference<decltype(ptr)>::type>::type;
                const size_t bytes = n * sizeof(T);
                if (!use_slab) {
                    if (pass == 1) ptr = dalloc<T>(n);
                    return;
                }
                const size_t o = act_slab_off;
                act_slab_off += (bytes + 255ull) & ~255ull;
                if (pass == 1) ptr = reinterpret_cast<T*>(d_act_slab + o);
            };
            al(d_cur, MB * kInkHidden);
            al(d_res, MB * kInkHidden);
            al(d_prelude, MB * kInkHidden);
            al(d_bank_out, MB * kInkHidden);
            for (int i = 0; i < 8; ++i) al(d_blocks[i], MB * kInkHidden);
            for (int i = 0; i < 2; ++i) al(d_prelude_blocks[i], MB * kInkHidden);
            for (int i = 0; i < 4; ++i) al(d_bank[i], MB * kInkHidden);
            al(d_mixed, 4 * MB * kInkHidden);
            al(d_msgs, 4 * MB * kInkHidden);
            al(d_inp0, MB * kInkHidden);
            al(d_inp1, MB * kInkHidden);
            al(d_inp2, MB * kInkHidden);
            al(d_q, MB * kInkQDim);
            al(d_k_raw, MB * kInkKVDim);
            al(d_v_raw, MB * kInkKVDim);
            al(d_k, MB * kInkKVDim);
            al(d_v, MB * kInkKVDim);
            al(d_r, MB * kInkHeads * kInkDRel);
            al(d_k_dyn, MB * kInkKVHeads * 4);
            al(d_v_dyn, MB * kInkKVHeads * 4);
            al(d_sinks, MB * kInkHeads);
            al(d_gate_attn, MB * kInkQDim);
            al(d_rel_table, ((m.fp8_trunk.ready && !m.ls_active && max_slots == 1) ? 32ull : MB) * kInkHeads * (kInkRelExtent + 1));
            al(d_attn_out, MB * kInkQDim);
            al(d_wo_raw, MB * kInkHidden);
            al(d_wo_conv, MB * kInkHidden);
            al(d_moe_in, MB * kInkHidden);
            al(d_moe_raw, MB * kInkHidden);
            al(d_moe_conv, MB * kInkHidden);
            al(d_h_norm, MB * kInkHidden);
            for (int i = 0; i < 4; ++i) {
                al(d_taps[i], MB * kInkHidden);
                al(d_taps_compact[i], (size_t)tree_cap * kInkHidden);
            }
            al(d_ctrl_cat, (m.fp8_trunk.ready ? 48ull : (m.all_sf4_vram ? std::max<size_t>(48ull, MB) : (3ull * MB))) * 2 * kInkHidden);
            al(d_ctrl_gate, 3ull * MB * kInkGateRank);
            al(d_ctrl_gain, m.all_sf4_vram ? std::max<size_t>(6144ull, MB * 64ull) : (MB * kInkHidden));
            al(d_ctrl_forget, m.all_sf4_vram ? std::max<size_t>(6144ull, MB * 64ull) : (MB * kInkHidden));
            al(d_attn_pm, sp_cap * kInkHeads);
            al(d_attn_pl, sp_cap * kInkHeads);
            al(d_attn_po, sp_cap * kInkHeads * kInkHeadDim);
            if (!(use_slab && max_slots >= 16)) {
                al(d_stage_k, (size_t)kInkVisits * tree_cap * kInkKVDim);
                al(d_stage_v, (size_t)kInkVisits * tree_cap * kInkKVDim);
                al(d_stage_sconv, (size_t)kInkVisits * tree_cap * kInkSconvCh);
            }
            al(d_kv_desc_arr, kInkVisits);
            al(d_sconv_pool_arr, kInkVisits);
            al(d_acc_path, kInkMaxTree);
            al(d_x_bf16, x_bf16_elems);
            al(d_w_scratch, w_scr_elems);
            al(d_xq8, mpad * kInkQDim);
            al(d_xs8, mpad * (kInkQDim / 32));
            if (!skip_bf16_screen) {
                al(d_dot_hat, (size_t)screen_cap * kInkVocabUnpad);
                al(d_h_norm2, (size_t)screen_cap);
                al(d_max_lb, (size_t)screen_cap);
                al(d_cand_ids, (size_t)screen_cap * 4096);
                al(d_cand_cnt, (size_t)screen_cap);
            }
            al(d_argmax, std::max(64, max_batch));
            al(d_tok_ids, MB);
            al(d_slots, MB);
            al(d_pos, MB);
            al(d_is_img, MB);
            al(d_img_emb, (m.all_sf4_vram ? 1ull : MB) * kInkHidden);
            al(d_sconv_lag, kInkMaxTree * 3);
            al(d_tree_anc, kInkMaxTree * kInkMaxTreeDepth);
            if (rtx5090::deep_tree_on()) al(d_tree_anc32, kInkMaxTree * kInkMaxTree);   // deep verify tree (RTX 50 fast mode)
            al(d_tree_dep, kInkMaxTree);
        };
        if (use_slab) {
            bind_acts(0);
            d_act_slab = dalloc<uint8_t>(act_slab_off);
        }
        bind_acts(1);
        INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
        INK_CK(cudaMemcpy(d_sconv_pool_arr, d_sconv_pool, sizeof(d_sconv_pool), cudaMemcpyHostToDevice));

        // Warmup cuBLAS Tensor Cores once at init (eliminates 61 ms first-call JIT tax from prompt #1!)
        {
            INK_CK(cudaMemsetAsync(d_w_scratch, 0, 64 * 64 * sizeof(__nv_bfloat16), st));
            INK_CK(cudaMemsetAsync(d_x_bf16, 0, 64 * 64 * sizeof(__nv_bfloat16), st));
            INK_CK(cudaMemsetAsync(d_ctrl_cat, 0, 64 * 64 * sizeof(float), st));
            const float a = 1.0f, b = 0.0f;
            if (m.sm_cc >= 80) {
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                             d_w_scratch, CUDA_R_16BF, 64, d_x_bf16, CUDA_R_16BF, 64, &b,
                             d_ctrl_gain, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else if (m.sm_cc >= 70) {
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                             d_w_scratch, CUDA_R_16F, 64, d_x_bf16, CUDA_R_16F, 64, &b,
                             d_ctrl_gain, CUDA_R_32F, 64, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, 64, &a,
                         d_ctrl_cat, CUDA_R_32F, 64, d_ctrl_cat, CUDA_R_32F, 64, &b,
                         d_ctrl_gain, CUDA_R_32F, 64,
                         m.sm_cc >= 80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            INK_CK(cudaStreamSynchronize(st));
        }

        meta_ring_stride = std::max<size_t>(MB * 3 + 256, d_tree_anc32 ? 1024 : 512);
        meta_ring_idx = 0;
        h_meta_pin  = halloc_pinned<int32_t>(64ull * meta_ring_stride);

        moe_ws.sm120_tc_smallm = ink_rtx5060ti_cfg(m.primary_gpu).sm120_tc_smallm;
        moe_ws.sm120_dev = m.primary_gpu;
        moe_ws.alloc(max_batch, m.all_sf4_vram && moe_ws.sm120_tc_smallm);
        // Opt-in RTX 5090 pre-packed fused BF16 QKV+R+Gate & BF16 Loop Controllers (OFF by default!)
        if (rtx5090::pf_attn_on() && !m.ls_active) {
            constexpr size_t kRelElems = (size_t)kInkRelExtent * (kInkDRel / 2); // 1024 * 8 half2
            d_rtx_rel_proj_pool = dalloc<__half2>((size_t)m.man.c.n_phys * kRelElems);
            for (int ip = 0; ip < m.man.c.n_phys; ++ip) {
                d_rtx_rel_proj_T[ip] = d_rtx_rel_proj_pool + (size_t)ip * kRelElems;
                const std::string rn = "model.llm.layers." + std::to_string(ip) + ".attn.rel_logits_proj.proj";
                rtx5090::k_ink_transpose_rel_proj_h2<<<(kInkRelExtent + 255) / 256, 256, 0, st>>>(
                    (const float*)m.trunk_ptr(rn), d_rtx_rel_proj_T[ip]);
            }
            INK_CK(cudaStreamSynchronize(st));
        }
        if (rtx5090::fp8_trunk_on() && !m.ls_active) {
            // the FP8 copy is built only when what comes after it still fits: DFlash 2's weights, rings and scratch (the drafter is built after the runner) and a 256 MiB crash margin
            size_t fp8_later = 0;
            if (m.man.has_dspark && df2_replaces_dspark(m.path)) {
                size_t vf_l = 0, vt_l = 0;
                if (ink_cuda_mem_info(&vf_l, &vt_l, m.primary_gpu) == cudaSuccess)
                    fp8_later = (df2_total_mib(df2_variant(vt_l < (24ull << 30)), max_slots) + 256ull) << 20;
            }
            fp8_eng.init(m, max_batch, st, fp8_later);
        }
        if (rtx5090::moe_opt_on() && !rtx5090::exact_mode_on() && !m.ls_active && !m.ep2_active && !m.hybrid.active) {
            rtx5090::transcode_mx8_to_sf4_inplace(m, st);
        }
        if (rtx5090::dense_opt_on() && !fp8_eng.cache_ready && !m.ls_active) {
            size_t vf = 0, vt = 0;
            INK_CK(ink_cuda_mem_info(&vf, &vt, m.primary_gpu));
            constexpr size_t kRDim = (size_t)(kInkHeads * kInkDRel);
            constexpr size_t kQkvCols = (size_t)(kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim);  // 8576
            const size_t qkv_elems = (size_t)m.man.c.n_phys * kQkvCols * kInkHidden;
            bool all_bf16 = true;
            for (int ip = 0; ip < m.man.c.n_phys && all_bf16; ++ip) {
                const std::string p = "model.llm.layers." + std::to_string(ip);
                if (m.tinfo(p + ".attn.wq_du.weight").fmt != FMT_BF16 ||
                    m.tinfo(p + ".attn.wr_du.weight").fmt != FMT_BF16 ||
                    m.tinfo(p + ".attn.gate_attn.weight").fmt != FMT_BF16 ||
                    m.tinfo(p + ".attn.wk_dv.weight").fmt != FMT_BF16 ||
                    m.tinfo(p + ".attn.wv_dv.weight").fmt != FMT_BF16) all_bf16 = false;
            }
            if (all_bf16 && vf > qkv_elems * 2 + (1536ull << 20)) {
                d_rtx_qkv_pool = dalloc<__nv_bfloat16>(qkv_elems);
                for (int ip = 0; ip < m.man.c.n_phys; ++ip) {
                    const std::string p = "model.llm.layers." + std::to_string(ip);
                    __nv_bfloat16* dst = d_rtx_qkv_pool + (size_t)ip * kQkvCols * kInkHidden;
                    d_rtx_qkv[ip] = dst;
                    size_t off = 0;
                    auto cp = [&](const std::string& tn, size_t rows) {
                        INK_CK(cudaMemcpyAsync(dst + off, m.trunk_ptr(tn), rows * kInkHidden * 2, cudaMemcpyDeviceToDevice, st));
                        off += rows * kInkHidden;
                    };
                    cp(p + ".attn.wq_du.weight",     kInkQDim);
                    cp(p + ".attn.wr_du.weight",     kRDim);
                    cp(p + ".attn.gate_attn.weight", kInkQDim);
                    cp(p + ".attn.wk_dv.weight",     kInkKVDim);
                    cp(p + ".attn.wv_dv.weight",     kInkKVDim);
                }
                vf -= qkv_elems * 2;
            }
            const int n_core = m.man.c.core_end - m.man.c.core_start;
            constexpr size_t kGfEl = (size_t)(2 * kInkHidden) * kInkGateRank;
            const size_t ctrl_elems = (size_t)n_core * 4 * kGfEl;
            if (n_core > 0 && n_core <= 12 && vf > ctrl_elems * 4 + (768ull << 20)) {
                d_rtx_ctrl_pool = dalloc<float>(ctrl_elems);
                static const char* kCN[4] = {"attn", "k", "v", "mlp"};
                size_t cur_c = 0;
                for (int ci = 0; ci < n_core; ++ci) {
                    const std::string p = "model.llm.layers." + std::to_string(m.man.c.core_start + ci);
                    for (int k = 0; k < 4; ++k) {
                        const std::string cp = p + ".controllers." + kCN[k];
                        d_rtx_ctrl_gf[ci][k] = d_rtx_ctrl_pool + cur_c; cur_c += kGfEl;
                        const size_t hgf = kGfEl / 2;
                        INK_CK(cudaMemcpyAsync(d_rtx_ctrl_gf[ci][k],       m.trunk_ptr(cp + ".gain_weight"),   hgf * 4, cudaMemcpyDeviceToDevice, st));
                        INK_CK(cudaMemcpyAsync(d_rtx_ctrl_gf[ci][k] + hgf, m.trunk_ptr(cp + ".forget_weight"), hgf * 4, cudaMemcpyDeviceToDevice, st));
                    }
                }
            }
            INK_CK(cudaStreamSynchronize(st));
        }
        // Pre-allocate O(1)-VRAM Tensor Core prefill attention workspace (~114 MiB fixed forever!)
        if (!rtx5090::pf_attn_on() && m.sm_cc >= 60 && max_slots <= 4 && max_batch >= 128) {
            size_t vf = 0, vt = 0;
            INK_CK(ink_cuda_mem_info(&vf, &vt, m.primary_gpu));
            if (vf > (m.all_sf4_vram ? (160ull << 20) : (320ull << 20))) ensure_pf_attn(kInkPfKvTile);
        }
        if (m.fp8_trunk.ready && !m.ls_active) {
            fp8_sm.init(m, m.fp8_trunk, max_batch, st);
        }

        if (m.ls_active) {
            use_cuda_graph = false;
            StageExec& s0 = ls_stages[0];
            s0.st = st; s0.cublas = cublas;
            INK_CK(cudaEventCreateWithFlags(&s0.ev_done, cudaEventDisableTiming));
            s0.d_cur = d_cur; s0.d_res = d_res; s0.d_bank_out = d_bank_out;
            for (int i = 0; i < 8; ++i) s0.d_blocks[i] = d_blocks[i];
            for (int i = 0; i < 4; ++i) s0.d_bank[i] = d_bank[i];
            s0.d_mixed = d_mixed; s0.d_msgs = d_msgs; s0.d_inp0 = d_inp0; s0.d_inp1 = d_inp1; s0.d_inp2 = d_inp2;
            s0.d_q = d_q; s0.d_k_raw = d_k_raw; s0.d_v_raw = d_v_raw; s0.d_k = d_k; s0.d_v = d_v; s0.d_r = d_r;
            s0.d_k_dyn = d_k_dyn; s0.d_v_dyn = d_v_dyn;
            s0.d_sinks = d_sinks; s0.d_gate_attn = d_gate_attn; s0.d_rel_table = d_rel_table;
            s0.d_attn_out = d_attn_out; s0.d_wo_raw = d_wo_raw; s0.d_wo_conv = d_wo_conv;
            s0.d_moe_in = d_moe_in; s0.d_moe_raw = d_moe_raw; s0.d_moe_conv = d_moe_conv;
            s0.d_ctrl_cat = d_ctrl_cat; s0.d_ctrl_gate = d_ctrl_gate; s0.d_ctrl_gain = d_ctrl_gain; s0.d_ctrl_forget = d_ctrl_forget;
            s0.d_attn_pm = d_attn_pm; s0.d_attn_pl = d_attn_pl; s0.d_attn_po = d_attn_po;
            s0.d_pf_q = d_pf_q; s0.d_pf_k = d_pf_k; s0.d_pf_v = d_pf_v; s0.d_pf_p = d_pf_p;
            s0.d_pf_s = d_pf_s; s0.d_pf_o = d_pf_o; s0.d_pf_ot = d_pf_ot; s0.d_pf_m = d_pf_m; s0.d_pf_l = d_pf_l;
            s0.pf_kv_cap = pf_kv_cap;
            s0.d_x_bf16 = d_x_bf16; s0.d_w_scratch = d_w_scratch;
            s0.d_slots = d_slots; s0.d_pos = d_pos;

            INK_CK(cudaHostAlloc(&h_ls_bounce, MB * kInkHidden * sizeof(float), cudaHostAllocPortable));
            INK_CK(cudaHostAlloc(&h_ls_meta_bounce, MB * sizeof(int32_t), cudaHostAllocPortable));

            for (int s = 1; s < m.n_stages; ++s) {
                INK_CK(cudaSetDevice(m.stage_gpu[s]));
                StageExec& se = ls_stages[s];
                INK_CK(cudaStreamCreate(&se.st));
                INK_CK(cudaEventCreateWithFlags(&se.ev_done, cudaEventDisableTiming));
                INK_CUBLAS_CK(cublasCreate(&se.cublas));
                INK_CUBLAS_CK(cublasSetStream(se.cublas, se.st));
                INK_CUBLAS_CK(cublasSetMathMode(se.cublas, CUBLAS_DEFAULT_MATH));
                se.d_cur = dalloc<float>(MB * kInkHidden);
                se.d_res = dalloc<float>(MB * kInkHidden);
                se.d_bank_out = dalloc<float>(MB * kInkHidden);
                for (int i = 0; i < 8; ++i) se.d_blocks[i] = dalloc<float>(MB * kInkHidden);
                for (int i = 0; i < 4; ++i) se.d_bank[i] = dalloc<float>(MB * kInkHidden);
                se.d_mixed = dalloc<float>(4 * MB * kInkHidden);
                se.d_msgs  = dalloc<float>(4 * MB * kInkHidden);
                se.d_inp0  = dalloc<float>(MB * kInkHidden);
                se.d_inp1  = dalloc<float>(MB * kInkHidden);
                se.d_inp2  = dalloc<float>(MB * kInkHidden);
                se.d_q     = dalloc<float>(MB * kInkQDim);
                se.d_k_raw = dalloc<float>(MB * kInkKVDim);
                se.d_v_raw = dalloc<float>(MB * kInkKVDim);
                se.d_k     = dalloc<float>(MB * kInkKVDim);
                se.d_v     = dalloc<float>(MB * kInkKVDim);
                se.d_r     = dalloc<float>(MB * kInkHeads * kInkDRel);
                se.d_k_dyn = dalloc<float>(MB * kInkKVHeads * 4);
                se.d_v_dyn = dalloc<float>(MB * kInkKVHeads * 4);
                se.d_sinks = dalloc<float>(MB * kInkHeads);
                se.d_gate_attn = dalloc<float>(MB * kInkQDim);
                se.d_rel_table = dalloc<float>(MB * kInkHeads * (kInkRelExtent + 1));
                se.d_attn_out  = dalloc<float>(MB * kInkQDim);
                se.d_wo_raw    = dalloc<float>(MB * kInkHidden);
                se.d_wo_conv   = dalloc<float>(MB * kInkHidden);
                se.d_moe_in    = dalloc<float>(MB * kInkHidden);
                se.d_moe_raw   = dalloc<float>(MB * kInkHidden);
                se.d_moe_conv  = dalloc<float>(MB * kInkHidden);
                se.d_ctrl_cat  = dalloc<float>(3ull * MB * 2 * kInkHidden);
                se.d_ctrl_gate = dalloc<float>(3ull * MB * kInkGateRank);
                se.d_ctrl_gain = dalloc<float>(MB * kInkHidden);
                se.d_ctrl_forget = dalloc<float>(MB * kInkHidden);
                se.d_attn_pm   = dalloc<float>(sp_cap * kInkHeads);
                se.d_attn_pl   = dalloc<float>(sp_cap * kInkHeads);
                se.d_attn_po   = dalloc<float>(sp_cap * kInkHeads * kInkHeadDim);
                se.d_x_bf16    = dalloc<__nv_bfloat16>(MB * std::max(kInkDsparkInter, std::max(kInkQDim, kInkHidden)));
                se.d_w_scratch = dalloc<__nv_bfloat16>((size_t)kInkDsparkInter * kInkHidden);
                se.d_slots     = dalloc<int32_t>(MB);
                se.d_pos       = dalloc<int32_t>(MB);
                se.ws.alloc(max_batch);
            }
            INK_CK(cudaSetDevice(m.primary_gpu));
        } else if (m.ep2_active && m.peer_gpu >= 0) {
            use_cuda_graph = false;
            INK_CK(cudaSetDevice(m.primary_gpu));
            INK_CK(cudaEventCreateWithFlags(&ev_moe_ready, cudaEventDisableTiming));
            d_peer_y_rx = dalloc<float>(MB * kInkHidden);
            INK_CK(cudaSetDevice(m.peer_gpu));
            INK_CK(cudaStreamCreateWithFlags(&peer_st, cudaStreamNonBlocking));
            INK_CK(cudaEventCreateWithFlags(&ev_peer_done, cudaEventDisableTiming));
            peer_ws.alloc(max_batch);
            d_peer_x = dalloc<float>(MB * kInkHidden);
            d_peer_y = dalloc<float>(MB * kInkHidden);
            INK_CK(cudaSetDevice(m.primary_gpu));
            if (!m.ep2_p2p) {
                INK_CK(cudaHostAlloc(&h_ep_x,   MB * kInkHidden * sizeof(float),   cudaHostAllocPortable));
                INK_CK(cudaHostAlloc(&h_ep_idx, MB * kInkTopK   * sizeof(int32_t), cudaHostAllocPortable));
                INK_CK(cudaHostAlloc(&h_ep_w,   MB * kInkTopK   * sizeof(float),   cudaHostAllocPortable));
                INK_CK(cudaHostAlloc(&h_ep_y,   MB * kInkHidden * sizeof(float),   cudaHostAllocPortable));
            }
        }
    }

    // Elastic Global KV Cache Growth: starts compact (4K..8K) and grows in 8K-token chunks up to max_ctx!
    // Returns false cleanly if VRAM is exhausted (NEVER aborts with exit(3)!).
    bool ensure_global_kv_cap(int needed_pos) {
        if (elastic_kv.enabled) return elastic_kv.ensure_cap(*this, needed_pos);
        if (needed_pos <= kv_alloc_ctx) return true;
        if (needed_pos > max_ctx) return false;
        const int old_cap = kv_alloc_ctx;
        int new_cap = strata::core::ink_kv_next_cap(old_cap, needed_pos, max_ctx);

        const size_t per_pos = kv_bytes_per_visit_pos(kv_dtype);
        for (int s = 0; s < (m.ls_active ? m.n_stages : 1); ++s) {
            int n_glob_s = 0;
            for (int v = 0; v < kInkVisits; ++v) {
                const int ip = ink_visit_phys(v);
                if (!m.man.c.is_swa[ip] && (!m.ls_active || m.layer_owner[ip] == s)) ++n_glob_s;
            }
            if (n_glob_s == 0) continue;
            size_t vf = 0, vt = 0;
            const int dev = m.ls_active ? m.stage_gpu[s] : m.primary_gpu;
            if (ink_cuda_mem_info(&vf, &vt, dev) != cudaSuccess) return false;
            const size_t floor_b = 256ull << 20;
            // #6770: the visits grow one at a time (new buffer, copy, free the old one), so the card needs the net growth
            // plus one held old visit, not the whole new K/V next to the old one (that refused growth at about half the context).
            if (!strata::core::ink_kv_grow_fits(vf, floor_b, n_glob_s, max_slots, old_cap, new_cap, per_pos)) {
                const int tight_cap = std::min(max_ctx, ((needed_pos + 255) & ~255));
                if (tight_cap <= old_cap ||
                    !strata::core::ink_kv_grow_fits(vf, floor_b, n_glob_s, max_slots, old_cap, tight_cap, per_pos)) {
                    // VRAM cannot grow global KV further: seamlessly spill tail [kv_dev_ctx..new_cap-1] into mapped pinned host RAM!
                    // The whole rest of the context first, so the K/V never has to grow again (a tail sized to just this request made the next, longer
                    // request end the engine, #6905); only when that does not fit in host memory, just what this request needs.
                    if (kv_offload_mode >= 0 && kv_host_ctx == 0) {
                      const int tails[2] = {strata::core::ink_kv_host_tail_full(kv_dev_ctx, max_ctx), strata::core::ink_kv_host_tail_need(kv_dev_ctx, new_cap)};
                      for (int ti = 0; ti < 2; ++ti) {
                        const int hcap = tails[ti];
                        if (ti == 1 && hcap == tails[0]) break;
                        if (hcap > 0) {
                            if (m.ls_active) {
                                for (int ss = 0; ss < m.n_stages; ++ss) {
                                    cudaSetDevice(m.stage_gpu[ss]);
                                    cudaStreamSynchronize(ls_stages[ss].st);
                                }
                            } else {
                                cudaSetDevice(m.primary_gpu);
                                cudaStreamSynchronize(st);
                            }
                            bool all_ok = true;
                            for (int v = 0; v < kInkVisits && all_ok; ++v) {
                                if (kv_desc[v].is_swa) continue;
                                const int ip = ink_visit_phys(v);
                                const int dev = m.ls_active ? m.stage_gpu[m.layer_owner[ip]] : m.primary_gpu;
                                cudaSetDevice(dev);
                                const size_t helems = (size_t)max_slots * hcap * kInkKVHeads * kInkHeadDim;
                                if (kv_dtype == InkKvDtype::FP16) {
                                    all_ok = alloc_mapped_host<__half>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f16)
                                          && alloc_mapped_host<__half>(helems, host_kv_raw_v[v], kv_desc[v].h_v_f16);
                                } else {
                                    const size_t v_b = (kv_dtype == InkKvDtype::K8V4) ? (helems / 2u) : helems;
                                    const size_t sc_n = (size_t)max_slots * hcap * kInkKVHeads;
                                    all_ok = alloc_mapped_host<uint8_t>(helems, host_kv_raw_k[v], kv_desc[v].h_k_f8)
                                          && alloc_mapped_host<uint8_t>(v_b, host_kv_raw_v[v], kv_desc[v].h_v_f8)
                                          && alloc_mapped_host<float>(sc_n, host_kv_raw_ksc[v], kv_desc[v].h_k_sc)
                                          && alloc_mapped_host<float>(sc_n, host_kv_raw_vsc[v], kv_desc[v].h_v_sc);
                                }
                            }
                            cudaSetDevice(m.primary_gpu);
                            if (all_ok) {
                                for (int v = 0; v < kInkVisits; ++v) {
                                    if (kv_desc[v].is_swa) continue;
                                    kv_desc[v].dev_cap  = kv_dev_ctx;
                                    kv_desc[v].host_cap = hcap;
                                    kv_desc[v].cap_pos  = kv_dev_ctx + hcap;
                                }
                                INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
                                drop_all_graphs();
                                kv_host_ctx = hcap;
                                kv_alloc_ctx = kv_dev_ctx + hcap;
                                std::fprintf(stderr, "[InkCUDA] Elastic global KV spilled to mapped host RAM: %d VRAM + %d host = %d tokens\n",
                                             kv_dev_ctx, kv_host_ctx, kv_alloc_ctx);
                                return true;
                            }
                            for (int v = 0; v < kInkVisits; ++v) if (!kv_desc[v].is_swa) free_host_kv_visit(v);
                        }
                      }
                    }
                    return false;
                }
                new_cap = std::min(new_cap, tight_cap);
            }
        }

        if (m.ls_active) {
            for (int s = 0; s < m.n_stages; ++s) {
                cudaSetDevice(m.stage_gpu[s]);
                cudaStreamSynchronize(ls_stages[s].st);
            }
        } else {
            cudaSetDevice(m.primary_gpu);
            cudaStreamSynchronize(st);
        }

        for (int v = 0; v < kInkVisits; ++v) {
            if (kv_desc[v].is_swa) continue;
            const int ip = ink_visit_phys(v);
            const int s  = m.ls_active ? m.layer_owner[ip] : 0;
            const int dev = m.ls_active ? m.stage_gpu[s] : m.primary_gpu;
            cudaStream_t vst = m.ls_active ? ls_stages[s].st : st;
            cudaSetDevice(dev);

            const size_t old_slot_el = (size_t)old_cap * kInkKVHeads * kInkHeadDim;
            const size_t new_slot_el = (size_t)new_cap * kInkKVHeads * kInkHeadDim;
            const size_t new_tot_el  = (size_t)max_slots * new_slot_el;

            if (kv_dtype == InkKvDtype::FP16) {
                __half *nk = nullptr, *nv = nullptr;
                if (cudaMalloc(&nk, new_tot_el * sizeof(__half)) != cudaSuccess ||
                    cudaMalloc(&nv, new_tot_el * sizeof(__half)) != cudaSuccess) {
                    if (nk) cudaFree(nk);
                    if (nv) cudaFree(nv);
                    cudaGetLastError();
                    cudaSetDevice(m.primary_gpu);
                    return false;
                }
                for (int sl = 0; sl < max_slots; ++sl) {
                    cudaMemcpyAsync(nk + (size_t)sl * new_slot_el, kv_desc[v].k_f16 + (size_t)sl * old_slot_el,
                                    old_slot_el * sizeof(__half), cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nv + (size_t)sl * new_slot_el, kv_desc[v].v_f16 + (size_t)sl * old_slot_el,
                                    old_slot_el * sizeof(__half), cudaMemcpyDeviceToDevice, vst);
                }
                cudaStreamSynchronize(vst);
                cudaFree(kv_desc[v].k_f16);
                cudaFree(kv_desc[v].v_f16);
                kv_desc[v].k_f16 = nk;
                kv_desc[v].v_f16 = nv;
            } else {
                const bool v4 = (kv_dtype == InkKvDtype::K8V4);
                const size_t old_v_b = v4 ? (old_slot_el / 2u) : old_slot_el;
                const size_t new_v_b = v4 ? (new_slot_el / 2u) : new_slot_el;
                const size_t old_sc  = (size_t)old_cap * kInkKVHeads;
                const size_t new_sc  = (size_t)new_cap * kInkKVHeads;
                uint8_t *nk = nullptr, *nv = nullptr;
                float *nks = nullptr, *nvs = nullptr;
                if (cudaMalloc(&nk, new_tot_el) != cudaSuccess ||
                    cudaMalloc(&nv, (size_t)max_slots * new_v_b) != cudaSuccess ||
                    cudaMalloc(&nks, (size_t)max_slots * new_sc * sizeof(float)) != cudaSuccess ||
                    cudaMalloc(&nvs, (size_t)max_slots * new_sc * sizeof(float)) != cudaSuccess) {
                    if (nk) cudaFree(nk); if (nv) cudaFree(nv);
                    if (nks) cudaFree(nks); if (nvs) cudaFree(nvs);
                    cudaGetLastError();
                    cudaSetDevice(m.primary_gpu);
                    return false;
                }
                for (int sl = 0; sl < max_slots; ++sl) {
                    cudaMemcpyAsync(nk + (size_t)sl * new_slot_el, kv_desc[v].k_f8 + (size_t)sl * old_slot_el,
                                    old_slot_el, cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nv + (size_t)sl * new_v_b, kv_desc[v].v_f8 + (size_t)sl * old_v_b,
                                    old_v_b, cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nks + (size_t)sl * new_sc, kv_desc[v].k_sc + (size_t)sl * old_sc,
                                    old_sc * sizeof(float), cudaMemcpyDeviceToDevice, vst);
                    cudaMemcpyAsync(nvs + (size_t)sl * new_sc, kv_desc[v].v_sc + (size_t)sl * old_sc,
                                    old_sc * sizeof(float), cudaMemcpyDeviceToDevice, vst);
                }
                cudaStreamSynchronize(vst);
                cudaFree(kv_desc[v].k_f8); cudaFree(kv_desc[v].v_f8);
                cudaFree(kv_desc[v].k_sc); cudaFree(kv_desc[v].v_sc);
                kv_desc[v].k_f8 = nk; kv_desc[v].v_f8 = nv;
                kv_desc[v].k_sc = nks; kv_desc[v].v_sc = nvs;
            }
            kv_desc[v].cap_pos = new_cap;
            kv_desc[v].dev_cap = new_cap;
        }
        cudaSetDevice(m.primary_gpu);
        INK_CK(cudaMemcpy(d_kv_desc_arr, kv_desc, sizeof(kv_desc), cudaMemcpyHostToDevice));
        drop_all_graphs();
        kv_dev_ctx = new_cap;
        kv_alloc_ctx = new_cap;
        std::fprintf(stderr, "[InkCUDA] Elastic global KV grown: %d -> %d tokens (max_ctx=%d)\n", old_cap, new_cap, max_ctx);
        return true;
    }

    void reset_slot(int slot) {
        elastic_kv.reset_slot(slot);
        constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
        constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const int s = m.ls_active ? m.layer_owner[ip] : 0;
            if (m.ls_active) cudaSetDevice(m.stage_gpu[s]);
            cudaStream_t vst = m.ls_active ? ls_stages[s].st : st;
            for (int st_i = 0; st_i < 4; ++st_i) {
                float* p = d_sconv_pool[v] + (size_t)c_off[st_i] * 3 * max_slots + (size_t)slot * 3 * c_len[st_i];
                INK_CK(cudaMemsetAsync(p, 0, (size_t)3 * c_len[st_i] * sizeof(float), vst));
            }
        }
        if (m.ls_active) cudaSetDevice(m.stage_gpu[cur_stage]);
    }

    void ensure_logits(int M) {
        if (M <= logits_cap) return;
        INK_CK(cudaStreamSynchronize(st));
        if (d_logits) INK_CK(cudaFree(d_logits));
        logits_cap = M;
        d_logits = dalloc<float>((size_t)logits_cap * m.man.c.vocab_unpadded);
    }

    // ---- High-Speed Rank-64 Loop Controller + RMSNorm Dispatcher (100x faster than single-CTA!) ----
    void run_loop_control_norm(
        const float* cur, const float* msg,
        const float* w_down, const float* w_gain, const float* w_forget,
        const float* norm_w, float* out, int M,
        bool cat_ready = false, bool exact_fp32 = false,
        const float* w_gf_f32 = nullptr,
        int fp8_ci = -1, int fp8_ck = -1,
        bool bf16_cur_msg = false) {
        constexpr int D = kInkHidden;
        constexpr int D2 = 2 * D;
        const bool use_fp8 = (fp8_ci >= 0 && fp8_ci < 12 && fp8_ck >= 0 && fp8_ck < 4 &&
                              fp8_eng.cache_ready && !m.ls_active && fp8_eng.w_ctrl_dn[fp8_ci][fp8_ck] != nullptr);
        if (M > 16 && use_fp8 && fp8_eng.lt_ok) {
            if (fp8_eng.run_ctrl_fp8(fp8_ci, fp8_ck, cur, msg, norm_w, out, d_ctrl_gate, d_ctrl_cat, M, cat_ready, st, nullptr, nullptr, bf16_cur_msg)) {
                return;
            }
            if (!cat_ready) k_ink_ctrl_cat_rms<<<M, 256, 0, st>>>(cur, msg, d_ctrl_cat, M, D);
            goto fallback_f32_ctrl;
        }
        if (!cat_ready) k_ink_ctrl_cat_rms<<<M, 256, 0, st>>>(cur, msg, d_ctrl_cat, M, D);
        if (M <= 16 && use_fp8) {
            const uint8_t* wd8 = fp8_eng.w_ctrl_dn[fp8_ci][fp8_ck];
            const float*   sd8 = fp8_eng.sc_ctrl_dn[fp8_ci][fp8_ck];
            const uint8_t* wgf8 = fp8_eng.w_ctrl_gf[fp8_ci][fp8_ck];
            const float*   sgf8 = fp8_eng.sc_ctrl_gf[fp8_ci][fp8_ck];
            if (M == 1) {
                rtx5090::k_ink_ctrl_1x_down_fp8<1><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, wd8, sd8, d_ctrl_gate, M);
                rtx5090::k_ink_ctrl_1x_apply_fp8<1><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, wgf8, sgf8, out, M);
            } else if (M <= 4) {
                rtx5090::k_ink_ctrl_1x_down_fp8<4><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, wd8, sd8, d_ctrl_gate, M);
                rtx5090::k_ink_ctrl_1x_apply_fp8<4><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, wgf8, sgf8, out, M);
            } else if (M <= 8) {
                rtx5090::k_ink_ctrl_1x_down_fp8<8><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, wd8, sd8, d_ctrl_gate, M);
                rtx5090::k_ink_ctrl_1x_apply_fp8<8><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, wgf8, sgf8, out, M);
            } else if (M <= 16) {
                rtx5090::k_ink_ctrl_1x_down_fp8<16><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, wd8, sd8, d_ctrl_gate, M);
                rtx5090::k_ink_ctrl_1x_apply_fp8<16><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, wgf8, sgf8, out, M);
            } else {
                rtx5090::k_ink_ctrl_1x_down_fp8<24><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, wd8, sd8, d_ctrl_gate, M);
                rtx5090::k_ink_ctrl_1x_apply_fp8<24><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, wgf8, sgf8, out, M);
            }
            if (norm_w) k_ink_rms_norm<<<M, 256, 0, st>>>(out, norm_w, out, nullptr, M, D);
        } else if (M <= 16) {
            if (M == 1) {
                k_ink_ctrl_down_smallM<1><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<1><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 4) {
                k_ink_ctrl_down_smallM<4><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<4><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 8) {
                k_ink_ctrl_down_smallM<8><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<8><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else {
                k_ink_ctrl_down_smallM<16><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_smallM<16><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            }
            if (norm_w) k_ink_rms_norm<<<M, 256, 0, st>>>(out, norm_w, out, nullptr, M, D);
        } else if (rtx5090::dense_opt_on() && w_gf_f32) {
            // Opt-in RTX 5090: Same exact TF32 precision as baseline, fused gain+forget in 1 GEMM!
            const float alpha = 1.0f, beta = 0.0f;
            const cublasComputeType_t ctype = (!exact_fp32) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                w_down, CUDA_R_32F, D2, d_ctrl_cat, CUDA_R_32F, D2, &beta,
                d_ctrl_gate, CUDA_R_32F, kInkGateRank, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            float* gf_out = d_ctrl_cat + (size_t)M * D2;
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N, D2, M, kInkGateRank, &alpha,
                w_gf_f32, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                gf_out, CUDA_R_32F, D2, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            rtx5090::k_ink_ctrl_blend_norm_fused_gf<<<M, 256, 0, st>>>(cur, msg, gf_out, norm_w, out, M, D);
        } else {
        fallback_f32_ctrl:
            const float alpha = 1.0f, beta = 0.0f;
            const int ctl_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;   // the stage that runs this, not the first card (#6796)
            if (!exact_fp32 && ctl_cc >= 70 && ctl_cc < 80) {
                // Volta sm_70 (V100) / Turing sm_75 (2080 Ti): 125 TFLOPS FP16 TC for loop controller GEMMs (4x faster than FP32!)
                __half* x_h = reinterpret_cast<__half*>(d_x_bf16);
                __half* w_h = reinterpret_cast<__half*>(d_w_scratch);
                const size_t nx_d = (size_t)M * D2;
                const size_t nw_d = (size_t)kInkGateRank * D2;
                k_ink_f32_to_fp16<<<(nx_d + 255) / 256, 256, 0, st>>>(d_ctrl_cat, x_h, nx_d);
                k_ink_f32_to_fp16<<<(nw_d + 255) / 256, 256, 0, st>>>(w_down, w_h, nw_d);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                    w_h, CUDA_R_16F, D2, x_h, CUDA_R_16F, D2, &beta,
                    d_ctrl_gate, CUDA_R_32F, kInkGateRank, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                const size_t nx_g = (size_t)M * kInkGateRank;
                const size_t nw_g = (size_t)D * kInkGateRank;
                k_ink_f32_to_fp16<<<(nx_g + 255) / 256, 256, 0, st>>>(d_ctrl_gate, x_h, nx_g);
                k_ink_f32_to_fp16<<<(nw_g + 255) / 256, 256, 0, st>>>(w_gain, w_h, nw_g);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_h, CUDA_R_16F, kInkGateRank, x_h, CUDA_R_16F, kInkGateRank, &beta,
                    d_ctrl_gain, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                k_ink_f32_to_fp16<<<(nw_g + 255) / 256, 256, 0, st>>>(w_forget, w_h, nw_g);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_h, CUDA_R_16F, kInkGateRank, x_h, CUDA_R_16F, kInkGateRank, &beta,
                    d_ctrl_forget, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                const cublasComputeType_t ctype = (!exact_fp32 && ctl_cc >= 80) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkGateRank, M, D2, &alpha,
                    w_down, CUDA_R_32F, D2, d_ctrl_cat, CUDA_R_32F, D2, &beta,
                    d_ctrl_gate, CUDA_R_32F, kInkGateRank, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_gain, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                    d_ctrl_gain, CUDA_R_32F, D, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, M, kInkGateRank, &alpha,
                    w_forget, CUDA_R_32F, kInkGateRank, d_ctrl_gate, CUDA_R_32F, kInkGateRank, &beta,
                    d_ctrl_forget, CUDA_R_32F, D, ctype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            k_ink_ctrl_blend_norm<<<M, 256, 0, st>>>(cur, msg, d_ctrl_gain, d_ctrl_forget, norm_w, out, M, D);
        }
    }

    // Opt-in Q8_ROW Rank-64 Loop Controller + RMSNorm (4x less DRAM traffic + INT8 TC for M > 16!)
    void run_loop_control_q8_norm(
        const float* cur, const float* msg,
        const uint8_t* w_down, const uint8_t* w_gain, const uint8_t* w_forget,
        const float* norm_w, float* out, int M,
        bool cat_ready = false) {
        constexpr int D  = kInkHidden;
        constexpr int D2 = 2 * D;
        if (M <= 16) {
            if (!cat_ready) k_ink_ctrl_cat_rms<<<M, 256, 0, st>>>(cur, msg, d_ctrl_cat, M, D);
            if (M == 1) {
                k_ink_ctrl_down_q8_smallM<1><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_q8_smallM<1><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 4) {
                k_ink_ctrl_down_q8_smallM<4><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_q8_smallM<4><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else if (M <= 8) {
                k_ink_ctrl_down_q8_smallM<8><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_q8_smallM<8><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            } else {
                k_ink_ctrl_down_q8_smallM<16><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, w_down, d_ctrl_gate, M, D2);
                k_ink_ctrl_apply_q8_smallM<16><<<D / 16, 256, 0, st>>>(cur, msg, d_ctrl_gate, w_gain, w_forget, out, M, D);
            }
            if (norm_w) k_ink_rms_norm<<<M, 256, 0, st>>>(out, norm_w, out, nullptr, M, D);
        } else {
            const int M_pad = (M + 3) & ~3;
            int8_t* x_i8 = reinterpret_cast<int8_t*>(d_x_bf16);
            float*  x_sc = d_sinks;  // free scratch [max_batch * 24] floats before QKV & before MLP!
            const int32_t alpha_i = 1, beta_i = 0;

            // 1. Fused Cat+RMS+I8Quant -> gate_i32 = cat[M, 4608] @ w_down[64, 4608]^T (ZERO FP32 d_ctrl_cat DRAM!)
            if (cat_ready) {
                k_ink_quant_x_row_i8<<<M, 256, 0, st>>>(d_ctrl_cat, x_i8, x_sc, M, D2);
            } else {
                k_ink_ctrl_cat_rms_i8<<<M, 256, 0, st>>>(cur, msg, x_i8, x_sc, M, D);
            }
            if (M_pad > M) INK_CK(cudaMemsetAsync(x_i8 + (size_t)M * D2, 0, (size_t)(M_pad - M) * D2, st));
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                kInkGateRank, M_pad, D2, &alpha_i,
                reinterpret_cast<const int8_t*>(w_down + (size_t)kInkGateRank * 4u), CUDA_R_8I, D2,
                x_i8, CUDA_R_8I, D2, &beta_i,
                d_ctrl_gate, CUDA_R_32I, kInkGateRank,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // 2. Direct-I32 100%-Coalesced DP4A Q8_ROW Apply+Blend (eliminates k_ink_i32_to_f32_scale + 2x K=64 cuBLAS!)
            k_ink_ctrl_apply_q8_i32_tile16<<<dim3((D + 31) / 32, (M + 7) / 8), 256, 0, st>>>(
                cur, msg, reinterpret_cast<const int32_t*>(d_ctrl_gate),
                reinterpret_cast<const float*>(w_down), x_sc, w_gain, w_forget, out, M, D);
            if (norm_w) k_ink_rms_norm<<<M, 256, 0, st>>>(out, norm_w, out, nullptr, M, D);
        }
    }

    // ---- Cross-Architecture cuBLAS GEMM for BF16 weights: sm_80+ native BF16 TC, sm_70/75 FP16 TC, sm_60/61 FP32 ----
    // When w_is_scratch==true, caller placed BF16 weights at d_w_scratch (or at d_w_scratch + 2*(cap/3) when sm_cc < 70).
    void gemm_bf16_compat(const __nv_bfloat16* W_bf16, const float* X, float* Y, int M, int N, int K,
                          int ldY, bool reuse_x, bool w_is_scratch) {
        const float alpha = 1.0f, beta = 0.0f;
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        if (cur_cc >= 80) {
            if (!reuse_x) {
                const size_t nx = (size_t)M * K;
                k_ink_f32_to_bf16<<<((nx + 3) / 4 + 255) / 256, 256, 0, st>>>(X, d_x_bf16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha,
                W_bf16, CUDA_R_16BF, K,
                d_x_bf16, CUDA_R_16BF, K,
                &beta, Y, CUDA_R_32F, ldY,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if (cur_cc >= 70) {
            __half* x_fp16 = reinterpret_cast<__half*>(d_x_bf16);
            if (!reuse_x) {
                const size_t nx = (size_t)M * K;
                k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, st>>>(X, x_fp16, nx);
            }
            const size_t cap = m.all_sf4_vram
                ? ((size_t)kInkQDim * kInkHidden)
                : ((size_t)kInkDsparkInter * kInkHidden);
            if (w_is_scratch) {
                const size_t nw = (size_t)N * K;
                __half* w_fp16 = reinterpret_cast<__half*>(d_w_scratch);
                k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16, w_fp16, nw);
                cublasStatus_t st_fp16 = cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    w_fp16, CUDA_R_16F, K,
                    x_fp16, CUDA_R_16F, K,
                    &beta, Y, CUDA_R_32F, ldY,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st_fp16 != CUBLAS_STATUS_SUCCESS) {
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, M, K, &alpha,
                        w_fp16, CUDA_R_16F, K,
                        x_fp16, CUDA_R_16F, K,
                        &beta, Y, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }
            } else {
                const int tile_n = std::min(N, (int)(cap / (size_t)K));
                __half* w_fp16 = reinterpret_cast<__half*>(d_w_scratch);
                for (int n0 = 0; n0 < N; n0 += tile_n) {
                    const int nt = std::min(tile_n, N - n0);
                    const size_t nw = (size_t)nt * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16 + (size_t)n0 * K, w_fp16, nw);
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nt, M, K, &alpha,
                        w_fp16, CUDA_R_16F, K,
                        x_fp16, CUDA_R_16F, K,
                        &beta, Y + n0, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
        } else {
            const size_t cap = m.all_sf4_vram
                ? ((size_t)kInkQDim * kInkHidden)
                : ((size_t)kInkDsparkInter * kInkHidden);
            const size_t max_elems = cap / 3;
            float* w_f32 = reinterpret_cast<float*>(d_w_scratch);
            if (w_is_scratch) {
                const size_t nw = (size_t)N * K;
                k_ink_bf16_to_f32<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16, w_f32, nw);
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    w_f32, CUDA_R_32F, K,
                    X, CUDA_R_32F, K,
                    &beta, Y, CUDA_R_32F, ldY,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else {
                const int tile_n = std::min(N, (int)(max_elems / (size_t)K));
                for (int n0 = 0; n0 < N; n0 += tile_n) {
                    const int nt = std::min(tile_n, N - n0);
                    const size_t nw = (size_t)nt * K;
                    k_ink_bf16_to_f32<<<(nw + 255) / 256, 256, 0, st>>>(W_bf16 + (size_t)n0 * K, w_f32, nw);
                    INK_CUBLAS_CK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nt, M, K, &alpha,
                        w_f32, CUDA_R_32F, K,
                        X, CUDA_R_32F, K,
                        &beta, Y + n0, CUDA_R_32F, ldY,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                }
            }
        }
    }

    // Tracks which converted representation of a shared X buffer is currently valid in d_x_bf16.
    enum XCacheFmt { XC_NONE = 0, XC_BF16 = 1, XC_FP16 = 2, XC_I8 = 3, XC_FP4 = 4 };

    XCacheFmt linear_x_cache_fmt(const std::string& tname, int M, int N_override, bool exact_fp32_x) const {
        const auto& ti = m.tinfo(tname);
        const int N = N_override > 0 ? N_override : (int)ti.rows();
        const int K = (int)ti.cols();
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        const bool cur_attn_fp16 = m.ls_active ? m.stage_trunk_attn_fp16[cur_stage] : m.trunk_attn_fp16;
        const XCacheFmt gemm_fmt = (cur_cc >= 80) ? XC_BF16 : ((cur_cc >= 70) ? XC_FP16 : XC_NONE);
        if (ti.fmt == FMT_BF16) {
            const bool is_fp16 = cur_attn_fp16 && (tname.find(".attn.") != std::string::npos);
            const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || cur_cc < 80) ? 16 : 7);
            if (M <= bf16_gemv_max) return XC_NONE;
            return is_fp16 ? XC_FP16 : gemm_fmt;
        }
        if (ti.fmt == FMT_F32) return XC_NONE;
        if (ti.fmt == FMT_Q8_0F) return XC_NONE;   // quantizes its own activations (k_ink_quant_x_q8f)
        if (ti.fmt == FMT_SF4) {
            return (M > 16 && sm120::available() && (K % 64 == 0) && (N % 128 == 0) && N_override == 0) ? XC_FP4 : XC_NONE;
        }
        if (ti.fmt == FMT_Q8ROW) {
            const bool sm120_i8 = ink_rtx5060ti_cfg(m.primary_gpu).sm120_tc_smallm && cur_cc >= 120 && (M % 4 == 0);
            if (M > 16 && (cur_cc == 61 || sm120_i8) && (K % 4 == 0) && (N % 4 == 0) && N_override == 0) return XC_I8;
            const size_t cap = m.all_sf4_vram
                ? ((size_t)kInkQDim * kInkHidden)
                : ((size_t)kInkDsparkInter * kInkHidden);
            const size_t max_tile = (cur_cc < 70) ? (cap / 3) : cap;
            if (M <= 16 || (size_t)N * K > max_tile) return XC_NONE;
            return gemm_fmt;
        }
        return gemm_fmt;
    }

    void linear_cached(const std::string& tname, const float* X, float* Y, int M, XCacheFmt& x_cached, bool exact_fp32_x = false) {
        const XCacheFmt need = linear_x_cache_fmt(tname, M, 0, exact_fp32_x);
        const bool reuse = (need != XC_NONE && need == x_cached);
        linear(tname, X, Y, M, 0, reuse, exact_fp32_x);
        if (need != XC_NONE) x_cached = need;
    }

    // Q8_0 dense matrices (FMT_Q8_0F): the activations are quantized once into the fragment order, then one GEMM launch for all the
    // matrices of mp (see k_ink_gemm_q8f for the layout and why the result does not depend on M).
    bool q8f_ok(int M, int K) const { return M >= 1 && M <= max_batch && K % 32 == 0 && K <= kInkQDim && d_xq8 != nullptr; }
    void gemm_q8f(const InkMultiProj& mp, const float* X, int M, int K) {
        const int nMT = (M + 7) / 8;
        const int nKB = K >> 5;
        const int total_warps = nMT * 8 * nKB;
        k_ink_quant_x_q8f<<<(total_warps + 7) / 8, 256, 0, st>>>(X, d_xq8, d_xs8, M, K, nMT);
        const int row_tiles = mp.cumN[mp.count] / 16;
        const int gx = (row_tiles + 1) / 2;
        // token tiles per warp: as many as fit, but split the tokens over more CTAs when the matrices are short of row tiles
        // (the output projection has 144) so that about 1500 warps are in flight; the result does not depend on this choice
        const int want_gy = std::max(1, (1536 + row_tiles - 1) / row_tiles);
        int mtw = 1;
        for (int c : {8, 4, 2}) { if (nMT >= c && (nMT + c - 1) / c >= std::min(want_gy, nMT)) { mtw = c; break; } }
        const dim3 grid(gx, (nMT + mtw - 1) / mtw);
        if (mtw == 8)      k_ink_gemm_q8f<8><<<grid, 64, 0, st>>>(mp, d_xq8, d_xs8, M, K, nMT);
        else if (mtw == 4) k_ink_gemm_q8f<4><<<grid, 64, 0, st>>>(mp, d_xq8, d_xs8, M, K, nMT);
        else if (mtw == 2) k_ink_gemm_q8f<2><<<grid, 64, 0, st>>>(mp, d_xq8, d_xs8, M, K, nMT);
        else               k_ink_gemm_q8f<1><<<grid, 64, 0, st>>>(mp, d_xq8, d_xs8, M, K, nMT);
    }

    // ---- Dense linear projection helper: Y[M, N] = X[M, K] @ W[N, K]^T ----
    void linear_multi(const std::string* names, float* const* Ys, int count, const float* X, int M, bool exact_fp32_x = false) {
        if (count <= 0) return;
        const auto& ti0 = m.tinfo(names[0]);
        const int K = (int)ti0.cols();
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        const bool cur_attn_fp16 = m.ls_active ? m.stage_trunk_attn_fp16[cur_stage] : m.trunk_attn_fp16;
        const bool is_fp16 = cur_attn_fp16 && (names[0].find(".attn.") != std::string::npos);
        const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || cur_cc < 80) ? 16 : 7);
        bool can_fuse = (count <= 5) && ((ti0.fmt == FMT_BF16 && M <= bf16_gemv_max) || (ti0.fmt == FMT_Q8ROW && M <= 16) ||
                                         (ti0.fmt == FMT_SF4 && M <= 16) || (ti0.fmt == FMT_Q8_0F && q8f_ok(M, K)));
        InkMultiProj mp{};
        mp.count = count;
        mp.cumN[0] = 0;
        for (int i = 0; i < count; ++i) {
            const auto& ti = m.tinfo(names[i]);
            if (ti.fmt != ti0.fmt || (int)ti.cols() != K) { can_fuse = false; break; }
            if (cur_attn_fp16 && ((names[i].find(".attn.") != std::string::npos) != is_fp16)) { can_fuse = false; break; }
            mp.W[i] = m.trunk_ptr_stage(cur_stage, names[i]);
            mp.Y[i] = Ys[i];
            mp.N[i] = (int)ti.rows();
            mp.cumN[i + 1] = mp.cumN[i] + mp.N[i];
        }
        if (can_fuse) {
            const int grid = (mp.cumN[count] + 7) / 8;
            if (ti0.fmt == FMT_Q8_0F) {
                gemm_q8f(mp, X, M, K);
                return;
            }
            if (ti0.fmt == FMT_SF4) {
                const int g4 = (mp.cumN[count] + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_sf4<1><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_sf4<4><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_sf4<8><<<g4, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_sf4<16><<<g4, 128, 0, st>>>(mp, X, M, K);
                return;
            }
            if (ti0.fmt == FMT_BF16) {
                if (is_fp16) {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<4, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                } else if (M > 16 && rtx5090::dense_opt_on()) {
                    const int g4 = (mp.cumN[count] + 3) / 4;
                    rtx5090::k_ink_gemv_multi_bf16_v2<24><<<g4, 64, 0, st>>>(mp, X, M, K);
                } else {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 4) k_ink_gemv_multi_bf16<4, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 8) k_ink_gemv_multi_bf16<8, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<16, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                }
                return;
            }
            if ((K % 64) == 0 && K <= kInkQDim && (cur_cc == 61 || cur_cc >= 75)) {
                if (M == 1) {
                    k_ink_gemv_multi_q8row_m1<<<grid, 128, 0, st>>>(mp, X, K);
                    return;
                }
                const int nblk = M * (K / 64);
                if (cur_moe_ws && (size_t)nblk <= (size_t)cur_moe_ws->cap_M * (kInkHidden / 64)) {
                    k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X, cur_moe_ws->xq_sk, M, K);
                    if (M <= 4)      k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                    else if (M <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                    else             k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                    return;
                }
            }
            const int g4 = (mp.cumN[count] + 3) / 4;
            if (M == 1)      k_ink_gemv_multi_q8row<1><<<g4, 128, 0, st>>>(mp, X, M, K);
            else if (M <= 4) k_ink_gemv_multi_q8row<4><<<g4, 128, 0, st>>>(mp, X, M, K);
            else if (M <= 8) k_ink_gemv_multi_q8row<8><<<g4, 128, 0, st>>>(mp, X, M, K);
            else             k_ink_gemv_multi_q8row<16><<<g4, 128, 0, st>>>(mp, X, M, K);
            return;
        }
        // Mixed-format or non-fusible projections: only reuse d_x_bf16 when a prior projection populated the same format.
        XCacheFmt x_cached = XC_NONE;
        for (int i = 0; i < count; ++i) linear_cached(names[i], X, Ys[i], M, x_cached, exact_fp32_x);
    }

    void linear(const std::string& tname, const float* X, float* Y, int M, int N_override = 0, bool reuse_x_bf16 = false, bool exact_fp32_x = false) {
        const auto& ti = m.tinfo(tname);
        const int N = N_override > 0 ? N_override : (int)ti.rows();
        const int K = (int)ti.cols();
        const uint8_t* W = m.trunk_ptr_stage(cur_stage, tname);
        const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
        const bool cur_attn_fp16 = m.ls_active ? m.stage_trunk_attn_fp16[cur_stage] : m.trunk_attn_fp16;

        if (ti.fmt == FMT_BF16) {
            const bool is_fp16 = cur_attn_fp16 && (tname.find(".attn.") != std::string::npos);
            const int bf16_gemv_max = is_fp16 ? 3 : ((exact_fp32_x || cur_cc < 80) ? 16 : 7);
            if (M <= bf16_gemv_max) {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                const int grid = (N + 7) / 8;
                if (is_fp16) {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<4, true><<<grid, 128, 0, st>>>(mp, X, M, K);
                } else if (M > 16 && rtx5090::dense_opt_on()) {
                    const int g4 = (N + 3) / 4;
                    rtx5090::k_ink_gemv_multi_bf16_v2<24><<<g4, 64, 0, st>>>(mp, X, M, K);
                } else {
                    if (M == 1)      k_ink_gemv_multi_bf16<1, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 4) k_ink_gemv_multi_bf16<4, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else if (M <= 8) k_ink_gemv_multi_bf16<8, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                    else             k_ink_gemv_multi_bf16<16, false><<<grid, 128, 0, st>>>(mp, X, M, K);
                }
            } else if (is_fp16) {
                __half* x_fp16 = reinterpret_cast<__half*>(d_x_bf16);
                if (!reuse_x_bf16) {
                    const size_t nx = (size_t)M * K;
                    k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, st>>>(X, x_fp16, nx);
                }
                const float alpha = 1.0f, beta = 0.0f;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    W, CUDA_R_16F, K,
                    x_fp16, CUDA_R_16F, K,
                    &beta, Y, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                gemm_bf16_compat(reinterpret_cast<const __nv_bfloat16*>(W), X, Y, M, N, K, N, reuse_x_bf16, false);
            }
        } else if (ti.fmt == FMT_F32) {
            if (M <= 16) {
                const int grid = (N + 7) / 8;
                if (M == 1)      k_ink_gemv_f32<1><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else if (M <= 4) k_ink_gemv_f32<4><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else if (M <= 8) k_ink_gemv_f32<8><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
                else             k_ink_gemv_f32<16><<<grid, 256, 0, st>>>((const float*)W, X, Y, M, N, K);
            } else if (rtx5090::dense_opt_on() && M <= 64 && N <= 64 && K == kInkHidden) {
                k_ink_gemm_f32_smallN<<<dim3(N, (M + 15) / 16), 256, 0, st>>>((const float*)W, X, Y, M, N, K);
            } else {
                const float alpha = 1.0f, beta = 0.0f;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                    W, CUDA_R_32F, K, X, CUDA_R_32F, K, &beta, Y, CUDA_R_32F, N,
                    cur_cc >= 80 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        } else if (ti.fmt == FMT_Q8_0F) {
            InkMultiProj mp{};
            mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
            gemm_q8f(mp, X, M, K);
        } else if (ti.fmt == FMT_SF4) {
            if (M > 16 && sm120::available() && (K % 64 == 0) && (N % 128 == 0) && N_override == 0) {
                uint8_t* xa = reinterpret_cast<uint8_t*>(d_x_bf16);
                const bool dense_direct = ink_rtx5060ti_cfg(m.primary_gpu).sm120_tc_smallm;   // direct-MMA SF4 (per-row scale) only when the table asks for it
                if (!reuse_x_bf16) sm120::quantize_act_fp4(X, M, K, xa, st, dense_direct);
                sm120::dense_sf4_gemm_tc(W, xa, Y, M, N, K, st, dense_direct);
            } else {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                const int g4 = (N + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_sf4<1><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_sf4<4><<<g4, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_sf4<8><<<g4, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_sf4<16><<<g4, 128, 0, st>>>(mp, X, M, K);
            }
        } else if (ti.fmt == FMT_Q8ROW) {
            const int M_pad = (M + 3) & ~3;
            const int i8_m_cap = m.all_sf4_vram ? std::max(max_batch, 256) : max_batch;
            const bool sm120_i8 = ink_rtx5060ti_cfg(m.primary_gpu).sm120_tc_smallm && cur_cc >= 120 && (M_pad <= i8_m_cap);
            if (M > 16 && (cur_cc == 61 || sm120_i8) && (K % 4 == 0) && (N % 4 == 0) && N_override == 0) {
                int8_t* x_i8 = reinterpret_cast<int8_t*>(d_x_bf16);
                float*  x_sc = d_ctrl_gain;
                if (!reuse_x_bf16) {
                    k_ink_quant_x_row_i8<<<M, 256, 0, st>>>(X, x_i8, x_sc, M, K);
                    if (M_pad > M) {
                        INK_CK(cudaMemsetAsync(x_i8 + (size_t)M * K, 0, (size_t)(M_pad - M) * K, st));
                    }
                }
                const float*  w_sc = reinterpret_cast<const float*>(W);
                const int8_t* w_i8 = reinterpret_cast<const int8_t*>(W + (size_t)ti.rows() * 4);
                const int32_t alpha_i = 1, beta_i = 0;
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M_pad, K, &alpha_i,
                    w_i8, CUDA_R_8I, K,
                    x_i8, CUDA_R_8I, K,
                    &beta_i, Y, CUDA_R_32I, N,
                    CUBLAS_COMPUTE_32I,
                    (cur_cc >= 75) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
                const size_t tot = (size_t)M * N;
                k_ink_i32_to_f32_scale<<<(tot + 255) / 256, 256, 0, st>>>(
                    reinterpret_cast<int32_t*>(Y), Y, w_sc, x_sc, M, N);
                return;
            }
            const size_t cap = m.all_sf4_vram
                ? ((size_t)kInkQDim * kInkHidden)
                : ((size_t)kInkDsparkInter * kInkHidden);
            const size_t max_tile = (cur_cc < 70) ? (cap / 3) : cap;
            if (M <= 16) {
                InkMultiProj mp{};
                mp.W[0] = W; mp.Y[0] = Y; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                if ((K % 64) == 0 && K <= (m.all_sf4_vram ? kInkDsparkInter : kInkQDim) && (cur_cc == 61 || cur_cc >= 75)) {
                    const int grid = (N + 7) / 8;
                    if (M == 1 && K <= kInkQDim) {
                        k_ink_gemv_multi_q8row_m1<<<grid, 128, 0, st>>>(mp, X, K);
                        return;
                    }
                    const int nblk = M * (K / 64);
                    const size_t xq_cap = cur_moe_ws ? (size_t)(m.all_sf4_vram ? 64u : (size_t)cur_moe_ws->cap_M) * (kInkHidden / 64) : 0u;   // (null stays guarded as in the shipped code)
                    if (cur_moe_ws && (size_t)nblk <= xq_cap) {
                        k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X, cur_moe_ws->xq_sk, M, K);
                        if (M == 1)      k_ink_gemv_multi_q8row_preq<1><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        else if (M <= 4) k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        else if (M <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        else             k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, M, K);
                        return;
                    }
                }
                const int grid = (N + 3) / 4;
                if (M == 1)      k_ink_gemv_multi_q8row<1><<<grid, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 4) k_ink_gemv_multi_q8row<4><<<grid, 128, 0, st>>>(mp, X, M, K);
                else if (M <= 8) k_ink_gemv_multi_q8row<8><<<grid, 128, 0, st>>>(mp, X, M, K);
                else             k_ink_gemv_multi_q8row<16><<<grid, 128, 0, st>>>(mp, X, M, K);
            } else if ((size_t)N * K > max_tile) {
                // Safe chunked M=16 Q8_ROW multi-GEMV across M > 16 when N*K exceeds d_w_scratch (0 extra scratch!)
                for (int m0 = 0; m0 < M; m0 += 16) {
                    const int mc = std::min(16, M - m0);
                    InkMultiProj mp{};
                    mp.W[0] = W; mp.Y[0] = Y + (size_t)m0 * N; mp.N[0] = N; mp.cumN[0] = 0; mp.cumN[1] = N; mp.count = 1;
                    const int grid = (N + 7) / 8;
                    if ((K % 64) == 0 && K <= kInkQDim && cur_moe_ws) {
                        const int nblk = mc * (K / 64);
                        k_ink_quant_q8_1<<<(nblk + 7) / 8, 256, 0, st>>>(X + (size_t)m0 * K, cur_moe_ws->xq_sk, mc, K);
                        if (mc <= 4)      k_ink_gemv_multi_q8row_preq<4><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, mc, K);
                        else if (mc <= 8) k_ink_gemv_multi_q8row_preq<8><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, mc, K);
                        else              k_ink_gemv_multi_q8row_preq<16><<<grid, 128, 0, st>>>(mp, cur_moe_ws->xq_sk, mc, K);
                    } else {
                        const int g4 = (N + 3) / 4;
                        k_ink_gemv_multi_q8row<16><<<g4, 128, 0, st>>>(mp, X + (size_t)m0 * K, mc, K);
                    }
                }
            } else {
                __nv_bfloat16* w_dst = (cur_cc < 70) ? (d_w_scratch + 2 * (cap / 3)) : d_w_scratch;
                dim3 dg((K + 255) / 256, N);
                k_ink_q8row_to_bf16<<<dg, 256, 0, st>>>(W, w_dst, N, K);
                gemm_bf16_compat(w_dst, X, Y, M, N, K, N, reuse_x_bf16, true);
            }
        } else {
            const size_t cap = m.all_sf4_vram
                ? ((size_t)kInkQDim * kInkHidden)
                : ((size_t)kInkDsparkInter * kInkHidden);
            const size_t max_tile = (cur_cc < 70) ? (cap / 3) : cap;
            const int tile_n = std::min(N, (int)(max_tile / (size_t)K));
            const size_t row_b = (size_t)(K / 256) * ink_sb_bytes(ti.fmt);
            __nv_bfloat16* w_dst = (cur_cc < 70) ? (d_w_scratch + 2 * (cap / 3)) : d_w_scratch;
            for (int n0 = 0; n0 < N; n0 += tile_n) {
                const int nt = std::min(tile_n, N - n0);
                dim3 dg((K + 255) / 256, nt);
                const uint8_t* Wsub = W + (size_t)n0 * row_b;
                if (ti.fmt == FMT_S5_K)      k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                else if (ti.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                else                         k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, st>>>(Wsub, w_dst, nt, K);
                gemm_bf16_compat(w_dst, X, Y + n0, M, nt, K, N, (n0 > 0) || reuse_x_bf16, true);
            }
        }
    }

    // Strictly O(1)-VRAM prefill attention scratch (fixed KV_TILE = 2048 -> ~114 MiB forever, fixes #6609 & #6605!).
    // Non-fatal: returns false if VRAM is tight so caller seamlessly falls back to zero-VRAM k_ink_flash_gqa_gqa3!
    bool ensure_pf_attn(int /*K_len*/) {
        constexpr int Q_CH = 256;
        if (d_pf_q && pf_kv_cap >= kInkPfKvTile) return true;
        INK_CK(cudaStreamSynchronize(st));
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p);
        fr(d_pf_s); fr(d_pf_o); fr(d_pf_ot); fr(d_pf_m); fr(d_pf_l);
        pf_kv_cap = 0;
        const size_t q_b  = (size_t)kInkHeads * max_batch * kInkHeadDim * sizeof(__nv_bfloat16);
        const size_t kv_b = (size_t)kInkHeads * kInkPfKvTile * kInkHeadDim * sizeof(__nv_bfloat16);
        const size_t s_b  = (size_t)kInkHeads * Q_CH * kInkPfKvTile * sizeof(float);
        const size_t p_b  = (size_t)kInkHeads * Q_CH * kInkPfKvTile * sizeof(__nv_bfloat16);
        const size_t o_b  = (size_t)kInkHeads * Q_CH * kInkHeadDim * sizeof(float);
        const size_t ml_b = (size_t)kInkHeads * Q_CH * sizeof(float);
        auto al256 = [](size_t b) { return (b + 255ull) & ~255ull; };
        const size_t need = (m.fp8_trunk.ready && !m.ls_active)
            ? (72ull << 20)
            : (al256(q_b) + 2 * al256(kv_b) + al256(s_b) + al256(p_b) + 2 * al256(o_b) + 2 * al256(ml_b));
        size_t vf = 0, vt = 0;
        const int dev = m.ls_active ? m.stage_gpu[cur_stage] : m.primary_gpu;
        const size_t pf_guard = m.all_sf4_vram ? (48ull << 20)
                              : (ink_rtx5060ti_cfg(dev).pack ? (64ull << 20) : (192ull << 20));
        if (ink_cuda_mem_info(&vf, &vt, dev) != cudaSuccess || vf < need + pf_guard) return false;
        if (m.all_sf4_vram && !m.ls_active) {
            if (cudaMalloc(&d_pf_slab, need) != cudaSuccess) {
                cudaGetLastError();
                d_pf_slab = nullptr;
                return false;
            }
            d_pf_slab_bytes = need;
            if (m.fp8_trunk.ready) {
                // FP8 engine uses zero-DRAM k_flash_gqa_pf_fp16acc; d_pf_slab (68 MiB) is reserved purely for Macro-Prefill MoE!
                pf_kv_cap = kInkPfKvTile;
                return true;
            }
            size_t off = 0;
            auto sl = [&](size_t b) -> uint8_t* {
                uint8_t* p = d_pf_slab + off;
                off += al256(b);
                return p;
            };
            d_pf_q  = reinterpret_cast<__nv_bfloat16*>(sl(q_b));
            d_pf_k  = reinterpret_cast<__nv_bfloat16*>(sl(kv_b));
            d_pf_v  = reinterpret_cast<__nv_bfloat16*>(sl(kv_b));
            d_pf_s  = reinterpret_cast<float*>(sl(s_b));
            d_pf_p  = reinterpret_cast<__nv_bfloat16*>(sl(p_b));
            d_pf_o  = reinterpret_cast<float*>(sl(o_b));
            d_pf_ot = reinterpret_cast<float*>(sl(o_b));
            d_pf_m  = reinterpret_cast<float*>(sl(ml_b));
            d_pf_l  = reinterpret_cast<float*>(sl(ml_b));
            pf_kv_cap = kInkPfKvTile;
            return true;
        }
        if (cudaMalloc(&d_pf_q, q_b) != cudaSuccess ||
            cudaMalloc(&d_pf_k, kv_b) != cudaSuccess ||
            cudaMalloc(&d_pf_v, kv_b) != cudaSuccess ||
            cudaMalloc(&d_pf_s, s_b) != cudaSuccess ||
            cudaMalloc(&d_pf_p, p_b) != cudaSuccess ||
            cudaMalloc(&d_pf_o, o_b) != cudaSuccess ||
            cudaMalloc(&d_pf_ot, o_b) != cudaSuccess ||
            cudaMalloc(&d_pf_m, ml_b) != cudaSuccess ||
            cudaMalloc(&d_pf_l, ml_b) != cudaSuccess) {
            cudaGetLastError();
            fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p);
            fr(d_pf_s); fr(d_pf_o); fr(d_pf_ot); fr(d_pf_m); fr(d_pf_l);
            return false;
        }
        pf_kv_cap = kInkPfKvTile;
        return true;
    }

    // ---- Execute 1 virtual visit (visit in 0..47, phys layer ip in 0..23) ----
    // mode: 0 = single-slot sequential (prefill or M=1 decode), 1 = batched multi-slot (B=1..64, 1 tok/slot), 2 = Tree-32 verify,
    //       3 = opt-in batched multi-slot speculative chain verify (B slots x K chain tokens = M)
    void run_visit(int visit, int ip, const DepthStatePtrs* bank_stp, int M, int mode, int slot0, int p0_val,
                   int& n_blocks, const float* (&block_ptrs)[8], bool pf_bf16_res = false, bool lm_stop_at_router = false) {
        const std::string p = "model.llm.layers." + std::to_string(ip);
        const bool is_swa = m.man.c.is_swa[ip] != 0;
        const bool has_msg = (bank_stp != nullptr);
        const bool exact_x = (mode == 2);
        const size_t n_hid = (size_t)M * kInkHidden;
        ink_tail_cur_visit() = visit;   // the opt-in tail skip (ink_tail_skip.cuh) reads the visit number at launch time

        // 1. Attention-Residual Depth Mix (+ Fused Looped Pre-Attn Msg + CatRMS -> eliminates 48+72 launches!)
        prof_mark_start();
        const float* nw_q = (const float*)m.trunk_ptr_stage(cur_stage, is_swa ? (p + ".attn_norm.weight") : (p + ".q_input_norm.weight"));
        const float* nw_k = (const float*)m.trunk_ptr_stage(cur_stage, is_swa ? (p + ".attn_norm.weight") : (p + ".k_input_norm.weight"));
        const float* nw_v = (const float*)m.trunk_ptr_stage(cur_stage, is_swa ? (p + ".attn_norm.weight") : (p + ".v_input_norm.weight"));

        const float* in_q = d_inp0;
        const float* in_k = d_inp0;
        const float* in_v = d_inp0;
        bool has_pre_amax = false;
        bool qkv_h2_ready = false;
        bool x_h2_ready = false;
        if (ip > 0) {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            const float* q_res = (const float*)m.trunk_ptr_stage(cur_stage, "model.llm.attn_res.attn_queries." + std::to_string(ip - 1));
            const int nq = is_swa ? 1 : 3;
            if (!has_msg) {
                if (pf_bf16_res || (M > 24 && fp8_eng.cache_ready && !m.ls_active && stp.count <= 6)) {
                    INK_CK(cudaMemsetAsync(fp8_eng.d_inp_amax, 0, 3 * sizeof(float), st));
                    if (pf_bf16_res) {
                        rtx5090::k_ink_attn_res_mix_1pass<true><<<M, 256, 0, st>>>(
                            stp, q_res, d_mixed, nq, M, kInkHidden, nw_q, is_swa ? nullptr : nw_k, is_swa ? nullptr : nw_v, fp8_eng.d_inp_amax);
                    } else {
                        rtx5090::k_ink_attn_res_mix_1pass<false><<<M, 256, 0, st>>>(
                            stp, q_res, d_mixed, nq, M, kInkHidden, nw_q, is_swa ? nullptr : nw_k, is_swa ? nullptr : nw_v, fp8_eng.d_inp_amax);
                    }
                    has_pre_amax = true;
                } else {
                    k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_res, d_mixed, nq, M, kInkHidden, nw_q, is_swa ? nullptr : nw_k, is_swa ? nullptr : nw_v);
                }
                in_q = d_mixed;
                in_k = is_swa ? d_mixed : (d_mixed + 1 * n_hid);
                in_v = is_swa ? d_mixed : (d_mixed + 2 * n_hid);
            } else if (fp8_sm.ready) {
                const int ci = ip - m.man.c.core_start;
                constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                const float* src0 = d_mixed;
                const float* src1 = d_mixed + 1 * n_hid;
                const float* src2 = d_mixed + 2 * n_hid;
                if (M <= 16) {
                    opt5060::k_attn_res_3x_fused_1cta<<<M, 256, 0, st>>>(
                        stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ci],
                        d_mixed, d_msgs, d_ctrl_cat, M, D, nullptr, 0);
                    const auto* tr = fp8_sm.tr;
                    if (M == 1) {
                        opt5060::k_ctrl_3x_down_fp8<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(
                            d_ctrl_cat, tr->w_ctrl_dn[ci][0], tr->sc_ctrl_dn[ci][0], tr->w_ctrl_dn[ci][1], tr->sc_ctrl_dn[ci][1], tr->w_ctrl_dn[ci][2], tr->sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        opt5060::k_ctrl_3x_apply_fp8<1><<<dim3(D / 16, 3), 256, 0, st>>>(
                            src0, src1, src2, d_msgs, d_ctrl_gate, tr->w_ctrl_gf[ci][0], tr->sc_ctrl_gf[ci][0], tr->w_ctrl_gf[ci][1], tr->sc_ctrl_gf[ci][1], tr->w_ctrl_gf[ci][2], tr->sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else if (M <= 4) {
                        opt5060::k_ctrl_3x_down_fp8<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(
                            d_ctrl_cat, tr->w_ctrl_dn[ci][0], tr->sc_ctrl_dn[ci][0], tr->w_ctrl_dn[ci][1], tr->sc_ctrl_dn[ci][1], tr->w_ctrl_dn[ci][2], tr->sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        opt5060::k_ctrl_3x_apply_fp8<4><<<dim3(D / 16, 3), 256, 0, st>>>(
                            src0, src1, src2, d_msgs, d_ctrl_gate, tr->w_ctrl_gf[ci][0], tr->sc_ctrl_gf[ci][0], tr->w_ctrl_gf[ci][1], tr->sc_ctrl_gf[ci][1], tr->w_ctrl_gf[ci][2], tr->sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else if (M <= 8) {
                        opt5060::k_ctrl_3x_down_fp8<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(
                            d_ctrl_cat, tr->w_ctrl_dn[ci][0], tr->sc_ctrl_dn[ci][0], tr->w_ctrl_dn[ci][1], tr->sc_ctrl_dn[ci][1], tr->w_ctrl_dn[ci][2], tr->sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        opt5060::k_ctrl_3x_apply_fp8<8><<<dim3(D / 16, 3), 256, 0, st>>>(
                            src0, src1, src2, d_msgs, d_ctrl_gate, tr->w_ctrl_gf[ci][0], tr->sc_ctrl_gf[ci][0], tr->w_ctrl_gf[ci][1], tr->sc_ctrl_gf[ci][1], tr->w_ctrl_gf[ci][2], tr->sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else {
                        opt5060::k_ctrl_3x_down_fp8<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(
                            d_ctrl_cat, tr->w_ctrl_dn[ci][0], tr->sc_ctrl_dn[ci][0], tr->w_ctrl_dn[ci][1], tr->sc_ctrl_dn[ci][1], tr->w_ctrl_dn[ci][2], tr->sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        opt5060::k_ctrl_3x_apply_fp8<16><<<dim3(D / 16, 3), 256, 0, st>>>(
                            src0, src1, src2, d_msgs, d_ctrl_gate, tr->w_ctrl_gf[ci][0], tr->sc_ctrl_gf[ci][0], tr->w_ctrl_gf[ci][1], tr->sc_ctrl_gf[ci][1], tr->w_ctrl_gf[ci][2], tr->sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    }
                    opt5060::k_rms_norm_3x_h2<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, fp8_sm.d_x_h2, M, D);
                    x_h2_ready = true;
                } else {
                    const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
                    opt5060::k_attn_res_3x_fused_1cta<<<M_pad, 256, 0, st>>>(
                        stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ci],
                        d_mixed, d_msgs, nullptr, M, D, fp8_sm.d_cat3_fp8, M_pad);
                    fp8_sm.run_ctrl_fp8(ci, 0, src0, d_msgs + 0 * n_hid, nw_q, d_inp0, d_ctrl_gate, M, true, st, fp8_sm.d_cat3_fp8 + 0ull * M_pad * D2);
                    fp8_sm.run_ctrl_fp8(ci, 1, src1, d_msgs + 1 * n_hid, nw_k, d_inp1, d_ctrl_gate, M, true, st, fp8_sm.d_cat3_fp8 + 1ull * M_pad * D2);
                    fp8_sm.run_ctrl_fp8(ci, 2, src2, d_msgs + 2 * n_hid, nw_v, d_inp2, d_ctrl_gate, M, true, st, fp8_sm.d_cat3_fp8 + 2ull * M_pad * D2);
                }
                in_q = d_inp0; in_k = d_inp1; in_v = d_inp2;
            } else if (ink_rtx5060ti_cfg(m.primary_gpu).enabled) {
                k_ink_attn_res_3x_cat_rms<<<dim3(M, 4), 256, 0, st>>>(
                    stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ip - m.man.c.core_start],
                    d_mixed, d_msgs, (M <= 16) ? d_ctrl_cat : nullptr, M, kInkHidden);
                const float* src0 = d_mixed;
                const float* src1 = d_mixed + 1 * n_hid;
                const float* src2 = d_mixed + 2 * n_hid;
                const bool ctrl_q8 = (m.tinfo(p + ".controllers.attn.down_weight").fmt == FMT_Q8ROW);
                const uint8_t* wd0_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.down_weight");
                const uint8_t* wg0_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.gain_weight");
                const uint8_t* wf0_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.forget_weight");
                const uint8_t* wd1_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.k.down_weight");
                const uint8_t* wg1_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.k.gain_weight");
                const uint8_t* wf1_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.k.forget_weight");
                const uint8_t* wd2_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.v.down_weight");
                const uint8_t* wg2_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.v.gain_weight");
                const uint8_t* wf2_u = m.trunk_ptr_stage(cur_stage, p + ".controllers.v.forget_weight");
                if (ctrl_q8) {
                    if (M <= 16) {
                        constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                        if (M == 1) {
                            k_ink_ctrl_3x_down_q8_smallM<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0_u, wd1_u, wd2_u, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_q8_smallM<1><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0_u, wg1_u, wg2_u, wf0_u, wf1_u, wf2_u, d_inp0, d_inp1, d_inp2, M, D);
                        } else if (M <= 4) {
                            k_ink_ctrl_3x_down_q8_smallM<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0_u, wd1_u, wd2_u, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_q8_smallM<4><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0_u, wg1_u, wg2_u, wf0_u, wf1_u, wf2_u, d_inp0, d_inp1, d_inp2, M, D);
                        } else if (M <= 8) {
                            k_ink_ctrl_3x_down_q8_smallM<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0_u, wd1_u, wd2_u, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_q8_smallM<8><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0_u, wg1_u, wg2_u, wf0_u, wf1_u, wf2_u, d_inp0, d_inp1, d_inp2, M, D);
                        } else {
                            k_ink_ctrl_3x_down_q8_smallM<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0_u, wd1_u, wd2_u, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_q8_smallM<16><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0_u, wg1_u, wg2_u, wf0_u, wf1_u, wf2_u, d_inp0, d_inp1, d_inp2, M, D);
                        }
                        k_ink_rms_norm_3x<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                    } else {
                        constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                        const int M_pad = (M + 3) & ~3;
                        int8_t* x_i8_3 = reinterpret_cast<int8_t*>(d_x_bf16);
                        float*  x_sc_3 = d_sinks;
                        int32_t* g3_i32 = reinterpret_cast<int32_t*>(d_ctrl_gate);
                        k_ink_ctrl_3x_cat_rms_i8<<<dim3(M, 3), 256, 0, st>>>(d_mixed, d_msgs, x_i8_3, x_sc_3, M, M_pad, D);
                        const uint8_t* wds[3] = { wd0_u, wd1_u, wd2_u };
                        const int32_t alpha_i = 1, beta_i = 0;
                        for (int ci = 0; ci < 3; ++ci) {
                            int8_t* xi = x_i8_3 + (size_t)ci * M_pad * D2;
                            if (M_pad > M) INK_CK(cudaMemsetAsync(xi + (size_t)M * D2, 0, (size_t)(M_pad - M) * D2, st));
                            INK_CUBLAS_CK(cublasGemmEx(
                                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                kInkGateRank, M_pad, D2, &alpha_i,
                                reinterpret_cast<const int8_t*>(wds[ci] + (size_t)kInkGateRank * 4u), CUDA_R_8I, D2,
                                xi, CUDA_R_8I, D2, &beta_i,
                                g3_i32 + (size_t)ci * M_pad * kInkGateRank, CUDA_R_32I, kInkGateRank,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                        }
                        k_ink_ctrl_3x_apply_q8_i32_tile16<<<dim3((D + 31) / 32, (M + 7) / 8, 3), 256, 0, st>>>(
                            d_mixed, d_msgs, g3_i32, wd0_u, wd1_u, wd2_u, x_sc_3,
                            wg0_u, wg1_u, wg2_u, wf0_u, wf1_u, wf2_u, d_inp0, d_inp1, d_inp2, M, M_pad, D);
                        k_ink_rms_norm_3x_fast<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                    }
                } else {
                    const float* wd0 = (const float*)wd0_u; const float* wg0 = (const float*)wg0_u; const float* wf0 = (const float*)wf0_u;
                    const float* wd1 = (const float*)wd1_u; const float* wg1 = (const float*)wg1_u; const float* wf1 = (const float*)wf1_u;
                    const float* wd2 = (const float*)wd2_u; const float* wg2 = (const float*)wg2_u; const float* wf2 = (const float*)wf2_u;
                    if (M <= 16) {
                        constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                        if (M == 1) {
                            k_ink_ctrl_3x_down_smallM<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_smallM<1><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                        } else if (M <= 4) {
                            k_ink_ctrl_3x_down_smallM<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_smallM<4><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                        } else if (M <= 8) {
                            k_ink_ctrl_3x_down_smallM<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_smallM<8><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                        } else {
                            k_ink_ctrl_3x_down_smallM<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                            k_ink_ctrl_3x_apply_smallM<16><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                        }
                        k_ink_rms_norm_3x<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                    } else {
                        run_loop_control_norm(src0, d_msgs + 0 * n_hid, wd0, wg0, wf0, nw_q, d_inp0, M, false, exact_x);
                        run_loop_control_norm(src1, d_msgs + 1 * n_hid, wd1, wg1, wf1, nw_k, d_inp1, M, false, exact_x);
                        run_loop_control_norm(src2, d_msgs + 2 * n_hid, wd2, wg2, wf2, nw_v, d_inp2, M, false, exact_x);
                    }
                }
                in_q = d_inp0; in_k = d_inp1; in_v = d_inp2;
            } else {
                const int ci = ip - m.man.c.core_start;
                const bool use_fp8_c = (!m.ls_active && ci >= 0 && ci < 12 && fp8_eng.cache_ready && fp8_eng.w_ctrl_dn[ci][0] != nullptr);
                const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
                const bool fuse_cat3_fp8 = (M > 16 && use_fp8_c && fp8_eng.lt_ok && fp8_eng.d_cat3_fp8 != nullptr && M_pad <= max_batch && M_pad <= fp8_eng.max_m);
                const bool need_cat3 = (M <= 16) && !fuse_cat3_fp8;
                if (fuse_cat3_fp8) {
                    if (pf_bf16_res) {
                        rtx5090::k_ink_attn_res_3x_fused_1cta<true><<<M_pad, 256, 0, st>>>(
                            stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ci],
                            d_mixed, d_msgs, nullptr, M, kInkHidden, fp8_eng.d_cat3_fp8, M_pad);
                    } else {
                        rtx5090::k_ink_attn_res_3x_fused_1cta<false><<<M_pad, 256, 0, st>>>(
                            stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ci],
                            d_mixed, d_msgs, nullptr, M, kInkHidden, fp8_eng.d_cat3_fp8, M_pad);
                    }
                } else if (rtx5090::dense_opt_on() && M > 4) {
                    rtx5090::k_ink_attn_res_3x_fused_1cta<false><<<M, 256, 0, st>>>(
                        stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ci],
                        d_mixed, d_msgs, need_cat3 ? d_ctrl_cat : nullptr, M, kInkHidden);
                } else {
                    k_ink_attn_res_3x_cat_rms<<<dim3(M, 4), 256, 0, st>>>(
                        stp, q_res, is_swa, *bank_stp, m.d_msg_queries[ci],
                        d_mixed, d_msgs, need_cat3 ? d_ctrl_cat : nullptr, M, kInkHidden);
                }
                const size_t msg_stride = pf_bf16_res ? (n_hid / 2) : n_hid;
                const float* src0 = d_mixed;
                const float* src1 = (is_swa && fuse_cat3_fp8) ? d_mixed : (d_mixed + 1 * msg_stride);
                const float* src2 = (is_swa && fuse_cat3_fp8) ? d_mixed : (d_mixed + 2 * msg_stride);
                const float* wd0 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.down_weight");
                const float* wg0 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.gain_weight");
                const float* wf0 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.attn.forget_weight");
                const float* wd1 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.k.down_weight");
                const float* wg1 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.k.gain_weight");
                const float* wf1 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.k.forget_weight");
                const float* wd2 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.v.down_weight");
                const float* wg2 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.v.gain_weight");
                const float* wf2 = (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.v.forget_weight");
                if (fuse_cat3_fp8) {
                    const size_t cstride = (size_t)M_pad * (2 * kInkHidden);
                    INK_CK(cudaMemsetAsync(fp8_eng.d_inp_amax, 0, 3 * sizeof(float), st));
                    const bool ok0 = fp8_eng.run_ctrl_fp8(ci, 0, src0, d_msgs + 0 * msg_stride, nw_q, d_inp0, d_ctrl_gate, d_ctrl_cat, M, true, st, fp8_eng.d_cat3_fp8 + 0 * cstride, fp8_eng.d_inp_amax + 0, pf_bf16_res);
                    const bool ok1 = fp8_eng.run_ctrl_fp8(ci, 1, src1, d_msgs + 1 * msg_stride, nw_k, d_inp1, d_ctrl_gate, d_ctrl_cat, M, true, st, fp8_eng.d_cat3_fp8 + 1 * cstride, fp8_eng.d_inp_amax + 1, pf_bf16_res);
                    const bool ok2 = fp8_eng.run_ctrl_fp8(ci, 2, src2, d_msgs + 2 * msg_stride, nw_v, d_inp2, d_ctrl_gate, d_ctrl_cat, M, true, st, fp8_eng.d_cat3_fp8 + 2 * cstride, fp8_eng.d_inp_amax + 2, pf_bf16_res);
                    if (ok0 && ok1 && ok2) {
                        has_pre_amax = true;
                    } else {
                        run_loop_control_norm(src0, d_msgs + 0 * n_hid, wd0, wg0, wf0, nw_q, d_inp0, M, false, exact_x, nullptr, ci, 0);
                        run_loop_control_norm(src1, d_msgs + 1 * n_hid, wd1, wg1, wf1, nw_k, d_inp1, M, false, exact_x, nullptr, ci, 1);
                        run_loop_control_norm(src2, d_msgs + 2 * n_hid, wd2, wg2, wf2, nw_v, d_inp2, M, false, exact_x, nullptr, ci, 2);
                    }
                } else if (M <= 16 && use_fp8_c) {
                    constexpr int D = kInkHidden;
                    if (M == 1) {
                        rtx5090::k_ink_ctrl_3x_down_fp8<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, fp8_eng.w_ctrl_dn[ci][0], fp8_eng.sc_ctrl_dn[ci][0], fp8_eng.w_ctrl_dn[ci][1], fp8_eng.sc_ctrl_dn[ci][1], fp8_eng.w_ctrl_dn[ci][2], fp8_eng.sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        rtx5090::k_ink_ctrl_3x_apply_fp8<1><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, fp8_eng.w_ctrl_gf[ci][0], fp8_eng.sc_ctrl_gf[ci][0], fp8_eng.w_ctrl_gf[ci][1], fp8_eng.sc_ctrl_gf[ci][1], fp8_eng.w_ctrl_gf[ci][2], fp8_eng.sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else if (M <= 4) {
                        rtx5090::k_ink_ctrl_3x_down_fp8<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, fp8_eng.w_ctrl_dn[ci][0], fp8_eng.sc_ctrl_dn[ci][0], fp8_eng.w_ctrl_dn[ci][1], fp8_eng.sc_ctrl_dn[ci][1], fp8_eng.w_ctrl_dn[ci][2], fp8_eng.sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        rtx5090::k_ink_ctrl_3x_apply_fp8<4><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, fp8_eng.w_ctrl_gf[ci][0], fp8_eng.sc_ctrl_gf[ci][0], fp8_eng.w_ctrl_gf[ci][1], fp8_eng.sc_ctrl_gf[ci][1], fp8_eng.w_ctrl_gf[ci][2], fp8_eng.sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else if (M <= 8) {
                        rtx5090::k_ink_ctrl_3x_down_fp8<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, fp8_eng.w_ctrl_dn[ci][0], fp8_eng.sc_ctrl_dn[ci][0], fp8_eng.w_ctrl_dn[ci][1], fp8_eng.sc_ctrl_dn[ci][1], fp8_eng.w_ctrl_dn[ci][2], fp8_eng.sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        rtx5090::k_ink_ctrl_3x_apply_fp8<8><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, fp8_eng.w_ctrl_gf[ci][0], fp8_eng.sc_ctrl_gf[ci][0], fp8_eng.w_ctrl_gf[ci][1], fp8_eng.sc_ctrl_gf[ci][1], fp8_eng.w_ctrl_gf[ci][2], fp8_eng.sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else if (M <= 16) {
                        rtx5090::k_ink_ctrl_3x_down_fp8<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, fp8_eng.w_ctrl_dn[ci][0], fp8_eng.sc_ctrl_dn[ci][0], fp8_eng.w_ctrl_dn[ci][1], fp8_eng.sc_ctrl_dn[ci][1], fp8_eng.w_ctrl_dn[ci][2], fp8_eng.sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        rtx5090::k_ink_ctrl_3x_apply_fp8<16><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, fp8_eng.w_ctrl_gf[ci][0], fp8_eng.sc_ctrl_gf[ci][0], fp8_eng.w_ctrl_gf[ci][1], fp8_eng.sc_ctrl_gf[ci][1], fp8_eng.w_ctrl_gf[ci][2], fp8_eng.sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    } else {
                        rtx5090::k_ink_ctrl_3x_down_fp8<24><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, fp8_eng.w_ctrl_dn[ci][0], fp8_eng.sc_ctrl_dn[ci][0], fp8_eng.w_ctrl_dn[ci][1], fp8_eng.sc_ctrl_dn[ci][1], fp8_eng.w_ctrl_dn[ci][2], fp8_eng.sc_ctrl_dn[ci][2], d_ctrl_gate, M);
                        rtx5090::k_ink_ctrl_3x_apply_fp8<24><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, fp8_eng.w_ctrl_gf[ci][0], fp8_eng.sc_ctrl_gf[ci][0], fp8_eng.w_ctrl_gf[ci][1], fp8_eng.sc_ctrl_gf[ci][1], fp8_eng.w_ctrl_gf[ci][2], fp8_eng.sc_ctrl_gf[ci][2], d_inp0, d_inp1, d_inp2, M);
                    }
                    if (M <= 8) {
                        rtx5090::k_ink_rms_norm_3x_h2<<<dim3(M, 3), 256, 0, st>>>(
                            d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, reinterpret_cast<__half2*>(d_x_bf16), M, D);
                        qkv_h2_ready = true;
                    } else {
                        k_ink_rms_norm_3x<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                    }
                } else if (M <= 16) {
                    constexpr int D = kInkHidden, D2 = 2 * kInkHidden;
                    if (M == 1) {
                        k_ink_ctrl_3x_down_smallM<1><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<1><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else if (M <= 4) {
                        k_ink_ctrl_3x_down_smallM<4><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<4><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else if (M <= 8) {
                        k_ink_ctrl_3x_down_smallM<8><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<8><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    } else {
                        k_ink_ctrl_3x_down_smallM<16><<<dim3(kInkGateRank, 3), 256, 0, st>>>(d_ctrl_cat, wd0, wd1, wd2, d_ctrl_gate, M, D2);
                        k_ink_ctrl_3x_apply_smallM<16><<<dim3(D / 16, 3), 256, 0, st>>>(src0, src1, src2, d_msgs, d_ctrl_gate, wg0, wg1, wg2, wf0, wf1, wf2, d_inp0, d_inp1, d_inp2, M, D);
                    }
                    k_ink_rms_norm_3x<<<dim3(M, 3), 256, 0, st>>>(d_inp0, d_inp1, d_inp2, nw_q, nw_k, nw_v, M, D);
                } else {
                    const bool has_bc = (!m.ls_active && ci >= 0 && ci < 12 && d_rtx_ctrl_gf[ci][0] != nullptr);
                    run_loop_control_norm(src0, d_msgs + 0 * n_hid, wd0, wg0, wf0, nw_q, d_inp0, M, false, exact_x,
                                          has_bc ? d_rtx_ctrl_gf[ci][0] : nullptr, ci, 0);
                    run_loop_control_norm(src1, d_msgs + 1 * n_hid, wd1, wg1, wf1, nw_k, d_inp1, M, false, exact_x,
                                          has_bc ? d_rtx_ctrl_gf[ci][1] : nullptr, ci, 1);
                    run_loop_control_norm(src2, d_msgs + 2 * n_hid, wd2, wg2, wf2, nw_v, d_inp2, M, false, exact_x,
                                          has_bc ? d_rtx_ctrl_gf[ci][2] : nullptr, ci, 2);
                }
                in_q = d_inp0; in_k = d_inp1; in_v = d_inp2;
            }
        } else {
            k_ink_rms_norm<<<M, 256, 0, st>>>(d_cur, nw_q, d_inp0, nullptr, M, kInkHidden);
            in_q = in_k = in_v = d_inp0;
        }

        // 3. Fused Q, K, V, R, Gate_Attn, Sinks projections (+ Dynamic K/V ShortConv tap projections)
        prof_mark_end(0); prof_mark_start();
        const bool share_qkv = (in_q == in_k && in_q == in_v);
        // Q8_0 dense matrices: one fused launch (one activation quantization) at any batch size, prefill chunks included
        const bool q8f_dense = (m.tinfo(p + ".attn.wq_du.weight").fmt == FMT_Q8_0F);
        bool qkv_fp8_done = false;
        bool qkv_bf16_io  = false;
        const int M_pad32 = (M <= 32) ? 32 : ((M + 31) & ~31);
        const bool try_pf_bf16_io = (mode == 0 && M >= 128 && rtx5090::pf_attn_on() &&
                                     fp8_eng.cache_ready && fp8_eng.lt_ok && !m.ls_active && !q8f_dense &&
                                     fp8_eng.w_attn[ip][0] != nullptr && d_rtx_rel_proj_T[ip] != nullptr &&
                                     kv_desc[visit].kv_fmt == 0 && kv_desc[visit].host_cap == 0 &&
                                     M_pad32 <= max_batch && M_pad32 <= fp8_eng.max_m && !dbg_hash_on);
        bool sink_sconv_done = false;
        auto run_sink_sconv_fused = [&]() {
            if (!rtx5090::dense_opt_on() || M > 64 || m.ls_active) return;
            const auto& ti_s = m.tinfo(p + ".attn.attn_sink_proj.weight");
            if (ti_s.fmt != FMT_F32) return;
            const float* w_s = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.attn_sink_proj.weight");
            const float* w_kd = nullptr;
            const float* w_vd = nullptr;
            if (!is_swa) {
                if (m.tinfo(p + ".attn.k_sconv.weight_proj").fmt != FMT_F32 ||
                    m.tinfo(p + ".attn.v_sconv.weight_proj").fmt != FMT_F32) return;
                w_kd = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.k_sconv.weight_proj");
                w_vd = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.v_sconv.weight_proj");
            }
            const int n_rows = is_swa ? kInkHeads : (kInkHeads + 2 * kInkKVHeads * 4);
            if (M == 1) {
                rtx5090::k_ink_f32_sink_sconv_fused<1><<<dim3(n_rows, 1), 256, 0, st>>>(
                    w_s, w_kd, w_vd, in_q, in_k, in_v, d_sinks, d_k_dyn, d_v_dyn, M);
            } else if (M <= 4) {
                rtx5090::k_ink_f32_sink_sconv_fused<4><<<dim3(n_rows, 1), 256, 0, st>>>(
                    w_s, w_kd, w_vd, in_q, in_k, in_v, d_sinks, d_k_dyn, d_v_dyn, M);
            } else {
                rtx5090::k_ink_f32_sink_sconv_fused<16><<<dim3(n_rows, (M + 15) / 16), 256, 0, st>>>(
                    w_s, w_kd, w_vd, in_q, in_k, in_v, d_sinks, d_k_dyn, d_v_dyn, M);
            }
            sink_sconv_done = true;
        };
        float* k_dyn_ptr = nullptr;
        float* v_dyn_ptr = nullptr;
        if (fp8_sm.ready) {
            constexpr int D = kInkHidden;
            const auto* tr = fp8_sm.tr;
            const float* w_sink = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.attn_sink_proj.weight");
            const float* w_kd   = is_swa ? nullptr : (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.k_sconv.weight_proj");
            const float* w_vd   = is_swa ? nullptr : (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.v_sconv.weight_proj");
            if (!is_swa) { k_dyn_ptr = d_k_dyn; v_dyn_ptr = d_v_dyn; }
            const int sink_rows = is_swa ? kInkHeads : (kInkHeads + 2 * kInkKVHeads * 4);
            if (M <= 16) {
                if (!x_h2_ready) {
                    const size_t n_pairs = (size_t)(share_qkv ? 1 : 3) * M * (D / 2);
                    opt5060::k_f32_to_h2<<<(n_pairs + 255) / 256, 256, 0, st>>>(in_q, fp8_sm.d_x_h2, n_pairs);
                }
                opt5060::InkMultiProjFp8 mp{};
                mp.count = 5;
                mp.cumN[0] = 0;
                static constexpr int kNs[5] = { kInkQDim, kInkHeads * kInkDRel, kInkQDim, kInkKVDim, kInkKVDim };
                float* ys[5] = { d_q, d_r, d_gate_attn, d_k_raw, d_v_raw };
                for (int i = 0; i < 5; ++i) {
                    mp.W[i] = tr->w_attn[ip][i]; mp.sc[i] = tr->sc_attn[ip][i]; mp.Y[i] = ys[i];
                    mp.N[i] = kNs[i]; mp.cumN[i + 1] = mp.cumN[i] + kNs[i];
                }
                const __half2* xh_k = share_qkv ? nullptr : (fp8_sm.d_x_h2 + (size_t)M * (D / 2));
                const __half2* xh_v = share_qkv ? nullptr : (fp8_sm.d_x_h2 + 2ull * M * (D / 2));
                const int g8 = (mp.cumN[5] + 7) / 8;
                if (M == 1) {
                    opt5060::k_gemv_multi_fp8_m1_l1<<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, xh_k, xh_v, D);
                    opt5060::k_f32_sink_sconv_fused<1><<<dim3(sink_rows, 1), 256, 0, st>>>(w_sink, w_kd, w_vd, in_q, in_k, in_v, d_sinks, k_dyn_ptr, v_dyn_ptr, 1);
                } else {
                    if (M <= 4)      opt5060::k_gemv_multi_fp8_smallM_l1<4><<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, xh_k, xh_v, M, D);
                    else if (M <= 8) opt5060::k_gemv_multi_fp8_smallM_l1<8><<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, xh_k, xh_v, M, D);
                    else             opt5060::k_gemv_multi_fp8_smallM_l1<16><<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, xh_k, xh_v, M, D);
                    opt5060::k_f32_sink_sconv_fused<4><<<dim3(sink_rows, (M + 3) / 4), 256, 0, st>>>(w_sink, w_kd, w_vd, in_q, in_k, in_v, d_sinks, k_dyn_ptr, v_dyn_ptr, M);
                }
            } else {
                fp8_sm.gemm_fp8_packed_qkv(ip, share_qkv, in_q, d_q, d_r, d_gate_attn, d_k_raw, d_v_raw, M, st);
                if (!share_qkv) {
                    fp8_sm.gemm_fp8_tn(tr->w_attn[ip][3], tr->sc_attn[ip][3], in_k, d_k_raw, M, kInkKVDim, D, false, st);
                    fp8_sm.gemm_fp8_tn(tr->w_attn[ip][4], tr->sc_attn[ip][4], in_v, d_v_raw, M, kInkKVDim, D, false, st);
                }
                linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M);
                if (!is_swa) {
                    linear(p + ".attn.k_sconv.weight_proj", in_k, d_k_dyn, M);
                    linear(p + ".attn.v_sconv.weight_proj", in_v, d_v_dyn, M);
                }
            }
            if (visit == dbg_tap_visit) { dbg_hash(201, in_q, M, kInkHidden); dbg_hash(199, d_q, M, kInkHeads * kInkHeadDim); dbg_hash(200, d_k_raw, M, kInkKVDim); dbg_hash(202, d_r, M, kInkHeads * kInkDRel); dbg_hash(203, d_gate_attn, M, kInkHeads * kInkHeadDim); }
        } else {
        if (fp8_eng.cache_ready && !m.ls_active && !q8f_dense && fp8_eng.w_attn[ip][0] != nullptr) {
            constexpr int kRDim = kInkHeads * kInkDRel;
            if (M <= 8 || !fp8_eng.lt_ok || M_pad32 > max_batch) {
                rtx5090::InkMultiProjFp8 mp{};
                const int ns[5] = {kInkQDim, kRDim, kInkQDim, kInkKVDim, kInkKVDim};
                const float* xs[5] = {in_q, in_q, in_q, in_k, in_v};
                float* ys[5] = {d_q, d_r, d_gate_attn, d_k_raw, d_v_raw};
                mp.count = 5; mp.cumN[0] = 0;
                for (int i = 0; i < 5; ++i) {
                    mp.W[i] = fp8_eng.w_attn[ip][i]; mp.sc[i] = fp8_eng.sc_attn[ip][i];
                    mp.X[i] = xs[i]; mp.Y[i] = ys[i]; mp.N[i] = ns[i]; mp.cumN[i + 1] = mp.cumN[i] + ns[i];
                }
                if (M == 1) {
                    __half2* xh0 = reinterpret_cast<__half2*>(d_x_bf16);
                    __half2* xhk = share_qkv ? xh0 : (xh0 + kInkHidden / 2);
                    __half2* xhv = share_qkv ? xh0 : (xh0 + kInkHidden);
                    if (!qkv_h2_ready) {
                        rtx5090::k_ink_f32_to_h2<<<(kInkHidden / 2 + 255) / 256, 256, 0, st>>>(in_q, xh0, kInkHidden / 2);
                        if (!share_qkv) {
                            rtx5090::k_ink_f32_to_h2<<<(kInkHidden / 2 + 255) / 256, 256, 0, st>>>(in_k, xhk, kInkHidden / 2);
                            rtx5090::k_ink_f32_to_h2<<<(kInkHidden / 2 + 255) / 256, 256, 0, st>>>(in_v, xhv, kInkHidden / 2);
                        }
                    }
                    rtx5090::k_ink_gemv_multi_fp8_m1_l1<<<(mp.cumN[5] + 7) / 8, 256, 0, st>>>(mp, xh0, xhk, xhv, kInkHidden);
                } else if (M <= 8) {
                    const size_t n2 = (size_t)M * (kInkHidden / 2);
                    __half2* xh0 = reinterpret_cast<__half2*>(d_x_bf16);
                    __half2* xhk = share_qkv ? xh0 : (xh0 + n2);
                    __half2* xhv = share_qkv ? xh0 : (xh0 + 2 * n2);
                    if (!qkv_h2_ready) {
                        rtx5090::k_ink_f32_to_h2<<<(n2 + 255) / 256, 256, 0, st>>>(in_q, xh0, n2);
                        if (!share_qkv) {
                            rtx5090::k_ink_f32_to_h2<<<(n2 + 255) / 256, 256, 0, st>>>(in_k, xhk, n2);
                            rtx5090::k_ink_f32_to_h2<<<(n2 + 255) / 256, 256, 0, st>>>(in_v, xhv, n2);
                        }
                    }
                    if (M <= 4) rtx5090::k_ink_gemv_multi_fp8_smallM_l1<4><<<(mp.cumN[5] + 7) / 8, 256, 0, st>>>(mp, xh0, xhk, xhv, M, kInkHidden);
                    else        rtx5090::k_ink_gemv_multi_fp8_smallM_l1<8><<<(mp.cumN[5] + 7) / 8, 256, 0, st>>>(mp, xh0, xhk, xhv, M, kInkHidden);
                } else {
                    rtx5090::k_ink_gemv_multi_fp8<8><<<(mp.cumN[5] + 3) / 4, 64, 0, st>>>(mp, in_q, M, kInkHidden);
                }
                run_sink_sconv_fused();
                if (!sink_sconv_done) linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
                qkv_fp8_done = true;
            } else if (fp8_eng.lt_ok) {
                const float* am0 = has_pre_amax ? (fp8_eng.d_inp_amax + 0) : nullptr;
                const float* am1 = has_pre_amax ? (fp8_eng.d_inp_amax + (share_qkv ? 0 : 1)) : nullptr;
                const float* am2 = has_pre_amax ? (fp8_eng.d_inp_amax + (share_qkv ? 0 : 2)) : nullptr;
                if (try_pf_bf16_io) {
                    bool ok = fp8_eng.gemm_fp8_packed_qkv_bf16(
                        ip, share_qkv, in_q, reinterpret_cast<__nv_bfloat16*>(d_ctrl_cat),
                        reinterpret_cast<__nv_bfloat16*>(d_q), reinterpret_cast<__nv_bfloat16*>(d_r),
                        reinterpret_cast<__nv_bfloat16*>(d_gate_attn), reinterpret_cast<__nv_bfloat16*>(d_k_raw),
                        reinterpret_cast<__nv_bfloat16*>(d_v_raw), M, st, am0);
                    if (ok && !share_qkv) {
                        ok = fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][3], fp8_eng.sc_attn[ip][3], in_k, reinterpret_cast<__nv_bfloat16*>(d_k_raw), M, kInkKVDim, kInkHidden, false, st, am1)
                          && fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][4], fp8_eng.sc_attn[ip][4], in_v, reinterpret_cast<__nv_bfloat16*>(d_v_raw), M, kInkKVDim, kInkHidden, false, st, am2);
                    }
                    if (!ok) {
                        ok = fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][0], fp8_eng.sc_attn[ip][0], in_q, reinterpret_cast<__nv_bfloat16*>(d_q), M, kInkQDim, kInkHidden, false, st, am0)
                          && fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][1], fp8_eng.sc_attn[ip][1], in_q, reinterpret_cast<__nv_bfloat16*>(d_r), M, kRDim, kInkHidden, true, st)
                          && fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][2], fp8_eng.sc_attn[ip][2], in_q, reinterpret_cast<__nv_bfloat16*>(d_gate_attn), M, kInkQDim, kInkHidden, true, st)
                          && fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][3], fp8_eng.sc_attn[ip][3], in_k, reinterpret_cast<__nv_bfloat16*>(d_k_raw), M, kInkKVDim, kInkHidden, share_qkv, st, am1)
                          && fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][4], fp8_eng.sc_attn[ip][4], in_v, reinterpret_cast<__nv_bfloat16*>(d_v_raw), M, kInkKVDim, kInkHidden, share_qkv, st, am2);
                    }
                    if (ok) {
                        linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M);
                        qkv_fp8_done = true;
                        qkv_bf16_io  = true;
                    }
                }
                if (!qkv_fp8_done && M <= 256) {
                    bool ok = fp8_eng.gemm_fp8_packed_qkv(ip, share_qkv, in_q, d_ctrl_cat, d_q, d_r, d_gate_attn, d_k_raw, d_v_raw, M, st, am0);
                    if (ok && !share_qkv) {
                        ok = fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][3], fp8_eng.sc_attn[ip][3], in_k, d_k_raw, M, kInkKVDim, kInkHidden, false, st, am1)
                          && fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][4], fp8_eng.sc_attn[ip][4], in_v, d_v_raw, M, kInkKVDim, kInkHidden, false, st, am2);
                    }
                    if (ok) {
                        run_sink_sconv_fused();
                        if (!sink_sconv_done) linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M);
                        qkv_fp8_done = true;
                    }
                }
                if (!qkv_fp8_done) {
                    bool ok = fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][0], fp8_eng.sc_attn[ip][0], in_q, d_q, M, kInkQDim, kInkHidden, false, st, am0)
                           && fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][1], fp8_eng.sc_attn[ip][1], in_q, d_r, M, kRDim, kInkHidden, true, st)
                           && fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][2], fp8_eng.sc_attn[ip][2], in_q, d_gate_attn, M, kInkQDim, kInkHidden, true, st)
                           && fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][3], fp8_eng.sc_attn[ip][3], in_k, d_k_raw, M, kInkKVDim, kInkHidden, share_qkv, st, am1)
                           && fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][4], fp8_eng.sc_attn[ip][4], in_v, d_v_raw, M, kInkKVDim, kInkHidden, share_qkv, st, am2);
                    if (ok) {
                        run_sink_sconv_fused();
                        if (!sink_sconv_done) linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M);
                        qkv_fp8_done = true;
                    }
                }
            }
        }
        if (qkv_fp8_done) {
            // handled via RTX 5090 FP8 engine
        } else if ((M <= 16 || q8f_dense) && share_qkv) {
            const std::string names[5] = {
                p + ".attn.wq_du.weight", p + ".attn.wr_du.weight", p + ".attn.gate_attn.weight",
                p + ".attn.wk_dv.weight", p + ".attn.wv_dv.weight"
            };
            float* ys[5] = { d_q, d_r, d_gate_attn, d_k_raw, d_v_raw };
            linear_multi(names, ys, 5, in_q, M, exact_x);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
        } else if (M <= 16 || q8f_dense) {
            const std::string names[3] = {
                p + ".attn.wq_du.weight", p + ".attn.wr_du.weight", p + ".attn.gate_attn.weight"
            };
            float* ys[3] = { d_q, d_r, d_gate_attn };
            linear_multi(names, ys, 3, in_q, M, exact_x);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, 0, false, exact_x);
            linear(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, 0, false, exact_x);
            linear(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, 0, false, exact_x);
        } else if (rtx5090::dense_opt_on() && M <= 256 && !m.ls_active && d_rtx_qkv[ip] != nullptr) {
            // Opt-in RTX 5090: Single fused BF16 cuBLAS GEMM for Q+R+Gate(+K+V) into d_ctrl_cat + 1 split launch!
            const int kRDim = kInkHeads * kInkDRel;
            const int n_fused = share_qkv ? (kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim)
                                          : (kInkQDim + kRDim + kInkQDim);
            gemm_bf16_compat(d_rtx_qkv[ip], in_q, d_ctrl_cat, M, n_fused, kInkHidden, n_fused, false, false);
            rtx5090::k_ink_split_qkv_5<<<dim3((n_fused + 255) / 256, M), 256, 0, st>>>(
                d_ctrl_cat, d_q, d_r, d_gate_attn, d_k_raw, d_v_raw, M, share_qkv);
            linear(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M);
            if (!share_qkv) {
                XCacheFmt x_k = XC_NONE, x_v = XC_NONE;
                linear_cached(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, x_k);
                linear_cached(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, x_v);
            }
        } else {
            XCacheFmt x_q = XC_NONE;
            linear_cached(p + ".attn.wq_du.weight", in_q, d_q, M, x_q);
            linear_cached(p + ".attn.wr_du.weight", in_q, d_r, M, x_q);
            linear_cached(p + ".attn.gate_attn.weight", in_q, d_gate_attn, M, x_q);
            linear_cached(p + ".attn.attn_sink_proj.weight", in_q, d_sinks, M, x_q);
            XCacheFmt x_k = XC_NONE, x_v = XC_NONE;
            linear_cached(p + ".attn.wk_dv.weight", in_k, d_k_raw, M, share_qkv ? x_q : x_k);
            linear_cached(p + ".attn.wv_dv.weight", in_v, d_v_raw, M, share_qkv ? x_q : x_v);
        }

        if (visit == dbg_tap_visit) { dbg_hash(201, in_q, M, kInkHidden); dbg_hash(199, d_q, M, kInkHeads * kInkHeadDim); dbg_hash(200, d_k_raw, M, kInkKVDim); dbg_hash(202, d_r, M, kInkHeads * kInkDRel); dbg_hash(203, d_gate_attn, M, kInkHeads * kInkHeadDim); }
        // 4. Dynamic K/V ShortConv tap projections on global layers
        if (!is_swa) {
            if (!sink_sconv_done) {
                linear(p + ".attn.k_sconv.weight_proj", in_k, d_k_dyn, M);
                linear(p + ".attn.v_sconv.weight_proj", in_v, d_v_dyn, M);
            }
            k_dyn_ptr = d_k_dyn;
            v_dyn_ptr = d_v_dyn;
            }
        }
        prof_mark_end(1); prof_mark_start();
        auto run_sconv_uniform = [&](const float* src, const float* sw, const float* dyn, float* dst, int C, int ch_off, int nh, bool add_to_out = false) {
            float* hbase = d_sconv_pool[visit] + (size_t)ch_off * 3 * max_slots;
            if (mode == 0) {
                float* hslot = hbase + (size_t)slot0 * 3 * C;
                if (M > 16) {
                    dim3 g((C + 255) / 256, (M + 15) / 16);
                    k_ink_sconv4_seq_par<<<g, 256, 0, st>>>(src, sw, dyn, hslot, dst, M, C, nh, add_to_out);
                    k_ink_sconv4_save_hist<<<(C + 255) / 256, 256, 0, st>>>(src, hslot, M, C);
                } else {
                    k_ink_sconv4_seq<<<(C + 255) / 256, 256, 0, st>>>(src, sw, dyn, hslot, dst, M, C, nh, true, add_to_out);
                }
            } else if (mode == 1) {
                dim3 g((C + 255) / 256, M);
                k_ink_sconv4_batched<<<g, 256, 0, st>>>(src, sw, dyn, hbase, d_slots, dst, M, C, nh, add_to_out);
            } else if (mode == 3 && d_bspec_sconv != nullptr) {
                // 5090-family Mode-3 batched speculative verify (bspec staging)
                dim3 g((C + 255) / 256, M);
                rtx5090::k_ink_sconv4_bspec<<<g, 256, 0, st>>>(
                    src, sw, hbase, d_slots, dst,
                    d_bspec_sconv + (size_t)visit * bspec_m_cap * kInkSconvCh,
                    ch_off, M, C, bspec_k_slot, add_to_out);
            } else if (mode == 3) {
                // 5060 Ti-family multi-chain speculative verify (K8V4 multi-slot DSpark)
                dim3 g((C + 255) / 256, mc_cur_B);
                uint16_t* stg = d_mc_sconv_stage ? (d_mc_sconv_stage + (size_t)visit * mc_cur_B * mc_cur_K * kInkSconvCh) : nullptr;
                k_ink_sconv4_multichain<<<g, 256, 0, st>>>(src, sw, dyn, hbase, d_mc_slot_ids, dst, stg, ch_off, mc_cur_B, mc_cur_K, C, nh, add_to_out);
            } else {
                dim3 g((C + 255) / 256, M);
                k_ink_sconv4_tree<<<g, 256, 0, st>>>(
                    src, sw, dyn, hbase + (size_t)slot0 * 3 * C, d_sconv_lag, dst,
                    d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh, ch_off, M, C, nh, add_to_out);
            }
        };

        // 5-6. FUSED Post-QKV (K/V ShortConv4 + Q/K/V/R Clip + V RMSNorm + Global Tau + KV Write/Stage) in 1 launch for decode/batched/tree!
        const auto& ti_qc = m.tinfo(p + ".attn.q_l2_clip.scale");
        const auto& ti_rc = m.tinfo(p + ".attn.rel_l2_clip.scale");
        const float* p_qc = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.q_l2_clip.scale");
        const float* p_rc = (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.rel_l2_clip.scale");
        const int qc_elems = (int)(ti_qc.bytes / 4);
        const int rc_elems = (int)(ti_rc.bytes / 4);
        const bool t2_write = (!is_swa && elastic_kv.tier2_active_for_pos(cur_max_pos));
        const bool cur_mc_pf = (mode == 3 && mc_kv_pf_active && mc_pf_slot[visit] >= 0);
        InkKvVisitDesc cur_mc_sdesc{};
        if (mode == 3 && d_bspec_k && d_bspec_v && d_bspec_sconv) {
            float* hk_pool = d_sconv_pool[visit];
            float* hv_pool = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots;
            rtx5090::k_ink_post_qkv_fused_bspec<<<dim3(64, M), 128, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k_raw, d_v_raw, m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_pool, hv_pool,
                d_k, d_v, d_r, p_rc, rc_elems,
                d_slots, d_pos,
                d_bspec_k + (size_t)visit * bspec_m_cap * kInkKVDim,
                d_bspec_v + (size_t)visit * bspec_m_cap * kInkKVDim,
                d_bspec_sconv + (size_t)visit * bspec_m_cap * kInkSconvCh,
                !is_swa, bspec_k_slot, M);
        } else if ((mode == 0 && M == 1) || (mode == 1 && !t2_write) || mode == 2) {
            float* hk_pool = d_sconv_pool[visit];
            float* hv_pool = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots;
            InkKvVisitDesc wdesc = (mode == 0 && M == 1 && t2_write)
                ? elastic_kv.prepare_single_write_desc(visit, p0_val, slot0, st)
                : kv_desc[visit];
            k_ink_post_qkv_fused<<<dim3(64, M), 128, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k_raw, d_v_raw, m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_pool, hv_pool,
                d_k, d_v, d_r, p_rc, rc_elems,
                (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                (mode == 2) ? d_sconv_lag : nullptr,
                (mode == 2) ? (d_stage_k + (size_t)visit * tree_cap * kInkKVDim) : nullptr,
                (mode == 2) ? (d_stage_v + (size_t)visit * tree_cap * kInkKVDim) : nullptr,
                (mode == 2) ? (d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh) : nullptr,
                wdesc, !is_swa, mode, M);
        } else if (qkv_bf16_io && !t2_write) {
            float* hk_slot = d_sconv_pool[visit] + (size_t)slot0 * 3 * kInkKVDim;
            float* hv_slot = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots + (size_t)slot0 * 3 * kInkKVDim;
            rtx5090::k_ink_post_qkv_prefill_sm120_bf16<<<M, 256, 0, st>>>(
                reinterpret_cast<__nv_bfloat16*>(d_q), p_qc, qc_elems,
                reinterpret_cast<const __nv_bfloat16*>(d_k_raw), reinterpret_cast<const __nv_bfloat16*>(d_v_raw),
                m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_slot, hv_slot,
                reinterpret_cast<__nv_bfloat16*>(d_r), p_rc, rc_elems,
                p0_val, slot0, kv_desc[visit], !is_swa, M);
            rtx5090::k_ink_save_kv_hist_2x_bf16<<<1, 256, 0, st>>>(
                reinterpret_cast<const __nv_bfloat16*>(d_k_raw), reinterpret_cast<const __nv_bfloat16*>(d_v_raw),
                hk_slot, hv_slot, M);
        } else if (mode == 0 && M > 16 && rtx5090::dense_opt_on() && kv_desc[visit].kv_fmt == 0 && kv_desc[visit].host_cap == 0 && !t2_write) {
            float* hk_slot = d_sconv_pool[visit] + (size_t)slot0 * 3 * kInkKVDim;
            float* hv_slot = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots + (size_t)slot0 * 3 * kInkKVDim;
            rtx5090::k_ink_post_qkv_prefill_sm120<<<M, 256, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k_raw, d_v_raw, m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_slot, hv_slot,
                d_r, p_rc, rc_elems,
                p0_val, slot0, kv_desc[visit], !is_swa, M);
            rtx5090::k_ink_save_kv_hist_2x<<<1, 256, 0, st>>>(d_k_raw, d_v_raw, hk_slot, hv_slot, M);
        } else if (mode == 3 && !t2_write) {
            // Fused 1-Launch Mode-3 Post-QKV (K/V ShortConv4 + Stage + Clip/RMS/Tau + KV Write + Optional sdesc Write!)
            if (cur_mc_pf) {
                const int cur_b = mc_pf_slot[visit];
                INK_CK(cudaStreamWaitEvent(st, c64_kv_pf.ev_dma[cur_b], 0));
                cur_mc_sdesc = c64_kv_pf.make_desc(cur_b, kv_desc[visit]);
            }
            float* hk_pool = d_sconv_pool[visit];
            float* hv_pool = d_sconv_pool[visit] + (size_t)kInkKVDim * 3 * max_slots;
            uint16_t* stg = d_mc_sconv_stage ? (d_mc_sconv_stage + (size_t)visit * mc_cur_B * mc_cur_K * kInkSconvCh) : nullptr;
            k_ink_post_qkv_multichain<<<dim3(43, mc_cur_B), 128, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k_raw, d_v_raw, m.d_sconv_k[ip], m.d_sconv_v[ip],
                k_dyn_ptr, v_dyn_ptr, hk_pool, hv_pool, stg,
                d_r, p_rc, rc_elems, d_mc_slot_ids, d_mc_base_pos,
                kv_desc[visit], cur_mc_sdesc, cur_mc_pf, !is_swa, mc_cur_B, mc_cur_K);
        } else {
            run_sconv_uniform(d_k_raw, m.d_sconv_k[ip], k_dyn_ptr, d_k, kInkKVDim, 0, kInkKVHeads);
            run_sconv_uniform(d_v_raw, m.d_sconv_v[ip], v_dyn_ptr, d_v, kInkKVDim, kInkKVDim, kInkKVHeads);
            k_ink_qkvr_clip_tau<<<dim3(64, M), 128, 0, st>>>(
                d_q, p_qc, qc_elems,
                d_k, d_v, d_r, p_rc, rc_elems,
                d_pos, !is_swa, M);
            k_ink_kv_write<<<dim3(kInkKVHeads, M), 128, 0, st>>>(d_k, d_v, (mode == 1 || mode == 3) ? d_slots : nullptr, slot0, d_pos, kv_desc[visit], M);
            if (t2_write) {
                const int p_min = (mode == 1 || mode == 3) ? 0 : p0_val;
                const int p_max = (mode == 1 || mode == 3) ? cur_max_pos : (p0_val + M - 1);
                elastic_kv.write_tier2_kv_batch(visit, d_k, d_v, (mode == 1 || mode == 3) ? d_slots : nullptr, slot0, d_pos, p_min, p_max, M, st);
            }
        }

        // 7. RelPos Table + Fused Sink Flash-GQA (Adaptive Split-K 16/32 for M==1 (#6591), O(1)-VRAM KV-Tiled TC for prefill M>=128 (#6609/#6605)!)
        prof_mark_end(2); prof_mark_start();
        bool has_attn_amax = false;
        bool wo_h2_ready = false;
        {
            const int cur_cc = m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc;
            dim3 gr(kInkHeads, M);
            const bool t2_attn = (!is_swa && elastic_kv.tier2_active_for_pos(cur_max_pos));
            const bool use_rtx_pf_attn = (mode == 0 && M >= 128 && !t2_attn && rtx5090::pf_attn_on() && d_rtx_rel_proj_T[ip] != nullptr);
            const bool want_wo_h2 = (M <= 8 && !t2_attn && fp8_eng.cache_ready && !m.ls_active && fp8_eng.w_attn[ip][5] != nullptr);
            const bool use_fp16_flash = (mode == 0 && M >= 32 && fp8_sm.ready && cur_cc >= 80 && !t2_attn);
            const bool use_mc_fp8rel = false;
            const bool mc_bspec = (mode == 3 && d_bspec_k && d_bspec_v && d_bspec_sconv);
            if (!use_rtx_pf_attn && !use_fp16_flash && !use_mc_fp8rel) {
                const int rel_md = (mode == 3 && !mc_bspec) ? std::min(kInkRelExtent, cur_max_pos + 1) : kInkRelExtent;
                if (rel_md == kInkRelExtent) k_ink_rel_table<<<gr, 256, 0, st>>>(d_r, (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.rel_logits_proj.proj"), d_rel_table, M);
                else                         k_ink_rel_table_md<<<gr, 256, 0, st>>>(d_r, (const float*)m.trunk_ptr_stage(cur_stage, p + ".attn.rel_logits_proj.proj"), d_rel_table, M, rel_md);
            }
            if (use_fp16_flash) {
                const __half2* rp_T = fp8_sm.d_rel_proj_T + (size_t)ip * kInkRelExtent * 8u;
                dim3 gfa(kInkHeads, (M + 63) / 64);
                opt5060::k_flash_gqa_pf_fp16acc<64><<<gfa, 128, 0, st>>>(
                    d_q, d_r, rp_T, d_sinks, d_gate_attn, kv_desc[visit], slot0, p0_val, M, d_attn_out);
            } else if (mc_bspec) {
                // fast only (TC_ATTN): batched drafting (C2..C16) at long context: one group per slot, the slot's K chain rows share every K/V tile; the chain's own K/V (d_k, d_v) is the tail
                bool tc_done = false;
                if (!is_swa && sm120_cfg(m.primary_gpu).tc_attn && tcattn::force_ref() != 0 && bspec_k_slot > 0 && M <= tree_cap) {
                    tc_done = tcattn::launch(st, d_q, d_rel_table, d_sinks, kv_desc[visit], d_slots, 0, d_pos, d_k, d_v, nullptr, nullptr, 0,
                                             d_attn_pm, d_attn_pl, d_attn_po, M, bspec_k_slot);
                    if (tc_done) tcattn::merge(st, d_attn_pm, d_attn_pl, d_attn_po, d_gate_attn, d_attn_out, nullptr, d_pos, true, kv_desc[visit].cap_pos, M, bspec_k_slot);
                }
                if (!tc_done) {
                dim3 g3(kInkKVHeads, M);
                rtx5090::k_ink_flash_gqa_bspec<<<g3, 128, 0, st>>>(
                    d_q, d_rel_table, d_sinks, d_gate_attn,
                    kv_desc[visit], d_slots, d_pos,
                    d_k, d_v, d_attn_out, bspec_k_slot, M);
                }
            } else if (mode == 3) {
                dim3 gmc(kInkKVHeads, mc_cur_B);
                const __half2* rp_T = use_mc_fp8rel ? (fp8_sm.d_rel_proj_T + (size_t)ip * kInkRelExtent * 8u) : nullptr;
                if (cur_mc_pf) {
                    const int cur_b = mc_pf_slot[visit];
                    // Launch NEXT non-pinned visit prefetch into the OTHER ping-pong buffer BEFORE running current attention!
                    const int nv = mc_pf_next[visit];
                    if (nv >= 0) {
                        const int nb = mc_pf_slot[nv];
                        if (!c64_kv_pf.prefetch_visit(nb, kv_desc[nv],
                                host_kv_raw_k[nv], host_kv_raw_v[nv], host_kv_raw_ksc[nv], host_kv_raw_vsc[nv],
                                cur_max_pos + 1, true)) {
                            mc_kv_pf_active = false;
                        }
                    }
                    if (t2_write) {
                        INK_CK(cudaStreamWaitEvent(st, c64_kv_pf.ev_dma[cur_b], 0));
                        cur_mc_sdesc = c64_kv_pf.make_desc(cur_b, kv_desc[visit]);
                        k_ink_kv_write<<<dim3(kInkKVHeads, M), 128, 0, st>>>(d_k, d_v, d_slots, 0, d_pos, cur_mc_sdesc, M);
                    }
                    if (use_mc_fp8rel) {
                        if (mc_cur_K <= 4) {
                            opt5060::k_flash_gqa_multichain_fp8rel<4><<<gmc, 128, 0, st>>>(
                                d_q, d_r, rp_T, d_sinks, d_gate_attn,
                                cur_mc_sdesc, d_mc_slot_ids, d_mc_base_pos, d_attn_out, mc_cur_B, mc_cur_K);
                        } else {
                            opt5060::k_flash_gqa_multichain_fp8rel<8><<<gmc, 128, 0, st>>>(
                                d_q, d_r, rp_T, d_sinks, d_gate_attn,
                                cur_mc_sdesc, d_mc_slot_ids, d_mc_base_pos, d_attn_out, mc_cur_B, mc_cur_K);
                        }
                    } else if (mc_cur_K == 3) {
                        k_ink_flash_gqa_multichain_k3<<<gmc, 96, 0, st>>>(
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            cur_mc_sdesc, d_mc_slot_ids, d_mc_base_pos,
                            d_attn_out, mc_cur_B);
                    } else if (mc_cur_K <= 4) {
                        k_ink_flash_gqa_multichain<4><<<gmc, 128, 0, st>>>(
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            cur_mc_sdesc, d_mc_slot_ids, d_mc_base_pos,
                            d_attn_out, mc_cur_B, mc_cur_K);
                    } else {
                        k_ink_flash_gqa_multichain<8><<<gmc, 128, 0, st>>>(
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            cur_mc_sdesc, d_mc_slot_ids, d_mc_base_pos,
                            d_attn_out, mc_cur_B, mc_cur_K);
                    }
                    INK_CK(cudaEventRecord(c64_kv_pf.ev_comp[cur_b], st));
                } else {
                    if (use_mc_fp8rel) {
                        if (mc_cur_K <= 4) {
                            opt5060::k_flash_gqa_multichain_fp8rel<4><<<gmc, 128, 0, st>>>(
                                d_q, d_r, rp_T, d_sinks, d_gate_attn,
                                kv_desc[visit], d_mc_slot_ids, d_mc_base_pos, d_attn_out, mc_cur_B, mc_cur_K);
                        } else {
                            opt5060::k_flash_gqa_multichain_fp8rel<8><<<gmc, 128, 0, st>>>(
                                d_q, d_r, rp_T, d_sinks, d_gate_attn,
                                kv_desc[visit], d_mc_slot_ids, d_mc_base_pos, d_attn_out, mc_cur_B, mc_cur_K);
                        }
                    } else if (mc_cur_K == 3) {
                        k_ink_flash_gqa_multichain_k3<<<gmc, 96, 0, st>>>(
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            kv_desc[visit], d_mc_slot_ids, d_mc_base_pos,
                            d_attn_out, mc_cur_B);
                    } else if (mc_cur_K <= 4) {
                        k_ink_flash_gqa_multichain<4><<<gmc, 128, 0, st>>>(
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            kv_desc[visit], d_mc_slot_ids, d_mc_base_pos,
                            d_attn_out, mc_cur_B, mc_cur_K);
                    } else {
                        k_ink_flash_gqa_multichain<8><<<gmc, 128, 0, st>>>(
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            kv_desc[visit], d_mc_slot_ids, d_mc_base_pos,
                            d_attn_out, mc_cur_B, mc_cur_K);
                    }
                }
            } else if (M <= 16 || mode == 2) {
              if (t2_attn) {
                    elastic_kv.run_global_decode_attn_3tier(
                        visit, kv_desc[visit], slot0, (mode == 1) ? d_slots : nullptr,
                        d_pos, cur_max_pos, M, mode, p0_val,
                        d_q, d_rel_table, d_sinks, d_gate_attn,
                        (mode == 2) ? d_k : nullptr, (mode == 2) ? d_v : nullptr, (mode == 2) ? d_tree_anc : nullptr,
                        d_attn_pm, d_attn_pl, d_attn_po, d_attn_out, st);
              } else {
                // #6671: a verify tree uses the one-token step's split count, so row m is the value the step at that position gives (the key split, the
                // warp walk and the merge are the same; only the count differed: 8 for M > 1 against 16 / 32 for M == 1)
                const int n_sp = (M == 1 || mode == 2) ? (is_swa ? 16 : kInkAttnMaxSplits) : kInkAttnSplits;
                dim3 gsp(kInkKVHeads, n_sp, M);
                // fast only (STRATA_INK_SM120_TC_ATTN): tensor-core split-K for the global visits; falls back to the old kernels for K8V8 / K8V4 KV, SWA visits and prefill chunks
                bool tc_done = false;
                if (!is_swa && sm120_cfg(m.primary_gpu).tc_attn && (mode == 2 || (mode == 0 && M <= 16) || mode == 1)) {
                    const bool deep = (mode == 2 && d_tree_anc32 != nullptr);
                    tc_done = tcattn::launch(st, d_q, d_rel_table, d_sinks, kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                                             (mode == 2) ? d_k : nullptr, (mode == 2) ? d_v : nullptr,
                                             (mode == 2) ? (deep ? d_tree_anc32 : d_tree_anc) : nullptr, (mode == 2) ? d_tree_dep : nullptr,
                                             deep ? (int)kInkMaxTree : (int)kInkMaxTreeDepth, d_attn_pm, d_attn_pl, d_attn_po, M);
                }
                if (tc_done) {
                } else if (mode == 2 && d_tree_anc32 != nullptr) {
                    rtx5090::k_ink_flash_gqa_splitk_deep<<<gsp, 128, 0, st>>>(
                        d_q, d_rel_table, d_sinks,
                        kv_desc[visit], nullptr, slot0, d_pos,
                        d_k, d_v, d_tree_anc32, d_tree_dep,
                        p0_val, d_attn_pm, d_attn_pl, d_attn_po, M, n_sp);
                } else {
                    // opt-in (STRATA_INK_VERIFY_SHARED_KV, off by default; LCTX, f-lctx 1cb531beb): a global-visit verify step reads each KV cell once for a group of rows; bit-identical partials
                    if (!(mode == 2 && ink_launch_splitk_shared(st, d_q, d_rel_table, d_sinks, kv_desc[visit], slot0, d_pos, d_k, d_v,
                                                                d_tree_anc, d_attn_pm, d_attn_pl, d_attn_po, M, n_sp)))
                    k_ink_flash_gqa_splitk<<<gsp, 128, 0, st>>>(
                        d_q, d_rel_table, d_sinks,
                        kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                        (mode == 2) ? d_k : nullptr, (mode == 2) ? d_v : nullptr,
                        (mode == 2) ? d_tree_anc : nullptr, (mode == 2) ? d_tree_dep : nullptr,
                        p0_val, d_attn_pm, d_attn_pl, d_attn_po, M, n_sp);
                }
                if (tc_done) {
                    tcattn::merge(st, d_attn_pm, d_attn_pl, d_attn_po, d_gate_attn, d_attn_out, want_wo_h2 ? reinterpret_cast<__half2*>(d_x_bf16) : nullptr,
                                  d_pos, mode == 2, kv_desc[visit].cap_pos, M);
                    if (want_wo_h2) wo_h2_ready = true;
                } else if (want_wo_h2) {
                    rtx5090::k_ink_flash_gqa_merge_h2<<<gr, 128, 0, st>>>(
                        d_attn_pm, d_attn_pl, d_attn_po, d_gate_attn, d_attn_out,
                        reinterpret_cast<__half2*>(d_x_bf16), M, n_sp);
                    wo_h2_ready = true;
                } else {
                    k_ink_flash_gqa_merge<<<gr, 128, 0, st>>>(d_attn_pm, d_attn_pl, d_attn_po, d_gate_attn, d_attn_out, M, n_sp);
                }
              }
            } else if (use_rtx_pf_attn) {
                // Opt-in RTX 5090: Pipelined cp.async + ldmatrix.x4.trans FP16-TC FlashAttention-2 (BN=32 SWA / BN=64 Global, BF16-IO + fused amax!)
                float* am_ptr = (fp8_eng.cache_ready && fp8_eng.d_inp_amax != nullptr) ? (fp8_eng.d_inp_amax + 3) : nullptr;
                if (am_ptr) {
                    INK_CK(cudaMemsetAsync(am_ptr, 0, sizeof(float), st));
                    has_attn_amax = true;
                }
                const dim3 gpf(kInkHeads, (M + 63) / 64);
                if (qkv_bf16_io) {
                    if (is_swa) rtx5090::k_ink_flash_gqa_pf_sm120<32, true><<<gpf, 128, 0, st>>>(d_q, d_r, d_rtx_rel_proj_T[ip], d_sinks, d_gate_attn, kv_desc[visit], slot0, p0_val, M, d_attn_out, am_ptr);
                    else        rtx5090::k_ink_flash_gqa_pf_sm120<64, true><<<gpf, 128, 0, st>>>(d_q, d_r, d_rtx_rel_proj_T[ip], d_sinks, d_gate_attn, kv_desc[visit], slot0, p0_val, M, d_attn_out, am_ptr);
                } else {
                    if (is_swa) rtx5090::k_ink_flash_gqa_pf_sm120<32, false><<<gpf, 128, 0, st>>>(d_q, d_r, d_rtx_rel_proj_T[ip], d_sinks, d_gate_attn, kv_desc[visit], slot0, p0_val, M, d_attn_out, am_ptr);
                    else        rtx5090::k_ink_flash_gqa_pf_sm120<64, false><<<gpf, 128, 0, st>>>(d_q, d_r, d_rtx_rel_proj_T[ip], d_sinks, d_gate_attn, kv_desc[visit], slot0, p0_val, M, d_attn_out, am_ptr);
                }
            } else if (mode == 0 && M >= 64 && cur_cc >= 80 && m.all_sf4_vram && ink_rtx5060ti_cfg(m.primary_gpu).prefill_lm && !t2_attn) {
                // 5060 Ti family: fused hardware-FP8->FP16x2 register-Q Flash-Attention-2 (64x32 tile, no d_pf_s/d_pf_p DRAM traffic, no pack_kv)
                k_ink_flash_pf_mma<<<dim3((M + 63) / 64, kInkHeads), 128, 0, st>>>(
                    d_q, kv_desc[visit], d_rel_table, d_sinks, d_gate_attn,
                    slot0, p0_val, M, is_swa, d_attn_out);
            } else if (mode == 0 && (M >= 128 || t2_attn) && (cur_cc >= 70 || cur_cc == 60) && ensure_pf_attn(kInkPfKvTile)) {
                // O(1)-VRAM KV-Tiled Online-Softmax Tensor Core Prefill Attention (fixed <= ~114 MiB even at 256K+ ctx!)
                const bool use_fp16_pf = (cur_cc < 80);
                const cudaDataType_t pf_dt = use_fp16_pf ? CUDA_R_16F : CUDA_R_16BF;
                const cublasGemmAlgo_t pf_algo = (cur_cc >= 70) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
                const int k_start = is_swa ? std::max(0, p0_val - kInkSwaWin + 1) : 0;
                const int K_len = (p0_val + M) - k_start;
                if (use_fp16_pf) k_ink_pf_pack_q<true><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);
                else             k_ink_pf_pack_q<false><<<dim3(kInkHeads, M), 128, 0, st>>>(d_q, d_pf_q, M);

                constexpr int Q_CH = 256;
                const float sc_qk = 0.08838834764831845f;
                const float one = 1.0f, zero = 0.0f;

                if (K_len <= kInkPfKvTile && !(!is_swa && elastic_kv.tier2_active_for_pos(cur_max_pos))) {
                    if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_len), 128, 0, st>>>(kv_desc[visit], slot0, k_start, K_len, d_pf_k, d_pf_v);
                    else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_len), 128, 0, st>>>(kv_desc[visit], slot0, k_start, K_len, d_pf_k, d_pf_v);
                    for (int q0 = 0; q0 < M; q0 += Q_CH) {
                        const int C = std::min(Q_CH, M - q0);
                        const int k_sub_beg = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
                        const int k_sub_end = p0_val + q0 + C;
                        const int k_off = k_sub_beg - k_start;
                        const int K_sub = std::max(1, k_sub_end - k_sub_beg);
                        INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            K_sub, C, kInkHeadDim, &sc_qk,
                            d_pf_k + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                            d_pf_q + (size_t)q0 * kInkHeadDim, pf_dt, kInkHeadDim, (long long)M * kInkHeadDim,
                            &zero, d_pf_s, CUDA_R_32F, K_sub, (long long)C * K_sub,
                            kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                        if (use_fp16_pf) {
                            k_ink_pf_softmax_rel_sink<true><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                d_pf_s, d_rel_table, d_sinks, d_pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
                        } else {
                            k_ink_pf_softmax_rel_sink<false><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                d_pf_s, d_rel_table, d_sinks, d_pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
                        }
                        INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            kInkHeadDim, C, K_sub, &one,
                            d_pf_v + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                            d_pf_p, pf_dt, K_sub, (long long)C * K_sub,
                            &zero, d_pf_o, CUDA_R_32F, kInkHeadDim, (long long)C * kInkHeadDim,
                            kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                        k_ink_pf_unpack_o_gate<<<dim3(kInkHeads, C), 128, 0, st>>>(d_pf_o, d_gate_attn, d_attn_out, q0, C);
                    }
                } else {
                    // Long-context KV-Tiled Online-Softmax TC Attention (K_len > 2048 -> O(1) 114 MiB VRAM forever!)
                    float* d_alpha_tmp = d_ctrl_gain;  // [max_batch * 2304] >> 24 * 256 floats!
                    if (!is_swa && elastic_kv.enabled && elastic_kv.ensure_pf_multi_qc((M + Q_CH - 1) / Q_CH)) {
                        elastic_kv.run_global_prefill_attn_3tier(
                            visit, kv_desc[visit], slot0, p0_val, M,
                            use_fp16_pf, pf_dt, pf_algo, cublas, st,
                            d_q, d_rel_table, d_sinks, d_gate_attn,
                            d_pf_q, d_pf_k, d_pf_v, d_pf_s, d_pf_p, d_pf_ot,
                            d_alpha_tmp, d_attn_out);
                    } else {
                        for (int q0 = 0; q0 < M; q0 += Q_CH) {
                            const int C = std::min(Q_CH, M - q0);
                            const int k_sub_beg = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
                            const int k_sub_end = p0_val + q0 + C;
                            k_ink_pf_online_init<<<dim3(kInkHeads, C), 128, 0, st>>>(d_sinks, d_pf_m, d_pf_l, d_pf_o, q0, C);
                            for (int kt = k_sub_beg; kt < k_sub_end; kt += kInkPfKvTile) {
                                const int K_t = std::min(kInkPfKvTile, k_sub_end - kt);
                                const bool is_last = (kt + K_t >= k_sub_end);
                                if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(kv_desc[visit], slot0, kt, K_t, d_pf_k, d_pf_v);
                                else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_t), 128, 0, st>>>(kv_desc[visit], slot0, kt, K_t, d_pf_k, d_pf_v);
                                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    K_t, C, kInkHeadDim, &sc_qk,
                                    d_pf_k, pf_dt, kInkHeadDim, (long long)K_t * kInkHeadDim,
                                    d_pf_q + (size_t)q0 * kInkHeadDim, pf_dt, kInkHeadDim, (long long)M * kInkHeadDim,
                                    &zero, d_pf_s, CUDA_R_32F, K_t, (long long)C * K_t,
                                    kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                                if (use_fp16_pf) {
                                    k_ink_pf_softmax_tile_online<true><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                        d_pf_s, d_rel_table, d_pf_p, d_pf_m, d_pf_l, d_alpha_tmp, q0, C, kt, K_t, p0_val, is_swa);
                                } else {
                                    k_ink_pf_softmax_tile_online<false><<<dim3(C, kInkHeads), 256, 0, st>>>(
                                        d_pf_s, d_rel_table, d_pf_p, d_pf_m, d_pf_l, d_alpha_tmp, q0, C, kt, K_t, p0_val, is_swa);
                                }
                                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                                    cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    kInkHeadDim, C, K_t, &one,
                                    d_pf_v, pf_dt, kInkHeadDim, (long long)K_t * kInkHeadDim,
                                    d_pf_p, pf_dt, K_t, (long long)C * K_t,
                                    &zero, d_pf_ot, CUDA_R_32F, kInkHeadDim, (long long)C * kInkHeadDim,
                                    kInkHeads, CUBLAS_COMPUTE_32F, pf_algo));
                                k_ink_pf_accum_tile<<<dim3(kInkHeads, C), 128, 0, st>>>(
                                    d_pf_ot, d_alpha_tmp, d_pf_l, d_pf_o, d_gate_attn, d_attn_out, q0, C, is_last);
                            }
                        }
                    }
                }
            } else if (!is_swa && elastic_kv.tier2_active_for_pos(cur_max_pos)) {
                elastic_kv.run_global_decode_attn_3tier(
                    visit, kv_desc[visit], slot0, (mode == 1) ? d_slots : nullptr,
                    d_pos, cur_max_pos, M, mode, p0_val,
                    d_q, d_rel_table, d_sinks, d_gate_attn,
                    nullptr, nullptr, nullptr,
                    d_attn_pm, d_attn_pl, d_attn_po, d_attn_out, st);
            } else {
                dim3 g3(kInkKVHeads, M);
                k_ink_flash_gqa_gqa3<<<g3, 128, 0, st>>>(
                    d_q, d_rel_table, d_sinks, d_gate_attn,
                    kv_desc[visit], (mode == 1) ? d_slots : nullptr, slot0, d_pos,
                    d_attn_out, M);
            }
        }
        if (mode == 0 && M == 1 && !is_swa) {   // STRATA_INK_ATTN_PROBE (debug): read-only look at the softmax mass of this global visit; a no-op when the env var is unset
            auto& probe = InkAttnProbe::get();
            if (probe.on) probe.run(st, visit, kv_desc[visit], slot0, p0_val, d_q, d_rel_table, d_sinks, d_gate_attn, d_attn_out);
        }
        prof_mark_end(3); prof_mark_start();

        // 8. Output projection wo + FUSED post-attn ShortConv + residual add / block boundary!
        bool wo_fp8_done = false;
        bool wo_bf16_io  = false;
        if (fp8_sm.ready) {
            const auto* tr = fp8_sm.tr;
            if (M <= 16) {
                const size_t n_pairs = (size_t)M * (kInkQDim / 2);
                opt5060::k_f32_to_h2<<<(n_pairs + 255) / 256, 256, 0, st>>>(d_attn_out, fp8_sm.d_x_h2, n_pairs);
                opt5060::InkMultiProjFp8 mp{};
                mp.count = 1; mp.W[0] = tr->w_attn[ip][5]; mp.sc[0] = tr->sc_attn[ip][5];
                mp.Y[0] = d_wo_raw; mp.N[0] = kInkHidden; mp.cumN[0] = 0; mp.cumN[1] = kInkHidden;
                const int g8 = (kInkHidden + 7) / 8;
                if (M == 1)      opt5060::k_gemv_multi_fp8_m1_l1<<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, nullptr, nullptr, kInkQDim);
                else if (M <= 4) opt5060::k_gemv_multi_fp8_smallM_l1<4><<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, nullptr, nullptr, M, kInkQDim);
                else if (M <= 8) opt5060::k_gemv_multi_fp8_smallM_l1<8><<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, nullptr, nullptr, M, kInkQDim);
                else             opt5060::k_gemv_multi_fp8_smallM_l1<16><<<g8, 256, 0, st>>>(mp, fp8_sm.d_x_h2, nullptr, nullptr, M, kInkQDim);
            } else {
                fp8_sm.gemm_fp8_tn(tr->w_attn[ip][5], tr->sc_attn[ip][5], d_attn_out, d_wo_raw, M, kInkHidden, kInkQDim, false, st);
            }
            wo_fp8_done = true;
        } else if (fp8_eng.cache_ready && !m.ls_active && fp8_eng.w_attn[ip][5] != nullptr) {
            if (M <= 8 || !fp8_eng.lt_ok || M_pad32 > max_batch) {
                rtx5090::InkMultiProjFp8 mp{};
                mp.count = 1; mp.cumN[0] = 0; mp.cumN[1] = kInkHidden; mp.N[0] = kInkHidden;
                mp.W[0] = fp8_eng.w_attn[ip][5]; mp.sc[0] = fp8_eng.sc_attn[ip][5]; mp.X[0] = d_attn_out; mp.Y[0] = d_wo_raw;
                if (M == 1) {
                    __half2* xh0 = reinterpret_cast<__half2*>(d_x_bf16);
                    if (!wo_h2_ready) rtx5090::k_ink_f32_to_h2<<<(kInkQDim / 2 + 255) / 256, 256, 0, st>>>(d_attn_out, xh0, kInkQDim / 2);
                    rtx5090::k_ink_gemv_multi_fp8_m1_l1<<<(kInkHidden + 7) / 8, 256, 0, st>>>(mp, xh0, nullptr, nullptr, kInkQDim);
                } else if (M <= 8) {
                    const size_t n2 = (size_t)M * (kInkQDim / 2);
                    __half2* xh0 = reinterpret_cast<__half2*>(d_x_bf16);
                    if (!wo_h2_ready) rtx5090::k_ink_f32_to_h2<<<(n2 + 255) / 256, 256, 0, st>>>(d_attn_out, xh0, n2);
                    if (M <= 4) rtx5090::k_ink_gemv_multi_fp8_smallM_l1<4><<<(kInkHidden + 7) / 8, 256, 0, st>>>(mp, xh0, nullptr, nullptr, M, kInkQDim);
                    else        rtx5090::k_ink_gemv_multi_fp8_smallM_l1<8><<<(kInkHidden + 7) / 8, 256, 0, st>>>(mp, xh0, nullptr, nullptr, M, kInkQDim);
                } else {
                    rtx5090::k_ink_gemv_multi_fp8<8><<<(kInkHidden + 3) / 4, 64, 0, st>>>(mp, d_attn_out, M, kInkQDim);
                }
                wo_fp8_done = true;
            } else if (fp8_eng.lt_ok) {
                const float* wo_am = has_attn_amax ? (fp8_eng.d_inp_amax + 3) : nullptr;
                if (mode == 0 && M > 16 && M_pad32 <= max_batch && !dbg_hash_on) {
                    wo_bf16_io = fp8_eng.gemm_fp8_tn_bf16(fp8_eng.w_attn[ip][5], fp8_eng.sc_attn[ip][5],
                                                          d_attn_out, reinterpret_cast<__nv_bfloat16*>(d_wo_raw),
                                                          M, kInkHidden, kInkQDim, false, st, wo_am);
                    wo_fp8_done = wo_bf16_io;
                }
                if (!wo_fp8_done) {
                    wo_fp8_done = fp8_eng.gemm_fp8_tn(fp8_eng.w_attn[ip][5], fp8_eng.sc_attn[ip][5],
                                                      d_attn_out, d_wo_raw, M, kInkHidden, kInkQDim, false, st, wo_am);
                }
            }
        }
        if (!wo_fp8_done) linear(p + ".attn.wo_ud.weight", d_attn_out, d_wo_raw, M, 0, false, (mode == 2));
        prof_mark_end(4); prof_mark_start();
        if (visit == dbg_tap_visit) dbg_hash(209, d_wo_raw, M, kInkHidden);
        if (visit == dbg_tap_visit) { dbg_hash(210, d_sconv_pool[visit] + (size_t)(2 * kInkKVDim) * 3 * max_slots + (size_t)slot0 * 3 * kInkHidden, 1, 3 * kInkHidden); dbg_hash(211, d_cur, M, kInkHidden); }
        const bool is_blk_bnd = (ip % m.man.c.block_size == 0);
        if (is_blk_bnd) {
            if (pf_bf16_res) {
                k_ink_f32_to_bf16<<<((n_hid + 3) / 4 + 255) / 256, 256, 0, st>>>(d_cur, reinterpret_cast<__nv_bfloat16*>(d_blocks[n_blocks]), n_hid);
            } else {
                INK_CK(cudaMemcpyAsync(d_blocks[n_blocks], d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            }
            block_ptrs[n_blocks] = d_blocks[n_blocks];
            ++n_blocks;
        }
        if (wo_bf16_io) {
            float* hslot = d_sconv_pool[visit] + (size_t)(2 * kInkKVDim) * 3 * max_slots + (size_t)slot0 * 3 * kInkHidden;
            dim3 gp((kInkHidden + 255) / 256, (M + 15) / 16);
            rtx5090::k_ink_sconv4_seq_par_bf16_in<<<gp, 256, 0, st>>>(
                reinterpret_cast<const __nv_bfloat16*>(d_wo_raw), m.d_sconv_attn[ip], hslot, d_cur, M, kInkHidden, !is_blk_bnd);
            rtx5090::k_ink_sconv4_save_hist_bf16_in<<<(kInkHidden + 255) / 256, 256, 0, st>>>(
                reinterpret_cast<const __nv_bfloat16*>(d_wo_raw), hslot, M, kInkHidden);
        } else {
            run_sconv_uniform(d_wo_raw, m.d_sconv_attn[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim, 0, !is_blk_bnd);
        }

        dbg_hash(visit * 4 + 0, d_cur, M, kInkHidden);
        if (visit == dbg_tap_visit) { dbg_hash(193, d_q, M, kInkHeads * kInkHeadDim); dbg_hash(194, d_k, M, kInkKVDim); dbg_hash(195, d_v, M, kInkKVDim); dbg_hash(196, d_attn_out, M, kInkHeads * kInkHeadDim); dbg_hash(198, d_sinks, M, kInkHeads); }
        // 9. MoE Sublayer: Fused Pre-MLP Mix+CatRMS -> SITU MoE -> Fused Reduce8 + Post-MLP ShortConv4 + Add!
        const float* mlp_nw = (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp_norm.weight");
        float* moe_in_ptr = d_moe_in;
        {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            const float* q_mlp = (const float*)m.trunk_ptr_stage(cur_stage, "model.llm.attn_res.mlp_queries." + std::to_string(ip));
            if (has_msg) {
                if (fp8_sm.ready) {
                    const int ci = ip - m.man.c.core_start;
                    constexpr int D = kInkHidden;
                    if (M <= 16) {
                        const auto* tr = fp8_sm.tr;
                        k_ink_mlp_res_cat_rms<<<M, 256, 0, st>>>(stp, q_mlp, d_msgs + 3 * n_hid, d_moe_in, d_ctrl_cat, M, D);
                        if (M == 1) {
                            opt5060::k_ctrl_1x_down_fp8<1><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, tr->w_ctrl_dn[ci][3], tr->sc_ctrl_dn[ci][3], d_ctrl_gate, M);
                            opt5060::k_ctrl_1x_apply_fp8<1><<<D / 16, 256, 0, st>>>(d_moe_in, d_msgs + 3 * n_hid, d_ctrl_gate, tr->w_ctrl_gf[ci][3], tr->sc_ctrl_gf[ci][3], d_moe_in, M);
                        } else if (M <= 4) {
                            opt5060::k_ctrl_1x_down_fp8<4><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, tr->w_ctrl_dn[ci][3], tr->sc_ctrl_dn[ci][3], d_ctrl_gate, M);
                            opt5060::k_ctrl_1x_apply_fp8<4><<<D / 16, 256, 0, st>>>(d_moe_in, d_msgs + 3 * n_hid, d_ctrl_gate, tr->w_ctrl_gf[ci][3], tr->sc_ctrl_gf[ci][3], d_moe_in, M);
                        } else if (M <= 8) {
                            opt5060::k_ctrl_1x_down_fp8<8><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, tr->w_ctrl_dn[ci][3], tr->sc_ctrl_dn[ci][3], d_ctrl_gate, M);
                            opt5060::k_ctrl_1x_apply_fp8<8><<<D / 16, 256, 0, st>>>(d_moe_in, d_msgs + 3 * n_hid, d_ctrl_gate, tr->w_ctrl_gf[ci][3], tr->sc_ctrl_gf[ci][3], d_moe_in, M);
                        } else {
                            opt5060::k_ctrl_1x_down_fp8<16><<<kInkGateRank, 256, 0, st>>>(d_ctrl_cat, tr->w_ctrl_dn[ci][3], tr->sc_ctrl_dn[ci][3], d_ctrl_gate, M);
                            opt5060::k_ctrl_1x_apply_fp8<16><<<D / 16, 256, 0, st>>>(d_moe_in, d_msgs + 3 * n_hid, d_ctrl_gate, tr->w_ctrl_gf[ci][3], tr->sc_ctrl_gf[ci][3], d_moe_in, M);
                        }
                        k_ink_rms_norm<<<M, 256, 0, st>>>(d_moe_in, mlp_nw, d_moe_in, nullptr, M, D);
                    } else {
                        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
                        opt5060::k_mlp_res_to_fp8<<<M_pad, 256, 0, st>>>(stp, q_mlp, d_msgs + 3 * n_hid, d_moe_in, fp8_sm.d_x_fp8, M, M_pad, D);
                        fp8_sm.run_ctrl_fp8(ci, 3, d_moe_in, d_msgs + 3 * n_hid, mlp_nw, d_moe_in, d_ctrl_gate, M, true, st);
                    }
                } else if (m.tinfo(p + ".controllers.mlp.down_weight").fmt == FMT_Q8ROW) {
                    if (M <= 16) {
                        k_ink_mlp_res_cat_rms<<<M, 256, 0, st>>>(stp, q_mlp, d_msgs + 3 * n_hid, d_moe_in, d_ctrl_cat, M, kInkHidden);
                        run_loop_control_q8_norm(
                            d_moe_in, d_msgs + 3 * n_hid,
                            m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.down_weight"),
                            m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.gain_weight"),
                            m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.forget_weight"),
                            mlp_nw, d_moe_in, M, /*cat_ready=*/true);
                    } else {
                        k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_mlp, d_moe_in, 1, M, kInkHidden);
                        run_loop_control_q8_norm(
                            d_moe_in, d_msgs + 3 * n_hid,
                            m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.down_weight"),
                            m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.gain_weight"),
                            m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.forget_weight"),
                            mlp_nw, d_moe_in, M, /*cat_ready=*/false);
                    }
                } else {
                const int ci = ip - m.man.c.core_start;
                const bool has_bc = (!m.ls_active && ci >= 0 && ci < 12 && d_rtx_ctrl_gf[ci][3] != nullptr);
                const int M_pad_m = (M <= 32) ? 32 : ((M + 31) & ~31);
                const bool use_fp8_m = (M > 16 && !m.ls_active && ci >= 0 && ci < 12 &&
                                        fp8_eng.cache_ready && fp8_eng.lt_ok && fp8_eng.w_ctrl_dn[ci][3] != nullptr &&
                                        M_pad_m <= max_batch && M_pad_m <= fp8_eng.max_m);
                const size_t msg_stride = (pf_bf16_res && use_fp8_m) ? (n_hid / 2) : n_hid;
                if (use_fp8_m) {
                    if (pf_bf16_res) {
                        rtx5090::k_ink_mlp_res_to_fp8<true><<<M_pad_m, 256, 0, st>>>(
                            stp, q_mlp, d_msgs + 3 * msg_stride, d_moe_in, fp8_eng.d_x_fp8, M, M_pad_m, kInkHidden);
                        moe_in_ptr = d_inp0;
                    } else {
                        rtx5090::k_ink_mlp_res_to_fp8<false><<<M_pad_m, 256, 0, st>>>(
                            stp, q_mlp, d_msgs + 3 * msg_stride, d_moe_in, fp8_eng.d_x_fp8, M, M_pad_m, kInkHidden);
                    }
                } else {
                    k_ink_mlp_res_cat_rms<<<M, 256, 0, st>>>(stp, q_mlp, d_msgs + 3 * n_hid, d_moe_in, d_ctrl_cat, M, kInkHidden);
                }
                run_loop_control_norm(
                    d_moe_in, d_msgs + 3 * msg_stride,
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.down_weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.gain_weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".controllers.mlp.forget_weight"),
                    mlp_nw, moe_in_ptr, M, /*cat_ready=*/true, exact_x,
                    has_bc ? d_rtx_ctrl_gf[ci][3] : nullptr, ci, 3, pf_bf16_res && use_fp8_m);
                }
            } else {
                if (pf_bf16_res || (M > 24 && fp8_eng.cache_ready && !m.ls_active && stp.count <= 6)) {
                    if (pf_bf16_res) rtx5090::k_ink_attn_res_mix_1pass<true><<<M, 256, 0, st>>>(stp, q_mlp, d_moe_in, 1, M, kInkHidden, mlp_nw);
                    else             rtx5090::k_ink_attn_res_mix_1pass<false><<<M, 256, 0, st>>>(stp, q_mlp, d_moe_in, 1, M, kInkHidden, mlp_nw);
                } else {
                    k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, q_mlp, d_moe_in, 1, M, kInkHidden, mlp_nw);
                }
            }
        }
        prof_mark_end(0); prof_mark_start();
        dbg_hash(visit * 4 + 1, moe_in_ptr, M, kInkHidden);

        if (lm_stop_at_router) {
            launch_ink_moe_router(
                st, cublas,
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                d_moe_in, M, *cur_moe_ws, m.man.c.route_scale, false);
            return;
        }

        if (m.ep2_active && m.layer_has_peer[ip]) {
            launch_ink_moe_router(
                st, cublas,
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                d_moe_in, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2));
            const size_t x_bytes   = (size_t)M * kInkHidden * sizeof(float);
            const size_t idx_bytes = (size_t)M * kInkTopK * sizeof(int32_t);
            const size_t w_bytes   = (size_t)M * kInkTopK * sizeof(float);
            if (m.ep2_p2p) {
                INK_CK(cudaEventRecord(ev_moe_ready, st));
                INK_CK(cudaSetDevice(m.peer_gpu));
                INK_CK(cudaStreamWaitEvent(peer_st, ev_moe_ready, 0));
                INK_CK(cudaMemcpyPeerAsync(d_peer_x, m.peer_gpu, d_moe_in, m.primary_gpu, x_bytes, peer_st));
                INK_CK(cudaMemcpyPeerAsync(peer_ws.topk_idx, m.peer_gpu, moe_ws.topk_idx, m.primary_gpu, idx_bytes, peer_st));
                INK_CK(cudaMemcpyPeerAsync(peer_ws.topk_w, m.peer_gpu, moe_ws.topk_w, m.primary_gpu, w_bytes, peer_st));
                launch_ink_moe_dispatch_only(peer_st, m.d_sec4_peer, m.d_experts_peer[ip], m.moe_info_peer[ip],
                                             d_peer_x, M, peer_ws, (mode == 2), false);
                k_ink_moe_reduce8_tier0<<<dim3((kInkHidden + 255) / 256, M), 256, 0, peer_st>>>(
                    m.d_experts_peer[ip], peer_ws.topk_idx, peer_ws.y_pair, d_peer_y, M);
                INK_CK(cudaMemcpyPeerAsync(d_peer_y_rx, m.primary_gpu, d_peer_y, m.peer_gpu, x_bytes, peer_st));
                INK_CK(cudaEventRecord(ev_peer_done, peer_st));
                INK_CK(cudaSetDevice(m.primary_gpu));
            } else {
                INK_CK(cudaMemcpyAsync(h_ep_x, d_moe_in, x_bytes, cudaMemcpyDeviceToHost, st));
                INK_CK(cudaMemcpyAsync(h_ep_idx, moe_ws.topk_idx, idx_bytes, cudaMemcpyDeviceToHost, st));
                INK_CK(cudaMemcpyAsync(h_ep_w, moe_ws.topk_w, w_bytes, cudaMemcpyDeviceToHost, st));
                INK_CK(cudaEventRecord(ev_moe_ready, st));
                INK_CK(cudaSetDevice(m.peer_gpu));
                INK_CK(cudaStreamWaitEvent(peer_st, ev_moe_ready, 0));
                INK_CK(cudaMemcpyAsync(d_peer_x, h_ep_x, x_bytes, cudaMemcpyHostToDevice, peer_st));
                INK_CK(cudaMemcpyAsync(peer_ws.topk_idx, h_ep_idx, idx_bytes, cudaMemcpyHostToDevice, peer_st));
                INK_CK(cudaMemcpyAsync(peer_ws.topk_w, h_ep_w, w_bytes, cudaMemcpyHostToDevice, peer_st));
                launch_ink_moe_dispatch_only(peer_st, m.d_sec4_peer, m.d_experts_peer[ip], m.moe_info_peer[ip],
                                             d_peer_x, M, peer_ws, (mode == 2), false);
                k_ink_moe_reduce8_tier0<<<dim3((kInkHidden + 255) / 256, M), 256, 0, peer_st>>>(
                    m.d_experts_peer[ip], peer_ws.topk_idx, peer_ws.y_pair, d_peer_y, M);
                INK_CK(cudaMemcpyAsync(h_ep_y, d_peer_y, x_bytes, cudaMemcpyDeviceToHost, peer_st));
                INK_CK(cudaEventRecord(ev_peer_done, peer_st));
                INK_CK(cudaSetDevice(m.primary_gpu));
            }
            launch_ink_moe_dispatch_only(st, m.d_sec4, m.d_experts[ip], m.moe_info[ip],
                                         d_moe_in, M, moe_ws, (mode == 2), false);
            k_ink_moe_reduce8_tier0<<<dim3((kInkHidden + 255) / 256, M), 256, 0, st>>>(
                m.d_experts[ip], moe_ws.topk_idx, moe_ws.y_pair, d_moe_raw, M);
            INK_CK(cudaStreamWaitEvent(st, ev_peer_done, 0));
            if (!m.ep2_p2p) {
                INK_CK(cudaMemcpyAsync(d_peer_y_rx, h_ep_y, x_bytes, cudaMemcpyHostToDevice, st));
            }
            const size_t n_add = (size_t)M * kInkHidden;
            k_ink_add_f32<<<(n_add + 255) / 256, 256, 0, st>>>(d_moe_raw, d_peer_y_rx, n_add);
            prof_mark_end(5); prof_mark_start();
            run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
        } else if (m.layer_has_host[ip] && m.hybrid.active && cur_stage == 0) {
            launch_ink_moe_hybrid(
                st, cublas, m.d_sec4, m.d_experts[ip], m.moe_info[ip], ip, m.hybrid,
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                d_moe_in, d_moe_raw, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2));
            prof_mark_end(5); prof_mark_start();
            run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
        } else {
            const bool rtx_moe = rtx5090::moe_opt_on() && !m.ls_active;
            const bool atomic_pf_moe = rtx_moe && (mode == 0 && M >= 1024 && !rtx5090::exact_mode_on() && !m.moe_info[ip].has_sk && !m.moe_info[ip].has_gguf);
            const bool fuse_pf_reduce = rtx_moe && (mode == 0 && M > 16) && !atomic_pf_moe;
            const bool fuse_moe_sconv = ((mode == 0 && M == 1) || mode == 1 || mode == 2 || (mode == 3 && d_bspec_sconv == nullptr) || fuse_pf_reduce);
            if (m.all_sf4_vram && fp8_sm.ready && cur_stage == 0 && M <= 3) {
                launch_ink_moe_router(
                    st, cublas,
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                    d_moe_in, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2));
                const size_t x_pairs = (size_t)M * (kInkHidden / 2);
                opt5060::k_f32_to_h2<<<(x_pairs + 255) / 256, 256, 0, st>>>(d_moe_in, fp8_sm.d_x_h2, x_pairs);
                if (M == 1) {
                    opt5060::k_moe_gu_m1_l1<<<dim3(kInkInter / 16, kInkTopK), 256, 0, st>>>(
                        m.d_sec4, m.d_experts[ip], cur_moe_ws->topk_idx, fp8_sm.d_x_h2, fp8_sm.d_h_h2);
                    opt5060::k_moe_dn_m1_l1<<<dim3((kInkHidden / 2) / 16, kInkTopK), 256, 0, st>>>(
                        m.d_sec4, m.d_experts[ip], cur_moe_ws->topk_idx, cur_moe_ws->topk_w, fp8_sm.d_h_h2, cur_moe_ws->y_pair);
                } else {
                    k_ink_moe_bin<<<1, 128, 0, st>>>(m.d_experts[ip], cur_moe_ws->topk_idx, M, 8, cur_moe_ws->tb);
                    const int max_act = std::min(kInkExperts, M * kInkTopK);
                    opt5060::k_moe_sf4_grouped_reg_gu<<<dim3(kInkInter / 16, max_act), 256, 0, st>>>(
                        m.d_sec4, m.d_experts[ip], cur_moe_ws->tb, fp8_sm.d_x_h2, fp8_sm.d_h_h2);
                    opt5060::k_moe_sf4_grouped_reg_dn<<<dim3((kInkHidden / 2) / 16, max_act), 256, 0, st>>>(
                        m.d_sec4, m.d_experts[ip], cur_moe_ws->tb, fp8_sm.d_h_h2, cur_moe_ws->topk_w, cur_moe_ws->y_pair);
                }
                if (!fuse_moe_sconv) {
                    k_ink_moe_reduce8<<<dim3((kInkHidden + 255) / 256, M), 256, 0, st>>>(cur_moe_ws->y_pair, d_moe_raw, M);
                }
            } else if (rtx_moe) {
                rtx5090::launch_ink_moe_rtx5090(
                    st, cublas, m.sec4_stage(cur_stage), m.d_experts[ip], m.moe_info[ip],
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                    moe_in_ptr, d_moe_raw, M, *cur_moe_ws,
                    d_ctrl_gain, reinterpret_cast<__half2*>(d_x_bf16),
                    m.man.c.route_scale, (mode == 2), !fuse_moe_sconv && !atomic_pf_moe, atomic_pf_moe,
                    fp8_eng.cache_ready ? fp8_eng.w_router_h2[ip] : nullptr);
            } else {
                launch_ink_moe(
                    st, cublas, m.sec4_stage(cur_stage), m.d_experts[ip], m.moe_info[ip],
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.weight"),
                    (const float*)m.trunk_ptr_stage(cur_stage, p + ".mlp.gate.bias"),
                    moe_in_ptr, d_moe_raw, M, *cur_moe_ws, m.man.c.route_scale, (mode == 2), !fuse_moe_sconv);
            }
            prof_mark_end(5); prof_mark_start();
            if (visit == dbg_tap_visit) { dbg_hash(204, cur_moe_ws->y_pair, M, kInkTopK * kInkHidden); dbg_hash(205, cur_moe_ws->topk_w, M, kInkTopK); dbg_hash(206, reinterpret_cast<const float*>(cur_moe_ws->topk_idx), M, kInkTopK); }
            if (fuse_moe_sconv) {
                constexpr int mlp_ch_off = 2 * kInkKVDim + kInkHidden;
                float* hbase = d_sconv_pool[visit] + (size_t)mlp_ch_off * 3 * max_slots;
                if (visit == dbg_tap_visit) { dbg_hash(207, hbase + (size_t)slot0 * 3 * kInkHidden, 1, 3 * kInkHidden); dbg_hash(208, d_cur, M, kInkHidden); }
                if (fuse_pf_reduce) {
                    float* hslot = hbase + (size_t)slot0 * 3 * kInkHidden;
                    dim3 gp((kInkHidden + 255) / 256, (M + 15) / 16);
                    rtx5090::k_ink_moe_reduce8_sconv_seq_par<<<gp, 256, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hslot, d_cur, M);
                    rtx5090::k_ink_moe_reduce8_save_hist<<<(kInkHidden + 255) / 256, 256, 0, st>>>(
                        cur_moe_ws->y_pair, hslot, M);
                } else if (mode == 0 && M == 1) {
                    k_ink_moe_reduce8_sconv_m1<<<(kInkHidden + 255) / 256, 256, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase + (size_t)slot0 * 3 * kInkHidden, d_cur);
                } else if (mode == 1) {
                    k_ink_moe_reduce8_sconv_batched<<<dim3((kInkHidden + 255) / 256, M), 256, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase, d_slots, d_cur, M);
                } else if (mode == 3) {
                    uint16_t* stg = d_mc_sconv_stage ? (d_mc_sconv_stage + (size_t)visit * mc_cur_B * mc_cur_K * kInkSconvCh) : nullptr;
                    k_ink_moe_reduce8_sconv_multichain<<<dim3((kInkHidden + 255) / 256, mc_cur_B), 256, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase, d_mc_slot_ids, d_cur, stg, mlp_ch_off, mc_cur_B, mc_cur_K);
                } else {
                    k_ink_moe_reduce8_sconv_tree<<<(kInkHidden + 127) / 128, 128, 0, st>>>(
                        cur_moe_ws->y_pair, m.d_sconv_mlp[ip], hbase + (size_t)slot0 * 3 * kInkHidden,
                        d_sconv_lag, d_stage_sconv + (size_t)visit * tree_cap * kInkSconvCh,
                        mlp_ch_off, d_cur, M);
                }
            } else {
                run_sconv_uniform(d_moe_raw, m.d_sconv_mlp[ip], nullptr, d_cur, kInkHidden, 2 * kInkKVDim + kInkHidden, 0, true);
            }
        }
        prof_mark_end(2);
        dbg_hash(visit * 4 + 3, cur_moe_ws->r_logits, M, kInkExperts);
        dbg_hash(visit * 4 + 2, d_cur, M, kInkHidden);
    }

    // ---- Full 48-Visit Body (Capturable inside a SINGLE CUDA Graph on single GPU, or Pipeline Layer-Split on 2..4 GPUs!) ----
    void emit_48_visits(int M, int mode, int slot0, int tree_p0, bool want_taps) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        const size_t n_hid = (size_t)M * kInkHidden;
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<M, 256, 0, st>>>(
            m.embed_ptr(), ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            d_tok_ids, has_img_step ? d_is_img : nullptr, has_img_step ? d_img_emb : nullptr, d_cur, M, kInkHidden);

        const float* block_ptrs[8] = {};
        int n_blocks = 0;
        int visit = 0;

        auto maybe_tap = [&](int v, const float* src) {
            if (!want_taps) return;
            float* dst_tap = nullptr;
            if (v == 5)       dst_tap = d_taps[0];
            else if (v == 23) dst_tap = d_taps[1];
            else if (v == 41) dst_tap = d_taps[2];
            if (!dst_tap) return;
            if (m.ls_active && cur_stage != 0) {
                copy_between_stages(cur_stage, 0, src, dst_tap, n_hid * 4);
                INK_CK(cudaSetDevice(m.stage_gpu[cur_stage]));
            } else {
                INK_CK(cudaMemcpyAsync(dst_tap, src, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            }
        };

        const bool pf_bf16_res = (mode == 0 && M >= 1024 && fp8_eng.cache_ready && fp8_eng.lt_ok &&
                                  !m.ls_active && !rtx5090::exact_mode_on() && !dbg_hash_on &&
                                  (((M + 31) & ~31) <= max_batch) && (((M + 31) & ~31) <= fp8_eng.max_m));
        const size_t blk_bytes = pf_bf16_res ? (n_hid * 2) : (n_hid * 4);

        // 1. Prelude: physical layers 0..5 (visits 0..5)
        for (int ip = 0; ip < m.man.c.core_start; ++ip) {
            if (m.ls_active && m.layer_owner[ip] != cur_stage) {
                switch_stage_handoff(m.layer_owner[ip], M, mode, n_blocks, block_ptrs, nullptr, 0);
            }
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs, pf_bf16_res);
            ++visit;
        }

        // Save prelude state & prelude_blocks on Stage 0
        if (m.ls_active && cur_stage != 0) {
            switch_stage_handoff(0, M, mode, n_blocks, block_ptrs, nullptr, 0);
        }
        INK_CK(cudaMemcpyAsync(d_prelude, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        const int n_pre_blk = n_blocks;
        for (int b = 0; b < n_pre_blk; ++b) {
            INK_CK(cudaMemcpyAsync(d_prelude_blocks[b], block_ptrs[b], blk_bytes, cudaMemcpyDeviceToDevice, st));
        }

        // 2. Looped Core: physical layers 6..17 x 3 passes (visits 6..41)
        const int n_passes = (max_passes_override > 0 && max_passes_override < m.man.c.num_passes)
                                 ? max_passes_override
                                 : m.man.c.num_passes;
        int n_bank = 0;
        for (int pass = 0; pass < n_passes; ++pass) {
            if (pass > 0) {
                if (m.ls_active && cur_stage != 0) {
                    switch_stage_handoff(0, M, mode, n_blocks, block_ptrs, nullptr, 0);
                }
                n_bank = n_blocks - (n_pre_blk + 1);
                for (int b = 0; b < n_bank; ++b) {
                    INK_CK(cudaMemcpyAsync(d_bank[b], block_ptrs[n_pre_blk + 1 + b], blk_bytes, cudaMemcpyDeviceToDevice, st));
                }
                if (pf_bf16_res) {
                    k_ink_f32_to_bf16<<<((n_hid + 3) / 4 + 255) / 256, 256, 0, st>>>(d_cur, reinterpret_cast<__nv_bfloat16*>(d_bank_out), n_hid);
                } else {
                    INK_CK(cudaMemcpyAsync(d_bank_out, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                }
                INK_CK(cudaMemcpyAsync(d_cur, d_prelude, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                n_blocks = n_pre_blk;
                for (int b = 0; b < n_pre_blk; ++b) {
                    INK_CK(cudaMemcpyAsync(d_blocks[b], d_prelude_blocks[b], blk_bytes, cudaMemcpyDeviceToDevice, st));
                    block_ptrs[b] = d_blocks[b];
                }
            }
            DepthStatePtrs bst{};
            if (pass > 0) {
                for (int b = 0; b < n_bank; ++b) bst.p[b] = d_bank[b];
                bst.p[n_bank] = d_bank_out;
                bst.count = n_bank + 1;
            }
            for (int ip = m.man.c.core_start; ip < m.man.c.core_end; ++ip) {
                if (m.ls_active && m.layer_owner[ip] != cur_stage) {
                    switch_stage_handoff(m.layer_owner[ip], M, mode, n_blocks, block_ptrs, (pass > 0) ? &bst : nullptr, n_bank);
                }
                maybe_tap(visit, d_cur);
                run_visit(visit, ip, (pass > 0) ? &bst : nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs, pf_bf16_res);
                ++visit;
            }
        }

        // 3. Coda: physical layers 18..23 (visits 42..47)
        visit = 42;
        for (int ip = m.man.c.core_end; ip < m.man.c.n_phys; ++ip) {
            if (m.ls_active && m.layer_owner[ip] != cur_stage) {
                switch_stage_handoff(m.layer_owner[ip], M, mode, n_blocks, block_ptrs, nullptr, 0);
            }
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs, pf_bf16_res);
            ++visit;
        }

        // Return to Stage 0 for Output Depth Mix + Output RMSNorm -> d_h_norm (and tap 48!)
        if (m.ls_active && cur_stage != 0) {
            switch_stage_handoff(0, M, mode, n_blocks, block_ptrs, nullptr, 0);
        }
        {
            DepthStatePtrs stp{};
            for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
            stp.p[n_blocks] = d_cur;
            stp.count = n_blocks + 1;
            if (pf_bf16_res) {
                rtx5090::k_ink_attn_res_mix_1pass<true><<<M, 256, 0, st>>>(
                    stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, M, kInkHidden);
            } else {
                k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, M, kInkHidden);
            }
            k_ink_rms_norm<<<M, 256, 0, st>>>(d_moe_in, (const float*)m.trunk_ptr("model.llm.norm.weight"), d_h_norm, nullptr, M, kInkHidden);
            if (want_taps) INK_CK(cudaMemcpyAsync(d_taps[3], d_h_norm, n_hid * 4, cudaMemcpyDeviceToDevice, st));
            dbg_hash(192, d_h_norm, M, kInkHidden);
        }
    }

    // ---- RTX 5060 Ti Opt-In Helpers: Prefix+Core CUDA Graph, Layer-Major Coda Prefill, and Mode-3 Batched Speculative Verify ----
    int first_host_coda_layer() const noexcept {
        for (int il = 0; il < m.man.c.n_phys; ++il) {
            if (m.layer_has_host[il]) return il;
        }
        return m.man.c.n_phys;
    }

    bool can_layer_major_prefill() const noexcept {
        if (!ink_rtx5060ti_cfg(m.primary_gpu).prefill_lm || !sm120::available()) return false;
        if (m.ls_active || m.ep2_active || !m.hybrid.active || m.hybrid.ssd_only || m.hybrid.tier_on) return false;
        if (max_passes_override != 0 || m.n_host_experts == 0) return false;
        const int fhc = first_host_coda_layer();
        return (fhc >= m.man.c.core_end && fhc < m.man.c.n_phys);
    }

    // Emit Visits 0 .. (42 + stop_coda_ip - core_end - 1) where ALL layers < stop_coda_ip are 100% VRAM-resident!
    void emit_visits_prefix_to(int stop_coda_ip, int M, int mode, int slot0, int tree_p0, bool want_taps) {
        const size_t n_hid = (size_t)M * kInkHidden;
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<M, 256, 0, st>>>(
            m.embed_ptr(), ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            d_tok_ids, has_img_step ? d_is_img : nullptr, has_img_step ? d_img_emb : nullptr, d_cur, M, kInkHidden);

        const float* block_ptrs[8] = {};
        int n_blocks = 0;
        int visit = 0;
        auto maybe_tap = [&](int v, const float* src) {
            if (!want_taps) return;
            float* dst_tap = nullptr;
            if (v == 5)       dst_tap = d_taps[0];
            else if (v == 23) dst_tap = d_taps[1];
            else if (v == 41) dst_tap = d_taps[2];
            if (dst_tap) INK_CK(cudaMemcpyAsync(dst_tap, src, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        };

        for (int ip = 0; ip < m.man.c.core_start; ++ip) {
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }
        INK_CK(cudaMemcpyAsync(d_prelude, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
        const int n_pre_blk = n_blocks;
        for (int b = 0; b < n_pre_blk; ++b) {
            INK_CK(cudaMemcpyAsync(d_prelude_blocks[b], block_ptrs[b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
        }

        const int n_passes = m.man.c.num_passes;
        int n_bank = 0;
        for (int pass = 0; pass < n_passes; ++pass) {
            if (pass > 0) {
                n_bank = n_blocks - (n_pre_blk + 1);
                for (int b = 0; b < n_bank; ++b) {
                    INK_CK(cudaMemcpyAsync(d_bank[b], block_ptrs[n_pre_blk + 1 + b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
                }
                INK_CK(cudaMemcpyAsync(d_bank_out, d_cur, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                INK_CK(cudaMemcpyAsync(d_cur, d_prelude, n_hid * 4, cudaMemcpyDeviceToDevice, st));
                n_blocks = n_pre_blk;
                for (int b = 0; b < n_pre_blk; ++b) {
                    INK_CK(cudaMemcpyAsync(d_blocks[b], d_prelude_blocks[b], n_hid * 4, cudaMemcpyDeviceToDevice, st));
                    block_ptrs[b] = d_blocks[b];
                }
            }
            DepthStatePtrs bst{};
            if (pass > 0) {
                for (int b = 0; b < n_bank; ++b) bst.p[b] = d_bank[b];
                bst.p[n_bank] = d_bank_out;
                bst.count = n_bank + 1;
            }
            for (int ip = m.man.c.core_start; ip < m.man.c.core_end; ++ip) {
                maybe_tap(visit, d_cur);
                run_visit(visit, ip, (pass > 0) ? &bst : nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
                ++visit;
            }
        }
        visit = 42;
        for (int ip = m.man.c.core_end; ip < stop_coda_ip; ++ip) {
            maybe_tap(visit, d_cur);
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }
    }

    // Emit remaining Coda layers start_coda_ip..23 + Output Depth Mix + Output RMSNorm (after emit_visits_prefix_to!)
    void emit_visits_coda_from(int start_coda_ip, int M, int mode, int slot0, int tree_p0, bool want_taps) {
        const size_t n_hid = (size_t)M * kInkHidden;
        int n_blocks = (start_coda_ip + m.man.c.block_size - 1) / m.man.c.block_size;
        const float* block_ptrs[8] = {};
        for (int b = 0; b < n_blocks; ++b) block_ptrs[b] = d_blocks[b];
        int visit = 42 + (start_coda_ip - m.man.c.core_end);
        for (int ip = start_coda_ip; ip < m.man.c.n_phys; ++ip) {
            run_visit(visit, ip, nullptr, M, mode, slot0, tree_p0, n_blocks, block_ptrs);
            ++visit;
        }
        DepthStatePtrs stp{};
        for (int b = 0; b < n_blocks; ++b) stp.p[b] = block_ptrs[b];
        stp.p[n_blocks] = d_cur;
        stp.count = n_blocks + 1;
        k_ink_attn_res_mix<<<M, 256, 0, st>>>(stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, M, kInkHidden);
        k_ink_rms_norm<<<M, 256, 0, st>>>(d_moe_in, (const float*)m.trunk_ptr("model.llm.norm.weight"), d_h_norm, nullptr, M, kInkHidden);
        if (want_taps) INK_CK(cudaMemcpyAsync(d_taps[3], d_h_norm, n_hid * 4, cudaMemcpyDeviceToDevice, st));
    }

    // MoE variant for the macro-prefill window (5060 Ti family): the tile size is the macro path's own rule (32 / 64 / 128 by window size). Direct-MMA kernels
    // (per-row scales) when the table asks for them, the base kernels otherwise (the exact form).
    Sm120MoeSel macro_moe_sel(int tile_rows) const {
        using strata::core::Sm120Tc;
        const auto& c = ink_rtx5060ti_cfg(m.primary_gpu);
        Sm120MoeSel s;
        s.active = true;
        s.tile_rows = tile_rows;
        s.sf4 = c.sm120_tc_smallm ? (tile_rows >= 128 ? Sm120Tc::Tr128Bk128 : Sm120Tc::Direct)
                                  : (tile_rows >= 128 ? Sm120Tc::Tr128Small : Sm120Tc::Base);
        s.mx8 = Sm120Tc::Base;
        s.pf_gu = -1;
        s.pf_dn = c.sm120_pf ? 128 : 0;
        s.row_quant = sm120::sf4_variant_row_quant((int)s.sf4);
        return s;
    }

    bool can_allvram_macro_prefill() const noexcept {
        if (!ink_rtx5060ti_cfg(m.primary_gpu).prefill_lm || !sm120::available()) return false;
        if (!m.all_sf4_vram || m.ls_active || m.ep2_active || max_passes_override != 0 || m.n_host_experts != 0) return false;
        return (d_pf_slab != nullptr && d_pf_slab_bytes >= (72ull << 20));
    }

    // Visit-Major All-VRAM Macro-Prefill (W = 2048..4096 tokens per MoE weight sweep -> >15,000-20,000 ppts on 8K+!):
    // Runs Attn+ShortConv+Controllers+Router in micro-chunks (mc <= max_batch) reusing d_pf_slab for Flash-GQA,
    // then executes ONE single W-token sm_120 NVFP4 MoE pass reusing d_pf_slab for d_y_accum + d_ha_fp4 + tb!
    template <typename TapFn>
    bool forward_seq_allvram_macro(const int32_t* tok_ids, int p0, int N, int slot0, bool want_taps, int& out_last_mc, TapFn&& on_taps_chunk) {
        if (!can_allvram_macro_prefill() || N < 1024) return false;
        // Reclaim cached small-M decode CUDA graphs & temporary head buffers before sizing W=4224 macro buffer!
        if (!graph_cache.empty() || d_q8_tile_i32 || d_logits) {
            INK_CK(cudaStreamSynchronize(st));
            for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
            graph_cache.clear();
            if (d_q8_tile_i32) { cudaFree(d_q8_tile_i32); d_q8_tile_i32 = nullptr; q8_argmax_cap_M = 0; q8_argmax_vt_nom = 0; }
            if (d_q8_tile_val) { cudaFree(d_q8_tile_val); d_q8_tile_val = nullptr; }
            if (d_q8_tile_id)  { cudaFree(d_q8_tile_id);  d_q8_tile_id  = nullptr; }
            if (d_logits)      { cudaFree(d_logits);      d_logits      = nullptr; logits_cap = 0; }
        }
        if (!opt5060_macro.ensure(std::min(N, 4224), m.primary_gpu)) return false;

        constexpr int D = kInkHidden;
        auto arch_for_block = [](int b) noexcept -> int { return (b < 6) ? b : (b + 1); };  // 0..5 -> 0..5, 6..7 -> 7..8
        auto pack_cur = [&](int off, int mc, const float* src) {
            k_ink_f32_to_bf16_rows<<<((size_t)mc * D + 255) / 256, 256, 0, st>>>(src, opt5060_macro.d_cur + (size_t)off * D, mc, D);
        };
        auto unpack_cur = [&](int off, int mc, float* dst) {
            k_ink_bf16_to_f32_rows<<<((size_t)mc * D + 255) / 256, 256, 0, st>>>(opt5060_macro.d_cur + (size_t)off * D, dst, mc, D);
        };
        auto save_arch = [&](int sl, int off, int mc, const float* src) {
            if (opt5060_macro.use_q8) {
                k_ink_f32_to_q8row_rows<<<mc, 256, 0, st>>>(
                    src, reinterpret_cast<int8_t*>(opt5060_macro.d_arch_q[sl]) + (size_t)off * D,
                    opt5060_macro.d_arch_sc[sl] + off, mc, D);
            } else {
                k_ink_f32_to_bf16_rows<<<((size_t)mc * D + 255) / 256, 256, 0, st>>>(
                    src, reinterpret_cast<uint16_t*>(opt5060_macro.d_arch_q[sl]) + (size_t)off * D, mc, D);
            }
        };
        auto load_arch = [&](int sl, int off, int mc, float* dst) {
            if (opt5060_macro.use_q8) {
                k_ink_q8row_to_f32_rows<<<mc, 256, 0, st>>>(
                    reinterpret_cast<const int8_t*>(opt5060_macro.d_arch_q[sl]) + (size_t)off * D,
                    opt5060_macro.d_arch_sc[sl] + off, dst, mc, D);
            } else {
                k_ink_bf16_to_f32_rows<<<((size_t)mc * D + 255) / 256, 256, 0, st>>>(
                    reinterpret_cast<const uint16_t*>(opt5060_macro.d_arch_q[sl]) + (size_t)off * D, dst, mc, D);
            }
        };
        auto copy_arch = [&](int dst_sl, int src_sl, int W_rows) {
            const size_t qb = (size_t)W_rows * D * (opt5060_macro.use_q8 ? 1u : 2u);
            INK_CK(cudaMemcpyAsync(opt5060_macro.d_arch_q[dst_sl], opt5060_macro.d_arch_q[src_sl], qb, cudaMemcpyDeviceToDevice, st));
            if (opt5060_macro.use_q8) {
                INK_CK(cudaMemcpyAsync(opt5060_macro.d_arch_sc[dst_sl], opt5060_macro.d_arch_sc[src_sl], (size_t)W_rows * 4u, cudaMemcpyDeviceToDevice, st));
            }
        };

        out_last_mc = 0;
        const auto& ei = m.tinfo("model.llm.embed.weight");
        double tm_emb = 0, tm_stepA = 0, tm_stepB = 0, tm_stepC = 0, tm_out = 0;
        auto mnow = [] { return std::chrono::steady_clock::now(); };
        auto mms  = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const int win_W = (N > opt5060_macro.cap_W && N <= 2 * opt5060_macro.cap_W)
            ? ((N + 1) / 2)
            : opt5060_macro.cap_W;
        int win_off = 0;
        while (win_off < N) {
            const int W = std::min(win_W, N - win_off);
            const int wp0 = p0 + win_off;
            cur_max_pos = wp0 + W - 1;
            std::vector<std::pair<int, int>> mcs;
            for (int off = 0; off < W; off += max_batch) mcs.push_back({off, std::min(max_batch, W - off)});

            // 1. Embed all W tokens -> opt5060_macro.d_cur (BF16)
            auto te0 = mnow();
            for (const auto& [off, mc] : mcs) {
                int32_t* sb = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
                std::memcpy(sb, tok_ids + win_off + off, (size_t)mc * 4u);
                INK_CK(cudaMemcpyAsync(d_tok_ids, sb, (size_t)mc * 4u, cudaMemcpyHostToDevice, st));
                k_ink_embed_norm<<<mc, 256, 0, st>>>(
                    m.embed_ptr(), ei.fmt, (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
                    d_tok_ids, nullptr, nullptr, d_cur, mc, D);
                pack_cur(off, mc, d_cur);
            }
            if (prof_prefill) { INK_CK(cudaStreamSynchronize(st)); tm_emb += mms(te0, mnow()); }

            // Helper: run ONE visit across all W tokens (micro Attn+Router -> ONE W-token sm120 MoE -> micro Post-MoE ShortConv!)
            int n_blocks = 0;
            int n_bank   = 0;
            auto run_macro_visit = [&](int visit, int ip, bool has_msg) {
                auto ta0 = mnow();
                // Step A: Attention + ShortConv + Controllers + Router in micro-chunks (mc <= max_batch)
                for (const auto& [off, mc] : mcs) {
                    if (want_taps) {
                        int ti = (visit == 5) ? 0 : (visit == 23 ? 1 : (visit == 41 ? 2 : -1));
                        if (ti >= 0) {
                            INK_CK(cudaMemcpyAsync(opt5060_macro.h_taps[ti] + (size_t)off * D,
                                                   opt5060_macro.d_cur + (size_t)off * D,
                                                   (size_t)mc * D * sizeof(uint16_t), cudaMemcpyDeviceToHost, st));
                        }
                    }
                    int32_t* sb = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
                    for (int i = 0; i < mc; ++i) sb[i] = wp0 + off + i;
                    INK_CK(cudaMemcpyAsync(d_pos, sb, (size_t)mc * 4u, cudaMemcpyHostToDevice, st));
                    unpack_cur(off, mc, d_cur);
                    const float* bptrs[8] = {};
                    if (ip > 0) {
                        for (int b = 0; b < n_blocks; ++b) {
                            load_arch(arch_for_block(b), off, mc, d_blocks[b]);
                            bptrs[b] = d_blocks[b];
                        }
                    }
                    DepthStatePtrs bst{};
                    if (has_msg) {
                        for (int b = 0; b < n_bank; ++b) {
                            load_arch(7 + b, off, mc, d_bank[b]);
                            bst.p[b] = d_bank[b];
                        }
                        load_arch(11, off, mc, d_bank_out);
                        bst.p[n_bank] = d_bank_out;
                        bst.count = n_bank + 1;
                    }
                    int nb_tmp = n_blocks;
                    run_visit(visit, ip, has_msg ? &bst : nullptr, mc, 0, slot0, wp0 + off, nb_tmp, bptrs, /*pf_bf16_res=*/false, /*lm_stop_at_router=*/true);
                    if (ip % m.man.c.block_size == 0) {
                        save_arch(arch_for_block(n_blocks), off, mc, d_blocks[n_blocks]);
                    }
                    pack_cur(off, mc, d_cur);
                    sm120::quantize_act_fp4(d_moe_in, mc, D, opt5060_macro.d_xa_fp4 + (size_t)off * (D / 64) * sm120::kActFp4SlotBytes, st,
                                            macro_moe_sel((W >= 1536) ? 128 : ((W <= 512) ? 32 : 64)).row_quant);
                    INK_CK(cudaMemcpyAsync(opt5060_macro.d_tidx + (size_t)off * kInkTopK, moe_ws.topk_idx, (size_t)mc * kInkTopK * 4u, cudaMemcpyDeviceToDevice, st));
                    INK_CK(cudaMemcpyAsync(opt5060_macro.d_tw   + (size_t)off * kInkTopK, moe_ws.topk_w,   (size_t)mc * kInkTopK * 4u, cudaMemcpyDeviceToDevice, st));
                }
                if (ip % m.man.c.block_size == 0) ++n_blocks;
                if (prof_prefill) {
                    prof_flush();
                    INK_CK(cudaStreamSynchronize(st));
                    tm_stepA += mms(ta0, mnow());
                }
                auto tb0 = mnow();

                // Step B: ONE W-token sm_120 NVFP4 MoE sweep reusing d_pf_slab (ZERO extra VRAM!)
                const int tile_rows = (W >= 1536) ? 128 : ((W <= 512) ? 32 : 64);
                const int max_tiles = ink_moe_max_tiles(W, tile_rows);
                const size_t P_tot  = (size_t)W * kInkTopK;
                size_t pf_off = 0;
                auto pf_slice = [&](size_t bytes) -> uint8_t* {
                    uint8_t* ptr = d_pf_slab + pf_off;
                    pf_off += (bytes + 255ull) & ~255ull;
                    return ptr;
                };
                float*   d_y_acc  = reinterpret_cast<float*>(pf_slice((size_t)W * D * sizeof(float)));
                uint8_t* d_ha_mac = pf_slice(P_tot * (kInkInter / 64) * sm120::kActFp4SlotBytes);
                InkMoeTables tb{};
                tb.cnt     = reinterpret_cast<int32_t*>(pf_slice(128 * 4u));
                tb.off     = reinterpret_cast<int32_t*>(pf_slice(129 * 4u));
                tb.active  = reinterpret_cast<int32_t*>(pf_slice(128 * 4u));
                tb.nact    = reinterpret_cast<int32_t*>(pf_slice(16u));
                tb.pairs   = reinterpret_cast<int32_t*>(pf_slice(P_tot * 4u));
                tb.tok_idx = reinterpret_cast<int32_t*>(pf_slice(P_tot * 4u));
                tb.n_tiles = reinterpret_cast<int32_t*>(pf_slice(16u));
                tb.tiles   = reinterpret_cast<int2*>(pf_slice((size_t)max_tiles * sizeof(int2)));

                INK_CK(cudaMemsetAsync(d_y_acc, 0, (size_t)W * D * sizeof(float), st));
                if (fp8_sm.ready && W >= 512) {
                    const int P_i32 = (int)P_tot;
                    INK_CK(cudaMemsetAsync(tb.cnt, 0, 128 * sizeof(int32_t), st));
                    const int n_blk = std::min(128, std::max(1, (P_i32 + 255) / 256));
                    opt5060::k_moe_bin_par_count<<<n_blk, 256, 0, st>>>(m.d_experts[ip], opt5060_macro.d_tidx, P_i32, tb.cnt);
                    opt5060::k_moe_bin_par_prefix<<<1, 128, 0, st>>>(tile_rows, tb, fp8_sm.d_bin_fill);
                    opt5060::k_moe_bin_par_scatter<<<n_blk, 256, 0, st>>>(m.d_experts[ip], opt5060_macro.d_tidx, P_i32, fp8_sm.d_bin_fill, tb);
                } else {
                    k_ink_moe_bin<<<1, 128, 0, st>>>(m.d_experts[ip], opt5060_macro.d_tidx, W, tile_rows, tb);
                }
                const Sm120MoeSel mac_sel = macro_moe_sel(tile_rows);
                sm120::moe_sf4_gu_tc(m.d_sec4, m.d_experts[ip], tb, opt5060_macro.d_xa_fp4, d_ha_mac, max_tiles, tile_rows, st, &mac_sel);
                sm120::moe_sf4_dn_tc(m.d_sec4, m.d_experts[ip], tb, d_ha_mac, opt5060_macro.d_tw, d_y_acc, max_tiles, tile_rows, st, /*atomic_reduce=*/true, &mac_sel);
                if (prof_prefill) { INK_CK(cudaStreamSynchronize(st)); tm_stepB += mms(tb0, mnow()); }
                auto tc0 = mnow();

                // Step C: Post-MoE ShortConv4 + Residual Add directly on GDDR7 in micro-chunks!
                constexpr int mlp_ch_off = 2 * kInkKVDim + D;
                float* hslot = d_sconv_pool[visit] + (size_t)mlp_ch_off * 3 * max_slots + (size_t)slot0 * 3 * D;
                for (const auto& [off, mc] : mcs) {
                    unpack_cur(off, mc, d_cur);
                    const float* src_moe = d_y_acc + (size_t)off * D;
                    dim3 g((D + 255) / 256, (mc + 15) / 16);
                    k_ink_sconv4_seq_par<<<g, 256, 0, st>>>(src_moe, m.d_sconv_mlp[ip], nullptr, hslot, d_cur, mc, D, 0, true);
                    k_ink_sconv4_save_hist<<<(D + 255) / 256, 256, 0, st>>>(src_moe, hslot, mc, D);
                    pack_cur(off, mc, d_cur);
                }
                if (prof_prefill) { INK_CK(cudaStreamSynchronize(st)); tm_stepC += mms(tc0, mnow()); }
            };

            // 2. Prelude: layers 0..5 (visits 0..5)
            int visit = 0;
            for (int ip = 0; ip < m.man.c.core_start; ++ip) {
                run_macro_visit(visit, ip, false);
                ++visit;
            }
            const int n_pre_blk = n_blocks;  // 2 (blocks[0..1] stay untouched in arch 0..1!)
            // Save prelude state into arch slot 6
            for (const auto& [off, mc] : mcs) {
                unpack_cur(off, mc, d_cur);
                save_arch(6, off, mc, d_cur);
            }

            // 3. Looped Core: layers 6..17 x 3 passes (visits 6..41)
            for (int pass = 0; pass < m.man.c.num_passes; ++pass) {
                if (pass > 0) {
                    n_bank = n_blocks - (n_pre_blk + 1);  // 3
                    for (int b = 0; b < n_bank; ++b) {
                        copy_arch(7 + b, arch_for_block(n_pre_blk + 1 + b), W);
                    }
                    for (const auto& [off, mc] : mcs) {
                        unpack_cur(off, mc, d_cur);
                        save_arch(11, off, mc, d_cur);  // bank_out -> arch 11
                        load_arch(6, off, mc, d_cur);   // prelude -> cur
                        pack_cur(off, mc, d_cur);
                    }
                    n_blocks = n_pre_blk;
                }
                for (int ip = m.man.c.core_start; ip < m.man.c.core_end; ++ip) {
                    run_macro_visit(visit, ip, pass > 0);
                    ++visit;
                }
            }

            // 4. Coda: layers 18..23 (visits 42..47; blocks[6..7] reuse arch slots 7..8!)
            visit = 42;
            for (int ip = m.man.c.core_end; ip < m.man.c.n_phys; ++ip) {
                run_macro_visit(visit, ip, false);
                ++visit;
            }

            // 5. Final Output Depth Mix + Output RMSNorm + Taps (0..3)
            auto to0 = mnow();
            for (const auto& [off, mc] : mcs) {
                const bool is_final = (win_off + off + mc >= N);
                if (!want_taps && !is_final) continue;
                unpack_cur(off, mc, d_cur);
                DepthStatePtrs stp{};
                for (int b = 0; b < n_blocks; ++b) {
                    load_arch(arch_for_block(b), off, mc, d_blocks[b]);
                    stp.p[b] = d_blocks[b];
                }
                stp.p[n_blocks] = d_cur;
                stp.count = n_blocks + 1;
                k_ink_attn_res_mix<<<mc, 256, 0, st>>>(stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, mc, D);
                k_ink_rms_norm<<<mc, 256, 0, st>>>(d_moe_in, (const float*)m.trunk_ptr("model.llm.norm.weight"), d_h_norm, nullptr, mc, D);
                if (want_taps) {
                    for (int t = 0; t < 3; ++t) {
                        INK_CK(cudaMemcpyAsync(d_x_bf16, opt5060_macro.h_taps[t] + (size_t)off * D,
                                               (size_t)mc * D * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
                        k_ink_bf16_to_f32_rows<<<((size_t)mc * D + 255) / 256, 256, 0, st>>>(
                            reinterpret_cast<const uint16_t*>(d_x_bf16), d_taps[t], mc, D);
                    }
                    INK_CK(cudaMemcpyAsync(d_taps[3], d_h_norm, (size_t)mc * D * sizeof(float), cudaMemcpyDeviceToDevice, st));
                    on_taps_chunk(wp0 + off, mc);
                }
                if (is_final) out_last_mc = mc;
            }
            if (prof_prefill) { INK_CK(cudaStreamSynchronize(st)); tm_out += mms(to0, mnow()); }
            win_off += W;
        }
        if (prof_prefill) {
            prof_print("MACRO-STEPA-SUM", 1);
            prof_reset();
            std::fprintf(stderr, "[MACRO-PF N=%d W=%d] emb=%.1fms stepA(attn+dense+ctrl)=%.1fms stepB(MoE)=%.1fms stepC(sconv)=%.1fms out+taps=%.1fms TOT=%.1fms (%.0f ppts)\n",
                         N, opt5060_macro.cap_W, tm_emb, tm_stepA, tm_stepB, tm_stepC, tm_out,
                         tm_emb + tm_stepA + tm_stepB + tm_stepC + tm_out,
                         (double)N * 1000.0 / std::max(0.1, tm_emb + tm_stepA + tm_stepB + tm_stepC + tm_out));
        }
        // Release macro prefill slab immediately after prefill so decode gets full VRAM headroom for CUDA graphs!
        INK_CK(cudaStreamSynchronize(st));
        opt5060_macro.free_all();
        return true;
    }

    // Layer-Major Single-Sweep Coda Prefill: streams each spilled Coda layer (L19..23) over PCIe ONCE for up to 8704 tokens!
    // Leaves final micro-chunk's d_h_norm in runner->d_h_norm (exact length `out_last_mc` starting at `p0 + N - out_last_mc`).
    // Calls `on_taps_chunk(chunk_p0, chunk_mc)` after computing d_taps[0..3] for each micro-chunk if want_taps==true.
    template <typename TapFn>
    bool forward_seq_layer_major(const int32_t* tok_ids, int p0, int N, int slot0, bool want_taps, int& out_last_mc, TapFn&& on_taps_chunk) {
        if (!can_layer_major_prefill() || N < 1024) return false;
        const int fhc = first_host_coda_layer();
        const int n_blk_pre = (fhc + m.man.c.block_size - 1) / m.man.c.block_size;
        constexpr int kMaxWin = 8704;
        const int first_W = std::min(N, kMaxWin);
        if (!opt5060_pf.ensure(first_W, m.primary_gpu)) return false;

        out_last_mc = 0;
        for (int win_off = 0; win_off < N; ) {
            int W = std::min(kMaxWin, N - win_off);
            if (N - win_off - W > 0 && N - win_off - W < 64) W -= 64;
            const int wp0 = p0 + win_off;
            cur_max_pos = wp0 + W - 1;

            // Build micro-chunk ranges (every chunk strictly >= 32 tokens so sm120::available() TC path is ALWAYS taken!)
            std::vector<std::pair<int, int>> mcs;
            for (int off = 0; off < W; ) {
                int mc = std::min(max_batch, W - off);
                if (W - off - mc > 0 && W - off - mc < 32) {
                    mc = (W - off) / 2;
                }
                mcs.push_back({off, mc});
                off += mc;
            }

            // Stage A: Pure-VRAM Prefix + Core (Visits 0 .. 42+(fhc-18)-1) in micro-chunks (ZERO PCIe expert traffic!)
            for (const auto& [off, mc] : mcs) {
                int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
                std::memcpy(slot_buf, tok_ids + win_off + off, (size_t)mc * 4);
                int32_t* h_pos = slot_buf + mc;
                for (int i = 0; i < mc; ++i) h_pos[i] = wp0 + off + i;
                INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)mc * 4, cudaMemcpyHostToDevice, st));
                INK_CK(cudaMemcpyAsync(d_pos, h_pos, (size_t)mc * 4, cudaMemcpyHostToDevice, st));

                emit_visits_prefix_to(fhc, mc, 0, slot0, wp0 + off, want_taps);

                const size_t hb = (size_t)mc * kInkHidden * sizeof(float);
                INK_CK(cudaMemcpyAsync(opt5060_pf.h_cur.data() + (size_t)off * kInkHidden, d_cur, hb, cudaMemcpyDeviceToHost, st));
                for (int b = 0; b < n_blk_pre; ++b) {
                    INK_CK(cudaMemcpyAsync(opt5060_pf.h_blocks[b].data() + (size_t)off * kInkHidden, d_blocks[b], hb, cudaMemcpyDeviceToHost, st));
                }
                if (want_taps) {
                    for (int t = 0; t < 3; ++t) {
                        INK_CK(cudaMemcpyAsync(opt5060_pf.h_taps[t].data() + (size_t)off * kInkHidden, d_taps[t], hb, cudaMemcpyDeviceToHost, st));
                    }
                }
                INK_CK(cudaStreamSynchronize(st));
            }

            // Stage B: Each Spilled Coda Layer ip = fhc..23 — streamed over PCIe EXACTLY ONCE across ALL W tokens!
            auto& hyb = m.hybrid;
            for (int ip = fhc; ip < m.man.c.n_phys; ++ip) {
                const int visit = 42 + (ip - m.man.c.core_end);
                const int n_blk_in = (ip + m.man.c.block_size - 1) / m.man.c.block_size;
                const bool is_blk_start = (ip % m.man.c.block_size == 0);
                const auto& info = m.moe_info[ip];

                // Sub-step B1: Attention + Post-Attn SConv + Pre-MoE Mix/Norm + Router + Quantize X into compact VRAM window!
                for (const auto& [off, mc] : mcs) {
                    const size_t hb = (size_t)mc * kInkHidden * sizeof(float);
                    int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
                    for (int i = 0; i < mc; ++i) slot_buf[i] = wp0 + off + i;
                    INK_CK(cudaMemcpyAsync(d_pos, slot_buf, (size_t)mc * 4, cudaMemcpyHostToDevice, st));
                    INK_CK(cudaMemcpyAsync(d_cur, opt5060_pf.h_cur.data() + (size_t)off * kInkHidden, hb, cudaMemcpyHostToDevice, st));
                    const float* bptrs[8] = {};
                    for (int b = 0; b < n_blk_in; ++b) {
                        INK_CK(cudaMemcpyAsync(d_blocks[b], opt5060_pf.h_blocks[b].data() + (size_t)off * kInkHidden, hb, cudaMemcpyHostToDevice, st));
                        bptrs[b] = d_blocks[b];
                    }
                    int nb = n_blk_in;
                    run_visit(visit, ip, nullptr, mc, 0, slot0, wp0 + off, nb, bptrs, /*pf_bf16_res=*/false, /*lm_stop_at_router=*/true);

                    INK_CK(cudaMemcpyAsync(opt5060_pf.h_cur.data() + (size_t)off * kInkHidden, d_cur, hb, cudaMemcpyDeviceToHost, st));
                    if (is_blk_start) {
                        INK_CK(cudaMemcpyAsync(opt5060_pf.h_blocks[n_blk_in].data() + (size_t)off * kInkHidden, d_blocks[n_blk_in], hb, cudaMemcpyDeviceToHost, st));
                    }
                    INK_CK(cudaMemcpyAsync(opt5060_pf.d_tidx_win + (size_t)off * kInkTopK, moe_ws.topk_idx, (size_t)mc * kInkTopK * 4, cudaMemcpyDeviceToDevice, st));
                    INK_CK(cudaMemcpyAsync(opt5060_pf.d_tw_win   + (size_t)off * kInkTopK, moe_ws.topk_w,   (size_t)mc * kInkTopK * 4, cudaMemcpyDeviceToDevice, st));
                    if (info.has_sf4) {
                        {
                            // the same variant choice launch_ink_moe_dispatch_only makes for these mc tokens (quant_x_done = true below)
                            const auto& c5060 = ink_rtx5060ti_cfg(m.primary_gpu);
                            const Sm120MoeSel qsel = c5060.sm120_tc_smallm ? sm120_moe_sel(sm120_cfg(m.primary_gpu), mc) : Sm120MoeSel{};
                            sm120::quantize_act_fp4(d_moe_in, mc, kInkHidden, opt5060_pf.d_xa_fp4_win + (size_t)off * (kInkHidden / 64) * sm120::kActFp4SlotBytes, st,
                                                    qsel.active && qsel.row_quant);
                        }
                    }
                    if (info.has_mx8) {
                        sm120::quantize_act_mx8(d_moe_in, mc, kInkHidden,
                                                opt5060_pf.d_xq_mx8_win + (size_t)off * kInkHidden,
                                                opt5060_pf.d_xsc_mx8_win + (size_t)off * (kInkHidden / 32), st);
                    }
                    INK_CK(cudaStreamSynchronize(st));
                }

                // Sub-step B2: Stream layer ip's 128 host experts ONCE in L2-resident <= 24 MiB ping-pong chunks!
                INK_CK(cudaMemsetAsync(opt5060_pf.d_moe_out_win, 0, (size_t)W * kInkHidden * sizeof(float), st));
                struct LmChunk { std::vector<int> eids; std::vector<size_t> offs; size_t total_bytes = 0; };
                std::vector<LmChunk> chunks;
                const size_t chunk_cap = std::min<size_t>(hyb.pf_stage_bytes, 24ull << 20);
                {
                    LmChunk cur;
                    size_t used = 0;
                    for (int e = 0; e < kInkExperts; ++e) {
                        const auto& hm = hyb.host_meta[ip * kInkExperts + e];
                        if (!hm.body) continue;
                        const size_t b = (hm.body_bytes + 255ull) & ~255ull;
                        if (used + b > chunk_cap && !cur.eids.empty()) {
                            cur.total_bytes = used;
                            chunks.push_back(std::move(cur));
                            cur = {};
                            used = 0;
                        }
                        cur.eids.push_back(e);
                        cur.offs.push_back(used);
                        used += b;
                    }
                    if (!cur.eids.empty()) { cur.total_bytes = used; chunks.push_back(std::move(cur)); }
                }

                auto issue_lm_dma = [&](size_t ci, int buf) {
                    if (hyb.pf_buf_busy[buf]) {
                        INK_CK(cudaEventSynchronize(hyb.pf_ev_done[buf]));
                        hyb.pf_buf_busy[buf] = false;
                    }
                    const auto& ch = chunks[ci];
                    InkExpertDev* h_exp = hyb.h_pf_exp_pin[buf];
                    for (int e = 0; e < kInkExperts; ++e) { h_exp[e] = {}; h_exp[e].tier = 2; }
                    const bool use_bounce = (!hyb.sec4_is_pinned && hyb.h_pf_bounce[buf] != nullptr);
                    for (size_t i = 0; i < ch.eids.size(); ++i) {
                        const int e = ch.eids[i];
                        const auto& hm = hyb.host_meta[ip * kInkExperts + e];
                        h_exp[e].body_off = ch.offs[i];
                        h_exp[e].dn_off   = hm.dn_off;
                        h_exp[e].gu_code  = hm.gu_code;
                        h_exp[e].dn_code  = hm.dn_code;
                        h_exp[e].tier     = 0;
                        if (use_bounce) std::memcpy(hyb.h_pf_bounce[buf] + ch.offs[i], hm.body, hm.body_bytes);
                        else INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf] + ch.offs[i], hm.body, hm.body_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
                    }
                    if (use_bounce && ch.total_bytes > 0) {
                        INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf], hyb.h_pf_bounce[buf], ch.total_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
                    }
                    INK_CK(cudaMemcpyAsync(hyb.d_pf_exp_tmp[buf], h_exp, (size_t)kInkExperts * sizeof(InkExpertDev), cudaMemcpyHostToDevice, hyb.pf_dma_st));
                    INK_CK(cudaEventRecord(hyb.pf_ev_dma[buf], hyb.pf_dma_st));
                };

                uint8_t* const orig_xa  = moe_ws.xa_fp4;
                uint8_t* const orig_xq  = moe_ws.xq_mx8;
                uint8_t* const orig_xsc = moe_ws.xsc_mx8;
                if (!chunks.empty()) issue_lm_dma(0, 0);
                for (size_t ci = 0; ci < chunks.size(); ++ci) {
                    const int buf = (int)(ci & 1);
                    if (ci + 1 < chunks.size()) issue_lm_dma(ci + 1, 1 - buf);
                    INK_CK(cudaStreamWaitEvent(st, hyb.pf_ev_dma[buf], 0));
                    for (const auto& [off, mc] : mcs) {
                        INK_CK(cudaMemcpyAsync(moe_ws.topk_idx, opt5060_pf.d_tidx_win + (size_t)off * kInkTopK, (size_t)mc * kInkTopK * 4, cudaMemcpyDeviceToDevice, st));
                        INK_CK(cudaMemcpyAsync(moe_ws.topk_w,   opt5060_pf.d_tw_win   + (size_t)off * kInkTopK, (size_t)mc * kInkTopK * 4, cudaMemcpyDeviceToDevice, st));
                        INK_CK(cudaMemsetAsync(moe_ws.y_pair, 0, (size_t)mc * kInkTopK * kInkHidden * sizeof(float), st));
                        moe_ws.xa_fp4  = opt5060_pf.d_xa_fp4_win  + (size_t)off * (kInkHidden / 64) * sm120::kActFp4SlotBytes;
                        moe_ws.xq_mx8  = opt5060_pf.d_xq_mx8_win  + (size_t)off * kInkHidden;
                        moe_ws.xsc_mx8 = opt5060_pf.d_xsc_mx8_win + (size_t)off * (kInkHidden / 32);
                        launch_ink_moe_dispatch_only(st, hyb.d_pf_stage[buf], hyb.d_pf_exp_tmp[buf], info, nullptr, mc, moe_ws, false, /*quant_x_done=*/true);
                        k_ink_moe_reduce8_add<<<dim3((kInkHidden + 255) / 256, mc), 256, 0, st>>>(
                            moe_ws.y_pair, opt5060_pf.d_moe_out_win + (size_t)off * kInkHidden, mc);
                    }
                    INK_CK(cudaEventRecord(hyb.pf_ev_done[buf], st));
                    hyb.pf_buf_busy[buf] = true;
                }
                moe_ws.xa_fp4  = orig_xa;
                moe_ws.xq_mx8  = orig_xq;
                moe_ws.xsc_mx8 = orig_xsc;

                // Sub-step B3: Post-MoE ShortConv4 + Residual Add across all W tokens in micro-chunks!
                constexpr int mlp_ch_off = 2 * kInkKVDim + kInkHidden;
                float* hslot = d_sconv_pool[visit] + (size_t)mlp_ch_off * 3 * max_slots + (size_t)slot0 * 3 * kInkHidden;
                for (const auto& [off, mc] : mcs) {
                    const size_t hb = (size_t)mc * kInkHidden * sizeof(float);
                    INK_CK(cudaMemcpyAsync(d_cur, opt5060_pf.h_cur.data() + (size_t)off * kInkHidden, hb, cudaMemcpyHostToDevice, st));
                    const float* src_moe = opt5060_pf.d_moe_out_win + (size_t)off * kInkHidden;
                    dim3 g((kInkHidden + 255) / 256, (mc + 15) / 16);
                    k_ink_sconv4_seq_par<<<g, 256, 0, st>>>(src_moe, m.d_sconv_mlp[ip], nullptr, hslot, d_cur, mc, kInkHidden, 0, true);
                    k_ink_sconv4_save_hist<<<(kInkHidden + 255) / 256, 256, 0, st>>>(src_moe, hslot, mc, kInkHidden);
                    INK_CK(cudaMemcpyAsync(opt5060_pf.h_cur.data() + (size_t)off * kInkHidden, d_cur, hb, cudaMemcpyDeviceToHost, st));
                    INK_CK(cudaStreamSynchronize(st));
                }
            }

            // Stage C: Output Depth Mix + Output RMSNorm + Tap 3 + DSpark Tap Ingest!
            const int n_blk_all = (m.man.c.n_phys + m.man.c.block_size - 1) / m.man.c.block_size;  // 8
            for (const auto& [off, mc] : mcs) {
                const bool is_final_chunk = (win_off + off + mc >= N);
                if (!want_taps && !is_final_chunk) continue;
                const size_t hb = (size_t)mc * kInkHidden * sizeof(float);
                INK_CK(cudaMemcpyAsync(d_cur, opt5060_pf.h_cur.data() + (size_t)off * kInkHidden, hb, cudaMemcpyHostToDevice, st));
                DepthStatePtrs stp{};
                for (int b = 0; b < n_blk_all; ++b) {
                    INK_CK(cudaMemcpyAsync(d_blocks[b], opt5060_pf.h_blocks[b].data() + (size_t)off * kInkHidden, hb, cudaMemcpyHostToDevice, st));
                    stp.p[b] = d_blocks[b];
                }
                stp.p[n_blk_all] = d_cur;
                stp.count = n_blk_all + 1;
                k_ink_attn_res_mix<<<mc, 256, 0, st>>>(stp, (const float*)m.trunk_ptr("model.llm.attn_res.output_query"), d_moe_in, 1, mc, kInkHidden);
                k_ink_rms_norm<<<mc, 256, 0, st>>>(d_moe_in, (const float*)m.trunk_ptr("model.llm.norm.weight"), d_h_norm, nullptr, mc, kInkHidden);
                if (want_taps) {
                    for (int t = 0; t < 3; ++t) {
                        INK_CK(cudaMemcpyAsync(d_taps[t], opt5060_pf.h_taps[t].data() + (size_t)off * kInkHidden, hb, cudaMemcpyHostToDevice, st));
                    }
                    INK_CK(cudaMemcpyAsync(d_taps[3], d_h_norm, hb, cudaMemcpyDeviceToDevice, st));
                    on_taps_chunk(wp0 + off, mc);
                }
                if (is_final_chunk) out_last_mc = mc;
                INK_CK(cudaStreamSynchronize(st));
            }
            win_off += W;
        }
        return true;
    }

    // Ensure Mode-3 (Batched Multi-Slot Speculative Chain Verify) staging buffers (VRAM BF16 when fits -> 1.8 ms, else mapped host!)
    // Indexed by flat B * K token-slots so adaptive tail-drain (e.g. B=32 K=6 or B=24 K=8 -> B*K=192) reuses the EXACT same VRAM buffer!
    bool ensure_mc_buffers(int need_bk = 192) {
        const int want_bk = std::max(max_slots, std::min(max_batch, need_bk));
        if (d_mc_sconv_stage && mc_stage_cap_k >= want_bk && d_mc_slot_ids && d_mc_base_pos && d_mc_n_acc) return true;
        if (d_mc_sconv_stage) {
            if (d_mc_sconv_is_vram) cudaFree(d_mc_sconv_stage);
            else if (h_mc_sconv_raw) cudaFreeHost(h_mc_sconv_raw);
            d_mc_sconv_stage = nullptr;
            h_mc_sconv_raw   = nullptr;
            d_mc_sconv_is_vram = false;
            mc_stage_cap_k   = 0;
        }
        const size_t stg_bytes = (size_t)kInkVisits * (size_t)want_bk * kInkSconvCh * sizeof(uint16_t);
        bool got_vram = false;
        if (m.all_sf4_vram) {
            size_t vf = 0, vt = 0;
            if (ink_cuda_mem_info(&vf, &vt, m.primary_gpu) == cudaSuccess && vf > stg_bytes + (60ull << 20)) {
                if (cudaMalloc(&d_mc_sconv_stage, stg_bytes) == cudaSuccess) {
                    d_mc_sconv_is_vram = true;
                    got_vram = true;
                    std::fprintf(stderr, "[Ink5060Ti] C64 ShortConv BF16 stage in GDDR7 VRAM (BK<=%d, %.1f MiB)\n",
                                 want_bk, (double)stg_bytes / 1048576.0);
                } else {
                    (void)cudaGetLastError();
                    d_mc_sconv_stage = nullptr;
                }
            }
        }
        if (!got_vram) {
            if (cudaHostAlloc(&h_mc_sconv_raw, stg_bytes, cudaHostAllocMapped) != cudaSuccess) {
                (void)cudaGetLastError();
                h_mc_sconv_raw = nullptr;
                return false;
            }
            void* dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, h_mc_sconv_raw, 0) != cudaSuccess) {
                (void)cudaGetLastError();
                cudaFreeHost(h_mc_sconv_raw);
                h_mc_sconv_raw = nullptr;
                return false;
            }
            d_mc_sconv_stage = static_cast<uint16_t*>(dp);
            d_mc_sconv_is_vram = false;
        }
        mc_stage_cap_k = want_bk;
        if (!d_mc_slot_ids) d_mc_slot_ids = dalloc<int32_t>(max_slots);
        if (!d_mc_base_pos) d_mc_base_pos = dalloc<int32_t>(max_slots);
        if (!d_mc_n_acc)    d_mc_n_acc    = dalloc<int32_t>(max_slots);
        return true;
    }

    // Mode-3 Batched Multi-Slot Speculative Chain Verify: B concurrent slots x K chain tokens (M = B * K <= max_batch)!
    void forward_multichain_verify(
        const int32_t* tok_ids,   // [B * K]
        const int32_t* slot_ids,  // [B]
        const int32_t* base_pos,  // [B]
        int B, int K, bool want_taps) {
        const int M = B * K;
        mc_cur_B = B;
        mc_cur_K = K;
        int mp = 0;
        for (int b = 0; b < B; ++b) mp = std::max(mp, base_pos[b] + K - 1);
        cur_max_pos = mp;

        int32_t* buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        int32_t* h_tok  = buf;
        int32_t* h_sl_m = buf + M;
        int32_t* h_pos  = buf + 2 * M;
        int32_t* h_sl_b = buf + 3 * M;
        int32_t* h_bp_b = buf + 3 * M + B;
        for (int b = 0; b < B; ++b) {
            h_sl_b[b] = slot_ids[b];
            h_bp_b[b] = base_pos[b];
            for (int k = 0; k < K; ++k) {
                const int m = b * K + k;
                h_tok[m]  = tok_ids[m];
                h_sl_m[m] = slot_ids[b];
                h_pos[m]  = base_pos[b] + k;
            }
        }
        INK_CK(cudaMemcpyAsync(d_tok_ids,     h_tok,  (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_slots,       h_sl_m, (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos,         h_pos,  (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_mc_slot_ids, h_sl_b, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_mc_base_pos, h_bp_b, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        mc_kv_pf_active = false;
        for (int v = 0; v < kInkVisits; ++v) { mc_pf_slot[v] = -1; mc_pf_next[v] = -1; }
        if (swa_in_host && kv_dev_ctx == 0 && c64_kv_pf.ensure(max_slots, kv_dtype, m.primary_gpu) &&
            cur_max_pos < c64_kv_pf.cap_pos) {
            int first_pf = -1, prev_pf = -1, pf_idx = 0;
            for (int v = 0; v < kInkVisits; ++v) {
                if (!swa_pinned[v] && host_kv_raw_k[v] != nullptr) {
                    mc_pf_slot[v] = (pf_idx++) & 1;
                    if (first_pf < 0) first_pf = v;
                    if (prev_pf >= 0) mc_pf_next[prev_pf] = v;
                    prev_pf = v;
                }
            }
            if (first_pf >= 0) {
                INK_CK(cudaEventRecord(c64_kv_pf.ev_comp[0], st));
                INK_CK(cudaEventRecord(c64_kv_pf.ev_comp[1], st));
                mc_kv_pf_active = c64_kv_pf.prefetch_visit(mc_pf_slot[first_pf], kv_desc[first_pf],
                    host_kv_raw_k[first_pf], host_kv_raw_v[first_pf], host_kv_raw_ksc[first_pf], host_kv_raw_vsc[first_pf],
                    cur_max_pos + 1, true);
            }
        }
        emit_48_visits(M, 3, 0, 0, want_taps);
        mc_kv_pf_active = false;
        if (prof_enabled) {
            prof_flush();
            static int mc_pf = 0;
            if (++mc_pf <= 3 || (mc_pf % 10 == 0)) {
                char tag[32];
                std::snprintf(tag, sizeof(tag), "C64-MC-M%d", M);
                prof_print(tag, 1);
            }
            prof_reset();
        }
    }

    void commit_multichain_sconv(const int32_t* n_acc_arr, int B, int K) {
        int32_t* h_na = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(h_na, n_acc_arr, (size_t)B * 4);
        INK_CK(cudaMemcpyAsync(d_mc_n_acc, h_na, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        k_ink_commit_sconv_multichain<<<dim3(kInkVisits, 4, B), 256, 0, st>>>(
            d_sconv_pool_arr, d_mc_sconv_stage, d_mc_slot_ids, d_mc_n_acc, B, K, max_slots);
    }

    // ---- Launch Full 48-Visit Forward (with Safe CUDA Graph Capture -> Eager Fallback, NEVER exit(3)!) ----
    void forward_seq(const int32_t* tok_ids, int p0, int M, int slot0, bool want_taps) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        cur_max_pos = p0 + M - 1;
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)M * 4);
        int32_t* h_pos = slot_buf + M;
        for (int i = 0; i < M; ++i) h_pos[i] = p0 + i;
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, h_pos, (size_t)M * 4, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && !swa_in_host && !m.ls_active && !has_img_step && max_passes_override == 0 && M == 1 && slot0 == 0 && !m.hybrid.needs_eager() && !elastic_kv.needs_eager_for_pos(cur_max_pos)) {
            GraphKey key{M, 0, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(M, 0, slot0, p0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = pdl::instantiate(&exec, gr, "decode");
                        cudaGraphDestroy(gr);
                        gr = nullptr;   // destroyed: the failure branch below must not destroy it a second time (cudaErrorInvalidValue left pending, read by the next cudaGetLastError as a fatal "invalid argument")
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    (void)cudaGetLastError();   // the eager fallback starts with a clean error state
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] CUDA Graph capture unavailable (err=%d): falling back to eager mode\n", (int)ge);
                    emit_48_visits(M, 0, slot0, p0, want_taps);
                    if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_after(1, m.d_sec4, m.d_experts, st);
                    return;
                }
                graph_cache.push_back({key, exec});
            }
            cudaError_t le = cudaGraphLaunch(exec, st);
            if (le != cudaSuccess) {
                cudaGetLastError();
                for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
                graph_cache.clear();
                use_cuda_graph = false;
                emit_48_visits(M, 0, slot0, p0, want_taps);
            }
            if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_after(1, m.d_sec4, m.d_experts, st);
            if (dbg_hash_on) dbg_hash_print("fwd", M, h_pos, nullptr, tok_ids, nullptr, -1);
            return;
        }
        emit_48_visits(M, 0, slot0, p0, want_taps);
        if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_after(M <= 16 ? M : 0, m.d_sec4, m.d_experts, st);
        if (dbg_hash_on) dbg_hash_print("fwd", M, h_pos, nullptr, tok_ids, nullptr, -1);
        if (prof_enabled) {   // STRATA_INK_PROF=1: where one decode step goes (eager, events only), every 100 steps
            prof_flush();
            if (M != 1) {
                if (prof_prefill && M >= 64) {   // STRATA_INK_PROF=2: one line per prefill chunk (the phases of this chunk, ms)
                    char tag[32];
                    std::snprintf(tag, sizeof(tag), "PREFILL-M%d", M);
                    prof_print(tag, 1);
                }
                prof_reset();
            }
            else if (++prof_steps % 100 == 0) { prof_print("DECODE-M1", (int)prof_steps); prof_steps = 0; prof_reset(); }
        }
    }

    // Launch probe for the cold-expert lane (STRATA_INK_EXACT unset): the same short run of decode steps is timed with the CPU lane and with
    // the card's own copy-in lane (exact), on the placement the loaders made (the tier is paused and learns nothing), and the faster one is kept:
    // the CPU lane unless the card's copy is clearly (10%) faster, because a CPU lane is the default of this engine and the probe is noisy.
    // Runs once after the runner exists, before any request; a slot is reset when a request takes it, so the probe leaves no state behind.
    void pick_cold_lane() {
        auto& h = m.hybrid;
        if (h.lane_pref != 0 || !h.active || m.n_host_experts == 0) return;   // forced lane, or nothing is cold
        if (m.ls_active || has_img_step) {
            std::fprintf(stderr, "[InkHybrid] cold experts: no launch probe on this configuration, computed on the CPU where they are\n");
            return;
        }
        // The first 24 token ids of a plain chat prompt (the chat template's header and a user question): the experts they route to are those of real text,
        // not of random ids, so the probe's share of cold experts is close to a request's.
        static const int32_t kProbeTok[24] = {200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004,
                                              40, 150480, 149882, 150617, 153048, 13, 153611, 154845, 30, 200010, 200001, 200004};
        constexpr int kWarm = 6, kTimed = 18;
        const auto run_lane = [&](bool exact) -> double {
            h.set_lane(exact);
            drop_all_graphs();
            std::vector<double> ms;
            for (int i = 0; i < kWarm + kTimed; ++i) {
                int32_t tok = kProbeTok[i];
                const auto t0 = std::chrono::steady_clock::now();
                forward_seq(&tok, i, 1, 0, false);
                INK_CK(cudaStreamSynchronize(st));
                if (i >= kWarm) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];
        };
        h.probing = true;
        const double t_cpu = run_lane(false);
        const double t_card = run_lane(true);
        h.probing = false;
        const bool exact = t_card < 0.9 * t_cpu;
        h.set_lane(exact);
        if (ink_rtx5060ti_cfg(m.primary_gpu).compact_hybrid && !exact && t_cpu > 0.0 && t_card > 0.0 && h.dec_share_user < 0.0) {
            // COMPACT_HYBRID row: the split of the batched cold experts comes from this box's own two lane times (see auto_pcie_share)
            // t_cpu / (t_cpu + t_card) would finish both lanes together if the copy lane had no overhead of its own; it has (the sync per visit, the activation copy, the bounce ring), so the
            // optimum measured on two boxes sits lower: RTX 5060 box ratio 0.20, best share 0.05-0.15 (prose steps/s 10.0 at 0.20 against 15.2 at 0.15); RTX 5070 box ratio 0.26,
            // flat from 0.15 to 0.30. 0.75 x ratio gives 0.15 and 0.20 there.
            h.auto_pcie_share = std::max(0.0, std::min(0.65, 0.75 * t_cpu / (t_cpu + t_card)));
            std::fprintf(stderr, "[InkHybrid] cold-expert split from the probe: CPU lane %.1f ms, copy lane %.1f ms -> %.2f of the verify steps' cold experts go over PCIe, the rest to the CPU\n",
                         t_cpu, t_card, h.auto_pcie_share);
        }
        drop_all_graphs();
        if (h.d_use_cnt) INK_CK(cudaMemsetAsync(h.d_use_cnt, 0, (size_t)h.n_layers * h.n_experts * sizeof(float), st));
        if (h.d_stat) INK_CK(cudaMemsetAsync(h.d_stat, 0, 2 * sizeof(unsigned long long), st));
        std::fill(h.h_use_batch.begin(), h.h_use_batch.end(), 0.0f);   // the eager copy lane counted its routing here
        if (h.d_trace_n) { INK_CK(cudaMemsetAsync(h.d_trace_n, 0, 4, st)); h.trace_flushed = 0; }
        INK_CK(cudaStreamSynchronize(st));
        std::fprintf(stderr, "[InkHybrid] cold experts: launch probe (%d decode steps each): CPU lane %.1f ms/token, copy into the card %.1f ms/token -> %s%s\n",
                     kTimed, t_cpu, t_card, exact ? "copied into the card and computed by its kernels (identical answers wherever an expert lives)"
                                                  : "computed on the CPU where they are (the answer can differ from an all-resident run in rounding)",
                     " (STRATA_INK_EXACT=1 or 0 forces a lane)");
    }

    // Speculative Tree-32 / Block-8 verify (100% CUDA-Graph capturable with safe eager fallback!)
    void forward_tree_verify(const InkTreeSpec& tree, int p0, int slot0, bool want_taps) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        const int M = tree.M;
        {
            int md = 0;
            for (int i = 0; i < M; ++i) md = std::max(md, (int)tree.depth[i]);
            cur_max_pos = p0 + md;
        }
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        int32_t* h_tok = slot_buf;
        int32_t* h_pos = slot_buf + kInkMaxTree;
        uint8_t* h_bytes = reinterpret_cast<uint8_t*>(slot_buf + 2 * kInkMaxTree);
        for (int i = 0; i < M; ++i) {
            h_tok[i] = tree.tok[i];
            h_pos[i] = p0 + (int)tree.depth[i];
        }
        std::memcpy(h_bytes,       tree.sconv_lag, kInkMaxTree * 3);
        std::memcpy(h_bytes + 128, tree.anc,       kInkMaxTree * kInkMaxTreeDepth);
        std::memcpy(h_bytes + 384, tree.depth,     kInkMaxTree);
        if (d_tree_anc32) {
            std::memcpy(h_bytes + 512, tree.anc32, kInkMaxTree * kInkMaxTree);
        }

        INK_CK(cudaMemcpyAsync(d_tok_ids,   h_tok,         (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos,       h_pos,         (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_sconv_lag, h_bytes,       kInkMaxTree * 3, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tree_anc,  h_bytes + 128, kInkMaxTree * kInkMaxTreeDepth, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tree_dep,  h_bytes + 384, kInkMaxTree, cudaMemcpyHostToDevice, st));
        if (d_tree_anc32) {
            INK_CK(cudaMemcpyAsync(d_tree_anc32, h_bytes + 512, kInkMaxTree * kInkMaxTree, cudaMemcpyHostToDevice, st));
        }

        if (use_cuda_graph && !m.ls_active && max_passes_override == 0 && slot0 == 0 && M <= tree_cap && m.n_host_experts == 0 && !elastic_kv.needs_eager_for_pos(cur_max_pos)) {
            GraphKey key{M, 2, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(M, 2, slot0, 0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = pdl::instantiate(&exec, gr, "decode");
                        cudaGraphDestroy(gr);
                        gr = nullptr;   // destroyed: the failure branch below must not destroy it a second time (cudaErrorInvalidValue left pending, read by the next cudaGetLastError as a fatal "invalid argument")
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    (void)cudaGetLastError();   // the eager fallback starts with a clean error state
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] Tree CUDA Graph capture unavailable (err=%d): falling back to eager mode\n", (int)ge);
                    emit_48_visits(M, 2, slot0, p0, want_taps);
                    return;
                }
                if (m.all_sf4_vram && graph_cache.size() >= 8) {
                    if (graph_cache.front().exec) cudaGraphExecDestroy(graph_cache.front().exec);
                    graph_cache.erase(graph_cache.begin());
                }
                graph_cache.push_back({key, exec});
            }
            cudaError_t le = cudaGraphLaunch(exec, st);
            if (le != cudaSuccess) {
                cudaGetLastError();
                for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
                graph_cache.clear();
                use_cuda_graph = false;
                emit_48_visits(M, 2, slot0, p0, want_taps);
                dbg_tree_print(tree, p0);
                return;
            }
            dbg_tree_print(tree, p0);
            return;
        }
        // Opt-in RTX 5060 Ti 42/43-Visit Prefix+Core CUDA Graph even when Coda layers live in Host RAM!
        if (ink_rtx5060ti_cfg(m.primary_gpu).core_graph && use_cuda_graph && !m.ls_active && max_passes_override == 0 &&
            slot0 == 0 && M <= tree_cap && m.n_host_experts > 0 && !m.hybrid.tier_on && !elastic_kv.needs_eager_for_pos(cur_max_pos)) {
            const int fhc = first_host_coda_layer();
            if (fhc >= m.man.c.core_end && fhc < m.man.c.n_phys) {
                GraphKey key{M, 200 + fhc, want_taps ? 1 : 0, (int)kv_dtype};
                cudaGraphExec_t exec = nullptr;
                for (const auto& e : prefix_core_graph_cache) if (e.key == key) { exec = e.exec; break; }
                if (!exec) {
                    cudaGraph_t gr = nullptr;
                    cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                    if (ge == cudaSuccess) {
                        emit_visits_prefix_to(fhc, M, 2, slot0, 0, want_taps);
                        ge = cudaStreamEndCapture(st, &gr);
                        if (ge == cudaSuccess && gr) {
                            ge = pdl::instantiate(&exec, gr, "decode");
                            cudaGraphDestroy(gr);
                            gr = nullptr;   // destroyed: the failure branch below must not destroy it a second time (cudaErrorInvalidValue left pending, read by the next cudaGetLastError as a fatal "invalid argument")
                        }
                    }
                    if (ge == cudaSuccess && exec) {
                        prefix_core_graph_cache.push_back({key, exec});
                    } else {
                        (void)cudaGetLastError();
                        if (gr) cudaGraphDestroy(gr);
                        (void)cudaGetLastError();   // the eager fallback starts with a clean error state
                        exec = nullptr;
                    }
                }
                if (exec) {
                    INK_CK(cudaGraphLaunch(exec, st));
                    emit_visits_coda_from(fhc, M, 2, slot0, p0, want_taps);
                    dbg_tree_print(tree, p0);
                    return;
                }
            }
        }
        emit_48_visits(M, 2, slot0, p0, want_taps);
        if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_after(M, m.d_sec4, m.d_experts, st);   // the hot tier learns from verify steps too
        dbg_tree_print(tree, p0);
    }
    void dbg_tree_print(const InkTreeSpec& tree, int p0) {
        if (!dbg_hash_on) return;
        int32_t pos[kInkMaxTree], dep[kInkMaxTree], par[kInkMaxTree];
        for (int i = 0; i < tree.M && i < kInkMaxTree; ++i) { pos[i] = p0 + (int)tree.depth[i]; dep[i] = (int)tree.depth[i]; par[i] = (int)tree.parent[i]; }
        dbg_hash_print("tree", tree.M, pos, dep, tree.tok, par, p0);
    }

    // Batched multi-slot 1-token-per-slot step (B = 1..64 concurrent requests, with safe eager fallback!)
    void forward_batched(const int32_t* tok_ids, const int32_t* slot_ids, const int32_t* pos_arr, int B, bool want_taps = false) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        {
            int mp = 0;
            for (int i = 0; i < B; ++i) mp = std::max(mp, pos_arr[i]);
            cur_max_pos = mp;
        }
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)B * 4);
        std::memcpy(slot_buf + B, slot_ids, (size_t)B * 4);
        std::memcpy(slot_buf + 2 * B, pos_arr, (size_t)B * 4);
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_slots, slot_buf + B, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, slot_buf + 2 * B, (size_t)B * 4, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && !swa_in_host && !m.ls_active && B <= 64 && (m.n_host_experts == 0 || (B == 1 && !m.hybrid.needs_eager())) && !elastic_kv.needs_eager_for_pos(cur_max_pos)) {
            GraphKey key{B, 10, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(B, 1, 0, 0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = pdl::instantiate(&exec, gr, "decode");
                        cudaGraphDestroy(gr);
                        gr = nullptr;   // destroyed: the failure branch below must not destroy it a second time (cudaErrorInvalidValue left pending, read by the next cudaGetLastError as a fatal "invalid argument")
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    (void)cudaGetLastError();   // the eager fallback starts with a clean error state
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] Batched CUDA Graph capture unavailable (err=%d): falling back to eager mode\n", (int)ge);
                    emit_48_visits(B, 1, 0, 0, want_taps);
                    return;
                }
                graph_cache.push_back({key, exec});
            }
            cudaError_t le = cudaGraphLaunch(exec, st);
            if (le != cudaSuccess) {
                cudaGetLastError();
                for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
                graph_cache.clear();
                use_cuda_graph = false;
                emit_48_visits(B, 1, 0, 0, want_taps);
            }
            if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_after(B, m.d_sec4, m.d_experts, st);
            return;
        }
        emit_48_visits(B, 1, 0, 0, want_taps);
        if (m.hybrid.active && m.n_host_experts > 0) m.hybrid.step_after(B, m.d_sec4, m.d_experts, st);
        if (prof_enabled) {   // STRATA_INK_PROF=1: the batched decode step, every 50 steps
            prof_flush();
            if (++prof_steps % 50 == 0) {
                char tag[32];
                std::snprintf(tag, sizeof(tag), "DECODE-B%d", B);
                prof_print(tag, (int)prof_steps);
                prof_steps = 0;
                prof_reset();
            }
        }
    }

    // Mode-3 Batched Multi-Slot Linear Speculative Verify (B slots x K_slot tokens = M <= 256, CUDA-Graph captured!)
    void forward_batched_spec(const int32_t* tok_ids, const int32_t* slot_ids, const int32_t* pos_arr, int B, int K_slot, bool want_taps = true) {
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        const int M = B * K_slot;
        bspec_k_slot = K_slot;
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf, tok_ids, (size_t)M * 4);
        std::memcpy(slot_buf + M, slot_ids, (size_t)M * 4);
        std::memcpy(slot_buf + 2 * M, pos_arr, (size_t)M * 4);
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf, (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_slots, slot_buf + M, (size_t)M * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos, slot_buf + 2 * M, (size_t)M * 4, cudaMemcpyHostToDevice, st));

        if (use_cuda_graph && !m.ls_active && M <= bspec_m_cap && m.n_host_experts == 0) {
            GraphKey key{B * 8 + K_slot, 30, want_taps ? 1 : 0, (int)kv_dtype};
            cudaGraphExec_t exec = nullptr;
            for (const auto& e : graph_cache) if (e.key == key) { exec = e.exec; break; }
            if (!exec) {
                cudaGraph_t gr = nullptr;
                cudaError_t ge = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                if (ge == cudaSuccess) {
                    emit_48_visits(M, 3, 0, 0, want_taps);
                    ge = cudaStreamEndCapture(st, &gr);
                    if (ge == cudaSuccess && gr) {
                        ge = pdl::instantiate(&exec, gr, "decode");
                        cudaGraphDestroy(gr);
                        gr = nullptr;   // destroyed: the failure branch below must not destroy it a second time (cudaErrorInvalidValue left pending, read by the next cudaGetLastError as a fatal "invalid argument")
                    }
                }
                if (ge != cudaSuccess || !exec) {
                    cudaGetLastError();
                    if (gr) cudaGraphDestroy(gr);
                    (void)cudaGetLastError();   // the eager fallback starts with a clean error state
                    use_cuda_graph = false;
                    std::fprintf(stderr, "[InkCUDA] Mode-3 Batched Spec Graph unavailable (err=%d): falling back to eager\n", (int)ge);
                    emit_48_visits(M, 3, 0, 0, want_taps);
                    return;
                }
                graph_cache.push_back({key, exec});
            }
            INK_CK(cudaGraphLaunch(exec, st));
            return;
        }
        emit_48_visits(M, 3, 0, 0, want_taps);
    }

    void commit_bspec(const int32_t* slots_b, const int32_t* p0_b, const int32_t* n_acc_b, int B, int K_slot) {
        int32_t* slot_buf = h_meta_pin + (meta_ring_idx++ & 63ull) * meta_ring_stride;
        std::memcpy(slot_buf,         slots_b, (size_t)B * 4);
        std::memcpy(slot_buf + B,     p0_b,    (size_t)B * 4);
        std::memcpy(slot_buf + 2 * B, n_acc_b, (size_t)B * 4);
        INK_CK(cudaMemcpyAsync(d_slots,   slot_buf,         (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_pos,     slot_buf + B,     (size_t)B * 4, cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_tok_ids, slot_buf + 2 * B, (size_t)B * 4, cudaMemcpyHostToDevice, st));
        rtx5090::k_ink_commit_sconv_bspec<<<dim3(kInkVisits, B), 256, 0, st>>>(
            d_sconv_pool_arr, d_bspec_sconv, d_slots, d_tok_ids, B, K_slot, max_slots, bspec_m_cap);
        rtx5090::k_ink_commit_kv_bspec<<<dim3(kInkVisits * kInkKVHeads, B), 128, 0, st>>>(
            d_kv_desc_arr, d_bspec_k, d_bspec_v, d_slots, d_pos, d_tok_ids, B, K_slot, bspec_m_cap);
    }

    // Fast 1-pass exact argmax for Q8_ROW unembed:
    //   M <= 8: single-pass dp4a scan (~1.0 ms @ 448 GB/s GDDR7 peak, ZERO 32x cuBLAS launch overhead!)
    //   M >  8: vocab-tiled INT8 Tensor Core (~2.2-3.8 ms for M=64..320!)
    bool head_argmax_q8row_tc(int M, int32_t* h_out_ids) {
        if (!m.unembed_screen.ready || m.unembed_screen.V_pad <= 0 || (m.unembed_screen.V_pad % 4) != 0) return false;
        const int V     = m.man.c.vocab_unpadded;
        const int V_pad = m.unembed_screen.V_pad;
        int8_t* x_i8 = reinterpret_cast<int8_t*>(d_x_bf16);
        float*  x_sc = d_ctrl_gain;
        k_ink_quant_x_row_i8<<<M, 256, 0, st>>>(d_h_norm, x_i8, x_sc, M, kInkHidden);
        if (M <= 8) {
            constexpr int kGrid = 256;
            float*   blk_v = d_ctrl_forget;                                       // [max_batch * 2304] >> 8 * 256
            int32_t* blk_i = reinterpret_cast<int32_t*>(d_ctrl_forget + 8 * kGrid);
            if (M == 1)      k_ink_q8row_argmax_smallm_p1<1><<<kGrid, 256, 0, st>>>(m.unembed_screen.q8, m.unembed_screen.sc, x_i8, M, V, blk_v, blk_i);
            else if (M <= 4) k_ink_q8row_argmax_smallm_p1<4><<<kGrid, 256, 0, st>>>(m.unembed_screen.q8, m.unembed_screen.sc, x_i8, M, V, blk_v, blk_i);
            else             k_ink_q8row_argmax_smallm_p1<8><<<kGrid, 256, 0, st>>>(m.unembed_screen.q8, m.unembed_screen.sc, x_i8, M, V, blk_v, blk_i);
            k_ink_q8row_argmax_smallm_p2<<<M, 256, 0, st>>>(blk_v, blk_i, kGrid, d_argmax);
            INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            return true;
        }
        const int M_pad = ((M + 3) / 4) * 4;
        const int n_tiles = (M_pad >= 64 && m.fp8_trunk.ready) ? 12 : 32;
        const int vt_nom = ((V_pad + n_tiles - 1) / n_tiles + 3) & ~3;
        if (M_pad > q8_argmax_cap_M || vt_nom > q8_argmax_vt_nom) {
            if (d_q8_tile_i32) { cudaFree(d_q8_tile_i32); d_q8_tile_i32 = nullptr; }
            if (d_q8_tile_val) { cudaFree(d_q8_tile_val); d_q8_tile_val = nullptr; }
            if (d_q8_tile_id)  { cudaFree(d_q8_tile_id);  d_q8_tile_id  = nullptr; }
            q8_argmax_cap_M  = std::max(M_pad, 32);
            q8_argmax_vt_nom = vt_nom;
            d_q8_tile_i32 = dalloc<int32_t>((size_t)q8_argmax_cap_M * q8_argmax_vt_nom);
            d_q8_tile_val = dalloc<float>((size_t)q8_argmax_cap_M * n_tiles);
            d_q8_tile_id  = dalloc<int32_t>((size_t)q8_argmax_cap_M * n_tiles);
        }
        if (M_pad > M) {
            INK_CK(cudaMemsetAsync(x_i8 + (size_t)M * kInkHidden, 0, (size_t)(M_pad - M) * kInkHidden, st));
        }
        const int32_t alpha_i = 1, beta_i = 0;
        for (int ti = 0; ti < n_tiles; ++ti) {
            const int v0 = ti * vt_nom;
            const int vt = std::max(0, std::min(vt_nom, V_pad - v0));
            if (vt > 0) {
                INK_CUBLAS_CK(cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    vt, M_pad, kInkHidden, &alpha_i,
                    m.unembed_screen.q8 + (size_t)v0 * kInkHidden, CUDA_R_8I, kInkHidden,
                    x_i8, CUDA_R_8I, kInkHidden, &beta_i,
                    d_q8_tile_i32, CUDA_R_32I, vt,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            k_ink_q8row_tile_argmax_i32<<<M, 256, 0, st>>>(
                d_q8_tile_i32, m.unembed_screen.sc, v0, vt, ti, V, n_tiles, d_q8_tile_val, d_q8_tile_id);
        }
        k_ink_q8row_final_argmax<<<M, 32, 0, st>>>(d_q8_tile_val, d_q8_tile_id, n_tiles, d_argmax);
        INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
        INK_CK(cudaStreamSynchronize(st));
        return true;
    }

    // ---- Unembed + Softcap(30.0) + Argmax ----
    // Fast exact 2-stage screened argmax (0.14 ms on RTX 5090, 100.000% mathematically bit-exact up to M <= 32!)
    void head_argmax_exact(int M, int32_t* h_out_ids) {
        prof_mark_start();
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        // Opt-in RTX 5090 fast exact FMT_Q8ROW argmax (SQ-20 unembed is in-place Q8_ROW -> 0 extra VRAM, 0.28 ms!)
        if (ui.fmt == FMT_Q8ROW && rtx5090::dspark_opt_on() && !rtx5090::exact_mode_on() &&
            M <= screen_cap && !m.ls_active) {
            constexpr int kEvalK = 32;
            const float*  w_sc = reinterpret_cast<const float*>(m.d_trunk + ui.off);
            const int8_t* w_i8 = reinterpret_cast<const int8_t*>(m.d_trunk + ui.off + (size_t)ui.rows() * sizeof(float));
            int8_t*  h_q8 = reinterpret_cast<int8_t*>(d_x_bf16);
            float*   h_sc = d_h_norm2;
            int32_t* d_i32 = reinterpret_cast<int32_t*>(d_dot_hat);
            int32_t* tmp_i = reinterpret_cast<int32_t*>(d_ctrl_cat);
            float*   tmp_v = d_ctrl_cat + (size_t)M * 512;
            rtx5090::k_ms_quant_h_q8<<<M, 256, 0, st>>>(d_h_norm, h_q8, h_sc, M);
            const int32_t alpha_i = 1, beta_i = 0;
            INK_CUBLAS_CK(cublasGemmEx(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                rtx5090::kBspecV8, M, kInkHidden, &alpha_i,
                w_i8, CUDA_R_8I, kInkHidden,
                h_q8, CUDA_R_8I, kInkHidden,
                &beta_i, d_i32, CUDA_R_32I, rtx5090::kBspecV8,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            rtx5090::k_ms_top_i32_s1<<<dim3(64, M), 256, 0, st>>>(d_i32, w_sc, h_sc, rtx5090::kBspecV8, tmp_i, tmp_v, M);
            rtx5090::k_ms_top_i32_s2<<<M, 256, 0, st>>>(tmp_i, tmp_v, kEvalK, d_cand_ids, d_cand_cnt, m.man.c.vocab_unpadded, rtx5090::kBspecV8, M);
            rtx5090::k_ink_screen_eval_q8row<<<M, 256, 0, st>>>(
                m.d_trunk + ui.off, (int)ui.rows(), d_h_norm, d_cand_ids, d_cand_cnt, kEvalK, d_argmax, M);
            prof_mark_end(7);
            INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            return;
        }
        if (ui.fmt == FMT_Q8ROW && head_argmax_q8row_tc(M, h_out_ids)) {
            prof_mark_end(7);
            return;
        }
        if (m.unembed_screen.ready && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K) && M <= kInkMaxTree) {
            const int V = m.man.c.vocab_unpadded;
            const int g1 = (V + 7) / 8;
            for (int m0 = 0; m0 < M; m0 += screen_cap) {
                const int mc = std::min(screen_cap, M - m0);
                const float* hm = d_h_norm + (size_t)m0 * kInkHidden;
                k_ink_row_l2norm<<<mc, 256, 0, st>>>(hm, d_h_norm2, d_max_lb, d_cand_cnt, mc);
                if (mc == 1) {
                    k_ink_screen_pass1_multiM<1><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else if (mc <= 4) {
                    k_ink_screen_pass1_multiM<4><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else if (mc <= 8) {
                    k_ink_screen_pass1_multiM<8><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else if (mc <= 16) {
                    rtx5090::k_ink_screen_pass1_multiM_rtx<16><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else {
                    rtx5090::k_ink_screen_pass1_multiM_rtx<24><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                }
                dim3 g2((V + 255) / 256, mc);
                k_ink_screen_pass2_bf16<<<g2, 256, 0, st>>>(
                    nullptr, m.unembed_screen.eps,
                    hm, d_h_norm2, d_dot_hat, d_max_lb, d_cand_ids, d_cand_cnt, 4096, mc, V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_screen_eval_bf16<<<mc, 256, 0, st>>>(
                        (const uint16_t*)(m.d_trunk + ui.off), hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                } else {
                    k_ink_screen_eval_s6k<<<mc, 256, 0, st>>>(
                        m.d_trunk + ui.off, hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                }
            }
            prof_mark_end(7);
            INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            return;
        }
        head_logits_full(M);
        k_ink_argmax_rows<<<M, 1024, 0, st>>>(d_logits, m.man.c.vocab_unpadded, d_argmax);
        prof_mark_end(7);
        INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
        INK_CK(cudaStreamSynchronize(st));
    }

    // Opt-in Batched Exact Screened Argmax for M up to max_batch (e.g. C64 x 8 = 512 rows, ZERO 400 MiB d_logits!)
    void head_argmax_exact_batch(int M, int32_t* h_out_ids) {
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        if (ui.fmt == FMT_Q8ROW && head_argmax_q8row_tc(M, h_out_ids)) return;
        if (m.unembed_screen.ready && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K) && M <= max_batch) {
            const int V = m.man.c.vocab_unpadded;
            const int g1 = (V + 7) / 8;
            for (int m0 = 0; m0 < M; m0 += 8) {
                const int mc = std::min(8, M - m0);
                const float* hm = d_h_norm + (size_t)m0 * kInkHidden;
                k_ink_row_l2norm<<<mc, 256, 0, st>>>(hm, d_h_norm2, d_max_lb, d_cand_cnt, mc);
                if (mc == 1) {
                    k_ink_screen_pass1_multiM<1><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else if (mc <= 4) {
                    k_ink_screen_pass1_multiM<4><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                } else {
                    k_ink_screen_pass1_multiM<8><<<g1, 256, 0, st>>>(
                        m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps,
                        hm, d_h_norm2, d_dot_hat, d_max_lb, mc, V);
                }
                dim3 g2((V + 255) / 256, mc);
                k_ink_screen_pass2_bf16<<<g2, 256, 0, st>>>(
                    nullptr, m.unembed_screen.eps,
                    hm, d_h_norm2, d_dot_hat, d_max_lb, d_cand_ids, d_cand_cnt, 4096, mc, V);
                if (ui.fmt == FMT_BF16) {
                    k_ink_screen_eval_bf16<<<mc, 256, 0, st>>>(
                        (const uint16_t*)(m.d_trunk + ui.off), hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                } else {
                    k_ink_screen_eval_s6k<<<mc, 256, 0, st>>>(
                        m.d_trunk + ui.off, hm, d_cand_ids, d_cand_cnt, 4096, d_argmax + m0, mc);
                }
            }
            INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            return;
        }
        // Chunked fallback (32 rows at a time -> max 25 MiB d_logits!)
        for (int m0 = 0; m0 < M; m0 += 32) {
            const int mc = std::min(32, M - m0);
            head_logits_from(d_h_norm + (size_t)m0 * kInkHidden, mc);
            k_ink_argmax_rows<<<mc, 1024, 0, st>>>(d_logits, m.man.c.vocab_unpadded, d_argmax + m0);
        }
        INK_CK(cudaMemcpyAsync(h_out_ids, d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, st));
        INK_CK(cudaStreamSynchronize(st));
    }

    // Full exact softcapped logits [n_want, 200058] for selected hidden rows in d_h_src -> d_logits
    // PERF-STEP: the exact top-128 keys of ONE row through the INT8 screening index (ink_head_screen.cuh); false = run the full head instead (no index, not BF16, or the guard fired)
    HeadScreen hs;
    bool head_screen_cands(const float* d_h, InkTopkScratch& tk) {
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        if (ui.fmt != FMT_BF16 || !m.unembed_screen.ready || m.unembed_screen.borrowed || m.unembed_screen.eps == nullptr) return false;
        const int V = m.man.c.vocab_unpadded;
        if (!hs.ensure(V) || !tk.ensure(1)) return false;
        const uint16_t* W = reinterpret_cast<const uint16_t*>(m.trunk_ptr("model.llm.unembed.weight"));
        k_ink_row_l2norm<<<1, 256, 0, st>>>(d_h, hs.d_hn, hs.d_misc + 6, hs.d_misc + 7, 1);
        k_ink_hs_pass1<<<(V + 7) / 8, 256, 0, st>>>(m.unembed_screen.q8, m.unembed_screen.sc, m.unembed_screen.eps, d_h, hs.d_hn, hs.d_dot, hs.d_hist, V);
        k_ink_hs_threshold<<<1, 1024, 0, st>>>(hs.d_hist, hs.d_T, hs.d_misc + 0, hs.d_misc + 1, hs.d_misc + 2);
        k_ink_hs_cand<<<(V + 255) / 256, 256, 0, st>>>(hs.d_dot, m.unembed_screen.eps, hs.d_hn, hs.d_T, hs.d_cand, hs.d_misc + 0, hs.d_misc + 2, V);
        k_ink_hs_exact<<<64, 128, 0, st>>>(W, d_h, hs.d_cand, hs.d_misc + 0, hs.d_keys, m.man.c.logit_softcap);
        k_ink_hs_sort<<<1, kTopkThreads, 0, st>>>(hs.d_keys, hs.d_misc + 0, hs.d_misc + 1, tk.d_out);
        INK_CK(cudaMemcpyAsync(tk.h_out, tk.d_out, (size_t)kTopkCand * 8, cudaMemcpyDeviceToHost, st));
        INK_CK(cudaMemcpyAsync(hs.h_misc, hs.d_misc, 3 * sizeof(int), cudaMemcpyDeviceToHost, st));
        INK_CK(cudaStreamSynchronize(st));
        ++hs.calls;
        hs.cand_sum += hs.h_misc[0];
        if (hs.h_misc[1] != 0) { ++hs.fallbacks; return false; }
        tk.h_fin[0] = hs.h_misc[2];
        if (hs.calls % 500 == 1) std::fprintf(stderr, "[InkHeadScreen] %ld steps, %ld fell back to the full head, mean %.0f candidates\n", hs.calls, hs.fallbacks, (double)hs.cand_sum / (double)hs.calls);
        return true;
    }

    void head_logits_from(const float* d_h_src, int n_want) {
        ensure_logits(n_want);
        const int V = m.man.c.vocab_unpadded;
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        if (ui.fmt == FMT_Q8ROW && m.unembed_screen.ready) {
            int8_t* x_i8 = reinterpret_cast<int8_t*>(d_x_bf16);
            float*  x_sc = d_ctrl_gain;
            k_ink_quant_x_row_i8<<<n_want, 256, 0, st>>>(d_h_src, x_i8, x_sc, n_want, kInkHidden);
            const int grid = (V + 7) / 8;
            k_ink_q8row_unembed_dp4a<<<grid, 256, 0, st>>>(
                m.unembed_screen.q8, m.unembed_screen.sc, x_i8, x_sc, d_logits, n_want, V, kInkHidden);
        } else {
            linear("model.llm.unembed.weight", d_h_src, d_logits, n_want, V);
        }
        const size_t n = (size_t)n_want * V;
        k_ink_softcap<<<(n + 255) / 256, 256, 0, st>>>(d_logits, n, m.man.c.logit_softcap);
    }

    // Full exact softcapped logits [M, 200058] into d_logits
    void head_logits_full(int M) {
        head_logits_from(d_h_norm, M);
    }

    void free_all() {
        hs.free_all();
        fp8_sm.free_all();
        fp8_eng.free_all();
        elastic_kv.free_all();
        opt5060_pf.free_all();
        opt5060_macro.free_all();
        c64_kv_pf.free_all();
        for (auto& e : prefix_core_graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        prefix_core_graph_cache.clear();
        if (d_mc_sconv_stage) {
            if (d_mc_sconv_is_vram) cudaFree(d_mc_sconv_stage);
            else if (h_mc_sconv_raw) cudaFreeHost(h_mc_sconv_raw);
            h_mc_sconv_raw = nullptr; d_mc_sconv_stage = nullptr; d_mc_sconv_is_vram = false; mc_stage_cap_k = 0;
        }
        if (d_mc_slot_ids)  { cudaFree(d_mc_slot_ids);  d_mc_slot_ids = nullptr; }
        if (d_mc_base_pos)  { cudaFree(d_mc_base_pos);  d_mc_base_pos = nullptr; }
        if (d_mc_n_acc)     { cudaFree(d_mc_n_acc);     d_mc_n_acc = nullptr; }
        if (d_q8_tile_i32)  { cudaFree(d_q8_tile_i32);  d_q8_tile_i32 = nullptr; }
        if (d_q8_tile_val)  { cudaFree(d_q8_tile_val);  d_q8_tile_val = nullptr; }
        if (d_q8_tile_id)   { cudaFree(d_q8_tile_id);   d_q8_tile_id  = nullptr; }
        q8_argmax_cap_M = 0; q8_argmax_vt_nom = 0;
        if (m.ls_active && cur_stage != 0) bind_stage(0);
        for (auto& e : graph_cache) if (e.exec) cudaGraphExecDestroy(e.exec);
        graph_cache.clear();
        if (m.ls_active) {
            auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            if (ls_stages[0].ev_done) { cudaEventDestroy(ls_stages[0].ev_done); ls_stages[0].ev_done = nullptr; }
            for (int s = 1; s < m.n_stages; ++s) {
                cudaSetDevice(m.stage_gpu[s]);
                StageExec& se = ls_stages[s];
                se.ws.free_all();
                fr(se.d_cur); fr(se.d_res); fr(se.d_bank_out);
                for (int i = 0; i < 8; ++i) fr(se.d_blocks[i]);
                for (int i = 0; i < 4; ++i) fr(se.d_bank[i]);
                fr(se.d_mixed); fr(se.d_msgs); fr(se.d_inp0); fr(se.d_inp1); fr(se.d_inp2);
                fr(se.d_q); fr(se.d_k_raw); fr(se.d_v_raw); fr(se.d_k); fr(se.d_v); fr(se.d_r);
                fr(se.d_k_dyn); fr(se.d_v_dyn); fr(se.d_sinks); fr(se.d_gate_attn); fr(se.d_rel_table);
                fr(se.d_attn_out); fr(se.d_wo_raw); fr(se.d_wo_conv); fr(se.d_moe_in); fr(se.d_moe_raw); fr(se.d_moe_conv);
                fr(se.d_ctrl_cat); fr(se.d_ctrl_gate); fr(se.d_ctrl_gain); fr(se.d_ctrl_forget);
                fr(se.d_attn_pm); fr(se.d_attn_pl); fr(se.d_attn_po);
                fr(se.d_pf_q); fr(se.d_pf_k); fr(se.d_pf_v); fr(se.d_pf_p);
                fr(se.d_pf_s); fr(se.d_pf_o); fr(se.d_pf_ot); fr(se.d_pf_m); fr(se.d_pf_l);
                fr(se.d_x_bf16); fr(se.d_w_scratch); fr(se.d_slots); fr(se.d_pos);
                if (se.ev_done) { cudaEventDestroy(se.ev_done); se.ev_done = nullptr; }
                if (se.cublas) { cublasDestroy(se.cublas); se.cublas = nullptr; }
                if (se.st) { cudaStreamDestroy(se.st); se.st = nullptr; }
            }
            cudaSetDevice(m.primary_gpu);
            if (h_ls_bounce) { cudaFreeHost(h_ls_bounce); h_ls_bounce = nullptr; }
            if (h_ls_meta_bounce) { cudaFreeHost(h_ls_meta_bounce); h_ls_meta_bounce = nullptr; }
        }
        if (m.ep2_active && m.peer_gpu >= 0) {
            cudaSetDevice(m.peer_gpu);
            peer_ws.free_all();
            if (d_peer_x) { cudaFree(d_peer_x); d_peer_x = nullptr; }
            if (d_peer_y) { cudaFree(d_peer_y); d_peer_y = nullptr; }
            if (ev_peer_done) { cudaEventDestroy(ev_peer_done); ev_peer_done = nullptr; }
            if (peer_st) { cudaStreamDestroy(peer_st); peer_st = nullptr; }
            cudaSetDevice(m.primary_gpu);
        }
        if (d_peer_y_rx) { cudaFree(d_peer_y_rx); d_peer_y_rx = nullptr; }
        if (ev_moe_ready) { cudaEventDestroy(ev_moe_ready); ev_moe_ready = nullptr; }
        if (h_ep_x)   { cudaFreeHost(h_ep_x); h_ep_x = nullptr; }
        if (h_ep_idx) { cudaFreeHost(h_ep_idx); h_ep_idx = nullptr; }
        if (h_ep_w)   { cudaFreeHost(h_ep_w); h_ep_w = nullptr; }
        if (h_ep_y)   { cudaFreeHost(h_ep_y); h_ep_y = nullptr; }
        moe_ws.free_all();
        const bool had_kv_slab = (d_kv_slab != nullptr);
        if (d_kv_slab) { cudaFree(d_kv_slab); d_kv_slab = nullptr; }
        for (int v = 0; v < kInkVisits; ++v) {
            const int ip = ink_visit_phys(v);
            const int dev = m.ls_active ? m.stage_gpu[m.layer_owner[ip]] : m.primary_gpu;
            cudaSetDevice(dev);
            if (had_kv_slab || (kv_desc[v].is_swa && swa_in_host)) {
                kv_desc[v].k_f16 = nullptr; kv_desc[v].v_f16 = nullptr;
                kv_desc[v].k_f8  = nullptr; kv_desc[v].v_f8  = nullptr;
                kv_desc[v].k_sc  = nullptr; kv_desc[v].v_sc  = nullptr;
            } else {
                if (kv_desc[v].k_f16) { cudaFree(kv_desc[v].k_f16); kv_desc[v].k_f16 = nullptr; }
                if (kv_desc[v].v_f16) { cudaFree(kv_desc[v].v_f16); kv_desc[v].v_f16 = nullptr; }
                if (kv_desc[v].k_f8)  { cudaFree(kv_desc[v].k_f8);  kv_desc[v].k_f8 = nullptr; }
                if (kv_desc[v].v_f8)  { cudaFree(kv_desc[v].v_f8);  kv_desc[v].v_f8 = nullptr; }
                if (kv_desc[v].k_sc)  { cudaFree(kv_desc[v].k_sc);  kv_desc[v].k_sc = nullptr; }
                if (kv_desc[v].v_sc)  { cudaFree(kv_desc[v].v_sc);  kv_desc[v].v_sc = nullptr; }
            }
            free_host_kv_visit(v);
            if (had_kv_slab) d_sconv_pool[v] = nullptr;
            else if (d_sconv_pool[v])  { cudaFree(d_sconv_pool[v]);  d_sconv_pool[v] = nullptr; }
        }
        cudaSetDevice(m.primary_gpu);
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        const bool had_act_slab = (d_act_slab != nullptr);
        if (d_act_slab) { cudaFree(d_act_slab); d_act_slab = nullptr; }
        auto fr_act = [&](auto*& p) { if (had_act_slab) p = nullptr; else fr(p); };
        fr_act(d_cur); fr_act(d_res); fr_act(d_prelude); fr_act(d_bank_out);
        for (int i = 0; i < 8; ++i) fr_act(d_blocks[i]);
        for (int i = 0; i < 2; ++i) fr_act(d_prelude_blocks[i]);
        for (int i = 0; i < 4; ++i) fr_act(d_bank[i]);
        fr_act(d_mixed); fr_act(d_msgs); fr_act(d_inp0); fr_act(d_inp1); fr_act(d_inp2);
        fr_act(d_q); fr_act(d_k_raw); fr_act(d_v_raw); fr_act(d_k); fr_act(d_v); fr_act(d_r);
        fr_act(d_k_dyn); fr_act(d_v_dyn); fr_act(d_sinks); fr_act(d_gate_attn); fr_act(d_rel_table);
        fr_act(d_attn_out); fr_act(d_wo_raw); fr_act(d_wo_conv); fr_act(d_moe_in); fr_act(d_moe_raw); fr_act(d_moe_conv);
        fr_act(d_h_norm);
        for (int i = 0; i < 4; ++i) { fr_act(d_taps[i]); fr_act(d_taps_compact[i]); }
        fr_act(d_ctrl_cat); fr_act(d_ctrl_gate); fr_act(d_ctrl_gain); fr_act(d_ctrl_forget);
        fr_act(d_attn_pm); fr_act(d_attn_pl); fr_act(d_attn_po);
        if (d_pf_slab) {
            cudaFree(d_pf_slab); d_pf_slab = nullptr; d_pf_slab_bytes = 0;
            d_pf_q = d_pf_k = d_pf_v = d_pf_p = nullptr;
            d_pf_s = d_pf_o = d_pf_ot = d_pf_m = d_pf_l = nullptr;
        } else {
            fr(d_pf_q); fr(d_pf_k); fr(d_pf_v); fr(d_pf_p); fr(d_pf_s); fr(d_pf_o); fr(d_pf_ot); fr(d_pf_m); fr(d_pf_l);
        }
        fr_act(d_stage_k); fr_act(d_stage_v); fr_act(d_stage_sconv);
        fr_act(d_kv_desc_arr); fr_act(d_sconv_pool_arr); fr_act(d_acc_path);
        fr_act(d_x_bf16); fr_act(d_w_scratch); fr(d_logits); fr_act(d_xq8); fr_act(d_xs8);
        fr_act(d_dot_hat); fr_act(d_h_norm2); fr_act(d_max_lb); fr_act(d_cand_ids); fr_act(d_cand_cnt); fr_act(d_argmax);
        fr_act(d_tok_ids); fr_act(d_slots); fr_act(d_pos); fr_act(d_is_img); fr_act(d_img_emb); fr_act(d_sconv_lag); fr_act(d_tree_anc); fr_act(d_tree_dep);
        fr(d_rtx_qkv_pool); fr(d_rtx_ctrl_pool); fr(d_rtx_rel_proj_pool); fr(d_tree_anc32);
        if (h_meta_pin) { cudaFreeHost(h_meta_pin); h_meta_pin = nullptr; }
        if (ev_p0) { cudaEventDestroy(ev_p0); ev_p0 = nullptr; }
        if (ev_p1) { cudaEventDestroy(ev_p1); ev_p1 = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (d_cublas_ws) { cudaFree(d_cublas_ws); d_cublas_ws = nullptr; }
        if (st) { cudaStreamDestroy(st); st = nullptr; }
    }
};

}  // namespace strata::ink
