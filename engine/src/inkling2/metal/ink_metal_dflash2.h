// engine/src/inkling2/metal/ink_metal_dflash2.h - Native Apple Silicon Metal DFlash 2 Drafter
// Exact host driver for ink_metal_dflash2.metal (single-CB Block-8 + Q8_ROW weights +
// single-pass Q8 unembed Top-16 screen + rank-256 bilinear selector + ceiling.py MoE adaptive M + PLD).
// Works on BOTH .strata (SQv2-20) and .gguf (GSQ-RCO-IQ3_S, UD-Q4_K_XL).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink_metal_common.h"
#include "../ink_dflash2_mix.h"
#include "../cuda/ink_dflash2_path.cuh"
#include "../../image/vae/safetensors_lite.hpp"

namespace ink_metal {

struct InkMetalDFlash2 {
    static constexpr int kHidden  = 2304;
    static constexpr int kTargets = 4;
    static constexpr int kLayers  = 5;
    static constexpr int kQHeads  = 32;
    static constexpr int kKvHeads = 8;
    static constexpr int kHeadDim = 128;
    static constexpr int kQ       = 4096;
    static constexpr int kKV      = 1024;
    static constexpr int kKV2     = 2048;
    static constexpr int kFF      = 6912;
    static constexpr int kDyn     = 576;
    static constexpr int kRank    = 256;
    static constexpr int kBlock   = 8;
    static constexpr int kDraft   = 7;
    static constexpr int kWin     = 2048;
    static constexpr int kRingCap = 2112;
    static constexpr int kTopK    = 16;
    static constexpr int kMaskId  = 200056;
    static constexpr float kTheta = 10000.0f;

    struct Df2AttnArgs {
        int slot;
        int pos0;
        int ctx_lo;
        int ctx_hi;
        float scale;
        int pad;
    };

    mtl::Context* cx = nullptr;
    InkMetalModel* mod = nullptr;
    int lib_df2 = -1;
    bool ready = false;
    bool use_flat_tape = false;
    bool use_untracked = false;

    int max_slots = 1;
    int V = 0;
    int nparts_head = 0;
    double conf_pow = 1.0;

    // Pipelines from ink_metal_dflash2.metal
    mtl::Pipeline p_bf16_to_q8, p_copy_q8_unemb, p_sk6_to_q8;
    mtl::Pipeline p_pack_taps4, p_ctx_kv_store;
    mtl::Pipeline p_init_bf16, p_init_q8, p_rms_norm;
    mtl::Pipeline p_conv_pre, p_conv_post_add;
    mtl::Pipeline p_q8_mv8, p_bf16_mv_m, p_qk_norm_rope;
    mtl::Pipeline p_attn_splitk, p_attn_reduce, p_swiglu;
    mtl::Pipeline p_head_top16_part, p_head_top16_reduce, p_select_chain;

    struct LayerW {
        mtl::Buffer wq_q, wq_s;
        mtl::Buffer wo_q, wo_s;
        mtl::Buffer wg_q, wg_s;
        mtl::Buffer wu_q, wu_s;
        mtl::Buffer wd_q, wd_s;
        mtl::Buffer wkv_bf16;       // packed [2048, 2304] BF16 (first 1024=K, next 1024=V)
        mtl::Buffer conv_a_bf16, conv_f_bf16;
        mtl::Buffer base_a, base_f;
        mtl::Buffer attn_norm, post_norm, q_norm, k_norm;
    } L[kLayers];

    mtl::Buffer fc_bf16, hidden_norm, norm, sel_h_bf16;
    mtl::Buffer pred_bf16, succ_bf16;
    mtl::Buffer mask_row;

    // Target Q8_ROW unembed screen (V x 2304 char + V float scales)
    mtl::Buffer unemb_q, unemb_s;

    // Per-layer context KV rings: ring_k[l], ring_v[l] = [max_slots, 2112, 1024] half
    mtl::Buffer ring_k[kLayers], ring_v[kLayers];
    std::vector<int64_t> lo, hi;

    // Block-8 draft scratch buffers
    mtl::Buffer b_x, b_xn, b_c, b_dyn, b_q, b_kv_blk, b_attn, b_z;
    mtl::Buffer b_g, b_u, b_act, b_hd, b_gsel;
    mtl::Buffer b_part_o, b_part_ml;
    mtl::Buffer b_part_ci, b_part_cv, b_part_tail, b_cand, b_tail;
    mtl::Buffer b_out_ids, b_out_conf;

    // Context ingest scratch buffers (chunked up to kIngChunk = 256 rows)
    static constexpr int kIngChunk = 256;
    mtl::Buffer bi_taps, bi_c, bi_cn, bi_kv;

    std::vector<mtl::FlatDispatch> flat_ds;
    std::vector<mtl::Buffer> owned;

    mtl::Buffer alloc_w(size_t bytes) {
        mtl::Buffer b = cx->alloc(bytes);
        if (b.handle != nullptr) owned.push_back(b);
        return b;
    }
    mtl::Buffer alloc_s(size_t bytes) {
        mtl::Buffer b = use_untracked ? cx->alloc_untracked(bytes) : cx->alloc(bytes);
        if (b.handle != nullptr) owned.push_back(b);
        return b;
    }

    ~InkMetalDFlash2() { release(); }

    void release() {
        if (!cx) return;
        for (auto& b : owned) if (b.handle != nullptr) cx->free(b);
        owned.clear();
        for (int l = 0; l < kLayers; ++l) { ring_k[l] = {}; ring_v[l] = {}; }
        ready = false;
    }

    bool init(mtl::Context& ctx, InkMetalModel& m, int lib_id, const std::string& path,
              int slots, bool flat_tape, bool untracked_scratch, std::string& err) {
        cx = &ctx;
        mod = &m;
        lib_df2 = lib_id;
        use_flat_tape = flat_tape;
        use_untracked = untracked_scratch;
        max_slots = std::max(1, std::min(slots, 64));
        V = m.man.c.vocab_unpadded;
        nparts_head = (V + 63) / 64;
        if (const char* cp = std::getenv("STRATA_INK_DFLASH2_CONF_POW")) {
            double v = std::atof(cp);
            if (v > 0.1 && v < 4.0) conf_pow = v;
        }
        if (lib_df2 < 0) { err = "dflash2 Metal shader library not compiled"; return false; }

        p_bf16_to_q8        = cx->pipeline(lib_df2, "df2_bf16_to_q8row");
        p_copy_q8_unemb     = cx->pipeline(lib_df2, "df2_copy_q8row_unembed");
        p_sk6_to_q8         = cx->pipeline(lib_df2, "df2_sk6_to_q8row");
        p_pack_taps4        = cx->pipeline(lib_df2, "df2_pack_taps4");
        p_ctx_kv_store      = cx->pipeline(lib_df2, "df2_ctx_kv_store");
        p_init_bf16         = cx->pipeline(lib_df2, "df2_init_block_bf16");
        p_init_q8           = cx->pipeline(lib_df2, "df2_init_block_q8row");
        p_rms_norm          = cx->pipeline(lib_df2, "df2_rms_norm");
        p_conv_pre          = cx->pipeline(lib_df2, "df2_conv_pre");
        p_conv_post_add     = cx->pipeline(lib_df2, "df2_conv_post_add");
        p_q8_mv8            = cx->pipeline(lib_df2, "df2_q8_mv8");
        p_bf16_mv_m         = cx->pipeline(lib_df2, "df2_bf16_mv_m");
        p_qk_norm_rope      = cx->pipeline(lib_df2, "df2_qk_norm_rope");
        p_attn_splitk       = cx->pipeline(lib_df2, "df2_block_attn_splitk");
        p_attn_reduce       = cx->pipeline(lib_df2, "df2_block_attn_reduce");
        p_swiglu            = cx->pipeline(lib_df2, "df2_swiglu");
        p_head_top16_part   = cx->pipeline(lib_df2, "df2_head_top16_part_q8");
        p_head_top16_reduce = cx->pipeline(lib_df2, "df2_head_top16_reduce");
        p_select_chain      = cx->pipeline(lib_df2, "df2_select_chain");

        double t0 = mtl::now_s();
        strata::image::StFile st;
        std::string he = strata::image::st_read_header(path, st);
        if (!he.empty()) { err = "dflash2 header: " + he; return false; }

        std::vector<unsigned char> host;
        auto get_bf16 = [&](const std::string& name, size_t want_elems, bool prefix_ok = false) -> bool {
            auto it = st.tensors.find(name);
            if (it == st.tensors.end()) { err = "dflash2 missing tensor: " + name; return false; }
            const size_t have = (size_t)(it->second.end - it->second.begin);
            const size_t want_bytes = want_elems * 2;
            if (it->second.dtype != "BF16" || (prefix_ok ? have < want_bytes : have != want_bytes)) {
                err = "dflash2 bad dtype/size for " + name;
                return false;
            }
            std::string e = strata::image::st_read_tensor(path, st, it->second, host);
            if (!e.empty()) { err = "dflash2 read " + name + ": " + e; return false; }
            return true;
        };
        auto up_bf16 = [&](const std::string& name, size_t elems, mtl::Buffer& dst, bool prefix_ok = false) -> bool {
            if (!get_bf16(name, elems, prefix_ok)) return false;
            dst = alloc_w(elems * 2);
            std::memcpy(dst.ptr, host.data(), elems * 2);
            return true;
        };
        auto up_f32 = [&](const std::string& name, size_t elems, mtl::Buffer& dst) -> bool {
            auto it = st.tensors.find(name);
            if (it == st.tensors.end()) { err = "dflash2 missing tensor: " + name; return false; }
            dst = alloc_w(elems * 4);
            if (it->second.dtype == "F32") {
                if ((size_t)(it->second.end - it->second.begin) != elems * 4) { err = "dflash2 bad F32 size: " + name; return false; }
                std::string e = strata::image::st_read_tensor(path, st, it->second, host);
                if (!e.empty()) { err = "dflash2 read " + name + ": " + e; return false; }
                std::memcpy(dst.ptr, host.data(), elems * 4);
                return true;
            }
            if (!get_bf16(name, elems)) return false;
            const uint16_t* bp = reinterpret_cast<const uint16_t*>(host.data());
            float* fp = dst.as<float>();
            for (size_t i = 0; i < elems; ++i) {
                uint32_t u = (uint32_t)bp[i] << 16;
                std::memcpy(&fp[i], &u, 4);
            }
            return true;
        };

        mtl::Buffer b_tmp = cx->alloc((size_t)kFF * kHidden * 2);
        auto up_q8 = [&](const std::string& name, int N, int K, mtl::Buffer& bq, mtl::Buffer& bs) -> bool {
            // the pre-quantized file ("strata-dflash2-q1" mix): NAME.q / NAME.s (/ NAME.o) instead of NAME (see ../ink_dflash2_mix.h)
            auto itq = st.tensors.find(name + ".q");
            if (itq != st.tensors.end()) {
                auto its = st.tensors.find(name + ".s");
                if (its == st.tensors.end()) { err = "dflash2 missing tensor: " + name + ".s"; return false; }
                const size_t ne = (size_t)N * K;
                const size_t hq = (size_t)(itq->second.end - itq->second.begin), hs = (size_t)(its->second.end - its->second.begin);
                if (itq->second.dtype == "I8" && its->second.dtype == "F32" && hq == ne && hs == (size_t)N * 4) {
                    bq = alloc_w(ne);
                    bs = alloc_w((size_t)N * 4);
                    std::string e = strata::image::st_read_tensor(path, st, itq->second, host);
                    if (!e.empty()) { err = "dflash2 read " + name + ".q: " + e; return false; }
                    std::memcpy(bq.ptr, host.data(), ne);
                    e = strata::image::st_read_tensor(path, st, its->second, host);
                    if (!e.empty()) { err = "dflash2 read " + name + ".s: " + e; return false; }
                    std::memcpy(bs.ptr, host.data(), (size_t)N * 4);
                    return true;
                }
                auto ito = st.tensors.find(name + ".o");
                if (!(itq->second.dtype == "U8" && its->second.dtype == "U8" && ito != st.tensors.end() && ito->second.dtype == "F32" &&
                      hq == ne / 2 && hs == ne / 16 && (size_t)(ito->second.end - ito->second.begin) == 4 && K % 16 == 0)) {
                    err = "dflash2 unsupported pre-quantized tensor " + name;
                    return false;
                }
                std::vector<unsigned char> hq_buf, hs_buf, ho_buf;
                std::string e = strata::image::st_read_tensor(path, st, itq->second, hq_buf);
                if (e.empty()) e = strata::image::st_read_tensor(path, st, its->second, hs_buf);
                if (e.empty()) e = strata::image::st_read_tensor(path, st, ito->second, ho_buf);
                if (!e.empty()) { err = "dflash2 read " + name + ": " + e; return false; }
                float outer; std::memcpy(&outer, ho_buf.data(), 4);
                ink2::df2mix::sf4_to_bf16(hq_buf.data(), hs_buf.data(), outer, N, K, reinterpret_cast<uint16_t*>(b_tmp.ptr));
                bq = alloc_w(ne);
                bs = alloc_w((size_t)N * 4);
                mtl::Dispatch d;
                d.p = &p_bf16_to_q8;
                d.grid[0] = (uint32_t)N;
                d.tg[0] = 256;
                d.args = {mtl::Arg::B(b_tmp), mtl::Arg::B(bq), mtl::Arg::B(bs), mtl::Arg::V(N), mtl::Arg::V(K)};
                cx->run({d});
                return true;
            }
            if (!get_bf16(name, (size_t)N * K)) return false;
            bq = alloc_w((size_t)N * K);
            bs = alloc_w((size_t)N * 4);
            std::memcpy(b_tmp.ptr, host.data(), (size_t)N * K * 2);
            mtl::Dispatch d;
            d.p = &p_bf16_to_q8;
            d.grid[0] = (uint32_t)N;
            d.tg[0] = 256;
            d.args = {mtl::Arg::B(b_tmp), mtl::Arg::B(bq), mtl::Arg::B(bs), mtl::Arg::V(N), mtl::Arg::V(K)};
            cx->run({d});
            return true;
        };

        bool ok = up_bf16("fc.weight", (size_t)kHidden * kTargets * kHidden, fc_bf16) &&
                  up_f32("hidden_norm.weight", kHidden, hidden_norm) &&
                  up_f32("norm.weight", kHidden, norm) &&
                  up_bf16("candidate_selector.hidden_projection.weight", (size_t)kRank * kHidden, sel_h_bf16) &&
                  up_bf16("candidate_selector.predecessor_codebook", (size_t)V * kRank, pred_bf16, true) &&
                  up_bf16("candidate_selector.successor_codebook", (size_t)V * kRank, succ_bf16, true);

        for (int l = 0; ok && l < kLayers; ++l) {
            LayerW& y = L[l];
            const std::string p = "layers." + std::to_string(l) + ".";
            y.wkv_bf16 = alloc_w((size_t)kKV2 * kHidden * 2);
            ok = up_q8(p + "self_attn.q_proj.weight", kQ, kHidden, y.wq_q, y.wq_s) &&
                 up_q8(p + "self_attn.o_proj.weight", kHidden, kQ, y.wo_q, y.wo_s) &&
                 up_q8(p + "mlp.gate_proj.weight", kFF, kHidden, y.wg_q, y.wg_s) &&
                 up_q8(p + "mlp.up_proj.weight", kFF, kHidden, y.wu_q, y.wu_s) &&
                 up_q8(p + "mlp.down_proj.weight", kHidden, kFF, y.wd_q, y.wd_s) &&
                 get_bf16(p + "self_attn.k_proj.weight", (size_t)kKV * kHidden);
            if (ok) std::memcpy(y.wkv_bf16.ptr, host.data(), (size_t)kKV * kHidden * 2);
            if (ok) ok = get_bf16(p + "self_attn.v_proj.weight", (size_t)kKV * kHidden);
            if (ok) std::memcpy(y.wkv_bf16.as<uint8_t>() + (size_t)kKV * kHidden * 2, host.data(), (size_t)kKV * kHidden * 2);
            ok = ok &&
                 up_bf16(p + "attention_conv.kernel_projection.weight", (size_t)kDyn * kHidden, y.conv_a_bf16) &&
                 up_bf16(p + "mlp_conv.kernel_projection.weight", (size_t)kDyn * kHidden, y.conv_f_bf16) &&
                 up_f32(p + "attention_conv.base_kernel", (size_t)4 * kHidden, y.base_a) &&
                 up_f32(p + "mlp_conv.base_kernel", (size_t)4 * kHidden, y.base_f) &&
                 up_f32(p + "input_layernorm.weight", kHidden, y.attn_norm) &&
                 up_f32(p + "post_attention_layernorm.weight", kHidden, y.post_norm) &&
                 up_f32(p + "self_attn.q_norm.weight", kHeadDim, y.q_norm) &&
                 up_f32(p + "self_attn.k_norm.weight", kHeadDim, y.k_norm);
        }
        cx->free(b_tmp);
        if (!ok) { release(); return false; }

        // 2. Build target Q8_ROW unembed screen from target model.llm.unembed.weight!
        unemb_q = alloc_w((size_t)V * kHidden);
        unemb_s = alloc_w((size_t)V * 4);
        {
            const auto& ui = m.man.get("model.llm.unembed.weight");
            mtl::Dispatch d;
            d.grid[0] = (uint32_t)V;
            d.tg[0] = 256;
            if (ui.fmt == FMT_BF16) {
                d.p = &p_bf16_to_q8;
                d.args = {mtl::Arg::B(m.sec1_buf, ui.off), mtl::Arg::B(unemb_q), mtl::Arg::B(unemb_s),
                          mtl::Arg::V(V), mtl::Arg::V(kHidden)};
            } else if (ui.fmt == FMT_Q8ROW) {
                const int Vp = m.man.c.vocab_padded;
                d.p = &p_copy_q8_unemb;
                d.args = {mtl::Arg::B(m.sec1_buf, ui.off), mtl::Arg::B(unemb_q), mtl::Arg::B(unemb_s),
                          mtl::Arg::V(V), mtl::Arg::V(Vp), mtl::Arg::V(kHidden)};
            } else if (ui.fmt == FMT_S6) {
                d.p = &p_sk6_to_q8;
                d.args = {mtl::Arg::B(m.sec1_buf, ui.off), mtl::Arg::B(unemb_q), mtl::Arg::B(unemb_s),
                          mtl::Arg::V(V), mtl::Arg::V(kHidden)};
            } else {
                err = "dflash2: unsupported target unembed format (" + std::string(fmt_name(ui.fmt)) + ")";
                release();
                return false;
            }
            cx->run({d});
        }

        // 3. Precompute mask_row = rms_norm(embed[kMaskId]) * embed_norm
        mask_row = alloc_w((size_t)kHidden * 4);
        {
            const auto& ei = m.man.get("model.llm.embed.weight");
            const auto& en = m.man.get("model.llm.embed_norm.weight");
            const float* nw = reinterpret_cast<const float*>(m.base + m.sec1_off + en.off);
            float* dst = mask_row.as<float>();
            const int mid = std::min(kMaskId, m.man.c.vocab_padded - 1);
            double ss = 0.0;
            if (ei.fmt == FMT_BF16) {
                const uint16_t* row = reinterpret_cast<const uint16_t*>(m.base + m.sec1_off + ei.off) + (size_t)mid * kHidden;
                for (int i = 0; i < kHidden; ++i) {
                    uint32_t u = (uint32_t)row[i] << 16;
                    float v; std::memcpy(&v, &u, 4);
                    dst[i] = v; ss += (double)v * v;
                }
            } else if (ei.fmt == FMT_Q8ROW) {
                const uint8_t* eb = m.base + m.sec1_off + ei.off;
                const float sc = reinterpret_cast<const float*>(eb)[mid];
                const int8_t* row = reinterpret_cast<const int8_t*>(eb + (size_t)m.man.c.vocab_padded * 4) + (size_t)mid * kHidden;
                for (int i = 0; i < kHidden; ++i) {
                    float v = (float)row[i] * sc;
                    dst[i] = v; ss += (double)v * v;
                }
            } else {
                err = "dflash2: unsupported target embed format";
                release();
                return false;
            }
            const float inv = 1.0f / std::sqrt((float)(ss / (double)kHidden) + m.man.c.rms_eps);
            for (int i = 0; i < kHidden; ++i) dst[i] = dst[i] * inv * nw[i];
        }

        // 4. Allocate per-layer KV rings ([max_slots, 2112, 1024] half each) & scratch
        const size_t ring_bytes = (size_t)max_slots * kRingCap * kKV * sizeof(uint16_t);
        for (int l = 0; l < kLayers; ++l) {
            ring_k[l] = alloc_w(ring_bytes);
            ring_v[l] = alloc_w(ring_bytes);
            std::memset(ring_k[l].ptr, 0, ring_bytes);
            std::memset(ring_v[l].ptr, 0, ring_bytes);
        }
        lo.assign((size_t)max_slots, 0);
        hi.assign((size_t)max_slots, 0);

        b_x         = alloc_s((size_t)kBlock * kHidden * 4);
        b_xn        = alloc_s((size_t)kBlock * kHidden * 4);
        b_c         = alloc_s((size_t)kBlock * kHidden * 4);
        b_dyn       = alloc_s((size_t)kBlock * kDyn * 4);
        b_q         = alloc_s((size_t)kBlock * kQ * 4);
        b_kv_blk    = alloc_s((size_t)kBlock * kKV2 * 4);
        b_attn      = alloc_s((size_t)kBlock * kQ * 4);
        b_z         = alloc_s((size_t)kBlock * kHidden * 4);
        b_g         = alloc_s((size_t)kBlock * kFF * 4);
        b_u         = alloc_s((size_t)kBlock * kFF * 4);
        b_act       = alloc_s((size_t)kBlock * kFF * 4);
        b_hd        = alloc_s((size_t)kBlock * kHidden * 4);
        b_gsel      = alloc_s((size_t)kBlock * kRank * 4);
        b_part_o    = alloc_s((size_t)kBlock * kQHeads * 8 * kHeadDim * 4);
        b_part_ml   = alloc_s((size_t)kBlock * kQHeads * 8 * 2 * 4);
        b_part_ci   = alloc_s((size_t)kDraft * nparts_head * kTopK * 4);
        b_part_cv   = alloc_s((size_t)kDraft * nparts_head * kTopK * 4);
        b_part_tail = alloc_s((size_t)kDraft * nparts_head * 2 * 4);
        b_cand      = alloc_s((size_t)kDraft * 32 * 4);
        b_tail      = alloc_s((size_t)kDraft * 2 * 4);
        b_out_ids   = alloc_s((size_t)kDraft * 4);
        b_out_conf  = alloc_s((size_t)kDraft * 4);

        bi_taps = alloc_s((size_t)kIngChunk * kTargets * kHidden * 4);
        bi_c    = alloc_s((size_t)kIngChunk * kHidden * 4);
        bi_cn   = alloc_s((size_t)kIngChunk * kHidden * 4);
        bi_kv   = alloc_s((size_t)kIngChunk * kKV2 * 4);

        ready = true;
        fprintf(stderr, "[ink_metal_dflash2] loaded %s in %.2fs (Q8_ROW 5L + Q8 unembed V=%d nparts=%d, slots=%d, rings=%.1f MiB)\n",
                path.c_str(), mtl::now_s() - t0, V, nparts_head, max_slots,
                (double)(2 * kLayers * ring_bytes) / (1024.0 * 1024.0));
        return true;
    }

    void reset_slot(int slot) {
        if (slot >= 0 && slot < max_slots) { lo[slot] = 0; hi[slot] = 0; }
    }

    void rollback_slot(int slot, int keep_len) {
        if (slot >= 0 && slot < max_slots) {
            if (keep_len < hi[slot]) hi[slot] = keep_len;
            if (lo[slot] > hi[slot]) lo[slot] = hi[slot];
        }
    }

    struct RingSnap {
        int64_t lo = 0, hi = 0;
        std::vector<uint16_t> rk[kLayers], rv[kLayers];
    };

    void save_slot_ring(int slot, RingSnap& out) {
        out.lo = 0; out.hi = 0;
        for (int l = 0; l < kLayers; ++l) { out.rk[l].clear(); out.rv[l].clear(); }
        if (!ready || slot < 0 || slot >= max_slots) return;
        const size_t n = (size_t)kRingCap * kKV;
        out.lo = lo[slot];
        out.hi = hi[slot];
        for (int l = 0; l < kLayers; ++l) {
            out.rk[l].resize(n);
            out.rv[l].resize(n);
            std::memcpy(out.rk[l].data(), ring_k[l].as<uint16_t>() + (size_t)slot * n, n * 2);
            std::memcpy(out.rv[l].data(), ring_v[l].as<uint16_t>() + (size_t)slot * n, n * 2);
        }
    }

    void restore_slot_ring(int slot, const RingSnap& in) {
        if (!ready || slot < 0 || slot >= max_slots || in.rk[0].empty()) return;
        const size_t n = (size_t)kRingCap * kKV;
        for (int l = 0; l < kLayers; ++l) {
            std::memcpy(ring_k[l].as<uint16_t>() + (size_t)slot * n, in.rk[l].data(), n * 2);
            std::memcpy(ring_v[l].as<uint16_t>() + (size_t)slot * n, in.rv[l].data(), n * 2);
        }
        lo[slot] = in.lo;
        hi[slot] = in.hi;
    }

    // Ingest 4 target taps (b_taps4[0..2, M, 2304] + b_h_norm[M, 2304]) into DFlash 2 context rings!
    void ingest_taps(const mtl::Buffer& b_taps4, const mtl::Buffer& b_h_norm,
                     const mtl::Buffer& b_slot_ids, const mtl::Buffer& b_pos,
                     const int32_t* slot_ids, const int32_t* pos_ids, int M) {
        if (!ready || M <= 0) return;
        constexpr int kKeep = kRingCap - 16;

        // Track per-slot lo/hi bounds and skip early prefill rows outside the 2096-token sliding window!
        std::vector<int64_t> mx((size_t)max_slots, -1), mn((size_t)max_slots, INT64_MAX);
        std::vector<int> cnt((size_t)max_slots, 0);
        for (int i = 0; i < M; ++i) {
            const int s = slot_ids[i];
            if (s < 0 || s >= max_slots) continue;
            mx[s] = std::max<int64_t>(mx[s], pos_ids[i]);
            mn[s] = std::min<int64_t>(mn[s], pos_ids[i]);
            ++cnt[s];
        }
        for (int s = 0; s < max_slots; ++s) {
            if (cnt[s] == 0) continue;
            if (mn[s] != hi[s]) lo[s] = mn[s];
            hi[s] = mx[s] + 1;
            lo[s] = std::max<int64_t>(lo[s], hi[s] - kKeep);
        }

        // For long single-slot prefill (e.g. M=8192), only ingest the final kKeep rows (zeros wasted work!)
        int start_row = 0;
        if (M > kKeep) {
            bool all_same = true;
            for (int i = 1; i < M; ++i) if (slot_ids[i] != slot_ids[0]) { all_same = false; break; }
            if (all_same) start_row = M - kKeep;
        }

        const int M_stride = M * kHidden;
        const int N_fc = kHidden, K_fc = kTargets * kHidden;
        const int N_kv2 = kKV2, K_h = kHidden;

        for (int r0 = start_row; r0 < M; r0 += kIngChunk) {
            const int cn = std::min(kIngChunk, M - r0);
            std::vector<mtl::Dispatch> ds;
            ds.reserve(20);
            auto add = [&](mtl::Pipeline& p, uint32_t gx, uint32_t gy, uint32_t tx, std::initializer_list<mtl::Arg> args, bool bar = true) {
                mtl::Dispatch d; d.p = &p; d.grid[0] = gx; d.grid[1] = gy; d.tg[0] = tx; d.args = args; d.barrier = bar;
                ds.push_back(std::move(d));
            };
            const uint32_t m_tiles = (uint32_t)((cn + 7) / 8);
            add(p_pack_taps4, (uint32_t)cn, 1, 256,
                {mtl::Arg::B(b_taps4), mtl::Arg::B(b_h_norm), mtl::Arg::B(bi_taps),
                 mtl::Arg::V(M_stride), mtl::Arg::V(r0)});
            add(p_bf16_mv_m, (uint32_t)((N_fc + 15) / 16), m_tiles, 256,
                {mtl::Arg::B(fc_bf16), mtl::Arg::B(bi_taps), mtl::Arg::B(bi_c),
                 mtl::Arg::V(N_fc), mtl::Arg::V(K_fc), mtl::Arg::V(cn)});
            add(p_rms_norm, (uint32_t)cn, 1, 256,
                {mtl::Arg::B(bi_c), mtl::Arg::B(hidden_norm), mtl::Arg::B(bi_cn)});
            for (int l = 0; l < kLayers; ++l) {
                add(p_bf16_mv_m, (uint32_t)((N_kv2 + 15) / 16), m_tiles, 256,
                    {mtl::Arg::B(L[l].wkv_bf16), mtl::Arg::B(bi_cn), mtl::Arg::B(bi_kv),
                     mtl::Arg::V(N_kv2), mtl::Arg::V(K_h), mtl::Arg::V(cn)});
                add(p_ctx_kv_store, (uint32_t)cn, kKvHeads, 64,
                    {mtl::Arg::B(bi_kv), mtl::Arg::B(L[l].k_norm), mtl::Arg::B(ring_k[l]), mtl::Arg::B(ring_v[l]),
                     mtl::Arg::B(b_slot_ids), mtl::Arg::B(b_pos), mtl::Arg::V(r0), mtl::Arg::V(kTheta)});
            }
            cx->run_graph(ds);
        }
    }

    // Run full 5-layer Block-8 DFlash 2 forward + Top-16 Q8 unembed + rank-256 bilinear selector in ONE CB!
    int draft_block8(int slot, int32_t anchor_tok, int anchor_pos, int32_t out_ids[7], float out_conf[7]) {
        if (!ready || slot < 0 || slot >= max_slots) return 0;
        const int ctx_lo = (int)lo[slot];
        const int ctx_hi = (int)std::min<int64_t>(hi[slot], anchor_pos);
        Df2AttnArgs aa{slot, anchor_pos, ctx_lo, ctx_hi, 1.0f / std::sqrt((float)kHeadDim), 0};

        const auto& ei = mod->man.get("model.llm.embed.weight");
        const auto& en = mod->man.get("model.llm.embed_norm.weight");
        const int M8 = kBlock;
        const int nh_q = kQHeads, nh_kv = kKvHeads, st_q = kQ, st_kv = kKV2;
        const int n_ff8 = kBlock * kFF;
        const int n_dr = kDraft;

        auto run_tape = [&](auto&& push_fn) {
            if (ei.fmt == FMT_BF16) {
                push_fn(p_init_bf16, 8, 1, 1, 256,
                        {mtl::Arg::B(mod->sec1_buf, ei.off), mtl::Arg::B(mod->sec1_buf, en.off),
                         mtl::Arg::B(mask_row), mtl::Arg::B(b_x), mtl::Arg::V(anchor_tok)}, true);
            } else {
                const int Vp = mod->man.c.vocab_padded;
                push_fn(p_init_q8, 8, 1, 1, 256,
                        {mtl::Arg::B(mod->sec1_buf, ei.off), mtl::Arg::B(mod->sec1_buf, en.off),
                         mtl::Arg::B(mask_row), mtl::Arg::B(b_x), mtl::Arg::V(anchor_tok), mtl::Arg::V(Vp)}, true);
            }

            for (int l = 0; l < kLayers; ++l) {
                const LayerW& y = L[l];
                push_fn(p_rms_norm, 8, 1, 1, 256,
                        {mtl::Arg::B(b_x), mtl::Arg::B(y.attn_norm), mtl::Arg::B(b_xn)}, true);
                push_fn(p_bf16_mv_m, (kDyn + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.conv_a_bf16), mtl::Arg::B(b_xn), mtl::Arg::B(b_dyn),
                         mtl::Arg::V(kDyn), mtl::Arg::V(kHidden), mtl::Arg::V(M8)}, true);
                push_fn(p_conv_pre, 8, 9, 1, 256,
                        {mtl::Arg::B(b_xn), mtl::Arg::B(b_dyn), mtl::Arg::B(y.base_a), mtl::Arg::B(b_c)}, true);

                // Q (Q8_ROW) & packed [K,V] (BF16) run concurrently!
                push_fn(p_q8_mv8, (kQ + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.wq_q), mtl::Arg::B(y.wq_s), mtl::Arg::B(b_c), mtl::Arg::B(b_q),
                         mtl::Arg::V(kQ), mtl::Arg::V(kHidden)}, true);
                push_fn(p_bf16_mv_m, (kKV2 + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.wkv_bf16), mtl::Arg::B(b_c), mtl::Arg::B(b_kv_blk),
                         mtl::Arg::V(kKV2), mtl::Arg::V(kHidden), mtl::Arg::V(M8)}, false);

                // Q & K head RMSNorm + RoPE run concurrently!
                push_fn(p_qk_norm_rope, 8, kQHeads, 1, 64,
                        {mtl::Arg::B(b_q), mtl::Arg::B(y.q_norm), mtl::Arg::V(nh_q),
                         mtl::Arg::V(st_q), mtl::Arg::V(anchor_pos), mtl::Arg::V(kTheta)}, true);
                push_fn(p_qk_norm_rope, 8, kKvHeads, 1, 64,
                        {mtl::Arg::B(b_kv_blk), mtl::Arg::B(y.k_norm), mtl::Arg::V(nh_kv),
                         mtl::Arg::V(st_kv), mtl::Arg::V(anchor_pos), mtl::Arg::V(kTheta)}, false);

                // Non-causal SWA-2048 32:8 GQA Split-K Attention
                push_fn(p_attn_splitk, 8, kKvHeads, 8, 128,
                        {mtl::Arg::B(b_q), mtl::Arg::B(b_kv_blk), mtl::Arg::B(ring_k[l]), mtl::Arg::B(ring_v[l]),
                         mtl::Arg::B(b_part_o), mtl::Arg::B(b_part_ml), mtl::Arg::V(aa)}, true);
                push_fn(p_attn_reduce, 8, kQHeads, 1, 128,
                        {mtl::Arg::B(b_part_o), mtl::Arg::B(b_part_ml), mtl::Arg::B(b_attn)}, true);

                // O projection + post-conv residual add
                push_fn(p_q8_mv8, (kHidden + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.wo_q), mtl::Arg::B(y.wo_s), mtl::Arg::B(b_attn), mtl::Arg::B(b_z),
                         mtl::Arg::V(kHidden), mtl::Arg::V(kQ)}, true);
                push_fn(p_conv_post_add, 8, 9, 1, 256,
                        {mtl::Arg::B(b_x), mtl::Arg::B(b_z), mtl::Arg::B(b_dyn), mtl::Arg::B(y.base_a)}, true);

                // Post-attention norm + MLP conv pre
                push_fn(p_rms_norm, 8, 1, 1, 256,
                        {mtl::Arg::B(b_x), mtl::Arg::B(y.post_norm), mtl::Arg::B(b_xn)}, true);
                push_fn(p_bf16_mv_m, (kDyn + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.conv_f_bf16), mtl::Arg::B(b_xn), mtl::Arg::B(b_dyn),
                         mtl::Arg::V(kDyn), mtl::Arg::V(kHidden), mtl::Arg::V(M8)}, true);
                push_fn(p_conv_pre, 8, 9, 1, 256,
                        {mtl::Arg::B(b_xn), mtl::Arg::B(b_dyn), mtl::Arg::B(y.base_f), mtl::Arg::B(b_c)}, true);

                // SwiGLU FFN (gate & up run concurrently!)
                push_fn(p_q8_mv8, (kFF + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.wg_q), mtl::Arg::B(y.wg_s), mtl::Arg::B(b_c), mtl::Arg::B(b_g),
                         mtl::Arg::V(kFF), mtl::Arg::V(kHidden)}, true);
                push_fn(p_q8_mv8, (kFF + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.wu_q), mtl::Arg::B(y.wu_s), mtl::Arg::B(b_c), mtl::Arg::B(b_u),
                         mtl::Arg::V(kFF), mtl::Arg::V(kHidden)}, false);
                push_fn(p_swiglu, (uint32_t)((n_ff8 + 255) / 256), 1, 1, 256,
                        {mtl::Arg::B(b_g), mtl::Arg::B(b_u), mtl::Arg::B(b_act), mtl::Arg::V(n_ff8)}, true);
                push_fn(p_q8_mv8, (kHidden + 15) / 16, 1, 1, 256,
                        {mtl::Arg::B(y.wd_q), mtl::Arg::B(y.wd_s), mtl::Arg::B(b_act), mtl::Arg::B(b_z),
                         mtl::Arg::V(kHidden), mtl::Arg::V(kFF)}, true);
                push_fn(p_conv_post_add, 8, 9, 1, 256,
                        {mtl::Arg::B(b_x), mtl::Arg::B(b_z), mtl::Arg::B(b_dyn), mtl::Arg::B(y.base_f)}, true);
            }

            // Final norm -> sel_h & single-DRAM-pass Q8 unembed Top-16 run concurrently!
            push_fn(p_rms_norm, 8, 1, 1, 256,
                    {mtl::Arg::B(b_x), mtl::Arg::B(norm), mtl::Arg::B(b_hd)}, true);
            push_fn(p_bf16_mv_m, (kRank + 15) / 16, 1, 1, 256,
                    {mtl::Arg::B(sel_h_bf16), mtl::Arg::B(b_hd), mtl::Arg::B(b_gsel),
                     mtl::Arg::V(kRank), mtl::Arg::V(kHidden), mtl::Arg::V(M8)}, true);
            push_fn(p_head_top16_part, (uint32_t)nparts_head, 1, 1, 256,
                    {mtl::Arg::B(unemb_q), mtl::Arg::B(unemb_s), mtl::Arg::B(b_hd),
                     mtl::Arg::B(b_part_ci), mtl::Arg::B(b_part_cv), mtl::Arg::B(b_part_tail),
                     mtl::Arg::V(V), mtl::Arg::V(nparts_head)}, false);
            push_fn(p_head_top16_reduce, kDraft, 1, 1, 256,
                    {mtl::Arg::B(b_part_ci), mtl::Arg::B(b_part_cv), mtl::Arg::B(b_part_tail),
                     mtl::Arg::B(b_cand), mtl::Arg::B(b_tail), mtl::Arg::V(nparts_head)}, true);
            push_fn(p_select_chain, 1, 1, 1, 512,
                    {mtl::Arg::B(b_cand), mtl::Arg::B(b_tail), mtl::Arg::B(b_gsel),
                     mtl::Arg::B(pred_bf16), mtl::Arg::B(succ_bf16),
                     mtl::Arg::B(b_out_ids), mtl::Arg::B(b_out_conf),
                     mtl::Arg::V(anchor_tok), mtl::Arg::V(n_dr)}, true);
        };

        if (use_flat_tape) {
            flat_ds.clear();
            if (flat_ds.capacity() < 128) flat_ds.reserve(128);
            run_tape([&](mtl::Pipeline& p, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx,
                         std::initializer_list<mtl::Arg> args, bool bar) {
                mtl::FlatDispatch d;
                d.p = &p; d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz; d.tg[0] = tx; d.barrier = bar;
                for (const auto& a : args) {
                    d.push_arg(mtl::FlatArg::from_arg(a));
                }
                flat_ds.push_back(d);
            });
            cx->run_graph_flat(flat_ds.data(), flat_ds.size());
        } else {
            std::vector<mtl::Dispatch> ds;
            ds.reserve(128);
            run_tape([&](mtl::Pipeline& p, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx,
                         std::initializer_list<mtl::Arg> args, bool bar) {
                mtl::Dispatch d;
                d.p = &p; d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz; d.tg[0] = tx; d.barrier = bar; d.args = args;
                ds.push_back(std::move(d));
            });
            cx->run_graph(ds);
        }

        const int32_t* ids = b_out_ids.as<int32_t>();
        const float* confs = b_out_conf.as<float>();
        for (int i = 0; i < kDraft; ++i) {
            out_ids[i] = ids[i];
            out_conf[i] = confs[i];
        }
        return kDraft;
    }

    // ceiling.py MoE expert-union adaptive verify width planner calibrated on Apple M5 Max (558 GB/s UMA)
    // Uses real M5 Max verify scaling T(M)/T(1) = 1.0 + extra_expert_slope * (E_union(M) - 8.0).
    int choose_verify_drafts(const float conf[7], int max_d, double extra_expert_slope, double draft_overhead_rel) const {
        if (max_d <= 0) return 0;
        if (const char* fixed = std::getenv("STRATA_INK_DFLASH2_FIXED_D")) {
            const int fd = std::atoi(fixed);
            if (fd >= 0 && fd <= 7) return std::min(fd, max_d);
        }
        // Baseline rate after draft_block8 has already run: d=0 emits 1 token for (draft_overhead_rel + 1.0)
        double best_rate = 1.0 / (draft_overhead_rel + 1.0);
        int best_d = 0;
        double prod = 1.0;
        double e_emit = 1.0;
        for (int d = 1; d <= std::min(7, max_d); ++d) {
            const double p = std::clamp((double)conf[d - 1], 0.0, 1.0);
            prod *= (conf_pow == 1.0) ? p : std::pow(p, conf_pow);
            e_emit += prod;
            const int M = 1 + d;
            const double e_union = 128.0 * (1.0 - std::pow(1.0 - 8.0 / 128.0, (double)M));
            const double rel_cost = 1.0 + extra_expert_slope * (e_union - 8.0);
            const double rate = e_emit / (draft_overhead_rel + rel_cost);
            if (rate > best_rate) {
                best_rate = rate;
                best_d = d;
            }
        }
        return best_d;
    }
};

}  // namespace ink_metal
