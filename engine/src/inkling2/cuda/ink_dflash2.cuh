// engine/src/inkling2/cuda/ink_dflash2.cuh - opt-in DFlash 2 block-diffusion drafter for Inkling 2 (CUDA)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// The drafter follows tjm8874's DFlash 2 implementation for the checkpoint incoai/Inkling-2-Mini-DFlash2 (Apache-2.0); the licence notices are in notices/.
// Strata's own parts: the loader of the sidecar safetensors file, Q8 row-scaled block weights with a dp4a skinny GEMV, per-slot
// sliding-window rings, the batched block attention and the hooks into the existing drafting paths.
//
// What it is: 5 non-causal sliding-window (2048) transformer layers (32 query / 8 KV heads of 128, SwiGLU 6912, RoPE theta 1e4), each of whose two
// sublayers is wrapped in 2-tap group-16 dynamic causal convolutions along the 8 rows of a block. Context: the target's 4 taps (visits 5, 23, 41 and the
// final normed state), [f0 | f1 | f2 | f3] -> fc (2304 x 9216) -> RMSNorm(hidden_norm) -> per layer K / V (K RMS-normed per head and roped) into a ring.
// Block: row 0 = the anchor token's embedding (RMS-normed, scaled by the target's embed_norm), rows 1..7 = the mask token's. Draft rows 1..7 are
// normed with `norm`, scored against the target's own head (the Q8 unembed that DSpark also screens with): the exact top-16 and the full-vocabulary tail,
// then a rank-256 bilinear selector walks the 7 steps. A draft only picks what the target verifies: the output is the target's greedy argmax either way.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ink_dflash2_path.cuh"
#include "ink_dflash2_kernels.cuh"
#include "../../image/vae/safetensors_lite.hpp"

namespace strata::ink {

struct InkDFlash2 {
    InkCudaRunner& r;
    InkCudaModel& m;
    bool ready = false;
    int max_slots = 1;
    int max_batch_rows = 0;
    int V = 0;
    int V_gemm = 0;
    int vt_nom = 0;
    float cap = 30.0f;
    float theta = 10000.0f;
    int mask_id = 200056;
    bool host_codebooks = false;
    double conf_pow = std::getenv("STRATA_INK_DFLASH2_CONF_POW") ? std::atof(std::getenv("STRATA_INK_DFLASH2_CONF_POW")) : 1.0;   // < 1 trusts the selector more
    bool w4 = false;   // 4-bit block weights (tight-VRAM cards; STRATA_INK_DFLASH2_W4=0|1)
    bool f32 = false;  // NV-OLDER: a card below sm_80: the BF16 matrices are held as FP32 and multiplied with cuBLAS FP32 (see df2_f32_path), the head screen is a dp4a kernel
    bool screen_cublas = false;   // the head screen runs through the cuBLAS INT8 GEMM (sm_80+; STRATA_INK_DFLASH2_SCREEN=cublas|kernel decides)
    float *d_ix_f = nullptr, *d_icn_f = nullptr;   // f32 form of the ingest inputs
    Df2Variant var;
    __nv_bfloat16* d_xb = nullptr;   // bf16 copy of the activation for the bf16 variant
    uint32_t* d_amax = nullptr;
    bool expand_bf16 = false;   // a pre-quantized sidecar on a card whose row is bf16: dequantized at load
    size_t bytes_weights = 0, bytes_scratch = 0;

    struct Layer {
        void *wq = nullptr, *wo = nullptr, *wg = nullptr, *wu = nullptr, *wd = nullptr;   // int8 [N][K] (+ float scale per row) or, with w4, 4-bit groups (+ half scale per 32)
        void *sq = nullptr, *so = nullptr, *sg = nullptr, *su = nullptr, *sd = nullptr;
        int kq = 1, ko = 1, kg = 1, ku = 1, kd = 1;            // 0 bf16, 1 Q8, 2 4-bit groups, 3 SF4 (see ink_dflash2_path.cuh)
        float oq = 1, oo = 1, og = 1, ou = 1, od = 1;           // SF4 tensor scales
        uint16_t *wkv = nullptr, *conv_a = nullptr, *conv_f = nullptr;   // bf16
        float *base_a = nullptr, *base_f = nullptr, *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    } L[df2::kLayers];
    uint16_t *fc = nullptr, *sel_h = nullptr, *pred = nullptr, *succ = nullptr;
    float *hidden_norm = nullptr, *norm = nullptr, *mask_row = nullptr;
    void *pred_raw = nullptr, *succ_raw = nullptr;

    // per-slot rings (allocated at the first ingest of a slot)
    std::vector<__half*> ring_k_h, ring_v_h;
    __half** d_rk_tab = nullptr;
    __half** d_rv_tab = nullptr;
    std::vector<int64_t> lo, hi;
    std::vector<char> slot_dead;

    // block scratch (rows R = 8 * max_slots)
    int32_t *d_meta = nullptr, *d_pos = nullptr, *d_srow = nullptr;
    float *d_anchor_emb = nullptr, *d_x = nullptr, *d_xn = nullptr, *d_c = nullptr, *d_dyn = nullptr, *d_q = nullptr, *d_kb = nullptr, *d_vb = nullptr;
    float *d_attn = nullptr, *d_z = nullptr, *d_g = nullptr, *d_u = nullptr, *d_act = nullptr, *d_hd = nullptr, *d_gsel = nullptr;
    __nv_bfloat16 *d_xn_bf = nullptr, *d_c_bf = nullptr, *d_hd_bf = nullptr;
    int8_t* d_aq = nullptr;
    float* d_as = nullptr;
    float *p_m = nullptr, *p_l = nullptr, *p_o = nullptr;
    int8_t* d_hq8 = nullptr;
    float* d_hsc = nullptr;
    int32_t* d_tile = nullptr;
    float2 *d_part = nullptr, *d_part_lse = nullptr;
    int32_t* d_cand = nullptr;
    float* d_tail = nullptr;
    int32_t* d_ids = nullptr;
    float* d_conf = nullptr;
    int32_t* h_ids = nullptr;
    float* h_conf = nullptr;
    // ingest scratch
    static constexpr int kCh = 256;
    __nv_bfloat16 *d_ix_bf = nullptr, *d_icn_bf = nullptr;
    float *d_ic = nullptr, *d_ikc = nullptr, *d_ivc = nullptr;
    int32_t* d_ing = nullptr;   // [3][max_batch_rows]: src rows, slots, positions
    static constexpr int kStageRing = 32;
    int32_t* h_stage = nullptr;
    size_t stage_stride = 0;
    int stage_idx = 0;
    // stats (STRATA_INK_DFLASH2_PROF=1: GPU time of the ingest and draft calls, printed every 200 drafts and at exit)
    long n_drafts_run = 0, n_ingest_rows = 0;
    bool prof = std::getenv("STRATA_INK_DFLASH2_PROF") != nullptr;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> prof_pend;
    double prof_ingest_ms = 0, prof_draft_ms = 0;
    long prof_ingest_calls = 0;
    void prof_flush() {
        for (auto& pr : prof_pend) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, pr.first, pr.second) == cudaSuccess) prof_ingest_ms += ms;
            cudaEventDestroy(pr.first);
            cudaEventDestroy(pr.second);
        }
        prof_pend.clear();
    }
    void prof_print() {
        prof_flush();
        std::fprintf(stderr, "[InkDFlash2 prof] ingest %ld calls %ld rows %.1f ms (%.3f ms/row) | %ld drafts %.1f ms (%.2f ms/draft)\n", prof_ingest_calls, n_ingest_rows,
                     prof_ingest_ms, n_ingest_rows ? prof_ingest_ms / (double)n_ingest_rows : 0.0, n_drafts_run, prof_draft_ms, n_drafts_run ? prof_draft_ms / (double)n_drafts_run : 0.0);
    }

    explicit InkDFlash2(InkCudaRunner& runner) : r(runner), m(runner.m) {}

    static bool requested() { return df2_requested(); }
    static std::string resolve_path(const std::string& model_path) { return df2_resolve_path(model_path); }

    // ---- weights ----------------------------------------------------------------------------------------------------------
    // every device buffer of the drafter, so that free_all() gives ALL of it back when the drafter cannot start (11-26: a bf16 start on a 16 GB card that then could not take its rings
    // left 21 MiB free and the engine died at the first request instead of running without a drafter)
    std::vector<void*> owned;
    template <typename T>
    T* wal(size_t n) {
        bytes_weights += n * sizeof(T);
        T* p = dalloc<T>(n);
        owned.push_back((void*)p);
        return p;
    }
    template <typename T>
    T* sal(size_t n) {
        bytes_scratch += n * sizeof(T);
        T* p = dalloc<T>(n);
        owned.push_back((void*)p);
        return p;
    }

    bool load(const std::string& path, int slots, std::string& err) {
        strata::image::StFile st;
        const std::string he = strata::image::st_read_header(path, st);
        if (!he.empty()) { err = "dflash2 file: " + he; return false; }
        max_slots = std::max(1, std::min(slots, 64));
        V = m.man.c.vocab_unpadded;
        const int V_u = (m.unembed_screen.V_pad > 0) ? m.unembed_screen.V_pad : V;
        V_gemm = V_u & ~3;   // the int8 tensor-core GEMM wants multiples of 4: the last <= 3 vocabulary rows (never drafted) are left out of the candidates
        f32 = m.sm_cc < 80;
        screen_cublas = m.sm_cc >= 61 && m.sm_cc != 70;   // the cuBLAS INT8 GEMM exists from sm_61; sm_60 (P100, emulated dp4a) and sm_70 (V100: hardware dp4a, the kernel is 0.4 ms a draft faster than the cuBLAS INT8 GEMM) run the dp4a kernel
        if (const char* e = std::getenv("STRATA_INK_DFLASH2_SCREEN")) {   // for the A/B of the two screens (the cuBLAS INT8 GEMM wants sm_61 or newer)
            if (std::string(e) == "kernel") screen_cublas = false;
            else if (std::string(e) == "cublas" && m.sm_cc >= 61) screen_cublas = true;
        }
        if (!m.unembed_screen.ready || m.sm_cc < 60 || V_gemm < 4096) {
            err = "the DFlash 2 drafter needs the Q8 unembed screen and an sm_60+ card";
            return false;
        }
        if (const char* e = std::getenv("STRATA_INK_DFLASH2_F32")) {   // test hook: the FP32 form on an sm_80+ card too
            if (std::string(e) == "1") f32 = true;
        }
        df2_f32_path() = f32;
        vt_nom = ((V_gemm + 7) / 8 + 3) & ~3;
        std::vector<unsigned char> host;
        auto get = [&](const std::string& name, size_t want_bytes, bool prefix_ok = false) -> bool {
            auto it = st.tensors.find(name);
            if (it == st.tensors.end()) { err = "dflash2 file: tensor " + name + " is missing"; return false; }
            const size_t have = (size_t)(it->second.end - it->second.begin);
            if (it->second.dtype != "BF16" || (prefix_ok ? have < want_bytes : have != want_bytes)) {
                err = "dflash2 file: tensor " + name + " has another type or size";
                return false;
            }
            const std::string e = strata::image::st_read_tensor(path, st, it->second, host);
            if (!e.empty()) { err = "dflash2 file: " + e; return false; }
            return true;
        };
        // raw tensor of any type (the pre-quantized sidecar's NAME.q / .s / .o), exactly want_bytes long
        auto getraw = [&](const std::string& name, size_t want_bytes, const char* dtype) -> bool {
            auto it = st.tensors.find(name);
            if (it == st.tensors.end() || (size_t)(it->second.end - it->second.begin) != want_bytes || it->second.dtype != dtype) {
                err = "dflash2 file: tensor " + name + " is missing or has another type or size";
                return false;
            }
            const std::string e = strata::image::st_read_tensor(path, st, it->second, host);
            if (!e.empty()) { err = "dflash2 file: " + e; return false; }
            return true;
        };
        auto up_bf16 = [&](const std::string& name, size_t n, uint16_t*& dst, size_t dst_off = 0, bool alloc = true, size_t alloc_n = 0) -> bool {
            if (!get(name, n * 2)) return false;
            if (f32) {   // the checkpoint's BF16 values, exactly, as FP32 (two uint16 slots per element)
                if (alloc) dst = wal<uint16_t>(2 * (alloc_n ? alloc_n : n));
                std::vector<float> fv(n);
                const uint16_t* b = reinterpret_cast<const uint16_t*>(host.data());
                for (size_t i = 0; i < n; ++i) { uint32_t u = (uint32_t)b[i] << 16; std::memcpy(&fv[i], &u, 4); }
                INK_CK(cudaMemcpy(reinterpret_cast<float*>(dst) + dst_off, fv.data(), n * 4, cudaMemcpyHostToDevice));
                return true;
            }
            if (alloc) dst = wal<uint16_t>(alloc_n ? alloc_n : n);
            INK_CK(cudaMemcpy(dst + dst_off, host.data(), n * 2, cudaMemcpyHostToDevice));
            return true;
        };
        auto up_f32 = [&](const std::string& name, size_t n, float*& dst) -> bool {
            {
                auto itf = st.tensors.find(name);
                if (itf != st.tensors.end() && itf->second.dtype == "F32") {   // a pre-quantized sidecar keeps the norms as f32
                    if (!getraw(name, n * 4, "F32")) return false;
                    dst = wal<float>(n);
                    INK_CK(cudaMemcpy(dst, host.data(), n * 4, cudaMemcpyHostToDevice));
                    return true;
                }
            }
            if (!get(name, n * 2)) return false;
            std::vector<float> f(n);
            const uint16_t* b = reinterpret_cast<const uint16_t*>(host.data());
            for (size_t i = 0; i < n; ++i) { uint32_t u = (uint32_t)b[i] << 16; std::memcpy(&f[i], &u, 4); }
            dst = wal<float>(n);
            INK_CK(cudaMemcpy(dst, f.data(), n * 4, cudaMemcpyHostToDevice));
            return true;
        };
        {
            size_t vf = 0, vt = 0;
            (void)ink_cuda_mem_info(&vf, &vt);
            const bool small_vram = m.all_sf4_vram || (vt > 0 && vt < (24ull << 30));
            var = df2_variant(small_vram);
            w4 = (var.attn == 2 || var.ffn == 2);
            expand_bf16 = (var.attn == 0 && var.ffn == 0);   // the format the card's row asks for is bf16: a pre-quantized sidecar is expanded to it
        }
        uint16_t* tmp = dalloc<uint16_t>((size_t)df2::kFF * df2::kHidden);   // the largest matrix, bf16, before it is quantized
        d_amax = dalloc<uint32_t>(1);
        auto up_q8 = [&](const std::string& name, int N, int K, int& kind, void*& q, void*& sc, float& outer) -> bool {
            if (st.tensors.count(name + ".q")) {   // pre-quantized: the format follows the tensors
                const auto& ts = st.tensors.at(name + ".s");
                const bool has_o = st.tensors.count(name + ".o") != 0;
                kind = has_o ? 3 : (ts.dtype == "F16" ? 2 : 1);
                const size_t qb = kind == 1 ? (size_t)N * K : (size_t)N * K / 2;
                const size_t sb = kind == 1 ? (size_t)N * 4 : (size_t)N * K / 16;
                if (!getraw(name + ".q", qb, kind == 1 ? "I8" : "U8")) return false;
                if (expand_bf16) {   // a big card: the matrix goes back to bf16 (cuBLAS tensor cores beat the skinny GEMV from 16 slots on), the download stayed small
                    uint8_t* tq = dalloc<uint8_t>(qb);
                    uint8_t* ts_ = dalloc<uint8_t>(sb);
                    INK_CK(cudaMemcpy(tq, host.data(), qb, cudaMemcpyHostToDevice));
                    if (!getraw(name + ".s", sb, kind == 1 ? "F32" : (kind == 2 ? "F16" : "U8"))) { cudaFree(tq); cudaFree(ts_); return false; }
                    INK_CK(cudaMemcpy(ts_, host.data(), sb, cudaMemcpyHostToDevice));
                    float wo = 1.0f;
                    if (kind == 3) {
                        if (!getraw(name + ".o", 4, "F32")) { cudaFree(tq); cudaFree(ts_); return false; }
                        std::memcpy(&wo, host.data(), 4);
                    }
                    uint16_t* ob = wal<uint16_t>((size_t)N * K);
                    if (kind == 1) df2::k_df2_q8row_to_bf16<<<1024, 256>>>((const int8_t*)tq, (const float*)ts_, ob, N, K);
                    else if (kind == 2) df2::k_df2_q4_to_bf16<<<1024, 256>>>(tq, (const __half*)ts_, ob, N, K);
                    else df2::k_df2_sf4_to_bf16<<<1024, 256>>>(tq, ts_, wo, ob, N, K);
                    INK_CK(cudaDeviceSynchronize());
                    cudaFree(tq);
                    cudaFree(ts_);
                    q = ob;
                    sc = nullptr;
                    kind = 0;
                    outer = 1.0f;
                    return true;
                }
                q = wal<uint8_t>(qb);
                INK_CK(cudaMemcpy(q, host.data(), qb, cudaMemcpyHostToDevice));
                if (!getraw(name + ".s", sb, kind == 1 ? "F32" : (kind == 2 ? "F16" : "U8"))) return false;
                sc = wal<uint8_t>(sb);
                INK_CK(cudaMemcpy(sc, host.data(), sb, cudaMemcpyHostToDevice));
                if (kind == 3) {
                    if (!getraw(name + ".o", 4, "F32")) return false;
                    std::memcpy(&outer, host.data(), 4);
                }
                return true;
            }
            if (!get(name, (size_t)N * K * 2)) return false;
            INK_CK(cudaMemcpy(tmp, host.data(), (size_t)N * K * 2, cudaMemcpyHostToDevice));
            if (kind == 0) {
                q = wal<uint16_t>((size_t)N * K);
                sc = nullptr;
                INK_CK(cudaMemcpy(q, host.data(), (size_t)N * K * 2, cudaMemcpyHostToDevice));
            } else if (kind == 2) {
                q = wal<uint8_t>((size_t)N * K / 2);
                sc = wal<__half>((size_t)N * (K / 32));
                df2::k_df2_bf16_to_q4<<<N, 256>>>(tmp, (uint8_t*)q, (__half*)sc, N, K);
            } else if (kind == 3) {
                q = wal<uint8_t>((size_t)N * K / 2);
                sc = wal<uint8_t>((size_t)N * K / 16);
                INK_CK(cudaMemset(d_amax, 0, 4));
                df2::k_df2_amax_bf16<<<256, 256>>>(tmp, (size_t)N * K, d_amax);
                uint32_t am_bits = 0;
                INK_CK(cudaMemcpy(&am_bits, d_amax, 4, cudaMemcpyDeviceToHost));
                float am;
                std::memcpy(&am, &am_bits, 4);
                outer = std::max(am / (6.0f * 448.0f), 1e-12f);
                df2::k_df2_bf16_to_sf4<<<N, 256>>>(tmp, (uint8_t*)q, (uint8_t*)sc, outer, N, K);
            } else {
                q = wal<int8_t>((size_t)N * K);
                sc = wal<float>((size_t)N);
                k_ink_bf16_to_q8row<<<N, 256>>>(tmp, (float*)sc, (int8_t*)q, N, N, K);
            }
            INK_CK(cudaDeviceSynchronize());
            return true;
        };
        bool ok = up_bf16("fc.weight", (size_t)df2::kHidden * df2::kTargets * df2::kHidden, fc) &&
                  up_f32("hidden_norm.weight", df2::kHidden, hidden_norm) && up_f32("norm.weight", df2::kHidden, norm) &&
                  up_bf16("candidate_selector.hidden_projection.weight", (size_t)df2::kRank * df2::kHidden, sel_h);
        for (int l = 0; ok && l < df2::kLayers; ++l) {
            Layer& y = L[l];
            const std::string p = "layers." + std::to_string(l) + ".";
            y.kq = y.ko = var.attn;
            y.kg = y.ku = y.kd = var.ffn;
            ok = up_q8(p + "self_attn.q_proj.weight", df2::kQ, df2::kHidden, y.kq, y.wq, y.sq, y.oq) &&
                 up_q8(p + "self_attn.o_proj.weight", df2::kHidden, df2::kQ, y.ko, y.wo, y.so, y.oo) &&
                 up_q8(p + "mlp.gate_proj.weight", df2::kFF, df2::kHidden, y.kg, y.wg, y.sg, y.og) &&
                 up_q8(p + "mlp.up_proj.weight", df2::kFF, df2::kHidden, y.ku, y.wu, y.su, y.ou) &&
                 up_q8(p + "mlp.down_proj.weight", df2::kHidden, df2::kFF, y.kd, y.wd, y.sd, y.od) &&
                 up_bf16(p + "self_attn.k_proj.weight", (size_t)df2::kKV * df2::kHidden, y.wkv, 0, true, (size_t)2 * df2::kKV * df2::kHidden) &&
                 up_bf16(p + "self_attn.v_proj.weight", (size_t)df2::kKV * df2::kHidden, y.wkv, (size_t)df2::kKV * df2::kHidden, false) &&
                 up_bf16(p + "attention_conv.kernel_projection.weight", (size_t)df2::kDyn * df2::kHidden, y.conv_a) &&
                 up_bf16(p + "mlp_conv.kernel_projection.weight", (size_t)df2::kDyn * df2::kHidden, y.conv_f) &&
                 up_f32(p + "attention_conv.base_kernel", (size_t)4 * df2::kHidden, y.base_a) &&
                 up_f32(p + "mlp_conv.base_kernel", (size_t)4 * df2::kHidden, y.base_f) &&
                 up_f32(p + "input_layernorm.weight", df2::kHidden, y.attn_norm) &&
                 up_f32(p + "post_attention_layernorm.weight", df2::kHidden, y.post_norm) &&
                 up_f32(p + "self_attn.q_norm.weight", df2::kHeadDim, y.q_norm) && up_f32(p + "self_attn.k_norm.weight", df2::kHeadDim, y.k_norm);
        }
        cudaFree(tmp);
        if (!ok) return false;
        // selector codebooks: bf16 [201024][256]; the first V rows are used. Mapped host memory on the tight-VRAM cards (7 steps x 17 rows of 512 B per draft).
        {
            size_t vf = 0, vt = 0;
            (void)ink_cuda_mem_info(&vf, &vt);
            host_codebooks = m.all_sf4_vram || (vt > 0 && vt < (24ull << 30));   // 206 MiB that 16 GB cards cannot spare; 7 x 17 rows of 512 B per draft
        }
        if (const char* e = std::getenv("STRATA_INK_DFLASH2_CODEBOOK")) host_codebooks = (std::string(e) == "host");
        for (int which = 0; which < 2; ++which) {
            const std::string name = which == 0 ? "candidate_selector.predecessor_codebook" : "candidate_selector.successor_codebook";
            const size_t n = (size_t)V * df2::kRank;
            if (!get(name, n * 2, true)) return false;
            uint16_t*& dev = which == 0 ? pred : succ;
            void*& raw = which == 0 ? pred_raw : succ_raw;
            if (host_codebooks) {
                if (!InkCudaRunner::alloc_mapped_host<uint16_t>(n, raw, dev)) { err = "pinned host memory for the DFlash 2 codebooks"; return false; }
                std::memcpy(raw, host.data(), n * 2);
            } else {
                dev = wal<uint16_t>(n);
                INK_CK(cudaMemcpy(dev, host.data(), n * 2, cudaMemcpyHostToDevice));
            }
        }
        // the mask row: embed(mask_id) RMS-normed, scaled by the target's embed norm
        mask_row = wal<float>(df2::kHidden);
        {
            int32_t* d_t = dalloc<int32_t>(1);
            INK_CK(cudaMemcpy(d_t, &mask_id, 4, cudaMemcpyHostToDevice));
            const auto& ei = m.tinfo("model.llm.embed.weight");
            k_ink_embed_norm<<<1, 256, 0, r.st>>>(m.embed_ptr(), ei.fmt, (const float*)m.trunk_ptr("model.llm.embed_norm.weight"), d_t, nullptr, nullptr,
                                                  mask_row, 1, df2::kHidden);
            INK_CK(cudaStreamSynchronize(r.st));
            cudaFree(d_t);
        }
        if (!alloc_scratch(err)) return false;
        if (const char* dump = std::getenv("STRATA_INK_DFLASH2_DUMP")) {
            if (!dump_sidecar(dump)) std::fprintf(stderr, "[InkDFlash2] could not write %s\n", dump);
            else std::fprintf(stderr, "[InkDFlash2] wrote the pre-quantized sidecar %s\n", dump);
        }
        return true;
    }

    // Writes the weights as they sit on the card (the engine's own quantization) as a safetensors file: format "strata-dflash2-q1".
    // bf16 / f32 tensors under the checkpoint's names; a quantized matrix NAME as NAME.q (I8 or U8 codes), NAME.s (F32 row scales, F16 scales per 32, or UE4M3 scales per 16),
    // NAME.o (F32 tensor scale, SF4 only); __metadata__ {"format": "strata-dflash2-q1", "variant": "q8" | "q4" | "sf4" | "mix" | "mixq4" | "bf16"}.
    bool dump_sidecar(const std::string& path_out) {
        if (f32) return false;   // the matrices sit as FP32 here; the sidecar format is BF16
        struct It { std::string name, dtype; std::vector<int64_t> shape; const void* dev; const void* host; size_t bytes; };
        std::vector<It> its;
        std::vector<float> outers;
        outers.reserve(32);
        auto add = [&](const std::string& n, const char* dt, std::vector<int64_t> sh, const void* d, size_t b) { its.push_back({n, dt, sh, d, nullptr, b}); };
        const int64_t H = df2::kHidden;
        add("fc.weight", "BF16", {H, 4 * H}, fc, (size_t)H * 4 * H * 2);
        add("hidden_norm.weight", "F32", {H}, hidden_norm, (size_t)H * 4);
        add("norm.weight", "F32", {H}, norm, (size_t)H * 4);
        add("candidate_selector.hidden_projection.weight", "BF16", {df2::kRank, H}, sel_h, (size_t)df2::kRank * H * 2);
        add("candidate_selector.predecessor_codebook", "BF16", {V, df2::kRank}, pred, (size_t)V * df2::kRank * 2);
        add("candidate_selector.successor_codebook", "BF16", {V, df2::kRank}, succ, (size_t)V * df2::kRank * 2);
        for (int l = 0; l < df2::kLayers; ++l) {
            const Layer& y = L[l];
            const std::string p = "layers." + std::to_string(l) + ".";
            add(p + "self_attn.k_proj.weight", "BF16", {df2::kKV, H}, y.wkv, (size_t)df2::kKV * H * 2);
            add(p + "self_attn.v_proj.weight", "BF16", {df2::kKV, H}, y.wkv + (size_t)df2::kKV * H, (size_t)df2::kKV * H * 2);
            add(p + "attention_conv.kernel_projection.weight", "BF16", {df2::kDyn, H}, y.conv_a, (size_t)df2::kDyn * H * 2);
            add(p + "mlp_conv.kernel_projection.weight", "BF16", {df2::kDyn, H}, y.conv_f, (size_t)df2::kDyn * H * 2);
            add(p + "attention_conv.base_kernel", "F32", {2, 2, H}, y.base_a, (size_t)4 * H * 4);
            add(p + "mlp_conv.base_kernel", "F32", {2, 2, H}, y.base_f, (size_t)4 * H * 4);
            add(p + "input_layernorm.weight", "F32", {H}, y.attn_norm, (size_t)H * 4);
            add(p + "post_attention_layernorm.weight", "F32", {H}, y.post_norm, (size_t)H * 4);
            add(p + "self_attn.q_norm.weight", "F32", {df2::kHeadDim}, y.q_norm, (size_t)df2::kHeadDim * 4);
            add(p + "self_attn.k_norm.weight", "F32", {df2::kHeadDim}, y.k_norm, (size_t)df2::kHeadDim * 4);
            struct Big { const char* nm; int64_t N, K; int kind; const void* w; const void* sc; float outer; };
            const Big bigs[5] = {{"self_attn.q_proj.weight", df2::kQ, H, y.kq, y.wq, y.sq, y.oq}, {"self_attn.o_proj.weight", H, df2::kQ, y.ko, y.wo, y.so, y.oo},
                                 {"mlp.gate_proj.weight", df2::kFF, H, y.kg, y.wg, y.sg, y.og}, {"mlp.up_proj.weight", df2::kFF, H, y.ku, y.wu, y.su, y.ou},
                                 {"mlp.down_proj.weight", H, df2::kFF, y.kd, y.wd, y.sd, y.od}};
            for (const Big& b : bigs) {
                const std::string n = p + b.nm;
                if (b.kind == 0) add(n, "BF16", {b.N, b.K}, b.w, (size_t)b.N * b.K * 2);
                else if (b.kind == 1) { add(n + ".q", "I8", {b.N * b.K}, b.w, (size_t)b.N * b.K); add(n + ".s", "F32", {b.N}, b.sc, (size_t)b.N * 4); }
                else if (b.kind == 2) { add(n + ".q", "U8", {b.N * b.K / 2}, b.w, (size_t)b.N * b.K / 2); add(n + ".s", "F16", {b.N * b.K / 32}, b.sc, (size_t)b.N * b.K / 16); }
                else {
                    add(n + ".q", "U8", {b.N * b.K / 2}, b.w, (size_t)b.N * b.K / 2);
                    add(n + ".s", "U8", {b.N * b.K / 16}, b.sc, (size_t)b.N * b.K / 16);
                    outers.push_back(b.outer);
                    its.push_back({n + ".o", "F32", {1}, nullptr, &outers.back(), 4});
                }
            }
        }
        const char* names[] = {"bf16", "q8", "q4", "sf4"};
        std::string vname;
        if (var.attn == var.ffn) vname = names[var.attn];
        else vname = (var.attn == 1 && var.ffn == 3) ? "mix" : ((var.attn == 1 && var.ffn == 2) ? "mixq4" : "mixed");
        std::string js = "{\"__metadata__\":{\"format\":\"strata-dflash2-q1\",\"variant\":\"" + vname + "\"}";
        size_t off = 0;
        for (const It& it : its) {
            js += ",\"" + it.name + "\":{\"dtype\":\"" + it.dtype + "\",\"shape\":[";
            for (size_t i = 0; i < it.shape.size(); ++i) js += (i ? "," : "") + std::to_string(it.shape[i]);
            js += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + it.bytes) + "]}";
            off += it.bytes;
        }
        js += "}";
        while (js.size() % 8) js += ' ';
        std::FILE* f = std::fopen(path_out.c_str(), "wb");
        if (!f) return false;
        unsigned char len8[8];
        uint64_t n = js.size();
        for (int k = 0; k < 8; ++k) len8[k] = (unsigned char)((n >> (8 * k)) & 0xFF);
        bool ok = std::fwrite(len8, 1, 8, f) == 8 && std::fwrite(js.data(), 1, js.size(), f) == js.size();
        std::vector<unsigned char> buf;
        for (const It& it : its) {
            if (!ok) break;
            buf.resize(it.bytes);
            if (it.host) std::memcpy(buf.data(), it.host, it.bytes);
            else if (cudaMemcpy(buf.data(), it.dev, it.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { ok = false; break; }
            ok = std::fwrite(buf.data(), 1, it.bytes, f) == it.bytes;
        }
        std::fclose(f);
        return ok;
    }

    bool alloc_scratch(std::string& err) {
        using namespace df2;
        const int B = max_slots;
        const int R = B * kBlock, R7 = B * kDraft;
        const int R7p = ((R7 + 3) / 4) * 4;
        max_batch_rows = std::max(r.max_batch, 64);
        d_meta = sal<int32_t>((size_t)B * 6 + 8);
        d_pos = sal<int32_t>(R);
        d_srow = sal<int32_t>(R);
        d_anchor_emb = sal<float>((size_t)B * kHidden);
        d_x = sal<float>((size_t)R * kHidden);
        d_xn = sal<float>((size_t)R * kHidden);
        d_xn_bf = f32 ? nullptr : sal<__nv_bfloat16>((size_t)R * kHidden);
        d_c = sal<float>((size_t)R * kHidden);
        d_c_bf = f32 ? nullptr : sal<__nv_bfloat16>((size_t)R * kHidden);
        d_dyn = sal<float>((size_t)R * kDyn);
        d_q = sal<float>((size_t)R * kQ);
        d_kb = sal<float>((size_t)R * kKV);
        d_vb = sal<float>((size_t)R * kKV);
        d_attn = sal<float>((size_t)R * kQ);
        d_z = sal<float>((size_t)R * kHidden);
        d_g = sal<float>((size_t)R * kFF);
        d_u = sal<float>((size_t)R * kFF);
        d_act = sal<float>((size_t)R * kFF);
        d_aq = sal<int8_t>((size_t)R * kFF);
        d_xb = f32 ? nullptr : sal<__nv_bfloat16>((size_t)R * kFF);
        d_as = sal<float>((size_t)R * (kFF / 32));
        const size_t bsplit = (size_t)std::max(24, B);   // B x split of the attention partials: split_for(B) is 1 above 24 slots, so it is B there (32 slots overran the old 24)
        p_m = sal<float>((size_t)8 * bsplit * 32 + 64);
        p_l = sal<float>((size_t)8 * bsplit * 32 + 64);
        p_o = sal<float>((size_t)8 * bsplit * 32 * kHeadDim + 64);
        d_hd = sal<float>((size_t)R * kHidden);
        d_hd_bf = f32 ? nullptr : sal<__nv_bfloat16>((size_t)R * kHidden);
        d_gsel = sal<float>((size_t)R * kRank);
        d_hq8 = sal<int8_t>((size_t)R7p * kHidden);
        d_hsc = sal<float>((size_t)R7p);
        d_tile = sal<int32_t>((size_t)R7p * vt_nom);
        d_part = sal<float2>((size_t)R7 * 64 * kTopK);
        d_part_lse = sal<float2>((size_t)R7 * 64);
        d_cand = sal<int32_t>((size_t)R7 * 2 * kTopK);
        d_tail = sal<float>((size_t)R7 * 2);
        d_ids = sal<int32_t>((size_t)B * kDraft);
        d_conf = sal<float>((size_t)B * kDraft);
        h_ids = halloc_pinned<int32_t>((size_t)B * kDraft);
        h_conf = halloc_pinned<float>((size_t)B * kDraft);
        d_ix_bf = f32 ? nullptr : sal<__nv_bfloat16>((size_t)kCh * kTargets * kHidden);
        d_icn_bf = f32 ? nullptr : sal<__nv_bfloat16>((size_t)kCh * kHidden);
        d_ix_f = f32 ? sal<float>((size_t)kCh * kTargets * kHidden) : nullptr;
        d_icn_f = f32 ? sal<float>((size_t)kCh * kHidden) : nullptr;
        d_ic = sal<float>((size_t)kCh * kHidden);
        d_ikc = sal<float>((size_t)kCh * kKV);
        d_ivc = sal<float>((size_t)kCh * kKV);
        d_ing = sal<int32_t>((size_t)3 * max_batch_rows);
        stage_stride = (size_t)std::max(3 * max_batch_rows, B * 6 + 8);
        h_stage = halloc_pinned<int32_t>(stage_stride * kStageRing);
        ring_k_h.assign(B, nullptr);
        ring_v_h.assign(B, nullptr);
        slot_dead.assign(B, 0);
        lo.assign(B, 0);
        hi.assign(B, 0);
        {
            // pointer tables (device), filled as rings are allocated
            __half** t = nullptr;
            if (cudaMalloc(&t, (size_t)2 * B * sizeof(__half*)) != cudaSuccess) { err = "DFlash 2 ring tables"; return false; }
            cudaMemset(t, 0, (size_t)2 * B * sizeof(__half*));
            d_rk_tab = t;
            d_rv_tab = t + B;
        }
        // the dp4a GEMV's dynamic shared memory: 9 * K bytes for K up to 6912
        // (below sm_80 the GEMVs take 4 activation rows per block where 8 rows would need more than 48 KB: nothing to opt in to)
        if (!f32) {
            INK_CK(cudaFuncSetAttribute(df2::k_df2_gemv_q8<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 9 * kFF + 64));
            INK_CK(cudaFuncSetAttribute(df2::k_df2_gemv_q4<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 9 * kFF + 64));
            INK_CK(cudaFuncSetAttribute(df2::k_df2_gemv_sf4<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 9 * kFF + 64));
        }
        // the first rings are taken now, inside the start-up reserve; later slots are taken at their first use
        // As many as fit: on a 16 GB card in compact mode the mix leaves room for 6 or 7 of the 8 (11-26: all 8 or nothing meant no drafter at all there); a slot without a ring decodes plainly.
        int with_ring = 0;
        for (int s = 0; s < std::min(max_slots, 8); ++s) {
            size_t vf_r = 0, vt_r = 0;
            const size_t ring_bytes = (size_t)2 * df2::kLayers * df2::kRingCap * df2::kKV * sizeof(__half);
            if (ink_cuda_mem_info(&vf_r, &vt_r) == cudaSuccess && vf_r < ring_bytes + (128ull << 20)) break;   // the card keeps 128 MiB free: no more rings (the first ring of a 16 GB card still fits: 161 MiB free at the end)
            if (!ensure_slot(s)) break;
            ++with_ring;
        }
        if (with_ring == 0) { err = "not enough GPU memory for the DFlash 2 context rings"; return false; }
        for (int s = with_ring; s < std::min(max_slots, 8); ++s) slot_dead[s] = 1;   // never retried later: the memory they would take is what the rest of the engine needs
        if (with_ring < std::min(max_slots, 8)) std::fprintf(stderr, "[InkDFlash2] only %d of %d context rings fit on this card; the other slots decode without a drafter\n", with_ring, std::min(max_slots, 8));
        ready = true;
        return true;
    }

    size_t ring_bytes_per_slot() const { return (size_t)2 * df2::kLayers * df2::kRingCap * df2::kKV * sizeof(__half); }

    bool ensure_slot(int s) {
        if (s < 0 || s >= max_slots || slot_dead[s]) return false;
        if (ring_k_h[s]) return true;
        const size_t one = (size_t)df2::kLayers * df2::kRingCap * df2::kKV * sizeof(__half);
        __half *k = nullptr, *v = nullptr;
        // The first 8 slots' rings are inside the start-up reserve. A later slot takes its ring only while 1 GiB stays free for the rest of the engine (the multi-slot paths allocate
        // working buffers lazily: at 64 slots on a 32 GB card the rings took everything and the engine died, 11-26); a slot that cannot has no drafts and decodes plainly.
        // Slots 1..7 (start-up) take their ring only while a 128 MiB crash margin (the same as the start-up loop in init) stays free after it: "as many as fit" ended a --parallel 8 start on a 5060 Ti with 13 MiB free and
        // the first launch probe died with out of memory (11-26); slot 0's ring is always taken (no ring, no drafter).
        bool room = true;
        if (s >= 1) {
            size_t vf = 0, vt = 0;
            const size_t margin = (s >= 8) ? (1024ull << 20) : (128ull << 20);
            room = ink_cuda_mem_info(&vf, &vt) == cudaSuccess && vf > 2 * one + margin;
        }
        if (!room || cudaMalloc(&k, one) != cudaSuccess || cudaMalloc(&v, one) != cudaSuccess) {
            (void)cudaGetLastError();
            if (k) cudaFree(k);
            slot_dead[s] = 1;
            std::fprintf(stderr, "[InkDFlash2] no GPU memory for slot %d's context ring (%.0f MiB); that slot does not draft\n", s, (double)(2 * one) / 1048576.0);
            return false;
        }
        cudaMemset(k, 0, one);
        cudaMemset(v, 0, one);
        ring_k_h[s] = k;
        ring_v_h[s] = v;
        INK_CK(cudaMemcpy(d_rk_tab + s, &k, sizeof(k), cudaMemcpyHostToDevice));
        INK_CK(cudaMemcpy(d_rv_tab + s, &v, sizeof(v), cudaMemcpyHostToDevice));
        return true;
    }

    // ---- GEMM helpers -----------------------------------------------------------------------------------------------------
    // Y[M][N] fp32 = X[M][K] bf16 * W[N][K]^T bf16
    void gemm_bf16(const uint16_t* W, const __nv_bfloat16* X, float* Y, int M, int N, int K) {
        const float a = 1.0f, b = 0.0f;
        INK_CUBLAS_CK(cublasGemmEx(r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &b, Y, CUDA_R_32F, N,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // the BF16 matrices of the checkpoint (fc, K / V, convolution kernels, selector): bf16 on sm_80+; FP32 (the same values) below it. Xb is the bf16 copy of the activation, Xf the fp32 one.
    void gemm_w(const uint16_t* W, const __nv_bfloat16* Xb, const float* Xf, float* Y, int M, int N, int K) {
        if (!f32) { gemm_bf16(W, Xb, Y, M, N, K); return; }
        const float a = 1.0f, b = 0.0f;
        INK_CUBLAS_CK(cublasGemmEx(r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, W, CUDA_R_32F, K, Xf, CUDA_R_32F, K, &b, Y, CUDA_R_32F, N,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    // a matrix offset in elements: bf16 slots, or fp32 elements of the same matrix when f32
    const uint16_t* woff(const uint16_t* W, size_t elems) const { return f32 ? reinterpret_cast<const uint16_t*>(reinterpret_cast<const float*>(W) + elems) : W + elems; }
    // Y[M][N] fp32 = X[M][K] fp32 * W[N][K]^T; kind 0 bf16 (cuBLAS), 1 Q8 row-scaled, 2 4-bit groups, 3 SF4 (the last three on activations quantized per 32 values; quant = false reuses the last)
    void mv(const void* W, const void* wsc, int kind, float outer, const float* X, float* Y, int M, int N, int K, bool quant = true) {
        using namespace df2;
        if (kind == 0) {
            if (quant) k_ink_f32_to_bf16<<<(unsigned)((((size_t)M * K + 3) / 4 + 255) / 256), 256, 0, r.st>>>(X, d_xb, (size_t)M * K);
            gemm_bf16((const uint16_t*)W, d_xb, Y, M, N, K);
            return;
        }
        if (quant) k_df2_quant_act<<<dim3((K / 32 + 7) / 8, M), 256, 0, r.st>>>(X, d_aq, d_as, M, K, 0);
        const int nrb = (N + 7) / 8;
        // 8 activation rows per block need 9 * K + 16 bytes of shared memory: 62 KB for the down projection, more than the 48 KB of an sm_60 / sm_61 block (and 64 KB of a Turing one is
        // tight): below sm_80 the GEMVs take 4 rows (4.5 * K + 16 bytes) there
        if (f32 && (size_t)9 * K + 16 > 48u * 1024u) {
            const int nch = (M + 3) / 4;
            const size_t sm = (size_t)9 * K / 2 + 16;
            if (kind == 2) k_df2_gemv_q4<4><<<nrb * nch, 256, sm, r.st>>>((const uint8_t*)W, (const __half*)wsc, d_aq, d_as, Y, M, N, K, nch);
            else if (kind == 3) k_df2_gemv_sf4<4><<<nrb * nch, 256, sm, r.st>>>((const uint8_t*)W, (const uint8_t*)wsc, outer, d_aq, d_as, Y, M, N, K, nch);
            else k_df2_gemv_q8<4><<<nrb * nch, 256, sm, r.st>>>((const int8_t*)W, (const float*)wsc, d_aq, d_as, Y, M, N, K, nch);
            return;
        }
        const int nch = (M + kGemvMR - 1) / kGemvMR;
        const size_t sm = (size_t)9 * K + 16;
        if (kind == 2) k_df2_gemv_q4<kGemvMR><<<nrb * nch, 256, sm, r.st>>>((const uint8_t*)W, (const __half*)wsc, d_aq, d_as, Y, M, N, K, nch);
        else if (kind == 3) k_df2_gemv_sf4<kGemvMR><<<nrb * nch, 256, sm, r.st>>>((const uint8_t*)W, (const uint8_t*)wsc, outer, d_aq, d_as, Y, M, N, K, nch);
        else k_df2_gemv_q8<kGemvMR><<<nrb * nch, 256, sm, r.st>>>((const int8_t*)W, (const float*)wsc, d_aq, d_as, Y, M, N, K, nch);
    }

    // ---- context ingest ---------------------------------------------------------------------------------------------------
    // Rows i = 0 .. n-1 of the four fp32 tap planes (row h_src[i], or i): the target's features at position h_pos[i] of slot h_slot[i].
    void ingest(const float* const taps[4], const int32_t* h_src, const int32_t* h_slot, const int32_t* h_pos, int n) {
        using namespace df2;
        if (!ready || n <= 0) return;
        // bookkeeping and the rows that can still matter (the last kKeep positions of each slot)
        constexpr int kKeep = kRingCap - 16;
        std::vector<int32_t> src, sl, ps;
        src.reserve(n); sl.reserve(n); ps.reserve(n);
        std::vector<int64_t> mx(max_slots, -1), mn(max_slots, INT64_MAX);
        std::vector<int> cnt(max_slots, 0);
        for (int i = 0; i < n; ++i) {
            const int s = h_slot[i];
            if (s < 0 || s >= max_slots) continue;
            mx[s] = std::max<int64_t>(mx[s], h_pos[i]);
            mn[s] = std::min<int64_t>(mn[s], h_pos[i]);
            ++cnt[s];
        }
        for (int s = 0; s < max_slots; ++s) {
            if (cnt[s] == 0) continue;
            if (!ensure_slot(s)) continue;
            if (mn[s] != hi[s]) lo[s] = mn[s];
            hi[s] = mx[s] + 1;
            lo[s] = std::max<int64_t>(lo[s], hi[s] - kKeep);
        }
        for (int i = 0; i < n; ++i) {
            const int s = h_slot[i];
            if (s < 0 || s >= max_slots || slot_dead[s]) continue;
            if (h_pos[i] < hi[s] - kKeep) continue;
            src.push_back(h_src ? h_src[i] : i);
            sl.push_back(s);
            ps.push_back(h_pos[i]);
        }
        const int nn = (int)src.size();
        if (nn == 0) return;
        n_ingest_rows += nn;
        cudaEvent_t pe0 = nullptr, pe1 = nullptr;
        if (prof) { cudaEventCreate(&pe0); cudaEventCreate(&pe1); cudaEventRecord(pe0, r.st); ++prof_ingest_calls; }
        int32_t* st = h_stage + (size_t)(stage_idx++ % kStageRing) * stage_stride;
        const int cap_rows = max_batch_rows;
        for (int off = 0; off < nn; off += cap_rows) {
            const int nb = std::min(cap_rows, nn - off);
            std::memcpy(st, src.data() + off, (size_t)nb * 4);
            std::memcpy(st + nb, sl.data() + off, (size_t)nb * 4);
            std::memcpy(st + 2 * nb, ps.data() + off, (size_t)nb * 4);
            int32_t* d_src = d_ing;
            int32_t* d_sl = d_ing + nb;
            int32_t* d_ps = d_ing + 2 * nb;
            INK_CK(cudaMemcpyAsync(d_ing, st, (size_t)3 * nb * 4, cudaMemcpyHostToDevice, r.st));
            for (int c0 = 0; c0 < nb; c0 += kCh) {
                const int cn = std::min(kCh, nb - c0);
                if (f32) k_df2_gather_taps_f32<<<cn, 256, 0, r.st>>>(taps[0], taps[1], taps[2], taps[3], d_src, c0, cn, d_ix_f);
                else k_df2_gather_taps<<<cn, 256, 0, r.st>>>(taps[0], taps[1], taps[2], taps[3], d_src, c0, cn, d_ix_bf);
                gemm_w(fc, d_ix_bf, d_ix_f, d_ic, cn, kHidden, kTargets * kHidden);
                k_ink_rms_norm<<<cn, 256, 0, r.st>>>(d_ic, hidden_norm, f32 ? d_icn_f : nullptr, f32 ? nullptr : d_icn_bf, cn, kHidden);
                for (int l = 0; l < kLayers; ++l) {
                    gemm_w(L[l].wkv, d_icn_bf, d_icn_f, d_ikc, cn, kKV, kHidden);
                    gemm_w(woff(L[l].wkv, (size_t)kKV * kHidden), d_icn_bf, d_icn_f, d_ivc, cn, kKV, kHidden);
                    k_df2_rmsnorm_rope<<<dim3(kKvHeads, cn), 128, 0, r.st>>>(d_ikc, L[l].k_norm, d_ps + c0, theta, cn, kKvHeads);
                    k_df2_ring_write<<<cn, 256, 0, r.st>>>(d_ikc, d_ivc, d_sl + c0, d_ps + c0, l, d_rk_tab, d_rv_tab, cn);
                }
            }
            // the pinned stage is read by the async copy above; the ring of kStageRing buffers keeps the next calls off it
            if (off + cap_rows < nn) INK_CK(cudaStreamSynchronize(r.st));
        }
        if (prof) { cudaEventRecord(pe1, r.st); prof_pend.emplace_back(pe0, pe1); }
    }

    void ingest_seq(const float* const taps[4], int slot, int p0, int M) {
        std::vector<int32_t> sl(M, slot), ps(M);
        for (int i = 0; i < M; ++i) ps[i] = p0 + i;
        ingest(taps, nullptr, sl.data(), ps.data(), M);
    }

    // the context of a slot is gone (a conversation was restored, the slot was reused): drafts see no context until the next ingest
    void reset_slot(int slot, int64_t pos) {
        if (slot < 0 || slot >= max_slots) return;
        lo[slot] = hi[slot] = pos;
    }

    // ---- the draft --------------------------------------------------------------------------------------------------------
    static int split_for(int B) { return std::max(1, std::min(df2::kMaxSplit, 24 / std::max(1, B))); }

    // B slots: slots[b], anchor tokens, block start positions pos0[b] (the anchor's position). out_ids / out_conf [B][7]. Returns false when nothing was drafted.
    bool draft(int B, const int32_t* slots, const int32_t* anchors, const int32_t* pos0, int32_t* out_ids, float* out_conf, const int32_t* ctx_hi = nullptr) {
        using namespace df2;
        if (!ready || B <= 0 || B > max_slots) return false;
        const int R = B * kBlock, R7 = B * kDraft;
        for (int b = 0; b < B; ++b)
            if (!ensure_slot(slots[b])) return false;
        int32_t* st = h_stage + (size_t)(stage_idx++ % kStageRing) * stage_stride;
        for (int b = 0; b < B; ++b) {
            const int s = slots[b];
            int64_t lo_eff = std::max<int64_t>(lo[s], (int64_t)pos0[b] - (kWindow - 1));
            int64_t hc = ctx_hi ? (int64_t)ctx_hi[b] : (int64_t)pos0[b];   // where the context ends (before the block, or before the gap of a second block)
            hc = std::min<int64_t>(hc, hi[s]);                              // nothing newer than what was ingested
            if (lo_eff > pos0[b]) lo_eff = pos0[b];
            st[B * 5 + b] = (int32_t)hc;
            st[b * 4 + 0] = s;
            st[b * 4 + 1] = pos0[b];
            st[b * 4 + 2] = (int32_t)lo_eff;
            st[b * 4 + 3] = anchors[b];
            st[B * 4 + b] = anchors[b];
        }
        cudaEvent_t de0 = nullptr, de1 = nullptr;
        if (prof) { cudaEventCreate(&de0); cudaEventCreate(&de1); cudaEventRecord(de0, r.st); }
        INK_CK(cudaMemcpyAsync(d_meta, st, (size_t)(B * 4 + 2 * B) * 4, cudaMemcpyHostToDevice, r.st));
        const int32_t* d_anchor = d_meta + B * 4;
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<B, 256, 0, r.st>>>(m.embed_ptr(), ei.fmt, (const float*)m.trunk_ptr("model.llm.embed_norm.weight"), d_anchor, nullptr, nullptr,
                                              d_anchor_emb, B, kHidden);
        k_df2_build_x<<<R, 256, 0, r.st>>>(d_anchor_emb, mask_row, d_x, B);
        k_df2_block_pos<<<(R + 127) / 128, 128, 0, r.st>>>(d_meta, B, d_pos, d_srow);
        const int split = split_for(B);
        for (int l = 0; l < kLayers; ++l) {
            const Layer& y = L[l];
            // ---- attention sublayer
            k_ink_rms_norm<<<R, 256, 0, r.st>>>(d_x, y.attn_norm, d_xn, d_xn_bf, R, kHidden);
            gemm_w(y.conv_a, d_xn_bf, d_xn, d_dyn, R, kDyn, kHidden);
            k_df2_conv_pre<<<R, 256, 0, r.st>>>(d_xn, d_dyn, y.base_a, d_c, d_c_bf, R);
            mv(y.wq, y.sq, y.kq, y.oq, d_c, d_q, R, kQ, kHidden);
            gemm_w(y.wkv, d_c_bf, d_c, d_kb, R, kKV, kHidden);
            gemm_w(woff(y.wkv, (size_t)kKV * kHidden), d_c_bf, d_c, d_vb, R, kKV, kHidden);
            k_df2_rmsnorm_rope<<<dim3(kHeads, R), 128, 0, r.st>>>(d_q, y.q_norm, d_pos, theta, R, kHeads);
            k_df2_rmsnorm_rope<<<dim3(kKvHeads, R), 128, 0, r.st>>>(d_kb, y.k_norm, d_pos, theta, R, kKvHeads);
            // block rows' K / V into the ring (cells of their positions), then attend
            k_df2_ring_write<<<R, 256, 0, r.st>>>(d_kb, d_vb, d_srow, d_pos, l, d_rk_tab, d_rv_tab, R);
            k_df2_attn_part<<<dim3(kKvHeads, B, split), 256, 0, r.st>>>(d_q, d_rk_tab, d_rv_tab, d_meta, d_meta + B * 5, l, split, p_m, p_l, p_o);
            k_df2_attn_combine<<<dim3(kKvHeads, B, 32), 128, 0, r.st>>>(p_m, p_l, p_o, split, d_attn);
            mv(y.wo, y.so, y.ko, y.oo, d_attn, d_z, R, kHidden, kQ);
            k_df2_conv_post_add<<<R, 256, 0, r.st>>>(d_z, d_dyn, y.base_a, d_x, R);
            // ---- feed-forward sublayer
            k_ink_rms_norm<<<R, 256, 0, r.st>>>(d_x, y.post_norm, d_xn, d_xn_bf, R, kHidden);
            gemm_w(y.conv_f, d_xn_bf, d_xn, d_dyn, R, kDyn, kHidden);
            k_df2_conv_pre<<<R, 256, 0, r.st>>>(d_xn, d_dyn, y.base_f, d_c, nullptr, R);
            // gate and up share the quantized activation
            mv(y.wg, y.sg, y.kg, y.og, d_c, d_g, R, kFF, kHidden);
            mv(y.wu, y.su, y.ku, y.ou, d_c, d_u, R, kFF, kHidden, false);
            k_df2_swiglu<<<(unsigned)(((size_t)R * kFF + 255) / 256), 256, 0, r.st>>>(d_g, d_u, d_act, (size_t)R * kFF);
            mv(y.wd, y.sd, y.kd, y.od, d_act, d_z, R, kHidden, kFF);
            k_df2_conv_post_add<<<R, 256, 0, r.st>>>(d_z, d_dyn, y.base_f, d_x, R);
        }
        // ---- head
        k_ink_rms_norm<<<R, 256, 0, r.st>>>(d_x, norm, d_hd, d_hd_bf, R, kHidden);
        gemm_w(sel_h, d_hd_bf, d_hd, d_gsel, R, kRank, kHidden);
        const int R7p = ((R7 + 3) / 4) * 4;
        k_dspark_quant_h_q8_slots<<<R7, 256, 0, r.st>>>(d_hd, d_hq8, d_hsc, B, kDraft);
        if (R7p > R7) INK_CK(cudaMemsetAsync(d_hq8 + (size_t)R7 * kHidden, 0, (size_t)(R7p - R7) * kHidden, r.st));
        const int32_t ialpha = 1, ibeta = 0;
        int chunk_base = 0;
        for (int ti = 0; ti < 8; ++ti) {
            const int v0 = ti * vt_nom;
            const int vt = std::max(0, std::min(vt_nom, V_gemm - v0));
            if (vt <= 0) continue;
            if (screen_cublas) {
                INK_CUBLAS_CK(cublasGemmEx(r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, vt, R7p, kHidden, &ialpha, m.unembed_screen.q8 + (size_t)v0 * kHidden, CUDA_R_8I,
                                           kHidden, d_hq8, CUDA_R_8I, kHidden, &ibeta, d_tile, CUDA_R_32I, vt, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                k_df2_screen_gemm<<<(vt + 7) / 8, 256, 0, r.st>>>(m.unembed_screen.q8 + (size_t)v0 * kHidden, d_hq8, d_tile, vt, R7p);
            }
            const int nchunk = (vt + kTopChunk - 1) / kTopChunk;
            k_df2_tile_part<<<dim3(nchunk, R7), 256, 0, r.st>>>(d_tile, m.unembed_screen.sc, d_hsc, v0, vt, V, cap, chunk_base, d_part, d_part_lse);
            chunk_base += nchunk;
        }
        k_df2_top16_merge<<<R7, 512, 0, r.st>>>(d_part, d_part_lse, chunk_base, V, cap, d_cand, d_tail);
        k_df2_select<<<B, 512, 0, r.st>>>(d_meta, d_cand, d_tail, d_gsel, pred, succ, kDraft, V, cap, d_ids, d_conf);
        INK_CK(cudaMemcpyAsync(h_ids, d_ids, (size_t)B * kDraft * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_conf, d_conf, (size_t)B * kDraft * 4, cudaMemcpyDeviceToHost, r.st));
        if (prof) cudaEventRecord(de1, r.st);
        INK_CK(cudaStreamSynchronize(r.st));
        std::memcpy(out_ids, h_ids, (size_t)B * kDraft * 4);
        std::memcpy(out_conf, h_conf, (size_t)B * kDraft * 4);
        ++n_drafts_run;
        if (prof) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, de0, de1) == cudaSuccess) prof_draft_ms += ms;
            cudaEventDestroy(de0);
            cudaEventDestroy(de1);
            if (n_drafts_run % 200 == 0) prof_print();
        }
        return true;
    }

    void free_all() {
        if (prof) prof_print();
        for (int s = 0; s < (int)ring_k_h.size(); ++s) {
            if (ring_k_h[s]) cudaFree(ring_k_h[s]);
            if (ring_v_h[s]) cudaFree(ring_v_h[s]);
        }
        ring_k_h.clear();
        ring_v_h.clear();
        if (d_rk_tab) cudaFree(d_rk_tab);
        d_rk_tab = d_rv_tab = nullptr;
        for (void* p : owned) cudaFree(p);
        owned.clear();
        if (d_amax) { cudaFree(d_amax); d_amax = nullptr; }
        if (pred_raw) { cudaFreeHost(pred_raw); pred_raw = nullptr; }
        if (succ_raw) { cudaFreeHost(succ_raw); succ_raw = nullptr; }
        ready = false;
    }
};

}  // namespace strata::ink
