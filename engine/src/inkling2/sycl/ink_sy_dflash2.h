// ink_sy_dflash2.h - Native Intel Arc SYCL SG16 DFlash 2 Block-Diffusion Drafter (dflash2.safetensors)
// CRITICAL BATTLEMAGE B70 RULE: 100% INK_REQD_SG16 (ZERO reqd_sub_group_size(8) anywhere!).
// Faithful SYCL port of engine/src/inkling2/cuda/ink_dflash2*.cuh:
//   - 5 SWA-2048 block-diffusion layers, H=2304, 32Q/8KV x 128, FF=6912, SwiGLU, 2-tap group-16 dynamic causal convs
//   - 4-tap ingest (visits 5, 23, 41 + final normed state) -> fc -> hidden_norm -> K/V FP16 ring (2112 cells)
//   - Rank-256 bilinear candidate selector (exact top-16 + full-vocab softcapped tail confidence)
//   - ceiling.py MoE expert-union adaptive draft width M* (1/4/6/8) + optional PLD suffix match
//   - On-GPU greedy verification + exact lossless stochastic verification (Leviathan et al., 2023)
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "../../image/vae/safetensors_lite.hpp"
#include "../ink_dflash2_mix.h"
#include "ink_sy_codecs.h"
#include "ink_sy_common.h"
#include "ink_sy_intelarc_cfg.h"
#include "ink_sy_ops.h"

namespace ink_sycl {

namespace df2_cfg {
constexpr int   kL       = 5;
constexpr int   kH       = 2304;
constexpr int   kQH      = 32;
constexpr int   kKVH     = 8;
constexpr int   kHD      = 128;
constexpr int   kQ       = kQH * kHD;    // 4096
constexpr int   kKV      = kKVH * kHD;   // 1024
constexpr int   kFF      = 6912;
constexpr int   kGroup   = 16;
constexpr int   kNG      = kH / kGroup;  // 144
constexpr int   kDyn     = 4 * kNG;      // 576
constexpr int   kBlock   = 8;
constexpr int   kRank    = 256;
constexpr int   kTopK    = 16;
constexpr int   kWindow  = 2048;
constexpr int   kRingCap = 2112;
constexpr float kTheta   = 10000.0f;
constexpr float kCap     = 30.0f;
constexpr float kEps     = 1e-6f;
constexpr int   kMaskId  = 200056;
constexpr int   kV       = 200058;
}  // namespace df2_cfg

inline bool df2_file_exists(const std::string& p) {
    if (p.empty()) return false;
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// DFlash 2 on SYCL loads by itself (a dflash2.safetensors beside the model, no switch) only when this is true. It is false for 0.2.23: the port is proven on one hybrid Arc card
// (A750: acceptance 70-90%, greedy tokens equal to plain), no resident Battlemage card has measured it yet (B70 CI). --draft on and STRATA_INK_DRAFTER=dflash2 load it.
inline constexpr bool kSyclDflash2DefaultOn = false;

inline bool df2_requested_env() {
    const char* e = std::getenv("STRATA_INK_DRAFTER");
    if (!e || !e[0]) return kSyclDflash2DefaultOn;
    std::string s = to_lower_ascii(e);
    return s == "dflash2" || s == "df2" || s == "auto" || s == "1";
}

inline std::string df2_check_file_or_dir(const std::string& p) {
    if (p.empty()) return "";
    if (df2_file_exists(p)) return p;
    std::string d = p;
    while (!d.empty() && (d.back() == '/' || d.back() == '\\')) d.pop_back();
    if (df2_file_exists(d + "/model.safetensors")) return d + "/model.safetensors";
    if (df2_file_exists(d + "/dflash2.safetensors")) return d + "/dflash2.safetensors";
    return "";
}

inline std::string df2_resolve_path(const std::string& model_path, const std::string& explicit_path = "") {
    if (!df2_requested_env()) return "";
    if (const char* e = std::getenv("STRATA_INK_DFLASH2"); e && e[0]) {
        std::string hit = df2_check_file_or_dir(e);
        if (!hit.empty()) return hit;
    }
    if (const char* e = std::getenv("STRATA_INK_DFLASH2_PATH"); e && e[0]) {
        std::string hit = df2_check_file_or_dir(e);
        if (!hit.empty()) return hit;
    }
    if (!explicit_path.empty()) {
        std::string hit = df2_check_file_or_dir(explicit_path);
        if (!hit.empty() && hit.size() > 12 &&
            hit.compare(hit.size() - 12, 12, ".safetensors") == 0) {
            return hit;
        }
    }
    std::string dir;
    size_t sl = model_path.find_last_of("/\\");
    if (sl != std::string::npos) dir = model_path.substr(0, sl + 1);
    std::string parent;
    if (dir.size() > 1) {
        size_t sl2 = dir.find_last_of("/\\", dir.size() - 2);
        if (sl2 != std::string::npos) parent = dir.substr(0, sl2 + 1);
    }
    const std::string cands[] = {
        dir + "dflash2.safetensors",
        dir + "dflash2/model.safetensors",
        dir + "Inkling-2-Mini-DFlash2/model.safetensors",
        dir + "Inkling-2-Mini-DFlash2/dflash2.safetensors",
        dir + "Inkling-2-Mini-DFlash2.safetensors",
        parent + "Inkling-2-Mini-DFlash2/model.safetensors",
        parent + "Inkling-2-Mini-DFlash2/dflash2.safetensors",
        parent + "dflash2.safetensors"
    };
    for (const auto& c : cands) {
        if (!c.empty() && df2_file_exists(c)) return c;
    }
    return "";
}

// ---- DFlash 2 SYCL SG16 Kernels ----

// Concatenate 4 taps [t0, t1, t2, t3] (each [M, 2304]) -> [M, 9216]
inline void k_df2_concat4_sg16(const float* taps4, float* out9216, int M, int M_stride_taps) {
    const int total = M * 4 * df2_cfg::kH;
    Q().parallel_for(NR((total + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int idx = (int)it.get_global_id(0);
        if (idx >= total) return;
        const int m = idx / (4 * df2_cfg::kH);
        const int rem = idx % (4 * df2_cfg::kH);
        const int t = rem / df2_cfg::kH;
        const int d = rem % df2_cfg::kH;
        out9216[idx] = taps4[(size_t)t * M_stride_taps + (size_t)m * df2_cfg::kH + d];
    });
}

// Head-RMSNorm + NeoX half-split RoPE (theta=10000) + store K/V into FP16 ring [slot, pos%2112, 1024]
inline void k_df2_ctx_kv_store_sg16(
    const float* k_in, const float* v_in, const float* k_norm_w,
    half* ring_k, half* ring_v, const int* slot_ids, const int* pos_buf, int M) {
    Q().parallel_for(NR2(df2_cfg::kKVH, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), kh = (int)it.get_group(1), d = (int)it.get_local_id(1);
        const int slot = slot_ids[m], pos = pos_buf[m];
        const int cell = ((pos % df2_cfg::kRingCap) + df2_cfg::kRingCap) % df2_cfg::kRingCap;
        const float* kp = k_in + ((size_t)m * df2_cfg::kKVH + kh) * df2_cfg::kHD;
        const float* vp = v_in + ((size_t)m * df2_cfg::kKVH + kh) * df2_cfg::kHD;

        const float kv = kp[d];
        const float inv = sycl::rsqrt(wg_sum16(it, kv * kv, sh) / (float)df2_cfg::kHD + df2_cfg::kEps);
        const float kn = kv * inv * k_norm_w[d];

        // Store normalized k into local memory to apply NeoX half-split RoPE (d vs d+64)
        float* sk = sh;  // wait: sh is 16 floats! Use sub-group / 2-step or local_accessor.
        (void)sk;
        // Compute paired partner directly without extra SLM:
        const int half_d = d & 63;
        const int other = (d < 64) ? (d + 64) : (d - 64);
        const float kv_o = kp[other];
        const float kn_o = kv_o * inv * k_norm_w[other];
        const float ang = (float)pos * sycl::pow(df2_cfg::kTheta, -2.0f * (float)half_d / (float)df2_cfg::kHD);
        const float cs = sycl::cos(ang), sn = sycl::sin(ang);
        const float kr = (d < 64) ? (kn * cs - kn_o * sn) : (kn * cs + kn_o * sn);

        const size_t dst = (((size_t)slot * df2_cfg::kRingCap + cell) * df2_cfg::kKVH + kh) * df2_cfg::kHD + d;
        ring_k[dst] = (half)kr;
        ring_v[dst] = (half)vp[d];
    });
}

// Block-8 initialization: row 0 = embed_norm(anchor_tok), rows 1..7 = embed_norm(200056)
inline void k_df2_init_block_sg16(
    const InkTensorInfo& eti, const float* norm_w, float* out8, int anchor_tok, int V_padded) {
    const uint8_t* ew = eti.dev_ptr;
    const bool is_bf16 = (eti.fmt == FMT_BF16);
    Q().parallel_for(NR(df2_cfg::kBlock, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int r = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const int id = (r == 0) ? anchor_tok : df2_cfg::kMaskId;
        float ss = 0.0f;
        if (is_bf16) {
            const uint16_t* row = reinterpret_cast<const uint16_t*>(ew) + (size_t)id * df2_cfg::kH;
            for (int d = tid; d < df2_cfg::kH; d += 256) { float v = bf16_to_f32(row[d]); ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)df2_cfg::kH + df2_cfg::kEps);
            for (int d = tid; d < df2_cfg::kH; d += 256) out8[(size_t)r * df2_cfg::kH + d] = bf16_to_f32(row[d]) * inv * norm_w[d];
        } else {
            const float sc = reinterpret_cast<const float*>(ew)[id];
            const int8_t* row = reinterpret_cast<const int8_t*>(ew + (size_t)V_padded * 4) + (size_t)id * df2_cfg::kH;
            for (int d = tid; d < df2_cfg::kH; d += 256) { float v = (float)row[d] * sc; ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)df2_cfg::kH + df2_cfg::kEps);
            for (int d = tid; d < df2_cfg::kH; d += 256) out8[(size_t)r * df2_cfg::kH + d] = (float)row[d] * sc * inv * norm_w[d];
        }
    });
}

// 2-tap group-16 dynamic causal conv pre:
// xp[r, c] = (b0[c] + dyn[r, g]) * xn[r, c] + (t > 0 ? (b1[c] + dyn[r, 144 + g]) * xn[r - 1, c] : 0)
inline void k_df2_conv_pre_sg16(
    const float* xn, const float* base_k, const float* dyn, float* xp, int B_rows) {
    Q().parallel_for(NR2((df2_cfg::kH + 255) / 256, (size_t)B_rows, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int r = (int)it.get_group(0);
        const int c = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (c >= df2_cfg::kH) return;
        const int t = r & 7;
        const int g = c >> 4;  // c / 16
        const float* dr = dyn + (size_t)r * df2_cfg::kDyn;
        const float w0 = base_k[0 * df2_cfg::kH + c] + dr[g];
        float v = w0 * xn[(size_t)r * df2_cfg::kH + c];
        if (t > 0) {
            const float w1 = base_k[1 * df2_cfg::kH + c] + dr[df2_cfg::kNG + g];
            v += w1 * xn[(size_t)(r - 1) * df2_cfg::kH + c];
        }
        xp[(size_t)r * df2_cfg::kH + c] = v;
    });
}

// 2-tap group-16 dynamic causal conv post-add:
// x[r, c] += (b2[c] + dyn[r, 288 + g]) * z[r, c] + (t > 0 ? (b3[c] + dyn[r, 432 + g]) * z[r - 1, c] : 0)
inline void k_df2_conv_post_add_sg16(
    float* x, const float* z, const float* base_k, const float* dyn, int B_rows) {
    Q().parallel_for(NR2((df2_cfg::kH + 255) / 256, (size_t)B_rows, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int r = (int)it.get_group(0);
        const int c = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (c >= df2_cfg::kH) return;
        const int t = r & 7;
        const int g = c >> 4;
        const float* dr = dyn + (size_t)r * df2_cfg::kDyn;
        const float w2 = base_k[2 * df2_cfg::kH + c] + dr[2 * df2_cfg::kNG + g];
        float v = w2 * z[(size_t)r * df2_cfg::kH + c];
        if (t > 0) {
            const float w3 = base_k[3 * df2_cfg::kH + c] + dr[3 * df2_cfg::kNG + g];
            v += w3 * z[(size_t)(r - 1) * df2_cfg::kH + c];
        }
        x[(size_t)r * df2_cfg::kH + c] += v;
    });
}

// In-place per-head RMSNorm + NeoX half-split RoPE (theta=10000) for block Q [8, 32, 128] or K [8, 8, 128]
inline void k_df2_qk_norm_rope_sg16(float* buf, const float* norm_w, int n_heads, int pos0) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        sycl::local_accessor<float, 1> xn(128, h);
        h.parallel_for(NR2((size_t)n_heads, df2_cfg::kBlock, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int r = (int)it.get_group(0), hd = (int)it.get_group(1), d = (int)it.get_local_id(1);
            float* p = buf + ((size_t)r * n_heads + hd) * df2_cfg::kHD;
            const float v = p[d];
            const float inv = sycl::rsqrt(wg_sum16(it, v * v, sh) / (float)df2_cfg::kHD + df2_cfg::kEps);
            xn[d] = v * inv * norm_w[d];
            sycl::group_barrier(it.get_group());
            const int half_d = d & 63;
            const int other = (d < 64) ? (d + 64) : (d - 64);
            const float ang = (float)(pos0 + r) * sycl::pow(df2_cfg::kTheta, -2.0f * (float)half_d / (float)df2_cfg::kHD);
            const float cs = sycl::cos(ang), sn = sycl::sin(ang);
            p[d] = (d < 64) ? (xn[d] * cs - xn[other] * sn) : (xn[d] * cs + xn[other] * sn);
        });
    });
}

// Non-causal intra-block + SWA-2048 ring-context attention (32Q : 8KV GQA, scale 1/sqrt(128))
// Row t in 0..7 attends to:
//   - context cells `cell` in [max(0, pos0 + t - 2047), min(ctx_hi, pos0)) from ring_k/ring_v
//   - all 8 block cells `b` in 0..7 (positions pos0..pos0+7) from blk_k/blk_v!
inline void k_df2_block_attn_sg16(
    const float* q_buf, const float* blk_k, const float* blk_v,
    const half* ring_k, const half* ring_v, float* out_buf,
    int slot, int pos0, int ctx_lo, int ctx_hi) {
    const float scale = 1.0f / std::sqrt((float)df2_cfg::kHD);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_m(8, h);
        sycl::local_accessor<float, 1> sh_l(8, h);
        sycl::local_accessor<float, 1> sh_o(8 * df2_cfg::kHD, h);
        h.parallel_for(NR2(df2_cfg::kQH, df2_cfg::kBlock, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int r = (int)it.get_group(0), head = (int)it.get_group(1);
            const int tid = (int)it.get_local_id(1);
            const int sg = tid >> 4, lane = tid & 15;
            const int kh = head >> 2;  // 32 / 8 = 4 Q heads per KV head

            float qv[8], o_acc[8];
            const float* qp = q_buf + ((size_t)r * df2_cfg::kQH + head) * df2_cfg::kHD + lane * 8;
#pragma unroll
            for (int u = 0; u < 8; ++u) { qv[u] = qp[u] * scale; o_acc[u] = 0.0f; }

            const int lo = sycl::max(ctx_lo, pos0 + r - (df2_cfg::kWindow - 1));
            const int c_end = sycl::min(ctx_hi, pos0);
            float m_max = -1e30f;
            float l_sum = 0.0f;
            const size_t slot_base = (size_t)slot * df2_cfg::kRingCap;

            // 1. Context cells in ring [lo, c_end)
            for (int k = lo + sg; k < c_end; k += 8) {
                const int cell = k % df2_cfg::kRingCap;
                const size_t kv_off = ((slot_base + cell) * df2_cfg::kKVH + kh) * df2_cfg::kHD + lane * 8;
                float dt = 0.0f;
#pragma unroll
                for (int u = 0; u < 8; ++u) dt += qv[u] * (float)ring_k[kv_off + u];
                const float sc = sg_sum(it, dt);
                const float mn = sycl::fmax(m_max, sc);
                const float corr = sycl::exp(m_max - mn);
                const float p = sycl::exp(sc - mn);
                l_sum = l_sum * corr + p;
                m_max = mn;
#pragma unroll
                for (int u = 0; u < 8; ++u) o_acc[u] = o_acc[u] * corr + p * (float)ring_v[kv_off + u];
            }

            // 2. Non-causal intra-block cells 0..7 (each of 8 SGs takes 1 block cell sg!)
            {
                const int b = sg;
                const size_t boff = ((size_t)b * df2_cfg::kKVH + kh) * df2_cfg::kHD + lane * 8;
                float dt = 0.0f;
#pragma unroll
                for (int u = 0; u < 8; ++u) dt += qv[u] * blk_k[boff + u];
                const float sc = sg_sum(it, dt);
                const float mn = sycl::fmax(m_max, sc);
                const float corr = sycl::exp(m_max - mn);
                const float p = sycl::exp(sc - mn);
                l_sum = l_sum * corr + p;
                m_max = mn;
#pragma unroll
                for (int u = 0; u < 8; ++u) o_acc[u] = o_acc[u] * corr + p * blk_v[boff + u];
            }

            if (lane == 0) { sh_m[sg] = m_max; sh_l[sg] = l_sum; }
#pragma unroll
            for (int u = 0; u < 8; ++u) sh_o[sg * df2_cfg::kHD + lane * 8 + u] = o_acc[u];
            sycl::group_barrier(it.get_group());

            if (sg == 0) {
                float gm = -1e30f;
                for (int s = 0; s < 8; ++s) gm = sycl::fmax(gm, sh_m[s]);
                float gl = 0.0f, go[8] = {};
                for (int s = 0; s < 8; ++s) {
                    const float w = sycl::exp(sh_m[s] - gm);
                    gl += sh_l[s] * w;
#pragma unroll
                    for (int u = 0; u < 8; ++u) go[u] += sh_o[s * df2_cfg::kHD + lane * 8 + u] * w;
                }
                const float il = 1.0f / gl;
                float* op = out_buf + ((size_t)r * df2_cfg::kQH + head) * df2_cfg::kHD + lane * 8;
#pragma unroll
                for (int u = 0; u < 8; ++u) op[u] = go[u] * il;
            }
        });
    });
}

// Standard SwiGLU: silu(gate) * up (NOT SITU!)
inline void k_df2_swiglu_sg16(const float* g, const float* u, float* out, int total) {
    Q().parallel_for(NR((total + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = (int)it.get_global_id(0);
        if (i < total) {
            const float gv = g[i];
            out[i] = (gv / (1.0f + sycl::exp(-gv))) * u[i];
        }
    });
}

// Extract exact top-16 raw logits + full-vocab softcapped tail (m_t, l_t) for 7 draft rows!
inline void k_df2_top16_tail_sg16(
    const float* raw_logits, int V, int n_rows,
    float* top_val, int32_t* top_idx, float* tail_m, float* tail_l) {
    const float cap = df2_cfg::kCap, icap = 1.0f / df2_cfg::kCap;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> sh_top(df2_cfg::kTopK, h);
        sycl::local_accessor<float, 1> sh_red(16, h);
        h.parallel_for(NR(n_rows, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int r = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            auto grp = it.get_group();
            const float* row = raw_logits + (size_t)r * V;

            // 1. Per-thread top-4 exact tracking (16 rounds guaranteed exact across 256 threads with refill guard)
            constexpr int TK = 4;
            float lv[TK] = {-1e38f, -1e38f, -1e38f, -1e38f};
            int32_t li[TK] = {V, V, V, V};
            for (int i = tid; i < V; i += 256) {
                const float v = row[i];
                if (v > lv[TK - 1] || (v == lv[TK - 1] && i < li[TK - 1])) {
                    int j = TK - 1;
                    while (j > 0 && (v > lv[j - 1] || (v == lv[j - 1] && i < li[j - 1]))) {
                        lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j;
                    }
                    lv[j] = v; li[j] = i;
                }
            }
            int cur = 0;
            for (int k = 0; k < df2_cfg::kTopK; ++k) {
                float cv = (cur < TK) ? lv[cur] : -1e38f;
                int32_t ci = (cur < TK) ? li[cur] : V;
                const float mx = sycl::reduce_over_group(grp, cv, sycl::maximum<float>());
                const int32_t cand = (cv == mx) ? ci : V;
                const int32_t bi = sycl::reduce_over_group(grp, cand, sycl::minimum<int32_t>());
                if (tid == 0) {
                    top_val[(size_t)r * df2_cfg::kTopK + k] = mx;
                    top_idx[(size_t)r * df2_cfg::kTopK + k] = bi;
                    sh_top[k] = bi;
                }
                if (ci == bi && cur < TK) {
                    ++cur;
                    if (cur == TK && k + 1 < df2_cfg::kTopK) {
                        float pv = lv[TK - 1]; int32_t pi = li[TK - 1];
                        for (int q = 0; q < TK; ++q) { lv[q] = -1e38f; li[q] = V; }
                        for (int i = tid; i < V; i += 256) {
                            const float v = row[i];
                            if (v > pv || (v == pv && i <= pi)) continue;
                            if (v > lv[TK - 1] || (v == lv[TK - 1] && i < li[TK - 1])) {
                                int j = TK - 1;
                                while (j > 0 && (v > lv[j - 1] || (v == lv[j - 1] && i < li[j - 1]))) {
                                    lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j;
                                }
                                lv[j] = v; li[j] = i;
                            }
                        }
                        cur = 0;
                    }
                }
                sycl::group_barrier(grp);
            }

            // 2. Compute softcapped tail max m_t and sum l_t over all j not in sh_top[0..15]
            float lm = -1e30f;
            for (int i = tid; i < V; i += 256) {
                bool in_top = false;
                for (int k = 0; k < df2_cfg::kTopK; ++k) if (sh_top[k] == i) { in_top = true; break; }
                if (!in_top) {
                    const float sv = cap * sycl::tanh(sycl::clamp(row[i] * icap, -15.0f, 15.0f));
                    lm = sycl::fmax(lm, sv);
                }
            }
            const float gm = sycl::reduce_over_group(grp, lm, sycl::maximum<float>());
            float ls = 0.0f;
            for (int i = tid; i < V; i += 256) {
                bool in_top = false;
                for (int k = 0; k < df2_cfg::kTopK; ++k) if (sh_top[k] == i) { in_top = true; break; }
                if (!in_top) {
                    const float sv = cap * sycl::tanh(sycl::clamp(row[i] * icap, -15.0f, 15.0f));
                    ls += sycl::exp(sv - gm);
                }
            }
            const float gl = wg_sum16(it, ls, sh_red);
            if (tid == 0) {
                tail_m[r] = gm;
                tail_l[r] = gl;
            }
        });
    });
}

// Sequential 7-step rank-256 bilinear selector on GPU (1 WG of 256 threads: 1 thread per rank dim k=0..255!)
inline void k_df2_select7_sg16(
    const float* top_val, const int32_t* top_idx, const float* tail_m, const float* tail_l,
    const float* g_proj, const uint16_t* pred_cb_bf16, const uint16_t* succ_cb_bf16,
    int32_t anchor_tok, int32_t* out_draft, float* out_conf) {
    const float cap = df2_cfg::kCap, icap = 1.0f / df2_cfg::kCap;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_red(16, h);
        sycl::local_accessor<float, 1> sh_sc(df2_cfg::kTopK, h);
        sycl::local_accessor<int32_t, 1> sh_prev(1, h);
        h.parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int k = (int)it.get_local_id(0);
            auto grp = it.get_group();
            if (k == 0) sh_prev[0] = anchor_tok;
            sycl::group_barrier(grp);

            for (int d = 0; d < 7; ++d) {
                const int32_t prev = sh_prev[0];
                const float pg = bf16_to_f32(pred_cb_bf16[(size_t)prev * df2_cfg::kRank + k])
                               * g_proj[(size_t)d * df2_cfg::kRank + k];
                for (int c = 0; c < df2_cfg::kTopK; ++c) {
                    const int32_t cid = top_idx[(size_t)d * df2_cfg::kTopK + c];
                    const float prod = pg * bf16_to_f32(succ_cb_bf16[(size_t)cid * df2_cfg::kRank + k]);
                    const float dot = wg_sum16(it, prod, sh_red);
                    if (k == 0) sh_sc[c] = top_val[(size_t)d * df2_cfg::kTopK + c] + dot;
                    sycl::group_barrier(grp);
                }
                if (k == 0) {
                    int best = 0;
                    float best_s = sh_sc[0];
                    for (int c = 1; c < df2_cfg::kTopK; ++c) {
                        if (sh_sc[c] > best_s) { best_s = sh_sc[c]; best = c; }
                    }
                    const int32_t pick = top_idx[(size_t)d * df2_cfg::kTopK + best];
                    out_draft[d] = pick;
                    sh_prev[0] = pick;

                    // Calibrated softmax confidence with exact full-vocab tail!
                    float sc_cap[df2_cfg::kTopK];
                    float m_ref = tail_m[d];
                    for (int c = 0; c < df2_cfg::kTopK; ++c) {
                        sc_cap[c] = cap * sycl::tanh(sycl::clamp(sh_sc[c] * icap, -15.0f, 15.0f));
                        if (sc_cap[c] > m_ref) m_ref = sc_cap[c];
                    }
                    float sum16 = 0.0f;
                    for (int c = 0; c < df2_cfg::kTopK; ++c) sum16 += sycl::exp(sc_cap[c] - m_ref);
                    const float denom = sum16 + tail_l[d] * sycl::exp(tail_m[d] - m_ref);
                    const float p_best = (denom > 0.0f) ? (sycl::exp(sc_cap[best] - m_ref) / denom) : 0.0f;
                    out_conf[d] = p_best;
                }
                sycl::group_barrier(grp);
            }
        });
    });
}

// ============================================================================
// InkSyDFlash2 State & Runner
// ============================================================================

struct InkSyDFlash2 {
    struct Layer {
        float* in_norm = nullptr;
        float* post_norm = nullptr;
        uint8_t* attn_conv_proj = nullptr;  // Q8_ROW [576, 2304]
        float* attn_conv_base = nullptr;    // F32 [4, 2304]
        uint8_t* mlp_conv_proj = nullptr;   // Q8_ROW [576, 2304]
        float* mlp_conv_base = nullptr;     // F32 [4, 2304]
        uint8_t* wq = nullptr;              // Q8_ROW [4096, 2304]
        uint8_t* wk = nullptr;              // Q8_ROW [1024, 2304]
        uint8_t* wv = nullptr;              // Q8_ROW [1024, 2304]
        uint8_t* wo = nullptr;              // Q8_ROW [2304, 4096]
        float* q_norm = nullptr;            // F32 [128]
        float* k_norm = nullptr;            // F32 [128]
        uint8_t* w_gate = nullptr;          // Q8_ROW [6912, 2304]
        uint8_t* w_up = nullptr;            // Q8_ROW [6912, 2304]
        uint8_t* w_dn = nullptr;            // Q8_ROW [2304, 6912]
    };

    bool loaded = false;
    std::string path;
    int max_slots = 1;
    int max_m_cap = 8;  // ceiling.py calibrated verify window cap (8 Hybrid, 16 all-VRAM)

    // Weights
    uint8_t* fc_q8 = nullptr;           // Q8_ROW [2304, 9216]
    float* hidden_norm = nullptr;       // F32 [2304]
    float* final_norm = nullptr;        // F32 [2304]
    float* sel_h_proj = nullptr;        // F32 [256, 2304]
    uint16_t* pred_cb = nullptr;        // BF16 [201024, 256]
    uint16_t* succ_cb = nullptr;        // BF16 [201024, 256]
    Layer layers[df2_cfg::kL]{};
    std::vector<void*> owned_ptrs;

    // Context KV ring per layer [max_slots, 2112, 1024] FP16 + per-slot [ctx_lo, ctx_hi) watermarks
    half* ring_k[df2_cfg::kL]{};
    half* ring_v[df2_cfg::kL]{};
    std::vector<int> ctx_lo;
    std::vector<int> ctx_hi;

    // Scratch buffers for Block-8 draft + Ingest
    float* d_cat9216 = nullptr;   // [max_batch_ing, 9216]
    int max_ing_m = 0;
    float* d_ing_h = nullptr;     // [max_batch_ing, 2304]
    float* d_ing_k = nullptr;     // [max_batch_ing, 1024]
    float* d_ing_v = nullptr;     // [max_batch_ing, 1024]

    float* b_x = nullptr;         // [8, 2304]
    float* b_xn = nullptr;        // [8, 2304]
    float* b_xp = nullptr;        // [8, 2304]
    float* b_dyn = nullptr;       // [8, 576]
    float* b_q = nullptr;         // [8, 4096]
    float* b_k = nullptr;         // [8, 1024]
    float* b_v = nullptr;         // [8, 1024]
    float* b_ao = nullptr;        // [8, 4096]
    float* b_z = nullptr;         // [8, 2304]
    float* b_mg = nullptr;        // [8, 6912]
    float* b_mu = nullptr;        // [8, 6912]
    float* b_mh = nullptr;        // [8, 6912]
    float* b_gproj = nullptr;     // [7, 256]
    float* b_raw_lg = nullptr;    // [7, V_unpadded]
    float* b_top_v = nullptr;     // [7, 16]
    int32_t* b_top_i = nullptr;   // [7, 16]
    float* b_tail_m = nullptr;    // [7]
    float* b_tail_l = nullptr;    // [7]
    int32_t* b_draft = nullptr;   // [7]
    float* b_conf = nullptr;      // [7]
    int8_t* b_xq = nullptr;       // [max(8, 256) * 9216]
    float* b_xsc = nullptr;       // [max(8, 256)]

    // Per-slot token history ring for PLD (Prompt Lookup Decoding) when STRATA_INK_INTELARC=1
    std::vector<std::vector<int32_t>> pld_hist;

    ~InkSyDFlash2() { release(); }

    template <typename T>
    T* track_dalloc(size_t n) {
        T* p = dalloc<T>(n);
        owned_ptrs.push_back(p);
        return p;
    }

    void release() {
        for (void* p : owned_ptrs) dfree(p);
        owned_ptrs.clear();
        for (int l = 0; l < df2_cfg::kL; ++l) {
            ring_k[l] = ring_v[l] = nullptr;
            layers[l] = Layer{};
        }
        fc_q8 = nullptr; hidden_norm = final_norm = sel_h_proj = nullptr;
        pred_cb = succ_cb = nullptr;
        d_cat9216 = d_ing_h = d_ing_k = d_ing_v = nullptr;
        b_x = b_xn = b_xp = b_dyn = b_q = b_k = b_v = b_ao = b_z = b_mg = b_mu = b_mh = nullptr;
        b_gproj = b_raw_lg = b_top_v = b_tail_m = b_tail_l = b_conf = b_xsc = nullptr;
        b_top_i = b_draft = nullptr; b_xq = nullptr;
        max_ing_m = 0;
        ctx_lo.clear();
        ctx_hi.clear();
        pld_hist.clear();
        loaded = false;
    }

    void reset_slot(int s, int pos = 0) {
        if (s >= 0 && s < (int)ctx_hi.size()) {
            const int p = std::max(0, pos);
            ctx_lo[s] = p;
            ctx_hi[s] = p;
            pld_hist[s].clear();
        }
    }

    void rollback_slot(int s, int keep_len) {
        if (s >= 0 && s < (int)ctx_hi.size()) {
            const int kl = std::max(0, keep_len);
            if (ctx_hi[s] > kl) ctx_hi[s] = kl;
            if (ctx_lo[s] > kl) ctx_lo[s] = kl;
            if ((int)pld_hist[s].size() > kl) pld_hist[s].resize((size_t)kl);
        }
    }

    void record_pld_tokens(int s, const int32_t* ids, int n) {
        if (s < 0 || s >= (int)pld_hist.size()) return;
        auto& h = pld_hist[s];
        for (int i = 0; i < n; ++i) {
            h.push_back(ids[i]);
            if (h.size() > 8192) h.erase(h.begin(), h.begin() + 2048);
        }
    }

    // Fast n-gram PLD suffix lookup (3-gram match in recent slot history -> 0-cost bonus draft on edits/code!)
    int try_pld_draft(int s, int32_t anchor_tok, int max_draft, int32_t* out_ids) const {
        if (s < 0 || s >= (int)pld_hist.size() || max_draft <= 0) return 0;
        const auto& h = pld_hist[s];
        const size_t n = h.size();
        if (n < 8) return 0;
        // Require a 3-token match ending at anchor_tok (h[n-2], h[n-1], anchor_tok)
        const int32_t a = h[n - 2], b = h[n - 1], c = anchor_tok;
        const size_t search_end = (n > 3) ? (n - 3) : 0;
        for (size_t i = search_end; i >= 2; --i) {
            if (h[i - 2] == a && h[i - 1] == b && h[i] == c) {
                int take = std::min(max_draft, (int)(n - (i + 1)));
                if (take >= 3) {
                    for (int k = 0; k < take; ++k) out_ids[k] = h[i + 1 + k];
                    return take;
                }
            }
        }
        return 0;
    }

    static std::vector<uint8_t> quantize_bf16_to_q8row(const uint16_t* bf16, int N, int K) {
        std::vector<uint8_t> out((size_t)N * 4 + (size_t)N * K);
        float* scales = reinterpret_cast<float*>(out.data());
        int8_t* qw = reinterpret_cast<int8_t*>(out.data() + (size_t)N * 4);
        for (int r = 0; r < N; ++r) {
            const uint16_t* src = bf16 + (size_t)r * K;
            float amax = 0.0f;
            for (int k = 0; k < K; ++k) {
                uint32_t u = (uint32_t)src[k] << 16;
                float v; std::memcpy(&v, &u, 4);
                float av = std::fabs(v);
                if (av > amax) amax = av;
            }
            const float sc = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
            const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            scales[r] = sc;
            int8_t* dst = qw + (size_t)r * K;
            for (int k = 0; k < K; ++k) {
                uint32_t u = (uint32_t)src[k] << 16;
                float v; std::memcpy(&v, &u, 4);
                int q = (int)std::lrint(v * inv);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                dst[k] = (int8_t)q;
            }
        }
        return out;
    }

    bool load(const std::string& st_path, int slots, int cap_m, std::string& err) {
        release();
        double t0 = now_s();
        strata::image::StFile sf;
        err = strata::image::st_read_header(st_path, sf);
        if (!err.empty()) {
            err = "dflash2 open failed: " + err;
            return false;
        }
        std::unique_ptr<unsigned char[]> blob;
        size_t blob_sz = 0;
        err = strata::image::st_read_all_parallel(st_path, sf, blob, blob_sz, 4);
        if (!err.empty()) {
            err = "dflash2 read failed: " + err;
            return false;
        }
        path = st_path;
        max_slots = std::max(1, std::min(64, slots));
        max_m_cap = std::max(2, std::min(16, cap_m));
        ctx_lo.assign((size_t)max_slots, 0);
        ctx_hi.assign((size_t)max_slots, 0);
        pld_hist.resize((size_t)max_slots);

        auto find_t = [&](const std::string& name) -> const strata::image::StTensor* {
            auto it = sf.tensors.find(name);
            if (it == sf.tensors.end()) { err = "dflash2 missing tensor: " + name; return nullptr; }
            if (it->second.end > blob_sz || it->second.end < it->second.begin) {
                err = "dflash2 out-of-bounds tensor: " + name;
                return nullptr;
            }
            return &it->second;
        };

        auto decode_to_f32 = [](const strata::image::StTensor& t, const unsigned char* raw, size_t n, std::vector<float>& f) -> bool {
            f.resize(n);
            const size_t avail = (size_t)(t.end - t.begin);
            if (t.dtype == "BF16" && avail >= n * 2) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
                for (size_t i = 0; i < n; ++i) {
                    uint32_t u = (uint32_t)p[i] << 16;
                    std::memcpy(&f[i], &u, 4);
                }
                return true;
            }
            if (t.dtype == "F32" && avail >= n * 4) {
                std::memcpy(f.data(), raw, n * 4);
                return true;
            }
            if (t.dtype == "F16" && avail >= n * 2) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
                for (size_t i = 0; i < n; ++i) f[i] = h2f_bits(p[i]);
                return true;
            }
            return false;
        };

        auto load_f32 = [&](const std::string& name, size_t expected_elems) -> float* {
            const auto* t = find_t(name);
            if (!t) return nullptr;
            std::vector<float> f;
            if (!decode_to_f32(*t, blob.get() + t->begin, expected_elems, f)) {
                err = "dflash2 unsupported dtype/size for " + name;
                return nullptr;
            }
            float* d = track_dalloc<float>(expected_elems);
            h2d_sync(d, f.data(), expected_elems * sizeof(float));
            return d;
        };

        auto load_bf16_raw = [&](const std::string& name, size_t expected_elems) -> uint16_t* {
            const auto* t = find_t(name);
            if (!t) return nullptr;
            const unsigned char* raw = blob.get() + t->begin;
            std::vector<uint16_t> bf(expected_elems);
            if (t->dtype == "BF16" && (size_t)(t->end - t->begin) >= expected_elems * 2) {
                std::memcpy(bf.data(), raw, expected_elems * 2);
            } else {
                std::vector<float> f;
                if (!decode_to_f32(*t, raw, expected_elems, f)) { err = "dflash2 bad bf16 tensor " + name; return nullptr; }
                for (size_t i = 0; i < expected_elems; ++i) {
                    uint32_t u; std::memcpy(&u, &f[i], 4);
                    bf[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
                }
            }
            uint16_t* d = track_dalloc<uint16_t>(expected_elems);
            h2d_sync(d, bf.data(), expected_elems * sizeof(uint16_t));
            return d;
        };

        auto load_q8 = [&](const std::string& name, int N, int K) -> uint8_t* {
            const size_t ne = (size_t)N * K;
            std::vector<uint16_t> bf;
            const uint16_t* bf_ptr = nullptr;
            // the pre-quantized file ("strata-dflash2-q1" mix): NAME.q / NAME.s (/ NAME.o) instead of NAME
            if (sf.tensors.find(name + ".q") != sf.tensors.end()) {
                const auto* tq = find_t(name + ".q");
                const auto* ts = tq ? find_t(name + ".s") : nullptr;
                if (!tq || !ts) return nullptr;
                const unsigned char* rq = blob.get() + tq->begin;
                const unsigned char* rs = blob.get() + ts->begin;
                if (tq->dtype == "I8" && ts->dtype == "F32" && (size_t)(tq->end - tq->begin) == ne && (size_t)(ts->end - ts->begin) == (size_t)N * 4) {
                    std::vector<uint8_t> q8((size_t)N * 4 + ne);
                    std::memcpy(q8.data(), rs, (size_t)N * 4);
                    std::memcpy(q8.data() + (size_t)N * 4, rq, ne);
                    uint8_t* d = track_dalloc<uint8_t>(q8.size());
                    h2d_sync(d, q8.data(), q8.size());
                    return d;
                }
                const auto* to = find_t(name + ".o");
                if (!to) return nullptr;
                if (!(tq->dtype == "U8" && ts->dtype == "U8" && to->dtype == "F32" && (size_t)(tq->end - tq->begin) == ne / 2 &&
                      (size_t)(ts->end - ts->begin) == ne / 16 && (size_t)(to->end - to->begin) == 4 && K % 16 == 0)) {
                    err = "dflash2 unsupported pre-quantized tensor " + name;
                    return nullptr;
                }
                float outer; std::memcpy(&outer, blob.get() + to->begin, 4);
                bf.resize(ne);
                ink2::df2mix::sf4_to_bf16(rq, rs, outer, N, K, bf.data());
                bf_ptr = bf.data();
                std::vector<uint8_t> q8 = quantize_bf16_to_q8row(bf_ptr, N, K);
                uint8_t* d = track_dalloc<uint8_t>(q8.size());
                h2d_sync(d, q8.data(), q8.size());
                return d;
            }
            const auto* t = find_t(name);
            if (!t) return nullptr;
            const unsigned char* raw = blob.get() + t->begin;
            if (t->dtype == "BF16" && (size_t)(t->end - t->begin) >= ne * 2) {
                bf_ptr = reinterpret_cast<const uint16_t*>(raw);
            } else {
                std::vector<float> f;
                if (!decode_to_f32(*t, raw, ne, f)) { err = "dflash2 bad q8 tensor " + name; return nullptr; }
                bf.resize(ne);
                for (size_t i = 0; i < ne; ++i) {
                    uint32_t u; std::memcpy(&u, &f[i], 4);
                    bf[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
                }
                bf_ptr = bf.data();
            }
            std::vector<uint8_t> q8 = quantize_bf16_to_q8row(bf_ptr, N, K);
            uint8_t* d = track_dalloc<uint8_t>(q8.size());
            h2d_sync(d, q8.data(), q8.size());
            return d;
        };

        if (!(fc_q8 = load_q8("fc.weight", df2_cfg::kH, 4 * df2_cfg::kH))) return false;
        if (!(hidden_norm = load_f32("hidden_norm.weight", df2_cfg::kH))) return false;
        if (!(final_norm = load_f32("norm.weight", df2_cfg::kH))) return false;
        if (!(sel_h_proj = load_f32("candidate_selector.hidden_projection.weight", (size_t)df2_cfg::kRank * df2_cfg::kH))) return false;
        if (!(pred_cb = load_bf16_raw("candidate_selector.predecessor_codebook", (size_t)df2_cfg::kV * df2_cfg::kRank))) return false;
        if (!(succ_cb = load_bf16_raw("candidate_selector.successor_codebook", (size_t)df2_cfg::kV * df2_cfg::kRank))) return false;

        for (int l = 0; l < df2_cfg::kL; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            auto& ly = layers[l];
            if (!(ly.in_norm        = load_f32(p + "input_layernorm.weight", df2_cfg::kH))) return false;
            if (!(ly.post_norm      = load_f32(p + "post_attention_layernorm.weight", df2_cfg::kH))) return false;
            if (!(ly.attn_conv_proj = load_q8 (p + "attention_conv.kernel_projection.weight", df2_cfg::kDyn, df2_cfg::kH))) return false;
            if (!(ly.attn_conv_base = load_f32(p + "attention_conv.base_kernel", 4ul * df2_cfg::kH))) return false;
            if (!(ly.mlp_conv_proj  = load_q8 (p + "mlp_conv.kernel_projection.weight", df2_cfg::kDyn, df2_cfg::kH))) return false;
            if (!(ly.mlp_conv_base  = load_f32(p + "mlp_conv.base_kernel", 4ul * df2_cfg::kH))) return false;
            if (!(ly.wq             = load_q8 (p + "self_attn.q_proj.weight", df2_cfg::kQ, df2_cfg::kH))) return false;
            if (!(ly.wk             = load_q8 (p + "self_attn.k_proj.weight", df2_cfg::kKV, df2_cfg::kH))) return false;
            if (!(ly.wv             = load_q8 (p + "self_attn.v_proj.weight", df2_cfg::kKV, df2_cfg::kH))) return false;
            if (!(ly.wo             = load_q8 (p + "self_attn.o_proj.weight", df2_cfg::kH, df2_cfg::kQ))) return false;
            if (!(ly.q_norm         = load_f32(p + "self_attn.q_norm.weight", df2_cfg::kHD))) return false;
            if (!(ly.k_norm         = load_f32(p + "self_attn.k_norm.weight", df2_cfg::kHD))) return false;
            if (!(ly.w_gate         = load_q8 (p + "mlp.gate_proj.weight", df2_cfg::kFF, df2_cfg::kH))) return false;
            if (!(ly.w_up           = load_q8 (p + "mlp.up_proj.weight", df2_cfg::kFF, df2_cfg::kH))) return false;
            if (!(ly.w_dn           = load_q8 (p + "mlp.down_proj.weight", df2_cfg::kH, df2_cfg::kFF))) return false;

            const size_t ring_e = (size_t)max_slots * df2_cfg::kRingCap * df2_cfg::kKV;
            ring_k[l] = track_dalloc<half>(ring_e);
            ring_v[l] = track_dalloc<half>(ring_e);
        }

        // Scratch allocations for Block-8 draft + chunked ingest (256-token chunks keep ingest scratch ~12 MiB!)
        max_ing_m = 256;
        d_cat9216 = track_dalloc<float>((size_t)max_ing_m * 4 * df2_cfg::kH);
        d_ing_h   = track_dalloc<float>((size_t)max_ing_m * df2_cfg::kH);
        d_ing_k   = track_dalloc<float>((size_t)max_ing_m * df2_cfg::kKV);
        d_ing_v   = track_dalloc<float>((size_t)max_ing_m * df2_cfg::kKV);

        b_x      = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kH);
        b_xn     = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kH);
        b_xp     = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kH);
        b_dyn    = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kDyn);
        b_q      = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kQ);
        b_k      = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kKV);
        b_v      = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kKV);
        b_ao     = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kQ);
        b_z      = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kH);
        b_mg     = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kFF);
        b_mu     = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kFF);
        b_mh     = track_dalloc<float>(df2_cfg::kBlock * df2_cfg::kFF);
        b_gproj  = track_dalloc<float>(7ul * df2_cfg::kRank);
        b_raw_lg = track_dalloc<float>(7ul * df2_cfg::kV);
        b_top_v  = track_dalloc<float>(7ul * df2_cfg::kTopK);
        b_top_i  = track_dalloc<int32_t>(7ul * df2_cfg::kTopK);
        b_tail_m = track_dalloc<float>(7);
        b_tail_l = track_dalloc<float>(7);
        b_draft  = track_dalloc<int32_t>(7);
        b_conf   = track_dalloc<float>(7);
        b_xq     = track_dalloc<int8_t>((size_t)max_ing_m * 4 * df2_cfg::kH);
        b_xsc    = track_dalloc<float>(max_ing_m);

        wait();
        loaded = true;
        fprintf(stderr, "[ink_sycl_dflash2] loaded %s in %.2fs (5 layers Q8_ROW, slots=%d, ring=%d, max_M*=%d)\n",
                st_path.c_str(), now_s() - t0, max_slots, df2_cfg::kRingCap, max_m_cap);
        return true;
    }

    void q8_linear(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
        if (M >= 16) {
            k_gemm_q8row_sg16(W, N, K, X, Y, M, N);
        } else {
            k_quantize_x_q8_sg16(X, M, K, b_xq, b_xsc);
            k_linear_q8row_dp4a_sg16(W, N, K, b_xq, b_xsc, Y, M, N);
        }
    }

    // GPU-only ingest (100% SYCL command_graph capturable!)
    void ingest_gpu(const float* b_taps4, const int* b_slot_ids, const int* b_pos, int M) {
        if (!loaded || M <= 0) return;
        const int M_stride = M * df2_cfg::kH;
        // Process in chunks of <= max_ing_m (256) so prefill of 8K tokens uses only 12 MiB scratch!
        float* dst_cat = d_cat9216;
        for (int off = 0; off < M; off += max_ing_m) {
            const int chunk = std::min(max_ing_m, M - off);
            const int total = chunk * 4 * df2_cfg::kH;
            Q().parallel_for(NR((total + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
                const int idx = (int)it.get_global_id(0);
                if (idx >= total) return;
                const int m = idx / (4 * df2_cfg::kH);
                const int rem = idx % (4 * df2_cfg::kH);
                const int t = rem / df2_cfg::kH;
                const int d = rem % df2_cfg::kH;
                dst_cat[idx] = b_taps4[(size_t)t * M_stride + (size_t)(off + m) * df2_cfg::kH + d];
            });
            q8_linear(fc_q8, df2_cfg::kH, 4 * df2_cfg::kH, d_cat9216, d_ing_h, chunk);
            k_rms_norm_sg16(d_ing_h, hidden_norm, d_ing_h, df2_cfg::kH, chunk);
            for (int l = 0; l < df2_cfg::kL; ++l) {
                q8_linear(layers[l].wk, df2_cfg::kKV, df2_cfg::kH, d_ing_h, d_ing_k, chunk);
                q8_linear(layers[l].wv, df2_cfg::kKV, df2_cfg::kH, d_ing_h, d_ing_v, chunk);
                k_df2_ctx_kv_store_sg16(d_ing_k, d_ing_v, layers[l].k_norm,
                                        ring_k[l], ring_v[l],
                                        b_slot_ids + off, b_pos + off, chunk);
            }
        }
    }

    void update_ctx_hi(const int* h_slot_ids, const int* h_pos, int M) {
        if (!loaded || M <= 0) return;
        for (int m = 0; m < M; ++m) {
            const int s = h_slot_ids[m];
            const int p = h_pos[m] + 1;
            if (s >= 0 && s < (int)ctx_hi.size() && p > ctx_hi[s]) ctx_hi[s] = p;
        }
    }

    void ingest(const float* b_taps4, const int* b_slot_ids, const int* b_pos,
                const int* h_slot_ids, const int* h_pos, int M) {
        ingest_gpu(b_taps4, b_slot_ids, b_pos, M);
        update_ctx_hi(h_slot_ids, h_pos, M);
    }

    // Run Block-8 DFlash 2 forward + rank-256 selector -> up to 7 drafts + per-token calibrated confidences!
    // Uses `unembed_fn(x_7x2304, out_7xV)` callback so target unembed (MX8/BF16/Q8_ROW/S_K) is 100% exact.
    template <class UnembedFn>
    int draft_block8(
        int slot, int32_t anchor_tok, int anchor_pos, int max_draft,
        const InkTensorInfo& emb_ti, const float* emb_norm_w, int V_padded,
        UnembedFn&& unembed_fn, int32_t* out_drafts, float* out_confs,
        bool allow_pld = false) {
        if (!loaded || max_draft <= 0 || slot < 0 || slot >= max_slots) return 0;

        // Optional 0-cost PLD suffix-match shortcut when STRATA_INK_INTELARC=1
        if (allow_pld) {
            int pld_n = try_pld_draft(slot, anchor_tok, std::min(7, max_draft), out_drafts);
            if (pld_n >= 3) {
                for (int i = 0; i < pld_n; ++i) out_confs[i] = 0.90f;
                return pld_n;
            }
        }

        const int clo = ctx_lo[slot];
        const int chi = ctx_hi[slot];
        k_df2_init_block_sg16(emb_ti, emb_norm_w, b_x, anchor_tok, V_padded);

        for (int l = 0; l < df2_cfg::kL; ++l) {
            const auto& ly = layers[l];
            // 1. Attention sublayer with 2-tap group-16 dynamic causal conv
            k_rms_norm_sg16(b_x, ly.in_norm, b_xn, df2_cfg::kH, df2_cfg::kBlock);
            q8_linear(ly.attn_conv_proj, df2_cfg::kDyn, df2_cfg::kH, b_xn, b_dyn, df2_cfg::kBlock);
            k_df2_conv_pre_sg16(b_xn, ly.attn_conv_base, b_dyn, b_xp, df2_cfg::kBlock);
            q8_linear(ly.wq, df2_cfg::kQ, df2_cfg::kH, b_xp, b_q, df2_cfg::kBlock);
            q8_linear(ly.wk, df2_cfg::kKV, df2_cfg::kH, b_xp, b_k, df2_cfg::kBlock);
            q8_linear(ly.wv, df2_cfg::kKV, df2_cfg::kH, b_xp, b_v, df2_cfg::kBlock);
            k_df2_qk_norm_rope_sg16(b_q, ly.q_norm, df2_cfg::kQH, anchor_pos);
            k_df2_qk_norm_rope_sg16(b_k, ly.k_norm, df2_cfg::kKVH, anchor_pos);
            k_df2_block_attn_sg16(b_q, b_k, b_v, ring_k[l], ring_v[l], b_ao, slot, anchor_pos, clo, chi);
            q8_linear(ly.wo, df2_cfg::kH, df2_cfg::kQ, b_ao, b_z, df2_cfg::kBlock);
            k_df2_conv_post_add_sg16(b_x, b_z, ly.attn_conv_base, b_dyn, df2_cfg::kBlock);

            // 2. SwiGLU MLP sublayer with 2-tap group-16 dynamic causal conv
            k_rms_norm_sg16(b_x, ly.post_norm, b_xn, df2_cfg::kH, df2_cfg::kBlock);
            q8_linear(ly.mlp_conv_proj, df2_cfg::kDyn, df2_cfg::kH, b_xn, b_dyn, df2_cfg::kBlock);
            k_df2_conv_pre_sg16(b_xn, ly.mlp_conv_base, b_dyn, b_xp, df2_cfg::kBlock);
            q8_linear(ly.w_gate, df2_cfg::kFF, df2_cfg::kH, b_xp, b_mg, df2_cfg::kBlock);
            q8_linear(ly.w_up,   df2_cfg::kFF, df2_cfg::kH, b_xp, b_mu, df2_cfg::kBlock);
            k_df2_swiglu_sg16(b_mg, b_mu, b_mh, df2_cfg::kBlock * df2_cfg::kFF);
            q8_linear(ly.w_dn, df2_cfg::kH, df2_cfg::kFF, b_mh, b_z, df2_cfg::kBlock);
            k_df2_conv_post_add_sg16(b_x, b_z, ly.mlp_conv_base, b_dyn, df2_cfg::kBlock);
        }

        // 3. Final norm + rank-256 selector on rows 1..7
        k_rms_norm_sg16(b_x, final_norm, b_xn, df2_cfg::kH, df2_cfg::kBlock);
        const float* hd7 = b_xn + df2_cfg::kH;  // rows 1..7
        k_linear_f32_sg16(sel_h_proj, df2_cfg::kRank, df2_cfg::kH, hd7, b_gproj, 7);
        unembed_fn(hd7, b_raw_lg, 7);
        k_df2_top16_tail_sg16(b_raw_lg, df2_cfg::kV, 7, b_top_v, b_top_i, b_tail_m, b_tail_l);
        k_df2_select7_sg16(b_top_v, b_top_i, b_tail_m, b_tail_l, b_gproj, pred_cb, succ_cb, anchor_tok, b_draft, b_conf);

        int32_t h_d[7]{};
        float h_c[7]{};
        d2h(h_d, b_draft, sizeof(h_d));
        d2h(h_c, b_conf, sizeof(h_c));
        wait();

        // ceiling.py MoE expert-union calibrated adaptive draft width!
        // Choose K in 0..min(7, max_draft, max_m_cap-1) maximizing E[emitted(K)] / rel_step_cost(K+1)
        const int k_max = std::min({7, max_draft, max_m_cap - 1});
        // Relative verify cost from MoE union law: E_union(M) = 128*(1-(1-8/128)^M), with dense+draft overhead
        // M=1: 1.00 | M=2: 1.48 | M=3: 1.92 | M=4: 2.33 | M=5: 2.71 | M=6: 3.07 | M=7: 3.40 | M=8: 3.71
        constexpr float kRelCost[9] = {1.0f, 1.00f, 1.48f, 1.92f, 2.33f, 2.71f, 3.07f, 3.40f, 3.71f};
        int best_k = k_max;
        float best_rate = 0.0f;
        float cum_p = 1.0f;
        float exp_emit = 1.0f;
        for (int k = 1; k <= k_max; ++k) {
            cum_p *= std::clamp(h_c[k - 1], 0.0f, 1.0f);
            exp_emit += cum_p;
            const float rate = exp_emit / kRelCost[k + 1];
            if (rate > best_rate) {
                best_rate = rate;
                best_k = k;
            }
        }
        // Fall back to plain M=1 only when confidence is extremely flat (h_c[0] < 0.05)
        if (best_rate < 0.65f && h_c[0] < 0.05f && k_max >= 1) best_k = 0;

        for (int i = 0; i < best_k; ++i) {
            out_drafts[i] = h_d[i];
            out_confs[i] = h_c[i];
        }
        return best_k;
    }
};

}  // namespace ink_sycl
