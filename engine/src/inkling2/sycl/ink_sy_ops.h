// ink_sy_ops.h - Fused Intel Arc SYCL SG16 48-Visit Inkling2Loop Ops
// Strictly uses INK_REQD_SG16 (zero SG8 ICE on Battlemage B70 bmg_g31!).
// Supports c = 1, 4, 8, 32, 64 continuous multi-request batching + prefill + Vision bypass.
// 100% MIT clean-room implementation.
#pragma once

#include "ink_sy_common.h"

namespace ink_sycl {

constexpr int INK_D = 2304;
constexpr int INK_NH = 24;
constexpr int INK_NKV = 8;
constexpr int INK_HD = 128;
constexpr int INK_DREL = 16;
constexpr float INK_RMS_EPS = 1e-6f;

inline void k_embed_sg16(const InkTensorInfo& eti, const float* norm_w, const int* tok_ids,
                         const uint8_t* is_img, const float* img_emb, float* cur_out, int M, int V_padded) {
    const uint8_t* ew = eti.dev_ptr;
    const bool is_bf16 = (eti.fmt == FMT_BF16);
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        if (is_img && is_img[m] != 0) {
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = img_emb[(size_t)m * INK_D + d];
            return;
        }
        const int id = tok_ids[m];
        float ss = 0.0f;
        if (is_bf16) {
            const uint16_t* row = reinterpret_cast<const uint16_t*>(ew) + (size_t)id * INK_D;
            for (int d = tid; d < INK_D; d += 256) { float v = bf16_to_f32(row[d]); ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = bf16_to_f32(row[d]) * inv * norm_w[d];
        } else {
            const float sc = reinterpret_cast<const float*>(ew)[id];
            const int8_t* row = reinterpret_cast<const int8_t*>(ew + (size_t)V_padded * 4) + (size_t)id * INK_D;
            for (int d = tid; d < INK_D; d += 256) { float v = (float)row[d] * sc; ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = (float)row[d] * sc * inv * norm_w[d];
        }
    });
}

// For one token, each state/query pair can form its two reductions independently.
inline void k_res_mix_decode(const float* blocks, const float* cur, const float* queries, float* out,
                             int n_blocks, int stride, int n_queries, float* stats, float* copy_cur = nullptr) {
    const int states = n_blocks + 1;
    Q().parallel_for(NR(states * n_queries, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int pair = it.get_group(0), state = pair % states, q = pair / states;
        const float* x = state < n_blocks ? blocks + (size_t)state * stride : cur;
        const float* query = queries + q * INK_D;
        float sq = 0.0f, dot = 0.0f;
        for (int d = it.get_local_id(0); d < INK_D; d += 256) {
            sq += x[d] * x[d];
            dot += x[d] * query[d];
        }
        auto* shared = lmem<float, 16>(it);
        const float inv = sycl::rsqrt(wg_sum16(it, sq, shared) / (float)INK_D + INK_RMS_EPS);
        const float score = wg_sum16(it, dot, shared) * inv * (1.0f / 48.0f);
        if (it.get_local_id(0) == 0) { stats[pair * 2] = inv; stats[pair * 2 + 1] = score; }
    });
    Q().parallel_for(NR2(INK_D / 64, n_queries, 64), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int q = it.get_group(0), d = it.get_global_id(1);
        const float* st = stats + q * states * 2;
        float peak = -1e30f, sum = 0.0f, p[8];
        for (int s = 0; s < states; ++s) peak = sycl::fmax(peak, st[2 * s + 1]);
        for (int s = 0; s < states; ++s) { p[s] = sycl::exp(st[2 * s + 1] - peak); sum += p[s]; }
        const float inv_sum = 1.0f / sum;
        float v = 0.0f;
        for (int s = 0; s < states; ++s) {
            p[s] *= inv_sum * st[2 * s];
            v += p[s] * (s < n_blocks ? blocks[(size_t)s * stride + d] : cur[d]);
        }
        out[q * INK_D + d] = v;
        if (copy_cur && q == 0) copy_cur[d] = cur[d];
    });
}

// Several work-group sums at once: the same operations as N calls of wg_sum16 (sub-group reduce, cross-sub-group sum through local memory,
// sub-group reduce), so every sum is bit for bit what wg_sum16 gives, but the barriers are shared: two for all N values instead of two each.
// sh holds N * 16 floats.
template <int N, class It, class P>
inline void wg_sum16_multi(const It& it, float (&v)[N], P sh) {
    auto sg = it.get_sub_group();
    const int w = (int)sg.get_group_linear_id();
    const int l = (int)sg.get_local_linear_id();
    const int nw = (int)sg.get_group_linear_range();
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = sycl::reduce_over_group(sg, v[i], sycl::plus<float>());
    sycl::group_barrier(it.get_group());
    if (l == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) sh[i * 16 + w] = v[i];
    }
    sycl::group_barrier(it.get_group());
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const float r = (l < nw) ? (float)sh[i * 16 + l] : 0.0f;
        v[i] = sycl::reduce_over_group(sg, r, sycl::plus<float>());
    }
}

// One work-group per (token, query): the queries are independent, and every query repeats the (cheap) block norms, so the values are bit for bit
// those of the one-group-does-everything kernel; the S block norms and the S dot products are each reduced in one batch (two barriers) instead
// of one after the other.
// copy_cur (optional): the kernel also writes its current-stream input to copy_cur (the residual of the sublayer), replacing a separate d2d copy.
template<bool Combined>
inline void k_attn_res_mix_prefill(const float* blocks_buf, const float* cur_buf, const float* queries, float* out_buf,
                                  int n_blocks, int block_stride, int M, int n_queries, float* copy_cur) {
    const int M_stride = M * INK_D;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh((Combined ? 16 : 8) * 16, h);
        sycl::local_accessor<float, 1> probs(8, h);
        h.parallel_for(NR2((size_t)n_queries, (size_t)M, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), q = (int)it.get_group(1), tid = (int)it.get_local_id(1);
            const int S = n_blocks + 1;
            const float* sp[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                sp[s] = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
            }
            constexpr float inv_sqrt_d = 1.0f / 48.0f;
            const float* qv = queries + (size_t)q * INK_D;
            float inv_s[8], dt[8];
            if constexpr (Combined) {
                float sums[16]{};
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    float sq = 0.0f, dot = 0.0f;
                    if (s < S) {
                        for (int d = tid; d < INK_D; d += 256) {
                            const float value = sp[s][d];
                            sq += value * value;
                            dot += value * qv[d];
                        }
                    }
                    sums[2 * s] = sq;
                    sums[2 * s + 1] = dot;
                }
                // Use the same SG16 reduction order for both sums.
                wg_sum16_multi<16>(it, sums, sh);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    inv_s[s] = sycl::rsqrt(sums[2 * s] / (float)INK_D + INK_RMS_EPS);
                    dt[s] = sums[2 * s + 1];
                }
            } else {
                float ss[8];
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    float a = 0.0f;
                    if (s < S) for (int d = tid; d < INK_D; d += 256) a += sp[s][d] * sp[s][d];
                    ss[s] = a;
                }
                wg_sum16_multi<8>(it, ss, sh);
#pragma unroll
                for (int s = 0; s < 8; ++s) inv_s[s] = sycl::rsqrt(ss[s] / (float)INK_D + INK_RMS_EPS);
                sycl::group_barrier(it.get_group());   // sh is read again below
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    float a = 0.0f;
                    if (s < S) for (int d = tid; d < INK_D; d += 256) a += sp[s][d] * qv[d];
                    dt[s] = a;
                }
                wg_sum16_multi<8>(it, dt, sh);
            }
            if (tid == 0) {
                float sc[8];
                for (int s = 0; s < S; ++s) sc[s] = dt[s] * inv_s[s] * inv_sqrt_d;
                float mx = -1e30f;
                for (int s = 0; s < S; ++s) mx = sycl::fmax(mx, sc[s]);
                float sum = 0.0f;
                for (int s = 0; s < S; ++s) { sc[s] = sycl::exp(sc[s] - mx); sum += sc[s]; }
                const float isum = 1.0f / sum;
                for (int s = 0; s < S; ++s) probs[s] = sc[s] * (isum * inv_s[s]);
            }
            sycl::group_barrier(it.get_group());
            float* dst = out_buf + (size_t)q * M_stride + (size_t)m * INK_D;
            for (int d = tid; d < INK_D; d += 256) {
                float acc = 0.0f;
                for (int s = 0; s < S; ++s) acc += probs[s] * sp[s][d];
                dst[d] = acc;
            }
            if (copy_cur != nullptr && q == 0) {
                for (int d = tid; d < INK_D; d += 256) copy_cur[(size_t)m * INK_D + d] = cur_buf[(size_t)m * INK_D + d];
            }
        });
    });
}

inline void k_attn_res_mix_sg16(const float* blocks_buf, const float* cur_buf, const float* queries, float* out_buf,
                                int n_blocks, int block_stride, int M, int n_queries, float* stats = nullptr, float* copy_cur = nullptr) {
    static const bool split = ink_bmg_dense();
    if (M == 1 && stats && split) {
        k_res_mix_decode(blocks_buf, cur_buf, queries, out_buf, n_blocks, block_stride, n_queries, stats, copy_cur);
        return;
    }
    static const bool combined = ink_bmg_dense();
    if (combined && M >= 512)
        k_attn_res_mix_prefill<true>(blocks_buf, cur_buf, queries, out_buf, n_blocks, block_stride, M, n_queries, copy_cur);
    else
        k_attn_res_mix_prefill<false>(blocks_buf, cur_buf, queries, out_buf, n_blocks, block_stride, M, n_queries, copy_cur);
}

inline void k_ctrl_cat_rms_sg16(const float* cur_buf, const float* msg_buf, float* cat_out, int M) {
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* cp = cur_buf + (size_t)m * INK_D;
        const float* mp = msg_buf + (size_t)m * INK_D;
        float ssc = 0.0f, ssm = 0.0f;
        for (int d = tid; d < INK_D; d += 256) { ssc += cp[d] * cp[d]; ssm += mp[d] * mp[d]; }
        const float invc = sycl::rsqrt(wg_sum16(it, ssc, sh) / (float)INK_D + INK_RMS_EPS);
        const float invm = sycl::rsqrt(wg_sum16(it, ssm, sh) / (float)INK_D + INK_RMS_EPS);
        float* dst = cat_out + (size_t)m * (2 * INK_D);
        for (int d = tid; d < INK_D; d += 256) { dst[d] = cp[d] * invc; dst[INK_D + d] = mp[d] * invm; }
    });
}

// The loop controller's gated mix, then (when do_norm) the RMS norm. Two kernels instead of one work-group per token: the mix runs one thread per
// output channel across D/64 work-groups (same order of the 64-term dots, so the same bits), the norm is one work-group per token that adds the
// squares in exactly the order the old single kernel did (thread t takes channels t, t+256, ...; then wg_sum16), so the result is bit for bit the same.
inline void k_ctrl_apply_norm_sg16(const float* cur_buf, const float* msg_buf, const float* gate_buf,
                                   const float* w_gain, const float* w_forget, const float* norm_w,
                                   float* out_buf, int M, int do_norm) {
    static const bool tiled = ink_bmg_dense();
    if (tiled && M >= 512) {
        // Consume four gate rows per load of each gain/forget weight row.
        constexpr int tokens = 4, channels = 64;
        Q().submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> gates(tokens * channels, h);
            h.parallel_for(NR2(INK_D / channels, (M + tokens - 1) / tokens, channels),
                           [=](sycl::nd_item<2> it) INK_REQD_SG16 {
                const int start = it.get_group(0) * tokens, tid = it.get_local_id(1);
                const int d = it.get_group(1) * channels + tid;
                const int rows = sycl::min(tokens, M - start);
#pragma unroll
                for (int m = 0; m < tokens; ++m)
                    gates[m * channels + tid] = m < rows ? gate_buf[(start + m) * channels + tid] : 0.0f;
                sycl::group_barrier(it.get_group());
                float gain[tokens]{}, forget[tokens]{};
                for (int r = 0; r < channels; r += 8) {
                    float wg[8], wf[8];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        wg[t] = w_gain[d * channels + r + t];
                        wf[t] = w_forget[d * channels + r + t];
                    }
#pragma unroll
                    for (int m = 0; m < tokens; ++m) {
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            const float gate = gates[m * channels + r + t];
                            gain[m] += wg[t] * gate;
                            forget[m] += wf[t] * gate;
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < rows; ++m) {
                    const size_t i = size_t(start + m) * INK_D + d;
                    out_buf[i] = (1.0f - sycl::tanh(sycl::clamp(forget[m], -15.0f, 15.0f))) * cur_buf[i]
                                 + sycl::tanh(sycl::clamp(gain[m], -15.0f, 15.0f)) * msg_buf[i];
                }
            });
        });
    } else {
        Q().submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> g_sh(64, h);
            h.parallel_for(NR2(INK_D / 64, (size_t)M, 64), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
                const int m = (int)it.get_group(0), blk = (int)it.get_group(1), tid = (int)it.get_local_id(1);
                g_sh[tid] = gate_buf[(size_t)m * 64 + tid];
                sycl::group_barrier(it.get_group());
                const int d = blk * 64 + tid;
                const float* wg = w_gain + (size_t)d * 64;
                const float* wf = w_forget + (size_t)d * 64;
                float sg = 0.0f, sf = 0.0f;
                for (int r = 0; r < 64; ++r) { sg += wg[r] * g_sh[r]; sf += wf[r] * g_sh[r]; }
                const float cp = cur_buf[(size_t)m * INK_D + d];
                const float mp = msg_buf[(size_t)m * INK_D + d];
                out_buf[(size_t)m * INK_D + d] = (1.0f - sycl::tanh(sycl::clamp(sf, -15.0f, 15.0f))) * cp
                                               + sycl::tanh(sycl::clamp(sg, -15.0f, 15.0f)) * mp;
            });
        });
    }
    if (!do_norm) return;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            float* dst = out_buf + (size_t)m * INK_D;
            float ss = 0.0f;
            for (int d = tid; d < INK_D; d += 256) { const float v = dst[d]; ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
        });
    });
}

inline void k_rms_norm_sg16(const float* x_in, const float* w, float* y_out, int D, int M) {
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* xp = x_in + (size_t)m * D;
        float ss = 0.0f;
        for (int d = tid; d < D; d += 256) ss += xp[d] * xp[d];
        const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)D + INK_RMS_EPS);
        float* yp = y_out + (size_t)m * D;
        for (int d = tid; d < D; d += 256) yp[d] = xp[d] * inv * w[d];
    });
}

inline void k_add_sg16(const float* a, const float* b, float* out, int N) {
    Q().parallel_for(NR((N + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = (int)it.get_global_id(0);
        if (i < N) out[i] = a[i] + b[i];
    });
}

inline void k_sconv4_prefill(const float* x, const float* weights, const float* taps,
                             float* history, const int* slots, float* y, int rows, int channels) {
    constexpr int tile = 8, threads = 64;
    Q().parallel_for(NR2((channels + threads - 1) / threads, (rows + tile - 1) / tile, threads),
                     [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int begin = it.get_group(0) * tile, c = it.get_global_id(1);
        if (c >= channels) return;
        float* hist = history + (size_t)slots[0] * 3 * channels + c;
        const float* prefix = begin ? x + (size_t)(begin - 3) * channels + c : hist;
        float h0 = prefix[0], h1 = prefix[channels], h2 = prefix[2 * channels];
        const auto w = *reinterpret_cast<const sycl::vec<float, 4>*>(weights + 4 * c);
        for (int m = begin; m < sycl::min(rows, begin + tile); ++m) {
            const float cur = x[(size_t)m * channels + c];
            float sum = h0 * w[0] + h1 * w[1] + h2 * w[2] + cur * w[3];
            if (taps) {
                const float* t = taps + ((size_t)m * 8 + (c >> 7)) * 4;
                sum += cur * t[0] + h2 * t[1] + h1 * t[2] + h0 * t[3];
            }
            y[(size_t)m * channels + c] = (cur + sum) * (taps ? 1.0f / 3.0f : 0.4472135954999579f);
            h0 = h1; h1 = h2; h2 = cur;
        }
        // Other tiles get their three causal inputs directly from x.
        if (begin == 0) {
            for (int j = 0; j < 3; ++j) hist[j * channels] = x[(size_t)(rows - 3 + j) * channels + c];
        }
    });
}

inline void k_sconv4_multi_sg16(
    const float* x_in, const float* static_w, const float* dyn_taps, float* hist_pool,
    const int* active_slots, const int* slot_tok_off, const int* slot_tok_cnt, const int* slot_tok_idx,
    float* y_out, int n_active_slots, int C, int has_dyn,
    float* ring_out = nullptr, int save_ring = 0) {
    Q().parallel_for(NR2((C + 255) / 256, (size_t)n_active_slots, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int si = (int)it.get_group(0);
        const int c = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (c >= C) return;
        const int slot = active_slots[si], off = slot_tok_off[si], cnt = slot_tok_cnt[si];
        float* h = hist_pool + (size_t)slot * 3 * C;
        float h0 = h[0 * C + c], h1 = h[1 * C + c], h2 = h[2 * C + c];
        const float* sw = static_w + (size_t)c * 4;
        const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
        const float scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;
        const int hd_idx = c >> 7;
        for (int k = 0; k < cnt; ++k) {
            const int m = slot_tok_idx[off + k];
            const float cur = x_in[(size_t)m * C + c];
            float acc = h0 * sw0 + h1 * sw1 + h2 * sw2 + cur * sw3;
            if (has_dyn) {
                const float* dt = dyn_taps + ((size_t)m * 8 + hd_idx) * 4;
                acc += cur * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
            }
            y_out[(size_t)m * C + c] = (cur + acc) * scale;
            h0 = h1; h1 = h2; h2 = cur;
            if (save_ring && ring_out && si == 0 && k < 8) {
                float* rdst = ring_out + (size_t)k * 3 * C;
                rdst[0 * C + c] = h0; rdst[1 * C + c] = h1; rdst[2 * C + c] = h2;
            }
        }
        h[0 * C + c] = h0; h[1 * C + c] = h1; h[2 * C + c] = h2;
    });
}

inline void k_qr_clip_tau_sg16(float* q_buf, const float* q_scale, float* r_buf, const float* r_scale,
                               const int* pos_buf, int M, int is_swa) {
    Q().parallel_for(NR2(INK_NH, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), h = (int)it.get_group(1), tid = (int)it.get_local_id(1);
        const float r_pos = (float)(pos_buf[m] + 1) * (1.0f / 8192.0f);
        const float tau = is_swa ? 1.0f : (1.0f + 0.1f * sycl::log(sycl::fmax(1.0f, r_pos)));

        float* qp = q_buf + ((size_t)m * INK_NH + h) * INK_HD;
        const float qv = qp[tid] * q_scale[h * INK_HD + tid];
        const float qss = wg_sum16(it, qv * qv, sh);
        const float qfac = sycl::fmin(1.0f, 22.627416997969522f / (sycl::sqrt(qss) + 1e-6f)) * tau;
        qp[tid] = qv * qfac;

        if (tid < 16) {
            float* rp = r_buf + ((size_t)m * INK_NH + h) * INK_DREL;
            const float rv = rp[tid] * r_scale[h * INK_DREL + tid];
            const float rss = sg_sum(it, rv * rv);
            const float rfac = sycl::fmin(1.0f, 8.0f / (sycl::sqrt(rss) + 1e-6f)) * tau;
            rp[tid] = rv * rfac;
        }
    });
}

inline void k_kv_norm_store_f16_sg16(const float* k_in, const float* v_in, half* k_cache, half* v_cache,
                                     const int* slot_ids, const int* pos_buf, int M, int is_swa, int kv_stride) {
    Q().parallel_for(NR2(INK_NKV, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), kh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
        const int slot = slot_ids[m], pos = pos_buf[m];
        const int kpos = is_swa ? pos % kv_stride : pos;

        const float kv = k_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        const float kss = wg_sum16(it, kv * kv, sh);
        const float kfac = sycl::fmin(1.0f, 22.627416997969522f / (sycl::sqrt(kss) + 1e-6f));

        const float vv = v_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        const float vss = wg_sum16(it, vv * vv, sh);
        const float vfac = sycl::rsqrt(vss / (float)INK_HD + INK_RMS_EPS);

        const size_t dst = (((size_t)slot * kv_stride + kpos) * INK_NKV + kh) * INK_HD + tid;
        k_cache[dst] = (half)(kv * kfac);
        v_cache[dst] = (half)(vv * vfac);
    });
}

// Fused RelPos + Sink 24:8 GQA Flash Attention + Sigmoid Gate (SG16: 8 SGs = 128 threads, 8 dims/lane!)
inline void k_relpos_sink_gqa_sg16(
    const float* q_buf, const float* r_buf, const float* rel_proj_pad, const float* sinks_buf,
    const float* gate_buf, const half* k_cache, const half* v_cache,
    const int* slot_ids, const int* pos_buf, float* attn_out,
    int M, int is_swa, int swa_win, int rel_extent, int kv_stride) {
    const float scale = 1.0f / std::sqrt((float)INK_HD);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_m(8, h);
        sycl::local_accessor<float, 1> sh_l(8, h);
        sycl::local_accessor<float, 1> sh_o(8 * INK_HD, h);
        sycl::local_accessor<float, 1> sh_r(16, h);
        h.parallel_for(NR2(INK_NH, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), head = (int)it.get_group(1);
            const int tid = (int)it.get_local_id(1);
            const int sg = tid >> 4, lane = tid & 15;
            const int kh = head / 3;
            const int slot = slot_ids[m], qpos = pos_buf[m];

            if (tid < 16) sh_r[tid] = r_buf[((size_t)m * INK_NH + head) * INK_DREL + tid];
            sycl::group_barrier(it.get_group());

            // Each SG16 lane holds 8 dims: d = lane*8 .. lane*8+7
            float qv[8], o_acc[8];
            const float* qp = q_buf + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
            for (int u = 0; u < 8; ++u) { qv[u] = qp[u] * scale; o_acc[u] = 0.0f; }

            const int k_start = is_swa ? sycl::max(0, qpos - swa_win + 1) : 0;
            const int k_end = qpos + 1;
            float m_max = (sg == 0) ? sinks_buf[(size_t)m * INK_NH + head] : -1e30f;
            float l_sum = (sg == 0) ? 1.0f : 0.0f;
            const size_t slot_base = (size_t)slot * kv_stride;
            // Keys are in the query's ring cycle or the immediately preceding one.
            const int cycle_start = is_swa ? qpos - qpos % kv_stride : 0;

            for (int k = k_start + sg; k < k_end; k += 8) {
                const int offset = k - cycle_start;
                const int kidx = offset < 0 ? offset + kv_stride : offset;
                const size_t kv_off = ((slot_base + kidx) * INK_NKV + kh) * INK_HD + lane * 8;
                float dt = 0.0f;
                for (int u = 0; u < 8; ++u) dt += qv[u] * (float)k_cache[kv_off + u];
                float sc = sg_sum(it, dt);

                const int dist = sycl::min(qpos - k, rel_extent);
                sc += sg_sum(it, sh_r[lane] * rel_proj_pad[(size_t)lane * 1025 + dist]);

                const float mn = sycl::fmax(m_max, sc);
                const float corr = sycl::exp(m_max - mn);
                const float p = sycl::exp(sc - mn);
                l_sum = l_sum * corr + p;
                m_max = mn;
                for (int u = 0; u < 8; ++u) o_acc[u] = o_acc[u] * corr + p * (float)v_cache[kv_off + u];
            }

            if (lane == 0) { sh_m[sg] = m_max; sh_l[sg] = l_sum; }
            for (int u = 0; u < 8; ++u) sh_o[sg * INK_HD + lane * 8 + u] = o_acc[u];
            sycl::group_barrier(it.get_group());

            if (sg == 0) {
                float gm = -1e30f;
                for (int s = 0; s < 8; ++s) gm = sycl::fmax(gm, sh_m[s]);
                float gl = 0.0f, go[8] = {};
                for (int s = 0; s < 8; ++s) {
                    const float w = sycl::exp(sh_m[s] - gm);
                    gl += sh_l[s] * w;
                    for (int u = 0; u < 8; ++u) go[u] += sh_o[s * INK_HD + lane * 8 + u] * w;
                }
                const float il = 1.0f / gl;
                const float* gp = gate_buf + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
                float* op = attn_out + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
                for (int u = 0; u < 8; ++u) op[u] = (go[u] * il) / (1.0f + sycl::exp(-gp[u]));
            }
        });
    });
}

// The winning score has a zero difference (or NaN for a non-finite
// self-difference). exp(0) is one; both expressions propagate NaN.
inline sycl::float2 swa_online_factors(float peak, float score, float mx) {
    const bool rise = score > peak;
    const float da = peak - mx, db = score - mx;
    const float factor = sycl::exp(rise ? da : db);
    const float unit = 1.f + (rise ? db : da);
    return {rise ? factor : unit, rise ? unit : factor};
}

// Three query heads share a KV head in Inkling. Stream each KV vector once.
template<bool Optimized = false, int Tile = 0, bool Sliding = false>
inline void k_relpos_prefill_shared_kv(const float* q_buf, const float* r_buf, const float* positions_table,
                                       const float* sinks, const float* gate, const half* k_cache,
                                       const half* v_cache, const int* slots, const int* positions,
                                       float* out, int rows, bool sliding, int window, int extent,
                                       int stride) {
    const float scale = 1.0f / std::sqrt((float)INK_HD);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(3 * 8 * (INK_HD + 2), h);
        h.parallel_for(NR2(INK_NKV, rows, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int m = it.get_group(0), kh = it.get_group(1);
            const int tid = it.get_local_id(1), sg = tid / 16, lane = tid % 16;
            const size_t head = (size_t)m * INK_NH + 3 * kh;
            const int pos = positions[m], start = sliding ? sycl::max(0, pos + 1 - window) : 0;
            const size_t base = (size_t)slots[m] * stride;
            const int cycle = sliding ? pos - pos % stride : 0;
            float query[3][8], rel[3], peak[3], norm[3], far_bias[3], accum[3][8] = {};
            for (int j = 0; j < 3; ++j) {
                for (int d = 0; d < 8; ++d)
                    query[j][d] = q_buf[(head + j) * INK_HD + lane * 8 + d] * scale;
                rel[j] = r_buf[(head + j) * INK_DREL + lane];
                if constexpr (Optimized)
                    far_bias[j] = sg_sum(it, rel[j] * positions_table[(size_t)lane * 1025 + extent]);
                peak[j] = sg == 0 ? sinks[head + j] : -1e30f;
                norm[j] = sg == 0 ? 1.0f : 0.0f;
            }
            auto accumulate = [&](int k, bool distant, const sycl::vec<half, 8>& key,
                                  const sycl::vec<half, 8>& val) {
                const float rp = positions_table[(size_t)lane * 1025 + sycl::min(pos - k, extent)];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < 8; ++d)
                        dot += query[j][d] * (float)key[d];
                    const float score = sg_sum(it, dot) + (distant ? far_bias[j] : sg_sum(it, rel[j] * rp));
                    const float mx = sycl::fmax(peak[j], score);
                    float a, b;
                    if constexpr (Optimized) {
                        const auto factors = swa_online_factors(peak[j], score, mx);
                        a = factors.x();
                        b = factors.y();
                    } else {
                        a = sycl::exp(peak[j] - mx);
                        b = sycl::exp(score - mx);
                    }
                    norm[j] = norm[j] * a + b;
                    peak[j] = mx;
                    for (int d = 0; d < 8; ++d)
                        accum[j][d] = accum[j][d] * a + b * (float)val[d];
                }
            };
            int k = start + sg;
            if constexpr (Tile > 0) {
                const int clip = pos - extent + 1;
                auto prefix = [&](int end, int adjustment, bool distant) {
                    for (; k + 8 * (Tile - 1) < end; k += 8 * Tile) {
                        sycl::vec<half, 8> keys[Tile], values[Tile];
#pragma unroll
                        for (int t = 0; t < Tile; ++t) {
                            const size_t idx = k + 8 * t + adjustment;
                            const size_t off = ((base + idx) * INK_NKV + kh) * INK_HD + lane * 8;
                            keys[t] = *reinterpret_cast<const sycl::vec<half, 8>*>(k_cache + off);
                            values[t] = *reinterpret_cast<const sycl::vec<half, 8>*>(v_cache + off);
                        }
#pragma unroll
                        for (int t = 0; t < Tile; ++t) accumulate(k + 8 * t, distant, keys[t], values[t]);
                    }
                    for (; k < end; k += 8) {
                        const size_t off = ((base + k + adjustment) * INK_NKV + kh) * INK_HD + lane * 8;
                        const auto key = *reinterpret_cast<const sycl::vec<half, 8>*>(k_cache + off);
                        const auto value = *reinterpret_cast<const sycl::vec<half, 8>*>(v_cache + off);
                        accumulate(k, distant, key, value);
                    }
                };
                auto stream = [&](int end, int adjustment) {
                    prefix(sycl::min(end, clip), adjustment, true);
                    prefix(end, adjustment, false);
                };
                if constexpr (Sliding) {
                    stream(sycl::min(pos + 1, cycle), stride - cycle);
                    stream(pos + 1, -cycle);
                } else {
                    stream(pos + 1, 0);
                }
            } else {
                auto step = [&](int k, bool distant) {
                    const int idx = k < cycle ? k - cycle + stride : k - cycle;
                    const size_t off = ((base + idx) * INK_NKV + kh) * INK_HD + lane * 8;
                    const auto key = *reinterpret_cast<const sycl::vec<half, 8>*>(k_cache + off);
                    const auto val = *reinterpret_cast<const sycl::vec<half, 8>*>(v_cache + off);
                    accumulate(k, distant, key, val);
                };
                // Relative positions in this prefix all use the same clipped bias.
                if constexpr (Optimized)
                    for (; k <= pos - extent; k += 8) step(k, true);
                for (; k <= pos; k += 8) step(k, false);
            }
            for (int j = 0; j < 3; ++j) {
                const int off = (j * 8 + sg) * (INK_HD + 2);
                if (lane == 0) {
                    partial[off] = peak[j];
                    partial[off + 1] = norm[j];
                }
                for (int d = 0; d < 8; ++d)
                    partial[off + 2 + lane * 8 + d] = accum[j][d];
            }
            sycl::group_barrier(it.get_group());
            if (sg >= 3)
                return;
            const int first = sg * 8 * (INK_HD + 2);
            float peak_all = -1e30f, norm_all = 0.0f, v[8] = {};
            for (int s = 0; s < 8; ++s)
                peak_all = sycl::fmax(peak_all, partial[first + s * (INK_HD + 2)]);
            for (int s = 0; s < 8; ++s) {
                const int off = first + s * (INK_HD + 2);
                const float w = sycl::exp(partial[off] - peak_all);
                norm_all += partial[off + 1] * w;
                for (int d = 0; d < 8; ++d)
                    v[d] += partial[off + 2 + lane * 8 + d] * w;
            }
            const float inv = 1.0f / norm_all;
            const size_t off = (head + sg) * INK_HD + lane * 8;
            for (int d = 0; d < 8; ++d)
                out[off + d] = (v[d] * inv) / (1.0f + sycl::exp(-gate[off + d]));
        });
    });
}

// A workgroup owns one of the original eight interleaved KV streams. Its
// subgroups score eight independent keys, then fold them in original order.
inline void k_relpos_decode_split_kv(const float* query, const float* rel, const float* table,
                                     const float* sinks, const float* gate, const half* keys,
                                     const half* values, int slot, int pos, float* out, float* partial,
                                     bool sliding, int window, int extent, int stride) {
    constexpr int pitch = INK_HD + 2;
    const float scale = 1.0f / std::sqrt((float)INK_HD);
    const int start = sliding ? std::max(0, pos + 1 - window) : 0;
    const int cycle = sliding ? pos - pos % stride : 0;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scores(8, h);
        sycl::local_accessor<half, 1> tile(8 * INK_HD, h);
        h.parallel_for(NR2(8, INK_NH, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int head = it.get_group(0), stream = it.get_group(1), tid = it.get_local_id(1);
            const int sg = tid / 16, lane = tid % 16, kh = head / 3;
            const size_t base = (size_t)slot * stride;
            float q[8];
            for (int d = 0; d < 8; ++d)
                q[d] = query[head * INK_HD + lane * 8 + d] * scale;
            const float r = rel[head * INK_DREL + lane];
            float peak = stream == 0 ? sinks[head] : -1e30f, norm = stream == 0 ? 1.0f : 0.0f, sum = 0.0f;
            for (int block = start + stream; block <= pos; block += 64) {
                const int k = block + 8 * sg;
                if (k <= pos) {
                    const int idx = k < cycle ? k - cycle + stride : k - cycle;
                    const auto key = *reinterpret_cast<const sycl::vec<half, 8>*>(
                        keys + ((base + idx) * INK_NKV + kh) * INK_HD + lane * 8);
                    const auto v = *reinterpret_cast<const sycl::vec<half, 8>*>(
                        values + ((base + idx) * INK_NKV + kh) * INK_HD + lane * 8);
#pragma unroll
                    for (int d = 0; d < 8; ++d) tile[sg * INK_HD + lane * 8 + d] = v[d];
                    float dot = 0.0f;
                    for (int d = 0; d < 8; ++d)
                        dot += q[d] * (float)key[d];
                    const float score =
                        sg_sum(it, dot) +
                        sg_sum(it, r * table[(size_t)lane * 1025 + sycl::min(pos - k, extent)]);
                    if (lane == 0)
                        scores[sg] = score;
                }
                sycl::group_barrier(it.get_group());
                for (int j = 0; j < 8 && block + 8 * j <= pos; ++j) {
                    const float score = scores[j], mx = sycl::fmax(peak, score);
                    const float a = sycl::exp(peak - mx), b = sycl::exp(score - mx);
                    norm = norm * a + b;
                    peak = mx;
                    const float v = (float)tile[j * INK_HD + tid];
                    sum = sum * a + b * v;
                }
                sycl::group_barrier(it.get_group());
            }
            float* dst = partial + (head * 8 + stream) * pitch;
            if (tid == 0) {
                dst[0] = peak;
                dst[1] = norm;
            }
            dst[2 + tid] = sum;
        });
    });
    Q().parallel_for(NR(INK_NH, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int head = it.get_group(0), d = it.get_local_id(0);
        const float* src = partial + head * 8 * pitch;
        float peak = -1e30f, norm = 0.0f, sum = 0.0f;
        for (int s = 0; s < 8; ++s)
            peak = sycl::fmax(peak, src[s * pitch]);
        for (int s = 0; s < 8; ++s) {
            const float w = sycl::exp(src[s * pitch] - peak);
            norm += src[s * pitch + 1] * w;
            sum += src[s * pitch + 2 + d] * w;
        }
        out[head * INK_HD + d] = (sum * (1.0f / norm)) / (1.0f + sycl::exp(-gate[head * INK_HD + d]));
    });
}

inline void k_router_topk8_sg16(const float* r_logits, const float* r_bias, int* topk_idx, float* topk_w,
                                float route_scale, int M) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_p(128, h);
        sycl::local_accessor<float, 1> sh_s(128, h);
        h.parallel_for(NR(M, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            auto grp = it.get_group();
            const float raw_l = r_logits[(size_t)m * 128 + tid];
            const float p_raw = 1.0f / (1.0f + sycl::exp(-raw_l));
            const float p = sycl::isfinite(p_raw) ? p_raw : 0.0f;
            const float sc = p + r_bias[tid];
            sh_p[tid] = p;
            sh_s[tid] = sycl::isfinite(sc) ? sc : -1e29f;
            sycl::group_barrier(grp);
            // eight rounds of "the best remaining expert, the lowest index among equals" (max and min are exact, so the choice is
            // the same as the serial scan it replaces); thread 0 keeps the weights in the old order
            int sel[8]; float pw[8]; float sum = 0.0f;
            for (int k = 0; k < 8; ++k) {
                const float v = sh_s[tid];
                const float mx = sycl::reduce_over_group(grp, v, sycl::maximum<float>());
                const int cand = (v == mx) ? tid : (1 << 30);
                int bi = sycl::reduce_over_group(grp, cand, sycl::minimum<int>());
                if (!(mx > -1e30f)) bi = k;
                if (tid == 0) { sel[k] = bi; pw[k] = sh_p[bi]; sum += pw[k]; }
                sycl::group_barrier(grp);
                if (tid == bi) sh_s[tid] = -1e30f;
                sycl::group_barrier(grp);
            }
            if (tid == 0) {
                const float inv = route_scale / (sum + 1e-20f);
                for (int k = 0; k < 8; ++k) { topk_idx[(size_t)m * 8 + k] = sel[k]; topk_w[(size_t)m * 8 + k] = pw[k] * inv; }
            }
        });
    });
}

// One subgroup keeps all 128 candidates in registers. Resolve ties by expert ID.
inline void k_sq20_decode_top8(const float *logits, const float *bias, int *ids, float *weights,
                               float scale, int rows = 1) {
    Q().parallel_for(NR(rows, 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int row = it.get_group(0), lane = it.get_local_id(0);
        const auto sg = it.get_sub_group();
        float p[8], score[8], my_weight = 0.f, sum = 0.f;
        int my_id = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int e = 16 * j + lane;
            const float raw = 1.f / (1.f + sycl::exp(-logits[row * 128 + e]));
            p[j] = sycl::isfinite(raw) ? raw : 0.f;
            const float s = p[j] + bias[e];
            score[j] = sycl::isfinite(s) ? s : -1e29f;
        }
        for (int rank = 0; rank < 8; ++rank) {
            float best = -1e30f;
            int local_id = 128;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (score[j] > best) {
                    best = score[j];
                    local_id = 16 * j + lane;
                }
            const float maximum = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
            int id = sycl::reduce_over_group(sg, best == maximum ? local_id : 128, sycl::minimum<int>());
            if (id == 128)
                id = rank;
            const int j = id / 16;
            const float chosen = sycl::select_from_group(sg, p[j], sycl::id<1>(id % 16));
            if (rank == lane) {
                my_id = id;
                my_weight = chosen;
            }
            sum += chosen;
            if (lane == id % 16)
                score[j] = -1e30f;
        }
        const float inv = scale / (sum + 1e-20f);
        if (lane < 8) {
            ids[row * 8 + lane] = my_id;
            weights[row * 8 + lane] = my_weight * inv;
        }
    });
}

// Single-token MLP ShortConv has static taps and consumes exactly one history slot.
inline void k_sq20_decode_tail(const float *weighted, const float *taps, float *history, const int *slots,
                               const float *residual, float *out) {
    Q().parallel_for(NR(9, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        constexpr int D = 2304;
        const int d = it.get_global_id(0);
        float cur = 0.f;
        for (int k = 0; k < 8; ++k)
            cur += weighted[k * D + d];
        float *h = history + (size_t)slots[0] * 3 * D + d;
        const float *sw = taps + 4 * d;
        const float h1 = h[D], h2 = h[2 * D];
        const float acc = h[0] * sw[0] + h1 * sw[1] + h2 * sw[2] + cur * sw[3];
        h[0] = h1;
        h[D] = h2;
        h[2 * D] = cur;
        const float conv = (cur + acc) * 0.4472135954999579f;
        out[d] = residual[d] + conv;
    });
}

inline void k_softcap_sg16(float* logits, int total, float cap) {
    const float inv = 1.0f / cap;
    Q().parallel_for(NR((total + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = (int)it.get_global_id(0);
        if (i < total) logits[i] = cap * sycl::tanh(sycl::clamp(logits[i] * inv, -15.0f, 15.0f));
    });
}

inline void k_argmax_rows_sg16(const float* logits, int* out_ids, int V, int M) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_v(256, h);
        sycl::local_accessor<int, 1> sh_i(256, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const float* row = logits + (size_t)m * V;
            float bv = -1e30f; int bi = 0;
            for (int i = tid; i < V; i += 256) {
                const float v = row[i];
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
            sh_v[tid] = bv; sh_i[tid] = bi;
            sycl::group_barrier(it.get_group());
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    const float v1 = sh_v[tid + s]; const int i1 = sh_i[tid + s];
                    if (v1 > sh_v[tid] || (v1 == sh_v[tid] && i1 < sh_i[tid])) { sh_v[tid] = v1; sh_i[tid] = i1; }
                }
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) out_ids[m] = sh_i[0];
        });
    });
}

}  // namespace ink_sycl
