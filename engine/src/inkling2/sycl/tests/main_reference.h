// Test-only SYCL reference kernels from Strata main 73d4f4196c25a65e02f6b9ef25d354826ce55105.
// Keep their original launch geometry and arithmetic independent of production dispatch.
#pragma once
#include "ink_sy_common.h"
namespace ink_sycl::main_reference {
inline void k_linear_f32_sg16(const float* W, int N, int K, const float* X, float* Y, int M) {
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const float* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane; k < K; k += 16) {
                const float w = wp[k];
                for (int i = 0; i < mm; ++i) a[i] += w * X[(size_t)(m0 + i) * K + k];
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, a[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

inline void k_linear_bf16_sg16(const uint16_t* W, int N, int K, const float* X, float* Y, int M) {
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4;
        const int lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const uint16_t* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane * 4; k < K; k += 64) {
                const float w0 = bf16_to_f32(wp[k + 0]), w1 = bf16_to_f32(wp[k + 1]);
                const float w2 = bf16_to_f32(wp[k + 2]), w3 = bf16_to_f32(wp[k + 3]);
                for (int i = 0; i < mm; ++i) {
                    const float* xp = X + (size_t)(m0 + i) * K + k;
                    a[i] += w0 * xp[0] + w1 * xp[1] + w2 * xp[2] + w3 * xp[3];
                }
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, a[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

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

            for (int k = k_start + sg; k < k_end; k += 8) {
                const int kidx = is_swa ? (k & 1023) : k;
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

inline void k_ctrl_apply_norm_sg16(const float* cur_buf, const float* msg_buf, const float* gate_buf,
                                   const float* w_gain, const float* w_forget, const float* norm_w,
                                   float* out_buf, int M, int do_norm) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        sycl::local_accessor<float, 1> g_sh(64, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            if (tid < 64) g_sh[tid] = gate_buf[(size_t)m * 64 + tid];
            sycl::group_barrier(it.get_group());
            const float* cp = cur_buf + (size_t)m * INK_D;
            const float* mp = msg_buf + (size_t)m * INK_D;
            float* dst = out_buf + (size_t)m * INK_D;
            float ss = 0.0f;
            for (int d = tid; d < INK_D; d += 256) {
                const float* wg = w_gain + (size_t)d * 64;
                const float* wf = w_forget + (size_t)d * 64;
                float sg = 0.0f, sf = 0.0f;
                for (int r = 0; r < 64; ++r) { sg += wg[r] * g_sh[r]; sf += wf[r] * g_sh[r]; }
                const float v = (1.0f - sycl::tanh(sycl::clamp(sf, -15.0f, 15.0f))) * cp[d]
                              + sycl::tanh(sycl::clamp(sg, -15.0f, 15.0f)) * mp[d];
                dst[d] = v;
                ss += v * v;
            }
            if (do_norm) {
                const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
                for (int d = tid; d < INK_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
            }
        });
    });
}

inline void k_attn_res_mix_sg16(const float* blocks_buf, const float* cur_buf, const float* queries, float* out_buf,
                                int n_blocks, int block_stride, int M, int n_queries) {
    const int M_stride = M * INK_D;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        sycl::local_accessor<float, 1> inv_s(8, h);
        sycl::local_accessor<float, 1> probs(32, h);  // [4][8]
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int S = n_blocks + 1;
            for (int s = 0; s < S; ++s) {
                const float* sp = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
                float ss = 0.0f;
                for (int d = tid; d < INK_D; d += 256) ss += sp[d] * sp[d];
                const float tot = wg_sum16(it, ss, sh);
                if (tid == 0) inv_s[s] = sycl::rsqrt(tot / (float)INK_D + INK_RMS_EPS);
            }
            sycl::group_barrier(it.get_group());
            const float inv_sqrt_d = 1.0f / 48.0f;
            for (int q = 0; q < n_queries; ++q) {
                const float* qv = queries + (size_t)q * INK_D;
                for (int s = 0; s < S; ++s) {
                    const float* sp = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
                    float dt = 0.0f;
                    for (int d = tid; d < INK_D; d += 256) dt += sp[d] * qv[d];
                    const float sc = wg_sum16(it, dt, sh) * inv_s[s] * inv_sqrt_d;
                    if (tid == 0) probs[q * 8 + s] = sc;
                }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    float mx = -1e30f;
                    for (int s = 0; s < S; ++s) mx = sycl::fmax(mx, probs[q * 8 + s]);
                    float sum = 0.0f;
                    for (int s = 0; s < S; ++s) { probs[q * 8 + s] = sycl::exp(probs[q * 8 + s] - mx); sum += probs[q * 8 + s]; }
                    const float isum = 1.0f / sum;
                    for (int s = 0; s < S; ++s) probs[q * 8 + s] *= isum * inv_s[s];
                }
                sycl::group_barrier(it.get_group());
                float* dst = out_buf + (size_t)q * M_stride + (size_t)m * INK_D;
                for (int d = tid; d < INK_D; d += 256) {
                    float acc = 0.0f;
                    for (int s = 0; s < S; ++s) {
                        const float* sp = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
                        acc += probs[q * 8 + s] * sp[d];
                    }
                    dst[d] = acc;
                }
            }
        });
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

} // namespace ink_sycl::main_reference
