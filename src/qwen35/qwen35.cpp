// src/qwen35/qwen35.cpp - the Qwen35MoE forward pass, reference (float) implementation.
//
// Every op is checked by src/qwen35/gdn_test.cpp against an independent implementation, and the end goal is
// parity against llama.cpp's own CPU ops (ggml_compute_forward_gated_delta_net_one_chunk is the source of the
// recurrence here).  Correctness first: this is plain float, no quantized types yet.
#include "strata/qwen35/qwen35.hpp"

#include <cmath>
#include <cstring>

namespace strata::qwen35 {

void rms_norm(const float* x, const float* w, int64_t n, float eps, float* y) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = 1.0f / std::sqrt((float) (ss / (double) n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * r * (w ? w[i] : 1.0f);
}

void l2_norm(const float* x, int64_t n, float eps, float* y) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = 1.0f / std::sqrt((float) ss + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * r;
}

void matvec(const float* w, const float* x, int64_t n_in, int64_t n_out, float* y) {
    for (int64_t o = 0; o < n_out; ++o) {
        const float* row = w + o * n_in;
        double s = 0.0;
        for (int64_t i = 0; i < n_in; ++i) s += (double) row[i] * x[i];
        y[o] = (float) s;
    }
}

void gdn_layer(const Qwen35Geometry& g, const GdnLayerWeights& w, GdnState& st, const float* x, float* out) {
    const int64_t H = g.n_embd;
    const int64_t d_conv = g.ssm_conv_kernel;
    const int64_t Hk = g.ssm_groups;         // key heads
    const int64_t Hv = g.ssm_dt_rank;        // value heads
    const int64_t S = g.ssm_state;           // head_k_dim == head_v_dim
    const int64_t key_dim = g.key_dim();
    const int64_t value_dim = g.value_dim();
    const int64_t qkv_dim = g.qkv_dim();
    const int64_t C = g.conv_channels();
    const float eps = g.rms_eps;

    // 1. normalize the residual input, then the four projections (qkv, z, beta, alpha).
    std::vector<float> xn((size_t) H);
    rms_norm(x, w.attn_norm, H, eps, xn.data());

    std::vector<float> qkv((size_t) qkv_dim), z((size_t) value_dim);
    std::vector<float> beta((size_t) Hv), alpha((size_t) Hv), gate((size_t) Hv);
    matvec(w.wqkv, xn.data(), H, qkv_dim, qkv.data());
    matvec(w.wgate, xn.data(), H, value_dim, z.data());
    for (int64_t h = 0; h < Hv; ++h) {
        const float* rb = w.ssm_beta + h * H;
        const float* ra = w.ssm_alpha + h * H;
        double ab = 0.0, aa = 0.0;
        for (int64_t i = 0; i < H; ++i) { ab += (double) rb[i] * xn[(size_t) i]; aa += (double) ra[i] * xn[(size_t) i]; }
        beta[(size_t) h] = sigmoid((float) ab);
        alpha[(size_t) h] = (float) aa + w.ssm_dt[h];
        gate[(size_t) h] = softplus(alpha[(size_t) h]) * w.ssm_a[h];
    }

    // 2. depthwise conv over [conv_state | qkv] and SiLU on the whole conv output.  Tap j of channel c is
    //    ssm_conv[c*d_conv + j]; j = d_conv-1 is the NEW frame, 0..d_conv-2 the history oldest-first.
    std::vector<float> h((size_t) C);
    for (int64_t c = 0; c < C; ++c) {
        const float* taps = w.ssm_conv + c * d_conv;
        double s = (double) qkv[(size_t) c] * taps[d_conv - 1];
        for (int64_t j = 0; j < d_conv - 1; ++j) s += (double) st.conv[(size_t) (j * C + c)] * taps[j];
        h[(size_t) c] = silu((float) s);
    }
    // The new state is the last d_conv-1 frames of [history | qkv].
    for (int64_t j = 0; j + 1 < d_conv - 1; ++j)
        std::memcpy(&st.conv[(size_t) (j * C)], &st.conv[(size_t) ((j + 1) * C)], (size_t) C * sizeof(float));
    if (d_conv >= 2)
        std::memcpy(&st.conv[(size_t) ((d_conv - 2) * C)], qkv.data(), (size_t) C * sizeof(float));

    // 3. split q | k | v, l2-normalise q and k per head (v is NOT normalised).
    std::vector<float> qn((size_t) key_dim), kn((size_t) key_dim);
    const float* qh = h.data();
    const float* kh = h.data() + key_dim;
    const float* vh = h.data() + 2 * key_dim;
    for (int64_t hh = 0; hh < Hk; ++hh) {
        l2_norm(qh + hh * S, S, eps, qn.data() + hh * S);
        l2_norm(kh + hh * S, S, eps, kn.data() + hh * S);
    }

    // 4. the recurrence, per value head, in ggml's transposed state layout: M[j*S + i] = S[i][j].
    const float scale = 1.0f / std::sqrt((float) S);
    std::vector<float> o((size_t) value_dim);
    std::vector<float> delta((size_t) S);
    for (int64_t hv = 0; hv < Hv; ++hv) {
        const int64_t hk = hv % Hk;                 // the CPU kernel's `iq1 = iv1 % neq1`
        const float* qd = qn.data() + hk * S;
        const float* kd = kn.data() + hk * S;
        const float* vd = vh + hv * S;
        float* M = st.rec.data() + (size_t) hv * S * S;
        const float decay = std::exp(gate[(size_t) hv]);
        for (int64_t j = 0; j < S; ++j) {
            float* row = M + j * S;
            for (int64_t i = 0; i < S; ++i) row[i] *= decay;
            double sk = 0.0;
            for (int64_t i = 0; i < S; ++i) sk += (double) row[i] * kd[i];
            delta[(size_t) j] = (vd[j] - (float) sk) * beta[(size_t) hv];
            for (int64_t i = 0; i < S; ++i) row[i] += kd[i] * delta[(size_t) j];
            double oo = 0.0;
            for (int64_t i = 0; i < S; ++i) oo += (double) row[i] * qd[i];
            o[(size_t) (hv * S + j)] = (float) oo * scale;
        }
    }

    // 5. gated RMS norm: rms_norm(o, ssm_norm) * silu(z), then the output projection.
    std::vector<float> y((size_t) value_dim);
    std::vector<float> tmp((size_t) S);
    for (int64_t hv = 0; hv < Hv; ++hv) {
        rms_norm(o.data() + hv * S, w.ssm_norm, S, eps, tmp.data());
        for (int64_t j = 0; j < S; ++j) y[(size_t) (hv * S + j)] = tmp[(size_t) j] * silu(z[(size_t) (hv * S + j)]);
    }
    matvec(w.ssm_out, y.data(), value_dim, H, out);
}

void rope_neox(float* v, int64_t head_dim, int64_t n_rot, float base, int64_t pos) {
    const int64_t half = n_rot / 2;
    const double theta_scale = std::pow((double) base, -2.0 / (double) n_rot);
    double theta = (double) pos;
    for (int64_t k = 0; k < half; ++k) {
        const float c = (float) std::cos(theta), s = (float) std::sin(theta);
        const float x0 = v[k], x1 = v[k + half];
        v[k] = x0 * c - x1 * s;
        v[k + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
    (void) head_dim;
}

void attn_layer(const Qwen35Geometry& g, const AttnLayerWeights& w, AttnState& st, const float* x, float* out) {
    const int64_t H = g.n_embd;
    const int64_t Nh = g.n_head, Nkv = g.n_head_kv, D = g.head_dim;
    const int64_t ratio = Nh / Nkv;
    const int64_t kv = Nkv * D;
    const float eps = g.rms_eps;
    const float scale = 1.0f / std::sqrt((float) D);
    const int64_t pos = st.n;

    std::vector<float> xn((size_t) H);
    rms_norm(x, w.attn_norm, H, eps, xn.data());

    std::vector<float> qfull((size_t) 2 * Nh * D), kf((size_t) kv), vf((size_t) kv);
    matvec(w.wq, xn.data(), H, 2 * Nh * D, qfull.data());
    matvec(w.wk, xn.data(), H, kv, kf.data());
    matvec(w.wv, xn.data(), H, kv, vf.data());

    // q and k each get a per-head RMS norm; the second half of every q head is the output gate.
    for (int64_t h = 0; h < Nh; ++h) {
        float* qh = qfull.data() + h * 2 * D;
        rms_norm(qh, w.q_norm, D, eps, qh);
        rope_neox(qh, D, g.rope_dim, (float) g.rope_freq_base, pos);
    }
    for (int64_t h = 0; h < Nkv; ++h) {
        float* kh = kf.data() + h * D;
        rms_norm(kh, w.k_norm, D, eps, kh);
        rope_neox(kh, D, g.rope_dim, (float) g.rope_freq_base, pos);
    }

    // append K/V
    std::memcpy(&st.k[(size_t) pos * kv], kf.data(), (size_t) kv * sizeof(float));
    std::memcpy(&st.v[(size_t) pos * kv], vf.data(), (size_t) kv * sizeof(float));
    st.n += 1;

    // dense causal attention over [0, pos].
    std::vector<float> qh((size_t) D), attn((size_t) D), o((size_t) Nh * D);
    std::vector<float> scores((size_t) (pos + 1));
    for (int64_t h = 0; h < Nh; ++h) {
        const int64_t hkv = h / ratio;
        const float* q = qfull.data() + h * 2 * D;
        const float* gate = q + D;
        const float* kbase = st.k.data() + hkv * D;
        const float* vbase = st.v.data() + hkv * D;
        float mx = -INFINITY;
        for (int64_t j = 0; j <= pos; ++j) {
            double s = 0.0;
            for (int64_t d = 0; d < D; ++d) s += (double) q[d] * st.k[(size_t) j * kv + hkv * D + d];
            scores[(size_t) j] = (float) s * scale;
            mx = std::max(mx, scores[(size_t) j]);
        }
        double sum = 0.0;
        for (int64_t j = 0; j <= pos; ++j) { scores[(size_t) j] = std::exp(scores[(size_t) j] - mx); sum += scores[(size_t) j]; }
        const float inv = (float) (1.0 / sum);
        for (int64_t d = 0; d < D; ++d) attn[(size_t) d] = 0.0f;
        for (int64_t j = 0; j <= pos; ++j) {
            const float p = scores[(size_t) j] * inv;
            for (int64_t d = 0; d < D; ++d) attn[(size_t) d] += p * st.v[(size_t) j * kv + hkv * D + d];
        }
        for (int64_t d = 0; d < D; ++d) o[(size_t) (h * D + d)] = attn[(size_t) d] * sigmoid(gate[d]);
        (void) kbase; (void) vbase; (void) qh;
    }
    matvec(w.wo, o.data(), Nh * D, H, out);
}

namespace {
void softmax_inplace(std::vector<float>& p) {
    float mx = -INFINITY;
    for (float v : p) mx = std::max(mx, v);
    double s = 0.0;
    for (float& v : p) { v = std::exp(v - mx); s += v; }
    for (float& v : p) v = (float) (v / s);
}
}  // namespace

void moe_layer(const Qwen35Geometry& g, const MoeLayerWeights& w, const float* x, float* out) {
    const int64_t H = g.n_embd, E = g.n_expert, K = g.n_expert_used;
    const int64_t F = g.n_ff_exp, Fs = g.n_ff_shexp;

    std::vector<float> logits((size_t) E);
    matvec(w.gate_inp, x, H, E, logits.data());
    std::vector<float> probs = logits;
    softmax_inplace(probs);

    std::vector<int64_t> idx((size_t) E);
    for (int64_t e = 0; e < E; ++e) idx[(size_t) e] = e;
    std::partial_sort(idx.begin(), idx.begin() + (size_t) K, idx.end(),
                      [&](int64_t a, int64_t b) { return probs[(size_t) a] > probs[(size_t) b]; });
    std::vector<float> wts((size_t) K);
    double wsum = 0.0;
    for (int64_t i = 0; i < K; ++i) { wts[(size_t) i] = probs[(size_t) idx[(size_t) i]]; wsum += wts[(size_t) i]; }
    for (int64_t i = 0; i < K; ++i) wts[(size_t) i] = (float) (wts[(size_t) i] / wsum);

    std::vector<float> acc((size_t) H, 0.0f), h((size_t) F), y((size_t) H);
    for (int64_t i = 0; i < K; ++i) {
        const ExpertWeights& ew = w.experts[idx[(size_t) i]];
        std::vector<float> gg((size_t) F), uu((size_t) F);
        matvec(ew.gate, x, H, F, gg.data());
        matvec(ew.up, x, H, F, uu.data());
        for (int64_t f = 0; f < F; ++f) h[(size_t) f] = silu(gg[(size_t) f]) * uu[(size_t) f];
        matvec(ew.down, h.data(), F, H, y.data());
        for (int64_t d = 0; d < H; ++d) acc[(size_t) d] += wts[(size_t) i] * y[(size_t) d];
    }

    // shared expert, scaled by its sigmoid gate
    std::vector<float> sg((size_t) Fs), su((size_t) Fs), sh((size_t) Fs), sy((size_t) H);
    matvec(w.gate_shexp, x, H, Fs, sg.data());
    matvec(w.up_shexp, x, H, Fs, su.data());
    for (int64_t f = 0; f < Fs; ++f) sh[(size_t) f] = silu(sg[(size_t) f]) * su[(size_t) f];
    matvec(w.down_shexp, sh.data(), Fs, H, sy.data());
    double gs = 0.0;
    for (int64_t i = 0; i < H; ++i) gs += (double) w.gate_inp_shexp[i] * x[i];
    const float sgate = sigmoid((float) gs);
    for (int64_t d = 0; d < H; ++d) out[d] = acc[(size_t) d] + sy[(size_t) d] * sgate;
}

void trunk_forward(const Qwen35Geometry& g, const TrunkWeights& w, TrunkState& st, int64_t token, float* logits) {
    const int64_t H = g.n_embd;
    std::vector<float> x((size_t) H), xn((size_t) H), a((size_t) H), m((size_t) H);
    std::memcpy(x.data(), w.token_embd + (size_t) token * H, (size_t) H * sizeof(float));
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const float* an = w.attn_norm + (size_t) l * H;
        const float* pn = w.post_attn_norm + (size_t) l * H;
        rms_norm(x.data(), an, H, g.rms_eps, xn.data());
        if (g.is_recurrent(l)) gdn_layer(g, w.gdn[(size_t) l], st.gdn[(size_t) l], xn.data(), a.data());
        else attn_layer(g, w.attn[(size_t) l], st.attn[(size_t) l], xn.data(), a.data());
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] += a[(size_t) i];
        rms_norm(x.data(), pn, H, g.rms_eps, xn.data());
        moe_layer(g, w.moe[(size_t) l], xn.data(), m.data());
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] += m[(size_t) i];
    }
    rms_norm(x.data(), w.output_norm, H, g.rms_eps, xn.data());
    matvec(w.output, xn.data(), H, g.n_vocab, logits);
}

}  // namespace strata::qwen35
