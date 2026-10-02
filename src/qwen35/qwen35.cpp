// src/qwen35/qwen35.cpp - the Qwen35MoE forward pass.
//
// Plain float for the non-quantized path; a `Mat` of any other ggml type is decoded by `g_quant_matvec`, which
// the ggml-backed build installs (src/qwen35/qwen35_ggml.cpp).  Every op is checked by the tests beside this
// file against an independent implementation, and the target is parity with llama.cpp's own CPU ops.
#include "strata/qwen35/qwen35.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::qwen35 {

// The quantized hooks.  Null in the float-only library (the unit tests); installed by qwen35_enable_ggml().
QuantMatvecFn g_quant_matvec = nullptr;
RowDequantFn g_row_dequant = nullptr;
GpuMatvecFn g_gpu_matvec = nullptr;
GpuBatchFn g_gpu_batch = nullptr;

Mat mat_row(const Mat& m, int64_t i) {
    Mat r = m;
    r.data = (const uint8_t*) m.data + (size_t) i * m.row_bytes;
    r.n_out = 1;
    return r;
}

float mat_dot(const Mat& m, const float* x) {
    float y = 0.0f;
    matvec(m, x, &y);
    return y;
}

namespace {
void no_quant(int) {
    std::fprintf(stderr, "qwen35: a quantized weight was used, but this build has no ggml type traits\n");
    std::exit(1);
}
}  // namespace

void matvec(const Mat& m, const float* x, float* y) {
    if (m.n_out <= 0) return;
    if (m.dev && g_gpu_matvec && g_gpu_matvec(m, x, y)) return;
    if (m.type == 0) {
        for (int64_t o = 0; o < m.n_out; ++o) {
            const float* row = (const float*) m.data + o * m.n_in;
            double s = 0.0;
            for (int64_t i = 0; i < m.n_in; ++i) s += (double) row[i] * x[i];
            y[o] = (float) s;
        }
        return;
    }
    if (!g_quant_matvec) no_quant(m.type);
    g_quant_matvec(m.type, m.data, m.n_in, m.n_out, x, y);
}

namespace {
/// fp16 -> fp32, the standard bit arithmetic (no table, no ggml dependency).
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else { exp = 127 - 15 + 1; while (!(mant & 0x400u)) { mant <<= 1; --exp; } mant &= 0x3FFu; f = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 31) { f = sign | 0x7F800000u | (mant << 13); }
    else { f = sign | ((exp + 112u) << 23) | (mant << 13); }
    float r; std::memcpy(&r, &f, 4); return r;
}
}  // namespace

void dequant_row(const Mat& m, int64_t row, float* out) {
    const uint8_t* p = (const uint8_t*) m.data + (size_t) row * m.row_bytes;
    // F32 (the tests) and Q8_0 (the embedding and the head) are decoded here; other types go to ggml.
    if (m.type == 0) { std::memcpy(out, p, (size_t) m.n_in * sizeof(float)); return; }
    if (m.type == 8) {  // Q8_0: 32 values per 34-byte block (fp16 scale + 32 int8)
        const int64_t nb = m.n_in / 32;
        for (int64_t b = 0; b < nb; ++b) {
            uint16_t dh; std::memcpy(&dh, p + b * 34, 2);
            const float d = fp16_to_fp32(dh);
            const int8_t* q = (const int8_t*) (p + b * 34 + 2);
            for (int j = 0; j < 32; ++j) out[b * 32 + j] = (float) q[j] * d;
        }
        return;
    }
    if (!g_row_dequant) no_quant(m.type);
    g_row_dequant(m.type, p, m.n_in, out);
}

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

void gdn_layer(const Qwen35Geometry& g, const GdnLayerWeights& w, GdnState& st, const float* x, float* out) {
    const int64_t H = g.n_embd;
    const int64_t d_conv = g.ssm_conv_kernel;
    const int64_t Hk = g.ssm_groups;
    const int64_t Hv = g.ssm_dt_rank;
    const int64_t S = g.ssm_state;
    const int64_t key_dim = g.key_dim();
    const int64_t value_dim = g.value_dim();
    const int64_t qkv_dim = g.qkv_dim();
    const int64_t C = g.conv_channels();
    const float eps = g.rms_eps;

    std::vector<float> xn((size_t) H);
    rms_norm(x, w.attn_norm, H, eps, xn.data());

    std::vector<float> qkv((size_t) qkv_dim), z((size_t) value_dim);
    std::vector<float> beta((size_t) Hv), alpha((size_t) Hv), gate((size_t) Hv);
    bool batched = false;
    if (g_gpu_batch) {
        const Mat* bm[4] = {&w.wqkv, &w.wgate, &w.ssm_beta, &w.ssm_alpha};
        const float* bx[4] = {xn.data(), xn.data(), xn.data(), xn.data()};
        float* by[4] = {qkv.data(), z.data(), beta.data(), alpha.data()};
        batched = g_gpu_batch(bm, bx, by, 4) == 4;
    }
    if (!batched) {
        matvec(w.wqkv, xn.data(), qkv.data());
        matvec(w.wgate, xn.data(), z.data());
        for (int64_t h = 0; h < Hv; ++h) {
            beta[(size_t) h] = mat_dot(mat_row(w.ssm_beta, h), xn.data());
            alpha[(size_t) h] = mat_dot(mat_row(w.ssm_alpha, h), xn.data());
        }
    }
    for (int64_t h = 0; h < Hv; ++h) {
        beta[(size_t) h] = sigmoid(beta[(size_t) h]);
        alpha[(size_t) h] += w.ssm_dt[h];
        gate[(size_t) h] = softplus(alpha[(size_t) h]) * w.ssm_a[h];
    }

    std::vector<float> h((size_t) C);
    for (int64_t c = 0; c < C; ++c) {
        const float* taps = w.ssm_conv + c * d_conv;
        double s = (double) qkv[(size_t) c] * taps[d_conv - 1];
        for (int64_t j = 0; j < d_conv - 1; ++j) s += (double) st.conv[(size_t) (j * C + c)] * taps[j];
        h[(size_t) c] = silu((float) s);
    }
    for (int64_t j = 0; j + 1 < d_conv - 1; ++j)
        std::memcpy(&st.conv[(size_t) (j * C)], &st.conv[(size_t) ((j + 1) * C)], (size_t) C * sizeof(float));
    if (d_conv >= 2)
        std::memcpy(&st.conv[(size_t) ((d_conv - 2) * C)], qkv.data(), (size_t) C * sizeof(float));

    std::vector<float> qn((size_t) key_dim), kn((size_t) key_dim);
    const float* qh = h.data();
    const float* kh = h.data() + key_dim;
    const float* vh = h.data() + 2 * key_dim;
    for (int64_t hh = 0; hh < Hk; ++hh) {
        l2_norm(qh + hh * S, S, eps, qn.data() + hh * S);
        l2_norm(kh + hh * S, S, eps, kn.data() + hh * S);
    }

    const float scale = 1.0f / std::sqrt((float) S);
    std::vector<float> o((size_t) value_dim), delta((size_t) S);
    for (int64_t hv = 0; hv < Hv; ++hv) {
        const int64_t hk = hv % Hk;
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

    std::vector<float> y((size_t) value_dim), tmp((size_t) S);
    for (int64_t hv = 0; hv < Hv; ++hv) {
        rms_norm(o.data() + hv * S, w.ssm_norm, S, eps, tmp.data());
        for (int64_t j = 0; j < S; ++j) y[(size_t) (hv * S + j)] = tmp[(size_t) j] * silu(z[(size_t) (hv * S + j)]);
    }
    matvec(w.ssm_out, y.data(), out);
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
    bool batched = false;
    if (g_gpu_batch) {
        const Mat* bm[3] = {&w.wq, &w.wk, &w.wv};
        const float* bx[3] = {xn.data(), xn.data(), xn.data()};
        float* by[3] = {qfull.data(), kf.data(), vf.data()};
        batched = g_gpu_batch(bm, bx, by, 3) == 3;
    }
    if (!batched) {
        matvec(w.wq, xn.data(), qfull.data());
        matvec(w.wk, xn.data(), kf.data());
        matvec(w.wv, xn.data(), vf.data());
    }

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

    std::memcpy(&st.k[(size_t) pos * kv], kf.data(), (size_t) kv * sizeof(float));
    std::memcpy(&st.v[(size_t) pos * kv], vf.data(), (size_t) kv * sizeof(float));
    st.n += 1;

    std::vector<float> attn((size_t) D), o((size_t) Nh * D);
    std::vector<float> scores((size_t) (pos + 1));
    for (int64_t h = 0; h < Nh; ++h) {
        const int64_t hkv = h / ratio;
        const float* q = qfull.data() + h * 2 * D;
        const float* gate = q + D;
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
    }
    matvec(w.wo, o.data(), out);
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
    for (int64_t e = 0; e < E; ++e) {
        const float* row = w.gate_inp + e * H;
        double s = 0.0;
        for (int64_t i = 0; i < H; ++i) s += (double) row[i] * x[i];
        logits[(size_t) e] = (float) s;
    }
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

    std::vector<float> acc((size_t) H, 0.0f);
    std::vector<std::vector<float>> gg((size_t) K, std::vector<float>((size_t) F));
    std::vector<std::vector<float>> uu((size_t) K, std::vector<float>((size_t) F));
    std::vector<std::vector<float>> hh((size_t) K, std::vector<float>((size_t) F));
    std::vector<std::vector<float>> yy((size_t) K, std::vector<float>((size_t) H));
    const auto expert = [&](int64_t i) -> const ExpertWeights& { return w.experts[idx[(size_t) i]]; };
    // The K experts' gate and up share the input x: one batched GPU call streams their weights and runs all
    // 2K projections with a single upload/sync/download.  Then the down projections share nothing but are
    // still batched (one input each).
    {
        bool ok = false;
        if (g_gpu_batch) {
            std::vector<const Mat*> bm;
            std::vector<const float*> bx;
            std::vector<float*> by;
            for (int64_t i = 0; i < K; ++i) {
                bm.push_back(&expert(i).gate); bx.push_back(x); by.push_back(gg[(size_t) i].data());
                bm.push_back(&expert(i).up);   bx.push_back(x); by.push_back(uu[(size_t) i].data());
            }
            ok = g_gpu_batch(bm.data(), bx.data(), by.data(), (int) (2 * K)) == 2 * K;
        }
        if (!ok) {
            for (int64_t i = 0; i < K; ++i) {
                matvec(expert(i).gate, x, gg[(size_t) i].data());
                matvec(expert(i).up, x, uu[(size_t) i].data());
            }
        }
    }
    for (int64_t i = 0; i < K; ++i)
        for (int64_t f = 0; f < F; ++f) hh[(size_t) i][(size_t) f] = silu(gg[(size_t) i][(size_t) f]) * uu[(size_t) i][(size_t) f];
    {
        bool ok = false;
        if (g_gpu_batch) {
            std::vector<const Mat*> bm;
            std::vector<const float*> bx;
            std::vector<float*> by;
            for (int64_t i = 0; i < K; ++i) {
                bm.push_back(&expert(i).down); bx.push_back(hh[(size_t) i].data()); by.push_back(yy[(size_t) i].data());
            }
            ok = g_gpu_batch(bm.data(), bx.data(), by.data(), (int) K) == K;
        }
        if (!ok) for (int64_t i = 0; i < K; ++i) matvec(expert(i).down, hh[(size_t) i].data(), yy[(size_t) i].data());
    }
    for (int64_t i = 0; i < K; ++i)
        for (int64_t d = 0; d < H; ++d) acc[(size_t) d] += wts[(size_t) i] * yy[(size_t) i][(size_t) d];

    std::vector<float> sg((size_t) Fs), su((size_t) Fs), sh((size_t) Fs), sy((size_t) H);
    bool sh_batched = false;
    if (g_gpu_batch) {
        const Mat* bm[2] = {&w.gate_shexp, &w.up_shexp};
        const float* bx[2] = {x, x};
        float* by[2] = {sg.data(), su.data()};
        sh_batched = g_gpu_batch(bm, bx, by, 2) == 2;
    }
    if (!sh_batched) {
        matvec(w.gate_shexp, x, sg.data());
        matvec(w.up_shexp, x, su.data());
    }
    for (int64_t f = 0; f < Fs; ++f) sh[(size_t) f] = silu(sg[(size_t) f]) * su[(size_t) f];
    matvec(w.down_shexp, sh.data(), sy.data());
    double gs = 0.0;
    for (int64_t i = 0; i < H; ++i) gs += (double) w.gate_inp_shexp[i] * x[i];
    const float sgate = sigmoid((float) gs);
    for (int64_t d = 0; d < H; ++d) out[d] = acc[(size_t) d] + sy[(size_t) d] * sgate;
}

void trunk_forward(const Qwen35Geometry& g, const TrunkWeights& w, TrunkState& st, int64_t token, float* logits) {
    const int64_t H = g.n_embd;
    std::vector<float> x((size_t) H), xn((size_t) H), a((size_t) H), m((size_t) H);
    dequant_row(w.token_embd, token, x.data());
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const float* an = w.attn_norm[(size_t) l];
        const float* pn = w.post_attn_norm[(size_t) l];
        rms_norm(x.data(), an, H, g.rms_eps, xn.data());
        if (g.is_recurrent(l)) gdn_layer(g, w.gdn[(size_t) l], st.gdn[(size_t) l], xn.data(), a.data());
        else attn_layer(g, w.attn[(size_t) l], st.attn[(size_t) l], xn.data(), a.data());
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] += a[(size_t) i];
        rms_norm(x.data(), pn, H, g.rms_eps, xn.data());
        moe_layer(g, w.moe[(size_t) l], xn.data(), m.data());
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] += m[(size_t) i];
    }
    rms_norm(x.data(), w.output_norm, H, g.rms_eps, xn.data());
    matvec(w.output, xn.data(), logits);
}

}  // namespace strata::qwen35
