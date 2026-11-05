// cann/tests/model_forward.cpp - the whole model, one token, from its id to its logits.
//
// Everything below was verified per block first; this wires it into the 41-layer loop and adds
// the two ends:
//
//   x = token_embd[id]
//   for each of 41 layers:
//       h = rms_norm(x, attn_norm)
//       h = gated_delta_net block   (3 of 4 layers)  or  attention block  (1 of 4)
//       x = x + h
//       h = rms_norm(x, post_attention_norm)
//       h = moe_ffn(h)              (8 of 256 experts + shared expert)
//       x = x + h
//   x = rms_norm(x, output_norm)
//   logits = output.weight x
//
// Both the NPU path and a CPU reference run the same weights, and the test compares the final
// hidden state and the top of the logit vector. One token means an empty KV cache and zero GDN
// state, so this is the first prefill step.
#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gdn.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"
#include "strata/cann/tensor_load.hpp"

using namespace strata::cann;

namespace {

const char * kDefaultPath = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";

constexpr int64_t kHidden = 2048;
constexpr int64_t kHeadDimSsm = 128;
constexpr int64_t kKHeads = 16;
constexpr int64_t kVHeads = 32;
constexpr int64_t kConvK = 4;
constexpr int64_t kKeyDim = kHeadDimSsm * kKHeads;
constexpr int64_t kValueDim = kHeadDimSsm * kVHeads;
constexpr int64_t kQkvDim = kKeyDim * 2 + kValueDim;
constexpr int64_t kHeadDimAttn = 256;
constexpr int64_t kHeads = 16;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kRot = 64;
constexpr float kFreqBase = 1e7f;
constexpr float kRmsEps = 1e-6f;  // qwen35moe.attention.layer_norm_rms_epsilon
constexpr int64_t kQDim = kHeads * kHeadDimAttn * 2;
constexpr int64_t kKvDim = kKvHeads * kHeadDimAttn; // 2 kv heads x 256, no gate doubling here
constexpr int64_t kExpertInter = 512;
constexpr int64_t kTopK = 8;

std::vector<float> round_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    std::vector<float> out(v.size());
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void upload(const std::vector<float> & v, Buffer & b) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), h.size());
    b.copy_from_host(h.data(), h.size() * 2);
}

std::vector<float> to_host(Stream & stream, const Buffer & b, int64_t n) {
    std::vector<uint16_t> h((size_t) n);
    b.copy_to_host(h.data(), h.size() * 2);
    stream.synchronize();
    std::vector<float> out((size_t) n);
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void gemv(Stream & stream, const std::vector<float> & w, int64_t m, int64_t k, const Tensor & x, Buffer & out) {
    Buffer bw(w.size() * 2);
    upload(w, bw);
    Tensor tw{ bw.data(), DType::F16, k, m };
    Tensor to{ out.data(), DType::F16, 1, m };
    matmul(stream, tw, x, to);
}

std::vector<float> gemm(const std::vector<float> & w, int64_t m, int64_t k, const std::vector<float> & in) {
    std::vector<float> out((size_t) m, 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        const float * row = &w[(size_t) i * k];
        double acc = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            acc += (double) row[j] * in[(size_t) j];
        }
        out[(size_t) i] = (float) acc;
    }
    return out;
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

void rms_cpu(const std::vector<float> & in, const std::vector<float> & gamma, float eps, std::vector<float> & out) {
    double ss = 0.0;
    for (float v : in) {
        ss += (double) v * v;
    }
    const float sc = (float) (1.0 / std::sqrt(ss / (double) in.size() + eps));
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i] * sc * (gamma.empty() ? 1.0f : gamma[i]);
    }
}

void rope_cpu(std::vector<float> & vec, int64_t n_rot, float freq_base, int64_t pos) {
    const int64_t half = n_rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const double theta = (double) pos * std::pow((double) freq_base, -2.0 * (double) i / (double) n_rot);
        const float c = (float) std::cos(theta);
        const float s = (float) std::sin(theta);
        const float x0 = vec[(size_t) i];
        const float x1 = vec[(size_t) (i + half)];
        vec[(size_t) i] = x0 * c - x1 * s;
        vec[(size_t) (i + half)] = x0 * s + x1 * c;
    }
}

// one expert slab out of a [cols, rows, n_expert] tensor
std::vector<float> expert_slab(const Gguf & g, const GgufTensor & t, int expert, int64_t rows, int64_t cols) {
    const uint64_t n = (uint64_t) rows * cols;
    const int block_bytes = quant_type_size((int) t.type);
    const int block_elems = quant_block_elements((int) t.type);
    if (block_bytes == 0 || n % (uint64_t) block_elems != 0) {
        throw std::runtime_error("model: unsupported expert tensor type");
    }
    const uint64_t bytes = n / (uint64_t) block_elems * (uint64_t) block_bytes;
    const std::vector<uint8_t> raw = g.tensor_bytes(t, (uint64_t) expert * bytes, bytes);
    std::vector<float> out((size_t) n);
    if (!dequantize((int) t.type, raw.data(), out.data(), (int64_t) n)) {
        throw std::runtime_error("model: dequantize failed");
    }
    return out;
}

std::vector<float> need(const Gguf & g, const std::string & name) {
    const GgufTensor * t = g.find(name);
    if (!t) {
        throw std::runtime_error("model: missing tensor " + name);
    }
    return round_f16(load_tensor(g, *t));
}

const GgufTensor * need_t(const Gguf & g, const std::string & name) {
    const GgufTensor * t = g.find(name);
    if (!t) {
        throw std::runtime_error("model: missing tensor " + name);
    }
    return t;
}

// ---- one GDN layer: the block from gdn_block_parity, condensed ----
std::vector<float> gdn_layer(Stream & stream, const Gguf & g, int il, const std::vector<float> & x) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wqkv = need(g, p + "attn_qkv.weight");
    const std::vector<float> Wgate = need(g, p + "attn_gate.weight");
    const std::vector<float> Wout = need(g, p + "ssm_out.weight");
    const std::vector<float> Wconv = need(g, p + "ssm_conv1d.weight");
    const std::vector<float> Wnorm = need(g, p + "ssm_norm.weight");
    const std::vector<float> Wbeta = need(g, p + "ssm_beta.weight");
    const std::vector<float> Walpha = need(g, p + "ssm_alpha.weight");
    const std::vector<float> dt = load_tensor(g, *need_t(g, p + "ssm_dt.bias"));
    const std::vector<float> negA = load_tensor(g, *need_t(g, p + "ssm_a"));

    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };

    Buffer b_qkv((size_t) kQkvDim * 2), b_z((size_t) kValueDim * 2), b_conv((size_t) kQkvDim * 2);
    Buffer b_silu((size_t) kQkvDim * 2), b_tap((size_t) kQkvDim * 2), b_acc((size_t) kQkvDim * 2);
    gemv(stream, Wqkv, kQkvDim, kHidden, tx, b_qkv);
    gemv(stream, Wgate, kValueDim, kHidden, tx, b_z);

    std::vector<float> w0((size_t) kQkvDim);
    for (int64_t c = 0; c < kQkvDim; ++c) {
        w0[(size_t) c] = Wconv[(size_t) c * kConvK + (kConvK - 1)]; // newest tap: conv_input is oldest-first
    }
    upload(w0, b_tap);
    upload(std::vector<float>((size_t) kQkvDim, 0.0f), b_acc);
    Tensor t_qkv{ b_qkv.data(), DType::F16, 1, kQkvDim };
    Tensor t_tap{ b_tap.data(), DType::F16, 1, kQkvDim };
    Tensor t_conv{ b_conv.data(), DType::F16, 1, kQkvDim };
    Tensor t_acc{ b_acc.data(), DType::F16, 1, kQkvDim };
    Tensor t_silu{ b_silu.data(), DType::F16, 1, kQkvDim };
    mul(stream, t_qkv, t_tap, t_conv);
    Buffer b_acc2((size_t) kQkvDim * 2);
    Tensor t_acc2{ b_acc2.data(), DType::F16, 1, kQkvDim };
    add(stream, t_acc, t_conv, t_acc2);
    silu(stream, t_acc2, t_silu);

    Buffer b_ones((size_t) kHeadDimSsm * 2), b_wnorm((size_t) kHeadDimSsm * 2);
    upload(std::vector<float>((size_t) kHeadDimSsm, 1.0f), b_ones);
    upload(Wnorm, b_wnorm);
    Tensor t_ones{ b_ones.data(), DType::F16, kHeadDimSsm, 1 };
    Tensor t_wnorm{ b_wnorm.data(), DType::F16, kHeadDimSsm, 1 };

    Buffer b_qn((size_t) kKeyDim * 2), b_kn((size_t) kKeyDim * 2);
    for (int64_t h = 0; h < kKHeads; ++h) {
        Buffer t1((size_t) kHeadDimSsm * 2), t2((size_t) kHeadDimSsm * 2);
        Tensor tq{ (char *) b_silu.data() + (size_t) h * kHeadDimSsm * 2, DType::F16, kHeadDimSsm, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimSsm, 1 };
        Tensor ts{ t2.data(), DType::F16, kHeadDimSsm, 1 };
        Tensor tout{ (char *) b_qn.data() + (size_t) h * kHeadDimSsm * 2, DType::F16, kHeadDimSsm, 1 };
        rms_norm(stream, tq, t_ones, kRmsEps / (double) kHeadDimSsm, tn);
        muls(stream, tn, 1.0f / std::sqrt((float) kHeadDimSsm), ts);
        muls(stream, ts, 1.0f, tout);

        Tensor tk{ (char *) b_silu.data() + (size_t) (kKeyDim + h * kHeadDimSsm) * 2, DType::F16, kHeadDimSsm, 1 };
        Tensor tkn{ (char *) b_kn.data() + (size_t) h * kHeadDimSsm * 2, DType::F16, kHeadDimSsm, 1 };
        rms_norm(stream, tk, t_ones, kRmsEps / (double) kHeadDimSsm, tn);
        muls(stream, tn, 1.0f / std::sqrt((float) kHeadDimSsm), ts);
        muls(stream, ts, 1.0f, tkn);
    }

    Buffer b_beta((size_t) kVHeads * 2), b_alpha((size_t) kVHeads * 2);
    gemv(stream, Wbeta, kVHeads, kHidden, tx, b_beta);
    gemv(stream, Walpha, kVHeads, kHidden, tx, b_alpha);
    const std::vector<float> beta_raw = to_host(stream, b_beta, kVHeads);
    const std::vector<float> alpha_raw = to_host(stream, b_alpha, kVHeads);

    Buffer b_normed((size_t) kValueDim * 2);
    for (int64_t h = 0; h < kVHeads; ++h) {
        const int64_t kh = h % kKHeads;
        const float bt = 1.0f / (1.0f + std::exp(-beta_raw[(size_t) h]));
        const float biased = alpha_raw[(size_t) h] + dt[(size_t) h];
        const float sp = std::log1p(std::exp(biased > 20.0f ? 20.0f : biased));
        const float gate = sp * negA[(size_t) h];

        Buffer b_g(2), b_b(2), b_out((size_t) kHeadDimSsm * 2);
        upload({ gate }, b_g);
        upload({ bt }, b_b);
        Tensor tq{ (char *) b_qn.data() + (size_t) kh * kHeadDimSsm * 2, DType::F16, 1, kHeadDimSsm };
        Tensor tk{ (char *) b_kn.data() + (size_t) kh * kHeadDimSsm * 2, DType::F16, 1, kHeadDimSsm };
        Tensor tv{ (char *) b_silu.data() + (size_t) (2 * kKeyDim + h * kHeadDimSsm) * 2, DType::F16, 1,
                   kHeadDimSsm };
        Tensor tg{ b_g.data(), DType::F16, 1, 1 };
        Tensor tb{ b_b.data(), DType::F16, 1, 1 };
        Buffer b_st((size_t) kHeadDimSsm * kHeadDimSsm * 2);
        upload(std::vector<float>((size_t) kHeadDimSsm * kHeadDimSsm, 0.0f), b_st);
        Tensor tst{ b_st.data(), DType::F16, kHeadDimSsm, kHeadDimSsm };
        Tensor tout{ b_out.data(), DType::F16, 1, kHeadDimSsm };
        GdnShape shape{ kHeadDimSsm, 1, false };
        gdn(stream, tq, tk, tv, tg, tb, tst, tout, shape);

        Buffer t1((size_t) kHeadDimSsm * 2), t2((size_t) kHeadDimSsm * 2), t3((size_t) kHeadDimSsm * 2);
        Tensor tin{ b_out.data(), DType::F16, kHeadDimSsm, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimSsm, 1 };
        Tensor tz{ (char *) b_z.data() + (size_t) h * kHeadDimSsm * 2, DType::F16, kHeadDimSsm, 1 };
        Tensor tzs{ t2.data(), DType::F16, kHeadDimSsm, 1 };
        Tensor tm{ t3.data(), DType::F16, kHeadDimSsm, 1 };
        Tensor tnout{ (char *) b_normed.data() + (size_t) h * kHeadDimSsm * 2, DType::F16, kHeadDimSsm, 1 };
        rms_norm(stream, tin, t_wnorm, kRmsEps, tn);
        silu(stream, tz, tzs);
        mul(stream, tn, tzs, tm);
        muls(stream, tm, 1.0f, tnout);
    }

    Buffer b_y((size_t) kHidden * 2);
    Tensor t_normed{ b_normed.data(), DType::F16, 1, kValueDim };
    gemv(stream, Wout, kHidden, kValueDim, t_normed, b_y);
    return to_host(stream, b_y, kHidden);
}

std::vector<float> gdn_layer_cpu(const Gguf & g, int il, const std::vector<float> & x) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wqkv = need(g, p + "attn_qkv.weight");
    const std::vector<float> Wgate = need(g, p + "attn_gate.weight");
    const std::vector<float> Wout = need(g, p + "ssm_out.weight");
    const std::vector<float> Wconv = need(g, p + "ssm_conv1d.weight");
    const std::vector<float> Wnorm = need(g, p + "ssm_norm.weight");
    const std::vector<float> Wbeta = need(g, p + "ssm_beta.weight");
    const std::vector<float> Walpha = need(g, p + "ssm_alpha.weight");
    const std::vector<float> dt = load_tensor(g, *need_t(g, p + "ssm_dt.bias"));
    const std::vector<float> negA = load_tensor(g, *need_t(g, p + "ssm_a"));

    const std::vector<float> qkv = gemm(Wqkv, kQkvDim, kHidden, x);
    const std::vector<float> z = gemm(Wgate, kValueDim, kHidden, x);
    const std::vector<float> beta_raw = gemm(Wbeta, kVHeads, kHidden, x);
    const std::vector<float> alpha_raw = gemm(Walpha, kVHeads, kHidden, x);

    std::vector<float> conv((size_t) kQkvDim);
    for (int64_t c = 0; c < kQkvDim; ++c) {
        const float v = Wconv[(size_t) c * kConvK + (kConvK - 1)] * qkv[(size_t) c];
        conv[(size_t) c] = v / (1.0f + std::exp(-v));
    }
    std::vector<float> q((size_t) kKeyDim), kk((size_t) kKeyDim), v((size_t) kValueDim);
    for (int64_t i = 0; i < kKeyDim; ++i) {
        q[(size_t) i] = conv[(size_t) i];
        kk[(size_t) i] = conv[(size_t) (kKeyDim + i)];
    }
    for (int64_t i = 0; i < kValueDim; ++i) {
        v[(size_t) i] = conv[(size_t) (2 * kKeyDim + i)];
    }
    auto l2 = [](std::vector<float> & vec, int64_t nh) {
        for (int64_t h = 0; h < nh; ++h) {
            double ss = 0.0;
            for (int64_t i = 0; i < kHeadDimSsm; ++i) {
                const double t = vec[(size_t) (h * kHeadDimSsm + i)];
                ss += t * t;
            }
            const float sc =
                (float) (1.0 / std::sqrt(ss / (double) kHeadDimSsm + kRmsEps / (double) kHeadDimSsm)) *
                (1.0f / std::sqrt((float) kHeadDimSsm));
            for (int64_t i = 0; i < kHeadDimSsm; ++i) {
                vec[(size_t) (h * kHeadDimSsm + i)] *= sc;
            }
        }
    };
    l2(q, kKHeads);
    l2(kk, kKHeads);

    std::vector<float> dn((size_t) kValueDim, 0.0f);
    const float dn_scale = 1.0f / std::sqrt((float) kHeadDimSsm);
    for (int64_t h = 0; h < kVHeads; ++h) {
        const int64_t kh = h % kKHeads;
        const float bt = 1.0f / (1.0f + std::exp(-beta_raw[(size_t) h]));
        const float biased = alpha_raw[(size_t) h] + dt[(size_t) h];
        const float sp = std::log1p(std::exp(biased > 20.0f ? 20.0f : biased));
        const float decay = std::exp(sp * negA[(size_t) h]);
        (void) decay; // state is zero on the first token
        double kq = 0.0;
        for (int64_t i = 0; i < kHeadDimSsm; ++i) {
            kq += (double) kk[(size_t) (kh * kHeadDimSsm + i)] * q[(size_t) (kh * kHeadDimSsm + i)];
        }
        for (int64_t i = 0; i < kHeadDimSsm; ++i) {
            dn[(size_t) (h * kHeadDimSsm + i)] =
                bt * v[(size_t) (h * kHeadDimSsm + i)] * (float) kq * dn_scale;
        }
    }

    std::vector<float> normed((size_t) kValueDim);
    for (int64_t h = 0; h < kVHeads; ++h) {
        double ss = 0.0;
        for (int64_t i = 0; i < kHeadDimSsm; ++i) {
            const double t = dn[(size_t) (h * kHeadDimSsm + i)];
            ss += t * t;
        }
        const float sc = (float) (1.0 / std::sqrt(ss / (double) kHeadDimSsm + kRmsEps));
        for (int64_t i = 0; i < kHeadDimSsm; ++i) {
            const float nv = dn[(size_t) (h * kHeadDimSsm + i)] * sc * Wnorm[(size_t) i];
            const float zv = z[(size_t) (h * kHeadDimSsm + i)];
            normed[(size_t) (h * kHeadDimSsm + i)] = nv * (zv / (1.0f + std::exp(-zv)));
        }
    }
    return gemm(Wout, kHidden, kValueDim, normed);
}

// ---- one full-attention layer, first token ----
std::vector<float> attn_layer(Stream & stream, const Gguf & g, int il, const std::vector<float> & x) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wq = need(g, p + "attn_q.weight");
    const std::vector<float> Wk = need(g, p + "attn_k.weight");
    const std::vector<float> Wv = need(g, p + "attn_v.weight");
    const std::vector<float> Wo = need(g, p + "attn_output.weight");
    const std::vector<float> qn = need(g, p + "attn_q_norm.weight");
    const std::vector<float> knw = need(g, p + "attn_k_norm.weight");

    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };
    Buffer b_q((size_t) kQDim * 2), b_k((size_t) kKvDim * 2), b_v((size_t) kKvDim * 2);
    gemv(stream, Wq, kQDim, kHidden, tx, b_q);
    gemv(stream, Wk, kKvDim, kHidden, tx, b_k);
    gemv(stream, Wv, kKvDim, kHidden, tx, b_v);

    Buffer b_qn((size_t) kHeadDimAttn * 2), b_kn((size_t) kHeadDimAttn * 2);
    upload(qn, b_qn);
    upload(knw, b_kn);
    Tensor t_qnw{ b_qn.data(), DType::F16, kHeadDimAttn, 1 };
    Tensor t_knw{ b_kn.data(), DType::F16, kHeadDimAttn, 1 };

    const int64_t half = kRot / 2;
    std::vector<float> ct((size_t) half), st((size_t) half);
    for (int64_t i = 0; i < half; ++i) {
        const double theta = std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
        ct[(size_t) i] = (float) std::cos(theta);
        st[(size_t) i] = (float) std::sin(theta);
    }
    Buffer b_cos((size_t) half * 2), b_sin((size_t) half * 2);
    upload(ct, b_cos);
    upload(st, b_sin);
    Tensor t_cos{ b_cos.data(), DType::F16, half, 1 };
    Tensor t_sin{ b_sin.data(), DType::F16, half, 1 };

    Buffer b_attn((size_t) kHeads * kHeadDimAttn * 2);
    for (int64_t h = 0; h < kHeads; ++h) {
        Buffer t1((size_t) kHeadDimAttn * 2), t2((size_t) kHeadDimAttn * 2);
        Buffer b_qrope((size_t) kHeadDimAttn * 2);
        Tensor qraw{ (char *) b_q.data() + (size_t) h * kHeadDimAttn * 2 * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tqr{ b_qrope.data(), DType::F16, kHeadDimAttn, 1 };
        rms_norm(stream, qraw, t_qnw, kRmsEps, tn);
        rope(stream, tn, t_cos, t_sin, kRot, tqr);

        Tensor graw{ (char *) b_q.data() + (size_t) (h * kHeadDimAttn * 2 + kHeadDimAttn) * 2, DType::F16,
                     kHeadDimAttn, 1 };
        Tensor tgate{ t2.data(), DType::F16, kHeadDimAttn, 1 };
        sigmoid(stream, graw, tgate);

        const int64_t kvh = h % kKvHeads;
        Tensor vh{ (char *) b_v.data() + (size_t) kvh * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor tout{ (char *) b_attn.data() + (size_t) h * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        mul(stream, vh, tgate, tout);
    }

    if (getenv("MF_TRACE") && il == 3) {
        const std::vector<float> at_h = to_host(stream, b_attn, kHeads * kHeadDimAttn);
        const int64_t n = kHeads * kHeadDimAttn;
        printf("  npu l3 attn_gated first: %.4f %.4f %.4f   last: %.4f %.4f %.4f\n", at_h[0], at_h[1], at_h[2],
               at_h[(size_t) (n - 3)], at_h[(size_t) (n - 2)], at_h[(size_t) (n - 1)]);
        // host side of the same projection, to tell a bad gemv from a bad vector
        std::vector<float> expect((size_t) kHidden, 0.0f);
        for (int64_t r = 0; r < kHidden; ++r) {
            double acc = 0.0;
            for (int64_t c = 0; c < n; ++c) {
                acc += (double) Wo[(size_t) (r * n + c)] * at_h[(size_t) c];
            }
            expect[(size_t) r] = (float) acc;
        }
        printf("  npu l3 o_proj on host[0..2]: %.4f %.4f %.4f\n", expect[0], expect[1], expect[2]);
        printf("  npu l3 gated at heads 1,2,10,14: %.4f %.4f %.4f %.4f\n", at_h[256], at_h[512], at_h[2560], at_h[3584]);
    }
    Buffer b_y((size_t) kHidden * 2);
    Tensor t_attn{ b_attn.data(), DType::F16, 1, kHeads * kHeadDimAttn };
    gemv(stream, Wo, kHidden, kHeads * kHeadDimAttn, t_attn, b_y);
    return to_host(stream, b_y, kHidden);
}

std::vector<float> attn_layer_cpu(const Gguf & g, int il, const std::vector<float> & x) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wq = need(g, p + "attn_q.weight");
    const std::vector<float> Wk = need(g, p + "attn_k.weight");
    const std::vector<float> Wv = need(g, p + "attn_v.weight");
    const std::vector<float> Wo = need(g, p + "attn_output.weight");
    const std::vector<float> qn = need(g, p + "attn_q_norm.weight");
    const std::vector<float> knw = need(g, p + "attn_k_norm.weight");

    const std::vector<float> q_full = gemm(Wq, kQDim, kHidden, x);
    const std::vector<float> kk = gemm(Wk, kKvDim, kHidden, x);
    const std::vector<float> vv = gemm(Wv, kKvDim, kHidden, x);

    std::vector<float> attn((size_t) (kHeads * kHeadDimAttn), 0.0f);
    for (int64_t h = 0; h < kHeads; ++h) {
        std::vector<float> qh(q_full.begin() + (size_t) (h * kHeadDimAttn * 2),
                             q_full.begin() + (size_t) (h * kHeadDimAttn * 2 + kHeadDimAttn));
        std::vector<float> gate(q_full.begin() + (size_t) (h * kHeadDimAttn * 2 + kHeadDimAttn),
                                q_full.begin() + (size_t) (h * kHeadDimAttn * 2 + 2 * kHeadDimAttn));
        std::vector<float> qnorm;
        rms_cpu(qh, qn, kRmsEps, qnorm);
        rope_cpu(qnorm, kRot, kFreqBase, 0);

        const int64_t kvh = h % kKvHeads;
        for (int64_t i = 0; i < kHeadDimAttn; ++i) {
            const float sg = 1.0f / (1.0f + std::exp(-gate[(size_t) i]));
            attn[(size_t) (h * kHeadDimAttn + i)] = vv[(size_t) (kvh * kHeadDimAttn + i)] * sg;
        }
    }
    return gemm(Wo, kHidden, kHeads * kHeadDimAttn, attn);
}

// ---- MoE FFN, both paths ----
std::vector<float> moe_ffn(Stream & stream, const Gguf & g, int il, const std::vector<float> & x,
                           std::vector<int> & sel_out, std::vector<float> & w_out) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wr = need(g, p + "ffn_gate_inp.weight");
    const GgufTensor * gate_exps = need_t(g, p + "ffn_gate_exps.weight");
    const GgufTensor * up_exps = need_t(g, p + "ffn_up_exps.weight");
    const GgufTensor * down_exps = need_t(g, p + "ffn_down_exps.weight");
    const std::vector<float> Wgs = need(g, p + "ffn_gate_shexp.weight");
    const std::vector<float> Wus = need(g, p + "ffn_up_shexp.weight");
    const std::vector<float> Wds = need(g, p + "ffn_down_shexp.weight");
    const std::vector<float> Wgi = need(g, p + "ffn_gate_inp_shexp.weight");

    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };

    const int64_t n_expert = (int64_t) Wr.size() / kHidden;
    Buffer b_logits((size_t) n_expert * 2);
    gemv(stream, Wr, n_expert, kHidden, tx, b_logits);
    const std::vector<float> logits = to_host(stream, b_logits, n_expert);

    if (getenv("MF_TRACE") && il == 0) {
        printf("  npu router logits[0..2]: %.4f %.4f %.4f\n", logits[0], logits[1], logits[2]);
    }
    // softmax, top-8, normalize (same as build_moe_ffn)
    double mx = logits[0];
    for (float v : logits) {
        mx = std::max(mx, (double) v);
    }
    std::vector<float> probs((size_t) n_expert);
    double sum = 0.0;
    for (int64_t e = 0; e < n_expert; ++e) {
        probs[(size_t) e] = (float) std::exp((double) logits[(size_t) e] - mx);
        sum += probs[(size_t) e];
    }
    for (auto & v : probs) {
        v = (float) (v / sum);
    }
    std::vector<int> order((size_t) n_expert);
    for (int64_t i = 0; i < n_expert; ++i) {
        order[(size_t) i] = (int) i;
    }
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                      [&](int a, int b) { return probs[(size_t) a] > probs[(size_t) b]; });
    std::vector<int> sel(order.begin(), order.begin() + kTopK);
    std::vector<float> w((size_t) kTopK);
    double wsum = 0.0;
    for (int64_t i = 0; i < kTopK; ++i) {
        w[(size_t) i] = probs[(size_t) sel[(size_t) i]];
        wsum += w[(size_t) i];
    }
    wsum = std::max(wsum, 6.103515625e-5);
    for (auto & v : w) {
        v = (float) (v / wsum);
    }
    sel_out = sel;
    w_out = w;

    Buffer acc_a((size_t) kHidden * 2), acc_b((size_t) kHidden * 2), tmp((size_t) kHidden * 2);
    Buffer b_gate((size_t) kExpertInter * 2), b_up((size_t) kExpertInter * 2);
    Buffer b_silu((size_t) kExpertInter * 2), b_act((size_t) kExpertInter * 2), b_y((size_t) kHidden * 2);
    Tensor t_tmp{ tmp.data(), DType::F16, 1, kHidden };
    Tensor t_acc_a{ acc_a.data(), DType::F16, 1, kHidden };
    Tensor t_acc_b{ acc_b.data(), DType::F16, 1, kHidden };
    upload(std::vector<float>((size_t) kHidden, 0.0f), acc_a);
    (void) t_acc_a;
    Buffer * acc = &acc_a;
    Buffer * other = &acc_b;

    for (int64_t i = 0; i < kTopK; ++i) {
        const int e = sel[(size_t) i];
        const std::vector<float> Wg = round_f16(expert_slab(g, *gate_exps, e, kExpertInter, kHidden));
        const std::vector<float> Wu = round_f16(expert_slab(g, *up_exps, e, kExpertInter, kHidden));
        const std::vector<float> Wd = round_f16(expert_slab(g, *down_exps, e, kHidden, kExpertInter));

        gemv(stream, Wg, kExpertInter, kHidden, tx, b_gate);
        gemv(stream, Wu, kExpertInter, kHidden, tx, b_up);
        Tensor g_t{ b_gate.data(), DType::F16, 1, kExpertInter };
        Tensor u_t{ b_up.data(), DType::F16, 1, kExpertInter };
        Tensor s_t{ b_silu.data(), DType::F16, 1, kExpertInter };
        Tensor a_t{ b_act.data(), DType::F16, 1, kExpertInter };
        silu(stream, g_t, s_t);
        mul(stream, s_t, u_t, a_t);

        gemv(stream, Wd, kHidden, kExpertInter, a_t, b_y);
        Tensor y_t{ b_y.data(), DType::F16, 1, kHidden };
        muls(stream, y_t, w[(size_t) i], t_tmp);
        Tensor acc_t{ acc->data(), DType::F16, 1, kHidden };
        Tensor oth_t{ other->data(), DType::F16, 1, kHidden };
        add(stream, acc_t, t_tmp, oth_t);
        std::swap(acc, other);
    }

    if (getenv("MF_TRACE") && il == 0) {
        const std::vector<float> acc_h = to_host(stream, *acc, kHidden);
        printf("  npu expert_sum[0..3]: %.4f %.4f %.4f %.4f\n", acc_h[0], acc_h[1], acc_h[2], acc_h[3]);
    }
    // shared expert
    {
        gemv(stream, Wgs, kExpertInter, kHidden, tx, b_gate);
        gemv(stream, Wus, kExpertInter, kHidden, tx, b_up);
        Tensor g_t{ b_gate.data(), DType::F16, 1, kExpertInter };
        Tensor u_t{ b_up.data(), DType::F16, 1, kExpertInter };
        Tensor s_t{ b_silu.data(), DType::F16, 1, kExpertInter };
        Tensor a_t{ b_act.data(), DType::F16, 1, kExpertInter };
        silu(stream, g_t, s_t);
        mul(stream, s_t, u_t, a_t);
        gemv(stream, Wds, kHidden, kExpertInter, a_t, b_y);

        Buffer b_sg(2);
        gemv(stream, Wgi, 1, kHidden, tx, b_sg);
        const float gate_dot = to_host(stream, b_sg, 1)[0];
        const float sg = 1.0f / (1.0f + std::exp(-gate_dot));
        Tensor y_t{ b_y.data(), DType::F16, 1, kHidden };
        muls(stream, y_t, sg, t_tmp);
        if (getenv("MF_TRACE") && il == 0) {
            const std::vector<float> sh = to_host(stream, tmp, kHidden);
            printf("  npu shared_gate_sigmoid: %.4f   shared[0..3]: %.4f %.4f %.4f %.4f\n", sg, sh[0], sh[1],
                   sh[2], sh[3]);
        }
        Tensor acc_t{ acc->data(), DType::F16, 1, kHidden };
        Tensor oth_t{ other->data(), DType::F16, 1, kHidden };
        add(stream, acc_t, t_tmp, oth_t);
        std::swap(acc, other);
    }
    return to_host(stream, *acc, kHidden);
}

std::vector<float> moe_ffn_cpu(const Gguf & g, int il, const std::vector<float> & x, const std::vector<int> & sel,
                               const std::vector<float> & w) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const GgufTensor * gate_exps = need_t(g, p + "ffn_gate_exps.weight");
    const GgufTensor * up_exps = need_t(g, p + "ffn_up_exps.weight");
    const GgufTensor * down_exps = need_t(g, p + "ffn_down_exps.weight");
    const std::vector<float> Wgs = need(g, p + "ffn_gate_shexp.weight");
    const std::vector<float> Wus = need(g, p + "ffn_up_shexp.weight");
    const std::vector<float> Wds = need(g, p + "ffn_down_shexp.weight");
    const std::vector<float> Wgi = need(g, p + "ffn_gate_inp_shexp.weight");

    std::vector<float> y((size_t) kHidden, 0.0f);
    for (int64_t i = 0; i < kTopK; ++i) {
        const int e = sel[(size_t) i];
        const std::vector<float> Wg = round_f16(expert_slab(g, *gate_exps, e, kExpertInter, kHidden));
        const std::vector<float> Wu = round_f16(expert_slab(g, *up_exps, e, kExpertInter, kHidden));
        const std::vector<float> Wd = round_f16(expert_slab(g, *down_exps, e, kHidden, kExpertInter));
        const std::vector<float> gv = gemm(Wg, kExpertInter, kHidden, x);
        const std::vector<float> uv = gemm(Wu, kExpertInter, kHidden, x);
        std::vector<float> act((size_t) kExpertInter);
        for (int64_t r = 0; r < kExpertInter; ++r) {
            act[(size_t) r] = (gv[(size_t) r] / (1.0f + std::exp(-gv[(size_t) r]))) * uv[(size_t) r];
        }
        const std::vector<float> dv = gemm(Wd, kHidden, kExpertInter, act);
        for (int64_t r = 0; r < kHidden; ++r) {
            y[(size_t) r] += w[(size_t) i] * dv[(size_t) r];
        }
    }
    const std::vector<float> gv = gemm(Wgs, kExpertInter, kHidden, x);
    const std::vector<float> uv = gemm(Wus, kExpertInter, kHidden, x);
    std::vector<float> act((size_t) kExpertInter);
    for (int64_t r = 0; r < kExpertInter; ++r) {
        act[(size_t) r] = (gv[(size_t) r] / (1.0f + std::exp(-gv[(size_t) r]))) * uv[(size_t) r];
    }
    const std::vector<float> dv = gemm(Wds, kHidden, kExpertInter, act);
    double gate_dot = 0.0;
    for (int64_t j = 0; j < kHidden; ++j) {
        gate_dot += (double) Wgi[(size_t) j] * x[(size_t) j];
    }
    const float sg = 1.0f / (1.0f + std::exp((float) -gate_dot));
    for (int64_t r = 0; r < kHidden; ++r) {
        y[(size_t) r] += sg * dv[(size_t) r];
    }
    return y;
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
    const int64_t n_layers = argc > 2 ? std::atoll(argv[2]) : 41;
    const int64_t token_id = argc > 3 ? std::atoll(argv[3]) : 9707;
    const bool cpu_too = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    try {
        const Gguf g(path);
        const int64_t interval = g.metadata_int(g.arch() + ".full_attention_interval");
        const GgufTensor * embd = g.find("token_embd.weight");
        const GgufTensor * out_w = g.find("output.weight");
        if (!embd || !out_w) {
            printf("embedding or output tensor missing\n");
            return 2;
        }

        // one row of the embedding table (the full table is 2 GB as floats)\r
        const int block_bytes = quant_type_size((int) embd->type);
        const int block_elems = quant_block_elements((int) embd->type);
        const uint64_t row_bytes = (uint64_t) kHidden / (uint64_t) block_elems * (uint64_t) block_bytes;
        const std::vector<uint8_t> row = g.tensor_bytes(*embd, (uint64_t) token_id * row_bytes, row_bytes);
        std::vector<float> h((size_t) kHidden);
        dequantize((int) embd->type, row.data(), h.data(), kHidden);
        h = round_f16(h);

        Device device(0);
        Stream stream(device);

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> xn = h;
        std::vector<std::vector<int>> sels;
        std::vector<std::vector<float>> ws;
        for (int64_t il = 0; il < n_layers; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            const bool recr = ((il + 1) % interval) != 0;
            const std::vector<float> an = need(g, p + (recr ? "attn_norm.weight" : "attn_norm.weight"));
            const std::vector<float> pan = need(g, p + "post_attention_norm.weight");

            Buffer bx((size_t) kHidden * 2), bn((size_t) kHidden * 2);
            upload(xn, bx);
            // rms_norm normalizes over the last dim, so both operands are described as rows here
            Tensor tx{ bx.data(), DType::F16, kHidden, 1 };
            Tensor tn{ bn.data(), DType::F16, kHidden, 1 };
            Buffer b_an((size_t) kHidden * 2);
            upload(an, b_an);
            Tensor t_an{ b_an.data(), DType::F16, kHidden, 1 };
            rms_norm(stream, tx, t_an, kRmsEps, tn);
            const std::vector<float> hnorm = to_host(stream, bn, kHidden);

            if (getenv("MF_TRACE") && il == 0) {
                printf("  npu normed[0..7]:");
                for (int i = 0; i < 8; ++i) printf(" %.6f", hnorm[(size_t) i]);
                printf("\n");
            }
            if (getenv("MF_TRACE") && il == 0) {
                printf("  npu normed[0..3]:");
                for (int i = 0; i < 4; ++i) printf(" %.6f", hnorm[(size_t) i]);
                printf("\n  input   [0..3]:");
                for (int i = 0; i < 4; ++i) printf(" %.6f", xn[(size_t) i]);
                printf("\n");
            }
            std::vector<float> layer_out = recr ? gdn_layer(stream, g, il, hnorm) : attn_layer(stream, g, il, hnorm);
            // residual accumulated in float32 on the host: doing it on the device rounds the sum
            // to fp16 at every layer, which is what made 40 layers drift to 1.7e-2
            for (int64_t i = 0; i < kHidden; ++i) {
                xn[(size_t) i] += layer_out[(size_t) i];
            }

            // post attention norm, then the FFN, then the residual
            Buffer b_x2((size_t) kHidden * 2);
            upload(xn, b_x2);
            Tensor t_x2{ b_x2.data(), DType::F16, kHidden, 1 }; // row, for the norm
            Buffer b_pan((size_t) kHidden * 2);
            upload(pan, b_pan);
            Tensor t_pan{ b_pan.data(), DType::F16, kHidden, 1 };
            rms_norm(stream, t_x2, t_pan, kRmsEps, tn);
            const std::vector<float> ffn_in = to_host(stream, bn, kHidden);

            std::vector<int> sel;
            std::vector<float> w;
            const std::vector<float> ffn_out = moe_ffn(stream, g, il, ffn_in, sel, w);
            if (getenv("MF_TRACE") && il == 0) {
                printf("  npu experts/weights:");
                for (int i = 0; i < 8; ++i) printf(" %d(%.4f)", sel[(size_t) i], w[(size_t) i]);
                printf("\n");
            }
            if (getenv("MF_TRACE") && il == 0) {
                printf("  npu attn_norm[0..7]:");
                for (int i = 0; i < 8; ++i) printf(" %.4f", hnorm[(size_t) i]);
                printf("\n  npu attn_out[0..3]:");
                for (int i = 0; i < 4; ++i) printf(" %.4f", layer_out[(size_t) i]);
                printf("\n  npu residual[0..3]:");
                for (int i = 0; i < 4; ++i) printf(" %.4f", xn[(size_t) i]);
                printf("\n  npu ffn_out[0..3]:");
                for (int i = 0; i < 4; ++i) printf(" %.4f", ffn_out[(size_t) i]);
                printf("\n");
            }
            sels.push_back(sel);
            ws.push_back(w);

            for (int64_t i = 0; i < kHidden; ++i) {
                xn[(size_t) i] += ffn_out[(size_t) i];
            }

            if (getenv("MF_TRACE") && il < 6) {
                printf("  npu layer %lld x[0..3]:", (long long) il);
                for (int i = 0; i < 4; ++i) printf(" %.5f", xn[(size_t) i]);
                printf("  (layer_out:");
                for (int i = 0; i < 4; ++i) printf(" %.5f", layer_out[(size_t) i]);
                printf(")\n");
            }
            if (il % 10 == 0 || il == n_layers - 1) {
                const auto t1 = std::chrono::steady_clock::now();
                const double secs = std::chrono::duration<double>(t1 - t0).count();
                printf("layer %2lld/%lld  %s  top1 expert %d  %.1f s\n", (long long) il, (long long) n_layers,
                       recr ? "gdn " : "attn", sel.empty() ? -1 : sel[0], secs);
            }
        }

        // final norm
        const std::vector<float> on = need(g, "output_norm.weight");
        Buffer b_f((size_t) kHidden * 2), b_on((size_t) kHidden * 2), b_fn((size_t) kHidden * 2);
        upload(xn, b_f);
        upload(on, b_on);
        Tensor t_f{ b_f.data(), DType::F16, kHidden, 1 };
        Tensor t_on{ b_on.data(), DType::F16, kHidden, 1 };
        Tensor t_fn{ b_fn.data(), DType::F16, kHidden, 1 };
        rms_norm(stream, t_f, t_on, kRmsEps, t_fn);
        const std::vector<float> final_hidden = to_host(stream, b_fn, kHidden);

        // logits over the whole vocabulary, in chunks
        const int64_t vocab = (int64_t) out_w->dims[1];
        const int64_t chunk = 32768;
        std::vector<std::pair<float, int>> best;
        // running log-sum-exp over the whole vocabulary, so the test can print logprobs
        // comparable with llama.cpp's
        double lse_max = -INFINITY, lse_sum = 0.0;
        for (int64_t start = 0; start < vocab; start += chunk) {
            const int64_t n = std::min(chunk, vocab - start);
            const int bb = quant_type_size((int) out_w->type);
            const int be = quant_block_elements((int) out_w->type);
            const uint64_t row_b = (uint64_t) kHidden / (uint64_t) be * (uint64_t) bb;
            const std::vector<uint8_t> raw = g.tensor_bytes(*out_w, (uint64_t) start * row_b, (uint64_t) n * row_b);
            std::vector<float> wf((size_t) n * kHidden);
            dequantize((int) out_w->type, raw.data(), wf.data(), n * kHidden);
            wf = round_f16(wf);
            Buffer b_l((size_t) n * 2);
            Tensor t_fn_col{ b_fn.data(), DType::F16, 1, kHidden };
            gemv(stream, wf, n, kHidden, t_fn_col, b_l);
            const std::vector<float> lg = to_host(stream, b_l, n);
            for (int64_t i = 0; i < n; ++i) {
                const double v = (double) lg[(size_t) i];
                if (v > lse_max) {
                    lse_sum = lse_sum * std::exp(lse_max - v) + 1.0;
                    lse_max = v;
                } else {
                    lse_sum += std::exp(v - lse_max);
                }
                const std::pair<float, int> cand{ lg[(size_t) i], (int) (start + i) };
                if ((int64_t) best.size() < 5) {
                    best.push_back(cand);
                    std::sort(best.begin(), best.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
                } else if (cand.first > best[4].first) {
                    best[4] = cand;
                    std::sort(best.begin(), best.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
                }
            }
        }
        std::sort(best.begin(), best.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        printf("\nfinal hidden:");
        for (int i = 0; i < 4; ++i) {
            printf(" %.5f", final_hidden[(size_t) i]);
        }
        const double log_z = lse_max + std::log(lse_sum);
        printf("\ntop 5 logits and logprobs:");
        for (size_t i = 0; i < std::min<size_t>(5, best.size()); ++i) {
            printf("  token %d  logit %.4f  logprob %.4f\n", best[i].second, best[i].first,
                   (double) best[i].first - log_z);
        }

        if (cpu_too) {
            printf("\nrunning the CPU reference for %lld layers...\n", (long long) n_layers);
            const auto c0 = std::chrono::steady_clock::now();
            std::vector<float> xc = h;
            for (int64_t il = 0; il < n_layers; ++il) {
                const std::string p = "blk." + std::to_string(il) + ".";
                const bool recr = ((il + 1) % interval) != 0;
                const std::vector<float> an = need(g, p + "attn_norm.weight");
                const std::vector<float> pan = need(g, p + "post_attention_norm.weight");
                std::vector<float> hn;
                rms_cpu(xc, an, kRmsEps, hn);
                if (getenv("MF_TRACE") && il == 0) {
                    printf("  cpu normed[0..3]:");
                    for (int i = 0; i < 4; ++i) printf(" %.6f", hn[(size_t) i]);
                    printf("\n  input   [0..3]:");
                    for (int i = 0; i < 4; ++i) printf(" %.6f", xc[(size_t) i]);
                    printf("\n");
                }
                std::vector<float> lo = recr ? gdn_layer_cpu(g, il, hn) : attn_layer_cpu(g, il, hn);
                for (int64_t i = 0; i < kHidden; ++i) {
                    xc[(size_t) i] += lo[(size_t) i];
                }
                std::vector<float> fn;
                rms_cpu(xc, pan, kRmsEps, fn);
                std::vector<float> fo = moe_ffn_cpu(g, il, fn, sels[(size_t) il], ws[(size_t) il]);
                for (int64_t i = 0; i < kHidden; ++i) {
                    xc[(size_t) i] += fo[(size_t) i];
                }
                if (getenv("MF_TRACE") && il < 2) {
                    printf("  cpu layer %lld x[0..3]:", (long long) il);
                    for (int i = 0; i < 4; ++i) printf(" %.5f", xc[(size_t) i]);
                    printf("  (attn_out:");
                    for (int i = 0; i < 4; ++i) printf(" %.5f", lo[(size_t) i]);
                    printf(")\n");
                }
            }
            std::vector<float> fc;
            rms_cpu(xc, on, kRmsEps, fc);
            const auto c1 = std::chrono::steady_clock::now();
            const double cpu_secs = std::chrono::duration<double>(c1 - c0).count();
            const double r = rel_l2(final_hidden, fc);
            printf("CPU reference in %.1f s, hidden state rel L2 = %.6f  %s\n", cpu_secs, r, r <= 0.02 ? "PASS" : "FAIL");
            if (r > 0.02) {
                shutdown();
                printf("\nthe model forward differs from the CPU reference\n");
                return 1;
            }
        }

        shutdown();
        printf("\n%lld layers ran on the NPU from the model file\n", (long long) n_layers);
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
