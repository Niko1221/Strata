// cann/tests/model_seq.cpp - the whole model over a token sequence.
//
// model_forward.cpp proves one token matches llama.cpp. A real engine also has to carry state:
// the recurrent layers keep a short-conv history and a 128x128 matrix per value head, and the
// attention layers keep a KV cache. This program walks a prompt of T tokens through the 40 layers
// with those states live on the device, then prints the next-token distribution for the last
// position, which is what llama.cpp reports for the same prompt.
//
//   model_seq <model.gguf> <tokens comma separated>
#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
constexpr float kRmsEps = 1e-6f;
constexpr int64_t kQDim = kHeads * kHeadDimAttn * 2;
constexpr int64_t kKvDim = kKvHeads * kHeadDimAttn;
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

std::vector<float> expert_slab(const Gguf & g, const GgufTensor & t, int expert, int64_t rows, int64_t cols) {
    const uint64_t n = (uint64_t) rows * cols;
    const int block_bytes = quant_type_size((int) t.type);
    const int block_elems = quant_block_elements((int) t.type);
    const uint64_t bytes = n / (uint64_t) block_elems * (uint64_t) block_bytes;
    const std::vector<uint8_t> raw = g.tensor_bytes(t, (uint64_t) expert * bytes, bytes);
    std::vector<float> out((size_t) n);
    if (!dequantize((int) t.type, raw.data(), out.data(), (int64_t) n)) {
        throw std::runtime_error("model: dequantize failed");
    }
    return out;
}

// per-layer recurrent state, allocated once and updated in place
struct GdnState {
    std::vector<std::unique_ptr<Buffer>> conv; // kConvK-1 past qkv vectors, newest first
    std::vector<std::unique_ptr<Buffer>> ssm;  // kVHeads x (128 x 128)
};

struct AttnState {
    std::unique_ptr<Buffer> k; // T rows x kHeadDimAttn per kv head, laid out [tok][dim]
    std::unique_ptr<Buffer> v;
};

// ---- one GDN layer step ----
std::vector<float> gdn_step(Stream & stream, const Gguf & g, int il, const std::vector<float> & x, const std::vector<std::vector<float>> & taps, GdnState & st) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wqkv = need(g, p + "attn_qkv.weight");
    const std::vector<float> Wgate = need(g, p + "attn_gate.weight");
    const std::vector<float> Wout = need(g, p + "ssm_out.weight");
    const std::vector<float> Wnorm = need(g, p + "ssm_norm.weight");
    const std::vector<float> Wbeta = need(g, p + "ssm_beta.weight");
    const std::vector<float> Walpha = need(g, p + "ssm_alpha.weight");
    const std::vector<float> dt = load_tensor(g, *need_t(g, p + "ssm_dt.bias"));
    const std::vector<float> negA = load_tensor(g, *need_t(g, p + "ssm_a"));

    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };

    Buffer b_qkv((size_t) kQkvDim * 2), b_z((size_t) kValueDim * 2);
    gemv(stream, Wqkv, kQkvDim, kHidden, tx, b_qkv);
    gemv(stream, Wgate, kValueDim, kHidden, tx, b_z);
    Tensor t_qkv{ b_qkv.data(), DType::F16, 1, kQkvDim };

    // short conv over the history: tap kConvK-1 goes with the current token, kConvK-2 with t-1
    Buffer b_acc((size_t) kQkvDim * 2), b_tmp((size_t) kQkvDim * 2), b_tap((size_t) kQkvDim * 2);
    upload(std::vector<float>((size_t) kQkvDim, 0.0f), b_acc);
    Buffer b_conv((size_t) kQkvDim * 2);
    for (int64_t d = 0; d < kConvK; ++d) {
        const int64_t tap = kConvK - 1 - d;
        upload(taps[(size_t) tap], b_tap);
        Tensor t_tap{ b_tap.data(), DType::F16, 1, kQkvDim };
        Tensor t_acc_in{ b_acc.data(), DType::F16, 1, kQkvDim };
        Tensor t_tmp{ b_tmp.data(), DType::F16, 1, kQkvDim };
        Tensor t_conv{ b_conv.data(), DType::F16, 1, kQkvDim };
        if (d == 0) {
            mul(stream, t_qkv, t_tap, t_conv);
        } else {
            Tensor hist{ st.conv[(size_t) (d - 1)]->data(), DType::F16, 1, kQkvDim };
            mul(stream, hist, t_tap, t_conv);
        }
        add(stream, t_acc_in, t_conv, t_tmp);
        muls(stream, t_tmp, 1.0f, t_acc_in);
    }
    Buffer b_silu((size_t) kQkvDim * 2);
    Tensor t_acc{ b_acc.data(), DType::F16, 1, kQkvDim };
    Tensor t_silu{ b_silu.data(), DType::F16, 1, kQkvDim };
    silu(stream, t_acc, t_silu);

    // rotate the history: newest first
    for (int64_t d = kConvK - 2; d > 0; --d) {
        Tensor dst{ st.conv[(size_t) d]->data(), DType::F16, 1, kQkvDim };
        Tensor src{ st.conv[(size_t) (d - 1)]->data(), DType::F16, 1, kQkvDim };
        muls(stream, src, 1.0f, dst);
    }
    {
        Tensor dst{ st.conv[0]->data(), DType::F16, 1, kQkvDim };
        muls(stream, t_qkv, 1.0f, dst);
    }

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
        Tensor tst{ st.ssm[(size_t) h]->data(), DType::F16, kHeadDimSsm, kHeadDimSsm };
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

// ---- one attention layer step ----
std::vector<float> attn_step(Stream & stream, const Gguf & g, int il, const std::vector<float> & x, int64_t t,
                             int64_t T, AttnState & st) {
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
    std::vector<float> ct((size_t) half), stv((size_t) half);
    for (int64_t i = 0; i < half; ++i) {
        const double theta = (double) t * std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
        ct[(size_t) i] = (float) std::cos(theta);
        stv[(size_t) i] = (float) std::sin(theta);
    }
    Buffer b_cos((size_t) half * 2), b_sin((size_t) half * 2);
    upload(ct, b_cos);
    upload(stv, b_sin);
    Tensor t_cos{ b_cos.data(), DType::F16, half, 1 };
    Tensor t_sin{ b_sin.data(), DType::F16, half, 1 };

    // k norm + rope, then write into the cache row t
    for (int64_t kvh = 0; kvh < kKvHeads; ++kvh) {
        Buffer t1((size_t) kHeadDimAttn * 2), t2((size_t) kHeadDimAttn * 2);
        Buffer b_krope((size_t) kHeadDimAttn * 2);
        Tensor kraw{ (char *) b_k.data() + (size_t) kvh * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tss{ t2.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tkr{ b_krope.data(), DType::F16, kHeadDimAttn, 1 };
        rms_norm(stream, kraw, t_knw, kRmsEps, tn);
        rope(stream, tn, t_cos, t_sin, kRot, tkr);
        Tensor krow{ (char *) st.k->data() + (size_t) (kvh * T + t) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn,
                     1 };
        muls(stream, tkr, 1.0f, krow);
        Tensor vraw{ (char *) b_v.data() + (size_t) kvh * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor vrow{ (char *) st.v->data() + (size_t) (kvh * T + t) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn,
                     1 };
        muls(stream, vraw, 1.0f, vrow);
        (void) tss;
    }

    Buffer b_attn((size_t) kHeads * kHeadDimAttn * 2);
    const int64_t n = t + 1;
    for (int64_t h = 0; h < kHeads; ++h) {
        Buffer t1((size_t) kHeadDimAttn * 2), t2((size_t) kHeadDimAttn * 2);
        Buffer b_qrope((size_t) kHeadDimAttn * 2);
        Tensor qraw{ (char *) b_q.data() + (size_t) h * kHeadDimAttn * 2 * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tqr{ b_qrope.data(), DType::F16, kHeadDimAttn, 1 };
        rms_norm(stream, qraw, t_qnw, kRmsEps, tn);
        rope(stream, tn, t_cos, t_sin, kRot, tqr);

        const int64_t kvh = h / (kHeads / kKvHeads);
        Buffer b_scores((size_t) n * 2), b_w((size_t) n * 2), b_scaled((size_t) n * 2);
        Buffer b_mix((size_t) kHeadDimAttn * 2);
        Tensor kc{ (char *) st.k->data() + (size_t) (kvh * T) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, n };
        Tensor q_col{ b_qrope.data(), DType::F16, 1, kHeadDimAttn };
        Tensor scores{ b_scores.data(), DType::F16, 1, n };
        matmul(stream, kc, q_col, scores);

        Tensor sc_in{ b_scores.data(), DType::F16, n, 1 };
        Tensor sc_out{ b_scaled.data(), DType::F16, n, 1 };
        muls(stream, sc_in, 1.0f / std::sqrt((float) kHeadDimAttn), sc_out);
        Tensor srow{ b_scaled.data(), DType::F16, n, 1 };
        Tensor wrow{ b_w.data(), DType::F16, n, 1 };
        softmax(stream, srow, wrow);

        Tensor wmat{ b_w.data(), DType::F16, n, 1 };
        Tensor vc{ (char *) st.v->data() + (size_t) (kvh * T) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, n };
        Tensor mix{ b_mix.data(), DType::F16, kHeadDimAttn, 1 };
        matmul(stream, wmat, vc, mix);

        Tensor graw{ (char *) b_q.data() + (size_t) (h * kHeadDimAttn * 2 + kHeadDimAttn) * 2, DType::F16,
                     kHeadDimAttn, 1 };
        Tensor tgate{ t2.data(), DType::F16, kHeadDimAttn, 1 };
        sigmoid(stream, graw, tgate);
        Tensor tout{ (char *) b_attn.data() + (size_t) h * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        mul(stream, mix, tgate, tout);
    }

    Buffer b_y((size_t) kHidden * 2);
    Tensor t_attn{ b_attn.data(), DType::F16, 1, kHeads * kHeadDimAttn };
    gemv(stream, Wo, kHidden, kHeads * kHeadDimAttn, t_attn, b_y);
    return to_host(stream, b_y, kHidden);
}

std::vector<float> moe_ffn(Stream & stream, const Gguf & g, int il, const std::vector<float> & x) {
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

    double mx = logits[0];
    for (float v : logits) mx = std::max(mx, (double) v);
    std::vector<float> probs((size_t) n_expert);
    double sum = 0.0;
    for (int64_t e = 0; e < n_expert; ++e) {
        probs[(size_t) e] = (float) std::exp((double) logits[(size_t) e] - mx);
        sum += probs[(size_t) e];
    }
    for (auto & v : probs) v = (float) (v / sum);
    std::vector<int> order((size_t) n_expert);
    for (int64_t i = 0; i < n_expert; ++i) order[(size_t) i] = (int) i;
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
    for (auto & v : w) v = (float) (v / wsum);

    Buffer acc_a((size_t) kHidden * 2), acc_b((size_t) kHidden * 2), tmp((size_t) kHidden * 2);
    Buffer b_gate((size_t) kExpertInter * 2), b_up((size_t) kExpertInter * 2);
    Buffer b_silu((size_t) kExpertInter * 2), b_act((size_t) kExpertInter * 2), b_y((size_t) kHidden * 2);
    upload(std::vector<float>((size_t) kHidden, 0.0f), acc_a);
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
        Tensor t_tmp{ tmp.data(), DType::F16, 1, kHidden };
        muls(stream, y_t, w[(size_t) i], t_tmp);
        Tensor acc_t{ acc->data(), DType::F16, 1, kHidden };
        Tensor oth_t{ other->data(), DType::F16, 1, kHidden };
        add(stream, acc_t, t_tmp, oth_t);
        std::swap(acc, other);
    }
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
        Tensor t_tmp{ tmp.data(), DType::F16, 1, kHidden };
        muls(stream, y_t, sg, t_tmp);
        Tensor acc_t{ acc->data(), DType::F16, 1, kHidden };
        Tensor oth_t{ other->data(), DType::F16, 1, kHidden };
        add(stream, acc_t, t_tmp, oth_t);
        std::swap(acc, other);
    }
    return to_host(stream, *acc, kHidden);
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        printf("usage: model_seq <model.gguf> <token ids, comma separated>\n");
        return 2;
    }
    const std::string path = argv[1];
    std::vector<int64_t> toks;
    {
        const std::string list = argv[2];
        size_t start = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!item.empty()) toks.push_back(std::atoll(item.c_str()));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    if (toks.empty()) {
        printf("no tokens\n");
        return 2;
    }
    const int64_t T = (int64_t) toks.size();

    try {
        const Gguf g(path);
        const int64_t interval = g.metadata_int(g.arch() + ".full_attention_interval");
        const GgufTensor * embd = g.find("token_embd.weight");
        const GgufTensor * out_w = g.find("output.weight");
        if (!embd || !out_w) {
            printf("embedding or output tensor missing\n");
            return 2;
        }

        Device device(0);
        Stream stream(device);

        // conv taps, newest last, per layer
        std::vector<std::vector<std::vector<float>>> layer_taps;
        std::vector<std::unique_ptr<GdnState>> gdn_states;
        std::vector<std::unique_ptr<AttnState>> attn_states;
        layer_taps.resize(40);
        gdn_states.resize(40);
        attn_states.resize(40);
        for (int64_t il = 0; il < 40; ++il) {
            const bool recr = ((il + 1) % interval) != 0;
            if (recr) {
                const std::string p = "blk." + std::to_string(il) + ".";
                const std::vector<float> Wconv = need(g, p + "ssm_conv1d.weight");
                std::vector<std::vector<float>> taps((size_t) kConvK, std::vector<float>((size_t) kQkvDim));
                for (int64_t k = 0; k < kConvK; ++k) {
                    for (int64_t c = 0; c < kQkvDim; ++c) {
                        taps[(size_t) k][(size_t) c] = Wconv[(size_t) c * kConvK + k];
                    }
                }
                layer_taps[(size_t) il] = std::move(taps);
                auto st = std::make_unique<GdnState>();
                for (int64_t d = 0; d < kConvK - 1; ++d) {
                    st->conv.push_back(std::make_unique<Buffer>((size_t) kQkvDim * 2));
                    upload(std::vector<float>((size_t) kQkvDim, 0.0f), *st->conv.back());
                }
                for (int64_t h = 0; h < kVHeads; ++h) {
                    st->ssm.push_back(std::make_unique<Buffer>((size_t) kHeadDimSsm * kHeadDimSsm * 2));
                    upload(std::vector<float>((size_t) kHeadDimSsm * kHeadDimSsm, 0.0f), *st->ssm.back());
                }
                gdn_states[(size_t) il] = std::move(st);
            } else {
                auto st = std::make_unique<AttnState>();
                st->k = std::make_unique<Buffer>((size_t) kKvHeads * T * kHeadDimAttn * 2);
                st->v = std::make_unique<Buffer>((size_t) kKvHeads * T * kHeadDimAttn * 2);
                const std::vector<float> zeros((size_t) kKvHeads * T * kHeadDimAttn, 0.0f);
                upload(zeros, *st->k);
                upload(zeros, *st->v);
                attn_states[(size_t) il] = std::move(st);
            }
        }

        // embedding rows for the prompt
        const int bb = quant_type_size((int) embd->type);
        const int be = quant_block_elements((int) embd->type);
        const uint64_t row_bytes = (uint64_t) kHidden / (uint64_t) be * (uint64_t) bb;
        std::vector<std::vector<float>> xs((size_t) T);
        for (int64_t t = 0; t < T; ++t) {
            const std::vector<uint8_t> row = g.tensor_bytes(*embd, (uint64_t) toks[(size_t) t] * row_bytes, row_bytes);
            std::vector<float> h((size_t) kHidden);
            dequantize((int) embd->type, row.data(), h.data(), kHidden);
            xs[(size_t) t] = round_f16(h);
        }

        // residual stream, fp32
        std::vector<std::vector<float>> res((size_t) T, std::vector<float>((size_t) kHidden, 0.0f));
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t t = 0; t < T; ++t) {
            std::vector<float> xn = xs[(size_t) t];
            for (int64_t il = 0; il < 40; ++il) {
                const std::string p = "blk." + std::to_string(il) + ".";
                const bool recr = ((il + 1) % interval) != 0;
                const std::vector<float> an = need(g, p + "attn_norm.weight");
                const std::vector<float> pan = need(g, p + "post_attention_norm.weight");

                Buffer bx((size_t) kHidden * 2), bn((size_t) kHidden * 2), b_an((size_t) kHidden * 2);
                upload(xn, bx);
                upload(an, b_an);
                Tensor tx{ bx.data(), DType::F16, kHidden, 1 };
                Tensor tn{ bn.data(), DType::F16, kHidden, 1 };
                Tensor t_an{ b_an.data(), DType::F16, kHidden, 1 };
                rms_norm(stream, tx, t_an, kRmsEps, tn);
                const std::vector<float> hnorm = to_host(stream, bn, kHidden);

                std::vector<float> layer_out;
                if (recr) {
                    layer_out = gdn_step(stream, g, il, hnorm, layer_taps[(size_t) il], *gdn_states[(size_t) il]);
                } else {
                    layer_out = attn_step(stream, g, il, hnorm, t, T, *attn_states[(size_t) il]);
                }
                for (int64_t i = 0; i < kHidden; ++i) xn[(size_t) i] += layer_out[(size_t) i];

                Buffer bx2((size_t) kHidden * 2), b_pan((size_t) kHidden * 2);
                upload(xn, bx2);
                upload(pan, b_pan);
                Tensor t_x2{ bx2.data(), DType::F16, kHidden, 1 };
                Tensor t_pan{ b_pan.data(), DType::F16, kHidden, 1 };
                rms_norm(stream, t_x2, t_pan, kRmsEps, tn);
                const std::vector<float> ffn_in = to_host(stream, bn, kHidden);
                const std::vector<float> ffn_out = moe_ffn(stream, g, il, ffn_in);
                for (int64_t i = 0; i < kHidden; ++i) xn[(size_t) i] += ffn_out[(size_t) i];
            }
            res[(size_t) t] = xn;
            const auto t1 = std::chrono::steady_clock::now();
            printf("token %lld/%lld done  %.1f s\n", (long long) (t + 1), (long long) T,
                   std::chrono::duration<double>(t1 - t0).count());
        }

        // final norm and logits for the last position
        const std::vector<float> on = need(g, "output_norm.weight");
        Buffer b_f((size_t) kHidden * 2), b_on((size_t) kHidden * 2), b_fn((size_t) kHidden * 2);
        upload(res[(size_t) (T - 1)], b_f);
        upload(on, b_on);
        Tensor t_f{ b_f.data(), DType::F16, kHidden, 1 };
        Tensor t_on{ b_on.data(), DType::F16, kHidden, 1 };
        Tensor t_fn{ b_fn.data(), DType::F16, kHidden, 1 };
        rms_norm(stream, t_f, t_on, kRmsEps, t_fn);
        const std::vector<float> final_hidden = to_host(stream, b_fn, kHidden);
        (void) final_hidden;

        const int64_t vocab = (int64_t) out_w->dims[1];
        const int64_t chunk = 32768;
        std::vector<std::pair<float, int>> best;
        double lse_max = -INFINITY, lse_sum = 0.0;
        Tensor t_fn_col{ b_fn.data(), DType::F16, 1, kHidden };
        for (int64_t start = 0; start < vocab; start += chunk) {
            const int64_t n = std::min(chunk, vocab - start);
            const int bb2 = quant_type_size((int) out_w->type);
            const int be2 = quant_block_elements((int) out_w->type);
            const uint64_t row_b = (uint64_t) kHidden / (uint64_t) be2 * (uint64_t) bb2;
            const std::vector<uint8_t> raw = g.tensor_bytes(*out_w, (uint64_t) start * row_b, (uint64_t) n * row_b);
            std::vector<float> wf((size_t) n * kHidden);
            dequantize((int) out_w->type, raw.data(), wf.data(), n * kHidden);
            wf = round_f16(wf);
            Buffer b_l((size_t) n * 2);
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
        const double log_z = lse_max + std::log(lse_sum);
        std::sort(best.begin(), best.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        printf("\nnext token from position %lld:\n", (long long) (T - 1));
        for (size_t i = 0; i < std::min<size_t>(5, best.size()); ++i) {
            printf("  token %d  logit %.4f  logprob %.4f\n", best[i].second, best[i].first,
                   (double) best[i].first - log_z);
        }
        shutdown();
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
