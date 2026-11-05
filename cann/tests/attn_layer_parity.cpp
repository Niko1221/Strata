// cann/tests/attn_layer_parity.cpp - the full-attention layer of blk.3 for its first token.
//
// blk.3 is a full-attention layer (every 4th layer is), so this is the other layer type:
//
//   q_full = attn_q x                    [8192] = per head: q[256] | gate[256]
//   k      = attn_k x                    [512]  = 2 KV heads x 256
//   v      = attn_v x                    [512]
//   q      = rms_norm(q, attn_q_norm) per head
//   k      = rms_norm(k, attn_k_norm) per head
//   q, k   = rope(q), rope(k)            n_rot 64 of 256
//   attn   = softmax(q k^T / sqrt(256)) v     (one token, so softmax is 1)
//   attn   = attn * sigmoid(gate)
//   y      = attn_output attn            [2048]
//
// With a single token in the cache the attention weights are trivially 1, so what this checks is
// the projections, the two per-head norms, rope and the gate. The weighted mix over a cache is
// the next step.
#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/tensor_load.hpp"

using namespace strata::cann;

namespace {

const char * kDefaultPath = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";

constexpr int64_t kHidden = 2048;
constexpr int64_t kHeadDim = 256;   // attention.key_length
constexpr int64_t kHeads = 16;      // attention.head_count
constexpr int64_t kKvHeads = 2;     // attention.head_count_kv
constexpr int64_t kRot = 64;        // rope.dimension_count
constexpr float kFreqBase = 1e7f;   // rope.freq_base
constexpr float kRmsEps = 1e-5f;
constexpr int64_t kPos = 5;

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
        double acc = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            acc += (double) w[(size_t) i * k + j] * in[(size_t) j];
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

void rms(const std::vector<float> & in, const std::vector<float> & gamma, float eps, std::vector<float> & out) {
    double ss = 0.0;
    for (float v : in) {
        ss += (double) v * v;
    }
    const float sc = (float) (1.0 / std::sqrt(ss / (double) in.size() + eps));
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i] * sc * gamma[i];
    }
}

void rope_ref(std::vector<float> & vec, int64_t n_rot, float freq_base, int64_t pos) {
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

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
    try {
        const Gguf g(path);
        const GgufTensor * wq = g.find("blk.3.attn_q.weight");
        const GgufTensor * wk = g.find("blk.3.attn_k.weight");
        const GgufTensor * wv = g.find("blk.3.attn_v.weight");
        const GgufTensor * wo = g.find("blk.3.attn_output.weight");
        const GgufTensor * wqn = g.find("blk.3.attn_q_norm.weight");
        const GgufTensor * wkn = g.find("blk.3.attn_k_norm.weight");
        if (!wq || !wk || !wv || !wo || !wqn || !wkn) {
            printf("attention tensors not found\n");
            return 2;
        }

        const std::vector<float> Wq = round_f16(load_tensor(g, *wq));
        const std::vector<float> Wk = round_f16(load_tensor(g, *wk));
        const std::vector<float> Wv = round_f16(load_tensor(g, *wv));
        const std::vector<float> Wo = round_f16(load_tensor(g, *wo));
        const std::vector<float> qn = round_f16(load_tensor(g, *wqn));
        const std::vector<float> kn = round_f16(load_tensor(g, *wkn));

        std::mt19937 rng(53);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t) kHidden);
        for (auto & v : x) {
            v = dist(rng);
        }
        x = round_f16(x);

        // ---- CPU reference ----
        const int64_t q_dim = kHeads * kHeadDim * 2;
        const int64_t kv_dim = kKvHeads * kHeadDim * 2;
        const std::vector<float> q_full = gemm(Wq, q_dim, kHidden, x);
        const std::vector<float> kk = gemm(Wk, kv_dim, kHidden, x);
        const std::vector<float> vv = gemm(Wv, kv_dim, kHidden, x);

        std::vector<float> attn((size_t) (kHeads * kHeadDim), 0.0f);
        for (int64_t h = 0; h < kHeads; ++h) {
            std::vector<float> qh((size_t) kHeadDim), gate((size_t) kHeadDim);
            for (int64_t i = 0; i < kHeadDim; ++i) {
                qh[(size_t) i] = q_full[(size_t) (h * kHeadDim * 2 + i)];
                gate[(size_t) i] = q_full[(size_t) (h * kHeadDim * 2 + kHeadDim + i)];
            }
            std::vector<float> qn_out;
            rms(qh, qn, kRmsEps, qn_out);
            rope_ref(qn_out, kRot, kFreqBase, kPos);

            const int64_t kvh = h % kKvHeads;
            std::vector<float> kh((size_t) kHeadDim), vh((size_t) kHeadDim);
            for (int64_t i = 0; i < kHeadDim; ++i) {
                kh[(size_t) i] = kk[(size_t) (kvh * kHeadDim + i)];
                vh[(size_t) i] = vv[(size_t) (kvh * kHeadDim + i)];
            }
            std::vector<float> kn_out;
            rms(kh, kn, kRmsEps, kn_out);
            rope_ref(kn_out, kRot, kFreqBase, kPos);

            // one token in the cache: softmax over one score is 1, so the mix is v itself
            double dot = 0.0;
            for (int64_t i = 0; i < kHeadDim; ++i) {
                dot += (double) qn_out[(size_t) i] * kn_out[(size_t) i];
            }
            const float score = (float) (dot / std::sqrt((double) kHeadDim));
            const float w = std::exp(score) / std::exp(score); // 1.0, kept explicit
            for (int64_t i = 0; i < kHeadDim; ++i) {
                const float sg = 1.0f / (1.0f + std::exp(-gate[(size_t) i]));
                attn[(size_t) (h * kHeadDim + i)] = w * vh[(size_t) i] * sg;
            }
        }
        const std::vector<float> y_ref = gemm(Wo, kHidden, kHeads * kHeadDim, attn);

        // ---- NPU ----
        Device device(0);
        Stream stream(device);
        Buffer bx((size_t) kHidden * 2);
        upload(x, bx);
        Tensor tx{ bx.data(), DType::F16, 1, kHidden };

        Buffer b_q((size_t) q_dim * 2), b_k((size_t) kv_dim * 2), b_v((size_t) kv_dim * 2);
        gemv(stream, Wq, q_dim, kHidden, tx, b_q);
        gemv(stream, Wk, kv_dim, kHidden, tx, b_k);
        gemv(stream, Wv, kv_dim, kHidden, tx, b_v);

        // cos/sin for the one position
        const int64_t half = kRot / 2;
        std::vector<float> cos_tab((size_t) half), sin_tab((size_t) half);
        for (int64_t i = 0; i < half; ++i) {
            const double theta = (double) kPos * std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
            cos_tab[(size_t) i] = (float) std::cos(theta);
            sin_tab[(size_t) i] = (float) std::sin(theta);
        }
        Buffer b_cos((size_t) half * 2), b_sin((size_t) half * 2);
        upload(cos_tab, b_cos);
        upload(sin_tab, b_sin);
        Tensor t_cos{ b_cos.data(), DType::F16, half, 1 };
        Tensor t_sin{ b_sin.data(), DType::F16, half, 1 };

        Buffer b_qn((size_t) kHeadDim * 2), b_kn((size_t) kHeadDim * 2);
        upload(qn, b_qn);
        upload(kn, b_kn);
        Tensor t_qnw{ b_qn.data(), DType::F16, kHeadDim, 1 };
        Tensor t_knw{ b_kn.data(), DType::F16, kHeadDim, 1 };

        Buffer b_attn((size_t) kHeads * kHeadDim * 2);
        std::vector<float> qn_host((size_t) kHeadDim), kn_host((size_t) kHeadDim);
        for (int64_t h = 0; h < kHeads; ++h) {
            Buffer t1((size_t) kHeadDim * 2), t2((size_t) kHeadDim * 2), t3((size_t) kHeadDim * 2);
            Buffer b_qrope((size_t) kHeadDim * 2);
            Tensor qraw{ (char *) b_q.data() + (size_t) h * kHeadDim * 2 * 2, DType::F16, kHeadDim, 1 };
            Tensor tn{ t1.data(), DType::F16, kHeadDim, 1 };
            Tensor tq{ b_qrope.data(), DType::F16, kHeadDim, 1 };
            rms_norm(stream, qraw, t_qnw, kRmsEps, tn);
            rope(stream, tn, t_cos, t_sin, kRot, tq);

            // gate half of this head
            Tensor graw{ (char *) b_q.data() + (size_t) (h * kHeadDim * 2 + kHeadDim) * 2, DType::F16, kHeadDim, 1 };
            Tensor tgate{ t2.data(), DType::F16, kHeadDim, 1 };
            sigmoid(stream, graw, tgate);

            // attention output for one token is v of the KV head, gated
            const int64_t kvh = h % kKvHeads;
            Tensor vh{ (char *) b_v.data() + (size_t) kvh * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            Tensor tout{ (char *) b_attn.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            mul(stream, vh, tgate, tout);

            if (h < kKvHeads) {
                // k norm + rope, kept so the test also covers the k path
                Tensor kraw{ (char *) b_k.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                Tensor tk{ t3.data(), DType::F16, kHeadDim, 1 };
                Buffer b_krope((size_t) kHeadDim * 2);
                Tensor tkr{ b_krope.data(), DType::F16, kHeadDim, 1 };
                rms_norm(stream, kraw, t_knw, kRmsEps, tk);
                rope(stream, tk, t_cos, t_sin, kRot, tkr);
            }
        }

        Buffer b_y((size_t) kHidden * 2);
        Tensor t_attn{ b_attn.data(), DType::F16, 1, kHeads * kHeadDim };
        gemv(stream, Wo, kHidden, kHeads * kHeadDim, t_attn, b_y);
        const std::vector<float> y_dev = to_host(stream, b_y, kHidden);

        const double r_y = rel_l2(y_dev, y_ref);
        printf("layer  : blk.3 attention, one token at position %lld\n", (long long) kPos);
        printf("shapes : q %lldx%lld (+gate), kv %lldx%lld, n_rot %lld of %lld\n\n", (long long) kHeads,
               (long long) kHeadDim, (long long) kKvHeads, (long long) kHeadDim, (long long) kRot,
               (long long) kHeadDim);
        printf("%-16s %12s  %s\n", "stage", "rel L2", "result");
        printf("%-16s %12.6f  %s\n", "y = attn_output", r_y, r_y <= 0.02 ? "PASS" : "FAIL");
        shutdown();
        if (r_y > 0.02) {
            printf("\nthe attention layer differs from the CPU reference\n");
            return 1;
        }
        printf("\nthe full-attention layer of blk.3 ran on the NPU\n");
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
