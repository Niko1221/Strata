// cann/tests/attn_cache_parity.cpp - the full-attention layer of blk.3 across tokens, with a KV cache.
//
// The single-token test could not exercise softmax: one score is always 1. Here T tokens go
// through the layer, each one writing its k and v into the cache and attending over everything
// up to itself:
//
//   scores[j] = <q_h, k_cache[j]> / sqrt(head_dim)     j = 0..t
//   w         = softmax(scores[0..t])
//   attn_h    = sum_j w[j] * v_cache[j]                (GQA: 8 q heads share one kv head)
//   attn_h    = attn_h * sigmoid(gate_h)
//   y_t       = attn_output attn
#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
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
constexpr int64_t kHeadDim = 256;
constexpr int64_t kHeads = 16;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kRot = 64;
constexpr float kFreqBase = 1e7f;
constexpr float kRmsEps = 1e-5f;
constexpr int64_t kQDim = kHeads * kHeadDim * 2;
constexpr int64_t kKvDim = kKvHeads * kHeadDim;

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

void rope_ref(std::vector<float> & vec, int64_t pos) {
    const int64_t half = kRot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const double theta = (double) pos * std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
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
    const int64_t T = argc > 2 ? std::atoll(argv[2]) : 4;
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

        std::mt19937 rng(59);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<std::vector<float>> xs((size_t) T, std::vector<float>((size_t) kHidden));
        for (auto & v : xs) {
            for (auto & q : v) q = dist(rng);
            v = round_f16(v);
        }

        // ---- CPU reference ----
        std::vector<std::vector<float>> kcache((size_t) kKvHeads), vcache((size_t) kKvHeads);
        std::vector<std::vector<float>> y_ref;
        for (int64_t t = 0; t < T; ++t) {
            const std::vector<float> q_full = gemm(Wq, kQDim, kHidden, xs[(size_t) t]);
            const std::vector<float> kk = gemm(Wk, kKvDim, kHidden, xs[(size_t) t]);
            const std::vector<float> vv = gemm(Wv, kKvDim, kHidden, xs[(size_t) t]);

            for (int64_t kvh = 0; kvh < kKvHeads; ++kvh) {
                std::vector<float> kh(kk.begin() + (size_t) (kvh * kHeadDim),
                                      kk.begin() + (size_t) ((kvh + 1) * kHeadDim));
                std::vector<float> knorm;
                rms(kh, kn, kRmsEps, knorm);
                rope_ref(knorm, t);
                kcache[(size_t) kvh].insert(kcache[(size_t) kvh].end(), knorm.begin(), knorm.end());
                vcache[(size_t) kvh].insert(vcache[(size_t) kvh].end(),
                                            vv.begin() + (size_t) (kvh * kHeadDim),
                                            vv.begin() + (size_t) ((kvh + 1) * kHeadDim));
            }

            std::vector<float> attn((size_t) (kHeads * kHeadDim), 0.0f);
            for (int64_t h = 0; h < kHeads; ++h) {
                std::vector<float> qh(q_full.begin() + (size_t) (h * kHeadDim * 2),
                                      q_full.begin() + (size_t) (h * kHeadDim * 2 + kHeadDim));
                std::vector<float> gate(q_full.begin() + (size_t) (h * kHeadDim * 2 + kHeadDim),
                                        q_full.begin() + (size_t) (h * kHeadDim * 2 + 2 * kHeadDim));
                std::vector<float> qnorm;
                rms(qh, qn, kRmsEps, qnorm);
                rope_ref(qnorm, t);

                const int64_t kvh = h / (kHeads / kKvHeads); // blocked, not interleaved: 8 q heads per kv head
                const int64_t n = t + 1;
                std::vector<float> scores((size_t) n);
                float mx = -INFINITY;
                for (int64_t j = 0; j < n; ++j) {
                    double dot = 0.0;
                    for (int64_t i = 0; i < kHeadDim; ++i) {
                        dot += (double) qnorm[(size_t) i] * kcache[(size_t) kvh][(size_t) (j * kHeadDim + i)];
                    }
                    scores[(size_t) j] = (float) (dot / std::sqrt((double) kHeadDim));
                    mx = std::max(mx, scores[(size_t) j]);
                }
                double ssum = 0.0;
                for (auto & v : scores) {
                    v = std::exp(v - mx);
                    ssum += v;
                }
                for (auto & v : scores) {
                    v = (float) (v / ssum);
                }
                for (int64_t i = 0; i < kHeadDim; ++i) {
                    double acc = 0.0;
                    for (int64_t j = 0; j < n; ++j) {
                        acc += (double) scores[(size_t) j] * vcache[(size_t) kvh][(size_t) (j * kHeadDim + i)];
                    }
                    const float sg = 1.0f / (1.0f + std::exp(-gate[(size_t) i]));
                    attn[(size_t) (h * kHeadDim + i)] = (float) acc * sg;
                }
            }
            y_ref.push_back(gemm(Wo, kHidden, kHeads * kHeadDim, attn));
        }

        // ---- NPU ----
        Device device(0);
        Stream stream(device);
        Buffer bx((size_t) kHidden * 2);
        Tensor tx{ bx.data(), DType::F16, 1, kHidden };

        Buffer b_q((size_t) kQDim * 2), b_k((size_t) kKvDim * 2), b_v((size_t) kKvDim * 2);
        Buffer b_qn((size_t) kHeadDim * 2), b_kn((size_t) kHeadDim * 2);
        upload(qn, b_qn);
        upload(kn, b_kn);
        Tensor t_qnw{ b_qn.data(), DType::F16, kHeadDim, 1 };
        Tensor t_knw{ b_kn.data(), DType::F16, kHeadDim, 1 };

        // per kv head caches: T rows of head_dim
        std::vector<std::unique_ptr<Buffer>> kbuf, vbuf;
        for (int64_t kvh = 0; kvh < kKvHeads; ++kvh) {
            kbuf.push_back(std::make_unique<Buffer>((size_t) T * kHeadDim * 2));
            vbuf.push_back(std::make_unique<Buffer>((size_t) T * kHeadDim * 2));
            std::vector<float> zeros((size_t) T * kHeadDim, 0.0f);
            upload(zeros, *kbuf.back());
            upload(zeros, *vbuf.back());
        }

        // cos/sin for every position
        Buffer b_cos((size_t) T * (kRot / 2) * 2), b_sin((size_t) T * (kRot / 2) * 2);
        {
            const int64_t half = kRot / 2;
            std::vector<float> ct((size_t) T * half), st((size_t) T * half);
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t i = 0; i < half; ++i) {
                    const double theta = (double) t * std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
                    ct[(size_t) (t * half + i)] = (float) std::cos(theta);
                    st[(size_t) (t * half + i)] = (float) std::sin(theta);
                }
            }
            upload(ct, b_cos);
            upload(st, b_sin);
        }

        Buffer b_attn((size_t) kHeads * kHeadDim * 2);
        std::vector<std::vector<float>> y_dev;
        for (int64_t t = 0; t < T; ++t) {
            upload(xs[(size_t) t], bx);
            gemv(stream, Wq, kQDim, kHidden, tx, b_q);
            gemv(stream, Wk, kKvDim, kHidden, tx, b_k);
            gemv(stream, Wv, kKvDim, kHidden, tx, b_v);

            // k norm + rope, then append to the cache
            for (int64_t kvh = 0; kvh < kKvHeads; ++kvh) {
                Buffer t1((size_t) kHeadDim * 2), t2((size_t) kHeadDim * 2);
                Buffer b_krope((size_t) kHeadDim * 2);
                Tensor kraw{ (char *) b_k.data() + (size_t) kvh * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                Tensor tn{ t1.data(), DType::F16, kHeadDim, 1 };
                Tensor tkr{ b_krope.data(), DType::F16, kHeadDim, 1 };
                rms_norm(stream, kraw, t_knw, kRmsEps, tn);
                Tensor tcos{ (char *) b_cos.data() + (size_t) t * (kRot / 2) * 2, DType::F16, kRot / 2, 1 };
                Tensor tsin{ (char *) b_sin.data() + (size_t) t * (kRot / 2) * 2, DType::F16, kRot / 2, 1 };
                rope(stream, tn, tcos, tsin, kRot, tkr);

                Tensor krow{ (char *) kbuf[(size_t) kvh]->data() + (size_t) t * kHeadDim * 2, DType::F16,
                             kHeadDim, 1 };
                muls(stream, tkr, 1.0f, krow);
                Tensor vraw{ (char *) b_v.data() + (size_t) kvh * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                Tensor vrow{ (char *) vbuf[(size_t) kvh]->data() + (size_t) t * kHeadDim * 2, DType::F16,
                             kHeadDim, 1 };
                muls(stream, vraw, 1.0f, vrow);
            }

            for (int64_t h = 0; h < kHeads; ++h) {
                Buffer t1((size_t) kHeadDim * 2), t2((size_t) kHeadDim * 2);
                Buffer b_qrope((size_t) kHeadDim * 2);
                Tensor qraw{ (char *) b_q.data() + (size_t) h * kHeadDim * 2 * 2, DType::F16, kHeadDim, 1 };
                Tensor tn{ t1.data(), DType::F16, kHeadDim, 1 };
                Tensor tqr{ b_qrope.data(), DType::F16, kHeadDim, 1 };
                rms_norm(stream, qraw, t_qnw, kRmsEps, tn);
                Tensor tcos{ (char *) b_cos.data() + (size_t) t * (kRot / 2) * 2, DType::F16, kRot / 2, 1 };
                Tensor tsin{ (char *) b_sin.data() + (size_t) t * (kRot / 2) * 2, DType::F16, kRot / 2, 1 };
                rope(stream, tn, tcos, tsin, kRot, tqr);

                const int64_t kvh = h / (kHeads / kKvHeads); // blocked, not interleaved: 8 q heads per kv head
                const int64_t n = t + 1;
                // scores = K_cache[0..n) * q   -> [n, 1]
                Buffer b_scores((size_t) n * 2), b_w((size_t) n * 2), b_mix((size_t) kHeadDim * 2);
                Tensor kc{ kbuf[(size_t) kvh]->data(), DType::F16, kHeadDim, n };
                Tensor q_col{ b_qrope.data(), DType::F16, 1, kHeadDim };
                Tensor scores{ b_scores.data(), DType::F16, 1, n };
                matmul(stream, kc, q_col, scores);

                // scores are q.k / sqrt(head_dim), then softmax over the n of them
                Buffer b_scaled((size_t) n * 2);
                Tensor sc_in{ b_scores.data(), DType::F16, n, 1 };
                Tensor sc_out{ b_scaled.data(), DType::F16, n, 1 };
                muls(stream, sc_in, 1.0f / std::sqrt((float) kHeadDim), sc_out);
                Tensor srow{ b_scaled.data(), DType::F16, n, 1 };
                Tensor wrow{ b_w.data(), DType::F16, n, 1 };
                softmax(stream, srow, wrow);

                // mix = w[1, n] * V_cache[n, head_dim]
                Tensor wmat{ b_w.data(), DType::F16, n, 1 }; // one row of n weights
                Tensor vc{ vbuf[(size_t) kvh]->data(), DType::F16, kHeadDim, n };
                Tensor mix{ b_mix.data(), DType::F16, kHeadDim, 1 };
                matmul(stream, wmat, vc, mix);

                // gate
                Tensor graw{ (char *) b_q.data() + (size_t) (h * kHeadDim * 2 + kHeadDim) * 2, DType::F16,
                             kHeadDim, 1 };
                Tensor tgate{ t2.data(), DType::F16, kHeadDim, 1 };
                sigmoid(stream, graw, tgate);
                Tensor tout{ (char *) b_attn.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                mul(stream, mix, tgate, tout);
            }

            Buffer b_y((size_t) kHidden * 2);
            Tensor t_attn{ b_attn.data(), DType::F16, 1, kHeads * kHeadDim };
            gemv(stream, Wo, kHidden, kHeads * kHeadDim, t_attn, b_y);
            y_dev.push_back(to_host(stream, b_y, kHidden));
        }

        printf("layer  : blk.3 attention across %lld tokens with a KV cache\n", (long long) T);
        printf("cache  : %lld kv heads x %lld dim, GQA %lld:%lld\n\n", (long long) kKvHeads, (long long) kHeadDim,
               (long long) (kHeads / kKvHeads), (long long) 1);
        printf("%-8s %12s  %s\n", "token", "y rel L2", "result");
        bool ok = true;
        for (int64_t t = 0; t < T; ++t) {
            const double r = rel_l2(y_dev[(size_t) t], y_ref[(size_t) t]);
            printf("%-8lld %12.6f  %s\n", (long long) t, r, r <= 0.02 ? "PASS" : "FAIL");
            ok = ok && r <= 0.02;
        }
        shutdown();
        if (!ok) {
            printf("\nthe cached attention differs from the CPU reference\n");
            return 1;
        }
        printf("\n%lld tokens went through the attention layer with a KV cache\n", (long long) T);
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
