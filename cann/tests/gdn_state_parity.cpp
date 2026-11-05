// cann/tests/gdn_state_parity.cpp - the GDN block of blk.0 across several tokens.
//
// The single-token test starts from a zero state, which is only the first decode step. This one
// carries both pieces of state between tokens:
//
//   conv state   the last three qkv vectors, so the kernel-4 short conv sees history
//   ssm state    one 128x128 matrix per value head, updated by the delta rule in place
//
// T tokens go through the same block; the test compares every token's output and the final
// state against a float CPU reference that runs the same recurrence.
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
#include "strata/cann/gdn.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/tensor_load.hpp"

using namespace strata::cann;

namespace {

const char * kDefaultPath = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";

constexpr int64_t kHidden = 2048;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kKHeads = 16;
constexpr int64_t kVHeads = 32;
constexpr int64_t kConvK = 4;
constexpr int64_t kKeyDim = kHeadDim * kKHeads;
constexpr int64_t kValueDim = kHeadDim * kVHeads;
constexpr int64_t kQkvDim = kKeyDim * 2 + kValueDim;
constexpr float kRmsEps = 1e-5f;

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

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
    const int64_t T = argc > 2 ? std::atoll(argv[2]) : 4;
    try {
        const Gguf g(path);
        const GgufTensor * wqkv = g.find("blk.0.attn_qkv.weight");
        const GgufTensor * wgate = g.find("blk.0.attn_gate.weight");
        const GgufTensor * wbeta = g.find("blk.0.ssm_beta.weight");
        const GgufTensor * walpha = g.find("blk.0.ssm_alpha.weight");
        const GgufTensor * wconv = g.find("blk.0.ssm_conv1d.weight");
        const GgufTensor * wnorm = g.find("blk.0.ssm_norm.weight");
        const GgufTensor * wout = g.find("blk.0.ssm_out.weight");
        const GgufTensor * wdt = g.find("blk.0.ssm_dt.bias");
        const GgufTensor * wa = g.find("blk.0.ssm_a");
        if (!wqkv || !wgate || !wbeta || !walpha || !wconv || !wnorm || !wout || !wdt || !wa) {
            printf("GDN tensors not found\n");
            return 2;
        }

        const std::vector<float> Wqkv = round_f16(load_tensor(g, *wqkv));
        const std::vector<float> Wgate = round_f16(load_tensor(g, *wgate));
        const std::vector<float> Wbeta = round_f16(load_tensor(g, *wbeta));
        const std::vector<float> Walpha = round_f16(load_tensor(g, *walpha));
        const std::vector<float> Wconv = round_f16(load_tensor(g, *wconv));
        const std::vector<float> Wnorm = round_f16(load_tensor(g, *wnorm));
        const std::vector<float> Wout = round_f16(load_tensor(g, *wout));
        const std::vector<float> dt = load_tensor(g, *wdt);
        const std::vector<float> negA = load_tensor(g, *wa);

        std::mt19937 rng(37);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<std::vector<float>> xs((size_t) T, std::vector<float>((size_t) kHidden));
        for (auto & v : xs) {
            for (auto & q : v) q = dist(rng);
            v = round_f16(v);
        }

        // conv taps: tap k is the per-channel weight Wconv[c][k]
        std::vector<std::vector<float>> taps((size_t) kConvK, std::vector<float>((size_t) kQkvDim));
        for (int64_t k = 0; k < kConvK; ++k) {
            for (int64_t c = 0; c < kQkvDim; ++c) {
                taps[(size_t) k][(size_t) c] = Wconv[(size_t) c * kConvK + k];
            }
        }

        const float dn_scale = 1.0f / std::sqrt((float) kHeadDim);

        // ---- CPU reference ----
        std::vector<std::vector<float>> y_ref((size_t) T, std::vector<float>((size_t) kHidden));
        std::vector<std::vector<float>> state_ref((size_t) kVHeads,
                                                  std::vector<float>((size_t) kHeadDim * kHeadDim, 0.0f));
        std::vector<std::vector<float>> hist((size_t) kConvK, std::vector<float>((size_t) kQkvDim, 0.0f));
        for (int64_t t = 0; t < T; ++t) {
            const std::vector<float> qkv = gemm(Wqkv, kQkvDim, kHidden, xs[(size_t) t]);
            const std::vector<float> z = gemm(Wgate, kValueDim, kHidden, xs[(size_t) t]);
            const std::vector<float> beta_raw = gemm(Wbeta, kVHeads, kHidden, xs[(size_t) t]);
            const std::vector<float> alpha_raw = gemm(Walpha, kVHeads, kHidden, xs[(size_t) t]);

            for (int64_t k = kConvK - 1; k > 0; --k) {
                hist[(size_t) k] = hist[(size_t) (k - 1)];
            }
            hist[0] = qkv;

            std::vector<float> conv((size_t) kQkvDim, 0.0f);
            for (int64_t k = 0; k < kConvK; ++k) {
                for (int64_t c = 0; c < kQkvDim; ++c) {
                    conv[(size_t) c] += taps[(size_t) k][(size_t) c] * hist[(size_t) k][(size_t) c];
                }
            }
            for (auto & v : conv) {
                v = v / (1.0f + std::exp(-v));
            }

            std::vector<float> q((size_t) kKeyDim), kk((size_t) kKeyDim), v((size_t) kValueDim);
            for (int64_t i = 0; i < kKeyDim; ++i) {
                q[(size_t) i] = conv[(size_t) i];
                kk[(size_t) i] = conv[(size_t) (kKeyDim + i)];
            }
            for (int64_t i = 0; i < kValueDim; ++i) {
                v[(size_t) i] = conv[(size_t) (2 * kKeyDim + i)];
            }
            auto l2 = [](std::vector<float> & vec, int64_t n_heads) {
                for (int64_t h = 0; h < n_heads; ++h) {
                    double ss = 0.0;
                    for (int64_t i = 0; i < kHeadDim; ++i) {
                        const double t2 = vec[(size_t) (h * kHeadDim + i)];
                        ss += t2 * t2;
                    }
                    const float sc = (float) (1.0 / std::sqrt(ss / (double) kHeadDim + kRmsEps / (double) kHeadDim)) *
                                     (1.0f / std::sqrt((float) kHeadDim));
                    for (int64_t i = 0; i < kHeadDim; ++i) {
                        vec[(size_t) (h * kHeadDim + i)] *= sc;
                    }
                }
            };
            l2(q, kKHeads);
            l2(kk, kKHeads);

            std::vector<float> dn((size_t) kValueDim, 0.0f);
            for (int64_t h = 0; h < kVHeads; ++h) {
                const int64_t kh = h % kKHeads;
                const float bt = 1.0f / (1.0f + std::exp(-beta_raw[(size_t) h]));
                const float biased = alpha_raw[(size_t) h] + dt[(size_t) h];
                const float sp = std::log1p(std::exp(biased > 20.0f ? 20.0f : biased));
                const float decay = std::exp(sp * negA[(size_t) h]);

                float * M = state_ref[(size_t) h].data();
                for (int64_t i = 0; i < kHeadDim * kHeadDim; ++i) {
                    M[i] *= decay;
                }
                std::vector<float> delta((size_t) kHeadDim, 0.0f);
                for (int64_t j = 0; j < kHeadDim; ++j) {
                    const float * row = M + j * kHeadDim;
                    double d = 0.0;
                    for (int64_t i = 0; i < kHeadDim; ++i) {
                        d += (double) row[i] * kk[(size_t) (kh * kHeadDim + i)];
                    }
                    delta[(size_t) j] = (v[(size_t) (h * kHeadDim + j)] - (float) d) * bt;
                }
                for (int64_t j = 0; j < kHeadDim; ++j) {
                    float * row = M + j * kHeadDim;
                    const float dv = delta[(size_t) j];
                    for (int64_t i = 0; i < kHeadDim; ++i) {
                        row[i] += kk[(size_t) (kh * kHeadDim + i)] * dv;
                    }
                }
                for (int64_t j = 0; j < kHeadDim; ++j) {
                    const float * row = M + j * kHeadDim;
                    double d = 0.0;
                    for (int64_t i = 0; i < kHeadDim; ++i) {
                        d += (double) row[i] * q[(size_t) (kh * kHeadDim + i)];
                    }
                    dn[(size_t) (h * kHeadDim + j)] = (float) d * dn_scale;
                }
            }

            std::vector<float> normed((size_t) kValueDim);
            for (int64_t h = 0; h < kVHeads; ++h) {
                double ss = 0.0;
                for (int64_t i = 0; i < kHeadDim; ++i) {
                    const double t2 = dn[(size_t) (h * kHeadDim + i)];
                    ss += t2 * t2;
                }
                const float sc = (float) (1.0 / std::sqrt(ss / (double) kHeadDim + kRmsEps));
                for (int64_t i = 0; i < kHeadDim; ++i) {
                    const float nv = dn[(size_t) (h * kHeadDim + i)] * sc * Wnorm[(size_t) i];
                    const float zv = z[(size_t) (h * kHeadDim + i)];
                    normed[(size_t) (h * kHeadDim + i)] = nv * (zv / (1.0f + std::exp(-zv)));
                }
            }
            y_ref[(size_t) t] = gemm(Wout, kHidden, kValueDim, normed);
        }

        // ---- NPU ----
        Device device(0);
        Stream stream(device);
        Buffer bx((size_t) kHidden * 2);
        Tensor tx{ bx.data(), DType::F16, 1, kHidden };

        Buffer b_qkv((size_t) kQkvDim * 2), b_z((size_t) kValueDim * 2), b_conv((size_t) kQkvDim * 2);
        Buffer b_silu((size_t) kQkvDim * 2);
        Buffer b_tap((size_t) kQkvDim * 2), b_acc((size_t) kQkvDim * 2);
        Tensor t_qkv{ b_qkv.data(), DType::F16, 1, kQkvDim };
        Tensor t_z{ b_z.data(), DType::F16, 1, kValueDim };
        Tensor t_conv{ b_conv.data(), DType::F16, 1, kQkvDim };
        Tensor t_silu{ b_silu.data(), DType::F16, 1, kQkvDim };
        Tensor t_tap{ b_tap.data(), DType::F16, 1, kQkvDim };
        Tensor t_acc{ b_acc.data(), DType::F16, 1, kQkvDim };

        Buffer b_q((size_t) kKeyDim * 2), b_k((size_t) kKeyDim * 2), b_normed((size_t) kValueDim * 2);
        Buffer b_ones((size_t) kHeadDim * 2), b_wnorm((size_t) kHeadDim * 2);
        upload(std::vector<float>((size_t) kHeadDim, 1.0f), b_ones);
        upload(Wnorm, b_wnorm);
        Tensor t_ones{ b_ones.data(), DType::F16, kHeadDim, 1 };
        Tensor t_wnorm{ b_wnorm.data(), DType::F16, kHeadDim, 1 };

        // persistent per-head state, kept on the device across tokens
        std::vector<std::unique_ptr<Buffer>> state_bufs;
        state_bufs.reserve((size_t) kVHeads);
        for (int64_t h = 0; h < kVHeads; ++h) {
            state_bufs.push_back(std::make_unique<Buffer>((size_t) kHeadDim * kHeadDim * 2));
            upload(std::vector<float>((size_t) kHeadDim * kHeadDim, 0.0f), *state_bufs.back());
        }

        // conv history on the device: three qkv vectors
        std::vector<std::unique_ptr<Buffer>> hist_bufs;
        for (int64_t k = 0; k < kConvK - 1; ++k) {
            hist_bufs.push_back(std::make_unique<Buffer>((size_t) kQkvDim * 2));
            upload(std::vector<float>((size_t) kQkvDim, 0.0f), *hist_bufs.back());
        }

        std::vector<std::vector<float>> y_dev;
        for (int64_t t = 0; t < T; ++t) {
            upload(xs[(size_t) t], bx);
            gemv(stream, Wqkv, kQkvDim, kHidden, tx, b_qkv);
            gemv(stream, Wgate, kValueDim, kHidden, tx, b_z);

            // conv: sum over the taps of tap_k * qkv[t-k], then silu
            upload(std::vector<float>((size_t) kQkvDim, 0.0f), b_acc);
            for (int64_t k = 0; k < kConvK; ++k) {
                upload(taps[(size_t) k], b_tap);
                if (k == 0) {
                    mul(stream, t_qkv, t_tap, t_conv);
                } else {
                    Tensor src{ hist_bufs[(size_t) (k - 1)]->data(), DType::F16, 1, kQkvDim };
                    mul(stream, src, t_tap, t_conv);
                }
                add(stream, t_acc, t_conv, t_acc);
            }
            silu(stream, t_acc, t_silu);

            // rotate history: hist[0] <- qkv(t), hist[k] <- hist[k-1]
            for (int64_t k = kConvK - 2; k > 0; --k) {
                Buffer & dst = *hist_bufs[(size_t) k];
                Buffer & src = *hist_bufs[(size_t) (k - 1)];
                Tensor td{ dst.data(), DType::F16, 1, kQkvDim };
                Tensor ts{ src.data(), DType::F16, 1, kQkvDim };
                muls(stream, ts, 1.0f, td);
            }
            {
                Tensor td{ hist_bufs[0]->data(), DType::F16, 1, kQkvDim };
                muls(stream, t_qkv, 1.0f, td);
            }

            // l2 norm per k head
            for (int64_t h = 0; h < kKHeads; ++h) {
                Buffer t1((size_t) kHeadDim * 2), t2((size_t) kHeadDim * 2);
                Tensor tin{ (char *) b_silu.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                Tensor tn{ t1.data(), DType::F16, kHeadDim, 1 };
                Tensor ts{ t2.data(), DType::F16, kHeadDim, 1 };
                Tensor tout{ (char *) b_q.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                rms_norm(stream, tin, t_ones, kRmsEps / (double) kHeadDim, tn);
                muls(stream, tn, 1.0f / std::sqrt((float) kHeadDim), ts);
                muls(stream, ts, 1.0f, tout);

                Tensor kin{ (char *) b_silu.data() + (size_t) (kKeyDim + h * kHeadDim) * 2, DType::F16, kHeadDim, 1 };
                Tensor kout{ (char *) b_k.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                rms_norm(stream, kin, t_ones, kRmsEps / (double) kHeadDim, tn);
                muls(stream, tn, 1.0f / std::sqrt((float) kHeadDim), ts);
                muls(stream, ts, 1.0f, kout);
            }

            // delta net per v head, state carried in place
            const std::vector<float> beta_raw_dev = gemm(Wbeta, kVHeads, kHidden, xs[(size_t) t]);
            const std::vector<float> alpha_raw_dev = gemm(Walpha, kVHeads, kHidden, xs[(size_t) t]);
            for (int64_t h = 0; h < kVHeads; ++h) {
                const int64_t kh = h % kKHeads;
                const float bt = 1.0f / (1.0f + std::exp(-beta_raw_dev[(size_t) h]));
                const float biased = alpha_raw_dev[(size_t) h] + dt[(size_t) h];
                const float sp = std::log1p(std::exp(biased > 20.0f ? 20.0f : biased));
                const float gate = sp * negA[(size_t) h];

                Buffer b_g(2), b_b(2), b_out((size_t) kHeadDim * 2);
                upload({ gate }, b_g);
                upload({ bt }, b_b);
                Tensor tq{ (char *) b_q.data() + (size_t) kh * kHeadDim * 2, DType::F16, 1, kHeadDim };
                Tensor tk{ (char *) b_k.data() + (size_t) kh * kHeadDim * 2, DType::F16, 1, kHeadDim };
                Tensor tv{ (char *) b_silu.data() + (size_t) (2 * kKeyDim + h * kHeadDim) * 2, DType::F16, 1,
                           kHeadDim };
                Tensor tg{ b_g.data(), DType::F16, 1, 1 };
                Tensor tb{ b_b.data(), DType::F16, 1, 1 };
                Tensor tst{ state_bufs[(size_t) h]->data(), DType::F16, kHeadDim, kHeadDim };
                Tensor tout{ b_out.data(), DType::F16, 1, kHeadDim };
                GdnShape shape{ kHeadDim, 1, false };
                gdn(stream, tq, tk, tv, tg, tb, tst, tout, shape);

                // gated rms norm into b_normed
                Buffer t1((size_t) kHeadDim * 2), t2((size_t) kHeadDim * 2), t3((size_t) kHeadDim * 2);
                Tensor tin{ b_out.data(), DType::F16, kHeadDim, 1 };
                Tensor tn{ t1.data(), DType::F16, kHeadDim, 1 };
                Tensor tz{ (char *) b_z.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                Tensor tzs{ t2.data(), DType::F16, kHeadDim, 1 };
                Tensor tm{ t3.data(), DType::F16, kHeadDim, 1 };
                Tensor tnout{ (char *) b_normed.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
                rms_norm(stream, tin, t_wnorm, kRmsEps, tn);
                silu(stream, tz, tzs);
                mul(stream, tn, tzs, tm);
                muls(stream, tm, 1.0f, tnout);
            }

            Buffer b_y((size_t) kHidden * 2);
            Tensor t_normed{ b_normed.data(), DType::F16, 1, kValueDim };
            gemv(stream, Wout, kHidden, kValueDim, t_normed, b_y);
            y_dev.push_back(to_host(stream, b_y, kHidden));
        }

        // final states back to the host
        std::vector<std::vector<float>> state_dev;
        for (int64_t h = 0; h < kVHeads; ++h) {
            state_dev.push_back(to_host(stream, *state_bufs[(size_t) h], kHeadDim * kHeadDim));
        }

        printf("block  : blk.0 GDN across %lld tokens, state carried on the device\n\n", (long long) T);
        printf("%-10s %12s  %s\n", "token", "y rel L2", "result");
        bool ok = true;
        for (int64_t t = 0; t < T; ++t) {
            const double r = rel_l2(y_dev[(size_t) t], y_ref[(size_t) t]);
            printf("%-10lld %12.6f  %s\n", (long long) t, r, r <= 0.02 ? "PASS" : "FAIL");
            ok = ok && r <= 0.02;
        }
        double worst_state = 0.0;
        for (int64_t h = 0; h < kVHeads; ++h) {
            worst_state = std::max(worst_state, rel_l2(state_dev[(size_t) h], state_ref[(size_t) h]));
        }
        printf("%-10s %12.6f  %s\n", "state", worst_state, worst_state <= 0.02 ? "PASS" : "FAIL");
        ok = ok && worst_state <= 0.02;

        shutdown();
        if (!ok) {
            printf("\nstate does not carry correctly across tokens\n");
            return 1;
        }
        printf("\n%lld tokens went through the GDN block with the state carried on the device\n", (long long) T);
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
