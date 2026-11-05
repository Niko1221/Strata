// cann/tests/gdn_block_parity.cpp - the whole GDN block of blk.0 for one token, from real weights.
//
// blk.0 is a recurrent (gated delta net) layer, so its attention half is:
//
//   qkv  = wqkv x                              [8192] = q[16x128] | k[16x128] | v[32x128]
//   z    = wqkv_gate x                         [4096]
//   beta = sigmoid(ssm_beta x)                 [32]
//   a    = softplus(ssm_alpha x + ssm_dt) * ssm_a   [32]
//   conv = silu(conv1d(qkv))                   short conv, kernel 4
//   q,k  = l2_norm over each 128-value head, repeated 16 -> 32 heads
//   out  = gated_delta_net(q, k, v, a, beta, state)
//   out  = rms_norm(out, ssm_norm) * silu(z)
//   y    = ssm_out out                         [2048]
//
// The test is one token with zero state, which is the decode step: the short conv sees only its
// own tap, and the delta net does one update.
#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// shape facts, from the file's metadata
constexpr int64_t kHidden = 2048;
constexpr int64_t kHeadDim = 128;   // ssm.state_size
constexpr int64_t kKHeads = 16;     // ssm.group_count
constexpr int64_t kVHeads = 32;     // ssm.time_step_rank
constexpr int64_t kConvK = 4;       // ssm.conv_kernel
constexpr int64_t kKeyDim = kHeadDim * kKHeads;   // 2048
constexpr int64_t kValueDim = kHeadDim * kVHeads; // 4096
constexpr float kRmsEps = 1e-5f;    // qwen35moe.attention.layer_norm_rms_epsilon is 1e-5

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

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
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
        const std::vector<float> Wconv = round_f16(load_tensor(g, *wconv)); // [kConvK][8192]
        const std::vector<float> Wnorm = round_f16(load_tensor(g, *wnorm)); // [128]
        const std::vector<float> Wout = round_f16(load_tensor(g, *wout));   // [4096][2048]
        const std::vector<float> dt = load_tensor(g, *wdt);
        const std::vector<float> negA = load_tensor(g, *wa);

        std::mt19937 rng(31);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t) kHidden);
        for (auto & v : x) {
            v = dist(rng);
        }
        x = round_f16(x);

        // ---- CPU reference ----
        auto gemm = [](const std::vector<float> & w, int64_t m, int64_t k, const std::vector<float> & in) {
            std::vector<float> out((size_t) m, 0.0f);
            for (int64_t i = 0; i < m; ++i) {
                double acc = 0.0;
                for (int64_t j = 0; j < k; ++j) {
                    acc += (double) w[(size_t) i * k + j] * in[(size_t) j];
                }
                out[(size_t) i] = (float) acc;
            }
            return out;
        };

        const std::vector<float> qkv = gemm(Wqkv, kKeyDim * 2 + kValueDim, kHidden, x);
        const std::vector<float> z = gemm(Wgate, kValueDim, kHidden, x);
        const std::vector<float> beta_raw = gemm(Wbeta, kVHeads, kHidden, x);
        const std::vector<float> alpha_raw = gemm(Walpha, kVHeads, kHidden, x);

        std::vector<float> beta((size_t) kVHeads), a((size_t) kVHeads);
        for (int64_t h = 0; h < kVHeads; ++h) {
            beta[(size_t) h] = 1.0f / (1.0f + std::exp(-beta_raw[(size_t) h]));
            const float biased = alpha_raw[(size_t) h] + dt[(size_t) h];
            const float sp = std::log1p(std::exp(biased > 20.0f ? 20.0f : biased));
            a[(size_t) h] = sp * negA[(size_t) h];
        }

        // short conv: with zero state and one token only the k=0 tap contributes
        std::vector<float> conv((size_t) (kKeyDim * 2 + kValueDim));
        for (int64_t c = 0; c < kKeyDim * 2 + kValueDim; ++c) {
            const float v = Wconv[(size_t) c * kConvK + 0] * qkv[(size_t) c];
            conv[(size_t) c] = v / (1.0f + std::exp(-v));
        }

        // split and l2-normalize q/k per head
        std::vector<float> q((size_t) kKeyDim), k((size_t) kKeyDim), v((size_t) kValueDim);
        for (int64_t i = 0; i < kKeyDim; ++i) {
            q[(size_t) i] = conv[(size_t) i];
            k[(size_t) i] = conv[(size_t) (kKeyDim + i)];
        }
        for (int64_t i = 0; i < kValueDim; ++i) {
            v[(size_t) i] = conv[(size_t) (2 * kKeyDim + i)];
        }
        auto l2 = [](std::vector<float> & vec, int64_t n_heads) {
            for (int64_t h = 0; h < n_heads; ++h) {
                double ss = 0.0;
                for (int64_t i = 0; i < kHeadDim; ++i) {
                    const double t = vec[(size_t) (h * kHeadDim + i)];
                    ss += t * t;
                }
                const float scale = (float) (1.0 / std::sqrt(ss / (double) kHeadDim + kRmsEps / (double) kHeadDim)) *
                                    (1.0f / std::sqrt((float) kHeadDim));
                for (int64_t i = 0; i < kHeadDim; ++i) {
                    vec[(size_t) (h * kHeadDim + i)] *= scale;
                }
            }
        };
        l2(q, kKHeads);
        l2(k, kKHeads);

        // delta net, one token, zero state: out[j] = beta * v[j] * <k, q> / sqrt(head_dim)
        // (ggml's GATED_DELTA_NET applies that 1/sqrt(S_v) scale internally)
        const float dn_scale = 1.0f / std::sqrt((float) kHeadDim);
        std::vector<float> dn((size_t) kValueDim, 0.0f);
        for (int64_t h = 0; h < kVHeads; ++h) {
            const int64_t kh = h % kKHeads;
            double kq = 0.0;
            for (int64_t i = 0; i < kHeadDim; ++i) {
                kq += (double) k[(size_t) (kh * kHeadDim + i)] * q[(size_t) (kh * kHeadDim + i)];
            }
            for (int64_t i = 0; i < kHeadDim; ++i) {
                dn[(size_t) (h * kHeadDim + i)] = beta[(size_t) h] * v[(size_t) (h * kHeadDim + i)] * (float) kq * dn_scale;
            }
        }

        // gated rms norm per head
        std::vector<float> normed((size_t) kValueDim);
        for (int64_t h = 0; h < kVHeads; ++h) {
            double ss = 0.0;
            for (int64_t i = 0; i < kHeadDim; ++i) {
                const double t = dn[(size_t) (h * kHeadDim + i)];
                ss += t * t;
            }
            const float scale = (float) (1.0 / std::sqrt(ss / (double) kHeadDim + kRmsEps));
            for (int64_t i = 0; i < kHeadDim; ++i) {
                const float nv = dn[(size_t) (h * kHeadDim + i)] * scale * Wnorm[(size_t) i];
                const float zv = z[(size_t) (h * kHeadDim + i)];
                normed[(size_t) (h * kHeadDim + i)] = nv * (zv / (1.0f + std::exp(-zv)));
            }
        }

        const std::vector<float> y_ref = gemm(Wout, kHidden, kValueDim, normed);

        // ---- NPU ----
        Device device(0);
        Stream stream(device);
        Buffer bx((size_t) kHidden * 2);
        upload(x, bx);
        Tensor tx{ bx.data(), DType::F16, 1, kHidden };

        const int64_t qkv_dim = kKeyDim * 2 + kValueDim;
        Buffer b_qkv((size_t) qkv_dim * 2), b_z((size_t) kValueDim * 2);
        gemv(stream, Wqkv, qkv_dim, kHidden, tx, b_qkv);
        gemv(stream, Wgate, kValueDim, kHidden, tx, b_z);

        // short conv, k=0 tap only
        std::vector<float> w0((size_t) qkv_dim);
        for (int64_t c = 0; c < qkv_dim; ++c) {
            w0[(size_t) c] = Wconv[(size_t) c * kConvK + 0];
        }
        Buffer bw0((size_t) qkv_dim * 2), b_conv((size_t) qkv_dim * 2), b_silu((size_t) qkv_dim * 2);
        upload(w0, bw0);
        Tensor t_qkv{ b_qkv.data(), DType::F16, 1, qkv_dim };
        Tensor t_w0{ bw0.data(), DType::F16, 1, qkv_dim };
        Tensor t_conv{ b_conv.data(), DType::F16, 1, qkv_dim };
        Tensor t_silu{ b_silu.data(), DType::F16, 1, qkv_dim };
        mul(stream, t_qkv, t_w0, t_conv);
        silu(stream, t_conv, t_silu);
        const std::vector<float> conv_dev = to_host(stream, b_silu, qkv_dim);

        // q/k l2 norm per head, on the device
        Buffer bones((size_t) kHeadDim * 2);
        {
            std::vector<float> ones((size_t) kHeadDim, 1.0f);
            upload(ones, bones);
        }
        Tensor t_ones{ bones.data(), DType::F16, kHeadDim, 1 };
        const float l2_scale = 1.0f / std::sqrt((float) kHeadDim);

        Buffer b_q((size_t) kKeyDim * 2), b_k((size_t) kKeyDim * 2);
        std::vector<float> qn((size_t) kKeyDim), kn((size_t) kKeyDim);
        for (int64_t h = 0; h < kKHeads; ++h) {
            Buffer tmpn((size_t) kHeadDim * 2), tmps((size_t) kHeadDim * 2);
            Tensor tq{ (char *) b_silu.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            Tensor tn{ tmpn.data(), DType::F16, kHeadDim, 1 };
            Tensor ts{ tmps.data(), DType::F16, kHeadDim, 1 };
            Tensor tqn{ (char *) b_q.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            rms_norm(stream, tq, t_ones, kRmsEps / (double) kHeadDim, tn);
            muls(stream, tn, l2_scale, ts);
            muls(stream, ts, 1.0f, tqn);

            Tensor tk{ (char *) b_silu.data() + (size_t) (kKeyDim + h * kHeadDim) * 2, DType::F16, kHeadDim, 1 };
            Tensor tkn{ (char *) b_k.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            rms_norm(stream, tk, t_ones, kRmsEps / (double) kHeadDim, tn);
            muls(stream, tn, l2_scale, ts);
            muls(stream, ts, 1.0f, tkn);
        }
        qn = to_host(stream, b_q, kKeyDim);
        kn = to_host(stream, b_k, kKeyDim);

        // delta net per v head, zero state
        Buffer b_dn((size_t) kValueDim * 2);
        for (int64_t h = 0; h < kVHeads; ++h) {
            const int64_t kh = h % kKHeads;
            Buffer b_st((size_t) kHeadDim * kHeadDim * 2), b_out((size_t) kHeadDim * 2);
            Buffer b_beta(2), b_gate(2);
            std::vector<float> zeros((size_t) kHeadDim * kHeadDim, 0.0f);
            upload(zeros, b_st);
            upload({ a[(size_t) h] }, b_gate);
            upload({ beta[(size_t) h] }, b_beta);

            Tensor tq{ (char *) b_q.data() + (size_t) kh * kHeadDim * 2, DType::F16, 1, kHeadDim };
            Tensor tk{ (char *) b_k.data() + (size_t) kh * kHeadDim * 2, DType::F16, 1, kHeadDim };
            Tensor tv{ (char *) b_silu.data() + (size_t) (2 * kKeyDim + h * kHeadDim) * 2, DType::F16, 1, kHeadDim };
            Tensor tg{ b_gate.data(), DType::F16, 1, 1 };
            Tensor tb{ b_beta.data(), DType::F16, 1, 1 };
            Tensor tst{ b_st.data(), DType::F16, kHeadDim, kHeadDim };
            Tensor tout{ (char *) b_dn.data() + (size_t) h * kHeadDim * 2, DType::F16, 1, kHeadDim };
            GdnShape shape{ kHeadDim, 1, false };
            gdn(stream, tq, tk, tv, tg, tb, tst, tout, shape);
        }
        const std::vector<float> dn_dev = to_host(stream, b_dn, kValueDim);

        // gated rms norm + ssm_out
        Buffer b_normed((size_t) kValueDim * 2);
        Buffer b_wnorm((size_t) kHeadDim * 2);
        upload(Wnorm, b_wnorm);
        Tensor t_wnorm{ b_wnorm.data(), DType::F16, kHeadDim, 1 };
        for (int64_t h = 0; h < kVHeads; ++h) {
            Buffer t1((size_t) kHeadDim * 2), t2((size_t) kHeadDim * 2), t3((size_t) kHeadDim * 2);
            Tensor tin{ (char *) b_dn.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            Tensor tn{ t1.data(), DType::F16, kHeadDim, 1 };
            Tensor tz{ (char *) b_z.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            Tensor tzs{ t2.data(), DType::F16, kHeadDim, 1 };
            Tensor tm{ t3.data(), DType::F16, kHeadDim, 1 };
            Tensor tout{ (char *) b_normed.data() + (size_t) h * kHeadDim * 2, DType::F16, kHeadDim, 1 };
            rms_norm(stream, tin, t_wnorm, kRmsEps, tn);
            silu(stream, tz, tzs);
            mul(stream, tn, tzs, tm);
            muls(stream, tm, 1.0f, tout);
        }
        const std::vector<float> normed_dev = to_host(stream, b_normed, kValueDim);

        Buffer b_y((size_t) kHidden * 2);
        Tensor t_normed{ b_normed.data(), DType::F16, 1, kValueDim };
        gemv(stream, Wout, kHidden, kValueDim, t_normed, b_y);
        const std::vector<float> y_dev = to_host(stream, b_y, kHidden);

        auto max_abs = [](const std::vector<float> & lhs, const std::vector<float> & rhs) {
            double m = 0.0;
            for (size_t i = 0; i < lhs.size(); ++i) {
                m = std::max(m, std::fabs((double) lhs[i] - rhs[i]));
            }
            return m;
        };
        auto rel = [](const std::vector<float> & lhs, const std::vector<float> & rhs) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < lhs.size(); ++i) {
                num += (double) (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
                den += (double) rhs[i] * rhs[i];
            }
            return den > 0.0 ? std::sqrt(num / den) : 0.0;
        };

        printf("block  : blk.0 GDN (recurrent) layer, one token, zero state\n");
        printf("shapes : q/k %lldx%lld heads, v %lldx%lld, conv k=%lld\n\n", (long long) kKHeads,
               (long long) kHeadDim, (long long) kVHeads, (long long) kHeadDim, (long long) kConvK);

        const double d_conv = max_abs(conv_dev, conv);
        const double d_q = max_abs(qn, q);
        const double d_k = max_abs(kn, k);
        const double d_dn = max_abs(dn_dev, dn);
        const double d_norm = max_abs(normed_dev, normed);
        const double d_y = max_abs(y_dev, y_ref);
        const double r_y = rel(y_dev, y_ref);

        // The activations here are small (the short conv weights are ~1e-3), so the pass
        // criterion is relative L2, not an absolute difference.
        const double r_conv = rel(conv_dev, conv);
        const double r_q = rel(qn, q);
        const double r_k = rel(kn, k);
        const double r_dn = rel(dn_dev, dn);
        const double r_norm = rel(normed_dev, normed);

        printf("%-24s %12s %12s  %s\n", "stage", "rel L2", "max|diff|", "result");
        printf("%-24s %12.6f %12.6f  %s\n", "conv = silu(w0 * qkv)", r_conv, d_conv, r_conv <= 0.01 ? "PASS" : "FAIL");
        printf("%-24s %12.6f %12.6f  %s\n", "q after l2 norm", r_q, d_q, r_q <= 0.01 ? "PASS" : "FAIL");
        printf("%-24s %12.6f %12.6f  %s\n", "k after l2 norm", r_k, d_k, r_k <= 0.01 ? "PASS" : "FAIL");
        printf("%-24s %12.6f %12.6f  %s\n", "delta net out", r_dn, d_dn, r_dn <= 0.02 ? "PASS" : "FAIL");
        printf("%-24s %12.6f %12.6f  %s\n", "gated rms norm", r_norm, d_norm, r_norm <= 0.02 ? "PASS" : "FAIL");
        printf("%-24s %12.6f %12.6f  %s\n", "y = ssm_out", r_y, d_y, r_y <= 0.02 ? "PASS" : "FAIL");

        const bool ok = r_conv <= 0.01 && r_q <= 0.01 && r_k <= 0.01 && r_dn <= 0.02 && r_norm <= 0.02 &&
                        r_y <= 0.02;
        shutdown();
        if (!ok) {
            printf("\nGDN block differs from the CPU reference\n");
            return 1;
        }
        printf("\nthe whole GDN block of blk.0 ran on the NPU\n");
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
