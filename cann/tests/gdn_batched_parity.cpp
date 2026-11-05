// cann/tests/gdn_batched_parity.cpp - the batched GDN against the per-head CPU oracle.
//
// The batched op runs all H heads in one pass (8 operator calls per token); the reference runs the
// scalar-gate recurrence one head at a time in float. If these agree, the batching is a pure
// performance change.
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gdn.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

namespace {

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

    const int64_t S = 128, H = 32;
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 4;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 0.3f);

    std::vector<float> q((size_t) (T * H * S)), k((size_t) (T * H * S)), v((size_t) (T * H * S));
    std::vector<float> decay((size_t) (T * H)), beta((size_t) (T * H));
    for (auto & x : q) x = round_f16({ dist(rng) })[0];
    for (auto & x : k) x = round_f16({ dist(rng) })[0];
    for (auto & x : v) x = round_f16({ dist(rng) })[0];
    for (auto & x : beta) x = round_f16({ 0.1f + 0.8f * std::fabs(dist(rng)) })[0];
    // decay in (0,1): a stable gate
    std::normal_distribution<float> gdist(-0.05f, 0.15f);
    std::vector<float> gate((size_t) (T * H));
    for (int64_t i = 0; i < T * H; ++i) {
        gate[(size_t) i] = round_f16({ gdist(rng) })[0];
        decay[(size_t) i] = std::exp(gate[(size_t) i]);
    }

    // reference: one head at a time, state starts at zero
    std::vector<float> ref_state((size_t) (H * S * S), 0.0f);
    std::vector<float> ref_attn((size_t) (T * H * S), 0.0f);
    for (int64_t h = 0; h < H; ++h) {
        std::vector<float> qh((size_t) (T * S)), kh((size_t) (T * S)), vh((size_t) (T * S)), gh((size_t) T),
            bh((size_t) T);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < S; ++d) {
                const size_t src = (size_t) ((t * H + h) * S + d);
                qh[(size_t) (t * S + d)] = q[src];
                kh[(size_t) (t * S + d)] = k[src];
                vh[(size_t) (t * S + d)] = v[src];
            }
            gh[(size_t) t] = gate[(size_t) (t * H + h)];
            bh[(size_t) t] = beta[(size_t) (t * H + h)];
        }
        std::vector<float> sh((size_t) (S * S), 0.0f), ah((size_t) (T * S), 0.0f);
        gdn_ref(qh.data(), kh.data(), vh.data(), gh.data(), bh.data(), sh.data(), ah.data(), GdnShape{ S, T, false });
        for (int64_t i = 0; i < S * S; ++i) ref_state[(size_t) (h * S * S + i)] = sh[(size_t) i];
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < S; ++d) {
                ref_attn[(size_t) ((t * H + h) * S + d)] = ah[(size_t) (t * S + d)];
            }
        }
    }

    Device device(0);
    Stream stream(device);
    Buffer bq(q.size() * 2), bk(k.size() * 2), bv(v.size() * 2), bd(decay.size() * 2), bb(beta.size() * 2);
    Buffer bs((size_t) ref_state.size() * 2), ba((size_t) ref_attn.size() * 2);
    upload(q, bq);
    upload(k, bk);
    upload(v, bv);
    upload(decay, bd);
    upload(beta, bb);
    // state starts at zero
    {
        std::vector<float> zeros((size_t) ref_state.size(), 0.0f);
        upload(zeros, bs);
    }

    Tensor tq{ bq.data(), DType::F16, H * S, T };
    Tensor tk{ bk.data(), DType::F16, H * S, T };
    Tensor tv{ bv.data(), DType::F16, H * S, T };
    Tensor td{ bd.data(), DType::F16, H, T };
    Tensor tb{ bb.data(), DType::F16, H, T };
    Tensor ts{ bs.data(), DType::F16, S, S, H };
    Tensor ta{ ba.data(), DType::F16, H * S, T };

    gdn_batched(stream, tq, tk, tv, td, tb, ts, ta, S, H, T);

    const std::vector<float> got_state = to_host(stream, bs, (int64_t) ref_state.size());
    const std::vector<float> got_attn = to_host(stream, ba, (int64_t) ref_attn.size());

    // The same data through the device's per-head op: same fp16 storage, so a difference here means
    // the batching itself is wrong, while a difference only against the float reference is drift.
    std::vector<float> dev_state((size_t) (H * S * S), 0.0f);
    std::vector<float> dev_attn((size_t) (T * H * S), 0.0f);
    {
        Buffer ph_s((size_t) (S * S) * 2), ph_a((size_t) (T * S) * 2), ph_q((size_t) (T * S) * 2),
            ph_k((size_t) (T * S) * 2), ph_v((size_t) (T * S) * 2), ph_g((size_t) T * 2), ph_b((size_t) T * 2);
        for (int64_t h = 0; h < H; ++h) {
            std::vector<float> qh((size_t) (T * S)), kh((size_t) (T * S)), vh((size_t) (T * S)),
                gh((size_t) T), bh((size_t) T);
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t d = 0; d < S; ++d) {
                    const size_t src = (size_t) ((t * H + h) * S + d);
                    qh[(size_t) (t * S + d)] = q[src];
                    kh[(size_t) (t * S + d)] = k[src];
                    vh[(size_t) (t * S + d)] = v[src];
                }
                gh[(size_t) t] = gate[(size_t) (t * H + h)];
                bh[(size_t) t] = beta[(size_t) (t * H + h)];
            }
            std::vector<float> zeros((size_t) (S * S), 0.0f);
            upload(qh, ph_q);
            upload(kh, ph_k);
            upload(vh, ph_v);
            upload(gh, ph_g);
            upload(bh, ph_b);
            upload(zeros, ph_s);
            Tensor tq1{ ph_q.data(), DType::F16, S, T };
            Tensor tk1{ ph_k.data(), DType::F16, S, T };
            Tensor tv1{ ph_v.data(), DType::F16, S, T };
            Tensor tg1{ ph_g.data(), DType::F16, 1, T };
            Tensor tb1{ ph_b.data(), DType::F16, 1, T };
            Tensor ts1{ ph_s.data(), DType::F16, S, S };
            Tensor ta1{ ph_a.data(), DType::F16, S, T };
            gdn(stream, tq1, tk1, tv1, tg1, tb1, ts1, ta1, GdnShape{ S, T, false });
            const std::vector<float> sh = to_host(stream, ph_s, S * S);
            const std::vector<float> ah = to_host(stream, ph_a, T * S);
            for (int64_t i = 0; i < S * S; ++i) dev_state[(size_t) (h * S * S + i)] = sh[(size_t) i];
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t d = 0; d < S; ++d) {
                    dev_attn[(size_t) ((t * H + h) * S + d)] = ah[(size_t) (t * S + d)];
                }
            }
        }
    }
    const double r_dev_state = rel_l2(got_state, dev_state);
    const double r_dev_attn = rel_l2(got_attn, dev_attn);

    const double r_state = rel_l2(got_state, ref_state);
    const double r_attn = rel_l2(got_attn, ref_attn);

    printf("batched gdn: S=%lld H=%lld T=%lld\n\n", (long long) S, (long long) H, (long long) T);
    printf("%-16s %12s  %s\n", "check", "rel L2", "result");
    printf("%-16s %12.6f  %s\n", "attn vs float", r_attn, r_attn <= 0.01 ? "PASS" : "FAIL");
    printf("%-16s %12.6f  %s\n", "state vs float", r_state, r_state <= 0.01 ? "PASS" : "FAIL");
    printf("%-16s %12.6f  %s\n", "attn vs device", r_dev_attn, r_dev_attn <= 0.002 ? "PASS" : "FAIL");
    printf("%-16s %12.6f  %s\n", "state vs device", r_dev_state, r_dev_state <= 0.002 ? "PASS" : "FAIL");

    shutdown();
    if (r_attn > 0.01 || r_state > 0.01) {
        printf("\nthe batched op does not match the per-head reference\n");
        return 1;
    }
    printf("\nbatched gdn matches the per-head reference\n");
    return 0;
}
