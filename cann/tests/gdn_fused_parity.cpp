// cann/tests/gdn_fused_parity.cpp - CANN's fused delta-rule operator against the float reference.
//
// aclnnRecurrentGatedDeltaRule is the operator vLLM-Ascend uses for this layer (through
// torch_npu.npu_recurrent_gated_delta_rule). It is bfloat16, while this port keeps the recurrence
// in fp32 and matches the reference exactly, so before switching anything this measures what the
// bf16 path actually costs.
//
// The calling convention is not documented beyond the argument types; the shapes below follow
// vLLM-Ascend's own smoke call in vllm_ascend/ops/gdn.py and its comments.
#include <acl/acl.h>
#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>

#include <acl/acl_rt.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gdn.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

namespace {

uint16_t to_bf16_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb; // round to nearest, ties to even
    return (uint16_t) (u >> 16);
}

float from_bf16_bits(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Buf {
    Buffer b;
    explicit Buf(size_t bytes) : b(bytes) {}
    void put_bf16(const std::vector<float> & v) {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = to_bf16_bits(v[i]);
        b.copy_from_host(h.data(), h.size() * 2);
    }
    std::vector<float> get_bf16(size_t n, Stream & s) const {
        std::vector<uint16_t> h(n);
        b.copy_to_host(h.data(), n * 2);
        s.synchronize();
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = from_bf16_bits(h[i]);
        return out;
    }
    void put_f32(const std::vector<float> & v) { b.copy_from_host(v.data(), v.size() * 4); }
    void put_i32(const std::vector<int32_t> & v) { b.copy_from_host(v.data(), v.size() * 4); }
};

aclTensor * mk(void * data, std::vector<int64_t> shape, aclDataType dt) {
    // shape is given fastest-last, the way ACL wants it
    int64_t stride[4] = { 1, 1, 1, 1 };
    int64_t acc = 1;
    for (int i = (int) shape.size() - 1; i >= 0; --i) {
        stride[i] = acc;
        acc *= shape[(size_t) i];
    }
    int64_t storage = acc;
    return aclCreateTensor(shape.data(), (uint64_t) shape.size(), dt, stride, 0, ACL_FORMAT_ND, &storage, 1, data);
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    // Nk / Nv / Dk / Dv / T are settable so the same test can start from the shape vLLM-Ascend's
    // smoke call uses and grow to this model's layer.
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 1;
    const int64_t Nk = argc > 2 ? std::atoll(argv[2]) : 1;
    const int64_t Nv = argc > 3 ? std::atoll(argv[3]) : 1;
    const int64_t Dk = argc > 4 ? std::atoll(argv[4]) : 128;
    const int64_t Dv = argc > 5 ? std::atoll(argv[5]) : 128;
    const int64_t B = 1;
    const bool blocked_heads = argc > 6 ? std::atoi(argv[6]) != 0 : true;
    if (Nv % Nk != 0) {
        printf("Nv must be a multiple of Nk\n");
        return 2;
    }

    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::normal_distribution<float> gdist(-0.05f, 0.1f);
    std::vector<float> q((size_t) (T * Nk * Dk)), k((size_t) (T * Nk * Dk)), v((size_t) (T * Nv * Dv));
    std::vector<float> beta((size_t) (T * Nv)), g((size_t) (T * Nv));
    for (auto & x : q) x = dist(rng);
    for (auto & x : k) x = dist(rng);
    for (auto & x : v) x = dist(rng);
    for (auto & x : beta) x = 0.1f + 0.8f * std::fabs(dist(rng));
    for (auto & x : g) x = gdist(rng);

    // reference, one head at a time in float; state starts at zero
    std::vector<float> ref_attn((size_t) (T * Nv * Dv), 0.0f);
    std::vector<float> ref_state((size_t) (B * Nv * Dv * Dk), 0.0f);
    for (int64_t h = 0; h < Nv; ++h) {
        const int64_t hk = blocked_heads ? (h / (Nv / Nk)) : (h % Nk);
        std::vector<float> qh((size_t) (T * Dk)), kh((size_t) (T * Dk)), vh((size_t) (T * Dv)), gh((size_t) T),
            bh((size_t) T);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < Dk; ++d) {
                qh[(size_t) (t * Dk + d)] = q[(size_t) ((t * Nk + hk) * Dk + d)];
                kh[(size_t) (t * Dk + d)] = k[(size_t) ((t * Nk + hk) * Dk + d)];
            }
            for (int64_t d = 0; d < Dv; ++d) {
                vh[(size_t) (t * Dv + d)] = v[(size_t) ((t * Nv + h) * Dv + d)];
            }
            gh[(size_t) t] = g[(size_t) (t * Nv + h)];
            bh[(size_t) t] = beta[(size_t) (t * Nv + h)];
        }
        std::vector<float> sh((size_t) (Dv * Dk), 0.0f), ah((size_t) (T * Dv), 0.0f);
        gdn_ref(qh.data(), kh.data(), vh.data(), gh.data(), bh.data(), sh.data(), ah.data(),
                GdnShape{ Dk, T, false });
        for (int64_t i = 0; i < Dv * Dk; ++i) ref_state[(size_t) (h * Dv * Dk + i)] = sh[(size_t) i];
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < Dv; ++d) {
                ref_attn[(size_t) ((t * Nv + h) * Dv + d)] = ah[(size_t) (t * Dv + d)];
            }
        }
    }

    Device device(0);
    Stream stream(device);
    Buf bq((size_t) (T * Nk * Dk) * 2), bk((size_t) (T * Nk * Dk) * 2), bv((size_t) (T * Nv * Dv) * 2);
    Buf bbeta((size_t) (T * Nv) * 2), bg((size_t) (T * Nv) * 4), bgk((size_t) (T * Nv) * 4);
    Buf bstate((size_t) (B * Nv * Dv * Dk) * 2), bout((size_t) (T * Nv * Dv) * 2);
    Buf blen((size_t) B * 4), bidx((size_t) T * 4), bacc(4);
    bq.put_bf16(q);
    bk.put_bf16(k);
    bv.put_bf16(v);
    bbeta.put_bf16(beta);
    bg.put_f32(g);
    bgk.put_f32(g); // the second gate: same values here, only its presence is being tested
    std::vector<float> zeros((size_t) (B * Nv * Dv * Dk), 0.0f);
    bstate.put_bf16(zeros);
    blen.put_i32(std::vector<int32_t>((size_t) B, (int32_t) T));
    bidx.put_i32(std::vector<int32_t>((size_t) T, 0));
    bacc.put_i32(std::vector<int32_t>{ 1 });

    std::vector<int64_t> qs{ T, Nk, Dk }, ks{ T, Nk, Dk }, vs{ T, Nv, Dv }, bs{ T, Nv }, ss{ B, Nv, Dv, Dk },
        os{ T, Nv, Dv }, ls{ B }, is{ T }, as{ 1 };
    aclTensor * tq = mk(bq.b.data(), qs, ACL_BF16);
    aclTensor * tk = mk(bk.b.data(), ks, ACL_BF16);
    aclTensor * tv = mk(bv.b.data(), vs, ACL_BF16);
    aclTensor * tb = mk(bbeta.b.data(), bs, ACL_BF16);
    aclTensor * tst = mk(bstate.b.data(), ss, ACL_BF16);
    aclTensor * tl = mk(blen.b.data(), ls, ACL_INT32);
    aclTensor * ti = mk(bidx.b.data(), is, ACL_INT32);
    aclTensor * tg = mk(bg.b.data(), bs, ACL_FLOAT);
    aclTensor * tgk = mk(bgk.b.data(), bs, ACL_FLOAT);
    aclTensor * tacc = mk(bacc.b.data(), as, ACL_INT32);
    aclTensor * to = mk(bout.b.data(), os, ACL_BF16);

    uint64_t ws_size = 0;
    aclOpExecutor * ex = nullptr;
    const float scale = 1.0f / std::sqrt((float) Dk);
    aclnnStatus st = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(tq, tk, tv, tb, tst, tl, ti, tg, tgk, tacc, scale,
                                                                  to, &ws_size, &ex);
    if (st != ACL_SUCCESS) {
        printf("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed: %d\n", (int) st);
        return 1;
    }
    void * ws = nullptr;
    if (ws_size > 0) {
        aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    st = aclnnRecurrentGatedDeltaRule(ws, ws_size, ex, (aclrtStream) stream.handle());
    if (st != ACL_SUCCESS) {
        printf("aclnnRecurrentGatedDeltaRule failed: %d\n", (int) st);
        return 1;
    }
    stream.synchronize();

    const std::vector<float> got_attn = bout.get_bf16((size_t) (T * Nv * Dv), stream);
    const std::vector<float> got_state = bstate.get_bf16((size_t) (B * Nv * Dv * Dk), stream);

    // the operator keeps the state as [Dv][Dk] (S, not its transpose), the reference as [Dk][Dv]
    std::vector<float> ref_state_t((size_t) (B * Nv * Dv * Dk));
    for (int64_t h = 0; h < Nv; ++h) {
        for (int64_t i = 0; i < Dv; ++i) {
            for (int64_t j = 0; j < Dk; ++j) {
                // ref_state[h][j][i] is S[i][j]; the op wants [i][j]
                ref_state_t[(size_t) ((h * Dv + i) * Dk + j)] = ref_state[(size_t) ((h * Dk + j) * Dv + i)];
            }
        }
    }
    // the two candidate layouts for the returned state
    std::vector<float> ref_state_same((size_t) (B * Nv * Dv * Dk));
    for (int64_t h = 0; h < Nv; ++h) {
        for (int64_t j = 0; j < Dk; ++j) {
            for (int64_t i = 0; i < Dv; ++i) {
                ref_state_same[(size_t) ((h * Dk + j) * Dv + i)] = ref_state[(size_t) ((h * Dk + j) * Dv + i)];
            }
        }
    }
    const double ra = rel_l2(got_attn, ref_attn);
    const double rs = rel_l2(got_state, ref_state_t);
    const double rs_same = rel_l2(got_state, ref_state_same);

    printf("fused GDN: T=%lld Nk=%lld Nv=%lld Dk=%lld Dv=%lld\n\n", (long long) T, (long long) Nk, (long long) Nv,
           (long long) Dk, (long long) Dv);
    printf("%-24s %12s\n", "check", "rel L2");
    printf("%-24s %12.6f  %s\n", "attn (fused bf16 vs float)", ra, ra <= 0.02 ? "PASS" : "FAIL");
    printf("%-24s %12.6f  %s\n", "state [Dv][Dk] vs float", rs, rs <= 0.02 ? "PASS" : "FAIL");
    printf("%-24s %12.6f  %s\n", "state [Dk][Dv] vs float", rs_same, rs_same <= 0.02 ? "PASS" : "FAIL");
    printf("\nfirst state values  fused %.6f %.6f   ref(S) %.6f %.6f   ref(S^T) %.6f %.6f\n", (double) got_state[0],
           (double) got_state[1], (double) ref_state_same[0], (double) ref_state_same[1], (double) ref_state_t[0],
           (double) ref_state_t[1]);
    printf("first attn values   fused %.6f   ref %.6f\n", (double) got_attn[0], (double) ref_attn[0]);

    shutdown();
    return 0;
}
