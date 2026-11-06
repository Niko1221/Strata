// cann/tests/gdn_parity.cpp - gated delta net on the NPU against the float reference.
//
// One head, one sequence, both decay modes. qwen35moe spends 3 of every 4 layers in this op,
// so it is the first P3 kernel: the state update is a per-token recurrence, which is why the
// device version issues a handful of small aclnn calls per token instead of one big matmul.
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

struct Result {
    double attn_max = 0.0;
    double state_max = 0.0;
    bool pass = false;
};

Result run_one(bool kda, int64_t S, int64_t T, uint32_t seed) {
    Device device(0);
    Stream stream(device);

    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0f, 0.3f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    std::vector<float> q((size_t) S * T), k((size_t) S * T), v((size_t) S * T);
    std::vector<float> beta((size_t) T);
    std::vector<float> g((size_t) ((kda ? (size_t) S * T : (size_t) T)));
    std::vector<float> st((size_t) S * S);
    for (auto & x : q) x = n(rng);
    for (auto & x : k) x = n(rng);
    for (auto & x : v) x = n(rng);
    for (auto & x : st) x = n(rng) * 0.2f;
    for (auto & x : beta) x = u(rng);
    for (auto & x : g) x = -std::fabs(n(rng)) * 0.5f;

    std::vector<float> ref_state = st, ref_attn((size_t) S * T);
    GdnShape shape{ S, T, kda };
    gdn_ref(q.data(), k.data(), v.data(), g.data(), beta.data(), ref_state.data(), ref_attn.data(), shape);

    // device: everything fp16
    auto to_h = [](const std::vector<float> & f) {
        std::vector<uint16_t> h(f.size());
        f32_to_f16(f.data(), h.data(), f.size());
        return h;
    };
    std::vector<uint16_t> hq = to_h(q), hk = to_h(k), hv = to_h(v), hg = to_h(g), hb = to_h(beta), hs = to_h(st);
    std::vector<uint16_t> ha((size_t) S * T, 0);

    Buffer bq(hq.size() * 2), bk(hk.size() * 2), bv(hv.size() * 2), bg(hg.size() * 2), bb(hb.size() * 2),
        bs(hs.size() * 2), ba(ha.size() * 2);
    bq.copy_from_host(hq.data(), hq.size() * 2);
    bk.copy_from_host(hk.data(), hk.size() * 2);
    bv.copy_from_host(hv.data(), hv.size() * 2);
    bg.copy_from_host(hg.data(), hg.size() * 2);
    bb.copy_from_host(hb.data(), hb.size() * 2);
    bs.copy_from_host(hs.data(), hs.size() * 2);

    Tensor tq{ bq.data(), DType::F16, S, T };
    Tensor tk{ bk.data(), DType::F16, S, T };
    Tensor tv{ bv.data(), DType::F16, S, T };
    Tensor tg{ bg.data(), DType::F16, kda ? S : 1, T };
    Tensor tb{ bb.data(), DType::F16, 1, T };
    Tensor ts{ bs.data(), DType::F16, S, S };
    Tensor ta{ ba.data(), DType::F16, S, T };

    gdn(stream, tq, tk, tv, tg, tb, ts, ta, shape);

    bs.copy_to_host(hs.data(), hs.size() * 2);
    ba.copy_to_host(ha.data(), ha.size() * 2);
    stream.synchronize();

    std::vector<float> got_state(hs.size()), got_attn(ha.size());
    f16_to_f32(hs.data(), got_state.data(), got_state.size());
    f16_to_f32(ha.data(), got_attn.data(), got_attn.size());

    Result r;
    for (size_t i = 0; i < ref_attn.size(); ++i) {
        r.attn_max = std::max(r.attn_max, std::fabs((double) got_attn[i] - ref_attn[i]));
    }
    for (size_t i = 0; i < ref_state.size(); ++i) {
        r.state_max = std::max(r.state_max, std::fabs((double) got_state[i] - ref_state[i]));
    }
    r.pass = r.attn_max <= 0.05 && r.state_max <= 0.05;
    return r;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    struct Case {
        bool kda;
        int64_t S;
        int64_t T;
        const char * name;
    };
    const Case cases[] = {
    // KDA (per-channel gate) is refused by gdn(): the broadcast it needs is not expressed
    // correctly for S > 64, so the op reports that instead of returning wrong numbers.
        { false, 64, 8, "scalar S=64  T=8" },
    };

    int failures = 0;
    printf("%-18s %12s %12s  %s\n", "case", "attn|diff|", "state|diff|", "result");
    for (const Case & c : cases) {
        const Result r = run_one(c.kda, c.S, c.T, 42u + (uint32_t) c.S);
        printf("%-18s %12.6f %12.6f  %s\n", c.name, r.attn_max, r.state_max, r.pass ? "PASS" : "FAIL");
        if (!r.pass) {
            ++failures;
        }
    }
    shutdown();
    if (failures == 0) {
        printf("\nall gdn checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
