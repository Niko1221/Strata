// engine/src/qwen35/memt_kernels_cpu_test.cpp - the MemT CUDA kernels on the CPU.
//
// memt_kernels_impl.cuh is compiled a second time with MEMT_HOST_TEST: every
// kernel runs with one simulated thread per block, so each `for (d = tid; ...)`
// loop walks its whole row and block_sum reduces a single value.  The outputs
// are compared against memt_math.h's scalar reference, which pins the released
// math - this is the index math check (proj's wide row, att's H*D row, the
// [h][k][v] state) that a GPU would otherwise be the first to see.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#define MEMT_HOST_TEST 1
#include "memt_kernels_impl.cuh"
#include "memt_math.h"

namespace M = strata::qwen35::memt;
namespace D = strata::qwen35::memt::device;
namespace H = strata::qwen35::memt::memt_host;

namespace {

int g_checks = 0;
int g_fail = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_fail;
    }
}

std::vector<float> values(size_t count, int seed) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = (float) ((int) ((i * 37 + (size_t) seed) % 29) - 14) / 14.0f;
    return out;
}

bool close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

// One host launch of a grid of `blocks` single-thread blocks.
template <class F>
void launch(int blocks, F fn) {
    H::g_nthread = 1;
    H::g_tid = 0;
    for (int b = 0; b < blocks; ++b) {
        H::g_bid = b;
        fn();
    }
}

}  // namespace

int main() {
    const int heads = 2, dim = 3, rank = 2, tokens = 3;
    const int hd = heads * dim;
    const int64_t stride = 5 * hd + 2 * rank;
    const float eps = 1e-6f, softcap = 2.0f;

    // The in_proj output, laid out as the engine leaves it: [tokens][stride].
    std::vector<float> proj = values((size_t) tokens * (size_t) stride, 5);
    std::vector<float> og = values((size_t) tokens * hd, 21);      // g_down's result
    std::vector<float> fdown = values((size_t) tokens * hd, 31);   // f_down's result (before the bias)
    std::vector<float> bias = values((size_t) hd, 41);
    std::vector<float> scale_log = values((size_t) heads, 47);
    std::vector<float> o_norm = values((size_t) dim, 53);
    std::vector<float> state0 = values((size_t) heads * dim * dim, 59);
    std::vector<float> att = values((size_t) tokens * hd, 61);

    // ---- gates
    std::vector<float> decay((size_t) tokens * hd, 0.0f);
    launch((int) (tokens * hd), [&] { D::k_gates(fdown.data(), bias.data(), scale_log.data(), decay.data(),
                                                 tokens * hd, heads, dim); });
    bool gates_ok = true;
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < dim; ++d) {
                const size_t i = (size_t) t * hd + (size_t) h * dim + d;
                const M::Gates g = M::token_gates(proj[(size_t) t * stride + 3 * hd + h * dim + d],
                                                  proj[(size_t) t * stride + 4 * hd + h * dim + d],
                                                  fdown[i] + bias[(size_t) h * dim + d], scale_log[(size_t) h]);
                gates_ok = gates_ok && std::fabs(decay[i] - g.log_decay) < 1e-6f;
            }
    check(gates_ok, "gates: log_decay matches memt_math");

    // ---- latents: the two gate latents out of proj's wide rows
    {
        std::vector<float> dlat((size_t) tokens * rank, 0.0f), olat((size_t) tokens * rank, 0.0f);
        launch(tokens * rank, [&] { D::k_latents(proj.data(), dlat.data(), olat.data(), tokens, stride, hd, rank); });
        bool ok = true;
        for (int t = 0; t < tokens; ++t)
            for (int r = 0; r < rank; ++r) {
                const size_t i = (size_t) t * rank + r;
                ok = ok && dlat[i] == proj[(size_t) t * stride + 5 * hd + r];
                ok = ok && olat[i] == proj[(size_t) t * stride + 5 * hd + rank + r];
            }
        check(ok, "latents: decay_latent at 5*H*D, out_gate_latent at 5*H*D+R, every row");
    }

    // ---- key_effective
    std::vector<float> k_eff((size_t) tokens * hd, 0.0f);
    std::vector<float> shared((size_t) (2 * dim + 1), 0.0f);
    H::g_shared = shared.data();
    launch(tokens * heads, [&] { D::k_key(proj.data(), k_eff.data(), heads, dim, stride); });
    bool key_ok = true;
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h) {
            std::vector<float> k((size_t) dim);
            for (int d = 0; d < dim; ++d) k[(size_t) d] = proj[(size_t) t * stride + hd + h * dim + d];
            std::vector<float> want = k;
            M::l2norm_in_place(want.data(), dim);
            for (int d = 0; d < dim; ++d) {
                const float erase_sqrt = std::exp(0.5f * M::logsigmoid(proj[(size_t) t * stride + 3 * hd + h * dim + d]));
                want[(size_t) d] *= erase_sqrt;
                const size_t i = (size_t) t * hd + (size_t) h * dim + d;
                key_ok = key_ok && std::fabs(k_eff[i] - want[(size_t) d]) < 1e-6f;
            }
        }
    check(key_ok, "key_effective: l2norm(k) * erase, with the H*D row (not the wide proj row)");

    // ---- read (and its addition into att)
    std::vector<float> att_got = att;
    H::g_shared = shared.data();
    launch(tokens * heads, [&] { D::k_read(proj.data(), og.data(), o_norm.data(), state0.data(), att_got.data(),
                                            heads, dim, stride, eps, softcap); });
    bool read_ok = true;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> q = values((size_t) hd, 0);
        for (int i = 0; i < hd; ++i) q[(size_t) i] = proj[(size_t) t * stride + i];
        for (int h = 0; h < heads; ++h) M::l2norm_in_place(q.data() + h * dim, dim);
        std::vector<float> raw((size_t) hd);
        M::memory_read(q.data(), state0.data(), raw.data(), heads, dim);
        std::vector<float> want((size_t) hd);
        M::gated_rmsnorm(raw.data(), og.data() + (size_t) t * hd, o_norm.data(), want.data(), heads, dim, eps, softcap);
        for (int i = 0; i < hd; ++i)
            read_ok = read_ok && std::fabs(att_got[(size_t) t * hd + i] - (att[(size_t) t * hd + i] + want[(size_t) i])) < 1e-6f;
    }
    check(read_ok, "read: l2norm(q) @ state, gated RMSNorm, added to att's H*D row");

    // A different state must give a different read (the state is indexed per head).
    std::vector<float> other = state0;
    for (int h = 0; h < heads; ++h) other[(size_t) h * dim * dim] += 1.0f;
    std::vector<float> att_other = att;
    H::g_shared = shared.data();
    launch(tokens * heads, [&] { D::k_read(proj.data(), og.data(), o_norm.data(), other.data(), att_other.data(),
                                            heads, dim, stride, eps, softcap); });
    check(!close(att_got, att_other, 1e-4f), "read: each head reads its own state");

    // ---- update
    std::vector<float> state = state0;
    H::g_shared = shared.data();
    launch(heads, [&] { D::k_update(proj.data(), k_eff.data(), decay.data(), state.data(), tokens, heads, dim, stride); });
    std::vector<float> want_state = state0;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> v((size_t) hd), write((size_t) hd);
        for (int i = 0; i < hd; ++i) {
            v[(size_t) i] = proj[(size_t) t * stride + 2 * hd + i];
            write[(size_t) i] = 1.0f / (1.0f + std::exp(-proj[(size_t) t * stride + 4 * hd + i]));
        }
        M::memory_update(want_state.data(), k_eff.data() + (size_t) t * hd, v.data(), decay.data() + (size_t) t * hd,
                         write.data(), heads, dim);
    }
    check(close(state, want_state, 1e-5f), "update: decayed state + k_eff (write*v - kv), token by token");
    check(!close(state, state0, 1e-6f), "update moves the state");
    // the state must be the same when the kernel sees a different proj row stride
    // (the wide row is the only place stride enters)
    {
        std::vector<float> wide((size_t) tokens * (size_t) (stride + 5), 0.0f);
        for (int t = 0; t < tokens; ++t)
            for (int64_t i = 0; i < stride; ++i) wide[(size_t) t * (size_t) (stride + 5) + (size_t) i] =
                proj[(size_t) t * (size_t) stride + (size_t) i];
        std::vector<float> s2 = state0;
        std::vector<float> k2((size_t) tokens * hd);
        H::g_shared = shared.data();
        launch(tokens * heads, [&] { D::k_key(wide.data(), k2.data(), heads, dim, stride + 5); });
        H::g_shared = shared.data();
        launch(heads, [&] { D::k_update(wide.data(), k2.data(), decay.data(), s2.data(), tokens, heads, dim, stride + 5); });
        check(close(s2, want_state, 1e-5f), "update: a wider proj stride reads the same rows");
    }

    // ---- a query is the commit's read: same att, untouched state
    {
        std::vector<float> s_query = state0;
        std::vector<float> att_q = att;
        H::g_shared = shared.data();
        launch(tokens * heads, [&] { D::k_read(proj.data(), og.data(), o_norm.data(), s_query.data(), att_q.data(),
                                                heads, dim, stride, eps, softcap); });
        check(close(att_q, att_got, 1e-6f) && s_query == state0, "read leaves the state unchanged");
    }

    if (g_fail == 0) std::printf("memt_kernels_cpu_test: %d checks passed\n", g_checks);
    return g_fail == 0 ? 0 : 1;
}
