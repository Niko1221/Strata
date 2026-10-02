// src/qwen35/layers_test.cpp - Qwen35 attention, MoE and trunk sanity/parity checks.
//
//   * rope_neox against a direct rotation formula;
//   * the attention layer's FIRST token against the closed form (softmax over one cell = 1, so the output is
//     `wo @ (v * sigmoid(gate))`, independent of q/k) - a check that the q-gate split is not swapped;
//   * the MoE block with top-1 against the selected expert plus the gated shared expert;
//   * the trunk runs, its logits are finite, and the state advances between tokens.
#include "strata/qwen35/qwen35.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <vector>

namespace q = strata::qwen35;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

namespace {
using strata::core::Qwen35Geometry;

/// Owns every weight buffer; `std::deque` so a push_back never moves an existing buffer and the pointers handed
/// to the layer structs stay valid.
struct Arena {
    std::mt19937 rng{7};
    std::deque<std::vector<float>> buf;
    const float* rnd(size_t n, float s = 0.2f) {
        std::normal_distribution<float> nd(0.f, 1.f);
        buf.emplace_back(n);
        for (auto& x : buf.back()) x = s * nd(rng);
        return buf.back().data();
    }
    const float* fill(size_t n, float v) { buf.emplace_back(n, v); return buf.back().data(); }
};

Qwen35Geometry tiny() {
    Qwen35Geometry g;
    g.n_layers = 4;
    g.n_embd = 32;
    g.n_expert = 8;
    g.n_expert_used = 2;
    g.n_ff_exp = 16;
    g.n_ff_shexp = 16;
    g.full_attention_interval = 4;
    g.n_head = 4;
    g.n_head_kv = 2;
    g.head_dim = 8;
    g.rope_dim = 8;
    g.rope_freq_base = 1e7;
    g.context_length = 16;
    g.ssm_state = 8;
    g.ssm_groups = 2;
    g.ssm_dt_rank = 4;
    g.ssm_inner = 4 * 8;
    g.ssm_conv_kernel = 4;
    g.rms_eps = 1e-6f;
    g.n_vocab = 16;
    return g;
}

void matvec(const float* w, const float* x, int64_t nin, int64_t nout, std::vector<float>& y) {
    for (int64_t o = 0; o < nout; ++o) {
        double s = 0; for (int64_t i = 0; i < nin; ++i) s += (double) w[o * nin + i] * x[i];
        y[(size_t) o] = (float) s;
    }
}

}  // namespace

int main() {
    std::printf("qwen35 layers_test\n");
    Qwen35Geometry g = tiny();
    Arena a;

    // ---- rope_neox against a direct formula
    {
        const int64_t D = 8, R = 8;
        std::vector<float> v((size_t) D);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (float) (i + 1) * 0.1f;
        std::vector<float> w = v;
        const int64_t pos = 3;
        q::rope_neox(w.data(), D, R, 1e7f, pos);
        bool ok = true;
        for (int64_t k = 0; k < R / 2; ++k) {
            const double th = (double) pos * std::pow(1e7, -2.0 * k / R);
            const float c = (float) std::cos(th), s = (float) std::sin(th);
            ok = ok && std::fabs(w[(size_t) k] - (v[(size_t) k] * c - v[(size_t) (k + R / 2)] * s)) < 1e-5f;
            ok = ok && std::fabs(w[(size_t) (k + R / 2)] - (v[(size_t) k] * s + v[(size_t) (k + R / 2)] * c)) < 1e-5f;
        }
        check(ok, "rope_neox matches the direct NEOX rotation");
    }

    // ---- attention: first token's output is wo @ (v * sigmoid(gate))
    {
        q::AttnLayerWeights aw;
        aw.attn_norm = a.rnd((size_t) g.n_embd, 0.9f);
        aw.wq = a.rnd((size_t) (2 * g.n_head * g.head_dim * g.n_embd));
        aw.wk = a.rnd((size_t) (g.n_head_kv * g.head_dim * g.n_embd));
        aw.wv = a.rnd((size_t) (g.n_head_kv * g.head_dim * g.n_embd));
        aw.wo = a.rnd((size_t) (g.n_embd * g.n_head * g.head_dim));
        aw.q_norm = a.rnd((size_t) g.head_dim, 0.9f);
        aw.k_norm = a.rnd((size_t) g.head_dim, 0.9f);
        q::AttnState ast; ast.resize(g.context_length, g);
        std::vector<float> x((size_t) g.n_embd);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.05f * (float) ((i % 5) - 2);
        std::vector<float> got((size_t) g.n_embd);
        q::attn_layer(g, aw, ast, x.data(), got.data());

        std::vector<float> xn((size_t) g.n_embd), qfull((size_t) (2 * g.n_head * g.head_dim));
        std::vector<float> vf((size_t) (g.n_head_kv * g.head_dim));
        q::rms_norm(x.data(), aw.attn_norm, g.n_embd, g.rms_eps, xn.data());
        matvec(aw.wq, xn.data(), g.n_embd, 2 * g.n_head * g.head_dim, qfull);
        matvec(aw.wv, xn.data(), g.n_embd, g.n_head_kv * g.head_dim, vf);
        std::vector<float> o((size_t) (g.n_head * g.head_dim));
        for (int64_t h = 0; h < g.n_head; ++h) {
            const int64_t hkv = h / (g.n_head / g.n_head_kv);
            const float* gate = qfull.data() + h * 2 * g.head_dim + g.head_dim;
            for (int64_t d = 0; d < g.head_dim; ++d)
                o[(size_t) (h * g.head_dim + d)] = vf[(size_t) (hkv * g.head_dim + d)] * q::sigmoid(gate[d]);
        }
        std::vector<float> want((size_t) g.n_embd);
        matvec(aw.wo, o.data(), g.n_head * g.head_dim, g.n_embd, want);
        double r = 0;
        for (int64_t i = 0; i < g.n_embd; ++i) r = std::max(r, (double) std::fabs(got[(size_t) i] - want[(size_t) i]));
        check(r < 1e-4, "attention first token == wo @ (v * sigmoid(gate))");
        check(ast.n == 1, "attention appended exactly one KV cell");
    }

    // ---- MoE: top-1
    {
        Qwen35Geometry g1 = g; g1.n_expert_used = 1;
        std::vector<q::ExpertWeights> ex((size_t) g1.n_expert);
        std::vector<const float*> eg((size_t) g1.n_expert), eu((size_t) g1.n_expert), ed((size_t) g1.n_expert);
        for (int64_t e = 0; e < g1.n_expert; ++e) {
            eg[(size_t) e] = a.rnd((size_t) (g1.n_ff_exp * g1.n_embd));
            eu[(size_t) e] = a.rnd((size_t) (g1.n_ff_exp * g1.n_embd));
            ed[(size_t) e] = a.rnd((size_t) (g1.n_embd * g1.n_ff_exp));
            ex[(size_t) e] = {eg[(size_t) e], eu[(size_t) e], ed[(size_t) e]};
        }
        const float* ginp = a.rnd((size_t) (g1.n_expert * g1.n_embd));
        const float* gsh = a.rnd((size_t) (g1.n_ff_shexp * g1.n_embd));
        const float* ush = a.rnd((size_t) (g1.n_ff_shexp * g1.n_embd));
        const float* dsh = a.rnd((size_t) (g1.n_embd * g1.n_ff_shexp));
        const float* gish = a.rnd((size_t) g1.n_embd);
        q::MoeLayerWeights mw{ginp, gsh, ush, dsh, gish, ex.data()};
        std::vector<float> x((size_t) g1.n_embd);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.05f * (float) ((i % 7) - 3);
        std::vector<float> got((size_t) g1.n_embd);
        q::moe_layer(g1, mw, x.data(), got.data());

        std::vector<float> logits((size_t) g1.n_expert);
        matvec(ginp, x.data(), g1.n_embd, g1.n_expert, logits);
        const int64_t best = (int64_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::vector<float> gg((size_t) g1.n_ff_exp), uu((size_t) g1.n_ff_exp), hh((size_t) g1.n_ff_exp);
        std::vector<float> y((size_t) g1.n_embd);
        matvec(eg[(size_t) best], x.data(), g1.n_embd, g1.n_ff_exp, gg);
        matvec(eu[(size_t) best], x.data(), g1.n_embd, g1.n_ff_exp, uu);
        for (int64_t f = 0; f < g1.n_ff_exp; ++f) hh[(size_t) f] = q::silu(gg[(size_t) f]) * uu[(size_t) f];
        matvec(ed[(size_t) best], hh.data(), g1.n_ff_exp, g1.n_embd, y);
        std::vector<float> ss((size_t) g1.n_ff_shexp), su((size_t) g1.n_ff_shexp), sh((size_t) g1.n_ff_shexp);
        std::vector<float> sy((size_t) g1.n_embd);
        matvec(gsh, x.data(), g1.n_embd, g1.n_ff_shexp, ss);
        matvec(ush, x.data(), g1.n_embd, g1.n_ff_shexp, su);
        for (int64_t f = 0; f < g1.n_ff_shexp; ++f) sh[(size_t) f] = q::silu(ss[(size_t) f]) * su[(size_t) f];
        matvec(dsh, sh.data(), g1.n_ff_shexp, g1.n_embd, sy);
        double gs = 0; for (int64_t i = 0; i < g1.n_embd; ++i) gs += (double) gish[i] * x[(size_t) i];
        const float sg = q::sigmoid((float) gs);
        double r = 0;
        for (int64_t i = 0; i < g1.n_embd; ++i)
            r = std::max(r, (double) std::fabs(got[(size_t) i] - (y[(size_t) i] + sy[(size_t) i] * sg)));
        check(r < 1e-4, "MoE top-1 == the selected expert + the gated shared expert");
    }

    // ---- the trunk runs, finite, and advances
    {
        q::TrunkWeights tw;
        tw.token_embd = a.rnd((size_t) (g.n_vocab * g.n_embd));
        tw.output_norm = a.rnd((size_t) g.n_embd, 0.9f);
        tw.output = a.rnd((size_t) (g.n_vocab * g.n_embd));
        tw.attn_norm = a.fill((size_t) (g.n_layers * g.n_embd), 0.9f);
        tw.post_attn_norm = a.fill((size_t) (g.n_layers * g.n_embd), 0.9f);
        tw.gdn.resize((size_t) g.n_layers);
        tw.attn.resize((size_t) g.n_layers);
        tw.moe.resize((size_t) g.n_layers);
        std::vector<std::vector<q::ExpertWeights>> exstore((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (g.is_recurrent(l)) {
                q::GdnLayerWeights& d = tw.gdn[(size_t) l];
                d.attn_norm = a.rnd((size_t) g.n_embd, 0.9f);
                d.wqkv = a.rnd((size_t) (g.qkv_dim() * g.n_embd));
                d.wgate = a.rnd((size_t) (g.value_dim() * g.n_embd));
                d.ssm_conv = a.rnd((size_t) (g.conv_channels() * g.ssm_conv_kernel));
                d.ssm_dt = a.rnd((size_t) g.ssm_dt_rank, 0.5f);
                d.ssm_a = a.rnd((size_t) g.ssm_dt_rank, 1.0f);
                d.ssm_beta = a.rnd((size_t) (g.ssm_dt_rank * g.n_embd));
                d.ssm_alpha = a.rnd((size_t) (g.ssm_dt_rank * g.n_embd));
                d.ssm_norm = a.rnd((size_t) g.ssm_state, 0.9f);
                d.ssm_out = a.rnd((size_t) (g.n_embd * g.value_dim()));
            } else {
                q::AttnLayerWeights& d = tw.attn[(size_t) l];
                d.attn_norm = a.rnd((size_t) g.n_embd, 0.9f);
                d.wq = a.rnd((size_t) (2 * g.n_head * g.head_dim * g.n_embd));
                d.wk = a.rnd((size_t) (g.n_head_kv * g.head_dim * g.n_embd));
                d.wv = a.rnd((size_t) (g.n_head_kv * g.head_dim * g.n_embd));
                d.wo = a.rnd((size_t) (g.n_embd * g.n_head * g.head_dim));
                d.q_norm = a.rnd((size_t) g.head_dim, 0.9f);
                d.k_norm = a.rnd((size_t) g.head_dim, 0.9f);
            }
            std::vector<q::ExpertWeights>& ex = exstore[(size_t) l];
            ex.resize((size_t) g.n_expert);
            for (int64_t e = 0; e < g.n_expert; ++e)
                ex[(size_t) e] = {a.rnd((size_t) (g.n_ff_exp * g.n_embd)), a.rnd((size_t) (g.n_ff_exp * g.n_embd)),
                                  a.rnd((size_t) (g.n_embd * g.n_ff_exp))};
            tw.moe[(size_t) l] = {a.rnd((size_t) (g.n_expert * g.n_embd)), a.rnd((size_t) (g.n_ff_shexp * g.n_embd)),
                                  a.rnd((size_t) (g.n_ff_shexp * g.n_embd)), a.rnd((size_t) (g.n_embd * g.n_ff_shexp)),
                                  a.rnd((size_t) g.n_embd), ex.data()};
        }
        q::TrunkState ts; ts.reset(g);
        std::vector<float> logits((size_t) g.n_vocab), logits2((size_t) g.n_vocab);
        q::trunk_forward(g, tw, ts, 3, logits.data());
        q::trunk_forward(g, tw, ts, 5, logits2.data());
        bool finite = true;
        for (float v : logits) finite = finite && std::isfinite(v);
        check(finite, "trunk_forward produces finite logits");
        bool moved = false;
        for (int64_t i = 0; i < g.n_vocab; ++i) moved = moved || std::fabs(logits2[(size_t) i] - logits[(size_t) i]) > 1e-9f;
        check(moved, "trunk state advances between tokens");
    }

    std::printf("qwen35 layers_test: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
