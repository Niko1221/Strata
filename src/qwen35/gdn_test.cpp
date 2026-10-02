// src/qwen35/gdn_test.cpp - the Qwen35 gated delta-net layer against an independent reference.
//
// The implementation (`qwen35.cpp`) uses ggml's transposed state layout `M[j*S + i]`; the reference below uses
// the natural `S[i][j]` layout and float64 arithmetic, so a mistake in the transposition, the head mapping, the
// conv tap order or the gate does not agree between them.  The recurrence itself is
// `ggml_compute_forward_gated_delta_net_one_chunk` (the pinned llama.cpp CPU op), read there and re-derived here.
//
//     ./gdn_test
#include "strata/qwen35/qwen35.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

namespace q = strata::qwen35;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

namespace {

q::Mat fmat(const std::vector<float>& v, int64_t nin, int64_t nout) {
    return {v.data(), nullptr, 0, nin, nout, (size_t) nin * sizeof(float)};
}

struct Weights {
    std::vector<float> attn_norm, wqkv, wgate, ssm_conv, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out;
    q::GdnLayerWeights view(const strata::core::Qwen35Geometry& g) const {
        q::GdnLayerWeights w;
        w.attn_norm = attn_norm.data();
        w.wqkv = fmat(wqkv, g.n_embd, g.qkv_dim());
        w.wgate = fmat(wgate, g.n_embd, g.value_dim());
        w.ssm_conv = ssm_conv.data();
        w.ssm_dt = ssm_dt.data();
        w.ssm_a = ssm_a.data();
        w.ssm_beta = fmat(ssm_beta, g.n_embd, g.ssm_dt_rank);
        w.ssm_alpha = fmat(ssm_alpha, g.n_embd, g.ssm_dt_rank);
        w.ssm_norm = ssm_norm.data();
        w.ssm_out = fmat(ssm_out, g.value_dim(), g.n_embd);
        return w;
    }
};

Weights make_weights(const strata::core::Qwen35Geometry& g, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    auto v = [&](size_t n, float scale = 0.1f) {
        std::vector<float> r(n);
        for (auto& x : r) x = scale * nd(rng);
        return r;
    };
    Weights w;
    w.attn_norm = v(g.n_embd, 0.9f);
    w.wqkv = v((size_t) g.qkv_dim() * g.n_embd);
    w.wgate = v((size_t) g.value_dim() * g.n_embd);
    w.ssm_conv = v((size_t) g.conv_channels() * g.ssm_conv_kernel);
    w.ssm_dt = v(g.ssm_dt_rank, 0.5f);
    // ssm_a is negative in the real model (the decay); keep it negative so exp(gate) < 1 like a real run.
    w.ssm_a = v(g.ssm_dt_rank, 1.0f);
    for (auto& x : w.ssm_a) x = -std::fabs(x);
    w.ssm_beta = v((size_t) g.ssm_dt_rank * g.n_embd);
    w.ssm_alpha = v((size_t) g.ssm_dt_rank * g.n_embd);
    w.ssm_norm = v(g.ssm_state, 0.9f);
    w.ssm_out = v((size_t) g.n_embd * g.value_dim());
    return w;
}

// The independent reference: natural [i][j] state, float64.
struct RefState { std::vector<double> conv, rec; };

void ref_layer(const strata::core::Qwen35Geometry& g, const Weights& w, RefState& st, const float* x, float* out) {
    const int64_t H = g.n_embd, d_conv = g.ssm_conv_kernel;
    const int64_t Hk = g.ssm_groups, Hv = g.ssm_dt_rank, S = g.ssm_state;
    const int64_t key_dim = g.key_dim(), value_dim = g.value_dim(), qkv_dim = g.qkv_dim(), C = g.conv_channels();
    const double eps = g.rms_eps;

    auto rms = [&](const auto* xx, const float* ww, int64_t n, double* o) {
        double ss = 0; for (int64_t i = 0; i < n; ++i) ss += (double) xx[i] * xx[i];
        const double r = 1.0 / std::sqrt(ss / n + eps);
        for (int64_t i = 0; i < n; ++i) o[i] = xx[i] * r * (ww ? ww[i] : 1.0);
    };
    auto dot = [&](const float* row, const double* xx, int64_t n) {
        double s = 0; for (int64_t i = 0; i < n; ++i) s += (double) row[i] * xx[i]; return s;
    };
    auto l2 = [&](const double* xx, int64_t n, double* o) {
        double ss = 0; for (int64_t i = 0; i < n; ++i) ss += xx[i] * xx[i];
        const double r = 1.0 / std::sqrt(ss + eps);
        for (int64_t i = 0; i < n; ++i) o[i] = xx[i] * r;
    };

    std::vector<double> xn(H), qkv(qkv_dim), z(value_dim), beta(Hv), gate(Hv);
    rms(x, w.attn_norm.data(), H, xn.data());
    for (int64_t o = 0; o < qkv_dim; ++o) qkv[o] = dot(w.wqkv.data() + o * H, xn.data(), H);
    for (int64_t o = 0; o < value_dim; ++o) z[o] = dot(w.wgate.data() + o * H, xn.data(), H);
    for (int64_t h = 0; h < Hv; ++h) {
        beta[h] = 1.0 / (1.0 + std::exp(-dot(w.ssm_beta.data() + h * H, xn.data(), H)));
        const double a = dot(w.ssm_alpha.data() + h * H, xn.data(), H) + w.ssm_dt[h];
        gate[h] = std::log1p(std::exp(a)) * w.ssm_a[h];
    }
    std::vector<double> hh(C);
    for (int64_t c = 0; c < C; ++c) {
        double s = qkv[c] * w.ssm_conv[c * d_conv + (d_conv - 1)];
        for (int64_t j = 0; j < d_conv - 1; ++j) s += st.conv[j * C + c] * w.ssm_conv[c * d_conv + j];
        hh[c] = s / (1.0 + std::exp(-s));
    }
    // shift the conv history and append qkv
    std::vector<double> nc((size_t) (d_conv - 1) * C);
    for (int64_t j = 0; j < d_conv - 2; ++j)
        for (int64_t c = 0; c < C; ++c) nc[j * C + c] = st.conv[(j + 1) * C + c];
    if (d_conv >= 2)
        for (int64_t c = 0; c < C; ++c) nc[(d_conv - 2) * C + c] = qkv[c];
    st.conv = nc;

    std::vector<double> qn(key_dim), kn(key_dim);
    for (int64_t hh2 = 0; hh2 < Hk; ++hh2) {
        l2(&hh[hh2 * S], S, &qn[hh2 * S]);
        l2(&hh[key_dim + hh2 * S], S, &kn[hh2 * S]);
    }

    std::vector<double> o(value_dim), y(value_dim), tmp(S);
    const double scale = 1.0 / std::sqrt((double) S);
    for (int64_t hv = 0; hv < Hv; ++hv) {
        const int64_t hk = hv % Hk;
        const double* qd = &qn[hk * S];
        const double* kd = &kn[hk * S];
        const double* vd = &hh[2 * key_dim + hv * S];
        double* M = &st.rec[(size_t) hv * S * S];   // natural [i][j]
        const double decay = std::exp(gate[hv]);
        std::vector<double> delta(S);
        for (int64_t i = 0; i < S; ++i) for (int64_t j = 0; j < S; ++j) M[i * S + j] *= decay;
        for (int64_t j = 0; j < S; ++j) {
            double sk = 0;
            for (int64_t i = 0; i < S; ++i) sk += M[i * S + j] * kd[i];
            delta[j] = (vd[j] - sk) * beta[hv];
        }
        for (int64_t i = 0; i < S; ++i) for (int64_t j = 0; j < S; ++j) M[i * S + j] += kd[i] * delta[j];
        for (int64_t j = 0; j < S; ++j) {
            double oo = 0;
            for (int64_t i = 0; i < S; ++i) oo += M[i * S + j] * qd[i];
            o[hv * S + j] = oo * scale;
        }
    }
    for (int64_t hv = 0; hv < Hv; ++hv) {
        rms(&o[hv * S], w.ssm_norm.data(), S, tmp.data());
        for (int64_t j = 0; j < S; ++j) y[hv * S + j] = tmp[j] * (z[hv * S + j] / (1.0 + std::exp(-z[hv * S + j])));
    }
    for (int64_t r = 0; r < H; ++r) out[r] = (float) dot(w.ssm_out.data() + r * value_dim, y.data(), value_dim);
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

}  // namespace

int main() {
    std::printf("qwen35 gdn_test\n");
    strata::core::Qwen35Geometry g;
    g.n_embd = 32;
    g.ssm_state = 8;
    g.ssm_groups = 2;
    g.ssm_dt_rank = 4;
    g.ssm_inner = 4 * 8;          // value_dim = v_heads * head_v_dim
    g.ssm_conv_kernel = 4;
    g.rms_eps = 1e-6f;

    std::mt19937 rng(12345);
    Weights w = make_weights(g, rng);
    q::GdnLayerWeights vw = w.view(g);

    // One token, fresh state.
    {
        std::vector<float> x((size_t) g.n_embd), got((size_t) g.n_embd), want((size_t) g.n_embd);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& t : x) t = nd(rng);
        q::GdnState st; st.resize(g);
        RefState rs; rs.conv.assign((size_t) (g.ssm_conv_kernel - 1) * g.conv_channels(), 0.0);
        rs.rec.assign((size_t) g.ssm_dt_rank * g.ssm_state * g.ssm_state, 0.0);
        q::gdn_layer(g, vw, st, x.data(), got.data());
        ref_layer(g, w, rs, x.data(), want.data());
        check(rel(got, want) < 1e-5, "one token, fresh state, matches the float64 reference");
    }

    // Eight tokens: the conv history and the recurrent state must both carry over correctly.
    {
        q::GdnState st; st.resize(g);
        RefState rs; rs.conv.assign((size_t) (g.ssm_conv_kernel - 1) * g.conv_channels(), 0.0);
        rs.rec.assign((size_t) g.ssm_dt_rank * g.ssm_state * g.ssm_state, 0.0);
        std::normal_distribution<float> nd(0.f, 1.f);
        double worst = 0.0;
        for (int t = 0; t < 8; ++t) {
            std::vector<float> x((size_t) g.n_embd), got((size_t) g.n_embd), want((size_t) g.n_embd);
            for (auto& q2 : x) q2 = nd(rng);
            q::gdn_layer(g, vw, st, x.data(), got.data());
            ref_layer(g, w, rs, x.data(), want.data());
            worst = std::max(worst, rel(got, want));
        }
        check(worst < 1e-5, "eight tokens: conv history and recurrent state carry over");
    }

    // zero() returns the state to the fresh-sequence reference.
    {
        q::GdnState a; a.resize(g), a.zero();
        q::GdnState b; b.resize(g);
        check(a.conv == b.conv && a.rec == b.rec, "GdnState::zero equals a fresh state");
    }

    // The primitives.
    {
        const float x[4] = {1.f, 2.f, 3.f, 4.f};
        float y[4];
        q::rms_norm(x, nullptr, 4, 1e-6f, y);
        const double ss = 1 + 4 + 9 + 16;
        const double r = 1.0 / std::sqrt(ss / 4 + 1e-6);
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && std::fabs(y[i] - x[i] * r) < 1e-6;
        check(ok, "rms_norm == x / sqrt(mean(x^2) + eps)");
        q::l2_norm(x, 4, 1e-6f, y);
        const double r2 = 1.0 / std::sqrt(ss + 1e-6);
        ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && std::fabs(y[i] - x[i] * r2) < 1e-6;
        check(ok, "l2_norm == x / sqrt(sum(x^2) + eps)");
    }

    std::printf("qwen35 gdn_test: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
