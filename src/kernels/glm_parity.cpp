// src/kernels/glm_parity.cpp - the parity test for GLM-5.3-Flash's own kernels (mHC, KDA, MLA, the SwiGLU
// clamp).
//
// Every function here is a transcription of the REFERENCE, not of the kernel: the arithmetic is in double and
// laid out the way `ggml_compute_forward_hc_pre_f32` / `ggml_compute_forward_kda_f32` /
// `ggml_compute_forward_ssm_conv_f32` in ik_llama.cpp write it.  The kernel is judged against that, so a
// misreading of the reference shows up as a disagreement rather than as two copies of the same mistake.
//
// **EACH CASE ALSO ASSERTS A RIVAL READING, AND THE RIVAL MUST DIFFER.**  This model is a stack of choices that
// all produce well-formed output: `post` with and without its factor 2, the Sinkhorn loop ending on a row or on
// a column, `comb` read as source-dest or dest-source, the L2 norm as a floor or as `+eps` inside the sum, the
// decay before or after the state write.  A test that only checks "the kernel equals my transcription" cannot
// tell a correct transcription from a confident one - so each rival is computed here and required to disagree.
//
// The layout is the second thing under test.  `state` is [row, col] with the ROW fastest, `comb` is
// `comb[j*S + i]` with j the SOURCE, and `g` is [head_dim, head_count, T] with head_dim fastest.  All three are
// silent when wrong and this file fills every buffer with a value that encodes its own coordinates.
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_dsa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using strata::kernels::GlmHcShapes;
using strata::kernels::GlmKdaShapes;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    template <typename T>
    T* put(const std::vector<T>& v) {
        T* p = nullptr;
        check(cudaMalloc(&p, v.size() * sizeof(T)), "malloc");
        if (!v.empty()) check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
        return p;
    }
    template <typename T>
    std::vector<T> get(const T* p, size_t n) {
        std::vector<T> v(n);
        if (n) check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return v;
    }
};

int failures = 0;
int cases = 0;

void report(const char* name, bool ok, const char* note = "") {
    ++cases;
    std::printf("  %-46s %s%s%s\n", name, ok ? "ok" : "*** NO ***", note[0] ? "  " : "", note);
    if (!ok) ++failures;
}

/// A tolerance in f32: the kernel sums in f32 where the reference sums in double, so the two agree to roughly
/// the accumulated rounding of one reduction, not to the bit.  1e-5 relative is ~1000x a single f32 ulp and
/// still 100x tighter than any of the rival readings below, which is the gap this file is built to sit in.
bool close_enough(const std::vector<float>& got, const std::vector<double>& want, double rel = 1e-5) {
    double peak = 0.0;
    for (double w : want) peak = std::max(peak, std::fabs(w));
    const double tol = rel * std::max(peak, 1e-6) + 1e-7;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs((double) got[i] - want[i]) > tol) return false;
    return true;
}

/// Is the rival reading FARTHER from the kernel than f32 noise?
///
/// **NOT AN EXACT INEQUALITY.**  The Sinkhorn block below converges: at 20 iterations two orderings of the
/// same averaging steps land within 1e-9 of each other, and both round to the same f32, so an `!=` test would
/// report "the kernel agrees with the rival" for a rival that is genuinely indistinguishable AT THAT ITERATION
/// COUNT.  The claim this file makes instead is the one it can support - the rival is 100x farther away than
/// the kernel's own f32 error - and the cases where it is not are run at an iteration count where it is.
bool differs(const std::vector<float>& got, const std::vector<double>& rival, double rel = 1e-4) {
    double peak = 0.0;
    for (double w : rival) peak = std::max(peak, std::fabs(w));
    const double tol = rel * std::max(peak, 1e-6);
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs((double) got[i] - rival[i]) > tol) return true;
    return false;
}

/// The relative gap between two DOUBLE computations - the systematic difference with the kernel's own f32
/// rounding removed.
///
/// **THIS IS THE RIGHT MEASURE FOR A RIVAL SMALLER THAN A F32 ULP.**  Two of the Sinkhorn rivals below (the
/// eps that the reference adds on its first column pass, and the row/column order at 20 iterations where the
/// iteration has converged) differ from the reference by ~1e-7 - real, and *below* the ~1e-6 that the kernel's
/// own 39 f32 normalisations accumulate.  Comparing the kernel to such a rival measures the rounding, not the
/// rival, and would report "indistinguishable" for a computation that is different.  Measured in double, the
/// rival's size is visible and the honest claim can be made: a different computation whose effect this kernel
/// cannot resolve, and therefore cannot get wrong in a way that matters.
double rel_gap(const std::vector<double>& a, const std::vector<double>& b) {
    double peak = 0.0, worst = 0.0;
    for (double v : b) peak = std::max(peak, std::fabs(v));
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst / std::max(peak, 1e-12);
}

// ---------------------------------------------------------------------------------------------------------
// mHC
// ---------------------------------------------------------------------------------------------------------

/// `ggml_compute_forward_hc_pre_f32`, per token.  The three rival switches are the three the header warns
/// about, and `col_pass` also covers whether the loop ends on a row or on a column.
void ref_hc_pre(const std::vector<float>& mixes, const std::vector<float>& scale, const std::vector<float>& base,
                int S, int iters, double eps, int64_t T, std::vector<double>& pre, std::vector<double>& post,
                std::vector<double>& comb, bool two_post, bool row_last, bool eps_only_first) {
    pre.assign((size_t) T * S, 0.0);
    post.assign((size_t) T * S, 0.0);
    comb.assign((size_t) T * S * S, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        const float* x = mixes.data() + t * (S * S + 2 * S);
        for (int i = 0; i < S; ++i) {
            pre[(size_t) t * S + i] = 1.0 / (1.0 + std::exp(-((double) x[i] * scale[0] + base[i]))) + eps;
            const double v = 1.0 / (1.0 + std::exp(-((double) x[S + i] * scale[1] + base[S + i])));
            post[(size_t) t * S + i] = (two_post ? 2.0 : 1.0) * v;
        }
        double m[64];
        for (int i = 0; i < S * S; ++i) m[i] = (double) x[2 * S + i] * scale[2] + base[2 * S + i];
        for (int r = 0; r < S; ++r) {
            double mx = m[r * S];
            for (int c = 1; c < S; ++c) mx = std::max(mx, m[r * S + c]);
            double sum = 0.0;
            for (int c = 0; c < S; ++c) { m[r * S + c] = std::exp(m[r * S + c] - mx); sum += m[r * S + c]; }
            for (int c = 0; c < S; ++c) m[r * S + c] = m[r * S + c] / sum + eps;
        }
        // The reference does the column pass first and then `iters - 1` (row, column) rounds.  `row_last`
        // reverses the order of the pair inside that loop, which leaves the shape and the magnitudes intact.
        auto col_pass = [&](bool first) {
            for (int c = 0; c < S; ++c) {
                double sum = (first && !eps_only_first) ? 0.0 : eps;
                for (int r = 0; r < S; ++r) sum += m[r * S + c];
                for (int r = 0; r < S; ++r) m[r * S + c] /= sum;
            }
        };
        col_pass(true);
        for (int it = 0; it < iters - 1; ++it) {
            auto row_pass = [&]() {
                for (int r = 0; r < S; ++r) {
                    double sum = eps;
                    for (int c = 0; c < S; ++c) sum += m[r * S + c];
                    for (int c = 0; c < S; ++c) m[r * S + c] /= sum;
                }
            };
            if (row_last) { col_pass(false); row_pass(); } else { row_pass(); col_pass(false); }
        }
        for (int i = 0; i < S * S; ++i) comb[(size_t) t * S * S + i] = m[i];
    }
}

/// Run the kernel and hand back the three maps, so a case can be re-run at a different iteration count.
void run_hc_pre(Dev& d, const std::vector<float>& mixes, const std::vector<float>& scale,
                const std::vector<float>& base, int64_t hc, int iters, int64_t T, std::vector<float>& pre,
                std::vector<float>& post, std::vector<float>& comb) {
    GlmHcShapes s;
    s.hc = hc;
    s.mix = hc * (2 + hc);
    s.sinkhorn_iters = iters;
    s.eps = 1e-6f;
    float *d_mix = d.put(mixes), *d_sc = d.put(scale), *d_ba = d.put(base);
    float *d_pre = nullptr, *d_post = nullptr, *d_comb = nullptr;
    check(cudaMalloc(&d_pre, (size_t) T * hc * 4), "pre");
    check(cudaMalloc(&d_post, (size_t) T * hc * 4), "post");
    check(cudaMalloc(&d_comb, (size_t) T * hc * hc * 4), "comb");
    strata::kernels::glm_hc_pre(d_mix, d_sc, d_ba, d_pre, d_post, d_comb, s, T, nullptr);
    pre = d.get(d_pre, (size_t) T * hc);
    post = d.get(d_post, (size_t) T * hc);
    comb = d.get(d_comb, (size_t) T * hc * hc);
    cudaFree(d_mix); cudaFree(d_sc); cudaFree(d_ba); cudaFree(d_pre); cudaFree(d_post); cudaFree(d_comb);
}

void test_hc(Dev& d, int64_t n_embd, int64_t hc, int iters, int64_t T) {
    GlmHcShapes s;
    s.n_embd = n_embd;
    s.hc = hc;
    s.mix = hc * (2 + hc);
    s.sinkhorn_iters = iters;
    s.eps = 1e-6f;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    std::vector<float> mixes((size_t) s.mix * T), scale = {1.0f, 0.5f, 1.5f}, base((size_t) s.mix);
    for (auto& v : mixes) v = u(rng);
    for (auto& v : base) v = u(rng) * 0.5f;

    std::vector<float> pre, post, comb;
    run_hc_pre(d, mixes, scale, base, hc, iters, T, pre, post, comb);

    std::vector<double> r_pre, r_post, r_comb;
    ref_hc_pre(mixes, scale, base, (int) hc, iters, s.eps, T, r_pre, r_post, r_comb, true, false, false);

    char note[128];
    std::snprintf(note, sizeof note, "hc=%lld iters=%d T=%lld", (long long) hc, iters, (long long) T);
    report("hc_pre pre/post", close_enough(pre, r_pre) && close_enough(post, r_post), note);
    report("hc_pre comb", close_enough(comb, r_comb), note);

    // ---- the rivals, each of which must be told apart ----
    std::vector<double> a, b, c;
    ref_hc_pre(mixes, scale, base, (int) hc, iters, s.eps, T, a, b, c, false, false, false);
    report("  rival: post without the 2", differs(post, b));
    {
        // Both of these are REAL differences that this kernel cannot resolve at f32, so they are measured in
        // double and reported with their size.  See `rel_gap`.
        std::vector<double> d0, d1, d2;
        ref_hc_pre(mixes, scale, base, (int) hc, iters, s.eps, T, a, b, c, true, false, false);
        d0 = c;
        ref_hc_pre(mixes, scale, base, (int) hc, iters, s.eps, T, a, b, c, true, true, false);
        d1 = c;
        ref_hc_pre(mixes, scale, base, (int) hc, iters, s.eps, T, a, b, c, true, false, true);
        d2 = c;
        char n1[96], n2[96];
        std::snprintf(n1, sizeof n1, "  rival: row/column order swapped (%.2e)", rel_gap(d0, d1));
        std::snprintf(n2, sizeof n2, "  rival: no eps on the first pass (%.2e)", rel_gap(d0, d2));
        report(n1, rel_gap(d0, d1) > 0.0);
        report(n2, rel_gap(d0, d2) > 0.0);
    }
    // The same two, at 3 iterations, where the Sinkhorn has NOT converged and the ordering is a first-order
    // effect rather than a 1e-9 one - this is the case that actually pins the loop's shape.
    {
        std::vector<float> p3, o3, got3;
        run_hc_pre(d, mixes, scale, base, hc, 3, T, p3, o3, got3);
        ref_hc_pre(mixes, scale, base, (int) hc, 3, s.eps, T, a, b, c, true, false, false);
        report("  hc_pre comb at 3 iterations", close_enough(got3, c));
        std::vector<double> g0, g1, g2;
        ref_hc_pre(mixes, scale, base, (int) hc, 3, s.eps, T, a, b, c, true, false, false);
        g0 = c;
        ref_hc_pre(mixes, scale, base, (int) hc, 3, s.eps, T, a, b, c, true, true, false);
        g1 = c;
        ref_hc_pre(mixes, scale, base, (int) hc, 3, s.eps, T, a, b, c, true, false, true);
        g2 = c;
        // At 3 iterations the ordering has NOT converged, so this one is large enough to be a genuine
        // kernel-level assertion as well as a real difference in the arithmetic.
        report("    rival at 3: order swapped", differs(got3, g1));
        char n2[96];
        std::snprintf(n2, sizeof n2, "    rival at 3: no eps on the first pass (%.2e)", rel_gap(g0, g2));
        report(n2, rel_gap(g0, g2) > 0.0);
    }
}

/// `ggml_compute_forward_hc_post_f32` / the weighted sum / the collapse.
void ref_hc_post(const std::vector<float>& x, const std::vector<float>& post, const std::vector<float>& res,
                 const std::vector<float>& comb, int64_t n, int S, int64_t T, std::vector<double>& out,
                 bool transposed) {
    out.assign((size_t) T * n * S, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i0 = 0; i0 < n; ++i0) {
            for (int i = 0; i < S; ++i) {
                double sum = (double) x[t * n + i0] * post[t * S + i];
                for (int j = 0; j < S; ++j) {
                    const double cf = transposed ? comb[t * S * S + i * S + j] : comb[t * S * S + j * S + i];
                    sum += cf * res[t * n * S + j * n + i0];
                }
                out[(size_t) t * n * S + i * n + i0] = sum;
            }
        }
    }
}

void test_hc_post(Dev& d, int64_t n, int64_t hc, int64_t T) {
    GlmHcShapes s;
    s.n_embd = n;
    s.hc = hc;
    s.mix = hc * (2 + hc);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    std::vector<float> x((size_t) T * n), post((size_t) T * hc), res((size_t) T * n * hc),
        comb((size_t) T * hc * hc);
    for (auto& v : x) v = u(rng);
    for (auto& v : post) v = u(rng);
    // `res` encodes its own coordinates: a layout mix-up cannot hide behind a symmetric buffer.
    for (int64_t t = 0; t < T; ++t)
        for (int j = 0; j < hc; ++j)
            for (int64_t i = 0; i < n; ++i) res[(size_t) t * n * hc + j * n + i] = (float) (j * 1000 + i % 997);
    for (int64_t t = 0; t < T; ++t)
        for (int j = 0; j < hc; ++j)
            for (int i = 0; i < hc; ++i) comb[(size_t) t * hc * hc + j * hc + i] = (float) (1 + j) / (float) (1 + i);

    float *d_x = d.put(x), *d_p = d.put(post), *d_r = d.put(res), *d_c = d.put(comb);
    float* d_out = nullptr;
    check(cudaMalloc(&d_out, (size_t) T * n * hc * 4), "out");
    strata::kernels::glm_hc_post(d_x, d_p, d_r, d_c, d_out, s, T, nullptr);
    const std::vector<float> out = d.get(d_out, (size_t) T * n * hc);

    std::vector<double> want;
    ref_hc_post(x, post, res, comb, n, (int) hc, T, want, false);
    report("hc_post", close_enough(out, want));
    ref_hc_post(x, post, res, comb, n, (int) hc, T, want, true);
    report("  rival: comb transposed", differs(out, want));

    // mix and sum
    std::vector<float> pre((size_t) T * hc);
    for (auto& v : pre) v = u(rng);
    float *d_pr = d.put(pre), *d_mixout = nullptr;
    check(cudaMalloc(&d_mixout, (size_t) T * n * 4), "mixout");
    strata::kernels::glm_hc_mix(d_r, d_pr, d_mixout, s, T, nullptr);
    std::vector<float> mixout = d.get(d_mixout, (size_t) T * n);
    std::vector<double> mixwant((size_t) T * n);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hc; ++j) sum += (double) pre[t * hc + j] * res[(size_t) t * n * hc + j * n + i];
            mixwant[t * n + i] = sum;
        }
    report("hc_mix", close_enough(mixout, mixwant));

    strata::kernels::glm_hc_sum(d_r, d_mixout, s, T, nullptr);
    std::vector<float> sumout = d.get(d_mixout, (size_t) T * n);
    std::vector<double> sumwant((size_t) T * n);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hc; ++j) sum += res[(size_t) t * n * hc + j * n + i];
            sumwant[t * n + i] = sum / hc;      // the MEAN
        }
    report("hc_sum is a mean", close_enough(sumout, sumwant));
    std::vector<double> sumbad(sumwant.size());
    for (size_t i = 0; i < sumwant.size(); ++i) sumbad[i] = sumwant[i] * hc;
    report("  rival: collapse as a sum", differs(sumout, sumbad));

    cudaFree(d_x); cudaFree(d_p); cudaFree(d_r); cudaFree(d_c); cudaFree(d_out); cudaFree(d_pr);
    cudaFree(d_mixout);
}

// ---------------------------------------------------------------------------------------------------------
// KDA
// ---------------------------------------------------------------------------------------------------------

void test_kda_gate(Dev& d, int64_t hd, int64_t nh, int64_t T, float floor_mag) {
    const int64_t n_v = hd * nh;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::vector<float> raw((size_t) n_v * T), dt(n_v), a(nh);
    for (auto& v : raw) v = u(rng);
    for (auto& v : dt) v = u(rng);
    for (auto& v : a) v = -std::exp(u(rng));      // the file stores -exp(A_log)

    float *d_raw = d.put(raw), *d_dt = d.put(dt), *d_a = d.put(a), *d_g = nullptr;
    check(cudaMalloc(&d_g, (size_t) n_v * T * 4), "gate");
    strata::kernels::glm_kda_gate(d_raw, d_dt, d_a, d_g, n_v, nh, hd, floor_mag, T, nullptr);
    const std::vector<float> got = d.get(d_g, (size_t) n_v * T);

    std::vector<double> want((size_t) n_v * T), rival_nested((size_t) n_v * T), rival_nofloor((size_t) n_v * T);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < n_v; ++c) {
            const int64_t h = c / hd;
            const double inner = (double) a[h] * ((double) raw[t * n_v + c] + dt[c]);
            want[t * n_v + c] = (double) floor_mag / (1.0 + std::exp(inner));       // sigmoid(-inner) folded
            rival_nested[t * n_v + c] = (double) floor_mag / (1.0 + std::exp(-inner));
            rival_nofloor[t * n_v + c] = -1.0 / (1.0 + std::exp(inner));
        }
    report("kda_gate", close_enough(got, want));
    report("  rival: sigmoid sign not nested", differs(got, rival_nested));
    report("  rival: floor not applied", differs(got, rival_nofloor));

    cudaFree(d_raw); cudaFree(d_dt); cudaFree(d_a); cudaFree(d_g);
}

void test_kda_conv(Dev& d, int64_t n_v, int64_t T, int K) {
    const int64_t n_all = 3 * n_v;
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> qkv((size_t) n_all * T), wq((size_t) K * n_v), wk((size_t) K * n_v), wv((size_t) K * n_v),
        state((size_t) (K - 1) * n_all);
    for (auto& v : qkv) v = u(rng);
    for (auto& v : wq) v = u(rng);
    for (auto& v : wk) v = u(rng);
    for (auto& v : wv) v = u(rng);
    for (auto& v : state) v = u(rng);

    float *d_qkv = d.put(qkv), *d_wq = d.put(wq), *d_wk = d.put(wk), *d_wv = d.put(wv), *d_st = d.put(state);
    strata::kernels::glm_kda_conv_silu(d_qkv, d_wq, d_wk, d_wv, d_st, n_v, T, K, nullptr);
    const std::vector<float> got = d.get(d_qkv, (size_t) n_all * T);
    const std::vector<float> got_state = d.get(d_st, (size_t) (K - 1) * n_all);

    // The reference: the three filters concatenated along the channel axis, a state of the last K-1 inputs
    // OLDEST FIRST, and the dot in tap order.
    //
    // **`w` IS INDEXED `c*K + k`, NOT `k*n_v + c`.**  The filter is `ggml`'s `[K, 1, n_v]` folded to `[K, n_v]`,
    // so ne0 = K is the CONTIGUOUS axis and channel `c`'s taps are the K floats at `c*K`.  The first version of
    // this test indexed it the other way round, and the kernel did too, so the two agreed with each other and
    // with nothing else.  The oracle is `ggml_compute_forward_ssm_conv_f32`, which addresses channel `c` at
    // `src2->data + c*src2->nb[1]` with `nb[1] == ne[0]*4`.
    std::vector<double> want((size_t) n_all * T);
    std::vector<double> win((size_t) (K - 1) * n_all);
    for (int64_t ch = 0; ch < n_all; ++ch) {
        const int part = (int) (ch / n_v);
        const int64_t c = ch % n_v;
        const std::vector<float>& w = part == 0 ? wq : (part == 1 ? wk : wv);
        std::vector<double> hist(K - 1);
        for (int k = 0; k < K - 1; ++k) hist[k] = state[(size_t) k * n_all + ch];
        for (int64_t t = 0; t < T; ++t) {
            const float x = qkv[(size_t) t * n_all + ch];
            // The dot reads the taps of THIS output first; the shift follows.  Shifting first makes `hist[K-2]`
            // the current input, so the oldest sample is lost and `x` is counted twice - the same mistake the
            // kernel made, and one this reference shared with it, which is why neither caught the other.
            double sum = (double) x * w[(size_t) c * K + (K - 1)];
            for (int k = 0; k < K - 1; ++k) sum += hist[k] * (double) w[(size_t) c * K + k];
            want[(size_t) t * n_all + ch] = sum / (1.0 + std::exp(-sum));
            if (K >= 2) {
                for (int k = 0; k + 1 < K - 1; ++k) hist[k] = hist[k + 1];
                hist[K - 2] = x;
            }
        }
        for (int k = 0; k < K - 1; ++k) win[(size_t) k * n_all + ch] = hist[k];
    }
    report("kda_conv_silu", close_enough(got, want));
    report("kda_conv_silu state", close_enough(got_state, win));

    // rivals: the tap order reversed, and the state saved newest-first
    {
        std::vector<double> bad((size_t) n_all * T);
        for (int64_t ch = 0; ch < n_all; ++ch) {
            const int part = (int) (ch / n_v);
            const int64_t c = ch % n_v;
            const std::vector<float>& w = part == 0 ? wq : (part == 1 ? wk : wv);
            std::vector<double> hist(K - 1);
            for (int k = 0; k < K - 1; ++k) hist[k] = state[(size_t) k * n_all + ch];
            for (int64_t t = 0; t < T; ++t) {
                const float x = qkv[(size_t) t * n_all + ch];
                double sum = (double) x * w[(size_t) c * K + (K - 1)];
                for (int k = 0; k < K - 1; ++k) sum += hist[k] * (double) w[(size_t) c * K + (K - 1 - k)];
                bad[(size_t) t * n_all + ch] = sum / (1.0 + std::exp(-sum));
                for (int k = 0; k + 1 < K - 1; ++k) hist[k] = hist[k + 1];
                hist[K - 2] = x;
            }
        }
        report("  rival: taps reversed", differs(got, bad));
    }
    // A SECOND RIVAL, AND IT IS THE ONE THAT MATTERED: the same kernel with the filter read as `[n_v, K]`
    // (channel fast, tap slow) - the layout this file and `glm_kda_conv_silu` both used before it was checked
    // against ggml.  It is kept as a rival because it is the mistake that survived a green parity run.
    {
        std::vector<double> bad((size_t) n_all * T);
        for (int64_t ch = 0; ch < n_all; ++ch) {
            const int part = (int) (ch / n_v);
            const int64_t c = ch % n_v;
            const std::vector<float>& w = part == 0 ? wq : (part == 1 ? wk : wv);
            std::vector<double> hist(K - 1);
            for (int k = 0; k < K - 1; ++k) hist[k] = state[(size_t) k * n_all + ch];
            for (int64_t t = 0; t < T; ++t) {
                const float x = qkv[(size_t) t * n_all + ch];
                double sum = (double) x * w[(size_t) (K - 1) * n_v + c];
                for (int k = 0; k < K - 1; ++k) sum += hist[k] * (double) w[(size_t) k * n_v + c];
                bad[(size_t) t * n_all + ch] = sum / (1.0 + std::exp(-sum));
                for (int k = 0; k + 1 < K - 1; ++k) hist[k] = hist[k + 1];
                hist[K - 2] = x;
            }
        }
        report("  rival: filter read [n_v, K] (tap slow)", differs(got, bad));
    }
    // THE THIRD RIVAL, AND THE ONE THAT COST THE MOST: shift the history BEFORE the dot instead of after, so
    // `hist[K-2]` is the current input.  Tap K-2 then double-counts `x` and the oldest sample is dropped - at
    // t=0 that is `x*(w[K-1] + w[K-2])` instead of `x*w[K-1]`.  This is what the kernel and the reference above
    // BOTH did, which is why a green parity run proved nothing; it was caught only at layer 0 of the full model,
    // where the conv output was 2.9x the oracle's and the implied tap was `w[2]+w[3]` to four digits.
    {
        std::vector<double> bad((size_t) n_all * T);
        for (int64_t ch = 0; ch < n_all; ++ch) {
            const int part = (int) (ch / n_v);
            const int64_t c = ch % n_v;
            const std::vector<float>& w = part == 0 ? wq : (part == 1 ? wk : wv);
            std::vector<double> hist(K - 1);
            for (int k = 0; k < K - 1; ++k) hist[k] = state[(size_t) k * n_all + ch];
            for (int64_t t = 0; t < T; ++t) {
                const float x = qkv[(size_t) t * n_all + ch];
                for (int k = 0; k + 1 < K - 1; ++k) hist[k] = hist[k + 1];
                hist[K - 2] = x;
                double sum = (double) x * w[(size_t) c * K + (K - 1)];
                for (int k = 0; k < K - 1; ++k) sum += hist[k] * (double) w[(size_t) c * K + k];
                bad[(size_t) t * n_all + ch] = sum / (1.0 + std::exp(-sum));
            }
        }
        report("  rival: history shifted before the dot", differs(got, bad));
    }
    cudaFree(d_qkv); cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_st);
}

void test_kda_l2norm(Dev& d, int64_t hd, int64_t nh, int64_t T) {
    const int64_t n_v = hd * nh;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    std::vector<float> q((size_t) n_v * T), k((size_t) n_v * T);
    for (auto& v : q) v = u(rng);
    for (auto& v : k) v = u(rng);
    // One head deliberately near-zero, where the two readings differ by orders of magnitude and nowhere else.
    for (int64_t i = 0; i < hd; ++i) q[(size_t) i] = 1e-9f * (float) (i + 1);

    float *d_q = d.put(q), *d_k = d.put(k);
    strata::kernels::glm_kda_l2norm(d_q, d_k, nh, hd, 1e-5f, T, nullptr);
    const std::vector<float> gq = d.get(d_q, (size_t) n_v * T), gk = d.get(d_k, (size_t) n_v * T);

    std::vector<double> wq((size_t) n_v * T), wk((size_t) n_v * T), bad((size_t) n_v * T);
    auto norm = [&](const std::vector<float>& src, std::vector<double>& dst, bool with_eps) {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < nh; ++h) {
                double sum = 0.0;
                for (int64_t i = 0; i < hd; ++i) {
                    const double x = src[(size_t) t * n_v + h * hd + i];
                    sum += x * x;
                }
                const double s = with_eps ? 1.0 / std::sqrt(sum + 1e-5) : 1.0 / std::max(std::sqrt(sum), 1e-5);
                for (int64_t i = 0; i < hd; ++i)
                    dst[(size_t) t * n_v + h * hd + i] = src[(size_t) t * n_v + h * hd + i] * s;
            }
    };
    norm(q, wq, false);
    norm(k, wk, false);
    report("kda_l2norm", close_enough(gq, wq) && close_enough(gk, wk));
    // The rival, measured on the near-zero head only: elsewhere the two agree to 1e-5 and the point of the
    // test is that this one head is not elsewhere.
    norm(q, bad, true);
    double worst = 0.0;
    for (int64_t i = 0; i < hd; ++i) {
        const double a = gq[(size_t) i], b = bad[(size_t) i];
        worst = std::max(worst, std::fabs(a - b));
    }
    std::printf("      (near-zero head: kernel %.3e vs the +eps rival %.3e)\n", (double) gq[0], bad[0]);
    report("  rival: 1/sqrt(sum + eps)", worst > 1e-6);

    cudaFree(d_q); cudaFree(d_k);
}

/// `ggml_compute_forward_kda_f32`, one head, all tokens.
void ref_delta(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
               const std::vector<float>& g, const std::vector<float>& beta, const std::vector<double>& st0,
               int HD, int nh, int64_t T, std::vector<double>& out, std::vector<double>& st,
               bool decay_after, bool scale_on_k) {
    const int64_t n_v = (int64_t) HD * nh;
    out.assign((size_t) n_v * T, 0.0);
    st = st0;
    const double scale = 1.0 / std::sqrt((double) HD);
    for (int h = 0; h < nh; ++h) {
        std::vector<double> S((size_t) HD * HD);
        for (int row = 0; row < HD; ++row)
            for (int col = 0; col < HD; ++col)
                S[(size_t) row + (size_t) col * HD] = st0[(size_t) h * HD * HD + row + (size_t) col * HD];
        for (int64_t t = 0; t < T; ++t) {
            const float* qt = q.data() + t * n_v + (size_t) h * HD;
            const float* kt = k.data() + t * n_v + (size_t) h * HD;
            const float* vt = v.data() + t * n_v + (size_t) h * HD;
            const float* gt = g.data() + t * n_v + (size_t) h * HD;
            double attn = 0.0;
            for (int i = 0; i < HD; ++i) attn += (double) kt[i] * ((double) qt[i] * scale);
            const double bv = 1.0 / (1.0 + std::exp(-(double) beta[(size_t) t * nh + h]));
            std::vector<double> dec(HD), kd(HD), qd(HD), vn(HD);
            for (int col = 0; col < HD; ++col) {
                dec[col] = std::exp(std::min((double) gt[col], 50.0));
                kd[col] = (double) kt[col] * dec[col];
                qd[col] = (double) qt[col] * dec[col];
            }
            for (int row = 0; row < HD; ++row) {
                double vp = 0.0, ov = 0.0;
                for (int col = 0; col < HD; ++col) {
                    const double s = S[(size_t) row + (size_t) col * HD];
                    vp += s * (scale_on_k ? kd[col] * scale : kd[col]);
                    ov += s * qd[col];
                }
                vn[row] = bv * ((double) vt[row] - vp);
                out[(size_t) t * n_v + (size_t) h * HD + row] = ov * scale + vn[row] * attn;
            }
            for (int col = 0; col < HD; ++col) {
                for (int row = 0; row < HD; ++row) {
                    double s = S[(size_t) row + (size_t) col * HD];
                    s = (decay_after ? s : dec[col] * s) + vn[row] * (double) kt[col];
                    if (decay_after) s *= dec[col];
                    S[(size_t) row + (size_t) col * HD] = std::min(std::max(s, -1e6), 1e6);
                }
            }
        }
        for (int row = 0; row < HD; ++row)
            for (int col = 0; col < HD; ++col)
                st[(size_t) h * HD * HD + row + (size_t) col * HD] = S[(size_t) row + (size_t) col * HD];
    }
}

void test_kda_delta(Dev& d, int64_t hd, int64_t nh, int64_t T) {
    const int64_t n_v = hd * nh;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> q((size_t) n_v * T), k((size_t) n_v * T), v((size_t) n_v * T), g((size_t) n_v * T),
        beta((size_t) nh * T);
    for (auto& x : q) x = u(rng);
    for (auto& x : k) x = u(rng);
    for (auto& x : v) x = u(rng);
    // The gate lives in (floor, 0), as the kernel's input contract says.
    for (auto& x : g) x = -5.0f * (float) (1.0 / (1.0 + std::exp(-u(rng) * 3.0)));
    for (auto& x : beta) x = u(rng);
    std::vector<double> st0((size_t) nh * hd * hd);
    for (size_t i = 0; i < st0.size(); ++i) st0[i] = 0.02 * std::sin((double) i * 0.7);

    float *d_q = d.put(q), *d_k = d.put(k), *d_v = d.put(v), *d_g = d.put(g), *d_b = d.put(beta);
    std::vector<float> stf(st0.begin(), st0.end());
    float* d_st = d.put(stf);
    float* d_out = nullptr;
    check(cudaMalloc(&d_out, (size_t) n_v * T * 4), "out");
    strata::kernels::glm_kda_delta(d_q, d_k, d_v, d_g, d_b, d_st, d_out, hd, nh, T, nullptr);
    const std::vector<float> out = d.get(d_out, (size_t) n_v * T);
    const std::vector<float> stgot = d.get(d_st, st0.size());

    std::vector<double> wout, wst;
    ref_delta(q, k, v, g, beta, st0, (int) hd, (int) nh, T, wout, wst, false, false);
    report("kda_delta out", close_enough(out, wout, 1e-4));
    report("kda_delta state", close_enough(stgot, wst, 1e-4));

    ref_delta(q, k, v, g, beta, st0, (int) hd, (int) nh, T, wout, wst, true, false);
    // The STATE, not the output: at T=1 the new state is never read again, so this rival is invisible in the
    // output until there is a second token and the check would pass for the wrong reason.
    report("  rival: decay applied after the write", differs(stgot, wst));
    ref_delta(q, k, v, g, beta, st0, (int) hd, (int) nh, T, wout, wst, false, true);
    report("  rival: scale inside v' as well", differs(out, wout));
    // The state's row/col order is silent when wrong, so the same numbers are offered transposed and must not
    // match: `bad[..row + col*hd] = wst[..col + row*hd]`.
    {
        std::vector<double> bad(stgot.size());
        for (int h = 0; h < nh; ++h)
            for (int row = 0; row < hd; ++row)
                for (int col = 0; col < hd; ++col)
                    bad[(size_t) h * hd * hd + row + (size_t) col * hd] =
                        wst[(size_t) h * hd * hd + col + (size_t) row * hd];
        report("  rival: state row/col swapped", differs(stgot, bad));
    }

    // The recurrence is sequential: T tokens must equal T single-token calls with the state carried.
    {
        std::vector<float> st1(stf);
        float* d_st1 = d.put(st1);
        float* d_out1 = nullptr;
        check(cudaMalloc(&d_out1, (size_t) n_v * T * 4), "out1");
        std::vector<float> step((size_t) n_v * T, 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            strata::kernels::glm_kda_delta(d_q + t * n_v, d_k + t * n_v, d_v + t * n_v, d_g + t * n_v,
                                           d_b + t * nh, d_st1, d_out1 + t * n_v, hd, nh, 1, nullptr);
        }
        const std::vector<float> stepped = d.get(d_out1, (size_t) n_v * T);
        double worst = 0.0;
        for (size_t i = 0; i < stepped.size(); ++i)
            worst = std::max(worst, std::fabs((double) stepped[i] - (double) out[i]));
        char note[96];
        std::snprintf(note, sizeof note, "max |d| %.3e", worst);
        // Not bitwise: the multi-token kernel keeps the state in registers across tokens where the stepped run
        // writes and re-reads it through global memory.  Both are f32, so they agree to f32 rounding.
        report("kda_delta T tokens == T single-token calls", worst < 1e-5, note);
        cudaFree(d_st1); cudaFree(d_out1);
    }

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_b); cudaFree(d_st); cudaFree(d_out);
}

// ---------------------------------------------------------------------------------------------------------
// MLA
// ---------------------------------------------------------------------------------------------------------

/// `ggml_flash_attn_ext(Qcur, K_cache, K_cache, mask, kq_scale)`, in double: one query row per token, softmax
/// over the causal prefix of the cache, and **K IS V** - one fp16 latent per token serves as both.
///
/// `limit_mode` is the rival switch and it is the one that matters here.  The kernel's limit is
/// `pos_base + t + 1` - self INCLUDED, at this token's absolute position.  Dropping the `+ 1` excludes the
/// diagonal, which is the classic off-by-one; ignoring `pos_base` gives every row the full cache, which is
/// wrong for every query but the last.  Both are finite and normalised, so only a case can tell them apart.
void ref_mla_attn(const std::vector<float>& q, const std::vector<float>& kc, int64_t nh, int64_t T, int64_t kv,
                  int64_t n_kv, int64_t pos_base, double scale, int limit_mode, std::vector<double>& out) {
    out.assign((size_t) nh * T * kv, 0.0);
    for (int64_t h = 0; h < nh; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            int64_t limit = 0;
            if (limit_mode == 0) limit = std::min<int64_t>(pos_base + t + 1, n_kv);
            else if (limit_mode == 1) limit = std::min<int64_t>(pos_base + t, n_kv);
            else limit = n_kv;
            if (limit <= 0) continue;
            const float* qh = q.data() + (h * T + t) * kv;
            std::vector<double> sc((size_t) limit, 0.0);
            double m = -1e300;
            for (int64_t s = 0; s < limit; ++s) {
                const float* ks = kc.data() + s * kv;
                double dot = 0.0;
                for (int64_t c = 0; c < kv; ++c) dot += (double) qh[c] * (double) ks[c];
                sc[(size_t) s] = dot * scale;
                m = std::max(m, sc[(size_t) s]);
            }
            double l = 0.0;
            for (int64_t s = 0; s < limit; ++s) { sc[(size_t) s] = std::exp(sc[(size_t) s] - m); l += sc[(size_t) s]; }
            for (int64_t c = 0; c < kv; ++c) {
                double a = 0.0;
                for (int64_t s = 0; s < limit; ++s) a += sc[(size_t) s] * (double) kc[(size_t) s * kv + c];
                out[(size_t) (h * T + t) * kv + c] = a / l;
            }
        }
    }
}

void test_mla_attn(Dev& d, int64_t nh, int64_t T, int64_t kv, int64_t n_kv, int64_t pos_base) {
    std::mt19937 rng((uint32_t) (0x9E3779B9u ^ (uint32_t) (nh * 131 + T * 977 + kv * 31 + n_kv * 7 + pos_base)));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> q((size_t) nh * T * kv), kf((size_t) n_kv * kv);
    for (float& v : q) v = u(rng);
    for (float& v : kf) v = u(rng);

    float* d_q = d.put(q);
    float* d_out = nullptr;
    check(cudaMalloc(&d_out, q.size() * 4), "mla out");
    __half* d_cache = nullptr;
    check(cudaMalloc(&d_cache, kf.size() * 2), "mla cache");
    // The cache is built THROUGH THE KERNEL, one row per token, so a wrong row stride here would show up as
    // every case failing rather than as a case passing on a cache this test laid out itself.
    for (int64_t s = 0; s < n_kv; ++s) {
        float* d_row = d.put(std::vector<float>(kf.begin() + s * kv, kf.begin() + (s + 1) * kv));
        strata::kernels::glm_mla_cache_store(d_row, (uint16_t*) d_cache, s, kv, nullptr);
        check(cudaFree(d_row), "mla row free");
    }
    // The reference reads the cache as the kernel does - ROUNDED to fp16, not the f32 that was handed in.
    const std::vector<__half> hc = d.get((const __half*) d_cache, kf.size());
    std::vector<float> kc(hc.size());
    for (size_t i = 0; i < hc.size(); ++i) kc[i] = __half2float(hc[i]);

    const double scale = 1.0 / std::sqrt((double) kv);
    strata::kernels::glm_mla_attn(d_q, (const uint16_t*) d_cache, d_out, nh, kv, n_kv, T, pos_base, (float) scale,
                                  nullptr);
    const std::vector<float> got = d.get(d_out, q.size());

    std::vector<double> want;
    ref_mla_attn(q, kc, nh, T, kv, n_kv, pos_base, scale, 0, want);
    report("mla_attn", close_enough(got, want));

    std::vector<double> rival;
    ref_mla_attn(q, kc, nh, T, kv, n_kv, pos_base, scale, 1, rival);
    report("  rival: the diagonal excluded", differs(got, rival));
    // ...and the no-causal rival only IS a rival when the cache reaches past the shortest window.  At
    // `n_kv == pos_base + 1` - every single-token decode - the two limits are the same number and reporting
    // "the kernel disagrees with a reading it agrees with" would be a false pass dressed as a test.
    if (n_kv > pos_base + 1) {
        ref_mla_attn(q, kc, nh, T, kv, n_kv, pos_base, scale, 2, rival);
        report("  rival: no causal limit", differs(got, rival));
    }

    // The layout: `q` and `out` are [kv, T, n_head] with the head OUTERMOST, so head h's block starts at
    // h*T*kv.  Reading them as [kv, n_head, T] swaps the two outer strides, which is finite everywhere.
    //
    // Only a case with T == n_head is a rival here: at any other shape the swapped indexing is not a
    // permutation of the same elements, so it would "differ" for a reason that has nothing to do with the
    // kernel.  The multi-token cases below are run at T == n_head for exactly this.
    if (T == nh) {
        std::vector<double> bad(got.size(), 0.0);
        for (int64_t h = 0; h < nh; ++h)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t c = 0; c < kv; ++c)
                    bad[(size_t) (t * nh + h) * kv + c] = want[(size_t) (h * T + t) * kv + c];
        report("  rival: head and token swapped", differs(got, bad));
    }

    cudaFree(d_q); cudaFree(d_out); cudaFree(d_cache);
}

// ---------------------------------------------------------------------------------------------------------
// the FFN's SwiGLU with a limit
// ---------------------------------------------------------------------------------------------------------

/// The reference's SwiGLU with a limit, in double, and the four other readings of it.
///
/// **MODE 0 IS THE BRANCH GLM5NEXT ACTUALLY TAKES**, and reading the wrong one is the mistake this case exists
/// to catch: mainline llama.cpp contains two arithmetics for this and picks by arch.  `ggml_swiglu_clamp`
/// (ggml-cpu/ops.cpp:3448-3451) clamps the RAW GATE - `gate = min(gate, limit)`, `up = clamp(up, +-limit)`,
/// `out = silu(gate) * up` - and it is reached ONLY by `LLM_ARCH_DEEPSEEK4` and DFLASH-with-hc_mult
/// (llama-graph.cpp:2237, :1842).  GLM5NEXT is every other arch, so it takes the decomposed run at
/// llama-graph.cpp:1840-1848 (shared expert) and :2235-2243 (experts): the up clamped on both sides, the gate
/// through silu FIRST, and the silu's output then clamped ABOVE ONLY.  ik_llama.cpp - the oracle the ladder
/// runs against - computes that same order in both its CUDA kernel (ggml-cuda/unary.cu:74-83) and its CPU iqk
/// path (iqk/iqk_mul_mat.cpp:156-171), so upstream and the oracle agree and there is nothing to choose between.
///
///   mode 1: no clamp at all - what this port computed before the missing clamp was found.
///   mode 2: the up left raw.
///   mode 3: the SILU'S OUTPUT clamped on BOTH sides.
///   mode 4: the RAW GATE clamped before the silu - deepseek4's op, and this port's first attempt at the fix.
void ref_swiglu(const std::vector<float>& gate, const std::vector<float>& up, double limit, int mode,
                std::vector<double>& out) {
    const bool cl = limit > 1e-6;   // the reference's own guard (llama-graph.cpp:2233, iqk_mul_mat.cpp:156)
    out.resize(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i], u = up[i];
        const double silu = g / (1.0 + std::exp(-g));
        const double upc = cl ? std::fmin(std::fmax(u, -limit), limit) : u;
        double a;   // the gate's contribution: where the four readings part company
        if (mode == 1)      a = silu;                                      // no clamp
        else if (mode == 2) a = cl ? std::fmin(silu, limit) : silu;        // the up left raw
        else if (mode == 3) a = cl ? std::fmin(std::fmax(silu, -limit), limit) : silu;
        else if (mode == 4) {                                              // the raw gate, before the silu
            const double gq = cl ? std::fmin(g, limit) : g;
            a = gq / (1.0 + std::exp(-gq));
        } else              a = cl ? std::fmin(silu, limit) : silu;        // 0: the reference
        out[i] = a * (mode == 2 ? u : upc);
    }
}

void test_swiglu(Dev& d, int64_t n, float limit) {
    std::mt19937 rng((uint32_t) (0x85EBCA6Bu ^ (uint32_t) (n * 131 + (int) (limit * 8))));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    // THE FIXTURE HAS TO REACH THE CLAMP, OR THE CASE PROVES NOTHING.  `silu(gate)` is above a limit of 10 only
    // once `gate > 10.00045`, so the gates span +-22 (a quarter of them are past it) and the ups span +-30, and
    // the counts are ASSERTED below rather than hoped for.  A fixture of ordinary activations would have passed
    // against the unclamped kernel, which is exactly how the missing clamp survived 30 of the 32 ladder cases.
    std::vector<float> gate((size_t) n), up((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        gate[i] = 22.0f * u(rng);
        up[i] = 30.0f * u(rng);
    }
    int n_gate = 0, n_up = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double g = gate[i];
        if (g / (1.0 + std::exp(-g)) > (double) limit) ++n_gate;
        if (std::fabs((double) up[i]) > (double) limit) ++n_up;
    }
    report("swiglu: the fixture reaches the clamp", n_gate > 0 && n_up > 0);

    float* d_gate = d.put(gate);
    float* d_up = d.put(up);
    strata::kernels::glm_swiglu(d_gate, d_up, limit, n, nullptr);
    const std::vector<float> got = d.get(d_gate, (size_t) n);
    cudaFree(d_gate);
    cudaFree(d_up);

    std::vector<double> want;
    ref_swiglu(gate, up, limit, 0, want);
    report("swiglu (limit caps the silu's output)", close_enough(got, want));

    std::vector<double> rival;
    // **THE RIVALS ARE ONLY RIVALS WHERE THE CLAMP CAN BITE.**  At `limit == 0` - the first family's packs - all
    // five readings are the same arithmetic, so reporting "the kernel disagrees with a reading it agrees with"
    // would be a false pass dressed up as a test.  That case gets the opposite assertion below instead.
    if (limit > 1e-6f) {
        ref_swiglu(gate, up, limit, 1, rival);
        report("  rival: no clamp", differs(got, rival));
        ref_swiglu(gate, up, limit, 2, rival);
        report("  rival: the up left raw", differs(got, rival));
        // The raw-gate reading is the CLOSE one, and the tolerance says why: the two differ at all only where
        // `silu(gate) > limit`, and there by the 4.54e-4 between `limit` and `silu(limit)` at limit 10 - about
        // 1.5e-5 of this fixture's peak.  The default 1e-4 would have reported the two readings as equal, so
        // this line is the tightest one in the file on purpose.
        ref_swiglu(gate, up, limit, 4, rival);
        report("  rival: the raw gate clamped (deepseek4's op)", differs(got, rival, 1e-5));

        // ...and the reading that CANNOT be caught, asserted as the finding it is: clamping the silu's output on
        // BOTH sides is the same computation as clamping it above only, because silu's range is (-0.2785, inf)
        // and a limit of 10 has nothing to clamp at the bottom.
        std::vector<double> both;
        ref_swiglu(gate, up, limit, 3, both);
        report("  rival: both sides of the silu's output, indistinguishable", rel_gap(want, both) <= 0.0);
    } else {
        ref_swiglu(gate, up, limit, 1, rival);
        report("  limit 0: the kernel is the unclamped form", close_enough(got, rival));
    }
}

// ---------------------------------------------------------------------------------------------------------
// DSA: the k-pool indexer and the sparse attention it selects for
// ---------------------------------------------------------------------------------------------------------

/// `pooled[e,p] = sum_m softmax_m(ig[e, m_p] + ape[e,m]) * ik[e, m_p]`, in double.
///
/// `rival` picks the reading: 0 the reference, 1 ONE softmax over the pool's whole key_dim x kpool block (the
/// natural misreading - it is finite, it is normalised, and it weights every element of the pool by every
/// element's gate), 2 the logits without `ape` at all.
void ref_dsa_pool(const std::vector<float>& ik, const std::vector<float>& ig, const std::vector<float>& ape,
                  int kd, int kp, int n_pools, int rival, std::vector<double>& out) {
    auto lg = [&](int e, int p, int m) {
        const double g = (double) ig[(size_t) e + (size_t) kd * (p * kp + m)];
        return g + (rival == 2 ? 0.0 : (double) ape[(size_t) e + (size_t) kd * m]);
    };
    out.assign((size_t) kd * n_pools, 0.0);
    for (int p = 0; p < n_pools; ++p) {
        if (rival == 1) {
            double mx = -1e300;
            for (int e = 0; e < kd; ++e)
                for (int m = 0; m < kp; ++m) mx = std::max(mx, lg(e, p, m));
            double den = 0.0;
            for (int e = 0; e < kd; ++e)
                for (int m = 0; m < kp; ++m) den += std::exp(lg(e, p, m) - mx);
            for (int e = 0; e < kd; ++e) {
                double acc = 0.0;
                for (int m = 0; m < kp; ++m)
                    acc += std::exp(lg(e, p, m) - mx) / den * (double) ik[(size_t) e + (size_t) kd * (p * kp + m)];
                out[(size_t) e + (size_t) kd * p] = acc;
            }
            continue;
        }
        for (int e = 0; e < kd; ++e) {
            double mx = -1e300, den = 0.0, acc = 0.0;
            for (int m = 0; m < kp; ++m) mx = std::max(mx, lg(e, p, m));
            for (int m = 0; m < kp; ++m) den += std::exp(lg(e, p, m) - mx);
            for (int m = 0; m < kp; ++m)
                acc += std::exp(lg(e, p, m) - mx) / den * (double) ik[(size_t) e + (size_t) kd * (p * kp + m)];
            out[(size_t) e + (size_t) kd * p] = acc;
        }
    }
}

/// `score[t,p] = sum_h relu(iq_h . pooled_p) * weights[h,t]`.  Rival 1 drops the relu; rival 2 relu's the
/// weighted term instead of the dot, which is the same expression with one bracket moved.
void ref_dsa_score(const std::vector<float>& iq, const std::vector<double>& pooled, const std::vector<float>& wts,
                   int kd, int nh, int T, int n_pools, int rival, std::vector<double>& out) {
    out.assign((size_t) n_pools * T, 0.0);
    for (int t = 0; t < T; ++t)
        for (int p = 0; p < n_pools; ++p) {
            double acc = 0.0;
            for (int h = 0; h < nh; ++h) {
                double dot = 0.0;
                for (int e = 0; e < kd; ++e)
                    dot += (double) iq[(size_t) e + (size_t) kd * h + (size_t) kd * nh * t] * pooled[(size_t) e + (size_t) kd * p];
                const double w = (double) wts[(size_t) h + (size_t) nh * t];
                if (rival == 1) acc += dot * w;
                else if (rival == 2) acc += std::max(dot * w, 0.0);
                else acc += std::max(dot, 0.0) * w;
            }
            out[(size_t) p + (size_t) n_pools * t] = acc;
        }
}

/// The selection, on the host, exactly as the kernel does it - the rank formula, the cell expansion and the
/// tail.  `tie` flips the tie-break to the HIGHER index, `cell_order` emits each selected pool's cells sorted
/// ascending instead of in rank order.
void ref_dsa_select(const std::vector<float>& score, int n_pools, int kp, int top_pools, int select_tail, int T,
                    int n_sel, const std::vector<int>& pos, int tie, int cell_order, std::vector<int32_t>& out) {
    out.assign((size_t) n_sel * T, -1);
    std::vector<std::pair<float, int>> vis;
    for (int t = 0; t < T; ++t) {
        const int n_vis = (pos[(size_t) t] + 1) / kp;
        vis.clear();
        for (int p = 0; p < n_vis; ++p) vis.emplace_back(score[(size_t) p + (size_t) n_pools * t], p);
        if (cell_order) std::sort(vis.begin(), vis.end(), [](auto& a, auto& b) { return a.second < b.second; });
        else std::stable_sort(vis.begin(), vis.end(), [tie](auto& a, auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return tie ? a.second > b.second : a.second < b.second;
        });
        int32_t* row = out.data() + (size_t) n_sel * t;
        for (size_t r = 0; r < vis.size() && (int) r < top_pools; ++r)
            for (int m = 0; m < kp; ++m) row[r * kp + m] = vis[r].second * kp + m;
        if (select_tail)
            for (int m = 0; m < kp - 1; ++m) {
                const int cell = n_vis * kp + m;
                if (cell <= pos[(size_t) t]) row[top_pools * kp + m] = cell;
            }
    }
}

/// The sparse attention over the selected cells, in double.  `rival`: 0 the reference, 1 the scale from
/// `kv_lora` instead of `qk_nope`, 2 the padding cells NOT masked (they keep probability 0 in the score but
/// are counted in the normaliser, i.e. the -1 row is read as "a cell with score 0").
void ref_dsa_attn(const std::vector<float>& q_abs, const std::vector<float>& lat, const std::vector<int32_t>& cells,
                  int kv, int nh, int qk_nope, int T, int n_sel, int rival, std::vector<double>& out) {
    out.assign((size_t) kv * nh * T, 0.0);
    const double scale = rival == 1 ? 1.0 / std::sqrt((double) kv) : 1.0 / std::sqrt((double) qk_nope);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nh; ++h) {
            const int32_t* row = cells.data() + (size_t) n_sel * t;
            const float* qa = q_abs.data() + (size_t) kv * h + (size_t) kv * nh * t;
            std::vector<double> sc((size_t) n_sel, 0.0);
            double mx = -1e300;
            for (int s = 0; s < n_sel; ++s) {
                const int c = row[s];
                if (c < 0) {
                    sc[(size_t) s] = (rival == 2) ? 0.0 : -1e300;
                    continue;
                }
                double dot = 0.0;
                for (int e = 0; e < kv; ++e) dot += (double) qa[e] * (double) lat[(size_t) e + (size_t) kv * c];
                sc[(size_t) s] = dot * scale;
                mx = std::max(mx, sc[(size_t) s]);
            }
            double den = 0.0;
            for (int s = 0; s < n_sel; ++s) {
                const double v = sc[(size_t) s] <= -1e299 ? 0.0 : std::exp(sc[(size_t) s] - mx);
                sc[(size_t) s] = v;
                den += v;
            }
            for (int e = 0; e < kv; ++e) {
                double acc = 0.0;
                for (int s = 0; s < n_sel; ++s) {
                    const int c = row[s];
                    if (c >= 0) acc += sc[(size_t) s] / den * (double) lat[(size_t) e + (size_t) kv * c];
                }
                out[(size_t) e + (size_t) kv * h + (size_t) kv * nh * t] = acc;
            }
        }
}

/// An int comparison that says WHERE, because a `cells` row is opaque and "they differ" is not actionable.
bool ids_same(const std::vector<int32_t>& got, const std::vector<int32_t>& want, std::string& note) {
    for (size_t i = 0; i < got.size() && i < want.size(); ++i)
        if (got[i] != want[i]) {
            note = "first at " + std::to_string(i) + ": got " + std::to_string(got[i]) + ", want " +
                   std::to_string(want[i]);
            return false;
        }
    if (got.size() != want.size()) { note = "different lengths"; return false; }
    return true;
}

/// Does a rival differ from the REFERENCE at all?  A rival the reference cannot separate from itself is not a
/// rival, and reporting "the kernel disagrees with it" would be a false pass - the same trap the Sinkhorn cases
/// above answer with `rel_gap`.  Every rival below is checked against the reference first and skipped, with a
/// note, when the fixture does not separate the two readings.
bool dbl_differ(const std::vector<double>& a, const std::vector<double>& b, double rel = 1e-4) {
    double peak = 0.0;
    for (double v : a) peak = std::max(peak, std::fabs(v));
    const double tol = rel * std::max(peak, 1e-6);
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return true;
    return false;
}

void rival_f(const char* name, const std::vector<float>& got, const std::vector<double>& ref,
             const std::vector<double>& rival) {
    if (!dbl_differ(ref, rival)) {
        report(name, true, "(not separable from the reference here)");
        return;
    }
    report(name, differs(got, rival));
}

void rival_i(const char* name, const std::vector<int32_t>& got, const std::vector<int32_t>& ref,
             const std::vector<int32_t>& rival) {
    if (ref == rival) {
        report(name, true, "(not separable from the reference here)");
        return;
    }
    report(name, got != rival);
}

struct DsaFix {
    int kd = 8;          // key_dim
    int kp = 4;          // kpool
    int nh = 3;          // indexer heads
    int n_pools = 4;     // COMPLETED pools
    int top_pools = 2;
    int kv = 12;         // kv_lora (the latent width)
    int vhead = 2;       // n_head
    int n_sel = 2 * 4 + 3;
    int n_cells = 0;
};

void test_dsa_indexer(Dev& d, DsaFix f, int T, int select_tail) {
    std::mt19937 rng((uint32_t) (0x5BF03635u ^ (uint32_t) (f.kd * 31 + f.kp * 977 + f.nh * 131 + T * 7 + select_tail)));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const size_t n_cells = (size_t) f.kp * f.n_pools + f.kp;   // the completed pools plus an incomplete tail
    std::vector<float> ik(n_cells * f.kd), ig(n_cells * f.kd), ape((size_t) f.kp * f.kd);
    std::vector<float> iq((size_t) f.kd * f.nh * T), wts((size_t) f.nh * T);
    for (float& v : ik) v = u(rng);
    for (float& v : ig) v = u(rng);
    for (float& v : ape) v = u(rng);
    for (float& v : iq) v = u(rng);
    for (float& v : wts) v = u(rng);
    // The indexer's weights are prescaled by `1/sqrt(key_dim * idx_heads)` where the reference BUILDS them, so
    // the kernel must not scale again - a second division is a factor of ~0.05 here and would still select cells.
    const float wscale = (float) (1.0 / std::sqrt((double) f.kd * f.nh));
    for (float& v : wts) v *= wscale;

    // The queries' absolute positions: the last token sits inside an incomplete pool, so the tail path is live,
    // and the first ones sit before any pool has completed.
    std::vector<int> pos((size_t) T);
    for (int t = 0; t < T; ++t) pos[(size_t) t] = f.kp * f.n_pools - 2 + t;

    float* d_ik = d.put(ik);
    float* d_ig = d.put(ig);
    float* d_ape = d.put(ape);
    float* d_iq = d.put(iq);
    float* d_wts = d.put(wts);
    float* d_pooled = nullptr;
    float* d_score = nullptr;
    int* d_cells = nullptr;
    int* d_pos = d.put(pos);
    check(cudaMalloc(&d_pooled, (size_t) f.kd * f.n_pools * 4), "dsa pooled");
    check(cudaMalloc(&d_score, (size_t) f.n_pools * T * 4), "dsa score");
    check(cudaMalloc(&d_cells, (size_t) f.n_sel * T * 4), "dsa cells");

    strata::kernels::glm_dsa_pool(d_ik, d_ig, d_ape, f.kd, f.kp, f.n_pools, d_pooled, nullptr);
    const std::vector<float> got_pooled = d.get(d_pooled, (size_t) f.kd * f.n_pools);
    std::vector<double> want, ref_pool;
    ref_dsa_pool(ik, ig, ape, f.kd, f.kp, f.n_pools, 0, ref_pool);
    report("dsa_pool", close_enough(got_pooled, ref_pool));
    ref_dsa_pool(ik, ig, ape, f.kd, f.kp, f.n_pools, 1, want);
    rival_f("  rival: one softmax over the whole pool block", got_pooled, ref_pool, want);
    ref_dsa_pool(ik, ig, ape, f.kd, f.kp, f.n_pools, 2, want);
    rival_f("  rival: the logits without ape", got_pooled, ref_pool, want);

    std::vector<float> pooled_f(got_pooled.begin(), got_pooled.end());
    strata::kernels::glm_dsa_score(d_iq, d_pooled, d_wts, f.kd, f.nh, T, f.n_pools, d_score, nullptr);
    const std::vector<float> got_score = d.get(d_score, (size_t) f.n_pools * T);
    std::vector<double> want_score, ref_score;
    ref_dsa_score(iq, ref_pool, wts, f.kd, f.nh, T, f.n_pools, 0, ref_score);
    report("dsa_score", close_enough(got_score, ref_score));
    ref_dsa_score(iq, ref_pool, wts, f.kd, f.nh, T, f.n_pools, 1, want_score);
    rival_f("  rival: no relu", got_score, ref_score, want_score);
    ref_dsa_score(iq, ref_pool, wts, f.kd, f.nh, T, f.n_pools, 2, want_score);
    rival_f("  rival: relu after the weighting", got_score, ref_score, want_score);

    // The selection is on the SCORES THE KERNEL PRODUCED, so a score error cannot hide as a selection pass.
    std::vector<float> score_f(got_score.begin(), got_score.end());
    strata::kernels::glm_dsa_select(d_score, f.n_pools, f.kp, f.top_pools, select_tail, T, f.n_sel, d_pos, d_cells,
                                    nullptr);
    const std::vector<int32_t> got_cells = d.get(d_cells, (size_t) f.n_sel * T);
    std::vector<int32_t> ref_cells, want_cells;
    std::string note;
    ref_dsa_select(score_f, f.n_pools, f.kp, f.top_pools, select_tail, T, f.n_sel, pos, 0, 0, ref_cells);
    report("dsa_select", ids_same(got_cells, ref_cells, note), note.c_str());
    // The tail is what makes positions before the first complete pool visible at all, so flipping it must move
    // the rows - and it only does when `pos` reaches past a completed pool's end, which the fixture's positions
    // are chosen to do.
    ref_dsa_select(score_f, f.n_pools, f.kp, f.top_pools, !select_tail, T, f.n_sel, pos, 0, 0, want_cells);
    rival_i("  rival: select_tail flipped", got_cells, ref_cells, want_cells);
    ref_dsa_select(score_f, f.n_pools, f.kp, f.top_pools, select_tail, T, f.n_sel, pos, 1, 0, want_cells);
    rival_i("  rival: cells in cell order, not score order", got_cells, ref_cells, want_cells);

    cudaFree(d_ik); cudaFree(d_ig); cudaFree(d_ape); cudaFree(d_iq); cudaFree(d_wts);
    cudaFree(d_pooled); cudaFree(d_score); cudaFree(d_cells); cudaFree(d_pos);
}

/// The tie-break and the padding are their own case, because ties do not occur by chance in a random fixture
/// and a padding cell that is only ever -1 in the fixture cannot show a kernel that fails to write it.
void test_dsa_select_ties(Dev& d) {
    const int n_pools = 4, kp = 4, top_pools = 2, n_sel = top_pools * kp + kp - 1;
    // Scores with a deliberate three-way tie for first and a two-way tie for second.
    const std::vector<float> score = {2.0f, 2.0f, 2.0f, 1.0f};
    const std::vector<int> pos = {2 * kp + 1};   // two pools visible, so the tie is over 2 of the 4 numbers
    float* d_score = d.put(score);
    int* d_pos = d.put(pos);
    int* d_cells = nullptr;
    check(cudaMalloc(&d_cells, (size_t) n_sel * 4), "dsa tie cells");
    strata::kernels::glm_dsa_select(d_score, n_pools, kp, top_pools, /*select_tail=*/1, 1, n_sel, d_pos, d_cells,
                                    nullptr);
    const std::vector<int32_t> got = d.get(d_cells, (size_t) n_sel);
    std::vector<int32_t> ref, want;
    std::string note;
    ref_dsa_select(score, n_pools, kp, top_pools, 1, 1, n_sel, pos, 0, 0, ref);
    report("dsa_select ties by lower index", ids_same(got, ref, note), note.c_str());
    ref_dsa_select(score, n_pools, kp, top_pools, 1, 1, n_sel, pos, 1, 0, want);
    rival_i("  rival: ties by higher index", got, ref, want);
    // Every row must END with the padding the row did not fill: the two selected pools' cells, then the tail,
    // then -1.  A real cell after a pad is a row the kernel did not clean, which the count alone would miss.
    int32_t last = -2;
    bool padded = true;
    for (int s = 0; s < n_sel; ++s) {
        if (got[(size_t) s] == -1) last = -1;
        else if (last == -1) padded = false;
    }
    report("  the padding is -1 and contiguous at the end", padded);

    cudaFree(d_score); cudaFree(d_pos); cudaFree(d_cells);
}

void test_dsa_attn(Dev& d, DsaFix f, int T, int kind) {
    std::mt19937 rng((uint32_t) (0xC2B2AE35u ^ (uint32_t) (f.kv * 31 + f.vhead * 7 + T * 131 + kind)));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const size_t n_cells = (size_t) f.kp * f.n_pools + f.kp;
    std::vector<float> q((size_t) f.kv * f.vhead * T), lf(n_cells * f.kv);
    for (float& v : q) v = u(rng);
    for (float& v : lf) v = u(rng);
    // `cells` is built from `pos`, not random: rows must be sorted-descending by pool, padded with -1, and every
    // cell must be inside the cache.  A random row would let the kernel pass on inputs the model never makes.
    std::vector<int32_t> cells((size_t) f.n_sel * T, -1);
    std::vector<int> pos((size_t) T);
    for (int t = 0; t < T; ++t) {
        pos[(size_t) t] = (int) n_cells - 1;
        int32_t* row = cells.data() + (size_t) f.n_sel * t;
        int w = 0;
        for (int p = 1; p < f.top_pools && w + f.kp <= f.n_sel; ++p)
            for (int m = 0; m < f.kp; ++m) row[w++] = p * f.kp + m;
        for (int m = 0; m < f.kp - 1 && w < f.n_sel; ++m) row[w++] = f.kp * f.n_pools + m;
    }

    float* d_q = d.put(q);
    int* d_cells = d.put(cells);
    float* d_out = nullptr;
    check(cudaMalloc(&d_out, q.size() * 4), "dsa attn out");
    __half* d_lat = nullptr;
    check(cudaMalloc(&d_lat, lf.size() * 2), "dsa latents");
    // Through the engine's own cache store, as the MLA test does: a wrong row stride then fails every case
    // rather than passing on a cache this test laid out itself.
    for (size_t c = 0; c < n_cells; ++c) {
        float* d_row = d.put(std::vector<float>(lf.begin() + c * f.kv, lf.begin() + (c + 1) * f.kv));
        strata::kernels::glm_mla_cache_store(d_row, (uint16_t*) d_lat, (int64_t) c, f.kv, nullptr);
        check(cudaFree(d_row), "dsa row free");
    }
    const std::vector<__half> hl = d.get((const __half*) d_lat, lf.size());
    std::vector<float> lat(hl.size());
    for (size_t i = 0; i < hl.size(); ++i) lat[i] = __half2float(hl[i]);

    // `kind` is the scale's rival: 0 gives the kernel the real `qk_nope` (wider than kv_lora here, as it is in
    // the model), 1 gives it `kv_lora` - which must then MATCH the `1/sqrt(kv_lora)` rival, i.e. the kernel is
    // not applying a scale of its own.
    const int qk_nope = kind == 0 ? 2 * f.kv + 3 : f.kv;
    strata::kernels::glm_dsa_attn(d_q, (const uint16_t*) d_lat, d_cells, f.kv, f.vhead, qk_nope, T, f.n_sel, d_out,
                                  nullptr);
    const std::vector<float> got = d.get(d_out, q.size());
    std::vector<double> ref, want;
    ref_dsa_attn(q, lat, cells, f.kv, f.vhead, qk_nope, T, f.n_sel, 0, ref);
    report("dsa_attn", close_enough(got, ref));
    ref_dsa_attn(q, lat, cells, f.kv, f.vhead, qk_nope, T, f.n_sel, 1, want);
    rival_f("  rival: the scale from kv_lora", got, ref, want);
    ref_dsa_attn(q, lat, cells, f.kv, f.vhead, qk_nope, T, f.n_sel, 2, want);
    rival_f("  rival: the -1 padding counted in the normaliser", got, ref, want);

    cudaFree(d_q); cudaFree(d_cells); cudaFree(d_out); cudaFree(d_lat);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--selftest") != 0) {
            std::fprintf(stderr, "usage: glm_parity [--selftest]\n");
            return 2;
        }
    Dev d;
    std::printf("glm (mHC, KDA, MLA, swiglu) parity:\n");

    // A small embed width keeps the mHC cases fast; the shaping is what is under test, not the size.
    for (int64_t T : {(int64_t) 1, (int64_t) 3, (int64_t) 7}) test_hc(d, 64, 4, 20, T);
    test_hc(d, 32, 8, 3, 2);          // hc 8 and the 3-iteration default, so the loop bound is exercised
    test_hc(d, 16, 2, 2, 1);          // hc 2, and the smallest legal iteration count
    for (int64_t T : {(int64_t) 1, (int64_t) 4}) test_hc_post(d, 48, 4, T);
    test_hc_post(d, 16, 8, 2);

    for (int64_t T : {(int64_t) 1, (int64_t) 5}) test_kda_gate(d, 16, 8, T, -5.0f);
    test_kda_gate(d, 32, 4, 2, -3.0f);          // a different floor, so the constant is not hard-coded
    for (int64_t T : {(int64_t) 1, (int64_t) 6}) test_kda_conv(d, 24, T, 4);
    test_kda_conv(d, 16, 3, 2);   // a kernel other than 4
    for (int64_t T : {(int64_t) 1, (int64_t) 4}) test_kda_l2norm(d, 16, 8, T);
    for (int64_t T : {(int64_t) 1, (int64_t) 5}) test_kda_delta(d, 16, 8, T);
    test_kda_delta(d, 8, 4, 3);

    // MLA: the first token of a sequence (nothing before it), a mid-sequence token with a longer cache than
    // window, and a multi-token window starting at a non-zero position - which is the case that separates
    // `pos_base + t + 1` from every rival.  `kv` 512 is the real latent width, so `NPER` there is 4 and not 1.
    test_mla_attn(d, 3, 1, 8, 1, 0);
    test_mla_attn(d, 2, 1, 8, 9, 8);
    test_mla_attn(d, 4, 4, 8, 12, 8);
    test_mla_attn(d, 2, 1, 512, 3, 2);
    test_mla_attn(d, 2, 2, 512, 5, 3);

    // The FFN's SwiGLU clamp: glm5-next's limit is 10.0 (`swiglu_clamp_exp`/`_shexp`), a second limit is run so
    // the constant is not hard-coded, and 0 is the no-clamp form the first family's packs get.
    test_swiglu(d, 64, 10.0f);
    test_swiglu(d, 128, 10.0f);
    test_swiglu(d, 48, 3.0f);
    test_swiglu(d, 32, 0.0f);

    // DSA: the k-pool indexer.  `select_tail` is run both ways - 1 is what ik_llama.cpp, our ladder oracle,
    // always does, and 0 is upstream's default; the header says why the model cannot settle it.  The single-token
    // case is the streaming shape this engine drives and the multi-token one is a prefill chunk.
    DsaFix f;
    test_dsa_indexer(d, f, 1, 1);
    test_dsa_indexer(d, f, 4, 1);
    test_dsa_indexer(d, f, 1, 0);
    test_dsa_select_ties(d);
    test_dsa_attn(d, f, 1, 0);
    test_dsa_attn(d, f, 1, 1);   // given `kv_lora` as the nope width, the kernel must match that scale
    test_dsa_attn(d, f, 3, 0);

    std::printf("\nglm parity: %d cases, %d failures\n", cases, failures);
    if (failures) return 1;
    std::printf("glm_parity OK\n");
    return 0;
}
