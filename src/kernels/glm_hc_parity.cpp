// src/kernels/glm_hc_parity.cpp - the glm5-next mHC kernels vs ref/glm.py, plus the wrong readings.
//
// Same discipline as gr_parity: every prose-reading trap is computed the WRONG way first and required to
// differ materially, because a test that only compares against the right reference passes against either.
// The mHC readings that are easy to get wrong (docs/GLM5-FLASH.md §3.4, glm_hc.hpp):
//
//   1. JOINT vs PER-STREAM rms norm.  gr's norm is one RMS per stream; mHC's is ONE RMS over the whole
//      flattened stack (glm5-next.cpp L472 reshapes to (hc_dim, T) and rms_norms).  The gr reading is the
//      natural port and is wrong here.
//   2. SRC-MAJOR comb rows.  comb[dst, src] = mixes[2hc + dst + hc*src] (ggml-h L2686); reading the rows
//      dst-major transposes the matrix and every hc_post with it.
//   3. THE +eps AFTER THE SOFTMAX.  The Sinkhorn adds hc_eps to the VALUES once the softmax is done, and
//      ALSO guards every divisor with +eps.  Dropping the first one still produces a doubly-stochastic-ish
//      matrix and plausible logits.
//   4. SUM vs MEAN over streams.  mixed = sum_s pre_s * R_s (build_hc_pre); the gr reading is a gated MEAN.
//   5. THE PRE-GATE FLOOR.  pre = sigmoid(...) + hc_eps (ggml_scale_bias(pre, 1, eps)); the post-gate has
//      no floor.  With the real 1e-6 this is invisible, so the structural fixture runs hc_eps = 1e-2.
//   6. 2*sigmoid on the post-gate (centred on 1), not sigmoid.
//
// The positive comparison holds the f32 kernel to ref/glm.py's f64 arithmetic.  Two fixtures: the
// structural one (hc_eps 1e-2, so the eps readings separate) and the real-epsilon one (1e-6, the released
// value - the floor and guards vanish below the f32 noise there, which is itself worth knowing).
#include "strata/kernels/glm_hc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Fixture {
    int n_embd, hc, iters;
    float norm_eps, hc_eps;
    std::vector<float> R, w_fn, w_scale, w_base;
};

Fixture make_fixture(int n_embd, int hc, int iters, float norm_eps, float hc_eps, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Fixture f;
    f.n_embd = n_embd;
    f.hc = hc;
    f.iters = iters;
    f.norm_eps = norm_eps;
    f.hc_eps = hc_eps;
    const int hc_dim = n_embd * hc, mix = (2 + hc) * hc;
    for (int i = 0; i < hc_dim; ++i) f.R.push_back(nd(rng));
    for (int i = 0; i < mix * hc_dim; ++i) f.w_fn.push_back(nd(rng) * 0.02f);
    f.w_scale = {1.0f + 0.1f * nd(rng), 1.0f + 0.1f * nd(rng), 1.0f + 0.1f * nd(rng)};
    for (int i = 0; i < mix; ++i) f.w_base.push_back(nd(rng) * 0.1f);
    return f;
}

// ---- the reference, straight from ref/glm.py (f64 internally, like the oracle)

double dsigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

void ref_sinkhorn(std::vector<double>& comb, int hc, int iters, double eps) {
    for (int s = 0; s < hc; ++s) {   // softmax over dst, per src column
        double mx = -1e300;
        for (int d = 0; d < hc; ++d) mx = std::max(mx, comb[(size_t) d * hc + s]);
        double denom = 0.0;
        for (int d = 0; d < hc; ++d) denom += std::exp(comb[(size_t) d * hc + s] - mx);
        for (int d = 0; d < hc; ++d) comb[(size_t) d * hc + s] = std::exp(comb[(size_t) d * hc + s] - mx) / denom;
    }
    for (auto& v : comb) v += eps;
    auto norm_dst = [&]() {
        for (int d = 0; d < hc; ++d) {
            double sum = eps;
            for (int s = 0; s < hc; ++s) sum += comb[(size_t) d * hc + s];
            for (int s = 0; s < hc; ++s) comb[(size_t) d * hc + s] /= sum;
        }
    };
    auto norm_src = [&]() {
        for (int s = 0; s < hc; ++s) {
            double sum = eps;
            for (int d = 0; d < hc; ++d) sum += comb[(size_t) d * hc + s];
            for (int d = 0; d < hc; ++d) comb[(size_t) d * hc + s] /= sum;
        }
    };
    norm_dst();
    for (int it = 1; it < iters; ++it) {
        norm_src();
        norm_dst();
    }
}

// `comb_dst_major` and `skip_value_eps` and `per_stream_norm` are the WRONG readings - the negative
// tests ask for them by name and require the kernel to differ from them materially.
void ref_hc_pre(const Fixture& f, std::vector<float>& mixed, std::vector<float>& pre,
                std::vector<float>& post, std::vector<float>& comb, bool per_stream_norm = false,
                bool comb_dst_major = false, bool skip_value_eps = false, bool mean_not_sum = false,
                bool no_pre_floor = false) {
    const int hc = f.hc, hc_dim = f.n_embd * hc;
    std::vector<double> inv(hc > 0 && per_stream_norm ? (size_t) hc : 1, 0.0);
    if (per_stream_norm) {
        for (int s = 0; s < hc; ++s) {
            double ms = 0.0;
            for (int e = 0; e < f.n_embd; ++e) ms += (double) f.R[(size_t) s * f.n_embd + e] * f.R[(size_t) s * f.n_embd + e];
            inv[s] = 1.0 / std::sqrt(ms / f.n_embd + f.norm_eps);
        }
    } else {
        double ms = 0.0;
        for (int i = 0; i < hc_dim; ++i) ms += (double) f.R[i] * f.R[i];
        inv[0] = 1.0 / std::sqrt(ms / hc_dim + f.norm_eps);
    }
    std::vector<double> mixes((size_t) (2 + hc) * hc, 0.0);
    for (int m = 0; m < (2 + hc) * hc; ++m) {
        double acc = 0.0;
        for (int k = 0; k < hc_dim; ++k) {
            const int s = k / f.n_embd;
            const double is = per_stream_norm ? inv[s] : inv[0];
            acc += (double) f.R[k] * is * f.w_fn[(size_t) m * hc_dim + k];
        }
        mixes[m] = acc;
    }
    pre.assign(hc, 0.0f);
    for (int s = 0; s < hc; ++s) {
        double g = dsigmoid(mixes[s] * f.w_scale[0] + f.w_base[s]);
        pre[s] = (float) (no_pre_floor ? g : g + f.hc_eps);
    }
    post.assign(hc, 0.0f);
    for (int s = 0; s < hc; ++s) post[s] = (float) (2.0 * dsigmoid(mixes[hc + s] * f.w_scale[1] + f.w_base[hc + s]));
    std::vector<double> comb_raw((size_t) hc * hc, 0.0);
    for (int d = 0; d < hc; ++d)
        for (int s = 0; s < hc; ++s) {
            const int row = comb_dst_major ? d * hc + s : d + hc * s;   // src-major is the real layout
            comb_raw[(size_t) d * hc + s] = mixes[2 * hc + row] * f.w_scale[2] + f.w_base[2 * hc + row];
        }
    ref_sinkhorn(comb_raw, hc, f.iters, skip_value_eps ? 0.0 : f.hc_eps);
    comb.assign(comb_raw.begin(), comb_raw.end());
    mixed.assign(f.n_embd, 0.0f);
    for (int e = 0; e < f.n_embd; ++e) {
        double acc = 0.0;
        for (int s = 0; s < hc; ++s) acc += pre[s] * f.R[(size_t) s * f.n_embd + e];
        mixed[e] = (float) (mean_not_sum ? acc / hc : acc);
    }
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

struct Device {
    float* d = nullptr;
    explicit Device(size_t bytes) { check(cudaMalloc(&d, bytes), "cudaMalloc"); }
    ~Device() { cudaFree(d); }
    float* at(size_t bytes_off) const { return d + bytes_off / sizeof(float); }
};

Device& upload(const Fixture& f, size_t& out_bytes) {
    // one arena: R | w_fn | w_scale | w_base | mixed | pre | post | comb | inv_rms | residual | block | post_out
    const int hc_dim = f.n_embd * f.hc, mix = (2 + f.hc) * f.hc;
    out_bytes = (hc_dim + (size_t) mix * hc_dim + 3 + mix + f.n_embd + 2 * f.hc + (size_t) f.hc * f.hc + 1 +
                 hc_dim + f.n_embd + hc_dim) * sizeof(float);
    static std::unique_ptr<Device> arena;   // resized when a wider fixture needs more
    static size_t have = 0;
    if (!arena || have < out_bytes) {
        arena = std::make_unique<Device>(out_bytes);
        have = out_bytes;
    }
    size_t off = 0;
    auto up = [&](const std::vector<float>& v) {
        check(cudaMemcpy(arena->at(off), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
        off += v.size() * sizeof(float);
    };
    up(f.R);
    up(f.w_fn);
    up(f.w_scale);
    up(f.w_base);
    return *arena;
}

// Run the kernels with the given wrong-reading reference for comparison; returns every output.
void run_kernels(const Fixture& f, std::vector<float>& mixed, std::vector<float>& pre,
                 std::vector<float>& post, std::vector<float>& comb, std::vector<float>& post_out,
                 const std::vector<float>& block_out) {
    const int hc = f.hc, hc_dim = f.n_embd * hc, mix = (2 + hc) * hc;
    size_t bytes = 0;
    Device& arena = upload(f, bytes);
    size_t off = 0;
    const auto* R = arena.at(off); off += hc_dim * sizeof(float);
    const auto* w_fn = arena.at(off); off += (size_t) mix * hc_dim * sizeof(float);
    const auto* w_scale = arena.at(off); off += 3 * sizeof(float);
    const auto* w_base = arena.at(off); off += mix * sizeof(float);
    float* d_mixed = arena.at(off); off += f.n_embd * sizeof(float);
    float* d_pre = arena.at(off); off += hc * sizeof(float);
    float* d_post = arena.at(off); off += hc * sizeof(float);
    float* d_comb = arena.at(off); off += (size_t) hc * hc * sizeof(float);
    float* d_inv = arena.at(off); off += sizeof(float);
    float* d_residual = arena.at(off); off += hc_dim * sizeof(float);
    float* d_block = arena.at(off); off += f.n_embd * sizeof(float);
    float* d_post_out = arena.at(off); off += hc_dim * sizeof(float);
    check(cudaMemcpy(d_residual, f.R.data(), hc_dim * sizeof(float), cudaMemcpyHostToDevice), "residual");
    check(cudaMemcpy(d_block, block_out.data(), f.n_embd * sizeof(float), cudaMemcpyHostToDevice), "block");

    strata::kernels::GlmHcShapes s;
    s.n_embd = f.n_embd;
    s.hc = hc;
    s.sinkhorn_iters = f.iters;
    strata::kernels::GlmHcWorkspace ws;
    strata::kernels::glm_hc_workspace_init(s, d_inv, ws);
    strata::kernels::glm_hc_pre(R, w_fn, w_scale, w_base, f.norm_eps, f.hc_eps, s, ws, d_mixed, d_pre, d_post,
                                d_comb, nullptr);
    check(cudaGetLastError(), "glm_hc_pre launch");
    strata::kernels::glm_hc_post(d_block, d_residual, d_post, d_comb, s, d_post_out, nullptr);
    check(cudaGetLastError(), "glm_hc_post launch");
    check(cudaDeviceSynchronize(), "kernel run");
    if (getenv("GLM_HC_DEBUG")) {
        float probe[5] = {0, 0, 0, 0, 0};
        check(cudaMemcpy(probe, R, 4 * sizeof(float), cudaMemcpyDeviceToHost), "probe R");
        check(cudaMemcpy(probe + 4, d_inv, sizeof(float), cudaMemcpyDeviceToHost), "probe inv");
        std::fprintf(stderr, "debug device R[0..3]: %g %g %g %g | inv_rms %g (host R[0] %g)\n",
                     probe[0], probe[1], probe[2], probe[3], probe[4], f.R[0]);
    }

    mixed.resize(f.n_embd);
    pre.resize(hc);
    post.resize(hc);
    comb.resize((size_t) hc * hc);
    post_out.resize((size_t) hc_dim);
    check(cudaMemcpy(mixed.data(), d_mixed, f.n_embd * sizeof(float), cudaMemcpyDeviceToHost), "mixed");
    check(cudaMemcpy(pre.data(), d_pre, hc * sizeof(float), cudaMemcpyDeviceToHost), "pre");
    check(cudaMemcpy(post.data(), d_post, hc * sizeof(float), cudaMemcpyDeviceToHost), "post");
    check(cudaMemcpy(comb.data(), d_comb, (size_t) hc * hc * sizeof(float), cudaMemcpyDeviceToHost), "comb");
    check(cudaMemcpy(post_out.data(), d_post_out, hc_dim * sizeof(float), cudaMemcpyDeviceToHost), "post_out");
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_hc_parity: %s\n", what.c_str());
        std::exit(1);
    }
}

void test_fixture(int n_embd, int hc, int iters, float hc_eps, bool structural) {
    const Fixture f = make_fixture(n_embd, hc, iters, 1e-5f, hc_eps, 1234u + (uint32_t) n_embd);
    std::vector<float> mixed, pre, post, comb, post_out;
    std::vector<float> block_out((size_t) n_embd);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : block_out) v = nd(rng);
    run_kernels(f, mixed, pre, post, comb, post_out, block_out);

    // ---- positive: the kernel matches the reference at f32-vs-f64 tolerance
    std::vector<float> r_mixed, r_pre, r_post, r_comb;
    ref_hc_pre(f, r_mixed, r_pre, r_post, r_comb);
    const double tol = 5e-4;
    if (getenv("GLM_HC_DEBUG")) {
        std::fprintf(stderr, "debug mixed[0..3]: kernel %.6f %.6f %.6f %.6f | ref %.6f %.6f %.6f %.6f\n",
                     mixed[0], mixed[1], mixed[2], mixed[3], r_mixed[0], r_mixed[1], r_mixed[2], r_mixed[3]);
        std::fprintf(stderr, "debug pre: kernel %.6f %.6f %.6f %.6f | ref %.6f %.6f %.6f %.6f\n",
                     pre[0], pre[1], pre[2], pre[3], r_pre[0], r_pre[1], r_pre[2], r_pre[3]);
        std::fprintf(stderr, "debug post: kernel %.6f %.6f %.6f %.6f | ref %.6f %.6f %.6f %.6f\n",
                     post[0], post[1], post[2], post[3], r_post[0], r_post[1], r_post[2], r_post[3]);
        std::fprintf(stderr, "debug comb: kernel %.6f %.6f %.6f %.6f | ref %.6f %.6f %.6f %.6f\n",
                     comb[0], comb[1], comb[4], comb[5], r_comb[0], r_comb[1], r_comb[4], r_comb[5]);
        std::fprintf(stderr, "debug deltas: mixed %.3e pre %.3e post %.3e comb %.3e\n",
                     max_abs(mixed, r_mixed), max_abs(pre, r_pre), max_abs(post, r_post), max_abs(comb, r_comb));
    }
    require(max_abs(mixed, r_mixed) < tol, "mixed diverges from ref/glm.py");
    require(max_abs(pre, r_pre) < tol, "pre diverges from ref/glm.py");
    require(max_abs(post, r_post) < tol, "post diverges from ref/glm.py");
    require(max_abs(comb, r_comb) < tol, "comb diverges from ref/glm.py");

    // the write half vs its reference
    std::vector<float> r_post_out((size_t) n_embd * hc, 0.0f);
    for (int d = 0; d < hc; ++d)
        for (int e = 0; e < n_embd; ++e) {
            double acc = block_out[e] * post[d];
            for (int s = 0; s < hc; ++s) acc += comb[(size_t) d * hc + s] * f.R[(size_t) s * n_embd + e];
            r_post_out[(size_t) d * n_embd + e] = (float) acc;
        }
    require(max_abs(post_out, r_post_out) < tol, "glm_hc_post diverges from ref/glm.py");

    // the Sinkhorn's dst rows sum to 1 at the working precision (the structural signature)
    for (int d = 0; d < hc; ++d) {
        double sum = 0.0;
        for (int s = 0; s < hc; ++s) sum += comb[(size_t) d * hc + s];
        require(std::fabs(sum - 1.0) < 4.0 * hc_eps + 1e-5, "sinkhorn dst rows do not sum to 1");
    }

    if (!structural) return;   // the real-epsilon fixture only does the positive comparison

    // ---- negative: each wrong reading must differ MATERIALLY (10x the positive tolerance)
    std::vector<float> w_mixed, w_pre, w_post, w_comb;
    ref_hc_pre(f, w_mixed, w_pre, w_post, w_comb, /*per_stream_norm=*/true);
    require(max_abs(r_mixed, w_mixed) > 10 * tol, "joint and per-stream norms are indistinguishable");
    ref_hc_pre(f, w_mixed, w_pre, w_post, w_comb, false, /*comb_dst_major=*/true);
    require(max_abs(r_comb, w_comb) > 10 * tol, "src-major and dst-major combs are indistinguishable");
    ref_hc_pre(f, w_mixed, w_pre, w_post, w_comb, false, false, /*skip_value_eps=*/true);
    require(max_abs(r_comb, w_comb) > 10 * tol, "the post-softmax +eps is invisible - enlarge hc_eps");
    ref_hc_pre(f, w_mixed, w_pre, w_post, w_comb, false, false, false, /*mean_not_sum=*/true);
    require(max_abs(r_mixed, w_mixed) > 10 * tol, "sum and mean over streams are indistinguishable");
    ref_hc_pre(f, w_mixed, w_pre, w_post, w_comb, false, false, false, false, /*no_pre_floor=*/true);
    require(max_abs(r_pre, w_pre) > 10 * tol, "the pre-gate floor is invisible - enlarge hc_eps");
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    if (!selftest) {
        std::fprintf(stderr, "glm_hc_parity: run with --selftest (the CTest form)\n");
        return 2;
    }
    // the structural fixture: hc_eps exaggerated to 3e-2 so the eps readings separate from f32 noise
    test_fixture(128, 4, 3, 3e-2f, true);
    test_fixture(128, 4, 20, 1e-6f, false);   // the real shape of the constants: 20 Sinkhorn iterations
    test_fixture(256, 4, 20, 1e-6f, false);   // and a wider stack
    std::printf("glm_hc_parity: PASS\n");
    return 0;
}
