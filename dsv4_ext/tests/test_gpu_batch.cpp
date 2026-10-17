// dsv4 test: the batched device kernels, one operation at a time, against the host reference.
//
// test_prefill_batch tells you THAT the device path differs from the CPU one; this tells you WHICH
// operation does it: gpu::matmul (identity / strided / gathered) and gpu::experts_batch, for F32 and
// Q8_0 weights, so no model file is needed. On the emulated backend it passes trivially; the point is to
// run it on the real device (build.sh --cuda builds and runs it as test_gpu_batch_cuda).
#include "dsv4/dequant.hpp"
#include "dsv4/gpu.hpp"
#include "dsv4/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace dsv4;

static int g_fail = 0;

static uint16_t f2h(float f) {   // plain float -> IEEE half, enough for scales in [1e-3, 1]
    uint32_t b; std::memcpy(&b, &f, 4);
    const int e = (int) ((b >> 23) & 0xff) - 127 + 15;
    return (uint16_t) (((b >> 16) & 0x8000) | ((uint32_t) std::max(1, std::min(30, e)) << 10) | ((b >> 13) & 0x3ff));
}

/// Random weights of `rows` x `in` for `type`, as the raw bytes the kernels read.
static std::vector<uint8_t> make_weights(uint32_t type, int rows, int in, std::mt19937& rng) {
    const size_t rb = row_bytes(type, in);
    std::vector<uint8_t> w((size_t) rows * rb, 0);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::uniform_int_distribution<int> Q(-100, 100);
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = w.data() + (size_t) r * rb;
        if (type == T_F32) {
            for (int i = 0; i < in; ++i) { const float v = U(rng) * 0.5f; std::memcpy(p + (size_t) i * 4, &v, 4); }
        } else {   // Q8_0: half scale + 32 int8
            for (int b = 0; b < in / 32; ++b) {
                const uint16_t d = f2h(0.004f + 0.004f * std::fabs(U(rng)));
                std::memcpy(p + (size_t) b * 34, &d, 2);
                for (int i = 0; i < 32; ++i) p[(size_t) b * 34 + 2 + i] = (uint8_t) (int8_t) Q(rng);
            }
        }
    }
    return w;
}

static std::vector<float> rand_vec(size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = U(rng);
    return v;
}

static double max_err(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double) a[i] - (double) b[i]) / std::max(1.0, std::fabs((double) b[i])));
    return m;
}

static void report(const char* what, uint32_t type, double err, double tol = 2e-4) {
    const bool ok = err <= tol;
    std::printf("%s %-34s %-5s max rel err %.3e\n", ok ? "ok  " : "FAIL", what, type == T_F32 ? "F32" : "Q8_0", err);
    if (!ok) ++g_fail;
}

template <class T> static T* dev_copy(const T* h, size_t n) {
    T* d = (T*) gpu::alloc(n * sizeof(T));
    if (d) gpu::h2d(d, h, n * sizeof(T));
    return d;
}

static void test_matmul(uint32_t type, std::mt19937& rng) {
    const int rows = 37, in = 192;     // 192 = 6 Q8_0 blocks, and not a multiple of the 256 threads
    const auto W = make_weights(type, rows, in, rng);
    const size_t rb = row_bytes(type, in);
    uint8_t* dW = dev_copy(W.data(), W.size());
    for (int T : {1, 2, 5, 13}) {
        for (int pad : {0, 7}) {       // pad > 0: x_stride/y_stride bigger than the row, as in o_a groups
            const int xs = in + pad, ys = rows + pad;
            const auto X = rand_vec((size_t) T * xs, rng);
            std::vector<float> ref((size_t) T * ys, 0.f), got((size_t) T * ys, 0.f);
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < rows; ++r)
                    ref[(size_t) t * ys + r] = row_dot(type, W.data() + (size_t) r * rb, X.data() + (size_t) t * xs, in);
            float* dX = dev_copy(X.data(), X.size());
            float* dY = dev_copy(got.data(), got.size());
            const bool ok = dW && dX && dY && gpu::matmul(type, dW, rows, in, dX, T, nullptr, dY, xs, ys);
            if (!ok) { std::printf("FAIL matmul returned false (T=%d pad=%d)\n", T, pad); ++g_fail; continue; }
            gpu::d2h(got.data(), dY, got.size() * 4);
            double e = 0;
            for (int t = 0; t < T; ++t) e = std::max(e, max_err(&got[(size_t) t * ys], &ref[(size_t) t * ys], rows));
            char name[64]; std::snprintf(name, sizeof name, "matmul T=%d stride pad=%d", T, pad);
            report(name, type, e);
            gpu::release(dX); gpu::release(dY);
        }
    }
    // gather: token k reads row sel[k] of X
    {
        const int T = 6; const int32_t sel[6] = {4, 0, 3, 3, 1, 5};
        const auto X = rand_vec((size_t) 8 * in, rng);
        std::vector<float> ref((size_t) T * rows), got((size_t) T * rows, 0.f);
        for (int k = 0; k < T; ++k)
            for (int r = 0; r < rows; ++r)
                ref[(size_t) k * rows + r] = row_dot(type, W.data() + (size_t) r * rb, X.data() + (size_t) sel[k] * in, in);
        float* dX = dev_copy(X.data(), X.size());
        float* dY = dev_copy(got.data(), got.size());
        if (!gpu::matmul(type, dW, rows, in, dX, T, sel, dY, in, rows)) { std::printf("FAIL gather matmul returned false\n"); ++g_fail; }
        else { gpu::d2h(got.data(), dY, got.size() * 4); report("matmul gather", type, max_err(got.data(), ref.data(), got.size())); }
        gpu::release(dX); gpu::release(dY);
    }
    gpu::release(dW);
}

static void test_experts(uint32_t type, std::mt19937& rng) {
    const int dim = 64, ff = 32, T = 9, NE = 3;
    const float lim = 3.0f;
    std::vector<std::vector<uint8_t>> G, U, D;
    for (int e = 0; e < NE; ++e) { G.push_back(make_weights(type, ff, dim, rng)); U.push_back(make_weights(type, ff, dim, rng)); D.push_back(make_weights(type, dim, ff, rng)); }
    // token -> experts: each expert gets a different bucket, a token can be in several buckets
    const std::vector<std::vector<int32_t>> bucket = {{0, 2, 3, 8}, {1, 2, 5, 6, 8}, {0, 4, 7}};
    const auto X = rand_vec((size_t) T * dim, rng);
    std::vector<std::vector<float>> W;
    for (auto& b : bucket) { std::vector<float> w; for (size_t i = 0; i < b.size(); ++i) w.push_back(0.1f + 0.2f * (float) i); W.push_back(w); }

    std::vector<float> ref((size_t) T * dim, 0.f);
    const size_t rg = row_bytes(type, dim), rd = row_bytes(type, ff);
    for (int e = 0; e < NE; ++e)
        for (size_t k = 0; k < bucket[e].size(); ++k) {
            const float* x = X.data() + (size_t) bucket[e][k] * dim;
            std::vector<float> g(ff), u(ff), a(ff);
            for (int r = 0; r < ff; ++r) { g[r] = row_dot(type, G[e].data() + r * rg, x, dim); u[r] = row_dot(type, U[e].data() + r * rg, x, dim); }
            swiglu_clamped(g.data(), u.data(), ff, lim, a.data());
            for (float& v : a) v *= W[e][k];
            for (int r = 0; r < dim; ++r) ref[(size_t) bucket[e][k] * dim + r] += row_dot(type, D[e].data() + r * rd, a.data(), ff);
        }

    std::vector<uint8_t*> dG(NE), dU(NE), dD(NE);
    std::vector<gpu::ExpBatch> eb(NE);
    for (int e = 0; e < NE; ++e) {
        dG[e] = dev_copy(G[e].data(), G[e].size()); dU[e] = dev_copy(U[e].data(), U[e].size()); dD[e] = dev_copy(D[e].data(), D[e].size());
        eb[e].gate = dG[e]; eb[e].up = dU[e]; eb[e].down = dD[e];
        eb[e].tok = bucket[e].data(); eb[e].w = W[e].data(); eb[e].n = (int) bucket[e].size();
    }
    float* dX = dev_copy(X.data(), X.size());
    float* dg = (float*) gpu::alloc((size_t) T * ff * 4); float* du = (float*) gpu::alloc((size_t) T * ff * 4);
    float* da = (float*) gpu::alloc((size_t) T * ff * 4); float* dn = (float*) gpu::alloc((size_t) T * dim * 4);
    float* dY = (float*) gpu::alloc((size_t) T * dim * 4);
    std::vector<float> got((size_t) T * dim, -7.f);
    if (!gpu::experts_batch(eb.data(), NE, type, type, type, ff, dim, lim, dX, T, dg, du, da, dn, dY)) {
        std::printf("FAIL experts_batch returned false\n"); ++g_fail;
    } else {
        gpu::d2h(got.data(), dY, got.size() * 4);
        report("experts_batch (device experts)", type, max_err(got.data(), ref.data(), got.size()));
    }
}

int main() {
    std::string err;
    if (!gpu::init(err)) { std::printf("SKIP: no device (%s)\n", err.c_str()); return 0; }
    std::printf("device: %s\n", gpu::is_emulated() ? "CPU-emulated" : "CUDA");
    std::mt19937 rng(1234);
    for (uint32_t type : {(uint32_t) T_F32, (uint32_t) T_Q8_0}) {
        if (!gpu::type_supported(type)) continue;
        test_matmul(type, rng);
        test_experts(type, rng);
    }
    std::printf("%s\n", g_fail ? "GPU BATCH TESTS FAILED" : "ALL GPU BATCH TESTS PASSED");
    return g_fail ? 1 : 0;
}
