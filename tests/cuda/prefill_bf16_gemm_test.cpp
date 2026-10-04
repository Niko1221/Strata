// prefill_bf16_gemm_test - Gemm::bf16 on the FP16 tensor-core route that cards below sm_80 take (gemm.cu; forced here
// with STRATA_PREFILL_BF16_F16=1 so every CUDA card checks it) against a double-precision product of the same BF16
// values. Shapes of the prompt path's BF16 projections: the hyper-connection down (X converted in several slices),
// up and inject, SSM alpha into a 96-wide row stride with beta = 1, the indexer query, and one product too large for
// the scratch (the BF16 GEMM fallback). X holds some values below 2^-14 (FP16 subnormals). The cells between the
// rows' N and ldy and before the output offset must keep their values.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {
namespace kn = strata::kernels;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

struct Case {
    const char* name;
    int64_t t, n, k, ldy, offset;
    float beta;
};

bool run(strata::prefill::Gemm& g, const Case& c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xd(-2.0f, 2.0f), wd(-0.25f, 0.25f), tiny(-3e-5f, 3e-5f);
    std::vector<uint16_t> x((size_t) (c.t * c.k)), w((size_t) (c.n * c.k));
    for (size_t i = 0; i < x.size(); ++i) x[i] = kn::bf16_from_f32(i % 17 == 0 ? tiny(rng) : xd(rng));
    for (auto& v : w) v = kn::bf16_from_f32(wd(rng));
    const size_t ny = (size_t) (c.offset + c.t * c.ldy);
    std::vector<float> y0(ny);
    for (size_t i = 0; i < ny; ++i) y0[i] = (float) ((int) (i % 23) - 11) * 0.125f;

    Dev dx(x.size() * 2), dw(w.size() * 2), dy(ny * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload x");
    ck(cudaMemcpy(dw.p, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "upload w");
    ck(cudaMemcpy(dy.p, y0.data(), ny * 4, cudaMemcpyHostToDevice), "upload y");
    g.bf16((const uint16_t*) dx.p, (const uint16_t*) dw.p, (float*) dy.p + c.offset, c.t, c.n, c.k, c.ldy, c.beta);
    ck(cudaDeviceSynchronize(), c.name);
    std::vector<float> y(ny);
    ck(cudaMemcpy(y.data(), dy.p, ny * 4, cudaMemcpyDeviceToHost), "download y");

    std::vector<double> xf(x.size()), wf(w.size());
    for (size_t i = 0; i < x.size(); ++i) xf[i] = kn::f32_from_bf16(x[i]);
    for (size_t i = 0; i < w.size(); ++i) wf[i] = kn::f32_from_bf16(w[i]);
    double worst = 0;   // error / (the absolute sum of the products), the scale FP32 accumulation errs on
    int64_t untouched_bad = 0;
    for (int64_t r = 0; r < c.t; ++r) {
        const double* xr = xf.data() + r * c.k;
        for (int64_t col = 0; col < c.ldy; ++col) {
            const size_t at = (size_t) (c.offset + r * c.ldy + col);
            if (col >= c.n) {
                untouched_bad += y[at] != y0[at];
                continue;
            }
            const double* wr = wf.data() + col * c.k;
            double ref = c.beta * (double) y0[at], abs_sum = std::fabs(c.beta * (double) y0[at]);
            for (int64_t i = 0; i < c.k; ++i) {
                ref += xr[i] * wr[i];
                abs_sum += std::fabs(xr[i] * wr[i]);
            }
            worst = std::max(worst, std::fabs((double) y[at] - ref) / std::max(abs_sum, 1e-30));
        }
    }
    for (int64_t i = 0; i < c.offset; ++i) untouched_bad += y[(size_t) i] != y0[(size_t) i];
    const bool ok = worst <= 1e-5 && untouched_bad == 0;
    std::printf("%s %-22s T=%-5lld N=%-5lld K=%-5lld ldy=%-5lld beta=%g: max error %.2e of the products' absolute sum, "
                "cells outside the product changed %lld\n",
                ok ? "PASS" : "FAIL", c.name, (long long) c.t, (long long) c.n, (long long) c.k, (long long) c.ldy,
                c.beta, worst, (long long) untouched_bad);
    return ok;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
#if defined(_WIN32)
    _putenv_s("STRATA_PREFILL_BF16_F16", "1");
#else
    setenv("STRATA_PREFILL_BF16_F16", "1", 1);
#endif
    // 8M elements: the hyper-connection down weight (320 x 10240) leaves room for 499 rows of X, so T = 3000 converts
    // in seven slices; 800 x 10240 leaves less than 256 rows and keeps the BF16 GEMM
    constexpr int64_t kScratch = 8ll << 20;
    strata::prefill::Gemm g;
    std::string err;
    if (!g.init(nullptr, kScratch, err)) {
        std::fprintf(stderr, "Gemm::init: %s\n", err.c_str());
        return 2;
    }
    const Case cases[] = {
        {"hc down (sliced)", 3000, 320, 10240, 320, 0, 0.0f},
        {"hc up", 3000, 10240, 320, 10240, 0, 0.0f},
        {"hc inject", 777, 4, 10240, 4, 5, 0.0f},
        {"ssm alpha, beta 1", 513, 48, 2560, 96, 48, 1.0f},
        {"indexer q", 300, 512, 2560, 512, 0, 0.0f},
        {"scratch too small", 100, 800, 10240, 800, 0, 0.0f},
    };
    bool ok = true;
    uint32_t seed = 1;
    for (const Case& c : cases) ok = run(g, c, seed++) && ok;
    std::puts(ok ? "PASS: Gemm::bf16 through FP16 matches FP64" : "FAIL");
    return ok ? 0 : 1;
}
