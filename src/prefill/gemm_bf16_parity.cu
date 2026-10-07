// src/prefill/gemm_bf16_parity.cu - Gemm::bf16 against cuBLAS's own BF16 product (GPU, synthetic, no model).
//
// Below sm_80 (no BF16 tensor cores) Gemm::bf16 converts the weight and the activations to FP16 and runs the FP16
// tensor-core GEMM (Volta by default, Turing with STRATA_BF16_TC=1) or widens them to fp32 (Pascal); everywhere else
// it is the cuBLAS BF16 call itself.  (On Pascal the cuBLAS BF16 reference itself may be refused: the test then fails
// at the reference, which says so.)  This
// checks the result against cublasGemmEx on the same BF16 inputs, within fp32-accumulation rounding, over the prompt
// path's shapes: the hyper-connection down / up projections, the router and indexer rows, the PLE value matrix, a T
// large enough to slice the activations, beta = 1 accumulation (the bf16x2 low parts) and an output row stride wider
// than N.  --bench adds the time of each against the cuBLAS BF16 product.
#include "strata/prefill/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <memory>
#include <string>
#include <vector>

namespace {

uint16_t to_bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

bool ok_or(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

struct Shape {
    int64_t T, N, K, ldy;
    const char* what;
};

void environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

int cpu_checks() {
    using strata::prefill::Gemm;
    int failures = 0;
    auto check = [&](bool pass, const char* what) {
        std::printf("  %-40s %s\n", what, pass ? "pass" : "FAIL");
        failures += !pass;
    };
    environment("STRATA_BF16_TC_CACHE_MIB", nullptr);
    environment("STRATA_BF16_TC_CACHE", nullptr);
    check(Gemm::bf16_cache_budget_bytes() == 0, "cache absent");
    environment("STRATA_BF16_TC_CACHE", "true");
    check(Gemm::bf16_cache_budget_bytes() == 0, "strict opt-in");
    environment("STRATA_BF16_TC_CACHE", "1");
    check(Gemm::bf16_cache_budget_bytes() == (256u << 20), "default 256 MiB reservation");
    environment("STRATA_BF16_TC_CACHE_MIB", "1");
    check(Gemm::bf16_cache_budget_bytes() == (1u << 20), "minimum budget");
    environment("STRATA_BF16_TC_CACHE_MIB", "65536");
    check(Gemm::bf16_cache_budget_bytes() == (size_t(65536) << 20), "maximum budget");
    for (const char* value : {"", "0", "-1", " 1", "+1", "1junk", "65537", "18446744073709551616"}) {
        environment("STRATA_BF16_TC_CACHE_MIB", value);
        check(Gemm::bf16_cache_budget_bytes() == 0, value);
    }
    std::printf("gemm_bf16_parity CPU-only: %d failures (no CUDA calls)\n", failures);
    return failures ? 1 : 0;
}

// Run separately with STRATA_BF16_TC=2 STRATA_BF16_TC_CACHE=1 STRATA_BF16_TC_CACHE_MIB=1.
int cache_checks(strata::prefill::Gemm& gemm, cudaStream_t st) {
    constexpr int64_t T = 8, N = 32, K = 64;
    const auto initial = gemm.bf16_cache_stats();
    if (!initial.budget || initial.budget > (4u << 20)) {
        std::printf("FAIL: --cache-test requires active BF16_TC cache, budget 1..4 MiB\n");
        return 1;
    }
    int failures = 0;
    auto check = [&](bool pass, const char* what) {
        std::printf("  cache %-34s %s\n", what, pass ? "pass" : "FAIL");
        failures += !pass;
    };
    check(strata::prefill::Gemm::bf16_cache_reserve_bytes() == initial.budget,
          "startup reserve matches arena");
    std::vector<uint16_t> x(T * K, to_bf16(0.25f)), w(N * K, to_bf16(0.125f));
    std::mt19937 rng(75);
    std::normal_distribution<float> distribution(0.0f, 0.25f);
    for (auto& value : x) value = to_bf16(distribution(rng));
    for (auto& value : w) value = to_bf16(distribution(rng));
    uint16_t *dx = nullptr, *dw = nullptr, *dm = nullptr, *dbig = nullptr;
    float *dy = nullptr, *dr = nullptr, *ybig = nullptr, *rbig = nullptr;
    const int64_t big_n = (int64_t) (initial.budget / (K * 2)) + 1;
    const size_t big_w_bytes = (size_t) big_n * K * 2, big_y_bytes = (size_t) T * big_n * 4;
    if (!ok_or(cudaMalloc((void**) &dx, x.size() * 2), "cache x") ||
        !ok_or(cudaMalloc((void**) &dw, w.size() * 2), "cache w") ||
        !ok_or(cudaMalloc((void**) &dm, w.size() * 2), "mutable w") ||
        !ok_or(cudaMalloc((void**) &dy, T * N * 2 * 4), "cache y") ||
        !ok_or(cudaMalloc((void**) &dr, T * N * 2 * 4), "cache ref") ||
        !ok_or(cudaMalloc((void**) &dbig, big_w_bytes), "budget w") ||
        !ok_or(cudaMalloc((void**) &ybig, big_y_bytes), "budget y") ||
        !ok_or(cudaMalloc((void**) &rbig, big_y_bytes), "budget ref")) return 1;
    auto upload = [&](uint16_t* dst, const std::vector<uint16_t>& src) {
        return ok_or(cudaMemcpyAsync(dst, src.data(), src.size() * 2, cudaMemcpyHostToDevice, st), "cache upload");
    };
    auto parity = [&](int64_t n = 0, int64_t k = 0) {
        if (!n) n = N;
        if (!k) k = K;
        gemm.bf16(dx, dw, dr, T, n, k);
        if (!ok_or(cudaStreamSynchronize(st), "cache sync")) return false;
        std::vector<float> y(T * n), r(y.size());
        if (!ok_or(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "cache read") ||
            !ok_or(cudaMemcpy(r.data(), dr, r.size() * 4, cudaMemcpyDeviceToHost), "cache ref read")) return false;
        return std::memcmp(y.data(), r.data(), y.size() * 4) == 0;
    };
    if (!upload(dx, x) || !upload(dw, w) || !upload(dm, w)) return 1;
    gemm.bf16_immutable(dx, dw, dy, T, N, K);
    gemm.bf16_immutable(dx, dw, dy, T, N, K);
    check(parity(), "repeat cached/uncached bit parity");
    auto s = gemm.bf16_cache_stats();
    check(s.hits == 1 && s.misses == 1 && s.conversions == 1, "one conversion, one hit");
    std::fill(x.begin(), x.end(), to_bf16(-0.5f));
    if (!upload(dx, x)) return 1;
    gemm.bf16_immutable(dx, dw, dy, T, N, K);
    check(parity(), "changed activations never cached");
    const auto before_mutable = gemm.bf16_cache_stats();
    gemm.bf16(dx, dm, dy, T, N, K);
    std::fill(w.begin(), w.end(), to_bf16(-0.25f));
    if (!upload(dm, w)) return 1;
    gemm.bf16(dx, dm, dy, T, N, K);
    if (!ok_or(cudaStreamSynchronize(st), "mutable sync")) return 1;
    std::vector<float> mutable_y(T * N);
    if (!ok_or(cudaMemcpy(mutable_y.data(), dy, mutable_y.size() * 4, cudaMemcpyDeviceToHost), "mutable read")) return 1;
    check(std::all_of(mutable_y.begin(), mutable_y.end(), [](float v) { return v == 8.0f; }),
          "nonimmutable weight mutation");
    s = gemm.bf16_cache_stats();
    check(s.hits == before_mutable.hits && s.misses == before_mutable.misses, "nonimmutable inputs bypass cache");
    // Same pointer and element count, different weight shape; X has ample storage for the narrower K.
    gemm.bf16_immutable(dx, dw, dy, T, N * 2, K / 2);
    check(parity(N * 2, K / 2), "same-size different-shape parity");
    s = gemm.bf16_cache_stats();
    check(s.conversions == 2 && s.misses == 2, "shape is part of cache key");
    if (!ok_or(cudaMemsetAsync(dbig, 0, big_w_bytes, st), "budget fill")) return 1;
    const auto before_budget = gemm.bf16_cache_stats();
    for (int i = 0; i < 2; ++i) gemm.bf16_immutable(dx, dbig, ybig, T, big_n, K);
    gemm.bf16(dx, dbig, rbig, T, big_n, K);
    if (!ok_or(cudaStreamSynchronize(st), "budget sync")) return 1;
    std::vector<float> by(T * big_n), br(by.size());
    if (!ok_or(cudaMemcpy(by.data(), ybig, big_y_bytes, cudaMemcpyDeviceToHost), "budget read") ||
        !ok_or(cudaMemcpy(br.data(), rbig, big_y_bytes, cudaMemcpyDeviceToHost), "budget ref read")) return 1;
    check(std::memcmp(by.data(), br.data(), big_y_bytes) == 0, "oversize fallback parity");
    s = gemm.bf16_cache_stats();
    check(s.budget_misses == before_budget.budget_misses + 2 &&
          s.conversions == before_budget.conversions && s.bytes <= s.budget, "budget bound and misses");
    gemm.bf16_immutable(dx, dw, dy, T, N, K);
    check(gemm.bf16_cache_stats().hits == s.hits + 1, "budget miss retains admitted weight");
    bool reset_pass = true;
    for (int i = 0; i < 64; ++i) {
        // This is also pointer-reuse stress: the same source address now denotes a different weight generation.
        gemm.invalidate_bf16_cache();
        reset_pass &= gemm.bf16_cache_stats().bytes == 0;
        std::fill(w.begin(), w.end(), to_bf16((i & 1) ? 0.125f : -0.25f));
        if (!upload(dw, w)) return 1;
        gemm.bf16_immutable(dx, dw, dy, T, N, K);
        gemm.bf16_immutable(dx, dw, dy, T, N, K);
        reset_pass &= parity();
        reset_pass &= gemm.bf16_cache_stats().bytes == w.size() * 2;
    }
    check(reset_pass && gemm.bf16_cache_stats().invalidations == 64, "invalidate/evict-all/reuse stress");
    cudaStream_t other = nullptr;
    cudaEvent_t ready = nullptr;
    if (!ok_or(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "other stream") ||
        !ok_or(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "source ready")) return 1;
    {
        strata::prefill::Gemm other_gemm;
        std::string error;
        if (!other_gemm.init(other, 0, error)) { std::printf("FAIL: %s\n", error.c_str()); return 1; }
        std::fill(x.begin(), x.end(), to_bf16(0.75f));
        if (!upload(dx, x) || !ok_or(cudaEventRecord(ready, st), "source event") ||
            !ok_or(cudaStreamWaitEvent(other, ready, 0), "source wait")) return 1;
        gemm.bf16_immutable(dx, dw, dy, T, N, K);
        other_gemm.bf16_immutable(dx, dw, dr, T, N, K);
        other_gemm.bf16_immutable(dx, dw, dr, T, N, K);
        if (!ok_or(cudaStreamSynchronize(st), "first stream sync") ||
            !ok_or(cudaStreamSynchronize(other), "other sync")) return 1;
        std::vector<float> cy(T * N), cr(cy.size());
        if (!ok_or(cudaMemcpy(cy.data(), dy, cy.size() * 4, cudaMemcpyDeviceToHost), "cross read") ||
            !ok_or(cudaMemcpy(cr.data(), dr, cr.size() * 4, cudaMemcpyDeviceToHost), "cross ref")) return 1;
        const auto os = other_gemm.bf16_cache_stats();
        check(os.conversions == 1 && os.hits == 1 && std::memcmp(cy.data(), cr.data(), cy.size() * 4) == 0,
              "separate stream owners and ordering");
    }
    cudaEventDestroy(ready);
    cudaStreamDestroy(other);
    s = gemm.bf16_cache_stats();
    std::printf("  cache counters: hits=%llu misses=%llu conversions=%llu budget_misses=%llu "
                "invalidations=%llu bytes=%zu budget=%zu\n",
                (unsigned long long) s.hits, (unsigned long long) s.misses, (unsigned long long) s.conversions,
                (unsigned long long) s.budget_misses, (unsigned long long) s.invalidations, s.bytes, s.budget);
    // Evict entries before freeing the source allocations to satisfy the immutable lifetime contract.
    gemm.invalidate_bf16_cache();
    cudaFree(dx); cudaFree(dw); cudaFree(dm); cudaFree(dbig);
    cudaFree(dy); cudaFree(dr); cudaFree(ybig); cudaFree(rbig);
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--cpu-only") return cpu_checks();
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool cache_shapes = argc > 1 && std::string(argv[1]) == "--cache-shapes";
    int dev = 0, maj = 0, min = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev);
    std::printf("gemm_bf16_parity: compute capability %d.%d\n", maj, min);

    cudaStream_t st = nullptr;
    if (!ok_or(cudaStreamCreate(&st), "stream")) return 1;
    auto owned_gemm = std::make_unique<strata::prefill::Gemm>();
    auto& gemm = *owned_gemm;
    std::string err;
    if (!gemm.init(st, 0, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    if (gemm.bf16_cache_stats().budget != strata::prefill::Gemm::bf16_cache_reserve_bytes()) {
        std::printf("FAIL: startup reserve differs from allocated cache budget\n");
        return 1;
    }
    if (cache_shapes && !gemm.bf16_cache_stats().budget) {
        std::printf("FAIL: --cache-shapes requires active BF16_TC conversion cache\n");
        return 1;
    }
    if (argc > 1 && std::string(argv[1]) == "--reserve-test") {
        owned_gemm.reset();
        cudaStreamDestroy(st);
        std::printf("gemm_bf16_parity: cache reservation matches active conversion path\n");
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--cache-test") {
        const int result = cache_checks(gemm, st);
        owned_gemm.reset();
        cudaStreamDestroy(st);
        return result;
    }
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::printf("FAIL: cublasCreate\n"); return 1; }
    cublasSetStream(h, st);

    const Shape shapes[] = {
        {4096, 320, 10240, 0, "hc down (activations sliced)"},
        {4096, 10240, 320, 0, "hc up"},
        {777, 2560, 2560, 0, "PLE value"},
        {512, 512, 2560, 0, "router / indexer q"},
        {300, 4, 10240, 0, "hc inject"},
        {2048, 1, 2560, 0, "single row"},
        {333, 128, 2560, 136, "row stride wider than N"},
    };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int failures = 0;
    for (const Shape& s : shapes) {
        const int64_t ldy = s.ldy > 0 ? s.ldy : s.N;
        std::vector<uint16_t> x((size_t) (s.T * s.K)), xlo(x.size()), w((size_t) (s.N * s.K));
        for (size_t i = 0; i < x.size(); ++i) {
            const float v = 3.0f * nd(rng);          // activations: normalized, a few units
            x[i] = to_bf16(v);
            xlo[i] = to_bf16(1e-3f * nd(rng));       // a bf16x2 low part
        }
        for (auto& v : w) v = to_bf16(0.02f * nd(rng));
        uint16_t *dx = nullptr, *dxlo = nullptr, *dw = nullptr;
        float *dy = nullptr, *dr = nullptr;
        const size_t ybytes = (size_t) (s.T * ldy) * 4;
        bool ok = ok_or(cudaMalloc((void**) &dx, x.size() * 2), "x") && ok_or(cudaMalloc((void**) &dxlo, x.size() * 2), "xlo") &&
                  ok_or(cudaMalloc((void**) &dw, w.size() * 2), "w") && ok_or(cudaMalloc((void**) &dy, ybytes), "y") &&
                  ok_or(cudaMalloc((void**) &dr, ybytes), "ref");
        if (!ok) return 1;
        cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dxlo, xlo.data(), xlo.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
        cudaMemset(dy, 0, ybytes);
        cudaMemset(dr, 0, ybytes);

        // the product under test: X . W^T, then the low part added with beta = 1
        gemm.bf16_immutable(dx, dw, dy, s.T, s.N, s.K, ldy);
        gemm.bf16(dxlo, dw, dy, s.T, s.N, s.K, ldy, 1.0f);
        // the reference: cuBLAS on the BF16 inputs, the same two calls
        const float one = 1.0f, zero = 0.0f;
        auto ref = [&](const uint16_t* X, float beta) {
            if (cache_shapes) {
                gemm.bf16(X, dw, dr, s.T, s.N, s.K, ldy, beta);
                return CUBLAS_STATUS_SUCCESS;
            }
            return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) s.N, (int) s.T, (int) s.K, &one, dw, CUDA_R_16BF,
                                (int) s.K, X, CUDA_R_16BF, (int) s.K, &beta, dr, CUDA_R_32F, (int) ldy,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        if (ref(dx, zero) != CUBLAS_STATUS_SUCCESS || ref(dxlo, one) != CUBLAS_STATUS_SUCCESS) {
            std::printf("FAIL: reference cublasGemmEx\n");
            return 1;
        }
        if (!ok_or(cudaStreamSynchronize(st), "sync")) return 1;
        std::vector<float> y((size_t) (s.T * ldy)), r(y.size());
        cudaMemcpy(y.data(), dy, ybytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(r.data(), dr, ybytes, cudaMemcpyDeviceToHost);
        double worst = 0.0, mag = 1e-30;
        int bad = 0;
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t c = 0; c < s.N; ++c) {
                const float a = y[(size_t) (t * ldy + c)], b = r[(size_t) (t * ldy + c)];
                if (!std::isfinite(a)) ++bad;
                worst = std::max(worst, (double) std::fabs(a - b));
                mag = std::max(mag, (double) std::fabs(b));
            }
        // the untouched columns of a wider row stride stay as they were
        for (int64_t t = 0; t < s.T && ldy > s.N; ++t)
            for (int64_t c = s.N; c < ldy; ++c)
                if (y[(size_t) (t * ldy + c)] != 0.0f) ++bad;
        const bool pass = bad == 0 && (cache_shapes ? worst == 0.0 : worst <= 1e-4 * mag);
        if (!pass) ++failures;
        std::printf("  %-32s T %5lld N %5lld K %5lld ldy %5lld: worst |diff| %.3e of max |ref| %.3e (rel %.2e) %s\n",
                    s.what, (long long) s.T, (long long) s.N, (long long) s.K, (long long) ldy, worst, mag, worst / mag,
                    pass ? "pass" : "FAIL");

        if (bench) {
            auto time = [&](auto&& f) {
                for (int i = 0; i < 3; ++i) f();
                cudaStreamSynchronize(st);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) f();
                cudaStreamSynchronize(st);
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
            };
            const double us_gemm = time([&] { gemm.bf16_immutable(dx, dw, dy, s.T, s.N, s.K, ldy); });
            const double us_ref = time([&] { ref(dx, zero); });
            std::printf("      bench: Gemm::bf16 %9.1f us   cuBLAS BF16 %9.1f us   (%.2fx)\n", us_gemm, us_ref,
                        us_ref / us_gemm);
        }
        gemm.report_bf16_cache();
        gemm.invalidate_bf16_cache();
        cudaFree(dx); cudaFree(dxlo); cudaFree(dw); cudaFree(dy); cudaFree(dr);
    }
    cublasDestroy(h);
    owned_gemm.reset();
    cudaStreamDestroy(st);
    std::printf("gemm_bf16_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
