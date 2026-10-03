// bench/bf16_gemm_bench.cu - the V100 (sm_70) BF16 GEMM micro benchmark: the BF16 cuBLAS call against the
// FP16 fallback this branch adds (`Gemm::bf16` on cc < 8.0), on the shapes the prompt path really uses.
//
// WHAT IT MEASURES, per shape (T x K by N, X row-major T x K, W row-major N x K, Y = X . W^T in fp32):
//
//   * bf16     - `Gemm::bf16` with STRATA_PREFILL_BF16_F16=0: the CUDA_R_16BF cublasGemmEx (the sm_75/80+
//                and old-V100 path, unchanged).
//   * f16      - `Gemm::bf16` with STRATA_PREFILL_BF16_F16=1: operands through f16_from_bf16 and the FP16
//                GEMM, W served from its cached FP16 twin after the first call.
//   * nocache  - the same FP16 path with STRATA_PREFILL_BF16_TWINS_MB=0: W converts per call into the
//                scratch.  What the twin cache buys (and the accuracy floor is the same either way).
//   * f16 cold - the same, but timing the FIRST call (W conversion included) so the twin's one-time cost
//                is visible instead of hidden by the cache.
//   * f16 raw  - the FP16 GEMM alone on pre-converted operands: the floor the fallback approaches.
//
// NUMERICS: an fp32 reference (cublasGemmEx, CUDA_R_32F, the same operands widened) and the max abs/rel
// difference of each path against it.  Both paths multiply the same numbers - bf16 -> f16 is exact for
// 2^-14 <= |x| <= 65280 - so the differences are accumulation order only, and both land in the fp32 noise.
//
// EXPERIMENT (report only, the default is not changed): the FP16 GEMM with CUBLAS_COMPUTE_16F and with
// CUBLAS_GEMM_DEFAULT_TENSOR_OP against the shipped CUBLAS_COMPUTE_32F + CUBLAS_DEFAULT_MATH.
//
// Run it on one GPU: `flock /tmp/opencode/gpu.lock -c 'CUDA_VISIBLE_DEVICES=0 ./bf16_gemm_bench'`.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::prefill::Gemm;

namespace {

constexpr int kIters = 30;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// The host-side conversion check: f16_from_bf16 must be exact through the fp16 range, clamp +-65504 past it,
// and pass inf/NaN through.  This is the property the whole fallback rests on; it is checked on the host so
// a failure names the bits and not a "GEMM looked off".
int check_conversion() {
    int bad = 0;
    // In-range values must survive EXACTLY: bf16_from_f32 of the value in, f16_from_bf16 of those bits, and
    // the result must be the very f16 bits f16_from_f32 produces for the value the bf16 bits DO hold (the
    // rounding of f32 -> bf16 happens first and is not this conversion's business).
    for (float v : {0.0f, -0.0f, 1.0f, -1.0f, 1.5f, 1.9921875f, 2.0f, 65280.0f, -65280.0f, 6.103515625e-5f,
                    0.0078125f, 123.456f, 255.875f}) {
        const uint16_t in = strata::kernels::bf16_from_f32(v);
        const uint16_t got = strata::kernels::f16_from_bf16(in);
        const uint16_t want = strata::kernels::f16_from_f32(strata::kernels::f32_from_bf16(in));
        if (got != want) {
            std::printf("  conversion %g: in %04X got %04X want %04X\n", v, in, got, want);
            ++bad;
        }
    }
    // |x| > 65504 clamps to +-65504 (0x7BFF); the bf16 grid jumps 65280 -> 65536, so 65536 is the first
    // clamped finite value and every larger magnitude lands on the same +-65504.
    for (float v : {65536.0f, 1.0e10f, 3.0e38f}) {
        for (int s = 0; s < 2; ++s) {
            const uint16_t in = strata::kernels::bf16_from_f32(s ? -v : v);
            const uint16_t got = strata::kernels::f16_from_bf16(in);
            const uint16_t want = (uint16_t) ((s ? 0x8000u : 0u) | 0x7BFFu);
            if (got != want) {
                std::printf("  conversion clamp %g: in %04X got %04X want %04X\n", s ? -v : v, in, got, want);
                ++bad;
            }
        }
    }
    // inf/NaN pass through (a NaN must NOT be clamped to 65504).
    if (strata::kernels::f16_from_bf16(strata::kernels::bf16_from_f32(INFINITY)) != 0x7C00u) {
        std::printf("  conversion: +inf did not pass through\n");
        ++bad;
    }
    if (strata::kernels::f16_from_bf16(strata::kernels::bf16_from_f32(-INFINITY)) != 0xFC00u) {
        std::printf("  conversion: -inf did not pass through\n");
        ++bad;
    }
    if ((strata::kernels::f16_from_bf16(0x7FC1) & 0x7C00u) != 0x7C00u ||
        (strata::kernels::f16_from_bf16(0x7FC1) & 0x03FFu) == 0) {
        std::printf("  conversion: NaN did not pass through\n");
        ++bad;
    }
    // Exhaustive exactness over the finite fp16 range: every bf16 whose value fp16 represents exactly must
    // survive with the same value.  256 exponents x 128 mantissas is 32k cases - cheap and it covers the
    // real operand range (2^-14 <= |x| <= 65280).
    for (int exp = 0; exp < 256; ++exp) {
        for (int man = 0; man < 128; ++man) {
            const uint16_t in = (uint16_t) ((exp << 7) | man);
            const float f = strata::kernels::f32_from_bf16(in);
            if (!std::isfinite(f) || std::fabs(f) < 6.103515625e-5f || std::fabs(f) > 65280.0f) continue;
            const uint16_t got = strata::kernels::f16_from_bf16(in);
            const float back = strata::kernels::f32_from_f16(got);
            if (back != f) {
                std::printf("  conversion not exact: in %04X (%g) -> %04X (%g)\n", in, f, got, back);
                ++bad;
                if (bad > 8) return bad;
            }
        }
    }
    std::printf("conversion check: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad;
}

struct Shape {
    int64_t T, N, K;
    const char* what;
};

struct Timing {
    double ms = 0.0;
};

// `iters` back-to-back GEMMs around CUDA events on `stream`.
template <class F>
Timing time_on_stream(void* stream, int iters, F&& f) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    f();   // warmup: covers lazy cublas algo selection and (for the fallback) the W twin
    ck(cudaStreamSynchronize((cudaStream_t) stream), "sync");
    ck(cudaEventRecord(a, (cudaStream_t) stream), "record");
    for (int i = 0; i < iters; ++i) f();
    ck(cudaEventRecord(b, (cudaStream_t) stream), "record");
    ck(cudaEventSynchronize(b), "event sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return Timing{ms / iters};
}

struct Diff {
    double max_abs = 0.0;
    double max_rel = 0.0;
};

Diff diff_against(const std::vector<float>& got, const std::vector<float>& ref) {
    Diff d;
    double scale = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) scale = std::max(scale, (double) std::fabs(ref[i]));
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = std::fabs((double) got[i] - (double) ref[i]);
        d.max_abs = std::max(d.max_abs, a);
        d.max_rel = std::max(d.max_rel, scale > 0 ? a / scale : 0.0);
    }
    return d;
}

// The experiment arm: the FP16 GEMM with a chosen compute type / algo, straight at cublasGemmEx.
cublasStatus_t f16_gemm_raw(cublasHandle_t h, const void* W, const void* X, float* Y, int64_t T, int64_t N,
                            int64_t K, int64_t ldy, cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W, CUDA_R_16F, (int) K,
                        X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy, compute, algo);
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    if (check_conversion() != 0) return 1;

    int dev = 0;
    ck(cudaGetDevice(&dev), "get device");
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, dev), "props");
    std::printf("device: %s, sm_%d%d\n", p.name, p.major, p.minor);

    void* stream = nullptr;
    ck(cudaStreamCreateWithFlags((cudaStream_t*) &stream, cudaStreamNonBlocking), "stream");
    cublasHandle_t raw = nullptr;
    ck(cublasCreate(&raw), "cublas");
    ck(cublasSetStream(raw, (cudaStream_t) stream), "cublas stream");

    const int iters = quick ? 5 : kIters;
    const std::vector<Shape> shapes = {
        {1, 48, 2560, "ssm_alpha/beta 2560x48"},
        {1, 128, 2560, "indexer.k 2560x128"},
        {1, 512, 2560, "indexer.q 2560x512"},
        {1, 2560, 2560, "ple_value 2560x2560"},
        {256, 48, 2560, "ssm_alpha/beta 2560x48"},
        {256, 128, 2560, "indexer.k 2560x128"},
        {256, 512, 2560, "indexer.q 2560x512"},
        {256, 2560, 2560, "ple_value 2560x2560"},
        {8192, 320, 10240, "prefill hc_down 10240x320"},
        {8192, 10240, 320, "prefill hc_up 320x10240"},
        {8192, 4, 10240, "prefill hc_inject 10240x4"},
        {8192, 2560, 2560, "prefill ple_value 2560x2560"},
        {8192, 512, 2560, "prefill indexer.q 2560x512"},
        {8192, 1, 2560, "prefill shared gate 2560x1"},
    };

    std::printf("\n%-26s %3s %10s %10s %10s %10s %10s | %10s %10s %10s\n", "shape", "T", "bf16 ms", "f16 ms", "nocache",
                "cold ms", "raw ms", "bf16/f16", "old vs ref", "new vs ref");
    for (const Shape& s : shapes) {
        const int64_t T = s.T, N = s.N, K = s.K;
        std::vector<float> Xf((size_t) (T * K)), Wf((size_t) (N * K));
        // Operands in a realistic activation/weight range; the bf16 rounding IS the input (both paths see the
        // very same numbers), so quantize to bf16 first and widen back exactly for the reference.
        unsigned seed = 12345u + (unsigned) (T * 131 + N * 17 + K);
        auto rnd = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return (float) ((double) (seed >> 8) / (double) (1u << 24) * 4.0 - 2.0);
        };
        for (auto& v : Xf) v = strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(rnd()));
        for (auto& v : Wf) v = strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(rnd()));

        std::vector<uint16_t> X16((size_t) (T * K)), W16((size_t) (N * K));
        for (size_t i = 0; i < X16.size(); ++i) X16[i] = strata::kernels::bf16_from_f32(Xf[i]);
        for (size_t i = 0; i < W16.size(); ++i) W16[i] = strata::kernels::bf16_from_f32(Wf[i]);

        uint16_t *dX = nullptr, *dW = nullptr;
        float *dY_old = nullptr, *dY_new = nullptr, *dY_ref = nullptr, *dY_raw = nullptr;
        float *dX32 = nullptr, *dW32 = nullptr;
        ck(cudaMalloc(&dX, X16.size() * 2), "X");
        ck(cudaMalloc(&dW, W16.size() * 2), "W");
        ck(cudaMalloc(&dY_old, (size_t) (T * N) * 4), "Y");
        ck(cudaMalloc(&dY_new, (size_t) (T * N) * 4), "Y");
        ck(cudaMalloc(&dY_ref, (size_t) (T * N) * 4), "Y");
        ck(cudaMalloc(&dY_raw, (size_t) (T * N) * 4), "Y");
        ck(cudaMalloc(&dX32, Xf.size() * 4), "X32");
        ck(cudaMalloc(&dW32, Wf.size() * 4), "W32");
        ck(cudaMemcpyAsync(dX, X16.data(), X16.size() * 2, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cx");
        ck(cudaMemcpyAsync(dW, W16.data(), W16.size() * 2, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cw");
        ck(cudaMemcpyAsync(dX32, Xf.data(), Xf.size() * 4, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cx3");
        ck(cudaMemcpyAsync(dW32, Wf.data(), Wf.size() * 4, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cw3");

        // fp32 reference: the same layout contract as Gemm::bf16's column-major view.
        const float alpha = 1.0f, beta = 0.0f;
        ck(cublasGemmEx(raw, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, dW32, CUDA_R_32F,
                        (int) K, dX32, CUDA_R_32F, (int) K, &beta, dY_ref, CUDA_R_32F, (int) N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "ref gemm");

        setenv("STRATA_PREFILL_BF16_F16", "0", 1);
        setenv("STRATA_PREFILL_BF16_TWINS_MB", "256", 1);
        Gemm old_gemm;
        std::string err;
        if (!old_gemm.init(stream, 32 << 20, err)) {
            std::fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }
        setenv("STRATA_PREFILL_BF16_F16", "1", 1);
        setenv("STRATA_PREFILL_BF16_TWINS_MB", "0", 1);
        Gemm nc_gemm;
        if (!nc_gemm.init(stream, 32 << 20, err)) {
            std::fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }
        setenv("STRATA_PREFILL_BF16_TWINS_MB", "256", 1);
        Gemm new_gemm;
        if (!new_gemm.init(stream, 32 << 20, err)) {
            std::fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }

        // Cold: the FIRST call through the fallback - W's twin conversion included.
        ck(cudaStreamSynchronize((cudaStream_t) stream), "sync");
        cudaEvent_t ca, cb;
        ck(cudaEventCreate(&ca), "event");
        ck(cudaEventCreate(&cb), "event");
        ck(cudaEventRecord(ca, (cudaStream_t) stream), "record");
        new_gemm.bf16(dX, dW, dY_new, T, N, K, 0, 0.0f);
        ck(cudaEventRecord(cb, (cudaStream_t) stream), "record");
        ck(cudaEventSynchronize(cb), "event sync");
        float cold_ms = 0.0f;
        ck(cudaEventElapsedTime(&cold_ms, ca, cb), "elapsed");
        cudaEventDestroy(ca);
        cudaEventDestroy(cb);

        const Timing t_old = time_on_stream(stream, iters, [&] { old_gemm.bf16(dX, dW, dY_old, T, N, K, 0, 0.0f); });
        const Timing t_new = time_on_stream(stream, iters, [&] { new_gemm.bf16(dX, dW, dY_new, T, N, K, 0, 0.0f); });
        const Timing t_nc = time_on_stream(stream, iters, [&] { nc_gemm.bf16(dX, dW, dY_raw, T, N, K, 0, 0.0f); });

        // f16 raw: pre-converted operands, the FP16 GEMM alone.
        std::vector<uint16_t> Xf16(X16.size()), Wf16(W16.size());
        for (size_t i = 0; i < Xf16.size(); ++i) Xf16[i] = strata::kernels::f16_from_bf16(X16[i]);
        for (size_t i = 0; i < Wf16.size(); ++i) Wf16[i] = strata::kernels::f16_from_bf16(W16[i]);
        uint16_t *dXf = nullptr, *dWf = nullptr;
        ck(cudaMalloc(&dXf, Xf16.size() * 2), "Xf");
        ck(cudaMalloc(&dWf, Wf16.size() * 2), "Wf");
        ck(cudaMemcpyAsync(dXf, Xf16.data(), Xf16.size() * 2, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cxf");
        ck(cudaMemcpyAsync(dWf, Wf16.data(), Wf16.size() * 2, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cwf");
        const Timing t_raw =
            time_on_stream(stream, iters, [&] { f16_gemm_raw(raw, dWf, dXf, dY_raw, T, N, K, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); });

        std::vector<float> Y_old((size_t) (T * N)), Y_new((size_t) (T * N)), Y_ref((size_t) (T * N));
        ck(cudaMemcpy(Y_old.data(), dY_old, Y_old.size() * 4, cudaMemcpyDeviceToHost), "dy");
        ck(cudaMemcpy(Y_new.data(), dY_new, Y_new.size() * 4, cudaMemcpyDeviceToHost), "dy");
        ck(cudaMemcpy(Y_ref.data(), dY_ref, Y_ref.size() * 4, cudaMemcpyDeviceToHost), "dy");
        const Diff d_old = diff_against(Y_old, Y_ref);
        const Diff d_new = diff_against(Y_new, Y_ref);
        const Diff d_between = diff_against(Y_new, Y_old);

        std::printf("%-26s %3lld %10.4f %10.4f %10.4f %10.4f %10.4f | %9.2fx %8.2e%% %8.2e%%\n", s.what, (long long) T,
                    t_old.ms, t_new.ms, t_nc.ms, (double) cold_ms, t_raw.ms, t_old.ms / t_new.ms, d_old.max_rel * 100.0,
                    d_new.max_rel * 100.0);
        std::printf("%-26s %3s max|new-old| = %.3e (rel %.2e%%), max|new-ref| = %.3e\n", "", "",
                    d_between.max_abs, d_between.max_rel * 100.0, d_new.max_abs);

        cudaFree(dX);
        cudaFree(dW);
        cudaFree(dY_old);
        cudaFree(dY_new);
        cudaFree(dY_ref);
        cudaFree(dY_raw);
        cudaFree(dX32);
        cudaFree(dW32);
        cudaFree(dXf);
        cudaFree(dWf);
    }

    // ---- overflow smoke: a W with |w| > 65504 must clamp to +-65504, never poison Y with NaN/Inf ----
    {
        const int64_t T = 4, N = 8, K = 8;
        std::vector<uint16_t> X16((size_t) (T * K), 0x3F80), W16((size_t) (N * K), 0x3F80);   // bf16 1.0
        W16[0] = 0x7F00;              // +1.7e38 bf16: past fp16's 65504 -> clamped to +65504
        W16[1] = 0xFF00;              // its negative -> clamped to -65504, cancels W16[0]
        W16[1 * K + 2] = 0x7F80;      // bf16 +inf in W row 1: passes through as fp16 inf, like the BF16 path
        uint16_t *dX = nullptr, *dW = nullptr;
        float* dY = nullptr;
        ck(cudaMalloc(&dX, X16.size() * 2), "X");
        ck(cudaMalloc(&dW, W16.size() * 2), "W");
        ck(cudaMalloc(&dY, (size_t) (T * N) * 4), "Y");
        ck(cudaMemcpy(dX, X16.data(), X16.size() * 2, cudaMemcpyHostToDevice), "cx");
        ck(cudaMemcpy(dW, W16.data(), W16.size() * 2, cudaMemcpyHostToDevice), "cw");
        setenv("STRATA_PREFILL_BF16_F16", "1", 1);
        setenv("STRATA_PREFILL_BF16_TWINS_MB", "0", 1);   // exercise the per-call W conversion too
        Gemm g;
        std::string err;
        if (!g.init(stream, 32 << 20, err)) {
            std::fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }
        g.bf16(dX, dW, dY, T, N, K, 0, 0.0f);
        std::vector<float> Y((size_t) (T * N));
        ck(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost), "dy");
        bool finite_row0 = true;
        for (int64_t t = 0; t < T; ++t) finite_row0 = finite_row0 && std::isfinite(Y[(size_t) (t * N)]);
        // row 0 of W sums clamped(+1.7e38) + clamped(-1.7e38) + 6 x 1.0 = 6.0 exactly; row 1 carries the inf
        // and is +inf here just as the BF16 path would leave it - a pass-through, not a clamp.
        std::printf("overflow smoke: row0 finite %s, Y[0] = %g (want 6), Y[1] = %g (want inf)\n",
                    finite_row0 ? "yes" : "NO", Y[0], Y[1]);
        if (!finite_row0 || std::fabs(Y[0] - 6.0) > 1e-3 || std::isfinite(Y[1])) {
            std::printf("overflow smoke: FAIL\n");
            return 1;
        }
        std::printf("overflow smoke: PASS\n");
        // beta = 1 accumulation (the bf16x2 remainder's shape): a second call must add, exactly as before.
        g.bf16(dX, dW, dY, T, N, K, 0, 1.0f);
        ck(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost), "dy");
        std::printf("beta=1 smoke: Y[0] = %g (want 12)\n", Y[0]);
        if (std::fabs(Y[0] - 12.0) > 1e-3) {
            std::printf("beta=1 smoke: FAIL\n");
            return 1;
        }
        std::printf("beta=1 smoke: PASS\n");
        cudaFree(dX);
        cudaFree(dW);
        cudaFree(dY);
    }

    // ---- the experiment: FP16 GEMM math modes (report only; nothing here changes the shipped default) ----    std::printf("\nFP16 GEMM math-mode experiment (T=256, N=K=2560, ms per call):\n");
    {
        const int64_t T = 256, N = 2560, K = 2560;
        std::vector<uint16_t> Xf16((size_t) (T * K)), Wf16((size_t) (N * K));
        unsigned seed = 999u;
        auto rnd = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return (float) ((double) (seed >> 8) / (double) (1u << 24) * 4.0 - 2.0);
        };
        for (auto& v : Xf16) v = strata::kernels::f16_from_f32(rnd());
        for (auto& v : Wf16) v = strata::kernels::f16_from_f32(rnd());
        uint16_t *dX = nullptr, *dW = nullptr;
        float* dY = nullptr;
        ck(cudaMalloc(&dX, Xf16.size() * 2), "X");
        ck(cudaMalloc(&dW, Wf16.size() * 2), "W");
        ck(cudaMalloc(&dY, (size_t) (T * N) * 4), "Y");
        ck(cudaMemcpyAsync(dX, Xf16.data(), Xf16.size() * 2, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cx");
        ck(cudaMemcpyAsync(dW, Wf16.data(), Wf16.size() * 2, cudaMemcpyHostToDevice, (cudaStream_t) stream), "cw");
        struct Arm {
            const char* what;
            cublasComputeType_t compute;
            cublasGemmAlgo_t algo;
        };
        const Arm arms[] = {
            {"COMPUTE_32F + DEFAULT (shipped)", CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT},
            {"COMPUTE_32F + DEFAULT_TENSOR_OP", CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP},
            {"COMPUTE_16F + DEFAULT_TENSOR_OP", CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP},
        };
        for (const Arm& a : arms) {
            const cublasStatus_t s = f16_gemm_raw(raw, dW, dX, dY, T, N, K, N, a.compute, a.algo);
            if (s != CUBLAS_STATUS_SUCCESS) {
                std::printf("  %-34s unsupported (cuBLAS status %d)\n", a.what, (int) s);
                continue;
            }
            const Timing t = time_on_stream(stream, iters, [&] {
                f16_gemm_raw(raw, dW, dX, dY, T, N, K, N, a.compute, a.algo);
            });
            std::printf("  %-34s %8.4f ms\n", a.what, t.ms);
        }
        cudaFree(dX);
        cudaFree(dW);
        cudaFree(dY);
    }

    setenv("STRATA_PREFILL_BF16_F16", "0", 1);   // leave the process on the shipped default
    cublasDestroy(raw);
    cudaStreamDestroy((cudaStream_t) stream);
    return 0;
}
