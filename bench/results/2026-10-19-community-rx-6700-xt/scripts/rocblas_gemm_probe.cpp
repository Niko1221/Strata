// rocBLAS GEMM probe for the Strata prompt path (HS/BS/HH/SS) on AMD gfx103x cards.
// Build: hipcc --offload-arch=gfx1030 --offload-arch=gfx1031 -O2 rocblas_gemm_probe.cpp -I/opt/rocm/include -L/opt/rocm/lib -lrocblas -o probe
// Run with and without HSA_OVERRIDE_GFX_VERSION=10.3.0.
// Gotcha: rocblas_gemm_ex alpha/beta are the COMPUTE type (f32), not the input type - a __half alpha gives ~zero output.
// Checked against a CPU reference at 512x384x320, timed at an engine prompt shape N=10240 x T=1024 x K=2560.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

static double ref_gemm(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int m, int n, int k) {
    std::vector<float> ref((size_t) m * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) acc += (double) A[i + (size_t) t * m] * B[t + (size_t) j * k];
            ref[i + (size_t) j * m] = (float) acc;
        }
    double scale = 0.0, worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) scale = std::max(scale, (double) std::fabs(ref[i]));
    for (size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, (double) std::fabs(C[i] - ref[i]) / (scale > 0 ? scale : 1.0));
    return worst;
}

int main() {
    hipDeviceProp_t p{};
    hipGetDeviceProperties(&p, 0);
    printf("device 0: %s arch=%s reported CUs=%d\n", p.name, p.gcnArchName, p.multiProcessorCount);
    rocblas_handle rb;
    rocblas_create_handle(&rb);
    char ver[64];
    rocblas_get_version_string(ver, sizeof(ver));
    printf("rocBLAS %s  ROCBLAS_TENSILE_LIBPATH=%s\n\n", ver,
           std::getenv("ROCBLAS_TENSILE_LIBPATH") ? std::getenv("ROCBLAS_TENSILE_LIBPATH") : "(unset)");

    const int m = 512, n = 384, k = 320;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> Af((size_t) m * k), Bf((size_t) k * n);
    for (auto& x : Af) x = u(rng);
    for (auto& x : Bf) x = u(rng);

    auto run = [&](const char* tag, rocblas_datatype at, rocblas_datatype ct, size_t eszA, size_t eszC, int tm, int tn, int tk, bool time_it) {
        const size_t nA = (size_t) tm * tk, nB = (size_t) tk * tn, nC = (size_t) tm * tn;
        std::vector<char> a(nA * eszA), b(nB * eszA), c(nC * eszC, 0);
        std::vector<float> Cget(nC);
        for (size_t i = 0; i < nA; ++i) {
            float v = Af[i % Af.size()];
            if (at == rocblas_datatype_f32_r) std::memcpy(a.data() + i * 4, &v, 4);
            else if (at == rocblas_datatype_f16_r) { __half h = __float2half(v); std::memcpy(a.data() + i * 2, &h, 2); }
            else { hip_bfloat16 h(v); std::memcpy(a.data() + i * 2, &h, 2); }
        }
        for (size_t i = 0; i < nB; ++i) {
            float v = Bf[i % Bf.size()];
            if (at == rocblas_datatype_f32_r) std::memcpy(b.data() + i * 4, &v, 4);
            else if (at == rocblas_datatype_f16_r) { __half h = __float2half(v); std::memcpy(b.data() + i * 2, &h, 2); }
            else { hip_bfloat16 h(v); std::memcpy(b.data() + i * 2, &h, 2); }
        }
        void *da, *db, *dc;
        if (hipMalloc(&da, a.size()) != hipSuccess || hipMalloc(&db, b.size()) != hipSuccess || hipMalloc(&dc, c.size()) != hipSuccess) {
            printf("[%-10s] hipMalloc failed\n", tag);
            return;
        }
        hipMemcpy(da, a.data(), a.size(), hipMemcpyHostToDevice);
        hipMemcpy(db, b.data(), b.size(), hipMemcpyHostToDevice);
        float alpha = 1.0f, beta = 0.0f;   // compute type f32
        rocblas_status st = rocblas_gemm_ex(rb, rocblas_operation_none, rocblas_operation_none, tm, tn, tk,
                                            &alpha, da, at, tm, db, at, tk, &beta, dc, ct, tm, dc, ct, tm,
                                            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        hipError_t se = hipDeviceSynchronize();
        hipMemcpy(c.data(), dc, c.size(), hipMemcpyDeviceToHost);
        for (size_t i = 0; i < nC; ++i)
            if (ct == rocblas_datatype_f32_r) std::memcpy(&Cget[i], c.data() + i * 4, 4);
            else if (ct == rocblas_datatype_f16_r) { __half h; std::memcpy(&h, c.data() + i * 2, 2); Cget[i] = __half2float(h); }
            else { hip_bfloat16 h; std::memcpy(&h, c.data() + i * 2, 2); Cget[i] = float(h); }
        double rel = (tm == m && tn == n && tk == k) ? ref_gemm(Af, Bf, Cget, m, n, k) : 1e9;
        printf("[%-10s] status=%d sync=%s rel_err=%.2e %s\n", tag, (int) st, hipGetErrorString(se), rel,
               (st == rocblas_status_success && rel < 2e-3) ? "OK" : "FAIL");
        if (time_it) {
            hipFree(da); hipFree(db); hipFree(dc);
            const int TM = 10240, TN = 1024, TK = 2560;
            size_t wa = (size_t) TM * TK * eszA, wb = (size_t) TK * TN * eszA, wc = (size_t) TM * TN * eszC;
            if (hipMalloc(&da, wa) != hipSuccess || hipMalloc(&db, wb) != hipSuccess || hipMalloc(&dc, wc) != hipSuccess) { printf("[%-10s] big-shape malloc failed\n", tag); return; }
            hipMemset(da, 0x11, wa); hipMemset(db, 0x11, wb);
            double ms = 0;
            for (int i = 0; i < 5; ++i) {
                hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
                st = rocblas_gemm_ex(rb, rocblas_operation_none, rocblas_operation_none, TM, TN, TK,
                                     &alpha, da, at, TM, db, at, TK, &beta, dc, ct, TM, dc, ct, TM,
                                     rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
                hipEventRecord(e1); hipEventSynchronize(e1);
                float t = 0; hipEventElapsedTime(&t, e0, e1); ms += t;
                hipEventDestroy(e0); hipEventDestroy(e1);
            }
            ms /= 5;
            printf("              N%d x T%d x K%d status=%d %.1f ms %.1f TFLOPS\n", TM, TN, TK, (int) st, ms,
                   2.0 * TM * TN * TK / (ms * 1e-3) / 1e12);
        }
        hipFree(da); hipFree(db); hipFree(dc);
    };

    run("HS f16>f32", rocblas_datatype_f16_r, rocblas_datatype_f32_r, 2, 4, m, n, k, true);
    run("BS bf16>f32", rocblas_datatype_bf16_r, rocblas_datatype_f32_r, 2, 4, m, n, k, true);
    run("HH f16>f16", rocblas_datatype_f16_r, rocblas_datatype_f16_r, 2, 2, m, n, k, true);
    run("SS f32>f32", rocblas_datatype_f32_r, rocblas_datatype_f32_r, 4, 4, m, n, k, true);
    rocblas_destroy_handle(rb);
    return 0;
}
