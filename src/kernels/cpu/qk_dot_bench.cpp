// Strata. ggml-cpu's quantized row dots, single thread, through ggml's public CPU traits: q4_K / q5_K /
// q6_K / q8_0 against their q8 partner. A benchmark, no test.
//
// WHY THIS EXISTS: on MSVC the native ggml detection (FindSIMD) probes /arch:AVX|AVX2|AVX512 only, so a native
// MSVC build compiles ggml-cpu with __AVX512F__/BW/VL alone; the -DSTRATA_X86_ISA=zen5 target additionally
// defines __AVX512VNNI__ (ggml's MSVC branch, ggml-cpu/CMakeLists 254-305), which swaps ggml's
// mul_sum_us8_pairs_float from the maddubs + madd pair to the fused _mm256_dpbusd_epi32. That fusion is the
// engine's Q8_0 path (the q8_0 KV/activation partner dots, the q8_0 expert rows): measured +34% on a Ryzen
// 9950X3D (1378 -> 1854 GB/s of row bytes, L1-resident, single thread), where the K-quant rows are L2-speed
// and bandwidth-bound (q4_K/q5_K/q6_K flat within noise). The IQ rows are NOT in here: their dots have no
// 512-bit variant in the pinned ggml (repack.cpp is AVX2), those rows are Strata's own kernels (iq_avx2_parity,
// iq512_gather_parity).
//
// The working set is cache-resident on purpose: the fused instruction is an ALU win and a DRAM-streaming run
// measures the memory, not the dot (a 1 GB ds4_gg_bench stream shows both builds at the same ~44 GB/s).
//
//   qk_dot_bench [--iters N]
#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double bench(ggml_type qt, int K, int rows, int iters) {
    const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(qt);
    const ggml_type at = tr->vec_dot_type;                    // q8_K for the K quants, q8_0 for q8_0
    const size_t qrow = (size_t) ggml_row_size(qt, K);
    const size_t arow = (size_t) ggml_row_size(at, K);
    std::vector<float> f(K);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.4f);
    std::vector<char> Q(qrow * rows), A(arow * rows);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < K; ++i) f[i] = nd(rng);
        tr->from_float(f.data(), Q.data() + qrow * r, K);
        for (int i = 0; i < K; ++i) f[i] = nd(rng);
        ggml_get_type_traits_cpu(at)->from_float(f.data(), A.data() + arow * r, K);
    }
    const int n = K / (qt == GGML_TYPE_Q8_0 ? 32 : 256);      // blocks the dot iterates over
    float sum = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        const int r = i % rows;
        tr->vec_dot(n, &sum, 0, Q.data() + qrow * r, qrow, A.data() + arow * r, arow, 1);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gbs = (double) iters * (qrow + arow) / s / 1e9;
    std::printf("  %-6s x %-5s : %7.2f GB/s of row bytes (last sum %.3f)\n", ggml_type_name(qt), ggml_type_name(at), gbs, sum);
    return gbs;
}

int main(int argc, char** argv) {
    int iters = 400000;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
    }
    ggml_cpu_init();
    const int K = 4096, rows = 1024;
    std::printf("qk_dot_bench: K=%d rows=%d iters=%d, one thread; ggml built with: avx2 %d avx512 %d vbmi %d vnni %d bf16 %d avx_vnni %d bmi2 %d\n",
                K, rows, iters, ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vbmi(),
                ggml_cpu_has_avx512_vnni(), ggml_cpu_has_avx512_bf16(), ggml_cpu_has_avx_vnni(), ggml_cpu_has_bmi2());
    bench(GGML_TYPE_Q4_K, K, rows, iters);
    bench(GGML_TYPE_Q5_K, K, rows, iters);
    bench(GGML_TYPE_Q6_K, K, rows, iters);
    bench(GGML_TYPE_Q8_0, K, rows, iters);
    return 0;
}
