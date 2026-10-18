// src/prefill/dense_mmq_bench.cu - gfx906: the prompt path's dense projections three ways, at the engine's shapes.
//
//     STRATA_MMQ_KQ=1 dense_mmq_bench [T]
//
// (a) today's path: Gemm::native - the GGUF weight dequantized to FP16, then hipBLAS FP16 GEMM, FP16 activations;
// (b) llama.cpp's MMQ (int8 dot products): the activations rounded to q8_1 (timed with it, and without it: one
//     quantized input can feed several projections), the weight read as its GGUF blocks;
// (c) mx-llama.cpp's repacked GEMM (rp_moe), where it covers the type (IQ4_NL), the weight repacked once.
// Weights: random N(0, 0.02) values quantized by ggml_quantize_chunk.  Activations: N(0, 1).  Accuracy: each path's
// relative RMS error against an FP64 product of the dequantized weight and the FP32 activations (CPU, on a sample
// of rows).  Medians of 15 runs.  Default T: 4096 (one --prefill 4096 chunk).
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/rp_moe.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace rp = strata::prefill::rp;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_mmq_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void to_f16(const float* x, __half* y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __float2half(x[i]);
}

template<class F>
float median_ms(int n, F&& fn) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> ts;
    for (int i = 0; i < n + 3; ++i) {
        cudaEventRecord(e0);
        fn();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "run");
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (i >= 3) ts.push_back(ms);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

struct Case {
    const char* name;
    ggml_type t;
    int64_t K, N;   // in, out
};

// relative RMS error of Y (T x N) against the FP64 reference on rows `rows`
double rel_err(const std::vector<float>& Y, const std::vector<double>& ref, const std::vector<int>& rows, int64_t N) {
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int64_t n = 0; n < N; ++n) {
            const double r = ref[i * N + n], y = Y[(size_t) rows[i] * N + n];
            e2 += (y - r) * (y - r);
            r2 += r * r;
        }
    return std::sqrt(e2 / std::max(r2, 1e-300));
}

void run_case(const Case& c, int64_t T, std::mt19937& rng) {
    const int64_t K = c.K, N = c.N;
    std::normal_distribution<float> wd(0.f, 0.02f), xd(0.f, 1.f);
    std::vector<float> wf((size_t) N * K), x((size_t) T * K);
    for (auto& v : wf) v = wd(rng);
    for (auto& v : x) v = xd(rng);
    const size_t wbytes = ggml_row_size(c.t, K) * (size_t) N;
    std::vector<uint8_t> wq(wbytes + 4096, 0);   // the MMQ tail zeroed
    ggml_quantize_chunk(c.t, wf.data(), wq.data(), 0, N, K, nullptr);
    // the reference: the weight as ggml dequantizes it, times the FP32 activations, in FP64 (a sample of rows)
    std::vector<float> wdq((size_t) N * K);
    ggml_get_type_traits(c.t)->to_float(wq.data(), wdq.data(), N * K);
    std::vector<int> rows;
    for (int i = 0; i < 24; ++i) rows.push_back((int) ((int64_t) i * (T - 1) / 23));
    std::vector<double> ref(rows.size() * (size_t) N);
    for (size_t i = 0; i < rows.size(); ++i)
        for (int64_t n = 0; n < N; ++n) {
            double s = 0;
            for (int64_t k = 0; k < K; ++k) s += (double) x[(size_t) rows[i] * K + k] * wdq[(size_t) n * K + k];
            ref[i * N + n] = s;
        }

    // for scale: the error the weight's own quantization already carries (the dequantized product vs the FP32 weight's)
    double wq2 = 0, wr2 = 0;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int64_t n = 0; n < N; ++n) {
            double sf = 0;
            for (int64_t k = 0; k < K; ++k) sf += (double) x[(size_t) rows[i] * K + k] * wf[(size_t) n * K + k];
            wq2 += (ref[i * N + n] - sf) * (ref[i * N + n] - sf);
            wr2 += sf * sf;
        }
    const double e_wq = std::sqrt(wq2 / std::max(wr2, 1e-300));
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    uint8_t *dw, *drw = nullptr;
    float *dx, *dy;
    __half* dx16;
    void* dq;
    int32_t *dids, *db;
    ck(cudaMalloc(&dw, wq.size()), "w");
    ck(cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 4), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dx16, x.size() * 2), "x16");
    to_f16<<<1024, 256, 0, s>>>(dx, dx16, (int64_t) x.size());
    ck(cudaMalloc(&dy, (size_t) T * N * 4), "y");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, K)), "xq");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids");
    ck(cudaMalloc(&db, 8), "bounds");
    const int32_t bounds[2] = {0, (int32_t) T};
    ck(cudaMemcpy(db, bounds, 8, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, s);
    std::vector<float> y((size_t) T * N);
    auto fetch = [&] {
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    };

    // (a) today's path
    strata::prefill::Gemm gm;
    std::string err;
    if (!gm.init(s, N * K, err)) { std::fprintf(stderr, "gemm init: %s\n", err.c_str()); std::exit(1); }
    gm.native((const uint16_t*) dx16, c.t, dw, dy, T, N, K);
    fetch();
    const double e_f16 = rel_err(y, ref, rows, N);
    const float t_f16 = median_ms(15, [&] { gm.native((const uint16_t*) dx16, c.t, dw, dy, T, N, K); });

    // (b) MMQ: one "expert" of T rows
    const bool mmq_ok = mmq::supported(c.t) && mmq::fits(c.t, N);
    float t_q = 0, t_mmq = 0, t_rp = 0;
    std::vector<float> y_mmq;
    size_t rp_diff = 0;
    double e_mmq = -1, e_rp = -1;
    mmq::Product p;
    p.w = dw; p.type = c.t; p.w_rows = N; p.w_cols = K; p.expert_bytes = wbytes; p.n = 1;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = T; p.dst = dy; p.ld_dst = N;
    mmq::Context mctx;
    if (mmq_ok) {
        mmq::quantize(dx, nullptr, dq, c.t, K, K, T, s);
        mctx.run(p, s);
        fetch();
        e_mmq = rel_err(y, ref, rows, N);
        y_mmq = y;
        t_q = median_ms(15, [&] { mmq::quantize(dx, nullptr, dq, c.t, K, K, T, s); });
        t_mmq = median_ms(15, [&] { mctx.run(p, s); });
    }
    // (c) the repacked GEMM
    rp::Context rctx;
    if (mmq_ok && rp::built() && rp::dense_supported(c.t, N, K)) {
        const size_t rb = rp::matrix_bytes(c.t, N, K);
        ck(cudaMalloc(&drw, rb), "rw");
        rp::repack(c.t, dw, wbytes, drw, rb, N, K, 1, s);
        mmq::Product q = p;
        q.w = drw; q.expert_bytes = rb;
        rctx.run(q, s);
        fetch();
        e_rp = rel_err(y, ref, rows, N);
        for (size_t i = 0; i < y.size(); ++i) rp_diff += std::memcmp(&y[i], &y_mmq[i], 4) != 0;
        t_rp = median_ms(15, [&] { rctx.run(q, s); });
    }
    const double gf = 2.0 * T * N * K / 1e9;
    std::printf("  %-26s %-7s [%5lld -> %5lld]  f16 %7.3f ms (%5.1f TF, err %.1e)", c.name, ggml_type_name(c.t),
                (long long) K, (long long) N, t_f16, gf / t_f16, e_f16);
    if (mmq_ok)
        std::printf(" | mmq %7.3f + q %5.3f ms (%5.1f TOPS, err %.1e) %+4.0f%%", t_mmq, t_q, gf / t_mmq, e_mmq,
                    100.0 * (t_f16 / (t_mmq + t_q) - 1.0));
    else
        std::printf(" | mmq: not covered");
    if (e_rp >= 0)
        std::printf(" | rp %7.3f + q ms (%5.1f TOPS, err %.1e) %+4.0f%%, vs mmq: %s", t_rp, gf / t_rp, e_rp,
                    100.0 * (t_f16 / (t_rp + t_q) - 1.0), rp_diff ? "DIFFERS" : "bitwise");
    std::printf(" | weight quant err %.1e\n", e_wq);
    for (void* v : {(void*) dw, (void*) dx, (void*) dx16, (void*) dy, dq, (void*) dids, (void*) db, (void*) drw})
        if (v) cudaFree(v);
    cudaStreamDestroy(s);
}

}  // namespace

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 4096;
    std::printf("dense_mmq_bench: T %lld; f16 = dequant + hipBLAS FP16 (today), mmq = llama.cpp int8 MMQ (+ q = its "
                "activation quantize), rp = mx repacked; %% = speedup over f16 incl. the quantize\n", (long long) T);
    const Case cases[] = {
        // Swift GSQ-RCO IQ3_S (our 2-GPU default)
        {"GDN attn_qkv (IQ3_S file)", GGML_TYPE_Q6_K, 2560, 10240},
        {"GDN attn_gate (IQ3_S file)", GGML_TYPE_IQ4_XS, 2560, 6144},
        {"GDN ssm_out (IQ3_S file)", GGML_TYPE_Q6_K, 6144, 2560},
        {"QSA attn_q (IQ3_S file)", GGML_TYPE_IQ4_XS, 2560, 12288},
        {"QSA attn_output (IQ3_S file)", GGML_TYPE_Q4_K, 6144, 2560},
        // Swift IQ4_NL
        {"GDN attn_qkv (IQ4_NL file)", GGML_TYPE_IQ4_NL, 2560, 10240},
        {"GDN attn_gate (IQ4_NL file)", GGML_TYPE_IQ4_NL, 2560, 6144},
        {"GDN ssm_out (IQ4_NL file)", GGML_TYPE_Q8_0, 6144, 2560},
        {"QSA attn_q (IQ4_NL file)", GGML_TYPE_Q6_K, 2560, 12288},
        {"QSA attn_output (IQ4_NL file)", GGML_TYPE_Q5_K, 6144, 2560},
    };
    std::mt19937 rng(11);
    for (const Case& c : cases) run_case(c, T, rng);
    return 0;
}
