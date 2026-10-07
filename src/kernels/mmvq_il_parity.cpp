// src/kernels/mmvq_il_parity.cpp - native_mmvq_il (2-4 columns from the interleaved q8_1 copy, fork F4) against
// native_mmvq's multi-column kernels, bitwise, for every rows-a-warp choice; --bench times both per call.
//
//     build/mmvq_il_parity [--bench]
//
// The same Q8_1 bytes feed both paths (quantize_q8_1_rows, then native_q8_1_interleave). Weights are random bytes with
// the block scales rewritten as normal fp16 values, so every output is finite (a NaN would compare equal whatever
// produced it). The bench cycles through enough copies of the weights to miss the L2 cache, as a layer's call does.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace strata::kernels;

struct Case {
    const char* name;
    int type, n_in, n_out, block_elems, block_bytes, scale_at[2];
};
// the engine's dense shapes: qkv 10240 / gate 6144 / out 2560 (GDN), q 12288 / k,v 512 / o 2560 (QSA), the head
const Case CASES[] = {
    {"IQ4_XS 2560x10240", 23, 2560, 10240, 256, 136, {0, -1}},
    {"IQ4_XS 2560x6144", 23, 2560, 6144, 256, 136, {0, -1}},
    {"IQ4_XS 4096x2560", 23, 4096, 2560, 256, 136, {0, -1}},
    {"IQ4_XS 2560x512", 23, 2560, 512, 256, 136, {0, -1}},
    {"Q4_K 2560x10240", 12, 2560, 10240, 256, 144, {0, 2}},
    {"Q4_K 4096x2560", 12, 4096, 2560, 256, 144, {0, 2}},
    {"Q4_K odd rows 2560x2051", 12, 2560, 2051, 256, 144, {0, 2}},
    {"Q5_K 2560x12288", 13, 2560, 12288, 256, 176, {0, 2}},
    {"Q5_K 2560x6144", 13, 2560, 6144, 256, 176, {0, 2}},
    {"Q5_K 4096x2560", 13, 4096, 2560, 256, 176, {0, 2}},
    {"Q5_K head 2560x248320", 13, 2560, 248320, 256, 176, {0, 2}},
    {"Q6_K 2560x12288", 14, 2560, 12288, 256, 210, {208, -1}},
    {"Q6_K 2560x10240", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 4096x2560", 14, 4096, 2560, 256, 210, {208, -1}},
    {"Q6_K odd rows 2560x2051", 14, 2560, 2051, 256, 210, {208, -1}},
    {"Q6_K 2560x512", 14, 2560, 512, 256, 210, {208, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

int run_case(const Case& c, bool bench, cudaStream_t s) {
    if (c.type == 13 && c.n_out > 100000 && !bench) { /* the head: parity too, just big */ }
    const std::size_t wbytes = native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::mt19937 rng(77u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%s: weight bytes mismatch\n", c.name);
        return 1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    const int copies = bench ? (int) std::max<std::size_t>(1, std::min<std::size_t>(24, (192u << 20) / wbytes)) : 1;
    std::vector<void*> dw(copies, nullptr);
    for (int i = 0; i < copies; ++i) {
        if (!ck(cudaMalloc(&dw[i], wbytes), "malloc w") || !ck(cudaMemcpy(dw[i], w.data(), wbytes, cudaMemcpyHostToDevice), "copy w"))
            return 1;
    }
    int bad = 0;
    for (int T = 2; T <= 4; ++T) {
        std::vector<float> x((std::size_t) T * c.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void *xq = nullptr, *xil = nullptr;
        float *yref = nullptr, *yil = nullptr;
        const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
            !ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T)), "malloc xq") ||
            !ck(cudaMalloc(&xil, native_q8_1_il_bytes(c.n_in, T)), "malloc xil") ||
            !ck(cudaMalloc((void**) &yref, ybytes), "malloc yref") || !ck(cudaMalloc((void**) &yil, ybytes), "malloc yil"))
            return 1;
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        native_q8_1_interleave(xq, xil, c.n_in, T, s);
        native_mmvq(c.type, dw[0], xq, yref, c.n_in, c.n_out, T, s);
        ck(cudaStreamSynchronize(s), "ref");
        std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
        ck(cudaMemcpy(a.data(), yref, ybytes, cudaMemcpyDeviceToHost), "read ref");
        double best_ref = 0, best_il[5] = {0, 0, 0, 0, 0};
        int table_r = 0;
        for (int r : {0, 1, 2, 4}) {
            native_mmvq_il_tune(r);
            cudaMemset(yil, 0xff, ybytes);
            native_mmvq_il(c.type, dw[0], xq, xil, yil, c.n_in, c.n_out, T, s);
            ck(cudaStreamSynchronize(s), "il");
            ck(cudaMemcpy(b.data(), yil, ybytes, cudaMemcpyDeviceToHost), "read il");
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                float fa;
                std::memcpy(&fa, &a[i], 4);
                if (!std::isfinite(fa)) ++nonfinite;
                if (a[i] != b[i]) ++diff;
            }
            const bool used = r != 0 || native_mmvq_il_supported(c.type, T, c.n_out);
            if (diff != 0 || nonfinite != 0) {
                ++bad;
                std::printf("FAIL %-26s T=%d rows=%d: %lld bits differ, %lld non-finite of %zu\n", c.name, T, r, diff, nonfinite, a.size());
            }
            (void) used;
            if (bench && r != 0) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                const int reps = 200;
                for (int rep = 0; rep < 2; ++rep) {   // rep 0 warms, rep 1 times
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < reps; ++i) native_mmvq_il(c.type, dw[i % copies], xq, xil, yil, c.n_in, c.n_out, T, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                }
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                best_il[r] = 1000.0 * ms / reps;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            }
        }
        native_mmvq_il_tune(0);
        if (bench) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int reps = 200;
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, s);
                for (int i = 0; i < reps; ++i) native_mmvq(c.type, dw[i % copies], xq, yref, c.n_in, c.n_out, T, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
            }
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            best_ref = 1000.0 * ms / reps;
            (void) table_r;
            std::printf("BENCH %-26s T=%d  multi %.1f us | il rows1 %.1f rows2 %.1f rows4 %.1f us\n", c.name, T, best_ref,
                        best_il[1], best_il[2], best_il[4]);
        }
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yref); cudaFree(yil);
    }
    for (void* p : dw) cudaFree(p);
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    cudaStreamCreate(&s);
    int bad = 0;
    for (const Case& c : CASES) bad += run_case(c, bench, s);
    std::printf("%s\n", bad ? "mmvq_il_parity: FAIL" : "mmvq_il_parity: OK (every case, T 2-4, rows 1/2/4 and the table, bitwise)");
    return bad ? 1 : 0;
}
