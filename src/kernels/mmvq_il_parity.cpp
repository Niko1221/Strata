// src/kernels/mmvq_il_parity.cpp - native_mmvq_il (2-4 columns from the interleaved q8_1 copy, fork F4) against
// native_mmvq's multi-column kernels, bitwise, for every rows-a-warp choice; --bench times both per call.
//
//     build/mmvq_il_parity [--bench | --sm75-bench]
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
int table_interleaved = 0, table_fallback = 0, forced_interleaved = 0;

struct Case {
    const char* name;
    int type, n_in, n_out, block_elems, block_bytes, scale_at[2];
};
// Original sm_86/sm_120 cases, plus reduction/row tails; current IQ3_S dense shapes follow below.
const Case CASES[] = {
    {"IQ4_XS tiny/tail 256x3", 23, 256, 3, 256, 136, {0, -1}},
    {"Q4_K tiny/tail 768x5", 12, 768, 5, 256, 144, {0, 2}},
    {"Q5_K tiny/tail 2304x17", 13, 2304, 17, 256, 176, {0, 2}},
    {"Q6_K tiny/tail 256x1", 14, 256, 1, 256, 210, {208, -1}},
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
// Unique supported dense projection shapes in the local IQ3_S GGUF headers, not its IQ3_S experts.
const Case SM75_CASES[] = {
    {"IQ4_XS 2560x512", 23, 2560, 512, 256, 136, {0, -1}},
    {"IQ4_XS 2560x640", 23, 2560, 640, 256, 136, {0, -1}},
    {"IQ4_XS 2560x6144", 23, 2560, 6144, 256, 136, {0, -1}},
    {"IQ4_XS 2560x10240", 23, 2560, 10240, 256, 136, {0, -1}},
    {"IQ4_XS 2560x12288", 23, 2560, 12288, 256, 136, {0, -1}},
    {"Q4_K 2560x640", 12, 2560, 640, 256, 144, {0, 2}},
    {"Q4_K 2560x6144", 12, 2560, 6144, 256, 144, {0, 2}},
    {"Q4_K 2560x10240", 12, 2560, 10240, 256, 144, {0, 2}},
    {"Q4_K 2560x12288", 12, 2560, 12288, 256, 144, {0, 2}},
    {"Q4_K 6144x2560", 12, 6144, 2560, 256, 144, {0, 2}},
    {"Q5_K 2560x512", 13, 2560, 512, 256, 176, {0, 2}},
    {"Q5_K 2560x640", 13, 2560, 640, 256, 176, {0, 2}},
    {"Q5_K 2560x6144", 13, 2560, 6144, 256, 176, {0, 2}},
    {"Q5_K 2560x10240", 13, 2560, 10240, 256, 176, {0, 2}},
    {"Q5_K 2560x12288", 13, 2560, 12288, 256, 176, {0, 2}},
    {"Q5_K 6144x2560", 13, 6144, 2560, 256, 176, {0, 2}},
    {"Q6_K 2560x512", 14, 2560, 512, 256, 210, {208, -1}},
    {"Q6_K 2560x640", 14, 2560, 640, 256, 210, {208, -1}},
    {"Q6_K 2560x6144", 14, 2560, 6144, 256, 210, {208, -1}},
    {"Q6_K 2560x10240", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 2560x12288", 14, 2560, 12288, 256, 210, {208, -1}},
    {"Q6_K 6144x2560", 14, 6144, 2560, 256, 210, {208, -1}},
    {"Q6_K head 2560x248320", 14, 2560, 248320, 256, 210, {208, -1}},
};

bool ck(cudaError_t e, const char* what);

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template<class Launch>
double time_calls(Launch launch, int reps, cudaStream_t s) {
    cudaEvent_t e0, e1;
    if (!ck(cudaEventCreate(&e0), "event") || !ck(cudaEventCreate(&e1), "event"))
        std::exit(1);
    ck(cudaEventRecord(e0, s), "start");
    for (int i = 0; i < reps; ++i) launch(i);
    ck(cudaEventRecord(e1, s), "end");
    if (!ck(cudaEventSynchronize(e1), "time sync")) std::exit(1);
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0 * ms / reps;
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}

int run_case(const Case& c, bool bench, cudaStream_t s) {
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
        double pack_us = 0;
        if (bench)
            pack_us = time_calls([&](int) { native_q8_1_interleave(xq, xil, c.n_in, T, s); }, 128, s);
        for (int r : {0, 1, 2, 4}) {
            native_mmvq_il_tune(r);
            const bool supported = native_mmvq_il_supported(c.type, T, c.n_out, c.n_in);
            ck(cudaMemsetAsync(yil, 0xff, ybytes, s), "poison il");
            native_mmvq_il(c.type, dw[0], xq, xil, yil, c.n_in, c.n_out, T, s);
            const int actual = native_mmvq_il_last_rows();
            if (r == 0) {
                if (actual) ++table_interleaved;
                else ++table_fallback;
            } else if (actual == r) ++forced_interleaved;
            if ((r && actual != r) || (supported != (actual != 0))) {
                std::printf("FAIL path %-26s T=%d requested=%d actual=%d supported=%d\n",
                            c.name, T, r, actual, supported);
                ++bad;
            }
            ck(cudaStreamSynchronize(s), "il");
            ck(cudaMemcpy(b.data(), yil, ybytes, cudaMemcpyDeviceToHost), "read il");
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                float fa;
                std::memcpy(&fa, &a[i], 4);
                if (!std::isfinite(fa)) ++nonfinite;
                if (a[i] != b[i]) ++diff;
            }
            if (diff != 0 || nonfinite != 0) {
                ++bad;
                std::printf("FAIL %-26s T=%d rows=%d: %lld bits differ, %lld non-finite of %zu\n", c.name, T, r, diff, nonfinite, a.size());
            }
            if (bench && r != 0) {
                const int reps = c.n_out > 100000 ? 16 : 64;
                auto ref = [&](int i) { native_mmvq(c.type, dw[i % copies], xq, yref, c.n_in, c.n_out, T, s); };
                auto il = [&](int i) { native_mmvq_il(c.type, dw[i % copies], xq, xil, yil, c.n_in, c.n_out, T, s); };
                for (int i = 0; i < reps; ++i) { ref(i); il(i); }
                std::vector<double> refs, ils, ratios;
                for (int pair = 0; pair < 7; ++pair) {
                    double a_us, b_us;
                    if (pair % 2) { b_us = time_calls(il, reps, s); a_us = time_calls(ref, reps, s); }
                    else { a_us = time_calls(ref, reps, s); b_us = time_calls(il, reps, s); }
                    refs.push_back(a_us);
                    ils.push_back(b_us);
                    ratios.push_back((b_us + pack_us) / a_us);
                }
                std::printf("BENCH %-26s T=%d rows=%d actual=%d multi=%.3f[%.3f,%.3f] il=%.3f[%.3f,%.3f] pack=%.3f us ratio=%.4f[%.4f,%.4f] pairs=7 reps=%d\n",
                            c.name, T, r, actual, median(refs), *std::min_element(refs.begin(), refs.end()),
                            *std::max_element(refs.begin(), refs.end()), median(ils),
                            *std::min_element(ils.begin(), ils.end()), *std::max_element(ils.begin(), ils.end()),
                            pack_us, median(ratios), *std::min_element(ratios.begin(), ratios.end()),
                            *std::max_element(ratios.begin(), ratios.end()), reps);
                std::fflush(stdout);
            }
        }
        native_mmvq_il_tune(0);
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yref); cudaFree(yil);
    }
    for (void* p : dw) cudaFree(p);
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    const bool sm75 = argc > 1 && std::string(argv[1]) == "--sm75-bench";
    const bool bench = sm75 || (argc > 1 && std::string(argv[1]) == "--bench");
    cudaDeviceProp prop{};
    if (!ck(cudaGetDeviceProperties(&prop, 0), "device")) return 1;
    std::printf("DEVICE logical=0 %s sm_%d%d STRATA_MMVQ_IL=%s\n", prop.name, prop.major, prop.minor,
                std::getenv("STRATA_MMVQ_IL") ? std::getenv("STRATA_MMVQ_IL") : "unset");
    if ((prop.major < 8 && !(prop.major == 7 && prop.minor == 5)) ||
        (sm75 && !(prop.major == 7 && prop.minor == 5))) {
        std::printf("SKIP: tuning requires sm_75 or sm_80+\n");
        return 0;
    }
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream")) return 1;
    int bad = 0;
    native_mmvq_set_multi_exact(false);
    native_mmvq_il_tune(1);
    if (native_mmvq_il_supported(14, 4, 10240, 2560)) ++bad;
    native_mmvq_set_multi_exact(true);
    if (native_mmvq_il_supported(21, 4, 10240, 2560) ||
        native_mmvq_il_supported(14, 1, 10240, 2560) ||
        native_mmvq_il_supported(14, 5, 10240, 2560) ||
        native_mmvq_il_supported(14, 4, 0, 2560)) ++bad;
    native_mmvq_il_tune(0);
    if (prop.major == 7 && prop.minor == 5) {
        const char* opt = std::getenv("STRATA_MMVQ_IL");
        const bool enabled = opt && std::strcmp(opt, "1") == 0;
        if (native_mmvq_il_supported(14, 4, 10240, 2560) != enabled ||
            native_mmvq_il_supported(14, 4, 10240, 4096) ||
            native_mmvq_il_supported(14, 4, 10241, 2560) ||
            native_mmvq_il_supported(14, 4, 10240, -256)) {
            std::printf("FAIL sm75 opt-in / exact-shape policy\n");
            ++bad;
        }
    }
    if (sm75) {
        for (const Case& c : SM75_CASES) bad += run_case(c, bench, s);
    } else {
        for (const Case& c : CASES) bad += run_case(c, bench, s);
        for (const Case& c : SM75_CASES) bad += run_case(c, false, s);
    }
    cudaStreamDestroy(s);
    std::printf("PATH forced_interleaved=%d table_interleaved=%d table_fallback=%d\n",
                forced_interleaved, table_interleaved, table_fallback);
    std::printf("%s\n", bad ? "mmvq_il_parity: FAIL" : "mmvq_il_parity: OK (every case, T 2-4, rows 1/2/4 and the table, bitwise)");
    return bad ? 1 : 0;
}
