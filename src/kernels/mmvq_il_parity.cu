// src/kernels/mmvq_il_parity.cu - native_mmvq_il (1 column from its q8_1 blocks, 2..8 from the interleaved copy)
// against native_mmvq, bitwise, on synthetic matrices of the model files' dense formats and shapes, with every rows a
// warp; the interleaving quantizer's q8_1 blocks against native_quantize_q8_1's.  --bench times both inside CUDA graphs
// (weights cycled past the L2).
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace {

struct Fmt { int type; const char* name; int block_elems, block_bytes, d_off, dmin_off; };
// the dense formats of the IQ3_XXS and UD-Q4_K_XL files; d_off: the fp16 block scale (random bytes elsewhere)
const Fmt kFmts[] = {
    {23, "IQ4_XS", 256, 136, 0, -1}, {12, "Q4_K", 256, 144, 0, 2}, {13, "Q5_K", 256, 176, 0, 2},
    {14, "Q6_K", 256, 210, 208, -1}, {8, "Q8_0", 32, 34, 0, -1},     {21, "IQ3_S", 256, 110, 0, -1},
};

cudaStream_t g_s;
std::mt19937 g_rng(29);

void* dev_weights(const Fmt& f, int n_in, int n_out) {
    const size_t bytes = native_mmvq_weight_bytes(f.type, n_in, n_out);
    std::vector<uint8_t> h(bytes);
    for (auto& b : h) b = (uint8_t) g_rng();
    std::uniform_int_distribution<int> e(0x1c00, 0x2400);   // fp16 scales in ~[0.004, 0.016]
    for (size_t o = 0; o < bytes; o += (size_t) f.block_bytes) {
        const uint16_t d = (uint16_t) e(g_rng);
        std::memcpy(&h[o + f.d_off], &d, 2);
        if (f.dmin_off >= 0) { const uint16_t m = (uint16_t) e(g_rng); std::memcpy(&h[o + f.dmin_off], &m, 2); }
    }
    void* p;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());   // a pageable copy may return before its DMA lands
    return p;
}

struct Act {
    float* x = nullptr;
    void *q_ref = nullptr, *q = nullptr, *il = nullptr;
    int n_in = 0;
};
Act make_act(int n_in) {
    Act a;
    a.n_in = n_in;
    std::vector<float> h((size_t) 8 * n_in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : h) v = nd(g_rng);
    for (size_t i = 0; i < h.size(); i += 97) h[i] *= 20.0f;   // outliers set some blocks' scales
    for (int j = 0; j < 32; ++j) h[(size_t) 3 * n_in + 64 + j] = 0.0f;   // an all-zero block
    CK(cudaMalloc(&a.x, h.size() * 4));
    CK(cudaMemcpy(a.x, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&a.q_ref, native_q8_1_bytes(n_in, 8)));
    CK(cudaMalloc(&a.q, native_q8_1_bytes(n_in, 8)));
    CK(cudaMalloc(&a.il, native_q8_1_il_bytes(n_in, 8)));
    CK(cudaDeviceSynchronize());
    return a;
}

template <typename T>
std::vector<T> fetch(const void* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// every column count 1..8 (1: the one-column kernels), with the table's rows a warp and each of 1, 2, 4
int check_shape(const Fmt& f, const Act& a, int n_out) {
    const int n_in = a.n_in;
    void* w = dev_weights(f, n_in, n_out);
    float *y0, *y1;
    CK(cudaMalloc(&y0, (size_t) 8 * n_out * 4));
    CK(cudaMalloc(&y1, (size_t) 8 * n_out * 4));
    int bad = 0;
    for (int nc = 1; nc <= 8; ++nc) {
        native_quantize_q8_1(a.x, a.q_ref, n_in, nc, g_s);
        if (nc >= 2) native_quantize_q8_1_il(a.x, a.q, a.il, n_in, nc, g_s);
        else native_quantize_q8_1(a.x, a.q, n_in, nc, g_s);
        CK(cudaMemsetAsync(y0, 0xff, (size_t) 8 * n_out * 4, g_s));
        native_mmvq(f.type, w, a.q_ref, y0, n_in, n_out, nc, g_s);
        CK(cudaStreamSynchronize(g_s));
        const size_t qb = native_q8_1_bytes(n_in, nc);
        if (fetch<uint8_t>(a.q_ref, qb) != fetch<uint8_t>(a.q, qb)) {
            ++bad;
            std::printf("  %s %d->%d, %d columns: the q8_1 blocks differ\n", f.name, n_in, n_out, nc);
        }
        const auto r0 = fetch<uint32_t>(y0, (size_t) nc * n_out);
        for (int rows : {0, 1, 2, 4}) {
            CK(cudaMemsetAsync(y1, 0xee, (size_t) 8 * n_out * 4, g_s));
            native_mmvq_il_tune(rows, nc == 1);
            native_mmvq_il(f.type, w, a.q, a.il, y1, n_in, n_out, nc, g_s);
            native_mmvq_il_tune(0, false);
            CK(cudaStreamSynchronize(g_s));
            const auto r1 = fetch<uint32_t>(y1, (size_t) nc * n_out);
            size_t diff = 0;
            for (size_t i = 0; i < r0.size(); ++i) diff += r0[i] != r1[i];
            if (diff) {
                ++bad;
                std::printf("  %s %d->%d, %d columns, rows %d: %zu of %zu values differ\n", f.name, n_in, n_out, nc,
                            rows, diff, r0.size());
            }
        }
    }
    cudaFree(w);
    cudaFree(y0);
    cudaFree(y1);
    return bad;
}

__global__ void spin_kernel(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

// The median of 20 launches, each after ~3 ms of a one-thread spin, as the engine's load: the GPU active at little
// power (a continuous load reaches the 3090's 260 W cap and lowers the clocks, an idle GPU lowers them too).
double time_calls(const std::function<void(int)>& fn, int reps = 48) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    for (int r = 0; r < reps; ++r) fn(r);
    CK(cudaStreamEndCapture(g_s, &g));
    cudaGraphExec_t ex;
    CK(cudaGraphInstantiate(&ex, g, 0));
    cudaGraphDestroy(g);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> ts;
    for (int k = 0; k < 24; ++k) {
        spin_kernel<<<1, 1, 0, g_s>>>(5'500'000);
        CK(cudaEventRecord(a, g_s));
        CK(cudaGraphLaunch(ex, g_s));
        CK(cudaEventRecord(b, g_s));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (k >= 4) ts.push_back(ms * 1000.0 / reps);
    }
    std::sort(ts.begin(), ts.end());
    cudaGraphExecDestroy(ex);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ts[ts.size() / 2];
}

void bench_shape(const Fmt& f, const Act& a, int n_out) {
    const int n_in = a.n_in;
    const double mb = (double) native_mmvq_weight_bytes(f.type, n_in, n_out) / 1e6;
    const int copies = std::max(4, (int) (24.0 / mb) + 1);
    std::vector<void*> w((size_t) copies);
    for (auto& p : w) p = dev_weights(f, n_in, n_out);
    float* y;
    CK(cudaMalloc(&y, (size_t) 8 * n_out * 4));
    std::printf("%-6s %5d -> %5d (%5.2f MB), us (GB/s):", f.name, n_in, n_out, mb);
    for (int nc : {2, 3, 4}) {
        native_quantize_q8_1(a.x, a.q_ref, n_in, nc, g_s);
        native_quantize_q8_1_il(a.x, a.q, a.il, n_in, nc, g_s);
        const double t0 = time_calls([&](int r) { native_mmvq(f.type, w[(size_t) (r % copies)], a.q_ref, y, n_in, n_out, nc, g_s); });
        const double t1 = time_calls([&](int r) {
            native_mmvq_il(f.type, w[(size_t) (r % copies)], a.q, a.il, y, n_in, n_out, nc, g_s);
        });
        std::printf("  %d: %5.1f -> %5.1f (%3.0f)", nc, t0, t1, mb * 1e3 / t1);
    }
    std::printf("\n");
    for (auto p : w) cudaFree(p);
    cudaFree(y);
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    Act a2560 = make_act(2560), a6144 = make_act(6144), a640 = make_act(640);
    if (bench) {
        for (const Fmt& f : kFmts) {
            bench_shape(f, a2560, 12288);
            bench_shape(f, a2560, 10240);
            bench_shape(f, a2560, 6144);
            bench_shape(f, a6144, 2560);
            bench_shape(f, a2560, 640);
        }
        return 0;
    }
    int bad = 0, cases = 0;
    for (const Fmt& f : kFmts) {
        const std::pair<Act*, int> shapes[] = {{&a2560, 10240}, {&a2560, 6144}, {&a6144, 2560}, {&a2560, 12288},
                                               {&a2560, 2050}, {&a2560, 4097}, {&a2560, 8192}, {&a2560, 512},
                                               {&a2560, 640}};
        for (const auto& sh : shapes) {
            bad += check_shape(f, *sh.first, sh.second);
            ++cases;
        }
        if (f.block_elems == 32) {   // Q8_0 also at the shared expert's down shape
            bad += check_shape(f, a640, 2560);
            ++cases;
        }
    }
    std::printf("mmvq_il_parity: %d shape/format cases x 1..8 columns x rows a warp: %s\n", cases,
                bad ? "FAILED" : "bitwise equal");
    return bad ? 1 : 0;
}
