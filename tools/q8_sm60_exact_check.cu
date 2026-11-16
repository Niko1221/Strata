// tools/q8_sm60_exact_check.cu - STRATA_Q8_SM60_EXACT=1 against the exact kernels, bit for bit, through the engine's own
// native_mmvq: random Q8_0 weights in two device copies, one packed for the Pascal kernel (native_q8_0_sm60_pack_quiet: its
// native_mmvq calls take q8sm60, here the bitwise twin) and one left in the GGUF layout (the exact kernels); the same q8_1
// activations; every float of y compared as bits. The Qwen3.8-Flash-Next Q8_0 shapes and more (n_in 512 .. 10240, the
// small-K path below 1024 included; n_out 512 .. 12288; the head 2560 -> 248320), 1 .. 8 columns. A shape the card has no
// VRAM for is skipped and counted. Needs a GP100 (sm_60). No CMake target; build by hand against a CUDA build:
//
//   nvcc -O2 -std=c++17 -arch=sm_60 -Iinclude tools/q8_sm60_exact_check.cu build/libstrata_kernels.a -o q8_sm60_exact_check
//   STRATA_Q8_SM60=1 STRATA_Q8_SM60_EXACT=1 ./q8_sm60_exact_check     # exit 0: every output bitwise equal
//   STRATA_Q8_SM60=1 ./q8_sm60_exact_check                            # the control: the plain kernel differs
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace K = strata::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
static cudaStream_t g_s;
static long run(int n_in, int n_out, int ncols, std::mt19937& rng) {
    const int nb = n_in / 32;
    std::vector<uint8_t> w((size_t) n_out * nb * 34);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f), xs(-2.f, 2.f);
    for (size_t b = 0; b < (size_t) n_out * nb; ++b) {
        const __half d = __float2half(sc(rng));
        std::memcpy(&w[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = (uint8_t) (int8_t) q(rng);
    }
    std::vector<float> x((size_t) ncols * n_in);
    for (auto& v : x) v = xs(rng);
    void *wa, *wb, *xq; float *xd, *ya, *yb;
    CK(cudaMalloc(&wa, w.size())); CK(cudaMalloc(&wb, w.size()));
    CK(cudaMemcpy(wa, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(wb, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&xd, x.size() * 4)); CK(cudaMemcpy(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&xq, (size_t) ncols * nb * 36));
    CK(cudaMalloc(&ya, (size_t) ncols * n_out * 4)); CK(cudaMalloc(&yb, (size_t) ncols * n_out * 4));
    if (!K::native_q8_0_sm60_pack_quiet(wa, n_in, n_out, "check")) { printf("skip %d x %d (no VRAM)\n", n_out, n_in); cudaFree(wa); cudaFree(wb); cudaFree(xq); cudaFree(xd); cudaFree(ya); cudaFree(yb); return -1; }
    K::native_quantize_q8_1(xd, xq, n_in, ncols, g_s);
    K::native_mmvq(8, wa, xq, ya, n_in, n_out, ncols, g_s);   // packed: the twin
    K::native_mmvq(8, wb, xq, yb, n_in, n_out, ncols, g_s);   // GGUF layout: the exact kernels
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> a((size_t) ncols * n_out), b(a.size());
    CK(cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), yb, b.size() * 4, cudaMemcpyDeviceToHost));
    long diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
    K::native_q8_0_sm60_release(wa);
    cudaFree(wa); cudaFree(wb); cudaFree(xq); cudaFree(xd); cudaFree(ya); cudaFree(yb);
    return diff;
}
int main() {
    CK(cudaStreamCreate(&g_s));
    std::mt19937 rng(1234);
    const int ins[] = {512, 640, 1024, 2048, 2560, 4096, 6144, 10240};
    const int outs[] = {512, 2560, 12288};
    long cases = 0, bad = 0, skipped = 0;
    for (int n_in : ins) for (int n_out : outs) for (int c = 1; c <= 8; ++c) {
        const long d = run(n_in, n_out, c, rng);
        if (d < 0) { ++skipped; continue; }
        ++cases;
        if (d) { ++bad; printf("DIFF n_in %d n_out %d cols %d: %ld of %d floats\n", n_in, n_out, c, d, c * n_out); }
    }
    for (int c : {1, 4, 8}) {                                     // the output head
        const long d = run(2560, 248320, c, rng);
        if (d < 0) { ++skipped; continue; }
        ++cases;
        if (d) { ++bad; printf("DIFF head cols %d: %ld of %d floats\n", c, d, c * 248320); }
    }
    printf("q8_sm60_exact_check: %ld cases, %ld with any differing bit, %ld skipped (VRAM)\n", cases, bad, skipped);
    return bad ? 1 : 0;
}
