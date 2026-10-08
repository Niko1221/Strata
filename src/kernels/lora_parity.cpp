// src/kernels/lora_parity.cpp - the LoRA kernels (strata/kernels/lora.hpp) against a host reference.
//   1. y += B (A x) for 1, 3 and 8 tokens (the decode / verify launches), strided rows, against double precision.
//   2. the same for 300 tokens (the prompt path's tiled products), fp32 and FP16 activations.
//   3. switched off: y BITWISE unchanged, eager and replayed from a graph captured while on (the flag is read on
//      the device); switched on again, the same graph adds the adapter again.
//   4. a layer without an adapter is untouched; a captured call of more than kLoraCaptureTokens tokens throws.

#include "strata/kernels/lora.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

template <typename T>
void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// max |got - want| / max(1, |want|) over the n x n_out outputs
double rel_err(const std::vector<float>& got, const std::vector<double>& want, int64_t n, int64_t n_out, int64_t ldy) {
    double worst = 0.0;
    for (int64_t t = 0; t < n; ++t)
        for (int64_t o = 0; o < n_out; ++o) {
            const double w = want[(size_t) (t * n_out + o)], g = got[(size_t) (t * ldy + o)];
            worst = std::max(worst, std::fabs(g - w) / std::max(1.0, std::fabs(w)));
        }
    return worst;
}

}  // namespace

int main() {
    constexpr int64_t L = 8, NIN = 6144, NOUT = 2560, R = 50, LDX = NIN + 64, LDY = NOUT + 32;
    constexpr int64_t kLayer = 3, kOff = 2;   // adapted / not adapted
    constexpr float kScale = 0.75f;
    std::mt19937 rng(4321);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<k::LoraLayerHost> layers((size_t) L);
    k::LoraLayerHost& h = layers[(size_t) kLayer];
    h.rank = R; h.n_in = NIN; h.n_out = NOUT;
    h.a.resize((size_t) (R * NIN));
    h.b.resize((size_t) (NOUT * R));
    for (float& v : h.a) v = nd(rng) * 0.02f;
    for (float& v : h.b) v = nd(rng) * 0.05f * kScale;   // the loader folds the scale into B
    layers[5] = h;                                       // a second adapted layer (5), kOff stays empty
    std::string err;
    if (!k::lora_upload(layers, err)) { std::fprintf(stderr, "lora_upload: %s\n", err.c_str()); return 2; }
    check(k::lora().covers(kLayer) && k::lora().covers(5) && !k::lora().covers(kOff) && !k::lora().covers(L),
          "covers(): adapted layers only");
    check(k::lora_enabled(), "starts on");

    constexpr int64_t TMAX = 300;
    std::vector<float> x((size_t) (TMAX * LDX)), y0((size_t) (TMAX * LDY));
    for (float& v : x) v = nd(rng);
    for (float& v : y0) v = nd(rng);
    std::vector<uint16_t> x16(x.size());
    for (size_t i = 0; i < x.size(); ++i) x16[i] = strata::kernels::f16_from_f32(x[i]);
    float* dx = dalloc<float>(x.size());
    uint16_t* dx16 = dalloc<uint16_t>(x16.size());
    float* dy = dalloc<float>(y0.size());
    up(dx, x);
    up(dx16, x16);

    // reference y0 + B (A x) in double, from the fp32 or the fp16-rounded x
    auto reference = [&](int64_t n, bool f16) {
        std::vector<double> want((size_t) (n * NOUT));
        std::vector<double> hv((size_t) R);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t r = 0; r < R; ++r) {
                double acc = 0.0;
                for (int64_t i = 0; i < NIN; ++i) {
                    const float xv = f16 ? strata::kernels::f32_from_f16(x16[(size_t) (t * LDX + i)]) : x[(size_t) (t * LDX + i)];
                    acc += (double) h.a[(size_t) (r * NIN + i)] * xv;
                }
                hv[(size_t) r] = acc;
            }
            for (int64_t o = 0; o < NOUT; ++o) {
                double acc = y0[(size_t) (t * LDY + o)];
                for (int64_t r = 0; r < R; ++r) acc += (double) h.b[(size_t) (o * R + r)] * hv[(size_t) r];
                want[(size_t) (t * NOUT + o)] = acc;
            }
        }
        return want;
    };

    std::printf("1-2. y += B (A x) against double precision\n");
    for (const int64_t n : {1, 3, 8, 300}) {
        for (const bool f16 : {false, true}) {
            up(dy, y0);
            if (f16) k::lora_apply_f16(kLayer, dx16, LDX, n, dy, LDY, nullptr);
            else k::lora_apply(kLayer, dx, LDX, n, dy, LDY, nullptr);
            ck(cudaDeviceSynchronize(), "apply");
            const double e = rel_err(down(dy, y0.size()), reference(n, f16), n, NOUT, LDY);
            char what[96];
            std::snprintf(what, sizeof what, "%3lld tokens, %s x: rel err %.2e (< 1e-4)", (long long) n, f16 ? "fp16" : "fp32", e);
            check(e < 1e-4, what);
        }
    }
    {   // the padding columns past n_out are never written
        up(dy, y0);
        k::lora_apply(kLayer, dx, LDX, 300, dy, LDY, nullptr);
        const std::vector<float> got = down(dy, y0.size());
        bool pad_ok = true;
        for (int64_t t = 0; t < 300; ++t)
            for (int64_t o = NOUT; o < LDY; ++o) pad_ok &= got[(size_t) (t * LDY + o)] == y0[(size_t) (t * LDY + o)];
        check(pad_ok, "row padding untouched");
    }

    std::printf("3. the request switch\n");
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
    k::lora_apply(kLayer, dx, LDX, 8, dy, LDY, s);
    ck(cudaStreamEndCapture(s, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");

    k::lora_set_enabled(false);
    check(!k::lora_enabled(), "lora_set_enabled(false)");
    for (const int64_t n : {1, 8, 300}) {
        up(dy, y0);
        k::lora_apply(kLayer, dx, LDX, n, dy, LDY, nullptr);
        k::lora_apply_f16(kLayer, dx16, LDX, n, dy, LDY, nullptr);
        ck(cudaDeviceSynchronize(), "apply off");
        char what[96];
        std::snprintf(what, sizeof what, "off, %3lld tokens: y bitwise unchanged", (long long) n);
        check(std::memcmp(down(dy, y0.size()).data(), y0.data(), y0.size() * sizeof(float)) == 0, what);
    }
    up(dy, y0);
    ck(cudaGraphLaunch(exec, s), "launch off");
    ck(cudaStreamSynchronize(s), "sync off");
    check(std::memcmp(down(dy, y0.size()).data(), y0.data(), y0.size() * sizeof(float)) == 0,
          "off, the graph captured on: y bitwise unchanged");
    k::lora_set_enabled(true);
    up(dy, y0);
    ck(cudaGraphLaunch(exec, s), "launch on");
    ck(cudaStreamSynchronize(s), "sync on");
    check(rel_err(down(dy, y0.size()), reference(8, false), 8, NOUT, LDY) < 1e-4, "on again, the same graph: the adapter");

    {   // not a check: the cost per adapted layer, decode / verify windows and a prompt chunk
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (const int64_t n : {1, 4, 8, 300}) {
            for (int i = 0; i < 20; ++i) k::lora_apply(kLayer, dx, LDX, n, dy, LDY, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < 500; ++i) k::lora_apply(kLayer, dx, LDX, n, dy, LDY, s);
            cudaEventRecord(e1, s);
            ck(cudaEventSynchronize(e1), "timing");
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("  timing: %lld tokens, %.1f us per call (rank %lld, %lld -> %lld)\n", (long long) n,
                        1000.0f * ms / 500.0f, (long long) R, (long long) NIN, (long long) NOUT);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }

    std::printf("4. edges\n");
    up(dy, y0);
    k::lora_apply(kOff, dx, LDX, 8, dy, LDY, nullptr);
    k::lora_apply(kOff, dx, LDX, 300, dy, LDY, nullptr);
    ck(cudaDeviceSynchronize(), "apply kOff");
    check(std::memcmp(down(dy, y0.size()).data(), y0.data(), y0.size() * sizeof(float)) == 0, "a layer without an adapter: untouched");
    bool threw = false;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture 2");
    try { k::lora_apply(kLayer, dx, LDX, k::kLoraCaptureTokens + 1, dy, LDY, s); } catch (const std::runtime_error&) { threw = true; }
    cudaGraph_t g2 = nullptr;
    cudaStreamEndCapture(s, &g2);
    if (g2) cudaGraphDestroy(g2);
    cudaGetLastError();
    check(threw, "captured call above kLoraCaptureTokens throws");

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    std::printf(g_fail ? "lora_parity: %d FAILED\n" : "lora_parity: all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
