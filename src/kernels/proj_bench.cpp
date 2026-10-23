// src/kernels/proj_bench.cpp - STRATA_FUSE_PROJ: several projections of one input in one launch against their own calls.
//
//     build/proj_bench [reps] [device]
//
// The groups of dense projections a verify window makes from one q8_1 input (GDN qkv + z, QSA k + v + q|gate) at the
// model's real quant-type mixes, 1-4 columns (and a few edge shapes): A = the calls one by one (native_mmvq,
// STRATA_Q6K_V2 on), B = native_mmvq_proj.  Weights rotate over enough copies to exceed the 64 MB Infinity Cache.  Two
// timings per path: the median of `reps` event-bracketed groups, and a graph of 64 groups (the decode's form: no host
// launch in between) per group.  Every output of B is compared bit for bit with A and with the calls on the V1 kernels
// (STRATA_Q6K_V2 off).  Exit code 1 on a difference.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {

struct Group {
    const char* name;
    int n_in;
    int type[4];
    int n_out[4];
    int n;
};
constexpr int Q4K = 12, Q5K = 13, Q6K = 14, IQ4XS = 23;
const Group GROUPS[] = {
    // GDN qkv + z: the model's seven type pairs (and the all-Q6_K one the Q6K_V2 body takes)
    {"gdn qkv+z Q6Q6", 2560, {Q6K, Q6K}, {10240, 6144}, 2},
    {"gdn qkv+z Q5Q5", 2560, {Q5K, Q5K}, {10240, 6144}, 2},
    {"gdn qkv+z Q4Q4", 2560, {Q4K, Q4K}, {10240, 6144}, 2},
    {"gdn qkv+z Q6Q4", 2560, {Q6K, Q4K}, {10240, 6144}, 2},
    {"gdn qkv+z Q6IQ", 2560, {Q6K, IQ4XS}, {10240, 6144}, 2},
    {"gdn qkv+z Q5Q6", 2560, {Q5K, Q6K}, {10240, 6144}, 2},
    {"gdn qkv+z Q6Q5", 2560, {Q6K, Q5K}, {10240, 6144}, 2},
    {"gdn qkv+z Q5IQ", 2560, {Q5K, IQ4XS}, {10240, 6144}, 2},
    {"gdn qkv+z IQ6", 2560, {IQ4XS, Q6K}, {10240, 6144}, 2},
    {"gdn qkv+z Q5Q4", 2560, {Q5K, Q4K}, {10240, 6144}, 2},
    // QSA k + v + q: the small keys and values must ride with q (k + v alone loses 8-11 us).  These are the model's
    // six triples: all-Q6_K plus (14,14,23), (14,14,13), (23,14,23), (14,13,12), (14,13,13).
    {"qsa kvq Q6Q6Q6", 2560, {Q6K, Q6K, Q6K}, {512, 512, 12288}, 3},
    {"qsa kvq Q6Q6IQ", 2560, {Q6K, Q6K, IQ4XS}, {512, 512, 12288}, 3},
    {"qsa kvq Q6Q6Q5", 2560, {Q6K, Q6K, Q5K}, {512, 512, 12288}, 3},
    {"qsa kvq IQ6IQ", 2560, {IQ4XS, Q6K, IQ4XS}, {512, 512, 12288}, 3},
    {"qsa kvq Q6Q5Q4", 2560, {Q6K, Q5K, Q4K}, {512, 512, 12288}, 3},
    {"qsa kvq Q6Q5Q5", 2560, {Q6K, Q5K, Q5K}, {512, 512, 12288}, 3},
    // a mix no fixed kernel covers: the generic fallback, still bitwise
    {"qsa kvq Q4Q4IQ", 2560, {Q4K, Q4K, IQ4XS}, {512, 512, 12288}, 3},
    {"qsa k+v", 2560, {Q6K, Q6K}, {512, 512}, 2},
    // the edges: odd and one-row segments, a partial last iteration (1280, 3840: 5, 15 blocks a row), 4 segments
    {"edge 1280", 1280, {Q6K, Q4K, IQ4XS}, {63, 1, 513}, 3},
    {"edge 3840", 3840, {Q4K, Q5K, Q6K, IQ4XS}, {5, 2048, 7, 301}, 4},
    {"edge 1024", 1024, {Q5K, IQ4XS}, {1, 1}, 2},
};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

struct Bufs {
    std::vector<std::vector<void*>> w;   // [segment][copy]
    void* xq = nullptr;
    std::vector<float*> y;               // [segment], 4 columns
    int copies = 1;
};

void run(const Group& g, const Bufs& b, int k, int T, bool fused, cudaStream_t s) {
    if (fused) {
        K::MmvqSeg sg[4];
        for (int i = 0; i < g.n; ++i) sg[i] = {g.type[i], b.w[(size_t) i][(size_t) k], b.y[(size_t) i], g.n_out[i]};
        if (!K::native_mmvq_proj(sg, g.n, b.xq, g.n_in, T, s)) { std::printf("native_mmvq_proj did not take %s\n", g.name); std::exit(1); }
        return;
    }
    for (int i = 0; i < g.n; ++i)
        K::native_mmvq(g.type[i], b.w[(size_t) i][(size_t) k], b.xq, b.y[(size_t) i], g.n_in, g.n_out[i], T, s);
}

double median_us(const Group& g, const Bufs& b, int T, bool fused, int reps, cudaStream_t s, std::vector<cudaEvent_t>& ev) {
    for (int r = 0; r < 30; ++r) run(g, b, r % b.copies, T, fused, s);
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(ev[(size_t) 2 * r], s);
        run(g, b, r % b.copies, T, fused, s);
        cudaEventRecord(ev[(size_t) 2 * r + 1], s);
    }
    cudaStreamSynchronize(s);
    std::vector<float> t((size_t) reps);
    for (int r = 0; r < reps; ++r) cudaEventElapsedTime(&t[(size_t) r], ev[(size_t) 2 * r], ev[(size_t) 2 * r + 1]);
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return 1e3 * t[(size_t) reps / 2];
}

double graph_us(const Group& g, const Bufs& b, int T, bool fused, cudaStream_t s) {
    constexpr int N = 64;
    cudaGraph_t gr;
    cudaGraphExec_t ex;
    if (!ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture")) return -1;
    for (int r = 0; r < N; ++r) run(g, b, r % b.copies, T, fused, s);
    if (!ck(cudaStreamEndCapture(s, &gr), "end capture") || !ck(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0), "instantiate"))
        return -1;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> t;
    for (int r = 0; r < 12; ++r) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ex, s);
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (r >= 2) t.push_back(ms);
    }
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(gr);
    return 1e3 * t[t.size() / 2] / N;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(50, std::atoi(argv[1])) : 300;
    const int dev = argc > 2 ? std::atoi(argv[2]) : 0;
    cudaSetDevice(dev);
    cudaStream_t s;
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) return 1;
    std::vector<cudaEvent_t> ev((size_t) 2 * reps);
    for (auto& e : ev) cudaEventCreate(&e);
    std::printf("%-12s %4s %9s %9s %7s %9s %9s %7s  %s\n", "group", "cols", "A us", "B us", "saved", "A graph", "B graph", "saved",
                "bitwise B=A, B=V1");
    long long bad = 0;
    for (const Group& g : GROUPS) {
        std::mt19937 rng(91u + (unsigned) g.n_out[0] + (unsigned) g.n);
        Bufs b;
        size_t total = 0;
        for (int i = 0; i < g.n; ++i) total += K::native_mmvq_weight_bytes(g.type[i], g.n_in, g.n_out[i]);
        b.copies = (int) std::max<size_t>(1, std::min<size_t>(8, (96u << 20) / total + 1));
        b.w.resize((size_t) g.n);
        auto block_bytes = [](int type) { return type == Q4K ? 144 : type == Q5K ? 176 : type == Q6K ? 210 : 136; };
        auto block_scale = [](int type) { return type == Q6K ? 208 : 0; };   // the fp16 d / dm's first half
        auto block_scale_bytes = [](int type) { return type == Q4K || type == Q5K ? 4 : 2; };
        for (int i = 0; i < g.n; ++i) {
            const size_t wb = K::native_mmvq_weight_bytes(g.type[i], g.n_in, g.n_out[i]);
            const int bb = block_bytes(g.type[i]), so = block_scale(g.type[i]);
            std::vector<uint8_t> w(wb);
            for (auto& v : w) v = (uint8_t) (rng() & 0xff);
            for (size_t k = 0; k < wb / (size_t) bb; ++k) {
                uint16_t h[2] = {sane_half(rng), sane_half(rng)};
                std::memcpy(&w[k * (size_t) bb + (size_t) so], h, (size_t) block_scale_bytes(g.type[i]));
            }
            for (int c = 0; c < b.copies; ++c) {
                void* d = nullptr;
                if (!ck(cudaMalloc(&d, wb), "malloc w") || !ck(cudaMemcpy(d, w.data(), wb, cudaMemcpyHostToDevice), "copy w")) return 1;
                b.w[(size_t) i].push_back(d);
            }
            float* y = nullptr;
            if (!ck(cudaMalloc(&y, (size_t) 4 * g.n_out[i] * 4), "malloc y")) return 1;
            b.y.push_back(y);
        }
        std::vector<float> x((size_t) 4 * g.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        float* xf = nullptr;
        if (!ck(cudaMalloc(&xf, x.size() * 4), "malloc x") || !ck(cudaMemcpy(xf, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x") ||
            !ck(cudaMalloc(&b.xq, K::native_q8_1_bytes(g.n_in, 4)), "malloc xq"))
            return 1;
        K::quantize_q8_1_rows(xf, 4, g.n_in, b.xq, s);
        for (int T = 1; T <= 4; ++T) {
            auto out = [&](std::vector<std::vector<uint32_t>>& h) {
                cudaStreamSynchronize(s);
                h.resize((size_t) g.n);
                for (int i = 0; i < g.n; ++i) {
                    h[(size_t) i].resize((size_t) T * g.n_out[i]);
                    cudaMemcpy(h[(size_t) i].data(), b.y[(size_t) i], h[(size_t) i].size() * 4, cudaMemcpyDeviceToHost);
                }
            };
            std::vector<std::vector<uint32_t>> hv1, ha, hb;
            K::native_mmvq_set_q6k_v2(0);
            run(g, b, 0, T, false, s);
            out(hv1);
            K::native_mmvq_set_q6k_v2(1);
            const double ta = median_us(g, b, T, false, reps, s, ev), tga = graph_us(g, b, T, false, s);
            run(g, b, 0, T, false, s);
            out(ha);
            const double tb = median_us(g, b, T, true, reps, s, ev), tgb = graph_us(g, b, T, true, s);
            for (int i = 0; i < g.n; ++i) cudaMemsetAsync(b.y[(size_t) i], 0xff, (size_t) 4 * 4 * g.n_out[i], s);   // poison: B writes all
            run(g, b, 0, T, true, s);
            out(hb);
            K::native_mmvq_set_q6k_v2(-1);
            const bool ba = hb == ha, bv = hb == hv1;
            if (!ba || !bv) ++bad;
            if (!ba || std::getenv("PROJ_DIAG"))
                for (int i = 0; i < g.n; ++i) {
                    long long nd = 0, first = -1, av = 0;
                    for (size_t q = 0; q < ha[(size_t) i].size(); ++q) {
                        if (ha[(size_t) i][q] != hb[(size_t) i][q]) { if (first < 0) first = (long long) q; ++nd; }
                        if (ha[(size_t) i][q] != hv1[(size_t) i][q]) ++av;
                    }
                    std::printf("  seg %d: %lld of %zu differ (first %lld), A vs V1 %lld\n", i, nd, ha[(size_t) i].size(), first, av);
                }
            std::printf("%-12s %4d %9.2f %9.2f %7.2f %9.2f %9.2f %7.2f  %s %s\n", g.name, T, ta, tb, ta - tb, tga, tgb, tga - tgb,
                        ba ? "ok" : "DIFF", bv ? "ok" : "DIFF");
        }
        for (auto& v : b.w) for (void* d : v) cudaFree(d);
        for (float* y : b.y) cudaFree(y);
        cudaFree(xf);
        cudaFree(b.xq);
    }
    std::printf("\nproj_bench: %s (%lld comparisons failed)\n", bad ? "FAILED" : "ok", bad);
    return bad ? 1 : 0;
}
