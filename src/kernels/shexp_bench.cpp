// src/kernels/shexp_bench.cpp - the shared expert's dense calls at the window's widths, timed cold, checked bitwise.
//
//     build/shexp_bench [replays] [B rows mode, default -1]
//
// The shared expert's down (IQ4_NL 640 -> 2560, native_mmvq) and its gate / up pairs (native_mmvq_pair_mixed, 2560 ->
// 640) at 2 and 4 columns.  The weights rotate over ~256 MiB of copies (past the L2 and the MALL, as the decode reads a
// layer's weights cold); 256 calls are captured in one graph, and the median replay / 256 is the time per call (the
// graph's own launch gap included, as in the engine's window graph).  Every output is checked bit for bit: the down
// against ncols one-column exact calls, a pair against its two native_mmvq calls.  Exit code 1 on any difference.
// Variants are environment switches read once (STRATA_B7_PF, STRATA_B7_R, ...): one process per variant.
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

struct Fmt { int type, block_elems, block_bytes, scale_at; };
const Fmt IQ4NL{20, 32, 18, 0}, Q4K{12, 256, 144, 0}, IQ4XS{23, 256, 136, 0}, Q6K{14, 256, 210, 208};

struct Case { const char* name; Fmt f1, f2; bool pair; int n_in, n_out; };
const Case CASES[] = {
    {"shexp down IQ4_NL", IQ4NL, IQ4NL, false, 640, 2560},
    {"shexp gu Q4_K+IQ4_XS", Q4K, IQ4XS, true, 2560, 640},
    {"shexp gu Q4_K+Q6_K", Q4K, Q6K, true, 2560, 640},
};
constexpr int kCalls = 256;

#if defined(__HIPCC__)
// floors at the down's grid (320 blocks of 8 warps): nothing at all; the weights streamed once (16-byte loads, one
// float a block written); the activations staged in LDS as B7 does, then the weights streamed
__global__ void floor_empty(float* y) { if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 100000) y[0] = 0.f; }
template<bool STAGE>
__global__ void floor_stream(const uint4* __restrict__ w, long long n16, const int* __restrict__ x, int n_int, float* y) {
    extern __shared__ int sx[];
    const int t = threadIdx.y * 32 + threadIdx.x, nt = blockDim.x * blockDim.y;
    if (STAGE) {
        for (int i = t; i < n_int; i += nt) sx[i] = x[i];
        __syncthreads();
    }
    unsigned acc = STAGE ? (unsigned) sx[t % n_int] : 0u;
    for (long long i = (long long) blockIdx.x * nt + t; i < n16; i += (long long) gridDim.x * nt) {
        const uint4 v = w[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) y[blockIdx.x] = (float) acc;
}
#endif

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
std::vector<uint8_t> weights(const Fmt& f, int n_in, int n_out, std::mt19937& rng) {
    std::vector<uint8_t> w(K::native_mmvq_weight_bytes(f.type, n_in, n_out));
    for (auto& x : w) x = (uint8_t) (rng() & 0xff);
    const size_t nb = (size_t) n_out * (size_t) (n_in / f.block_elems);
    for (size_t k = 0; k < nb; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[k * (size_t) f.block_bytes + (size_t) f.scale_at], &h, 2);
    }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("STRATA_FUSE_SHPAIR", "1", 0);
    const int replays = argc > 1 ? std::max(5, std::atoi(argv[1])) : 30;
    const int bmode = argc > 2 ? std::atoi(argv[2]) : -1;
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream")) return 1;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    long long bad = 0;
    std::printf("%-22s %4s %9s  %s   (B rows mode %d; STRATA_B7_PF=%s STRATA_B7_R=%s STRATA_DENSE_LDS=%s)\n", "case", "cols",
                "us/call", "bitwise", bmode, getenv("STRATA_B7_PF") ? getenv("STRATA_B7_PF") : "-",
                getenv("STRATA_B7_R") ? getenv("STRATA_B7_R") : "-", getenv("STRATA_DENSE_LDS") ? getenv("STRATA_DENSE_LDS") : "-");
    for (const Case& c : CASES) {
        std::mt19937 rng(91u + (unsigned) c.f1.type * 7u + (unsigned) c.f2.type);
        const std::vector<uint8_t> w1 = weights(c.f1, c.n_in, c.n_out, rng);
        const std::vector<uint8_t> w2 = c.pair ? weights(c.f2, c.n_in, c.n_out, rng) : std::vector<uint8_t>();
        const size_t per = w1.size() + w2.size();
        const int copies = (int) std::max<size_t>(2, (256u << 20) / per);
        std::vector<void*> d1((size_t) copies), d2((size_t) copies, nullptr);
        for (int k = 0; k < copies; ++k) {
            if (!ck(cudaMalloc(&d1[(size_t) k], w1.size()), "malloc") ||
                !ck(cudaMemcpy(d1[(size_t) k], w1.data(), w1.size(), cudaMemcpyHostToDevice), "copy")) return 1;
            if (c.pair && (!ck(cudaMalloc(&d2[(size_t) k], w2.size()), "malloc") ||
                           !ck(cudaMemcpy(d2[(size_t) k], w2.data(), w2.size(), cudaMemcpyHostToDevice), "copy"))) return 1;
        }
        std::vector<float> x((size_t) 8 * c.n_in);
        std::normal_distribution<float> nx(0.f, 1.f);
        for (auto& v : x) v = nx(rng);
        float *xf = nullptr, *y1 = nullptr, *y2 = nullptr, *r1 = nullptr, *r2 = nullptr;
        void* xq = nullptr;
        const size_t yb = (size_t) 8 * c.n_out * 4;
        if (!ck(cudaMalloc(&xf, x.size() * 4), "x") || !ck(cudaMemcpy(xf, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x") ||
            !ck(cudaMalloc(&xq, K::native_q8_1_bytes(c.n_in, 8)), "xq") || !ck(cudaMalloc(&y1, yb), "y") ||
            !ck(cudaMalloc(&y2, yb), "y") || !ck(cudaMalloc(&r1, yb), "r") || !ck(cudaMalloc(&r2, yb), "r")) return 1;
        K::quantize_q8_1_rows(xf, 8, c.n_in, xq, s);
        for (int T : {2, 4}) {
            auto call = [&](int k, float* a, float* b) -> bool {
                if (!c.pair) { K::native_mmvq(c.f1.type, d1[(size_t) k], xq, a, c.n_in, c.n_out, T, s); return true; }
                return K::native_mmvq_pair_mixed(c.f1.type, c.f2.type, d1[(size_t) k], d2[(size_t) k], xq, a, b, c.n_in,
                                                 c.n_out, T, s);
            };
            K::native_mmvq_set_b6_rows(bmode);
            if (!call(0, y1, y2)) { std::printf("%-22s %4d  (no pair launch)\n", c.name, T); continue; }
            // the reference
            if (!c.pair) {
                K::native_mmvq_set_b6_rows(0);
                for (int j = 0; j < T; ++j)
                    K::native_mmvq(c.f1.type, d1[0], (const uint8_t*) xq + K::native_q8_1_bytes(c.n_in, 1) * (size_t) j,
                                   r1 + (size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
                K::native_mmvq_set_b6_rows(bmode);
            } else {
                K::native_mmvq(c.f1.type, d1[0], xq, r1, c.n_in, c.n_out, T, s);
                K::native_mmvq(c.f2.type, d2[0], xq, r2, c.n_in, c.n_out, T, s);
            }
            if (!ck(cudaStreamSynchronize(s), "run")) return 1;
            const size_t n = (size_t) T * c.n_out;
            std::vector<uint32_t> a(n), b(n), ra(n), rb(n);
            cudaMemcpy(a.data(), y1, n * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(ra.data(), r1, n * 4, cudaMemcpyDeviceToHost);
            bool same = std::memcmp(a.data(), ra.data(), n * 4) == 0;
            if (c.pair) {
                cudaMemcpy(b.data(), y2, n * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(rb.data(), r2, n * 4, cudaMemcpyDeviceToHost);
                same = same && std::memcmp(b.data(), rb.data(), n * 4) == 0;
            }
            if (!same) ++bad;
            // the graph of kCalls calls over the copies
            cudaGraph_t g;
            cudaGraphExec_t ge;
            if (!ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture")) return 1;
            for (int i = 0; i < kCalls; ++i) call(i % copies, y1, y2);
            if (!ck(cudaStreamEndCapture(s, &g), "capture end") || !ck(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "inst"))
                return 1;
            std::vector<float> t;
            for (int r = 0; r < replays + 3; ++r) {
                cudaEventRecord(e0, s);
                cudaGraphLaunch(ge, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                if (r >= 3) t.push_back(ms);
            }
            std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
            uint64_t hsh = 1469598103934665603ull;   // FNV-1a of the outputs: equal across builds = bitwise across builds
            for (uint32_t u : a) hsh = (hsh ^ u) * 1099511628211ull;
            if (c.pair) for (uint32_t u : b) hsh = (hsh ^ u) * 1099511628211ull;
            std::printf("%-22s %4d %9.3f  %s  hash %016llx\n", c.name, T, 1000.0 * t[t.size() / 2] / kCalls, same ? "ok" : "DIFF",
                        (unsigned long long) hsh);
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
        }
        for (int k = 0; k < copies; ++k) { cudaFree(d1[(size_t) k]); if (d2[(size_t) k]) cudaFree(d2[(size_t) k]); }
        cudaFree(xf); cudaFree(xq); cudaFree(y1); cudaFree(y2); cudaFree(r1); cudaFree(r2);
    }
#if defined(__HIPCC__)
    {   // the floors at the down's shape: 0.92 MB of weights, 4 columns of activations (2880 bytes)
        const size_t wb = K::native_mmvq_weight_bytes(20, 640, 2560);
        const int copies = (int) ((256u << 20) / wb);
        std::vector<void*> d((size_t) copies);
        for (auto& p : d) { cudaMalloc(&p, wb); cudaMemset(p, 1, wb); }
        int* xi = nullptr;
        float* y = nullptr;
        cudaMalloc(&xi, 2880); cudaMemset(xi, 0, 2880); cudaMalloc(&y, 4096 * 4);
        for (int f = 0; f < 3; ++f) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            for (int i = 0; i < kCalls; ++i) {
                if (f == 0) floor_empty<<<320, dim3(32, 8), 0, s>>>(y);
                else if (f == 1) floor_stream<false><<<320, dim3(32, 8), 0, s>>>((const uint4*) d[(size_t) (i % copies)], (long long) (wb / 16), xi, 720, y);
                else floor_stream<true><<<320, dim3(32, 8), 2880, s>>>((const uint4*) d[(size_t) (i % copies)], (long long) (wb / 16), xi, 720, y);
            }
            cudaStreamEndCapture(s, &g);
            cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
            std::vector<float> t;
            for (int r = 0; r < replays + 3; ++r) {
                cudaEventRecord(e0, s); cudaGraphLaunch(ge, s); cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                if (r >= 3) t.push_back(ms);
            }
            std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
            const char* nm[3] = {"floor: empty kernel", "floor: stream 0.92MB", "floor: stage + stream"};
            std::printf("%-22s %4s %9.3f\n", nm[f], "", 1000.0 * t[t.size() / 2] / kCalls);
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        }
        for (auto& p : d) cudaFree(p);
        cudaFree(xi); cudaFree(y);
    }
#endif
    std::printf("shexp_bench: %s (%lld differ)\n", bad ? "FAILED" : "ok", bad);
    return bad ? 1 : 0;
}
