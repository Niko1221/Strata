// engine/src/inkling2/cuda/ink_shared_test.cu - bitwise test of k_ink_flash_gqa_splitk_shared against k_ink_flash_gqa_splitk (and a kernel-level timing).
//
// Build (no model needed):  nvcc -O3 -std=c++20 --expt-relaxed-constexpr -I<repo>/engine/src/inkling2/cuda -I<repo>/engine/include ink_shared_test.cu -lcublas -o ink_shared_test
// Run: ./ink_shared_test [--time-only]    exit code 0 = every partial (m, l, o) and every merged output is bit-identical in every case.
#include "ink_attn.cuh"
#include "ink_attn_shared.cuh"
#include "ink_cuda.cuh"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ink;

static __device__ __forceinline__ uint32_t mix32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((x ^ (x >> 31)) >> 16);
}
static __device__ __forceinline__ float rnd(uint64_t i, uint64_t seed) { return ((mix32(i * 2654435761ull + seed) & 0xFFFFFF) * (1.0f / 16777216.0f)) - 0.5f; }

__global__ void k_fill_f16(__half* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half_rn(rnd(i, seed) * amp);
}
__global__ void k_fill_u8(uint8_t* p, size_t n, uint64_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint8_t v = (uint8_t)(mix32(i + seed * 77) & 0xFF);
        if ((v & 0x7F) == 0x7F) v &= 0xBF;   // no e4m3 NaN
        p[i] = v;
    }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed, float amp, float off) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = rnd(i, seed) * amp + off;
}

struct Tree { int M; int tp0; uint8_t anc[32 * 8]; uint8_t dep[32]; int32_t pos[32]; };

static Tree make_tree(int M, int tp0, uint32_t seed) {
    std::mt19937 rng(seed);
    Tree t{};
    t.M = M; t.tp0 = tp0;
    int parent[32] = {};
    t.dep[0] = 0;
    for (int i = 1; i < M; ++i) {
        int p;
        do { p = (int)(rng() % i); } while (t.dep[p] >= 7);
        parent[i] = p;
        t.dep[i] = t.dep[p] + 1;
    }
    for (int i = 0; i < M; ++i) {
        int chain[8], n = 0;
        for (int c = i;; c = parent[c]) { chain[n++] = c; if (c == 0) break; }
        for (int d = 0; d < n; ++d) t.anc[i * 8 + d] = (uint8_t)chain[n - 1 - d];
        t.pos[i] = tp0 + t.dep[i];
    }
    return t;
}

struct Bufs {
    InkKvVisitDesc kv{};
    float *q = nullptr, *rel = nullptr, *sinks = nullptr, *tk = nullptr, *tv = nullptr;
    int32_t* pos = nullptr;
    uint8_t* anc = nullptr; uint8_t* dep = nullptr;
    float *pm[2] = {}, *pl[2] = {}, *po[2] = {}, *out[2] = {}, *gate = nullptr;
};

static int run_case(Bufs& b, int fmt, int M, int tp0, uint32_t seed, int n_sp, const char* tag, bool verbose, const int* rgs, int n_rg, bool tree_mode) {
    Tree t = make_tree(M, tp0, seed);
    if (!tree_mode) {   // plain consecutive rows (mode 0 style) are not routed to the new kernel by the runner, but the kernel handles tree_k == nullptr too
        for (int i = 0; i < M; ++i) t.pos[i] = tp0 + i;
    }
    cudaMemcpy(b.pos, t.pos, M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.anc, t.anc, 32 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(b.dep, t.dep, 32, cudaMemcpyHostToDevice);
    const size_t nrel = (size_t)M * kInkHeads * (kInkRelExtent + 1);
    k_fill_f32<<<256, 256>>>(b.q, (size_t)M * kInkHeads * kInkHeadDim, seed * 3 + 1, 3.0f, 0.0f);
    k_fill_f32<<<256, 256>>>(b.rel, nrel, seed * 3 + 2, 2.0f, 0.0f);
    k_fill_f32<<<4, 256>>>(b.sinks, (size_t)M * kInkHeads, seed * 3 + 3, 4.0f, 0.0f);
    k_fill_f32<<<256, 256>>>(b.tk, (size_t)M * kInkKVHeads * kInkHeadDim, seed * 5 + 1, 2.0f, 0.0f);
    k_fill_f32<<<256, 256>>>(b.tv, (size_t)M * kInkKVHeads * kInkHeadDim, seed * 5 + 2, 2.0f, 0.0f);
    k_fill_f32<<<4, 256>>>(b.gate, (size_t)M * kInkHeads * kInkHeadDim, seed * 7, 2.0f, 0.0f);
    cudaDeviceSynchronize();
    const size_t np = (size_t)M * kInkHeads * n_sp;
    const float* tk = tree_mode ? b.tk : nullptr;
    const float* tv = tree_mode ? b.tv : nullptr;
    const uint8_t* anc = tree_mode ? b.anc : nullptr;
    const uint8_t* dep = tree_mode ? b.dep : nullptr;
    cudaMemset(b.pm[0], 0xFF, np * 4); cudaMemset(b.pl[0], 0xFF, np * 4); cudaMemset(b.po[0], 0xFF, np * 128 * 4);
    k_ink_flash_gqa_splitk<<<dim3(kInkKVHeads, n_sp, M), 128>>>(b.q, b.rel, b.sinks, b.kv, nullptr, 0, b.pos, tk, tv, anc, dep, 0, b.pm[0], b.pl[0], b.po[0], M, n_sp);
    k_ink_flash_gqa_merge<<<dim3(kInkHeads, M), 128>>>(b.pm[0], b.pl[0], b.po[0], b.gate, b.out[0], M, n_sp);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("CUDA error %s\n", cudaGetErrorString(e)); return 1000; }
    std::vector<float> hm0(np), hl0(np), ho0(np * 128), hout0((size_t)M * kInkHeads * kInkHeadDim);
    cudaMemcpy(hm0.data(), b.pm[0], np * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hl0.data(), b.pl[0], np * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ho0.data(), b.po[0], np * 128 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hout0.data(), b.out[0], hout0.size() * 4, cudaMemcpyDeviceToHost);
    int bad_total = 0;
    for (int ri = 0; ri < n_rg; ++ri) {
        const int rg = rgs[ri];
        cudaMemset(b.pm[1], 0xFF, np * 4); cudaMemset(b.pl[1], 0xFF, np * 4); cudaMemset(b.po[1], 0xFF, np * 128 * 4);
        const bool ok = ink_launch_splitk_shared_raw(0, rg, b.q, b.rel, b.sinks, b.kv, 0, b.pos, tk, tv, anc, b.pm[1], b.pl[1], b.po[1], M, n_sp);
        if (!ok) { std::printf("  [%s] rg=%d not launched\n", tag, rg); return 1000; }
        k_ink_flash_gqa_merge<<<dim3(kInkHeads, M), 128>>>(b.pm[1], b.pl[1], b.po[1], b.gate, b.out[1], M, n_sp);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("CUDA error %s\n", cudaGetErrorString(e)); return 1000; }
        std::vector<float> hm1(np), hl1(np), ho1(np * 128), hout1(hout0.size());
        cudaMemcpy(hm1.data(), b.pm[1], np * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hl1.data(), b.pl[1], np * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ho1.data(), b.po[1], np * 128 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hout1.data(), b.out[1], hout1.size() * 4, cudaMemcpyDeviceToHost);
        long bm = 0, bl = 0, bo = 0, bf = 0;
        for (size_t i = 0; i < np; ++i) { if (std::memcmp(&hm0[i], &hm1[i], 4)) ++bm; if (std::memcmp(&hl0[i], &hl1[i], 4)) ++bl; }
        for (size_t i = 0; i < np * 128; ++i) if (std::memcmp(&ho0[i], &ho1[i], 4)) ++bo;
        for (size_t i = 0; i < hout0.size(); ++i) if (std::memcmp(&hout0[i], &hout1[i], 4)) ++bf;
        const int bad = (bm || bl || bo || bf) ? 1 : 0;
        bad_total += bad;
        if (verbose || bad)
            std::printf("  [%s fmt=%d M=%2d tp0=%6d n_sp=%d rg=%d] partial m/l/o mismatches %ld/%ld/%ld, merged-output mismatches %ld of %zu -> %s\n", tag, fmt, M, tp0, n_sp, rg, bm, bl, bo, bf,
                        hout0.size(), bad ? "DIFFERENT" : "bit-identical");
    }
    return bad_total;
}

static float time_kernel(bool shared, int rg, Bufs& b, int M, int n_sp, bool tree_mode, int iters) {
    const float* tk = tree_mode ? b.tk : nullptr;
    const float* tv = tree_mode ? b.tv : nullptr;
    const uint8_t* anc = tree_mode ? b.anc : nullptr;
    const uint8_t* dep = tree_mode ? b.dep : nullptr;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int w = 0; w < 3; ++w) {
        if (shared) ink_launch_splitk_shared_raw(0, rg, b.q, b.rel, b.sinks, b.kv, 0, b.pos, tk, tv, anc, b.pm[1], b.pl[1], b.po[1], M, n_sp);
        else k_ink_flash_gqa_splitk<<<dim3(kInkKVHeads, n_sp, M), 128>>>(b.q, b.rel, b.sinks, b.kv, nullptr, 0, b.pos, tk, tv, anc, dep, 0, b.pm[0], b.pl[0], b.po[0], M, n_sp);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) {
        if (shared) ink_launch_splitk_shared_raw(0, rg, b.q, b.rel, b.sinks, b.kv, 0, b.pos, tk, tv, anc, b.pm[1], b.pl[1], b.po[1], M, n_sp);
        else k_ink_flash_gqa_splitk<<<dim3(kInkKVHeads, n_sp, M), 128>>>(b.q, b.rel, b.sinks, b.kv, nullptr, 0, b.pos, tk, tv, anc, dep, 0, b.pm[0], b.pl[0], b.po[0], M, n_sp);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

int main(int argc, char** argv) {
    bool time_only = false;
    int cap = 140000;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--time-only")) time_only = true;
        else if (!std::strcmp(argv[i], "--cap")) cap = std::atoi(argv[++i]);
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU %s sm_%d%d, KV cap %d positions\n", prop.name, prop.major, prop.minor, cap);
    const int MAXM = 32, N_SP = 32;
    Bufs b;
    const size_t kv_elems = (size_t)cap * kInkKVHeads * kInkHeadDim;
    __half *kf16, *vf16; uint8_t *kf8, *vf8; float *ksc, *vsc;
    cudaMalloc(&kf16, kv_elems * 2); cudaMalloc(&vf16, kv_elems * 2);
    cudaMalloc(&kf8, kv_elems); cudaMalloc(&vf8, kv_elems);
    cudaMalloc(&ksc, (size_t)cap * kInkKVHeads * 4); cudaMalloc(&vsc, (size_t)cap * kInkKVHeads * 4);
    k_fill_f16<<<1024, 256>>>(kf16, kv_elems, 11, 3.0f);
    k_fill_f16<<<1024, 256>>>(vf16, kv_elems, 12, 2.0f);
    k_fill_u8<<<1024, 256>>>(kf8, kv_elems, 13);
    k_fill_u8<<<1024, 256>>>(vf8, kv_elems, 14);
    k_fill_f32<<<256, 256>>>(ksc, (size_t)cap * kInkKVHeads, 15, 0.02f, 0.03f);
    k_fill_f32<<<256, 256>>>(vsc, (size_t)cap * kInkKVHeads, 16, 0.02f, 0.03f);
    cudaMalloc(&b.q, (size_t)MAXM * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&b.rel, (size_t)MAXM * kInkHeads * (kInkRelExtent + 1) * 4);
    cudaMalloc(&b.sinks, (size_t)MAXM * kInkHeads * 4);
    cudaMalloc(&b.tk, (size_t)MAXM * kInkKVHeads * kInkHeadDim * 4);
    cudaMalloc(&b.tv, (size_t)MAXM * kInkKVHeads * kInkHeadDim * 4);
    cudaMalloc(&b.gate, (size_t)MAXM * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&b.pos, MAXM * 4); cudaMalloc(&b.anc, 32 * 8); cudaMalloc(&b.dep, 32);
    const size_t npmax = (size_t)MAXM * kInkHeads * N_SP;
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&b.pm[i], npmax * 4); cudaMalloc(&b.pl[i], npmax * 4); cudaMalloc(&b.po[i], npmax * 128 * 4);
        cudaMalloc(&b.out[i], (size_t)MAXM * kInkHeads * kInkHeadDim * 4);
    }
    cudaDeviceSynchronize();

    int bad = 0, ncases = 0;
    const int rgs[5] = {1, 2, 4, 6, 8};
    if (!time_only) {
        for (int fmt : {0, 1, 3}) {
            b.kv = InkKvVisitDesc{};
            b.kv.k_f16 = kf16; b.kv.v_f16 = vf16;
            b.kv.k_f8 = kf8; b.kv.v_f8 = vf8; b.kv.k_sc = ksc; b.kv.v_sc = vsc;
            b.kv.cap_pos = cap; b.kv.dev_cap = cap; b.kv.host_cap = 0; b.kv.is_swa = false; b.kv.kv_fmt = (uint8_t)fmt;
            if (fmt == 3) b.kv.v_f8 = vf8;   // K8V4 reads v as packed 4-bit: any bytes are valid
            for (int M : {2, 3, 5, 8, 12, 16}) {
                for (int tp0 : {1, 3, 63, 64, 1000, 4097, 33333, 100003, cap - 20}) {
                    for (int tree_mode = 1; tree_mode >= 0; --tree_mode) {
                        const int r = run_case(b, fmt, M, tp0, 1000 + M * 31 + tp0 % 97, N_SP, tree_mode ? "tree" : "flat", false, rgs, 5, tree_mode);
                        bad += r; ++ncases;
                        if (r >= 1000) return 2;
                    }
                }
            }
        }
        std::printf("RESULT: %d cases x 5 row-group sizes, %d case(s) differ -> %s\n", ncases, bad, bad ? "DIFFERENT" : "ALL BIT-IDENTICAL");
    }
    // kernel-level timing on VRAM-resident FP16 KV, M rows
    b.kv = InkKvVisitDesc{};
    b.kv.k_f16 = kf16; b.kv.v_f16 = vf16; b.kv.cap_pos = cap; b.kv.dev_cap = cap; b.kv.kv_fmt = 0;
    for (int tp0 : {7990, 32000, 130000}) {
        if (tp0 + 20 > cap) continue;
        for (int M : {4, 8, 12}) {
            Tree t = make_tree(M, tp0, 5);
            cudaMemcpy(b.pos, t.pos, M * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(b.anc, t.anc, 32 * 8, cudaMemcpyHostToDevice);
            cudaMemcpy(b.dep, t.dep, 32, cudaMemcpyHostToDevice);
            k_fill_f32<<<256, 256>>>(b.q, (size_t)M * kInkHeads * kInkHeadDim, 3, 3.0f, 0.0f);
            k_fill_f32<<<256, 256>>>(b.rel, (size_t)M * kInkHeads * (kInkRelExtent + 1), 4, 2.0f, 0.0f);
            k_fill_f32<<<4, 256>>>(b.sinks, (size_t)M * kInkHeads, 5, 4.0f, 0.0f);
            cudaDeviceSynchronize();
            const float t_old = time_kernel(false, 0, b, M, N_SP, true, 20);
            std::printf("timing FP16 VRAM KV, one global visit, tp0=%6d M=%2d: old %.3f ms", tp0, M, t_old);
            for (int rg : {1, 2, 4, 6, 8}) std::printf(" | shared rg=%d %.3f ms (x%.2f)", rg, time_kernel(true, rg, b, M, N_SP, true, 20), t_old / time_kernel(true, rg, b, M, N_SP, true, 20));
            std::printf("\n");
        }
        // one-row decode reference
        {
            const int M = 1;
            int32_t p = tp0; cudaMemcpy(b.pos, &p, 4, cudaMemcpyHostToDevice);
            k_fill_f32<<<256, 256>>>(b.q, (size_t)kInkHeads * kInkHeadDim, 3, 3.0f, 0.0f);
            k_fill_f32<<<256, 256>>>(b.rel, (size_t)kInkHeads * (kInkRelExtent + 1), 4, 2.0f, 0.0f);
            k_fill_f32<<<4, 256>>>(b.sinks, (size_t)kInkHeads, 5, 4.0f, 0.0f);
            cudaDeviceSynchronize();
            std::printf("timing FP16 VRAM KV, one global visit, tp0=%6d M=%2d (plain decode row): old %.3f ms\n", tp0, M, time_kernel(false, 0, b, M, N_SP, false, 20));
        }
    }
    // kernel-level timing with the KV tail in mapped pinned HOST memory (the default placement of a 16 GB card at long context): dev_cap cells in VRAM, the rest read over PCIe
    {
        const int dev_cap = 6400;
        const size_t host_n = (size_t)(cap - dev_cap) * kInkKVHeads * kInkHeadDim;
        __half *hk, *hv; uint8_t *hk8, *hv8; float *hksc, *hvsc;
        cudaHostAlloc(&hk, host_n * 2, cudaHostAllocMapped); cudaHostAlloc(&hv, host_n * 2, cudaHostAllocMapped);
        cudaHostAlloc(&hk8, host_n, cudaHostAllocMapped); cudaHostAlloc(&hv8, host_n, cudaHostAllocMapped);
        cudaHostAlloc(&hksc, (size_t)(cap - dev_cap) * kInkKVHeads * 4, cudaHostAllocMapped); cudaHostAlloc(&hvsc, (size_t)(cap - dev_cap) * kInkKVHeads * 4, cudaHostAllocMapped);
        const size_t dev_off = (size_t)dev_cap * kInkKVHeads * kInkHeadDim;
        cudaMemcpy(hk, kf16 + dev_off, host_n * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hv, vf16 + dev_off, host_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hk8, kf8 + dev_off, host_n, cudaMemcpyDeviceToHost); cudaMemcpy(hv8, vf8 + dev_off, host_n, cudaMemcpyDeviceToHost);
        cudaMemcpy(hksc, ksc + (size_t)dev_cap * kInkKVHeads, (size_t)(cap - dev_cap) * kInkKVHeads * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hvsc, vsc + (size_t)dev_cap * kInkKVHeads, (size_t)(cap - dev_cap) * kInkKVHeads * 4, cudaMemcpyDeviceToHost);
        void *dhk, *dhv, *dhk8, *dhv8, *dhksc, *dhvsc;
        cudaHostGetDevicePointer(&dhk, hk, 0); cudaHostGetDevicePointer(&dhv, hv, 0); cudaHostGetDevicePointer(&dhk8, hk8, 0); cudaHostGetDevicePointer(&dhv8, hv8, 0);
        cudaHostGetDevicePointer(&dhksc, hksc, 0); cudaHostGetDevicePointer(&dhvsc, hvsc, 0);
        for (int fmt : {0, 1}) {
            b.kv = InkKvVisitDesc{};
            b.kv.k_f16 = kf16; b.kv.v_f16 = vf16; b.kv.k_f8 = kf8; b.kv.v_f8 = vf8; b.kv.k_sc = ksc; b.kv.v_sc = vsc;
            b.kv.h_k_f16 = (__half*)dhk; b.kv.h_v_f16 = (__half*)dhv; b.kv.h_k_f8 = (uint8_t*)dhk8; b.kv.h_v_f8 = (uint8_t*)dhv8; b.kv.h_k_sc = (float*)dhksc; b.kv.h_v_sc = (float*)dhvsc;
            b.kv.cap_pos = cap; b.kv.dev_cap = dev_cap; b.kv.host_cap = cap - dev_cap; b.kv.is_swa = false; b.kv.kv_fmt = (uint8_t)fmt;
            const int tp0 = 130000;
            for (int M : {1, 4, 8, 12}) {
                Tree t = make_tree(M, tp0, 5);
                cudaMemcpy(b.pos, t.pos, M * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(b.anc, t.anc, 32 * 8, cudaMemcpyHostToDevice);
                cudaMemcpy(b.dep, t.dep, 32, cudaMemcpyHostToDevice);
                k_fill_f32<<<256, 256>>>(b.q, (size_t)M * kInkHeads * kInkHeadDim, 3, 3.0f, 0.0f);
                k_fill_f32<<<256, 256>>>(b.rel, (size_t)M * kInkHeads * (kInkRelExtent + 1), 4, 2.0f, 0.0f);
                k_fill_f32<<<4, 256>>>(b.sinks, (size_t)M * kInkHeads, 5, 4.0f, 0.0f);
                cudaDeviceSynchronize();
                const float t_old = time_kernel(false, 0, b, M, N_SP, M > 1, 3);
                std::printf("timing HOST-tail KV fmt=%d (%d cells in VRAM, rest mapped host), one global visit, tp0=%d M=%2d: old %.2f ms", fmt, dev_cap, tp0, M, t_old);
                if (M > 1) for (int rg : {1, 2, 4, 6, 8}) { const float ts = time_kernel(true, rg, b, M, N_SP, true, 3); std::printf(" | shared rg=%d %.2f ms (x%.2f)", rg, ts, t_old / ts); }
                std::printf("\n");
            }
        }
    }
    return bad ? 1 : 0;
}
