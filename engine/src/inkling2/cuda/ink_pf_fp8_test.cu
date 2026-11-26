// engine/src/inkling2/cuda/ink_pf_fp8_test.cu - bit-identity and timing of the FP8-KV tile loader of k_ink_flash_gqa_pf_sm120 (prefill attention).
//
// Build twice from the same source and compare the printed output hashes:
//   nvcc ... ink_pf_fp8_test.cu -o pf_new                       (the register-prefetch loader)
//   nvcc ... -DSTRATA_PF_FP8_LEGACY ink_pf_fp8_test.cu -o pf_old (the generic loader)
// Run: pf_new [--reps R]     Output: one line per case with an FNV-1a hash of the output bytes and the kernel time; equal hashes = bit-identical.
#include "ink_rtx5090_kernels.cuh"
#include "ink_attn.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::ink;

static __device__ __forceinline__ uint32_t mix32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((x ^ (x >> 31)) >> 16);
}
static __device__ __forceinline__ float rnd(uint64_t i, uint64_t seed) { return ((mix32(i * 2654435761ull + seed) & 0xFFFFFF) * (1.0f / 16777216.0f)) - 0.5f; }
__global__ void k_fill_f8(uint8_t* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const __nv_fp8_e4m3 v(rnd(i, seed) * amp); p[i] = v.__x; }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed, float amp, float off) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = rnd(i, seed) * amp + off;
}
__global__ void k_fill_h2(__half2* p, size_t n, uint64_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __floats2half2_rn(rnd(2 * i, seed), rnd(2 * i + 1, seed));
}

static uint64_t fnv(const void* d, size_t n) {
    const unsigned char* p = (const unsigned char*)d;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    int reps = 5;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
#ifdef STRATA_PF_FP8_LEGACY
    std::printf("device %s: LEGACY generic FP8 loader\n", prop.name);
#else
    std::printf("device %s: register-prefetch FP8 loader\n", prop.name);
#endif
    const int cap = 40000;
    const size_t cells = (size_t)cap * kInkKVHeads;
    InkKvVisitDesc kv{};
    cudaMalloc(&kv.k_f8, cells * kInkHeadDim); cudaMalloc(&kv.v_f8, cells * kInkHeadDim);
    cudaMalloc(&kv.k_sc, cells * 4); cudaMalloc(&kv.v_sc, cells * 4);
    k_fill_f8<<<1024, 256>>>(kv.k_f8, cells * kInkHeadDim, 1, 400.0f);
    k_fill_f8<<<1024, 256>>>(kv.v_f8, cells * kInkHeadDim, 2, 400.0f);
    k_fill_f32<<<256, 256>>>(kv.k_sc, cells, 3, 0.01f, 0.02f);
    k_fill_f32<<<256, 256>>>(kv.v_sc, cells, 4, 0.01f, 0.02f);
    const int MMAX = 4096;
    float *q, *r, *sinks, *gate, *out;
    __half2* rel;
    cudaMalloc(&q, (size_t)MMAX * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&r, (size_t)MMAX * kInkHeads * kInkDRel * 4);
    cudaMalloc(&sinks, (size_t)MMAX * kInkHeads * 4);
    cudaMalloc(&gate, (size_t)MMAX * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&out, (size_t)MMAX * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&rel, 1024 * 8 * sizeof(__half2));
    k_fill_f32<<<512, 256>>>(q, (size_t)MMAX * kInkHeads * kInkHeadDim, 5, 3.0f, 0.0f);
    k_fill_f32<<<256, 256>>>(r, (size_t)MMAX * kInkHeads * kInkDRel, 6, 1.0f, 0.0f);
    k_fill_f32<<<16, 256>>>(sinks, (size_t)MMAX * kInkHeads, 7, 4.0f, 0.0f);
    k_fill_f32<<<512, 256>>>(gate, (size_t)MMAX * kInkHeads * kInkHeadDim, 8, 2.0f, 0.0f);
    k_fill_h2<<<8, 256>>>(rel, 1024 * 8, 9);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { std::printf("setup failed\n"); return 2; }

    struct Case { const char* name; bool swa; int p0; int M; int ring; };
    const Case cases[] = {{"global p0=0 M=1024", false, 0, 1024, 0},      {"global p0=3000 M=2048", false, 3000, 2048, 0},
                          {"global p0=30000 M=4096", false, 30000, 4096, 0}, {"global p0=777 M=129", false, 777, 129, 0},
                          {"swa p0=5000 M=1024", true, 5000, 1024, 2048},  {"swa p0=0 M=300", true, 0, 300, 2048}};
    int bad = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (const Case& c : cases) {
        InkKvVisitDesc d = kv;
        d.kv_fmt = 1; d.is_swa = c.swa; d.dev_cap = c.swa ? c.ring : cap; d.host_cap = 0; d.cap_pos = c.swa ? c.ring : cap;
        const dim3 grid(kInkHeads, (c.M + 63) / 64);
        auto go = [&]() {
            if (c.swa) rtx5090::k_ink_flash_gqa_pf_sm120<32, false><<<dim3(kInkHeads, (c.M + 63) / 64), 128>>>(q, r, rel, sinks, gate, d, 0, c.p0, c.M, out, nullptr);
            else       rtx5090::k_ink_flash_gqa_pf_sm120<64, false><<<grid, 128>>>(q, r, rel, sinks, gate, d, 0, c.p0, c.M, out, nullptr);
        };
        cudaMemset(out, 0, (size_t)MMAX * kInkHeads * kInkHeadDim * 4);
        go();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("%-26s CUDA ERROR %s\n", c.name, cudaGetErrorString(e)); bad = 1; break; }
        std::vector<float> h((size_t)c.M * kInkHeads * kInkHeadDim);
        cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
        bool finite = true;
        for (float x : h) if (!(x == x) || x > 1e30f || x < -1e30f) { finite = false; break; }
        cudaEventRecord(e0);
        for (int i = 0; i < reps; ++i) go();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= reps;
        std::printf("%-26s hash %016llx  %.3f ms  %s\n", c.name, (unsigned long long)fnv(h.data(), h.size() * 4), ms, finite ? "finite" : "NOT FINITE");
        if (!finite) bad = 1;
    }
    std::printf("PF-TEST-RC=%d\n", bad);
    return bad;
}
