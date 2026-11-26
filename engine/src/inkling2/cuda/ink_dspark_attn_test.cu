// engine/src/inkling2/cuda/ink_dspark_attn_test.cu - the DSpark drafter's block attention (k_dspark_block_attn: 8 block rows over the committed context + the 8 block keys,
// no bias, no sink) against the tensor-core kernel of ink_attn_tc.cuh in block mode (flags 7). Accuracy and time per call.
// Build: nvcc -O3 -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_120,code=sm_120 -I<cuda dir> -I<engine/include> ink_dspark_attn_test.cu -o t
#include "ink_dspark.cuh"
#include "ink_attn_tc.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::ink;

static __device__ __forceinline__ uint32_t mix32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((x ^ (x >> 31)) >> 16);
}
static __device__ __forceinline__ float rnd(uint64_t i, uint64_t seed) { return ((mix32(i * 2654435761ull + seed) & 0xFFFFFF) * (1.0f / 16777216.0f)) - 0.5f; }
__global__ void k_fill_h(__half* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half_rn(rnd(i, seed) * amp);
}
__global__ void k_fill_f(float* p, size_t n, uint64_t seed, float amp) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = rnd(i, seed) * amp;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    const int B = 8, cap = 131072 + 64;
    const size_t cells = (size_t)cap * kInkKVHeads * kInkHeadDim;
    __half *ck, *cv;
    float *q, *kd, *vd, *o0, *o1, *pm, *pl, *po;
    int32_t* apos;
    cudaMalloc(&ck, cells * 2); cudaMalloc(&cv, cells * 2);
    cudaMalloc(&q, (size_t)B * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&kd, (size_t)B * kInkKVHeads * kInkHeadDim * 4); cudaMalloc(&vd, (size_t)B * kInkKVHeads * kInkHeadDim * 4);
    cudaMalloc(&o0, (size_t)B * kInkHeads * kInkHeadDim * 4); cudaMalloc(&o1, (size_t)B * kInkHeads * kInkHeadDim * 4);
    const size_t np = (size_t)B * kInkHeads * tcattn::kSlots;
    cudaMalloc(&pm, np * 4); cudaMalloc(&pl, np * 4); cudaMalloc(&po, np * kInkHeadDim * 4);
    cudaMalloc(&apos, 4);
    k_fill_h<<<1024, 256>>>(ck, cells, 1, 4.0f);
    k_fill_h<<<1024, 256>>>(cv, cells, 2, 4.0f);
    k_fill_f<<<64, 256>>>(q, (size_t)B * kInkHeads * kInkHeadDim, 3, 6.0f);
    k_fill_f<<<16, 256>>>(kd, (size_t)B * kInkKVHeads * kInkHeadDim, 4, 4.0f);
    k_fill_f<<<16, 256>>>(vd, (size_t)B * kInkKVHeads * kInkHeadDim, 5, 4.0f);
    cudaDeviceSynchronize();
    InkKvVisitDesc kv{};
    kv.k_f16 = ck; kv.v_f16 = cv; kv.cap_pos = cap; kv.dev_cap = cap; kv.host_cap = 0; kv.is_swa = false; kv.kv_fmt = 0;
    int bad = 0;
    for (int nc : {300, 4099, 32768, 131072}) {
        cudaMemcpy(apos, &nc, 4, cudaMemcpyHostToDevice);
        auto old_path = [&]() { k_dspark_block_attn<<<dim3(kInkHeads, B), 128>>>(q, kd, vd, ck, cv, apos, false, 1025, cap, B, o0); };
        auto new_path = [&]() {
            tcattn::launch(0, q, nullptr, nullptr, kv, nullptr, 0, apos, kd, vd, nullptr, nullptr, 0, pm, pl, po, B, B,
                           tcattn::kFlagNoBias | tcattn::kFlagNoSink | tcattn::kFlagBlockTail);
            tcattn::merge(0, pm, pl, po, nullptr, o1, nullptr, apos, true, cap, B, B, tcattn::kFlagNoBias | tcattn::kFlagNoSink | tcattn::kFlagBlockTail);
        };
        old_path(); new_path();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("CUDA ERROR %s\n", cudaGetErrorString(e)); return 2; }
        const size_t n = (size_t)B * kInkHeads * kInkHeadDim;
        std::vector<float> a(n), b(n);
        cudaMemcpy(a.data(), o0, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), o1, n * 4, cudaMemcpyDeviceToHost);
        double d2 = 0, r2 = 0;
        bool fin = true;
        for (size_t i = 0; i < n; ++i) { if (!std::isfinite(b[i])) fin = false; const double d = (double)b[i] - a[i]; d2 += d * d; r2 += (double)a[i] * a[i]; }
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 3; ++i) { old_path(); new_path(); }
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < 20; ++i) old_path();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t_old = 0; cudaEventElapsedTime(&t_old, e0, e1); t_old /= 20;
        cudaEventRecord(e0);
        for (int i = 0; i < 20; ++i) new_path();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t_new = 0; cudaEventElapsedTime(&t_new, e0, e1); t_new /= 20;
        const double rl2 = std::sqrt(d2 / std::max(r2, 1e-30));
        const bool ok = fin && rl2 < 2e-3;
        std::printf("drafter block attn nc=%d: relL2 %.2e  old %.3f ms  new %.3f ms  x%.1f  %s\n", nc, rl2, t_old, t_new, t_old / t_new, ok ? "ok" : "FAIL");
        if (!ok) bad = 1;
    }
    std::printf("DSPARK-TEST-RC=%d\n", bad);
    return bad;
}
