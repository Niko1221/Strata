// engine/src/inkling2/cuda/ink_pf_acc_test.cu - the output sum of k_ink_flash_gqa_pf_sm120 (prefill attention) over many keys of equal score.
//
// The kernel sums P @ V over the K/V tiles before it divides by the sum of P. With every key at the same score the sum of P is the number of
// keys, so the output sum reaches N x |V|: a sum held in FP16 passes the largest half (65504) at N x |V| > 65504 and the row comes out as inf
// or NaN (a 128K prompt of similar records did that from about 55K tokens on, and the engine ended with a GPU error further on); below that it
// loses the tiles that are smaller than half a unit of the sum. This test asks for the mean of V, which is V itself when V is the same for
// every key: K = 0 (every score 0), no relative term, a sink far below, no gate. It prints the worst relative error per case.
//
//   nvcc -O3 -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_120,code=sm_120 -I. -I../../../include -I../../hy4flash/run ink_pf_acc_test.cu -o pf_acc
// Run: pf_acc     Exit code 0 when every case is within 1%.
#include "ink_rtx5090_kernels.cuh"
#include "ink_attn.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::ink;

__global__ void k_fill_half(__half* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half(v);
}
// V differs by dimension and sign (the mean over the keys is still V of that dimension), so a wrong column or a lost sign shows too
__global__ void k_fill_v(__half* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(i % kInkHeadDim);
        p[i] = __float2half(((d & 1) ? -v : v) * (1.0f + 0.25f * (float)(d % 4)));
    }
}
__global__ void k_fill_f32c(float* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device %s\n", prop.name);
    const int cap = 131072;
    const float v0 = 1.0f;
    const size_t cells = (size_t)cap * kInkKVHeads * kInkHeadDim;
    InkKvVisitDesc kv{};
    if (cudaMalloc(&kv.k_f16, cells * 2) != cudaSuccess || cudaMalloc(&kv.v_f16, cells * 2) != cudaSuccess) { std::printf("no memory for the K/V\n"); return 2; }
    k_fill_half<<<1024, 256>>>(kv.k_f16, cells, 0.0f);
    k_fill_v<<<1024, 256>>>(kv.v_f16, cells, v0);
    const int M = 128;
    float *q, *r, *sinks, *out;
    __half2* rel;
    cudaMalloc(&q, (size_t)M * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&r, (size_t)M * kInkHeads * kInkDRel * 4);
    cudaMalloc(&sinks, (size_t)M * kInkHeads * 4);
    cudaMalloc(&out, (size_t)M * kInkHeads * kInkHeadDim * 4);
    cudaMalloc(&rel, 1024 * 8 * sizeof(__half2));
    k_fill_f32c<<<64, 256>>>(q, (size_t)M * kInkHeads * kInkHeadDim, 0.5f);
    k_fill_f32c<<<64, 256>>>(r, (size_t)M * kInkHeads * kInkDRel, 0.0f);
    k_fill_f32c<<<16, 256>>>(sinks, (size_t)M * kInkHeads, -40.0f);
    cudaMemset(rel, 0, 1024 * 8 * sizeof(__half2));
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { std::printf("setup failed\n"); return 2; }

    kv.kv_fmt = 0; kv.is_swa = false; kv.dev_cap = cap; kv.host_cap = 0; kv.cap_pos = cap;
    const int ends[] = {1024, 4096, 8192, 16384, 32768, 50000, 60000, 100000, 131072};   // the number of keys the last row sees
    int bad = 0;
    for (int n : ends) {
        const int p0 = n - M;
        cudaMemset(out, 0, (size_t)M * kInkHeads * kInkHeadDim * 4);
        rtx5090::k_ink_flash_gqa_pf_sm120<64, false><<<dim3(kInkHeads, (M + 63) / 64), 128>>>(q, r, rel, sinks, nullptr, kv, 0, p0, M, out, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("keys %6d  CUDA ERROR %s\n", n, cudaGetErrorString(e)); return 2; }
        std::vector<float> h((size_t)M * kInkHeads * kInkHeadDim);
        cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
        double worst = 0.0;
        size_t not_finite = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            const int d = (int)(i % kInkHeadDim);
            const double want = ((d & 1) ? -v0 : v0) * (1.0 + 0.25 * (d % 4));
            if (!std::isfinite(h[i])) { ++not_finite; continue; }
            worst = std::max(worst, std::fabs(h[i] - want) / std::fabs(want));
        }
        const bool ok = (not_finite == 0 && worst <= 0.01);
        std::printf("keys %6d  worst relative error %8.4f%%  not finite %zu of %zu  %s\n", n, worst * 100.0, not_finite, h.size(), ok ? "ok" : "WRONG");
        if (!ok) bad = 1;
    }
    std::printf("PF-ACC-TEST-RC=%d\n", bad);
    return bad;
}
