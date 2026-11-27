// src/kernels/cuda/loop_gate.cu - the recurrent-depth loop's pass-boundary gate (issue #7295, M5). New kernels,
// not a port - see include/strata/kernels/loop_gate.hpp for why, and for the two-step formula they compute.
//
// Deliberately plain: a serial per-block reduction in the down kernel (no warp tricks) and a per-thread serial
// loop over `gate_rank` in the up kernel (no reduction needed - `gate_rank` is small and stays in registers).
// `gr.cu`'s own comments are the record of what happens when a kernel here is tuned before anything measures
// it; this one is not tuned at all, on purpose, until a trained model exists to measure it against.
#include "strata/kernels/loop_gate.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

/// `g_out[c * gate_rank + k] = sum_d R[c, d] * down[k, d]`, one block per (c, k) pair, a plain shared-memory
/// tree-free reduction (every thread's partial strided-sums into one shared slot, thread 0 adds them up).
__global__ void loop_gate_down_kernel(const float* __restrict__ R, const uint16_t* __restrict__ down, int n_embd,
                                      int gate_rank, float* __restrict__ g_out) {
    __shared__ float part[THREADS];
    const int idx = blockIdx.x;              // idx = c * gate_rank + k
    const int k = idx % gate_rank;
    const int c = idx / gate_rank;
    const float* Rc = R + (size_t) c * n_embd;
    const uint16_t* row = down + (size_t) k * n_embd;   // (gate_rank, n_embd) row-major, n_embd contiguous
    float acc = 0.0f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) acc += Rc[d] * f32_from_bf16(row[d]);
    part[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < (int) blockDim.x; ++i) sum += part[i];
        g_out[idx] = sum;
    }
}

/// `z[c, d] = sigmoid(sum_k g[c, k] * up[d, k])`, then `R_out[c, d] = z*R[c,d] + (1-z)*R0[c,d]`. One thread per
/// (c, d) output element, a flat grid-stride loop over `hc * n_embd`.
__global__ void loop_gate_up_kernel(const float* __restrict__ R, const float* __restrict__ R0,
                                    const float* __restrict__ g_in, const uint16_t* __restrict__ up, int n_embd,
                                    int hc, int gate_rank, float* __restrict__ R_out) {
    const long long n = (long long) hc * n_embd;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int c = (int) (i / n_embd), d = (int) (i % n_embd);
        const float* gc = g_in + (size_t) c * gate_rank;
        const uint16_t* row = up + (size_t) d * gate_rank;   // (n_embd, gate_rank) row-major, gate_rank contiguous
        float acc = 0.0f;
        for (int k = 0; k < gate_rank; ++k) acc += gc[k] * f32_from_bf16(row[k]);
        const float z = 1.0f / (1.0f + expf(-acc));
        R_out[i] = z * R[i] + (1.0f - z) * R0[i];
    }
}

}  // namespace

size_t loop_gate_workspace_bytes(const LoopGateShapes& s) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.gate_rank <= 0) return 0;
    return (size_t) s.hc * (size_t) s.gate_rank * sizeof(float);
}

void loop_gate_mix(const float* R, const float* R0, const uint16_t* down, const uint16_t* up,
                   const LoopGateShapes& s, float* workspace, float* R_out, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.gate_rank <= 0) return;   // "no gate": the caller does not reach here
    cudaStream_t cs = (cudaStream_t) stream;
    const int blocks1 = (int) (s.hc * s.gate_rank);
    loop_gate_down_kernel<<<blocks1, THREADS, 0, cs>>>(R, down, (int) s.n_embd, (int) s.gate_rank, workspace);
    const long long n = (long long) s.hc * s.n_embd;
    const int blocks2 = (int) ((n + THREADS - 1) / THREADS);
    loop_gate_up_kernel<<<blocks2, THREADS, 0, cs>>>(R, R0, workspace, up, (int) s.n_embd, (int) s.hc,
                                                     (int) s.gate_rank, R_out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "loop_gate_mix launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) {
            std::fprintf(stderr, "loop_gate_mix: %s\n", cudaGetErrorString(se));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
