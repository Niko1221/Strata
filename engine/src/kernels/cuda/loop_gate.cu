// src/kernels/cuda/loop_gate.cu - the recurrent-depth loop's per-layer message + gate (issue #7295). A
// port of Inkling2Loop's attn_res_mix_f32/loop_control_f32 - see include/strata/kernels/loop_gate.hpp for
// the exact math and the two adaptations (2 channels, plain-vector bank).
#include "strata/kernels/loop_gate.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr float RMS_EPS = 1e-6f;

/// `attn_res_mix_f32`, ported directly: softmax attention over `bank_n + 1` states (the bank, then
/// `bank_out`), RMS-normalizing each state before the score AND the weighted sum. The normalization is
/// folded algebraically rather than materialized: `dot(norm(x), q) == dot(x, q) * rsqrt(mean(x^2) + eps)`,
/// so a state's sum-of-squares and its dot with the query are the only things read from it. One thread per
/// query/channel (2 threads total): `bank_n` is small (the loop range is QSA-only, so for Strata's default
/// qsa_interval it is almost always 1) and a serial `O(n_states * n_embd)` scan is simple and obviously
/// correct - see the file header on why this stays unoptimized for now.
__global__ void loop_msg_attn_kernel(const float* __restrict__ bank, const float* __restrict__ bank_out,
                                     const float* __restrict__ query, int n_embd, int bank_n,
                                     float* __restrict__ msgs) {
    const int q = threadIdx.x;
    if (q >= 2) return;
    const float* qv = query + (size_t) q * n_embd;
    const int n_states = bank_n + 1;
    auto state_ptr = [&](int s) { return (s < bank_n) ? (bank + (size_t) s * n_embd) : bank_out; };
    const float inv_sqrt_d = 1.0f / sqrtf((float) n_embd);

    // pass 1: the max score, for a numerically stable softmax (online max, no storage of n_states scores)
    float max_score = -1e30f;
    for (int s = 0; s < n_states; ++s) {
        const float* sv = state_ptr(s);
        float dot = 0.0f, ss = 0.0f;
        for (int d = 0; d < n_embd; ++d) { dot += sv[d] * qv[d]; ss += sv[d] * sv[d]; }
        const float rs = rsqrtf(ss / (float) n_embd + RMS_EPS);
        const float score = dot * rs * inv_sqrt_d;
        if (score > max_score) max_score = score;
    }
    // pass 2: the softmax weights and the weighted sum of NORMALIZED states, recomputing dot/sum-of-
    // squares rather than caching n_states of them - the same "simple over fast" choice as pass 1.
    float* out = msgs + (size_t) q * n_embd;
    for (int d = 0; d < n_embd; ++d) out[d] = 0.0f;
    float sum_w = 0.0f;
    for (int s = 0; s < n_states; ++s) {
        const float* sv = state_ptr(s);
        float dot = 0.0f, ss = 0.0f;
        for (int d = 0; d < n_embd; ++d) { dot += sv[d] * qv[d]; ss += sv[d] * sv[d]; }
        const float rs = rsqrtf(ss / (float) n_embd + RMS_EPS);
        const float score = dot * rs * inv_sqrt_d;
        const float w = expf(score - max_score);
        sum_w += w;
        for (int d = 0; d < n_embd; ++d) out[d] += w * sv[d] * rs;
    }
    const float inv = 1.0f / sum_w;
    for (int d = 0; d < n_embd; ++d) out[d] *= inv;
}

/// `cat = rmsnorm(cur) ++ rmsnorm(msg)`, written into `workspace[0 .. 2*n_embd)`. One block, `2*n_embd`
/// covered by a grid-stride loop over threads - small relative to the down-projection below, so this is
/// not split further.
__global__ void loop_ctrl_cat_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                     int n_embd, float* __restrict__ cat) {
    __shared__ float part[THREADS];
    // cur's sum of squares
    float acc = 0.0f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) acc += cur[d] * cur[d];
    part[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) blockDim.x; ++i) s += part[i];
        part[0] = rsqrtf(s / (float) n_embd + RMS_EPS);
    }
    __syncthreads();
    const float rs_cur = part[0];
    __syncthreads();
    // msg's sum of squares (reuses `part`, guarded by the barrier above having already read it)
    acc = 0.0f;
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) acc += msg[d] * msg[d];
    part[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) blockDim.x; ++i) s += part[i];
        part[0] = rsqrtf(s / (float) n_embd + RMS_EPS);
    }
    __syncthreads();
    const float rs_msg = part[0];
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        cat[d] = cur[d] * rs_cur;
        cat[n_embd + d] = msg[d] * rs_msg;
    }
}

/// `gate[r] = sum_i down[r, i] * cat[i]` over `i` in `0..2*n_embd`, one block per output `r` - the same
/// plain block-wide reduction `loop_gate_down_kernel` used in the predecessor of this file.
__global__ void loop_ctrl_down_kernel(const float* __restrict__ cat, const uint16_t* __restrict__ down,
                                      int n_embd2, float* __restrict__ gate) {
    __shared__ float part[THREADS];
    const int r = blockIdx.x;
    const uint16_t* row = down + (size_t) r * n_embd2;   // (gate_rank, 2*n_embd) row-major, 2*n_embd contiguous
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_embd2; i += blockDim.x) acc += cat[i] * f32_from_bf16(row[i]);
    part[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) blockDim.x; ++i) s += part[i];
        gate[r] = s;
    }
}

/// `g = tanh(gain @ gate)`, `f = tanh(forget @ gate)`, `out[d] = (1-f)*cur[d] + g*msg[d]` - one thread per
/// output element, a serial loop over `gate_rank` (small, no reduction needed).
__global__ void loop_ctrl_updown_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                        const float* __restrict__ gate, const uint16_t* __restrict__ gain,
                                        const uint16_t* __restrict__ forget, int n_embd, int gate_rank,
                                        float* __restrict__ out) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < n_embd; d += gridDim.x * blockDim.x) {
        const uint16_t* grow = gain + (size_t) d * gate_rank;      // (n_embd, gate_rank) row-major
        const uint16_t* frow = forget + (size_t) d * gate_rank;
        float ga = 0.0f, fa = 0.0f;
        for (int r = 0; r < gate_rank; ++r) {
            ga += gate[r] * f32_from_bf16(grow[r]);
            fa += gate[r] * f32_from_bf16(frow[r]);
        }
        const float g = tanhf(ga), f = tanhf(fa);
        out[d] = (1.0f - f) * cur[d] + g * msg[d];
    }
}

}  // namespace

void loop_msg_attn(const float* bank, const float* bank_out, const float* query, const LoopMsgShapes& s,
                   float* msgs, void* stream) {
    if (s.n_embd <= 0) return;
    loop_msg_attn_kernel<<<1, 2, 0, (cudaStream_t) stream>>>(bank, bank_out, query, (int) s.n_embd,
                                                             (int) s.bank_n, msgs);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "loop_msg_attn launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) { std::fprintf(stderr, "loop_msg_attn: %s\n", cudaGetErrorString(se)); std::exit(1); }
    }
}

size_t loop_ctrl_workspace_bytes(const LoopCtrlShapes& s) {
    if (s.n_embd <= 0 || s.gate_rank <= 0) return 0;
    return (size_t) (2 * s.n_embd + s.gate_rank) * sizeof(float);
}

void loop_ctrl_gate(const float* cur, const float* msg, const uint16_t* down, const uint16_t* gain,
                    const uint16_t* forget, const LoopCtrlShapes& s, float* workspace, float* out,
                    void* stream) {
    if (s.n_embd <= 0 || s.gate_rank <= 0) return;   // "no gate": the caller does not reach here
    cudaStream_t cs = (cudaStream_t) stream;
    float* cat = workspace;
    float* gate = workspace + 2 * s.n_embd;
    loop_ctrl_cat_kernel<<<1, THREADS, 0, cs>>>(cur, msg, (int) s.n_embd, cat);
    loop_ctrl_down_kernel<<<(int) s.gate_rank, THREADS, 0, cs>>>(cat, down, (int) (2 * s.n_embd), gate);
    const int blocks = (int) ((s.n_embd + THREADS - 1) / THREADS);
    loop_ctrl_updown_kernel<<<blocks, THREADS, 0, cs>>>(cur, msg, gate, gain, forget, (int) s.n_embd,
                                                        (int) s.gate_rank, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "loop_ctrl_gate launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) { std::fprintf(stderr, "loop_ctrl_gate: %s\n", cudaGetErrorString(se)); std::exit(1); }
    }
}

}  // namespace strata::kernels
