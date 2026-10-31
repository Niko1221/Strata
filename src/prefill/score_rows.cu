// src/prefill/score_rows.cu - see include/strata/prefill/score_rows.hpp.
#include "strata/prefill/score_rows.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace strata::prefill {
namespace {

constexpr int kThreads = 256;          // two warps' worth of rows' worth of lanes; one block owns one row
constexpr int kWarps = kThreads / 32;

// Fold `(om, os)` into `(m, s)`: the two partial log-sum-exps describe different halves of the row, so the merged
// max is the larger and each sum is rescaled to it.  Both being -inf means neither half saw a finite logit; the
// early return keeps exp(-inf - -inf) = NaN out of the result.
__device__ __forceinline__ void lse_merge(float& m, float& s, float om, float os) {
    const float nm = fmaxf(m, om);
    if (nm == -INFINITY) {
        m = nm;
        s = 0.f;
        return;
    }
    s = s * expf(m - nm) + os * expf(om - nm);
    m = nm;
}

// One block per row.  The row is walked ONCE: every element updates the online pair, so the vocabulary is read a
// single time (248320 floats, 994 KB) instead of twice for max-then-sum.  expf, not __expf: a row sums 248320
// terms and the training gradient is a difference of two such sums, so the ~2 ulp fast intrinsic is not worth the
// few microseconds it saves.
__global__ void score_row_kernel(const float* __restrict__ logits, int vocab, int64_t stride,
                                 const int32_t* __restrict__ targets, float* __restrict__ out) {
    const int row = blockIdx.x;
    const float* r = logits + (size_t) row * (size_t) stride;
    float m = -INFINITY, s = 0.f;
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
        const float v = r[i];
        if (v > m) {
            s = s * expf(m - v) + 1.f;
            m = v;
        } else {
            s += expf(v - m);
        }
    }
    __shared__ float wm[kWarps], ws[kWarps];
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_down_sync(0xffffffffu, m, off);
        const float os = __shfl_down_sync(0xffffffffu, s, off);
        lse_merge(m, s, om, os);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        wm[warp] = m;
        ws[warp] = s;
    }
    __syncthreads();
    if (warp != 0) return;
    m = lane < kWarps ? wm[lane] : -INFINITY;
    s = lane < kWarps ? ws[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
        const float om = __shfl_down_sync(0xffffffffu, m, off);
        const float os = __shfl_down_sync(0xffffffffu, s, off);
        lse_merge(m, s, om, os);
    }
    if (lane != 0) return;
    const int tgt = targets[row];
    // A position with no target (the last of a sequence, or a masked one) has nothing to score.  NaN, not 0: a
    // caller that forgets to skip it must see it, not average a plausible number into a loss.
    if (tgt < 0 || tgt >= vocab) {
        out[row] = NAN;
        return;
    }
    out[row] = r[tgt] - (m + logf(s));
}

__global__ void f32_to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half_as_ushort(__float2half_rn(in[i]));
}

}  // namespace

size_t score_rows_scratch_bytes(int rows) {
    if (rows < 1) return 0;
    return (size_t) rows * 2 * sizeof(float);
}

bool score_rows(const float* logits, int rows, int vocab, int64_t stride, const int32_t* targets, void* scratch,
                float* out, void* stream) {
    if (logits == nullptr || targets == nullptr || scratch == nullptr || out == nullptr) return false;
    if (rows < 1 || vocab < 1 || stride < vocab) return false;
    // The scratch is only here so the kernel's shape can grow without a new allocation at every call site; this
    // implementation keeps the pair in registers, so a caller may pass any buffer of the documented size.
    (void) scratch;
    score_row_kernel<<<rows, kThreads, 0, (cudaStream_t) stream>>>(logits, vocab, stride, targets, out);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool f32_to_f16(const float* in, uint16_t* out, int64_t n, void* stream) {
    if (in == nullptr || out == nullptr || n < 0) return false;
    if (n == 0) return true;
    const int64_t blocks = (n + kThreads - 1) / kThreads;
    f32_to_f16_kernel<<<(unsigned) blocks, kThreads, 0, (cudaStream_t) stream>>>(in, out, n);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::prefill
