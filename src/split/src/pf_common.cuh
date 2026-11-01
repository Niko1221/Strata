// What both models' prompt paths (prefill.cu, s27_prefill.cu) share: their error checks, the zeroed owned allocation
// every buffer comes from, the dequantizations and grid helper, and the copy-engine ring (pf_ring_plan.hpp's steps: a
// card's addend packed to fp16, added to what came in, the gathered sum read back), each setting the window's error
// bit 2 past fp16's range. The ring round itself is pf_ring_run, which both files' ring shapes run.
#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "kernels.hpp"
#include "owned.hpp"
#include "pf_ring_plan.hpp"

namespace fnx {

inline void ckc(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("prefill: ") + what + ": " + cudaGetErrorString(e));
}
inline void ckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("prefill: cublas ") + what + ": " + std::to_string((int) s));
}
inline void cud(CUresult e, const char* what) {
    if (e != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(e, &s);
        throw std::runtime_error(std::string("prefill: ") + what + ": " + (s ? s : "?"));
    }
}

// the prompt path's allocations on the card being set up: reset per card as it is built; Flash-Next's later stages
// save and restore it around their own, and the 27B allocates nothing after prefill_alloc
inline size_t pf_card_bytes = 0;
// a buffer of n T's, zeroed and owned. A failure names the card, what is free there and what the prompt path already
// holds: which of the two this allocation died between is the whole question when the pool is sized from free VRAM
template <typename T> T* palloc(size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, n * sizeof(T));
    if (e != cudaSuccess) {
        cudaGetLastError();
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        int dev = -1;
        cudaGetDevice(&dev);
        throw std::runtime_error("prefill: cudaMalloc of " + std::to_string(n * sizeof(T) >> 20) + " MiB on card " + std::to_string(dev) +
                                 ": " + cudaGetErrorString(e) + " (" + std::to_string(fr >> 20) + " MiB free; the prompt path holds " +
                                 std::to_string(pf_card_bytes >> 20) + " MiB there already)");
    }
    pf_card_bytes += n * sizeof(T);
    ckc(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    own_dev(p);
    return (T*) p;
}

// a grid of at most 4096 blocks over n values, 256 threads each
inline unsigned grid_for(size_t n) { return (unsigned) std::min<size_t>(4096, (n + 255) / 256); }

// the split Q8_0 layout (int8 rows, f16 scale per 32) to f16, d * q as ggml's dequantization to f16 rounds it
static __global__ void k_q8s_f16(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, size_t n, __half* __restrict__ out) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        out[e] = __float2half(__half2float(__ushort_as_half(d[e / 32])) * (float) q[e]);
}
static __global__ void k_f32_f16(const float* __restrict__ x, size_t n, __half* __restrict__ y) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) y[e] = __float2half(x[e]);
}

constexpr int PF_RING_STEPS = pf_ring_steps(NCARD);   // flags a set holds

// out = f16(x) over n values (n even)
static __global__ void k_pfr_pack(const float2* __restrict__ x, int n2, __half2* __restrict__ out, unsigned* err) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n2; e += gridDim.x * blockDim.x) {
        const float2 v = x[e];
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(err, 2u);   // past fp16's range (or NaN)
        out[e] = __floats2half2_rn(v.x, v.y);
    }
}
// out = f16(in + x) over n values (n even)
static __global__ void k_pfr_add(const __half2* __restrict__ in, const float2* __restrict__ x, int n2, __half2* __restrict__ out,
                                 unsigned* err) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n2; e += gridDim.x * blockDim.x) {
        const float2 a = __half22float2(in[e]), b = x[e];
        const float2 v = make_float2(a.x + b.x, a.y + b.y);
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(err, 2u);
        out[e] = __floats2half2_rn(v.x, v.y);
    }
}
static __global__ void k_pfr_out(const __half2* __restrict__ ag, float2* __restrict__ x, int n2) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n2; e += gridDim.x * blockDim.x) x[e] = __half22float2(ag[e]);
}

// one direction's sum of x (C rows of N) over the cards, on stream s; this card is at position g of the direction's
// ring (its next card: g + 1 there), by the steps of r.plan (pf_ring_plan.hpp, which describes the algorithm). Each
// copy into the next card is followed by a flag there; the step after waits on this card's flag (stream memory
// operations: no SM time, no host). Both files' ring shapes run it: Flash-Next's PfRing and the 27B's Pf, which holds
// the same buffers and takes its sequence number and copy-engine stream from itself rather than from the caller.
template <typename Ring>
inline void pf_ring_run(const Ring& r, int dev, int g, float* x, int C, unsigned seq, unsigned* err, cudaStream_t s) {
    if (r.plan.empty() || r.plan.front().part != g) throw std::runtime_error("prefill: the ring's plan is not card " + std::to_string(g) + "'s");
    const int set = (int) (seq & 1u);
    auto r0 = [&](int k) { return k * C / NCARD; };
    auto rows = [&](int k) { return r0(k + 1) - r0(k); };
    CUstream cs = (CUstream) s;
    auto signal = [&](int step) {
        cud(cuStreamWriteValue32(cs, (CUdeviceptr) (r.nflags + set * PF_RING_STEPS + step), seq, 0), "ring flag");
    };
    auto wait = [&](int step) {
        cud(cuStreamWaitValue32(cs, (CUdeviceptr) (r.flags + set * PF_RING_STEPS + step), seq, CU_STREAM_WAIT_VALUE_GEQ), "ring wait");
    };
    auto copy = [&](__half* dst, const __half* src, int nrows) {
        ckc(cudaMemcpyPeerAsync(dst, r.next_dev, src, dev, (size_t) nrows * N * sizeof(__half), s), "ring copy");
    };
    // a part can be empty (a chunk of fewer rows than cards: a read cut at a checkpoint): its kernels are skipped, its
    // copies move nothing, its flags still step
    auto add = [&](const __half* in, int part, __half* out) {
        if (rows(part) == 0) return;
        k_pfr_add<<<grid_for((size_t) rows(part) * N / 2), 256, 0, s>>>((const __half2*) in, (const float2*) (x + (size_t) r0(part) * N),
                                                                       rows(part) * N / 2, (__half2*) out, err);
    };
    for (const PfRingOp& o : r.plan) {
        switch (o.k) {
            case PfRingOp::PACK:
                if (rows(o.part) > 0)
                    k_pfr_pack<<<grid_for((size_t) rows(o.part) * N / 2), 256, 0, s>>>((const float2*) (x + (size_t) r0(o.part) * N),
                                                                                       rows(o.part) * N / 2, (__half2*) r.send, err);
                break;
            case PfRingOp::ADD: add(r.in[o.hop][set], o.part, o.fin ? r.ag[set] + (size_t) r0(o.part) * N : r.send); break;
            case PfRingOp::SEND: copy(r.nin[o.hop][set], r.send, rows(o.part)); break;
            case PfRingOp::PASS: copy(r.nag[set] + (size_t) r0(o.part) * N, r.ag[set] + (size_t) r0(o.part) * N, rows(o.part)); break;
            case PfRingOp::SIGNAL: signal(o.step); break;
            case PfRingOp::WAIT: wait(o.step); break;
            case PfRingOp::OUT:
                k_pfr_out<<<grid_for((size_t) C * N / 2), 256, 0, s>>>((const __half2*) r.ag[set], (float2*) x, C * N / 2);
                break;
            default: throw std::runtime_error("prefill: a ring step of unknown kind " + std::to_string((int) o.k));
        }
    }
}

}  // namespace fnx
