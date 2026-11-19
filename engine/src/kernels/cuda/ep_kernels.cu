// src/kernels/cuda/ep_kernels.cu - see include/strata/kernels/ep_kernels.hpp.  HIP port of the SYCL expert
// parallelism's card-to-card protocol (upstream PR #1798), through host-mapped memory because HIP IPC between the
// two 7900 XT fails ("invalid device context"): every region is one host allocation with a device alias per card;
// the flags are volatile loads/stores the engine's doorbell path already uses, and each writer fences at system
// scope before it raises the flag.
#include "strata/kernels/ep_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

/// *ctr += 1 (device memory of the card that owns it).
__global__ void ep_bump_kernel(uint32_t* ctr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *ctr += 1;
}

/// The ids and the q8_1 rows into the peer's inbox, then the flag: one block, every store fenced at system scope
/// before the single flag store so the peer's spin cannot see the flag ahead of the bytes.
__global__ void ep_push_kernel(const int32_t* __restrict__ ids, int n_ids, const uint32_t* __restrict__ xq,
                               size_t words, int32_t* __restrict__ peer_ids, uint32_t* __restrict__ peer_xq,
                               volatile uint32_t* peer_flag, const uint32_t* ctr) {
    const int lid = threadIdx.x;
    for (int i = lid; i < n_ids; i += blockDim.x) peer_ids[i] = ids[i];
    for (size_t i = lid; i < words; i += blockDim.x) peer_xq[i] = xq[i];
    __threadfence_system();
    __syncthreads();
    if (lid == 0) *peer_flag = *ctr;
}

/// Spin until the flag (this card's memory, written by the other card) reaches this card's epoch; bounded, with the
/// bound recorded in mapped host memory the host reads after the window.  The counters only grow and wrap: compare
/// the difference, not the values.
__global__ void ep_wait_kernel(const volatile uint32_t* flag, const uint32_t* ctr, uint32_t spin_max,
                               volatile uint32_t* err, uint32_t code) {
    const uint32_t want = *ctr;
    uint32_t spin = 0;
    while ((int32_t) (*flag - want) < 0) {
        if (++spin >= spin_max) {
            if (err != nullptr) *err = code;
            break;
        }
        strata_spin_pause();
    }
    __threadfence_system();
}

/// The peer's rows of the entries card 0 does not own (`res0[eid] < 0`) into the mapped region card 0 reads; the
/// last work-group to finish clears the counter and raises the return flag.  Each group fences at system scope
/// after its row, so the flag is never raised ahead of any row.
__global__ void ep_send_rows_kernel(float* __restrict__ peer_rows, const float* __restrict__ rows,
                                    const int32_t* __restrict__ ids, const int32_t* __restrict__ res0, int n,
                                    long long row, uint32_t* done, volatile uint32_t* peer_flag, const uint32_t* ctr) {
    const int r = blockIdx.x;
    const int32_t e = ids[r];
    if (e >= 0 && res0[e] < 0) {   // card 0 does not hold this entry: its row goes to card 0
        const float4* s = (const float4*) (rows + (size_t) r * (size_t) row);
        float4* d = (float4*) (peer_rows + (size_t) r * (size_t) row);
        for (long long i = threadIdx.x; i < row / 4; i += blockDim.x) d[i] = s[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prev = atomicAdd(done, 1u);
        if (prev == (uint32_t) n - 1) {
            *done = 0;   // every group has counted itself: the next call starts from zero
            __threadfence_system();
            *peer_flag = *ctr;
        }
    }
}

/// Card 0: the peer-written rows into parts_, read past this card's caches (volatile), at the entries card 0 does
/// not own.
__global__ void ep_copy_peer_rows_kernel(float4* __restrict__ dst, const volatile float4* __restrict__ src,
                                         const int32_t* __restrict__ ids, const int32_t* __restrict__ res0,
                                         int64_t row4) {
    const int r = blockIdx.x;
    const int32_t e = ids[r];
    if (!(e >= 0 && res0[e] < 0)) return;   // card 0's own entry: its row is already in parts_ (a bad id is a plan error)
    float4* d = dst + (int64_t) r * row4;
    const volatile float4* s = src + (int64_t) r * row4;
    for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = const_cast<const float4*>(s)[i];
}

__global__ void ep_copy_uncached_kernel(uint32_t* __restrict__ dst, const volatile uint32_t* __restrict__ src,
                                        size_t words) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

}  // namespace

void ep_bump(uint32_t* ctr, void* stream) {
    ep_bump_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ctr);
    check("ep_bump");
}

void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    if (n_ids <= 0 && xq_bytes == 0) return;
    if ((xq_bytes & 3) != 0 || ((uintptr_t) xq & 3) != 0 || ((uintptr_t) peer_xq & 3) != 0) {
        std::fprintf(stderr, "ep_push: xq_bytes must be a multiple of 4 and both pointers 4-byte aligned\n");
        std::exit(1);
    }
    ep_push_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(ids, n_ids, (const uint32_t*) xq, xq_bytes / 4,
                                                         peer_ids, (uint32_t*) peer_xq,
                                                         (volatile uint32_t*) peer_flag, ctr);
    check("ep_push");
}

void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream) {
    ep_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>((const volatile uint32_t*) flag, ctr, spin_max,
                                                       (volatile uint32_t*) err, code);
    check("ep_wait");
}

void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    if (n <= 0) return;
    if ((row & 3) != 0 || ((uintptr_t) peer_rows & 15) != 0 || ((uintptr_t) rows & 15) != 0) {
        std::fprintf(stderr, "ep_send_rows: row must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    ep_send_rows_kernel<<<(unsigned) n, 128, 0, (cudaStream_t) stream>>>(peer_rows, rows, ids, res0, n, row, done,
                                                                         (volatile uint32_t*) peer_flag, ctr);
    check("ep_send_rows");
}

void ep_copy_peer_rows(float* dst, const float* src, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                       void* stream) {
    if (n <= 0) return;
    if ((row & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "ep_copy_peer_rows: row must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    ep_copy_peer_rows_kernel<<<(unsigned) n, 128, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src,
                                                                              ids, res0, row / 4);
    check("ep_copy_peer_rows");
}

void ep_copy_uncached(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return;
    if ((bytes & 3) != 0 || ((uintptr_t) dst & 3) != 0 || ((uintptr_t) src & 3) != 0) {
        std::fprintf(stderr, "ep_copy_uncached: bytes must be a multiple of 4 and both pointers 4-byte aligned\n");
        std::exit(1);
    }
    const size_t words = bytes / 4;
    const size_t groups = (words + 255) / 256 < 64 ? (words + 255) / 256 : 64;
    ep_copy_uncached_kernel<<<(unsigned) groups, 256, 0, (cudaStream_t) stream>>>((uint32_t*) dst,
                                                                                  (const volatile uint32_t*) src, words);
    check("ep_copy_uncached");
}

}  // namespace strata::kernels
