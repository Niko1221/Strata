// src/ds4/ds4_pcie.cuh - the decode path's PCIe lane on the device (see ds4_pcie.hpp): the GPU pulls the experts the
// host read into pinned memory into its staging slots itself (as GLM's moe_fetch_kernel does) - the host only reads and
// writes flags, never calls the GPU's API while the device waits on it (a thread that enqueues the layers can be
// blocked in the driver until the device moves on, and with it every other thread's call).  Included by ds4_model.cu
// alone (its one translation unit), and the tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "ds4_launch.h"   // strata_ds4_launch

namespace strata::ds4::pcie {

/// A request's pull (host-mapped, a ring of them by seq): `seq` once src is set; src[i] the device address of pick i's
/// expert in pinned memory (0: not streamed), laid out as its blob (gate, up, down at the layer's offsets);
/// ready[i][r] == seq once its matrix r is read (bad[i]: a read failed - the pick is left out, its weight lost); done ==
/// seq once the device pulled every one (the host times the stream lane by it).
struct Pull {
    unsigned seq;
    unsigned done;
    unsigned pad[2];
    unsigned long long src[8];
    unsigned ready[8][4];
    unsigned bad[8];
};
// the fetch kernel's device flags: the pull, each pick's matrices, then the blocks done (the last one sets Pull::done)
constexpr int kPullCnt = 1 + 8 * 3, kPullFlags = kPullCnt + 1;

/// A layer's matrices in its blob: offsets and bytes (each a multiple of 16 when rounded up: the slots' matrices start
/// 256-aligned, the copy may run into the padding after one)
struct Mats {
    size_t off[3], bytes[3];
};

__device__ __forceinline__ void pull_nap() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(500);
#endif
}

// grid: up to one block a multiprocessor.  Block 0's thread 0 polls the host's flags (the others wait on device copies
// of them in dflag [kPullFlags] - so the link is not flooded with polls); each streamed pick's matrices, each once read,
// copied from pinned memory into its slot (vram + i stride; copy 0: the kernels read them where they lie - an
// integrated GPU); then pick i's slot into plan2[i] (the second gate / up's plan: the streamed picks alone) and plan[i]
// (the down pass and the sum take a streamed expert as a VRAM one); the last block to finish sets the pull's done
__global__ void __launch_bounds__(256) stream_fetch_kernel(Pull* p, unsigned seq, unsigned* dflag, uint8_t* vram,
                                                           size_t stride, Mats m, int copy, unsigned long long* plan,
                                                           unsigned long long* plan2, int k) {
    __shared__ unsigned long long s_src[8];
    __shared__ unsigned s_bad[8];
    const int tid = threadIdx.x;
    volatile const Pull* vp = (volatile const Pull*) p;
    volatile unsigned* df = dflag;
    if (tid == 0) {
        if (blockIdx.x == 0) {
            while (vp->seq != seq) pull_nap();
            __threadfence_system();
            df[0] = seq;
            __threadfence();
        } else {
            while (df[0] != seq) pull_nap();
            __threadfence();
        }
    }
    __syncthreads();
    if (tid < 8) {
        s_src[tid] = tid < k ? vp->src[tid] : 0ull;
        s_bad[tid] = 0u;
    }
    __syncthreads();
    for (int i = 0; i < k; ++i) {
        if (s_src[i] == 0ull) continue;
        for (int r = 0; r < 3; ++r) {
            const int f = 1 + 3 * i + r;
            if (tid == 0) {
                if (blockIdx.x == 0) {
                    while (vp->ready[i][r] != seq) pull_nap();
                    __threadfence_system();
                    df[f] = vp->bad[i] != 0u ? ~seq : seq;   // (a failed read: ~seq - no other request's seq soon)
                    __threadfence();
                } else {
                    while (df[f] != seq && df[f] != ~seq) pull_nap();
                    __threadfence();
                }
                if (df[f] != seq) s_bad[i] = 1u;
            }
            __syncthreads();
            if (s_bad[i] || !copy) continue;
            const uint4* src = (const uint4*) (s_src[i] + m.off[r]);
            uint4* dst = (uint4*) (vram + (size_t) i * stride + m.off[r]);
            const size_t n16 = (m.bytes[r] + 15) / 16;
            for (size_t j = (size_t) blockIdx.x * blockDim.x + tid; j < n16; j += (size_t) gridDim.x * blockDim.x)
                dst[j] = src[j];
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&dflag[kPullCnt], 1u) == gridDim.x - 1) {
            dflag[kPullCnt] = 0u;
            ((volatile Pull*) p)->done = seq;
            __threadfence_system();
        }
    }
    if (blockIdx.x == 0 && tid < k) {
        const unsigned long long d = s_src[tid] == 0ull || s_bad[tid] ? 0ull
                                     : copy                          ? (unsigned long long) (vram + (size_t) tid * stride)
                                                                     : s_src[tid];
        plan2[tid] = d;
        if (d != 0ull) plan[tid] = d;
    }
}

inline void stream_fetch(Pull* p, unsigned seq, unsigned* dflag, uint8_t* vram, size_t stride, const Mats& m,
                         bool copy, unsigned long long* plan, unsigned long long* plan2, int k, int blocks,
                         cudaStream_t s) {
    strata_ds4_launch(stream_fetch_kernel, blocks, 256, 0, s, p, seq, dflag, vram, stride, m, copy ? 1 : 0, plan, plan2, k);
}

}  // namespace strata::ds4::pcie
