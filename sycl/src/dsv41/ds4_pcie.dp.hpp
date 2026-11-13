// src/ds4/ds4_pcie.cuh - the decode path's PCIe lane on the device (see ds4_pcie.hpp): the GPU pulls the experts the
// host read into pinned memory into its staging slots itself (as GLM's moe_fetch_kernel does) - the host only reads and
// writes flags, never calls the GPU's API while the device waits on it (a thread that enqueues the layers can be
// blocked in the driver until the device moves on, and with it every other thread's call).  Included by ds4_model.cu
// alone (its one translation unit), and the tests.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstddef>
#include <cstdint>
#if defined(STRATA_METAL)
#include <vector_types.h>   // uint4
#endif

#include "ds4_launch.h"
#include <time.h>
   // strata_ds4_launch

namespace strata::ds4::pcie {

/// A request's pull (host-mapped, a ring of them by seq): `seq` once src is set; src[i] the device address of pick i's
/// expert in pinned memory (0: not streamed), laid out as its blob (gate, up, down at the layer's offsets);
/// ready[i][r] == seq once its matrix r is read (bad[i]: a read failed - the pick is left out, its weight lost); done ==
/// seq once the device pulled every one (the host times the stream lane by it).
constexpr int kPullEnt = 16 * 8;   // a window's entries (kWin tokens x 8 picks)
constexpr int kSlots = 16;          // a pull's streamed experts at most (a window's; a token's: its picks)
struct Pull {
    unsigned seq;
    unsigned done;
    unsigned pad[2];
    unsigned long long src[kSlots];
    unsigned ready[kSlots][4];
    unsigned bad[kSlots];
    unsigned long long mat[kSlots][3];   // (0: src + the matrix's offset) pick i's matrix r where it lies - pulled in place
    signed char ent[kPullEnt];      // (a verify window) entry t * 8 + i's slot when its expert streams, else -1
    unsigned long long t_beg, t_end;   // the device's clock (ns) when it saw the pull, when its last block was done
    unsigned long long dst[kSlots];    // (0: its staging slot) where pick i lands - a recent slot it then stays in
};
// the fetch kernel's device flags: the pull, each pick's matrices, then the blocks done (the last one sets Pull::done)
constexpr int kPullCnt = 1 + kSlots * 3, kPullFlags = kPullCnt + 1;

/// A layer's matrices in its blob: offsets and bytes (each a multiple of 16 when rounded up: the slots' matrices start
/// 256-aligned, the copy may run into the padding after one)
struct Mats {
    size_t off[3], bytes[3];
};

// the device's clock in ns (NVIDIA's globaltimer; HIP's wall clock, 100 MHz on gfx11/gfx12); 0 on other AMD GPUs,
// whose wall clock may tick at another rate - the host then times the pull itself
inline unsigned long long pull_clock() {
    // Use the existing host-timed path on a backend without NVIDIA globaltimer.
    return 0ull;
}

__dpct_inline__ void pull_nap() {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 700
    /*
    DPCT1008:496: __nanosleep function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    /* CUDA pause hint; polling order is unchanged. */
#endif
}

// grid: up to one block a multiprocessor.  Block 0's thread 0 polls the host's flags (the others wait on device copies
// of them in dflag [kPullFlags] - so the link is not flooded with polls); each streamed pick's matrices, each once read,
// copied from pinned memory into its slot (vram + i stride; copy 0: the kernels read them where they lie - an
// integrated GPU); then pick i's slot into plan2[i] (the second gate / up's plan: the streamed picks alone) and plan[i]
// (the down pass and the sum take a streamed expert as a VRAM one); the last block to finish sets the pull's done
__dpct_inline__ void stream_fetch_kernel(Pull *p, unsigned seq, unsigned *dflag,
                                         uint8_t *vram, size_t stride, Mats m,
                                         int copy, unsigned long long *plan,
                                         unsigned long long *plan2, int k,
                                         unsigned long long *wplan, int went) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_src = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    unsigned long long[kSlots]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_mat = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[kSlots][3]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_dst = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[kSlots]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned[kSlots]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = item_ct1.get_local_id(2);
    volatile const Pull* vp = (volatile const Pull*) p;
    volatile unsigned* df = dflag;
    if (tid == 0) {
        if (item_ct1.get_group(2) == 0) {
            while (vp->seq != seq) pull_nap();
            /*
            DPCT1078:219: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::system);
            ((volatile Pull*) p)->t_beg = pull_clock();
            df[0] = seq;
            /*
            DPCT1078:220: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
        } else {
            while (df[0] != seq) pull_nap();
            /*
            DPCT1078:221: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
        }
    }
    /*
    DPCT1065:497: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid < kSlots) {
        s_src[tid] = tid < k ? vp->src[tid] : 0ull;
#pragma unroll
        for (int r = 0; r < 3; ++r) s_mat[tid][r] = tid < k ? vp->mat[tid][r]
                                                            : 0ull;
        s_dst[tid] = tid < k && vp->dst[tid] != 0ull ? vp->dst[tid] : (unsigned long long) (vram + (size_t) tid * stride);
        s_bad[tid] = 0u;
    }
    /*
    DPCT1065:498: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = 0; i < k; ++i) {
        if (s_src[i] == 0ull) continue;
        for (int r = 0; r < 3; ++r) {
            const int f = 1 + 3 * i + r;
            if (tid == 0) {
                if (item_ct1.get_group(2) == 0) {
                    while (vp->ready[i][r] != seq) pull_nap();
                    /*
                    DPCT1078:223: Consider replacing memory_order::acq_rel with
                    memory_order::seq_cst for correctness if strong memory order
                    restrictions are needed.
                    */
                    sycl::atomic_fence(sycl::memory_order::acq_rel,
                                       sycl::memory_scope::system);
                    df[f] = vp->bad[i] != 0u ? ~seq : seq;   // (a failed read: ~seq - no other request's seq soon)
                    /*
                    DPCT1078:224: Consider replacing memory_order::acq_rel with
                    memory_order::seq_cst for correctness if strong memory order
                    restrictions are needed.
                    */
                    sycl::atomic_fence(sycl::memory_order::acq_rel,
                                       sycl::memory_scope::device);
                } else {
                    while (df[f] != seq && df[f] != ~seq) pull_nap();
                    /*
                    DPCT1078:225: Consider replacing memory_order::acq_rel with
                    memory_order::seq_cst for correctness if strong memory order
                    restrictions are needed.
                    */
                    sycl::atomic_fence(sycl::memory_order::acq_rel,
                                       sycl::memory_scope::device);
                }
                if (df[f] != seq) s_bad[i] = 1u;
            }
            /*
            DPCT1118:222: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065:500: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if (s_bad[i] || !copy) continue;
            const sycl::uint4 *src =
                (const sycl::uint4 *)(s_mat[i][r] != 0ull
                                          ? s_mat[i][r]
                                          : s_src[i] + m.off[r]);
            sycl::uint4 *dst = (sycl::uint4 *)(s_dst[i] + m.off[r]);
            const size_t n16 = (m.bytes[r] + 15) / 16;
#pragma unroll
            for (size_t j = (size_t)item_ct1.get_group(2) *
                                item_ct1.get_local_range(2) +
                            tid;
                 j < n16; j += (size_t)item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))
                dst[j] = src[j];
        }
    }
    /*
    DPCT1065:499: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid == 0) {
        /*
        DPCT1078:226: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        if (dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dflag[kPullCnt], 1u) == item_ct1.get_group_range(2) - 1) {
            dflag[kPullCnt] = 0u;
            ((volatile Pull*) p)->t_end = pull_clock();
            /*
            DPCT1078:227: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::system);
            ((volatile Pull*) p)->done = seq;
            /*
            DPCT1078:228: Consider replacing memory_order::acq_rel with
            memory_order::seq_cst for correctness if strong memory order
            restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::system);
        }
    }
    if (item_ct1.get_group(2) == 0 && tid < k && wplan == nullptr) {
        const unsigned long long d = s_src[tid] == 0ull || s_bad[tid] ? 0ull : copy ? s_dst[tid] : s_src[tid];
        plan2[tid] = d;
        if (d != 0ull) plan[tid] = d;
    }
    // (a verify window) each entry whose expert streamed: its slot into the window's plan - the grouping that follows
    // takes it as a VRAM expert (a failed read: the entry stays the CPU lane's, which left it out - the token goes
    // without it, as a token does)
    if (item_ct1.get_group(2) == 0 && wplan != nullptr)
        for (int e = tid; e < went; e += item_ct1.get_local_range(2)) {
            const int i = vp->ent[e];
            if (i < 0 || i >= k || s_src[i] == 0ull || s_bad[i]) continue;
            wplan[e] = copy ? s_dst[i] : s_src[i];
        }
}

inline void stream_fetch(Pull *p, unsigned seq, unsigned *dflag, uint8_t *vram,
                         size_t stride, const Mats &m, bool copy,
                         unsigned long long *plan, unsigned long long *plan2,
                         int k, int blocks, dpct::queue_ptr s) {
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            stream_fetch_kernel(p, seq, dflag, vram, stride, m, copy ? 1 : 0,
                                plan, plan2, k, nullptr, 0);
        });
}
// (a verify window of T tokens) the streamed experts pulled as above, then their entries' slots into its plan
inline void stream_fetch_win(Pull *p, unsigned seq, unsigned *dflag,
                             uint8_t *vram, size_t stride, const Mats &m,
                             bool copy, unsigned long long *wplan, int T,
                             int blocks, dpct::queue_ptr s) {
    s->submit([&](sycl::handler &cgh) {
        auto kSlots_ct9 = kSlots;
        auto T_ct11 = T * 8;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                               sycl::range(1, 1, 256),
                                           sycl::range(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             stream_fetch_kernel(p, seq, dflag, vram, stride, m,
                                                 copy ? 1 : 0, nullptr, nullptr,
                                                 kSlots_ct9, wplan, T_ct11);
                         });
    });
}

}  // namespace strata::ds4::pcie
