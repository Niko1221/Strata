// What both models' prompt paths (prefill.cu, s27_prefill.cu) share: their error checks, the zeroed owned allocation
// every buffer comes from, the dequantizations and grid helper, and the copy-engine ring (pf_ring_plan.hpp's steps: a
// card's addend packed to fp16, added to what came in, the gathered sum read back), each setting the window's error
// bit 2 past fp16's range. The ring round itself is pf_ring_run, which both files' ring shapes run.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels.hpp"
#include "owned.hpp"
#include "pf_ring_plan.hpp"
#include <cmath>

#ifndef FNX_ZERO_DEV
#define FNX_ZERO_DEV(p, bytes) DPCT_CHECK_ERROR(sycl_port::zero_device(&dpct::get_in_order_queue(), (p), (bytes)).wait_and_throw())
#endif

namespace fnx {

inline void ckc(dpct::err0 e, const char *what) {
    sycl_port::check_errors();
    
    
    
    if (e != 0) throw std::runtime_error(std::string("prefill: ") + what +
                                         ": " +
                                         dpct::get_error_string_dummy(e));
}
inline void ckb(int s, const char *what) {
    if (s != 0) throw std::runtime_error(std::string("prefill: cublas ") +
                                         what + ": " + std::to_string((int)s));
}
inline void cud(int e, const char *what) {
    sycl_port::check_errors();
    
    if (e != 0) {
        const char* s = nullptr;
        
        
        s = dpct::get_error_string_dummy(e);
        throw std::runtime_error(std::string("prefill: ") + what + ": " + (s ? s : "?"));
    }
}

// the prompt path's allocations on the card being set up: reset per card as it is built; Flash-Next's later stages
// save and restore it around their own, and the 27B allocates nothing after prefill_alloc
inline size_t pf_card_bytes = 0;
// a buffer of n T's, zeroed and owned. A failure names the card, what is free there and what the prompt path already
// holds: which of the two this allocation died between is the whole question when the pool is sized from free VRAM
template <typename T> T *palloc(size_t n) try {
    void* p = nullptr;
    const dpct::err0 e =
        DPCT_CHECK_ERROR(p = (void *)sycl_port::alloc_device(
                             n * sizeof(T), dpct::get_in_order_queue()));
    
    if (e != 0) {
        
        
        size_t fr = 0, tot = 0;
        
        dpct::get_current_device().get_memory_info(fr, tot);
        int dev = -1;
        dev = dpct::get_current_device_id();
        throw std::runtime_error(
            "prefill: cudaMalloc of " + std::to_string(n * sizeof(T) >> 20) +
            " MiB on card " + std::to_string(dev) +
            
            ": " +
            dpct::get_error_string_dummy(e) + " (" + std::to_string(fr >> 20) +
            " MiB free; the prompt path holds " +
            std::to_string(pf_card_bytes >> 20) + " MiB there already)");
    }
    pf_card_bytes += n * sizeof(T);
    ckc(FNX_ZERO_DEV(p, n * sizeof(T)), "SYCL memset");
    own_dev(p);
    return (T*) p;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// FNX_HOSTRING: n zeroed values of page-locked, mapped, portable host memory both cards reach (the prompt path's exchange
// buffers and flags), released by ~Engine
template <typename T> T *hpalloc(size_t n) try {
    void* p = nullptr;
    
    
    const dpct::err0 e =
        DPCT_CHECK_ERROR(p = (void *)sycl_port::alloc_host(
                             n * sizeof(T), dpct::get_in_order_queue()));
    
    
    
    if (e != 0) throw std::runtime_error(
        "prefill: cudaHostAlloc of " + std::to_string(n * sizeof(T) >> 20) +
        " MiB: " + dpct::get_error_string_dummy(e));
    std::memset(p, 0, n * sizeof(T));
    own_pinned(p);
    return (T*) p;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#if FNX_HIP
// FNX_PF_PEER: n zeroed values of fine-grained device memory on the current card (the other cards' kernels store into it; hip/peer.h enabled the access), released by ~Engine
template <typename T> T* ppalloc(size_t n) {
    void* p = nullptr;
    if (hipExtMallocWithFlags(&p, n * sizeof(T), hipDeviceMallocFinegrained) != hipSuccess)
        throw std::runtime_error("prefill: fine-grained device allocation of " + std::to_string(n * sizeof(T) >> 10) + " KiB failed");
    pf_card_bytes += n * sizeof(T);
    ckc(FNX_ZERO_DEV(p, n * sizeof(T)), "peer buffer clear");
    own_dev(p);
    return (T*) p;
}
#endif

// a grid of at most 4096 blocks over n values, 256 threads each
inline unsigned grid_for(size_t n) { return (unsigned) std::min<size_t>(4096, (n + 255) / 256); }

// the split Q8_0 layout (int8 rows, f16 scale per 32) to f16, d * q as ggml's dequantization to f16 rounds it
static void k_q8s_f16(const int8_t *__restrict__ q,
                      const uint16_t *__restrict__ d, size_t n,
                      sycl::half *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (size_t e =
             item_ct1.get_group(2) * (size_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         e < n;
         e += (size_t)item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        out[e] = sycl::vec<float, 1>(
                     sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(d[e / 32]))
                         .convert<float, sycl::rounding_mode::automatic>()[0] *
                     (float)q[e])
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}
static void k_f32_f16(const float *__restrict__ x, size_t n,
                      sycl::half *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (size_t e =
             item_ct1.get_group(2) * (size_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         e < n;
         e += (size_t)item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        y[e] = sycl::vec<float, 1>(x[e])
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

constexpr int PF_RING_STEPS = pf_ring_steps(NCARD);   // flags a set holds

// out = f16(x) over n values (n even)
static void k_pfr_pack(const sycl::float2 *__restrict__ x, int n2,
                       sycl::half2 *__restrict__ out, unsigned *err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         e < n2;
         e += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        const sycl::float2 v = x[e];
        if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                err, 2u); // past fp16's range (or NaN)
        out[e] = sycl::float2(v.x(), v.y())
                     .convert<sycl::half, sycl::rounding_mode::rte>();
    }
}
// out = f16(in + x) over n values (n even)
static void k_pfr_add(const sycl::half2 *__restrict__ in,
                      const sycl::float2 *__restrict__ x, int n2,
                      sycl::half2 *__restrict__ out, unsigned *err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         e < n2;
         e += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        const sycl::float2
            a = in[e].template convert<float, sycl::rounding_mode::automatic>(),
            b = x[e];
        const sycl::float2 v = sycl::float2(a.x() + b.x(), a.y() + b.y());
        if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                err, 2u);
        out[e] = sycl::float2(v.x(), v.y())
                     .convert<sycl::half, sycl::rounding_mode::rte>();
    }
}
// FNX_HOSTRING: a finished part from this card's gather buffer to the next card's, both in host memory, copied by a kernel. A host-to-host
// cudaMemcpyAsync blocks the calling thread until the stream has caught up, and the prompt path enqueues the cards one after the other on
// one thread: card 0's copy waits for a flag only card 1's (not yet enqueued) work sets, so the first read stalled. A kernel does not block.
static void k_pfr_copy(const sycl::uint4 *__restrict__ src,
                       sycl::uint4 *__restrict__ dst, size_t n16,
                       int sysfence = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (size_t e =
             item_ct1.get_group(2) * (size_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         e < n16; e += (size_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2)) dst[e] = src[e];
#if FNX_HIP
    if (sysfence) __threadfence_system();   // (the destination is the next card's host memory)
#endif
}
// the direct gather's finishing add: f16(in + x) stored into every card's gathered rows (up to NCARD destinations, all host memory)
struct PfGatherDst {
    sycl::half2 *p[NCARD];
    int sysfence;   // DMA readers (FNX_PF_DMA=1) need the stores visible system-wide at once; kernel readers never did, and the fence flushes the L2
};
static void k_pfr_add_gather(const sycl::half2 *__restrict__ in,
                             const sycl::float2 *__restrict__ x, int n2,
                             PfGatherDst dst, unsigned *err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         e < n2;
         e += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        const sycl::float2
            a = in[e].template convert<float, sycl::rounding_mode::automatic>(),
            b = x[e];
        const sycl::float2 v = sycl::float2(a.x() + b.x(), a.y() + b.y());
        if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                err, 2u);
        const sycl::half2 h =
            sycl::float2(v.x(), v.y())
                .convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
        for (int d = 0; d < NCARD; ++d) dst.p[d][e] = h;
    }
#if FNX_HIP
    // the stores into the other cards' host buffers must be visible system-wide before the stream's flag write tells them so (a copy engine reads
    // that memory at once; a kernel read came late enough to hide it)
    if (dst.sysfence) __threadfence_system();
#endif
}
#if FNX_HIP
// FNX_PF_PEER, small chunks (rows <= p.a2a_rows): the sum in ONE cross-card latency instead of the ring's NCARD hops of copy + flag + wait. Every card stores its f32 addends of the
// chunk into slot [its number] of every other card's mailbox (device memory of the receiver, fine-grained), the last block to finish its stores raises the flag [its number] on
// every other card; then every card waits for its NCARD - 1 flags and makes the whole sum itself, row by row in the ring's own order and roundings: part k of the rows
// (rows [k C / n, (k + 1) C / n)) starts as f16(addend of card k), then card k + 1, k + 2, ... add their f32 addend to the f16 value and round to f16 again, so the result is
// bit for bit what the ring (PACK, ADD, GADD, OUT) produces. A flag is the sum's number (a wait passes on >=); one flag word per sender is enough: a sender cannot be a
// sum ahead of a set this card still reads (its next write to the same set follows this card's next sum).
static __global__ void k_pfa_push(const float4* __restrict__ x, int n4, float4* const* __restrict__ peer_box, unsigned* const* __restrict__ peer_flag, unsigned* __restrict__ counter,
                                  int g, unsigned seq, size_t set_off4, size_t slot4) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n4; e += gridDim.x * blockDim.x) {
        const float4 v = x[e];
#pragma unroll
        for (int d = 0; d < NCARD; ++d)
            if (d != g) peer_box[d][set_off4 + (size_t) g * slot4 + e] = v;
    }
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence_system();
        *counter = 0u;
#pragma unroll
        for (int d = 0; d < NCARD; ++d)
            if (d != g) *(volatile unsigned*) (peer_flag[d] + g) = seq;
        __threadfence_system();
    }
}
static __global__ void k_pfa_sum(float* __restrict__ x, int C, const float4* __restrict__ box, const unsigned* __restrict__ flag, int g, unsigned seq, size_t set_off4,
                                 size_t slot4, unsigned* err) {
    if (threadIdx.x < NCARD && (int) threadIdx.x != g) {
        const long long t0 = clock64();
        while ((int) (*(volatile const unsigned*) (flag + threadIdx.x) - seq) < 0) {
            __builtin_amdgcn_s_sleep(1);
            if (clock64() - t0 > 4000000000LL) { atomicOr(err, 1u); break; }
        }
    }
    __syncthreads();
    __threadfence();
    const int n4 = C * N / 4;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n4; e += gridDim.x * blockDim.x) {
        const int row = (e * 4) / N;
        int k = 0;
        while (k + 1 < NCARD && row >= (k + 1) * C / NCARD) ++k;   // the part this row is in
        float4 a[NCARD];
#pragma unroll
        for (int c = 0; c < NCARD; ++c) {
            if (c == g) a[c] = reinterpret_cast<const float4*>(x)[e];
            else {
                const volatile float* src = reinterpret_cast<const volatile float*>(box + set_off4 + (size_t) c * slot4 + e);
                a[c] = make_float4(src[0], src[1], src[2], src[3]);
            }
        }
        float r[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int s = 0; s < NCARD; ++s) {
                const int c = (k + s) % NCARD;
                const float v = (j == 0 ? a[c].x : j == 1 ? a[c].y : j == 2 ? a[c].z : a[c].w);
                const float t = s == 0 ? v : acc + v;
                if (!(fabsf(t) <= RING_MAX)) atomicOr(err, 2u);
                acc = __half2float(__float2half(t));
            }
            r[j] = acc;
        }
        reinterpret_cast<float4*>(x)[e] = make_float4(r[0], r[1], r[2], r[3]);
    }
}
#endif
// FNX_PF_DMA_CHECK=1: the device copy a DMA made against the host buffer it read, compared by a kernel right after the copy; a block that finds a
// difference prints once (the DMA saw older data than a later load: the peer's stores were not visible to the copy engine yet)
static void k_pfr_cmp(const sycl::uint4 *__restrict__ dev,
                      const sycl::uint4 *__restrict__ host, size_t n16,
                      int card, int what, int rows0, unsigned *err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned bad = 0;
    for (size_t e =
             item_ct1.get_group(2) * (size_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         e < n16; e += (size_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2)) {
        const sycl::uint4 a = dev[e];
        const unsigned* h = reinterpret_cast<const unsigned*>(host + e);   // uncached loads: what the memory holds now
#if FNX_HIP
        const unsigned b0 = __hip_atomic_load(h + 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM), b1 = __hip_atomic_load(h + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM),
                       b2 = __hip_atomic_load(h + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM), b3 = __hip_atomic_load(h + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        
        const unsigned b0 = *(h + 0), b1 = *(h + 1), b2 = *(h + 2),
                       b3 = *(h + 3); // NVIDIA: uncached loads (ld.global.cv)
#endif
        if (a.x() != b0 || a.y() != b1 || a.z() != b2 || a.w() != b3)++ bad;
    }
    if (bad) dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
        err,
        what ? 128u : 64u); // "prefill: ring error 64" (receive) / 128 (gather)
    (void) card; (void) what; (void) rows0;
}
static void k_pfr_out(const sycl::half2 *__restrict__ ag,
                      sycl::float2 *__restrict__ x, int n2) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         e < n2; e += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        x[e] = ag[e].template convert<float, sycl::rounding_mode::automatic>();
}
// RCCL's fp16 sum back to x as f32, with the range check the host exchange's adds make (a sum past fp16's range, or NaN: error bit 2)
static void k_pfr_out_chk(const sycl::half2 *__restrict__ ag,
                          sycl::float2 *__restrict__ x, int n2, unsigned *err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         e < n2;
         e += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        const sycl::float2 v =
            ag[e].template convert<float, sycl::rounding_mode::automatic>();
        if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                err, 2u);
        x[e] = v;
    }
}

// one direction's sum of x (C rows of N) over the cards, on stream s; this card is at position g of the direction's
// ring (its next card: g + 1 there), by the steps of r.plan (pf_ring_plan.hpp, which describes the algorithm). Each
// copy into the next card is followed by a flag there; the step after waits on this card's flag (stream memory
// operations: no SM time, no host). Both files' ring shapes run it: Flash-Next's PfRing and the 27B's Pf, which holds
// the same buffers and takes its sequence number and copy-engine stream from itself rather than from the caller.
//
// own_out (FNX_PF_HCSPLIT): the sum's consumer works on this card's own part of the rows alone (the part it finishes, (g + 1) % n), so the
// gather half of the ring is skipped: no PASS, no gather flags, and OUT converts only the finished part into x. The part's values are the
// ones the full OUT would give.
template <typename Ring>
inline void pf_ring_run(const Ring &r, int dev, int g, float *x, int C,
                        unsigned seq, unsigned *err, dpct::queue_ptr s,
                        bool own_out = false,
                        sycl_port::EventMailbox* events = nullptr,
                        sycl_port::HostTrace* trace = nullptr) try {
    sycl_port::HostScope scope(trace, own_out ? "ring.own" : "ring.full");
    if (r.plan.empty() || r.plan.front().part != g) throw std::runtime_error("prefill: the ring's plan is not card " + std::to_string(g) + "'s");
    const int set = (int) (seq & 1u), fin_part = (g + 1) % NCARD;   // the part this card finishes (pf_ring_plan)
    sycl::half *const own_ag =
        r.dma ? r.dag[set] : r.ag[set]; // where this card's gathered rows are
                                        // (device with DMA, host without)
    auto r0 = [&](int k) { return k * C / NCARD; };
    auto rows = [&](int k) { return r0(k + 1) - r0(k); };
    dpct::queue_ptr cs = (dpct::queue_ptr)s;
    auto signal = [&](int step) {
        if (events) {
            events->signal(cs, r.nflags + set * PF_RING_STEPS + step, seq, trace);
            return;
        }
        
        cud(cuStreamWriteValue32(
                cs, (dpct::device_ptr)(r.nflags + set * PF_RING_STEPS + step),
                seq, 0),
            "ring flag");
    };
    auto wait = [&](int step) {
        if (events) {
            events->wait(cs, r.flags + set * PF_RING_STEPS + step, seq, trace);
            return;
        }
        
        cud(cuStreamWaitValue32(
                cs, (dpct::device_ptr)(r.flags + set * PF_RING_STEPS + step),
                seq, CU_STREAM_WAIT_VALUE_GEQ),
            "ring wait");
    };
    auto copy = [&](sycl::half *dst, const sycl::half *src, int nrows) {
        if (r.send_kernel) {   // FNX_PF_SEND=kernel: the sender's device -> host copy by a kernel (its stores are done when the kernel is)
            if (nrows > 0)
                s->submit([&](sycl::handler &cgh) {
                    auto size_t_nrows_N_ct2 = (size_t)nrows * N / 8;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1,
                                           grid_for((size_t)nrows * N / 8)) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_pfr_copy((const sycl::uint4 *)src,
                                       (sycl::uint4 *)dst, size_t_nrows_N_ct2,
                                       0);
                        });
                });
            return;
        }
        
        ckc(DPCT_CHECK_ERROR(
                s->memcpy(dst, src, (size_t)nrows * N * sizeof(sycl::half))),
            "ring copy"); // device -> host buffer
    };
    // FNX_PF_DMA_BARRIER=1 (HIP): after a host -> device copy, an event recorded and waited on in the same stream, so no later kernel of the stream starts
    // before the copy engine is done (a copy engine's completion is not otherwise ordered against the compute queue's next kernel on every runtime)
    auto dma_fence = [&] {
        if (r.fence_ev) {
            
            ckc(DPCT_CHECK_ERROR(dpct::sync_barrier(r.fence_ev, s)),
                "dma fence");
            ckc(DPCT_CHECK_ERROR(sycl_port::stream_barrier(s, {*r.fence_ev})),
                "dma fence");
        }
    };
    // a part can be empty (a chunk of fewer rows than cards: a read cut at a checkpoint): its kernels are skipped, its
    // copies move nothing, its flags still step
    auto add = [&](const sycl::half *in, int part, sycl::half *out) {
        if (rows(part) == 0) return;
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto x_size_t_r0_part_N_ct1 =
                    (const sycl::float2 *)(x + (size_t)r0(part) * N);
                auto rows_part_N_ct2 = rows(part) * N / 2;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1,
                                       grid_for((size_t)rows(part) * N / 2)) *
                            sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_pfr_add((const sycl::half2 *)in,
                                  x_size_t_r0_part_N_ct1, rows_part_N_ct2,
                                  (sycl::half2 *)out, err);
                    });
            });
        }
    };
    for (const PfRingOp& o : r.plan) {
        if (own_out) {   // only the reduce-scatter half and this card's finished part
            if (o.k == PfRingOp::PASS || o.k == PfRingOp::SIGNALA || o.k == PfRingOp::WAITA) continue;
            if ((o.k == PfRingOp::SIGNAL || o.k == PfRingOp::WAIT) && o.step >= NCARD - 1) continue;
            if (o.k == PfRingOp::GADD) {   // the finishing add into this card's rows alone
                if (rows(o.part) > 0) {
                    if (r.dma) {
                        ckc(DPCT_CHECK_ERROR(s->memcpy(
                                r.din[o.hop][set], r.in[o.hop][set],
                                (size_t)rows(o.part) * N * sizeof(sycl::half))),
                            "ring in");
                        dma_fence();
                    }
                    add(r.dma ? r.din[o.hop][set] : r.in[o.hop][set], o.part, own_ag + (size_t) r0(o.part) * N);
                }
                continue;
            }
            if (o.k == PfRingOp::OUT) {
                if (rows(fin_part) > 0)
                {
                    dpct::has_capability_or_fail(s->get_device(),
                                                 {sycl::aspect::fp16});

                    s->submit([&](sycl::handler &cgh) {
                        auto own_ag_size_t_r0_fin_part_N_ct0 =
                            (const sycl::half2 *)(own_ag +
                                                  (size_t)r0(fin_part) * N);
                        auto x_size_t_r0_fin_part_N_ct1 =
                            (sycl::float2 *)(x + (size_t)r0(fin_part) * N);
                        auto rows_fin_part_N_ct2 = rows(fin_part) * N / 2;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(
                                    1, 1,
                                    grid_for((size_t)rows(fin_part) * N / 2)) *
                                    sycl::range<3>(1, 1, 256),
                                sycl::range<3>(1, 1, 256)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_pfr_out(own_ag_size_t_r0_fin_part_N_ct0,
                                          x_size_t_r0_fin_part_N_ct1,
                                          rows_fin_part_N_ct2);
                            });
                    });
                }
                continue;
            }
        }
        switch (o.k) {
            case PfRingOp::PACK:
                if (rows(o.part) > 0)
            {
                dpct::has_capability_or_fail(s->get_device(),
                                             {sycl::aspect::fp16});

                s->submit([&](sycl::handler &cgh) {
                    auto x_size_t_r0_o_part_N_ct0 =
                        (const sycl::float2 *)(x + (size_t)r0(o.part) * N);
                    auto rows_o_part_N_ct1 = rows(o.part) * N / 2;
                    auto r_send_ct2 = (sycl::half2 *)r.send;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(
                                1, 1, grid_for((size_t)rows(o.part) * N / 2)) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_pfr_pack(x_size_t_r0_o_part_N_ct0,
                                       rows_o_part_N_ct1, r_send_ct2, err);
                        });
                });
            }
                break;
            case PfRingOp::ADD:
                // the slot that arrived in host memory goes to device memory on the copy engine (asynchronous, beside the next chunk's
                // compute), and the add reads it there: a kernel reading host memory stalls its SMs at PCIe speed (FNX_PF_DMA=0: it reads the host slot)
                if (rows(o.part) > 0 && r.dma) {
                    ckc(DPCT_CHECK_ERROR(s->memcpy(
                            r.din[o.hop][set], r.in[o.hop][set],
                            (size_t)rows(o.part) * N * sizeof(sycl::half))),
                        "ring in");
                    dma_fence();
                    if (r.dma_check)
                    s->submit([&](sycl::handler &cgh) {
                        auto r_din_o_hop_set_ct0 =
                            (const sycl::uint4 *)r.din[o.hop][set];
                        auto r_in_o_hop_set_ct1 =
                            (const sycl::uint4 *)r.in[o.hop][set];
                        auto size_t_rows_o_part_N_ct2 =
                            (size_t)rows(o.part) * N / 8;
                        auto rows_o_part_ct5 = rows(o.part);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(
                                    1, 1,
                                    grid_for((size_t)rows(o.part) * N / 8)) *
                                    sycl::range<3>(1, 1, 256),
                                sycl::range<3>(1, 1, 256)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_pfr_cmp(r_din_o_hop_set_ct0,
                                          r_in_o_hop_set_ct1,
                                          size_t_rows_o_part_N_ct2, g, 0,
                                          rows_o_part_ct5, err);
                            });
                    });
                }
                add(r.dma ? r.din[o.hop][set] : r.in[o.hop][set], o.part, o.fin ? own_ag + (size_t) r0(o.part) * N : r.send);
                break;
            case PfRingOp::SEND: copy(r.nin[o.hop][set], r.send, rows(o.part)); break;
            case PfRingOp::PASS:
                if (rows(o.part) > 0) {
                    const size_t off = (size_t) r0(o.part) * N;
                    if (o.part == fin_part && r.dma)   // the part finished here: device -> the next card's host gather buffer, by DMA
                        
                        ckc(DPCT_CHECK_ERROR(s->memcpy(
                                r.nag[set] + off, r.dag[set] + off,
                                (size_t)rows(o.part) * N * sizeof(sycl::half))),
                            "ring gather");
                    else   // a part that arrived (three or more cards), or no DMA: host -> host by a kernel (a host-to-host memcpy would block the thread)
                    s->submit([&](sycl::handler &cgh) {
                        auto r_ag_set_off_ct0 =
                            (const sycl::uint4 *)(r.ag[set] + off);
                        auto r_nag_set_off_ct1 =
                            (sycl::uint4 *)(r.nag[set] + off);
                        auto size_t_rows_o_part_N_ct2 =
                            (size_t)rows(o.part) * N / 8;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(
                                    1, 1,
                                    grid_for((size_t)rows(o.part) * N / 8)) *
                                    sycl::range<3>(1, 1, 256),
                                sycl::range<3>(1, 1, 256)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_pfr_copy(r_ag_set_off_ct0, r_nag_set_off_ct1,
                                           size_t_rows_o_part_N_ct2, 0);
                            });
                    });
                }
                break;
            case PfRingOp::GADD:
                if (rows(o.part) > 0) {
                    // the received slot to device memory by DMA (as ADD), the sum stored into this card's device rows and every other card's
                    // host rows (stores to host memory are posted, they do not stall the SMs as loads do)
                    if (r.dma) {
                        
                        ckc(DPCT_CHECK_ERROR(s->memcpy(
                                r.din[o.hop][set], r.in[o.hop][set],
                                (size_t)rows(o.part) * N * sizeof(sycl::half))),
                            "ring in");
                        dma_fence();
                        if (r.dma_check)
                        s->submit([&](sycl::handler &cgh) {
                            auto r_din_o_hop_set_ct0 =
                                (const sycl::uint4 *)r.din[o.hop][set];
                            auto r_in_o_hop_set_ct1 =
                                (const sycl::uint4 *)r.in[o.hop][set];
                            auto size_t_rows_o_part_N_ct2 =
                                (size_t)rows(o.part) * N / 8;
                            auto rows_o_part_ct5 = rows(o.part);

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(
                                        1, 1,
                                        grid_for((size_t)rows(o.part) * N /
                                                 8)) *
                                        sycl::range<3>(1, 1, 256),
                                    sycl::range<3>(1, 1, 256)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    k_pfr_cmp(r_din_o_hop_set_ct0,
                                              r_in_o_hop_set_ct1,
                                              size_t_rows_o_part_N_ct2, g, 0,
                                              rows_o_part_ct5, err);
                                });
                        });
                    }
                    PfGatherDst dst;
                    dst.sysfence = r.dma ? 1 : 0;
                    for (int d = 0; d < NCARD; ++d)
                        dst.p[d] =
                            (sycl::half2 *)((d == g ? own_ag : r.pag[d][set]) +
                                            (size_t)r0(o.part) * N);
                {
                    dpct::has_capability_or_fail(s->get_device(),
                                                 {sycl::aspect::fp16});

                    s->submit([&](sycl::handler &cgh) {
                        auto r_dma_r_din_o_hop_set_r_in_o_hop_set_ct0 =
                            (const sycl::half2 *)(r.dma ? r.din[o.hop][set]
                                                        : r.in[o.hop][set]);
                        auto x_size_t_r0_o_part_N_ct1 =
                            (const sycl::float2 *)(x + (size_t)r0(o.part) * N);
                        auto rows_o_part_N_ct2 = rows(o.part) * N / 2;

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(
                                    1, 1,
                                    grid_for((size_t)rows(o.part) * N / 2)) *
                                    sycl::range<3>(1, 1, 256),
                                sycl::range<3>(1, 1, 256)),
                            [=](sycl::nd_item<3> item_ct1) {
                                k_pfr_add_gather(
                                    r_dma_r_din_o_hop_set_r_in_o_hop_set_ct0,
                                    x_size_t_r0_o_part_N_ct1, rows_o_part_N_ct2,
                                    dst, err);
                            });
                    });
                }
                }
                break;
            case PfRingOp::SIGNALA:
                for (int d = 0; d < NCARD; ++d)
                    if (d != g) {
                        if (events)
                            events->signal(cs, r.pflags[d] + set * PF_RING_STEPS +
                                NCARD - 1 + pf_gather_slot(NCARD, g, d), seq);
                        else cud(cuStreamWriteValue32(
                                cs,
                                (dpct::device_ptr)(r.pflags[d] +
                                                   set * PF_RING_STEPS +
                                                   (NCARD - 1 +
                                                    pf_gather_slot(NCARD, g,
                                                                   d))),
                                seq, 0),
                            "gather flag");
                    }
                break;
            case PfRingOp::WAITA:
                for (int j = 0; j < NCARD - 1; ++j) {
                    if (events)
                        events->wait(cs, r.flags + set * PF_RING_STEPS + NCARD - 1 + j, seq);
                    else cud(cuStreamWaitValue32(
                            cs,
                            (dpct::device_ptr)(r.flags + set * PF_RING_STEPS +
                                               NCARD - 1 + j),
                            seq, CU_STREAM_WAIT_VALUE_GEQ),
                        "gather wait");
                }
                break;
            case PfRingOp::SIGNAL: signal(o.step); break;
            case PfRingOp::WAIT: wait(o.step); break;
            case PfRingOp::OUT:
                // the parts that arrived (everything but the one finished here) go to device memory by DMA, then one kernel reads it all
                if (r.dma)
                    for (int p = 0; p < NCARD; ++p)
                        if (p != fin_part && rows(p) > 0)
                        {
                            
                            ckc(DPCT_CHECK_ERROR(s->memcpy(
                                    r.dag[set] + (size_t)r0(p) * N,
                                    r.ag[set] + (size_t)r0(p) * N,
                                    (size_t)rows(p) * N * sizeof(sycl::half))),
                                "ring out");
                            dma_fence();
                            if (r.dma_check)
                            s->submit([&](sycl::handler &cgh) {
                                auto r_dag_set_size_t_r0_p_N_ct0 =
                                    (const sycl::uint4 *)(r.dag[set] +
                                                          (size_t)r0(p) * N);
                                auto r_ag_set_size_t_r0_p_N_ct1 =
                                    (const sycl::uint4 *)(r.ag[set] +
                                                          (size_t)r0(p) * N);
                                auto size_t_rows_p_N_ct2 =
                                    (size_t)rows(p) * N / 8;
                                auto rows_p_ct5 = rows(p);

                                cgh.parallel_for(
                                    sycl::nd_range<3>(
                                        sycl::range<3>(
                                            1, 1,
                                            grid_for((size_t)rows(p) * N / 8)) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        k_pfr_cmp(r_dag_set_size_t_r0_p_N_ct0,
                                                  r_ag_set_size_t_r0_p_N_ct1,
                                                  size_t_rows_p_N_ct2, g, 1,
                                                  rows_p_ct5, err);
                                    });
                            });
                        }
            {
                dpct::has_capability_or_fail(s->get_device(),
                                             {sycl::aspect::fp16});

                s->submit([&](sycl::handler &cgh) {
                    auto C_N_ct2 = C * N / 2;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, grid_for((size_t)C * N / 2)) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_pfr_out((const sycl::half2 *)own_ag,
                                      (sycl::float2 *)x, C_N_ct2);
                        });
                });
            }
                break;
            default: throw std::runtime_error("prefill: a ring step of unknown kind " + std::to_string((int) o.k));
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace fnx
