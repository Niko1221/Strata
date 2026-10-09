// src/kernels/cuda/ep_kernels.dp.cpp (SYCL port only) - see include/strata/kernels/ep_kernels.hpp.
#include <sycl/sycl.hpp>
#include <algorithm>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/ep_kernels.hpp"

namespace strata::kernels {
namespace {
// the doorbell's uncached L1+L3 read (sycl_doorbell.hpp), for whole buffers
inline uint32_t load_uncached(const uint32_t* p) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(const_cast<uint32_t*>(p));
    return u[0];
}
}  // namespace

void ep_copy_uncached(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return;
    constexpr size_t WG = 256;
    const size_t words = bytes / 4;
    const size_t groups = std::min<size_t>((words + WG - 1) / WG, 64);
    uint32_t* d = (uint32_t*) dst;
    const uint32_t* s = (const uint32_t*) src;
    q_of(stream)->parallel_for<class ep_copy_uncached_kernel>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < words; i += groups * WG) d[i] = load_uncached(s + i);
    });
}

void ep_bump(uint32_t* ctr, void* stream) {
    q_of(stream)->single_task<class ep_bump_kernel>([=] { *ctr += 1; });
}

void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    constexpr size_t WG = 256;
    // q8_1 rows are 36-byte blocks: copied as 4-byte words (both buffers are 4-byte aligned, xq_bytes a multiple of 4)
    const size_t words = xq_bytes / 4;
    q_of(stream)->parallel_for<class ep_push_kernel>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t) n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = (const uint32_t*) xq;
        uint32_t* d = (uint32_t*) peer_xq;
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        // every work-item's stores into the peer before the flag
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (lid == 0) sys_store(peer_flag, *ctr);
    });
}

void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream) {
    q_of(stream)->single_task<class ep_wait_kernel>([=] {
        const uint32_t want = *ctr;
        uint32_t spin = 0;
        // the epoch counters only grow and wrap: compare the difference, not the values
        while ((int32_t) (sys_load(flag) - want) < 0) {
            if (++spin >= spin_max) {
                if (err != nullptr) sys_store(err, code);
                break;
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    });
}

void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    using sys_rmw = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system>;
    q_of(stream)->parallel_for<class ep_send_rows_kernel>(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        const int32_t e = ids[r];
        if (!(e >= 0 && res0[e] >= 0)) {   // the peer's entry: its row goes to card 0
            const sycl::float4* s = (const sycl::float4*) (rows + (size_t) r * row);
            sycl::float4* d = (sycl::float4*) (peer_rows + (size_t) r * row);
            for (int64_t i = it.get_local_id(0); i < row / 4; i += WG) d[i] = s[i];
        }
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t) n - 1) {
            *done = 0;   // every group has counted itself: the next call starts from zero
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_store(peer_flag, *ctr);
        }
    });
}


void ep_compare(const float* parts, const float* peer_rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                uint32_t* mismatch, void* stream) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    q_of(stream)->parallel_for<class ep_compare_kernel>(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        const int32_t e = ids[r];
        if (e >= 0 && res0[e] >= 0) return;
        const uint32_t* a = (const uint32_t*) (parts + (size_t) r * row);
        const uint32_t* b = (const uint32_t*) (peer_rows + (size_t) r * row);
        int bad = 0;
        for (int64_t i = it.get_local_id(0); i < row; i += WG) bad |= a[i] != load_uncached(b + i);
        bad = sycl::reduce_over_group(it.get_group(), bad, sycl::bit_or<int>());
        if (it.get_local_id(0) == 0 && bad)
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>(*mismatch).fetch_add(1u);
    });
}

}  // namespace strata::kernels
