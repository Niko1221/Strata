// tp_barrier_test.cpp - nd_range + group_barrier + atomic_ref + staging hints on two devices.
#include "strata/sycl_doorbell.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

int main() {
    dpct::select_device(0);
    sycl::queue& q0 = dpct::get_in_order_queue();
    float* h = sycl::malloc_host<float>(2048, q0);
    uint32_t* flag = sycl::malloc_host<uint32_t>(8, q0);
    for (int i = 0; i < 2048; ++i) h[i] = 1.0f;
    for (int i = 0; i < 8; ++i) flag[i] = 0;
    dpct::select_device(1);
    sycl::queue& q1 = dpct::get_in_order_queue();
    uint32_t* tok = sycl::malloc_device<uint32_t>(8, q1);
    q1.memset(tok, 0, 8 * sizeof(uint32_t)).wait();
    try {
        q1.parallel_for(sycl::nd_range<1>(2048, 256), [=](sycl::nd_item<1> it) {
            const int bid = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            if (tid == 0) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> r(tok[bid]);
                r.fetch_add(1u);
                strata::sys_store(flag + bid, 1u);
            }
            sycl::group_barrier(it.get_group());
            strata::sys_store_mapped(h + bid * 256 + tid, 2.0f);
        }).wait();
    } catch (const sycl::exception& e) {
        std::printf("device 1 kernel faulted: %s\n", e.what());
        return 1;
    }
    std::printf("device 1 kernel OK; h[0]=%.2f flag[0]=%u\n", h[0], flag[0]);
    return 0;
}
