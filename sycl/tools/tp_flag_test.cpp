// tp_flag_test.cpp - cross-device host flag handshake: device 1 stores, device 0 spins.
#include "strata/sycl_doorbell.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

int main() {
    dpct::select_device(0);
    sycl::queue& q0 = dpct::get_in_order_queue();
    uint32_t* flag = sycl::malloc_host<uint32_t>(1, q0);
    *flag = 0;
    dpct::select_device(1);
    sycl::queue& q1 = dpct::get_in_order_queue();
    q1.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { strata::sys_store(flag, 1u); }).wait();
    dpct::select_device(0);
    uint32_t* seen = sycl::malloc_device<uint32_t>(1, q0);
    try {
        q0.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            uint32_t spins = 0;
            while (strata::sys_load(flag) == 0 && spins < 20000) ++spins;
            *seen = strata::sys_load(flag);
        }).wait();
    } catch (const sycl::exception& e) {
        std::printf("spin faulted: %s\n", e.what());
        return 1;
    }
    uint32_t v = 0;
    q0.memcpy(&v, seen, sizeof(v)).wait();
    std::printf("flag handshake OK; seen=%u\n", v);
    return 0;
}
