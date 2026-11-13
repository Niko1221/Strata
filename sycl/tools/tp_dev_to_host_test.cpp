// tp_dev_to_host_test.cpp - read a device value and store it to host USM with the uncached hint.
#include "strata/sycl_doorbell.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

int main() {
    dpct::select_device(1);
    sycl::queue& q1 = dpct::get_in_order_queue();
    float* dev = sycl::malloc_device<float>(1024, q1);
    std::vector<float> host_in(1024, 2.0f);
    q1.memcpy(dev, host_in.data(), 1024 * sizeof(float)).wait();
    dpct::select_device(0);
    sycl::queue& q0 = dpct::get_in_order_queue();
    float* h = sycl::malloc_host<float>(1024, q0);
    dpct::select_device(1);
    try {
        q1.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) {
            strata::sys_store_mapped(h + i, dev[i]);
        }).wait();
    } catch (const sycl::exception& e) {
        std::printf("dev->host hinted store faulted: %s\n", e.what());
        return 1;
    }
    std::printf("dev->host hinted store OK; h[0]=%.2f\n", h[0]);
    return 0;
}
