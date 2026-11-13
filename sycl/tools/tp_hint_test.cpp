// tp_hint_test.cpp - do the doorbell's uncached hints work on plain host USM across two devices?
#include "strata/sycl_doorbell.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

int main() {
    dpct::select_device(0);
    sycl::queue& q0 = dpct::get_in_order_queue();
    float* h = sycl::malloc_host<float>(1024, q0);
    float* out = sycl::malloc_device<float>(1024, q0);
    for (int i = 0; i < 1024; ++i) h[i] = 1.0f;
    dpct::select_device(1);
    sycl::queue& q1 = dpct::get_in_order_queue();
    try {
        q1.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { strata::sys_store_mapped(h + i, 2.0f); }).wait();
    } catch (const sycl::exception& e) {
        std::printf("device 1 hinted store faulted: %s\n", e.what());
        return 1;
    }
    dpct::select_device(0);
    try {
        q0.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { out[i] = strata::sys_load_mapped(h + i); }).wait();
    } catch (const sycl::exception& e) {
        std::printf("device 0 hinted load faulted: %s\n", e.what());
        return 1;
    }
    float v = 0.0f;
    q0.memcpy(&v, out, sizeof(float)).wait();
    std::printf("hinted store/load OK; out[0]=%.2f\n", v);
    return 0;
}
