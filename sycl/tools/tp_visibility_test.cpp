// tp_visibility_test.cpp - can a kernel on device 1 read/write host USM allocated in device 0's context?
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <vector>

int main() {
    dpct::select_device(0);
    sycl::queue& q0 = dpct::get_in_order_queue();
    float* h = sycl::malloc_host<float>(1024, q0);
    if (h == nullptr) { std::printf("malloc_host failed\n"); return 1; }
    for (int i = 0; i < 1024; ++i) h[i] = 1.0f;
    // device 1 writes to device 0's host allocation
    dpct::select_device(1);
    sycl::queue& q1 = dpct::get_in_order_queue();
    try {
        q1.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { h[i] = 2.0f; }).wait();
    } catch (const sycl::exception& e) {
        std::printf("device 1 write faulted: %s\n", e.what());
        return 1;
    }
    std::printf("device 1 write OK; h[0]=%.2f\n", h[0]);
    // device 0 reads it back
    dpct::select_device(0);
    q0.wait_and_throw();
    std::printf("device 0 read h[0]=%.2f\n", h[0]);
    sycl::free(h, q0);
    return 0;
}
