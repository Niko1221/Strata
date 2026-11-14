// tp_bulk_test.cpp - bulk staging: device 1 writes 1024 floats with the hint, device 0 reads them.
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
    q1.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { strata::sys_store_mapped(h + i, 2.0f); }).wait();
    dpct::select_device(0);
    q0.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { out[i] = strata::sys_load_mapped(h + i); }).wait();
    std::vector<float> host(1024);
    q0.memcpy(host.data(), out, 1024 * sizeof(float)).wait();
    int bad = 0;
    for (float v : host) if (v != 2.0f) ++bad;
    std::printf("bulk staging: %d of 1024 wrong; host[0]=%.2f\n", bad, h[0]);
    return bad == 0 ? 0 : 1;
}
