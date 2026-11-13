// tp_uncached_visibility_test.cpp - is an uncached zeMemAllocHost allocation in a multi-device context
// visible to a kernel on another device?
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <cstdio>

int main() {
    std::vector<sycl::device> devs;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) devs.push_back(d);
    if (devs.size() < 2) { std::printf("need 2 GPUs\n"); return 1; }
    sycl::context ctx(devs);
    ze_context_handle_t zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_host_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                  ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
    float* h = nullptr;
    if (zeMemAllocHost(zctx, &desc, 4096, 64, (void**)&h) != ZE_RESULT_SUCCESS) {
        std::printf("zeMemAllocHost failed\n"); return 1;
    }
    h[0] = 1.0f;
    sycl::queue q1(devs[1]);
    try {
        q1.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { h[0] = 2.0f; }).wait();
    } catch (const sycl::exception& e) {
        std::printf("device 1 plain store faulted: %s\n", e.what());
        zeMemFree(zctx, h);
        return 1;
    }
    std::printf("device 1 plain store OK; h[0]=%.2f\n", h[0]);
    zeMemFree(zctx, h);
    return 0;
}
