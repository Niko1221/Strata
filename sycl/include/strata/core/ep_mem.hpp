// include/strata/core/ep_mem.hpp (SYCL port only) - device memory one Arc card writes and the other reads
// (ep_kernels.hpp).  dpct gives each device its own sycl::context, so the region is allocated from Level Zero on its
// owner and opened on the other card through an IPC handle.  It must be an allocation of its own: on a small
// sycl::malloc_device block (carved from the runtime's USM pool) the other card page-faulted or disturbed neighbours.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <dpct/dpct.hpp>

#include <cstddef>
#include <string>

namespace strata::core {

struct EpRegion {
    void* local = nullptr;    ///< the owner's address
    void* remote = nullptr;   ///< the same bytes as the other card's kernels address them
    ze_context_handle_t owner_ctx = nullptr, other_ctx = nullptr;
};

inline ze_context_handle_t ep_ze_context(int dev) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dpct::dev_mgr::instance().get_device(dev).get_context());
}
inline ze_device_handle_t ep_ze_device(int dev) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        static_cast<sycl::device&>(dpct::dev_mgr::instance().get_device(dev)));
}

/// `bytes` zeroed on dpct device `owner`, opened for dpct device `other`.
inline bool ep_region_alloc(int owner, int other, size_t bytes, EpRegion& r, std::string& err) {
    r = EpRegion{};
    r.owner_ctx = ep_ze_context(owner);
    r.other_ctx = ep_ze_context(other);
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ze_result_t z = zeMemAllocDevice(r.owner_ctx, &desc, bytes, 4096, ep_ze_device(owner), &r.local);
    if (z != ZE_RESULT_SUCCESS) {
        err = "expert parallel: zeMemAllocDevice of " + std::to_string(bytes) + " bytes failed (" + std::to_string((unsigned) z) + ")";
        return false;
    }
    ze_ipc_mem_handle_t h{};
    z = zeMemGetIpcHandle(r.owner_ctx, r.local, &h);
    if (z == ZE_RESULT_SUCCESS) z = zeMemOpenIpcHandle(r.other_ctx, ep_ze_device(other), h, 0, &r.remote);
    if (z != ZE_RESULT_SUCCESS) {
        zeMemFree(r.owner_ctx, r.local);
        r.local = nullptr;
        err = "expert parallel: opening card " + std::to_string(owner) + "'s memory on card " + std::to_string(other) +
              " failed (" + std::to_string((unsigned) z) + ")";
        return false;
    }
    dpct::dev_mgr::instance().get_device(owner).in_order_queue().memset(r.local, 0, bytes).wait();
    return true;
}

inline void ep_region_free(EpRegion& r) {
    if (r.remote != nullptr) zeMemCloseIpcHandle(r.other_ctx, r.remote);
    if (r.local != nullptr) zeMemFree(r.owner_ctx, r.local);
    r = EpRegion{};
}

}  // namespace strata::core
