// Device-scoped defaults for kernels measured on Arc Pro B70.
#pragma once
#include "strata/sycl_queue.hpp"

namespace strata {
inline bool is_b70(const void* stream) {
#ifdef STRATA_HAVE_ZE
    const auto* q = q_of(stream);
    if (q->get_backend() != sycl::backend::ext_oneapi_level_zero) return false;
    const auto device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_device());
    static thread_local ze_device_handle_t cached_device = nullptr;
    static thread_local bool supported = false;
    if (device != cached_device) {
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        supported = zeDeviceGetProperties(device, &props) == ZE_RESULT_SUCCESS &&
                    props.vendorId == 0x8086 && props.deviceId == 0xe223;
        cached_device = device;
    }
    return supported;
#else
    return false;
#endif
}
}  // namespace strata
