#pragma once
#include <sycl/sycl.hpp>
#if defined(__linux__) && __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::sycl_driver_allocation {
class Api {
    void* library=nullptr;
    template<class T> T symbol(const char* name) {
        auto value=reinterpret_cast<T>(dlsym(library,name));
        if(!value)throw std::runtime_error(std::string("Level Zero loader lacks ")+name);
        return value;
    }
public:
    decltype(&zeMemAllocHost) host;
    decltype(&zeMemAllocDevice) device;
    decltype(&zeMemFree) release;
    decltype(&zeDriverGetExtensionProperties) extensions;
    Api() {
        library=dlopen("libze_loader.so.1",RTLD_NOW|RTLD_LOCAL);
        if(!library)throw std::runtime_error("Level Zero allocation loader unavailable");
        try {
            host=symbol<decltype(host)>("zeMemAllocHost");device=symbol<decltype(device)>("zeMemAllocDevice");
            release=symbol<decltype(release)>("zeMemFree");
            extensions=symbol<decltype(extensions)>("zeDriverGetExtensionProperties");
        }catch(...){dlclose(library);throw;}
    }
    ~Api(){dlclose(library);}
    Api(const Api&)=delete;
    Api& operator=(const Api&)=delete;
};
inline Api& api(){static Api value;return value;}
// Every consuming SYCL module must also be built with
//   SYCL_PROGRAM_APPEND_COMPILE_OPTIONS=-ze-intel-greater-than-4GB-buffer-required
// The relaxed allocation descriptor permits the size but does not enable
// 64-bit buffer offsets in kernels. Without the compiler flag an A770 kernel
// can silently wrap offsets above 4 GiB, despite successful copies/allocation.
inline void* allocate(const sycl::context& context,const sycl::device& device,std::size_t bytes,bool host) {
    auto& a=api();const auto driver=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform());
    uint32_t count=0;
    if(a.extensions(driver,&count,nullptr)!=ZE_RESULT_SUCCESS)throw std::bad_alloc();
    std::vector<ze_driver_extension_properties_t> extensions(count);
    if(a.extensions(driver,&count,extensions.data())!=ZE_RESULT_SUCCESS)throw std::bad_alloc();
    bool supported=false;
    for(uint32_t i=0;i<count;++i)if(std::strcmp(extensions[i].name,ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME)==0)supported=true;
    if(!supported)throw std::runtime_error("Level Zero lacks relaxed allocation limits required by large arenas");
    ze_relaxed_allocation_limits_exp_desc_t relaxed{};
    relaxed.stype=ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
    relaxed.flags=ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    void* result=nullptr;ze_result_t status;
    auto native_context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
    if(host) {
        ze_host_mem_alloc_desc_t desc{};desc.stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;desc.pNext=&relaxed;
        status=a.host(native_context,&desc,bytes,256,&result);
    } else {
        ze_device_mem_alloc_desc_t desc{};desc.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;desc.pNext=&relaxed;
        status=a.device(native_context,&desc,bytes,256,
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device),&result);
    }
    if(status!=ZE_RESULT_SUCCESS || !result)
        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
            "Level Zero relaxed allocation failed, result="+std::to_string(static_cast<unsigned>(status)));
    return result;
}
inline void free(void* p,const sycl::context& context) {
    const auto status=api().release(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context),p);
    if(status!=ZE_RESULT_SUCCESS)throw std::runtime_error("Level Zero allocation release failed");
}
}
#define STRATA_HAS_RELAXED_ALLOCATION 1
#endif

#include <mutex>
#include <unordered_set>
namespace strata::sycl_large_allocation {
inline std::mutex allocation_mutex;
inline std::unordered_set<void*> native_allocations;
inline void* allocate(std::size_t bytes, sycl::queue& q, bool host) {
#ifdef STRATA_HAS_RELAXED_ALLOCATION
    if (q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
        bytes > q.get_device().get_info<sycl::info::device::max_mem_alloc_size>()) {
        void* p = sycl_driver_allocation::allocate(q.get_context(), q.get_device(), bytes, host);
        try { std::lock_guard<std::mutex> lock(allocation_mutex); native_allocations.insert(p); }
        catch (...) { sycl_driver_allocation::free(p, q.get_context()); throw; }
        return p;
    }
#endif
    void* p = host ? sycl::malloc_host(bytes, q) : sycl::malloc_device(bytes, q);
    if (!p) throw std::bad_alloc();
    return p;
}
inline void release(void* p, sycl::queue& q) {
    if (!p) return;
#ifdef STRATA_HAS_RELAXED_ALLOCATION
    std::lock_guard<std::mutex> lock(allocation_mutex);
    auto it = native_allocations.find(p);
    if (it != native_allocations.end()) {
        sycl_driver_allocation::free(p, q.get_context());
        native_allocations.erase(it);
        return;
    }
#endif
    sycl::free(p, q);
}
}
