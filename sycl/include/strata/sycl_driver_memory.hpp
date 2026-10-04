#pragma once

// Keep Sysman optional: the SYCL extension remains usable on installations
// without standalone Sysman headers or a loader. Never substitute total VRAM
// (or this process's allocation bookkeeping) for driver-reported free memory.
#if defined(__linux__) && __has_include(<level_zero/zes_api.h>)
#include <level_zero/zes_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <dlfcn.h>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::sycl_driver_memory {
inline void checked(ze_result_t result,const char* operation) {
    if(result!=ZE_RESULT_SUCCESS)
        throw std::runtime_error(std::string(operation)+" failed (Level Zero result "+
                                 std::to_string(static_cast<unsigned>(result))+")");
}
class Sysman {
    void* library=nullptr;
    template<class T> T symbol(const char* name) {
        auto value=reinterpret_cast<T>(dlsym(library,name));
        if(!value) throw std::runtime_error(std::string("Level Zero loader lacks ")+name);
        return value;
    }
public:
    decltype(&zesDriverGet) drivers;
    decltype(&zesDriverGetDeviceByUuidExp) by_uuid;
    decltype(&zeDeviceGetProperties) device_properties;
    decltype(&zesDeviceEnumMemoryModules) modules;
    decltype(&zesMemoryGetProperties) memory_properties;
    decltype(&zesMemoryGetState) memory_state;
    Sysman() {
        library=dlopen("libze_loader.so.1",RTLD_NOW|RTLD_LOCAL);
        if(!library) throw std::runtime_error("Level Zero Sysman loader unavailable");
        try {
            drivers=symbol<decltype(drivers)>("zesDriverGet");
            by_uuid=symbol<decltype(by_uuid)>("zesDriverGetDeviceByUuidExp");
            device_properties=symbol<decltype(device_properties)>("zeDeviceGetProperties");
            modules=symbol<decltype(modules)>("zesDeviceEnumMemoryModules");
            memory_properties=symbol<decltype(memory_properties)>("zesMemoryGetProperties");
            memory_state=symbol<decltype(memory_state)>("zesMemoryGetState");
            checked(symbol<decltype(&zesInit)>("zesInit")(0),"zesInit");
        } catch(...) {dlclose(library);throw;}
    }
    ~Sysman(){dlclose(library);}
    Sysman(const Sysman&)=delete;
    Sysman& operator=(const Sysman&)=delete;
    std::pair<std::size_t,std::size_t> query(const sycl::device& device) {
        ze_device_properties_t properties{};properties.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        checked(device_properties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device),&properties),
                "zeDeviceGetProperties");
        zes_uuid_t uuid{};
        static_assert(sizeof(uuid.id)==sizeof(properties.uuid.id));
        std::memcpy(uuid.id,properties.uuid.id,sizeof(uuid.id));
        uint32_t count=0;checked(drivers(&count,nullptr),"zesDriverGet count");
        std::vector<zes_driver_handle_t> handles(count);
        checked(drivers(&count,handles.data()),"zesDriverGet");
        handles.resize(count);
        zes_device_handle_t matched=nullptr;ze_bool_t subdevice=false;uint32_t subdevice_id=0;
        for(auto driver:handles) {
            zes_device_handle_t candidate=nullptr;
            if(by_uuid(driver,uuid,&candidate,&subdevice,&subdevice_id)==ZE_RESULT_SUCCESS) {
                matched=candidate;break;
            }
        }
        if(!matched) throw std::runtime_error("Sysman could not match the selected GPU UUID");
        count=0;checked(modules(matched,&count,nullptr),"zesDeviceEnumMemoryModules count");
        std::vector<zes_mem_handle_t> memory(count);
        checked(modules(matched,&count,memory.data()),"zesDeviceEnumMemoryModules");
        memory.resize(count);
        std::size_t available=0,total=0;
        bool root_modules=false,child_modules=false;
        for(auto module:memory) {
            zes_mem_properties_t info{};info.stype=ZES_STRUCTURE_TYPE_MEM_PROPERTIES;
            checked(memory_properties(module,&info),"zesMemoryGetProperties");
            if(info.location!=ZES_MEM_LOC_DEVICE) continue;
            if(subdevice && (!info.onSubdevice || info.subdeviceId!=subdevice_id)) continue;
            root_modules|=!info.onSubdevice;child_modules|=bool(info.onSubdevice);
            // Intel currently enumerates root modules OR per-tile modules.
            // A mixed hierarchy has ambiguous overlap; refuse to double-count.
            if(root_modules && child_modules)
                throw std::runtime_error("Sysman returned ambiguous root and subdevice memory modules");
            zes_mem_state_t state{};state.stype=ZES_STRUCTURE_TYPE_MEM_STATE;
            checked(memory_state(module,&state),"zesMemoryGetState");
            if(!state.size || state.free>state.size ||
               state.size>std::numeric_limits<std::size_t>::max()-total)
                throw std::runtime_error("Sysman returned invalid device memory telemetry");
            available+=static_cast<std::size_t>(state.free);
            total+=static_cast<std::size_t>(state.size);
        }
        if(!total) throw std::runtime_error("Sysman has no usable device-local memory telemetry");
        return {available,total};
    }
};
inline std::pair<std::size_t,std::size_t> query(const sycl::device& device) {
    static Sysman sysman;
    return sysman.query(device);
}
}
#define STRATA_HAS_STANDALONE_SYSMAN 1
#endif
