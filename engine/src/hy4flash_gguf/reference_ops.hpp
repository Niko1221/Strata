#pragma once

#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace strata::hy4_flash {
inline const char* reference_ops_path() {
    static const char* path = std::getenv("STRATA_HY4_FLASH_REFERENCE_OPS_LIBRARY");
    return path && *path ? path : nullptr;
}
inline void* reference_ops_library() {
    static void* library = [] {
        void* handle = dlopen(reference_ops_path(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (!handle)
            throw std::runtime_error(std::string("HY4 reference library: ") + dlerror());
        return handle;
    }();
    return library;
}

template <class Fn> Fn reference_op(const char* symbol) {
    auto fn = reinterpret_cast<Fn>(dlsym(reference_ops_library(), symbol));
    if (!fn)
        throw std::runtime_error(std::string("HY4 reference library missing ") + symbol);
    return fn;
}
} // namespace strata::hy4_flash
