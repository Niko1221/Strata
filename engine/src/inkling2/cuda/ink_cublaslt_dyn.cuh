// engine/src/inkling2/cuda/ink_cublaslt_dyn.cuh - find cublasLt symbols at run time (the RTX 50 FP8 engines call cublasLt without linking it)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// One loader for both FP8 engines (they each had a copy). The rules, from the review of the 5090 hand-over:
//   * a cublasLt that is already in the process wins: cublas (which the engine links) pulls its own cublasLt in, and a second copy of another
//     major version next to it must never be loaded;
//   * otherwise the major version of the toolkit the engine was built with is tried first (13 for CUDA 13.x, 12 for 12.x), then the other one;
//   * Windows: the DLL is looked for in the application folder and System32 only (no PATH / current directory search: no DLL planting).
// Only reached with an RTX 50 fast-mode part on (the FP8 trunk).
#pragma once

#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strata::ink {

inline void* ink_cublaslt_symbol(const char* name) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    constexpr int kFirst = 13, kSecond = 12;
#else
    constexpr int kFirst = 12, kSecond = 13;
#endif
#ifndef _WIN32
    // already loaded (the usual case: libcublas pulls libcublasLt in)
    if (void* p = dlsym(RTLD_DEFAULT, name)) return p;
    static void* lh = [] {
        const char* names[3] = {kFirst == 13 ? "libcublasLt.so.13" : "libcublasLt.so.12",
                                kSecond == 13 ? "libcublasLt.so.13" : "libcublasLt.so.12", "libcublasLt.so"};
        for (const char* n : names) {
            if (void* h = dlopen(n, RTLD_LAZY | RTLD_GLOBAL)) return h;
        }
        return static_cast<void*>(nullptr);
    }();
    return lh ? dlsym(lh, name) : nullptr;
#else
    static HMODULE hm = [] {
        const char* names[2] = {kFirst == 13 ? "cublasLt64_13.dll" : "cublasLt64_12.dll",
                                kSecond == 13 ? "cublasLt64_13.dll" : "cublasLt64_12.dll"};
        for (const char* n : names) {
            if (HMODULE h = GetModuleHandleA(n)) return h;   // already in the process
        }
        for (const char* n : names) {
            if (HMODULE h = LoadLibraryExA(n, nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32)) return h;
        }
        return static_cast<HMODULE>(nullptr);
    }();
    return hm ? reinterpret_cast<void*>(GetProcAddress(hm, name)) : nullptr;
#endif
}

}  // namespace strata::ink
