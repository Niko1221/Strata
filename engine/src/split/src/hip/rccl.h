// RCCL (AMD) or NCCL (NVIDIA, FNX_PF_NCCL=1) for the prompt path's chunk sums (the rules and the loader are hip/rccl_dl.hpp: loaded at run time, never linked; the two
// libraries share the API). open() loads the library, makes one communicator per card and runs a self-test (a four-card fp16 all-reduce whose sums must come back
// exact) before the engine may use it; any failure is one line and the host exchange. The engine calls the six functions only through the RcclApi pointers.
#pragma once

#if FNX_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rccl_dl.hpp"

#if !FNX_HIP
// NVIDIA: the HIP calls of the self-test below, by their CUDA names (undefined again at the end of this file)
#define hipSetDevice cudaSetDevice
#define hipFree cudaFree
#define hipMalloc cudaMalloc
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipStream_t cudaStream_t
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamQuery cudaStreamQuery
#define hipErrorNotReady cudaErrorNotReady
#define hipGetLastError cudaGetLastError
#define hipSuccess cudaSuccess
#define FNX_RCCL_CUDA_NAMES 1
#endif

namespace fnx {

struct RcclState {
    RcclApi api;
    std::vector<void*> comm;   // [card]
    ~RcclState() {
        for (void* c : comm)
            if (c && api.CommDestroy) api.CommDestroy(c);
        // (the library stays loaded: RCCL's helper threads may still be winding down)
    }
};

namespace rccl_detail {
static __global__ void k_fill16(__half* p, int n, float scale) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) p[e] = __float2half((float) (e % 16) * scale);
}
static __global__ void k_check16(const __half* p, int n, float scale, unsigned* bad) {
    unsigned b = 0;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) b += __half2float(p[e]) != (float) (e % 16) * scale;
    if (b) atomicAdd(bad, b);
}
}  // namespace rccl_detail

// devs: the cards' device ordinals. On success st holds the communicators; why is the library path (or the reason for the failure).
inline bool rccl_open(const std::vector<int>& devs, std::shared_ptr<RcclState>& st, std::string& why) {
    auto s = std::make_shared<RcclState>();
#if FNX_HIP
    if (!rccl_load(s->api, "", why)) return false;   // librccl.so.1
#else
    if (!rccl_load(s->api, NCCL_LIB, why)) return false;
#endif
    const std::string lib = why;
    const int n = (int) devs.size();
    s->comm.assign((size_t) n, nullptr);
    auto fail = [&](const char* what, int r) {
        why = std::string(what) + (s->api.ErrorString ? std::string(": ") + s->api.ErrorString(r) : std::string(" failed"));
        return false;
    };
    int r = s->api.CommInitAll(s->comm.data(), n, devs.data());
    if (r != RCCL_SUCCESS) {
        s->comm.assign((size_t) n, nullptr);
        return fail("ncclCommInitAll", r);
    }
    // the self-test: every card holds (e % 16) * (card + 1) as fp16, the sum must be (e % 16) * n (n + 1) / 2 on every card, in full
    const int cnt = 1 << 18;
    const float scale_sum = (float) (n * (n + 1) / 2);
    std::vector<__half*> buf((size_t) n, nullptr);
    std::vector<unsigned*> bad((size_t) n, nullptr);
    std::vector<hipStream_t> str((size_t) n, nullptr);
    auto cleanup = [&] {
        for (int i = 0; i < n; ++i) {
            hipSetDevice(devs[(size_t) i]);
            if (buf[(size_t) i]) hipFree(buf[(size_t) i]);
            if (bad[(size_t) i]) hipFree(bad[(size_t) i]);
            if (str[(size_t) i]) hipStreamDestroy(str[(size_t) i]);
        }
    };
    for (int i = 0; i < n; ++i) {
        hipSetDevice(devs[(size_t) i]);
        if (hipMalloc((void**) &buf[(size_t) i], (size_t) cnt * 2) != hipSuccess || hipMalloc((void**) &bad[(size_t) i], 4) != hipSuccess ||
            hipStreamCreateWithFlags(&str[(size_t) i], hipStreamNonBlocking) != hipSuccess) {
            hipGetLastError();
            cleanup();
            why = "the self-test buffers could not be allocated";
            return false;
        }
        hipMemset(bad[(size_t) i], 0, 4);
        rccl_detail::k_fill16<<<256, 256, 0, str[(size_t) i]>>>(buf[(size_t) i], cnt, (float) (i + 1));
    }
    r = s->api.GroupStart();
    for (int i = 0; i < n && r == RCCL_SUCCESS; ++i) {
        hipSetDevice(devs[(size_t) i]);
        r = s->api.AllReduce(buf[(size_t) i], buf[(size_t) i], (size_t) cnt, RCCL_FLOAT16, RCCL_SUM, s->comm[(size_t) i], (void*) str[(size_t) i]);
    }
    const int r2 = s->api.GroupEnd();
    if (r == RCCL_SUCCESS) r = r2;
    if (r != RCCL_SUCCESS) {
        cleanup();
        return fail("ncclAllReduce", r);
    }
    for (int i = 0; i < n; ++i) {
        hipSetDevice(devs[(size_t) i]);
        rccl_detail::k_check16<<<256, 256, 0, str[(size_t) i]>>>(buf[(size_t) i], cnt, scale_sum, bad[(size_t) i]);
    }
    // a bounded wait (30 s): a collective that never completes must not hang the start
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        hipSetDevice(devs[(size_t) i]);
        while (hipStreamQuery(str[(size_t) i]) == hipErrorNotReady)
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
                why = "the self-test all-reduce did not finish in 30 s";
                return false;   // (buffers stay: the kernels may still be running)
            }
    }
    unsigned total = 0;
    for (int i = 0; i < n; ++i) {
        hipSetDevice(devs[(size_t) i]);
        unsigned b = 0;
        hipMemcpy(&b, bad[(size_t) i], 4, hipMemcpyDeviceToHost);
        total += b;
    }
    cleanup();
    if (hipGetLastError() != hipSuccess) {
#if FNX_HIP
        why = "a HIP error during the self-test";
#else
        why = "a CUDA error during the self-test";
#endif
        return false;
    }
    if (total) {
        why = "the self-test sum was wrong in " + std::to_string(total) + " values";
        return false;
    }
    st = s;
    why = lib;
    return true;
}

}  // namespace fnx

#ifdef FNX_RCCL_CUDA_NAMES
#undef hipSetDevice
#undef hipFree
#undef hipMalloc
#undef hipMemset
#undef hipMemcpy
#undef hipMemcpyDeviceToHost
#undef hipStream_t
#undef hipStreamCreateWithFlags
#undef hipStreamNonBlocking
#undef hipStreamDestroy
#undef hipStreamQuery
#undef hipErrorNotReady
#undef hipGetLastError
#undef hipSuccess
#undef FNX_RCCL_CUDA_NAMES
#endif
