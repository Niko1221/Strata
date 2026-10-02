// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/core/device.hpp"
#include "strata/core/emulate.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw CudaError(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
// "gfx1201:sramecc-:xnack-" -> "gfx1201"
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {   // HIP without a usable device reports an error, not 0
        cudaGetLastError();
        return 0;
    }
    return count < 0 ? 0 : count;
}

bool device_summary(int ordinal, std::string& name, std::string& detail) {
    cudaDeviceProp p{};
    if (ordinal < 0 || ordinal >= device_count() || cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    char buf[160];
#if defined(STRATA_USE_HIP)
    std::snprintf(buf, sizeof(buf), "arch %s, %.1f GiB, wave%d", base_arch(p.gcnArchName).c_str(),
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024), p.warpSize);
#else
    std::snprintf(buf, sizeof(buf), "compute capability %d.%d, %.1f GiB", p.major, p.minor,
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024));
#endif
    name = p.name;
    detail = buf;
    return true;
}

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

std::string device_code_error() {
#if defined(STRATA_USE_HIP)
    return "";   // gpu_arch_problem() checks the HIP architectures against STRATA_HIP_ARCHS, before this point
#else
    // every .cu of the engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so this kernel stands for all
    cudaFuncAttributes a{};
    const cudaError_t e = cudaFuncGetAttributes(&a, poison_kernel);
    if (e == cudaSuccess) return {};
    cudaGetLastError();
    return cudaGetErrorString(e);
#endif
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
#if defined(STRATA_USE_HIP)
        throw CudaError(std::string("no HIP device is present; this engine was compiled for ") + STRATA_HIP_ARCHS, -1);
#else
        throw CudaError("no CUDA device is present; Strata needs an NVIDIA GPU (RTX 20 series or newer)", -1);
#endif
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    check(cudaSetDevice(ordinal), "cudaSetDevice");

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;

    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");

    // The engine supports compute capability 7.5 and newer (Turing: the QSA scorer's tf32 mma has a portable
    // fp32-FMA fallback below sm_80, the tensor-core prompt kernels refuse and fall back).  Compiling for a
    // supported arch is enforced by CMake; RUNNING on an older card is caught here, because a binary can be carried
    // to a machine with an older card and would otherwise silently take whatever path the driver chose.  The HIP
    // backend checks the card against the architectures the binary was compiled for (and wave32).
#if defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#else
    // #236: the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON: Pascal sm_60, Volta sm_70) runs on the cards it
    // was built for - refusing them below 7.5 there made the flag useless; the release engine keeps 7.5
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr int kMinCc = 60;
    const char* const kNeed = "6.0 or newer (this is the experimental Pascal / Volta build)";
#else
    constexpr int kMinCc = 75;
    const char* const kNeed = "7.5 or newer (RTX 20 / 30 / 40 / 50 series)";
#endif
    if (d.cc_major * 10 + d.cc_minor < kMinCc) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata needs compute capability " + kNeed,
                        -1);
    }
#endif
    return d;
}

namespace {

// Fase 4 (hetero multi-GPU): the capability queries need a device's context (the free-VRAM probe
// runs cudaSetDevice), and cudaSetDevice moves the CALLER'S ambient current device.  Measured leak
// 2026-10-02: a roles view (generate.cpp) left the last probed ordinal current, and the engine's
// later device picks then read the wrong device all the way to an illegal access.  This guard hands
// the ambient device back on every path (the throw paths too).
struct CurrDeviceGuard {
    CurrDeviceGuard() { cudaGetDevice(&prev_); cudaGetLastError(); }
    ~CurrDeviceGuard() { if (prev_ >= 0 && cudaSetDevice(prev_) == cudaSuccess) cudaGetLastError(); }
    CurrDeviceGuard(const CurrDeviceGuard&) = delete;
    CurrDeviceGuard& operator=(const CurrDeviceGuard&) = delete;
    int prev_ = 0;
};

DeviceCaps caps_from(const int ordinal, const cudaDeviceProp& p) {
    CurrDeviceGuard cdg_;   // hands the caller's ambient device back (see above)
    DeviceCaps c;
    c.ordinal = ordinal;
    c.name = p.name;
    // the effective cc, so a test run answers as the emulated card would (emulate.hpp)
    c.cc_major = strata::cc_major_of(p.major);
    c.cc_minor = strata::cc_minor_of(p.minor);
    c.multi_processor_count = p.multiProcessorCount;
    size_t free_b = 0, total_b = 0;
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    c.total_bytes = total_b;
    c.free_bytes = free_b;
    c.shared_mem_default = (int) p.sharedMemPerBlock;
    c.shared_mem_optin = strata::smem_optin_of((int) p.sharedMemPerBlockOptin);
    const int cc = c.cc_major * 10 + c.cc_minor;
    c.dp4a = cc >= 61;
    c.mma_int8 = cc >= 75;
    c.mma_tf32 = cc >= 80;
    c.cp_async = cc >= 80;
    check(cudaDriverGetVersion(&c.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&c.runtime_version), "cudaRuntimeGetVersion");
#if defined(STRATA_USE_HIP)
    c.arch = base_arch(p.gcnArchName);
#endif
    return c;
}

}  // namespace

std::vector<DeviceCaps> device_caps() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    std::vector<DeviceCaps> out;
    out.reserve((size_t) count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
        out.push_back(caps_from(ordinal, p));
    }
    return out;
}

int device_cc_major(int ordinal) {
    static int cc_major[64] = {};
    if (ordinal < 0 || ordinal >= 64) return -1;
    if (cc_major[ordinal] == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ordinal) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        cc_major[ordinal] = strata::cc_major_of(major);
    }
    return cc_major[ordinal];
}

std::vector<uint8_t> peer_access_matrix() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    std::vector<uint8_t> m((size_t) count * count, 0);
    for (int a = 0; a < count; ++a) {
        for (int b = 0; b < count; ++b) {
            if (a == b) continue;
            int can = 0;
            check(cudaDeviceCanAccessPeer(&can, a, b), "cudaDeviceCanAccessPeer");
            m[(size_t) a * count + b] = can ? 1 : 0;
        }
    }
    return m;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        // gridDim.x is 32-bit, so a large region needs a loop.  12 GB of floats is 3e9 elements = 1.2e7
        // blocks, which fits, but the loop keeps it correct for any size rather than for today's sizes.
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads);
            check(cudaGetLastError(), "poison_kernel");
        }
        check(cudaDeviceSynchronize(), "poison sync");
    }
}

DeviceArena::~DeviceArena() {
    if (base_) cudaFree(base_);          // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
