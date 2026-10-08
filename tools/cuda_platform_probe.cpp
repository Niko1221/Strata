// No model or large allocations: collect facts before choosing a UMA policy.
#include <cuda_runtime.h>
#include <cuda.h>
#include "strata/platform/memory.hpp"
#include "strata/platform/cpu_relax.hpp"
#include <cstdio>
#include <cstring>
#include <string>

static bool checked(cudaError_t e, const char* operation) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(e));
    return false;
}

static std::string json_string(const char* text) {
    std::string result = "\"";
    for (const unsigned char* p = (const unsigned char*) text; *p; ++p) {
        if (*p == '"' || *p == '\\') result += '\\';
        if (*p < 32) {
            char escaped[7];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", *p);
            result += escaped;
        } else result += (char) *p;
    }
    return result + '"';
}

int main(int argc, char** argv) {
    const bool json = argc == 2 && std::strcmp(argv[1], "--json") == 0;
    if (argc > 1 && !json) { std::fprintf(stderr, "usage: strata-cuda-probe [--json]\n"); return 2; }
    strata::platform::cpu_relax();
    int count = 0, driver = 0, runtime = 0;
    if (!checked(cudaGetDeviceCount(&count), "cudaGetDeviceCount") ||
        !checked(cudaDriverGetVersion(&driver), "cudaDriverGetVersion") ||
        !checked(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion")) return 1;
    if (count == 0) { std::fprintf(stderr, "No CUDA devices\n"); return 1; }
    if (cuInit(0) != CUDA_SUCCESS) { std::fprintf(stderr, "cuInit failed\n"); return 1; }
    std::printf(json ? "{\"driver_version\":%d,\"runtime_version\":%d,\"host_physical_bytes\":%llu,\"devices\":["
                     : "driver_version=%d runtime_version=%d host_physical_bytes=%llu\n", driver, runtime,
                (unsigned long long) strata::platform::total_physical_memory());
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        size_t free = 0, total = 0;
        if (!checked(cudaSetDevice(i), "cudaSetDevice") ||
            !checked(cudaGetDeviceProperties(&p, i), "cudaGetDeviceProperties") ||
            !checked(cudaMemGetInfo(&free, &total), "cudaMemGetInfo")) return 1;
        int vmm = 0, registration = 0;
        CUdevice device;
        if (cuDeviceGet(&device, i) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device) != CUDA_SUCCESS) {
            std::fprintf(stderr, "VMM attribute query failed for device %d\n", i);
            return 1;
        }
        if (!checked(cudaDeviceGetAttribute(&registration, cudaDevAttrHostRegisterSupported, i), "host registration attribute")) return 1;
        if (json) {
            std::printf("%s{\"device\":%d,\"name\":%s,\"sm\":%d,\"integrated\":%d,\"unified_addressing\":%d,"
                        "\"cuda_total_bytes\":%llu,\"cuda_free_bytes\":%llu,\"managed_memory\":%d,"
                        "\"concurrent_managed_access\":%d,\"pageable_memory_access\":%d,"
                        "\"can_map_host_memory\":%d,\"host_register_supported\":%d,\"vmm_supported\":%d}",
                        i ? "," : "", i, json_string(p.name).c_str(), p.major * 10 + p.minor, p.integrated,
                        p.unifiedAddressing, (unsigned long long) total, (unsigned long long) free,
                        p.managedMemory, p.concurrentManagedAccess, p.pageableMemoryAccess,
                        p.canMapHostMemory, registration, vmm);
        } else std::printf("device=%d name=%s sm=%d%d integrated=%d unified_addressing=%d\n"
                    "  cuda_total_bytes=%llu cuda_free_bytes=%llu\n"
                    "  managed_memory=%d concurrent_managed_access=%d pageable_memory_access=%d\n"
                    "  can_map_host_memory=%d host_register_supported=%d vmm_supported=%d\n",
                    i, p.name, p.major, p.minor, p.integrated, p.unifiedAddressing,
                    (unsigned long long) total, (unsigned long long) free,
                    p.managedMemory, p.concurrentManagedAccess, p.pageableMemoryAccess,
                    p.canMapHostMemory, registration, vmm);
    }
    if (json) std::printf("]}\n");
    return 0;
}
