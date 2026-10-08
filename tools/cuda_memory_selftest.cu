// Small, model-free tests of the memory primitives used by the engine on Orin.
#include <cuda_runtime.h>
#include <cuda.h>
#include "strata/platform/cpu_relax.hpp"
#include "strata/platform/mapped_memory.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

static bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}
__global__ void increment(int* p) { if (threadIdx.x == 0) ++*p; }

struct Doorbell { uint32_t seq, flag, gpu_value, cpu_value, result; };
__global__ void exchange(volatile Doorbell* p, unsigned long long timeout) {
    const auto start = clock64();
    for (uint32_t i = 1; i <= 512; ++i) {
        p->gpu_value = i * 3;
        __threadfence_system();
        p->seq = i;
        __threadfence_system();
        while (p->flag < i) {
            if (clock64() - start > timeout) { p->result = 1; return; }
        }
        __threadfence_system();
        if (p->cpu_value != i * 7) { p->result = 2; return; }
    }
    __threadfence_system();
}

int main() {
    cudaDeviceProp prop{};
    if (!ck(cudaGetDeviceProperties(&prop, 0), "device properties")) return 1;
    // Managed allocations are tested with synchronization at every ownership
    // transition. concurrentManagedAccess=0 forbids the doorbell use case.
    int* managed = nullptr;
    if (!ck(cudaMallocManaged(&managed, 4096), "managed allocation")) return 1;
    *managed = 41;
    increment<<<1, 1>>>(managed);
    bool ok = ck(cudaGetLastError(), "managed launch") && ck(cudaDeviceSynchronize(), "managed sync");
    ok = ok && *managed == 42;
    cudaFree(managed);
    if (!ok) return 1;
    std::puts("managed sequential CPU/GPU ownership: PASS");

    // The large host expert arena uses registration, not managed memory.
    int* registered = (int*) std::aligned_alloc(4096, 4096);
    if (!registered) return 1;
    *registered = 41;
    if (!ck(cudaHostRegister(registered, 4096, cudaHostRegisterMapped), "host registration")) {
        std::free(registered); return 1;
    }
    int* alias = nullptr;
    ok = ck(cudaHostGetDevicePointer(&alias, registered, 0), "registered alias");
    if (ok) {
        increment<<<1, 1>>>(alias);
        ok = ck(cudaGetLastError(), "registered launch") && ck(cudaDeviceSynchronize(), "registered sync");
        ok = ok && *registered == 42;
    }
    cudaHostUnregister(registered);
    std::free(registered);
    if (!ok) return 1;
    std::puts("host registration and mapped access: PASS");

    Doorbell* host = nullptr;
    Doorbell* device = nullptr;
    if (!ck(cudaHostAlloc(&host, sizeof(Doorbell), cudaHostAllocMapped), "mapped allocation")) return 1;
    *host = {};
    if (!ck(cudaHostGetDevicePointer(&device, host, 0), "mapped alias")) { cudaFreeHost(host); return 1; }
    strata::platform::mapped_memory_release();
    exchange<<<1, 1>>>(device, (unsigned long long) prop.clockRate * 10000); // 10 seconds
    ok = ck(cudaGetLastError(), "doorbell launch");
    volatile Doorbell* poll = host;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    for (uint32_t i = 1; ok && i <= 512; ++i) {
        while (poll->seq < i && poll->result == 0 && std::chrono::steady_clock::now() < deadline)
            strata::platform::cpu_relax();
        strata::platform::mapped_memory_acquire();
        if (poll->seq < i || poll->result != 0 || poll->gpu_value != i * 3) { ok = false; break; }
        poll->cpu_value = i * 7;
        strata::platform::mapped_memory_release();
        poll->flag = i;
    }
    ok = ck(cudaDeviceSynchronize(), "doorbell sync") && ok && host->result == 0;
    cudaFreeHost(host);
    if (!ok) { std::fprintf(stderr, "mapped doorbell exchange failed\n"); return 1; }
    std::puts("512 mapped CPU/GPU publication round trips: PASS");

    CUdevice dev;
    int supported = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS)
        return 1;
    if (!supported) { std::puts("VMM unsupported: fallback required"); return 0; }
    CUmemAllocationProp alloc{};
    alloc.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc.location.id = dev;
    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &alloc, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) return 1;
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool reserved = false, created = false, mapped = false;
    reserved = cuMemAddressReserve(&address, granularity, granularity, 0, 0) == CUDA_SUCCESS;
    created = reserved && cuMemCreate(&handle, granularity, &alloc, 0) == CUDA_SUCCESS;
    mapped = created && cuMemMap(address, granularity, 0, handle, 0) == CUDA_SUCCESS;
    CUmemAccessDesc access{};
    access.location = alloc.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ok = mapped && cuMemSetAccess(address, granularity, &access, 1) == CUDA_SUCCESS;
    int value = 41;
    if (ok) {
        int* p = (int*) address;
        ok = ck(cudaMemcpy(p, &value, sizeof value, cudaMemcpyHostToDevice), "VMM upload");
        if (ok) increment<<<1, 1>>>(p);
        ok = ck(cudaGetLastError(), "VMM launch") && ck(cudaDeviceSynchronize(), "VMM sync") && ok;
        ok = ck(cudaMemcpy(&value, p, sizeof value, cudaMemcpyDeviceToHost), "VMM download") && ok && value == 42;
    }
    if (mapped) cuMemUnmap(address, granularity);
    if (created) cuMemRelease(handle);
    if (reserved) cuMemAddressFree(address, granularity);
    if (!ok) { std::fprintf(stderr, "VMM allocation/map/access test failed\n"); return 1; }
    std::printf("VMM reserve/create/map/access/unmap (%zu bytes): PASS\n", granularity);
    return 0;
}
