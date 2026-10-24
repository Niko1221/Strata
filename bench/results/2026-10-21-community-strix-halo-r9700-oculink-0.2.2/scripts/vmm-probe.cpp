// vmm-probe.cpp N [MiB [HOST_GIB]] - the HIP virtual memory path Strata's --kv-grow uses (src/core/vmm.cpp): hipMemCreate
// of N chunks of MiB (default 2) on device 0, then reserve / map / set access / memset / unmap / free.  Prints where it
// fails.  HOST_GIB > 0 first page-locks that much host memory with hipHostRegister(PORTABLE), like Strata's expert arena.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const int maxn = argc > 1 ? std::atoi(argv[1]) : 64;
    const size_t chunk = (size_t) (argc > 2 ? std::atoll(argv[2]) : 2) << 20;
    const size_t host = (size_t) (argc > 3 ? std::atoll(argv[3]) : 0) << 30;
    hipSetDevice(0);
    if (host) {
        void* hp = std::aligned_alloc(2 << 20, host);
        if (!hp) { std::printf("host alloc of %zu GiB failed\n", host >> 30); return 1; }
        std::memset(hp, 1, host);
        hipError_t r = hipHostRegister(hp, host, hipHostRegisterPortable);
        std::printf("hipHostRegister PORTABLE %zu GiB: %s\n", host >> 30, hipGetErrorString(r));
    }
    hipDeviceProp_t pr;
    hipGetDeviceProperties(&pr, 0);
    std::printf("device 0: %s %s\n", pr.name, pr.gcnArchName);
    int vmm = 0;
    hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0);
    size_t fr = 0, tot = 0;
    hipMemGetInfo(&fr, &tot);
    std::printf("VirtualMemoryManagementSupported %d, free %.2f of %.2f GiB\n", vmm, fr / 1073741824.0, tot / 1073741824.0);
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = 0;
    size_t gmin = 0, grec = 0;
    hipError_t e = hipMemGetAllocationGranularity(&gmin, &p, hipMemAllocationGranularityMinimum);
    hipMemGetAllocationGranularity(&grec, &p, hipMemAllocationGranularityRecommended);
    std::printf("granularity minimum %zu (%s), recommended %zu\n", gmin, hipGetErrorString(e), grec);
    std::vector<hipMemGenericAllocationHandle_t> hs;
    for (int i = 0; i < maxn; ++i) {
        hipMemGenericAllocationHandle_t h = nullptr;
        e = hipMemCreate(&h, chunk, &p, 0);
        if (e != hipSuccess) {
            std::printf("hipMemCreate #%d failed: %s\n", i, hipGetErrorString(e));
            (void) hipGetLastError();
            break;
        }
        hs.push_back(h);
    }
    std::printf("hipMemCreate: %zu of %d chunks of %zu MiB\n", hs.size(), maxn, chunk >> 20);
    if (!hs.empty()) {
        void* base = nullptr;
        e = hipMemAddressReserve(&base, hs.size() * chunk, 0, nullptr, 0);
        std::printf("hipMemAddressReserve %zu MiB: %s\n", (hs.size() * chunk) >> 20, hipGetErrorString(e));
        size_t mapped = 0;
        if (e == hipSuccess) {
            for (size_t i = 0; i < hs.size(); ++i, ++mapped) {
                e = hipMemMap((char*) base + i * chunk, chunk, 0, hs[i], 0);
                if (e != hipSuccess) { std::printf("hipMemMap #%zu failed: %s\n", i, hipGetErrorString(e)); break; }
            }
            hipMemAccessDesc d{};
            d.location = p.location;
            d.flags = hipMemAccessFlagsProtReadWrite;
            if (mapped) {
                e = hipMemSetAccess(base, mapped * chunk, &d, 1);
                std::printf("hipMemSetAccess: %s\n", hipGetErrorString(e));
                e = hipMemset(base, 0x5a, mapped * chunk);
                hipError_t s = hipDeviceSynchronize();
                std::printf("hipMemset %zu MiB: %s, sync %s\n", (mapped * chunk) >> 20, hipGetErrorString(e), hipGetErrorString(s));
            }
            for (size_t i = 0; i < mapped; ++i) hipMemUnmap((char*) base + i * chunk, chunk);
            hipMemAddressFree(base, hs.size() * chunk);
        }
    }
    for (auto h : hs) hipMemRelease(h);
    hipMemGetInfo(&fr, &tot);
    std::printf("free after %.2f GiB\n", fr / 1073741824.0);
    return 0;
}
