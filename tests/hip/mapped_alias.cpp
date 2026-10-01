// Is a hipHostGetDevicePointer alias host-visible on this stack?
//
// tests/hip/handoff.cpp fails on gfx1201/Windows with "separate copy/ring timeout": it does a
// hipMemcpyAsync(..., hipMemcpyDeviceToDevice) into a hipHostGetDevicePointer alias, then waits for
// a doorbell that copy should have published. This test pins down the property that decides it, so
// that failure has a diagnosed cause instead of a shrug.
//
// On Linux the alias is a real device pointer into pinned host memory, a D2D copy lands, the host
// sees it, and the verdict is "host-visible". On Windows the runtime hands back the HOST pointer
// (this test prints both addresses when they are equal), so the "device" address is not device
// memory at all: passing it to a D2D copy is a host address in a device pointer, which faults.
// That is also why src/core/native_head.cpp keeps the token embedding in VRAM on Windows rather
// than using the mapped alias, and why it does so silently rather than fatally - the mapping check
// there is a null test, and this probe shows a non-null-but-wrong pointer is the real hazard.
//
// Reporting, not gating: a Windows host is EXPECTED to fail the host-visibility checks below, so
// this test exits 0 whenever it ran at all. It only fails when it could not run (no device).

#include "strata/kernels/elementwise.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

namespace {

int not_held = 0;

void report(const char* what, bool holds, const char* detail = "") {
    std::printf("  %-44s %s%s%s\n", what, holds ? "holds" : "DOES NOT HOLD", detail[0] ? " - " : "", detail);
    if (!holds) ++not_held;
}

}  // namespace

int main() {
    int runtime = 0;
    if (hipRuntimeGetVersion(&runtime) != hipSuccess) return 1;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) {
        std::printf("no HIP device\n");
        return 77;  // ctest "skipped"
    }
    std::printf("HIP runtime %d, %d device(s)\n", runtime, count);

    // Enough elements for the float4 path copy_from_mapped uses.
    const int64_t n = 4096;
    const size_t bytes = size_t(n) * sizeof(float);
    // Braces, not parens: `std::vector<float> pattern(size_t(n))` is a function declaration.
    std::vector<float> pattern(size_t(n), 0.0f);
    for (int64_t i = 0; i < n; ++i) pattern[size_t(i)] = float(i + 1);

    float* host = nullptr;
    if (hipHostMalloc(reinterpret_cast<void**>(&host), bytes, hipHostMallocMapped) != hipSuccess) return 1;
    void* alias = nullptr;
    if (hipHostGetDevicePointer(&alias, host, 0) != hipSuccess) {
        std::printf("hipHostGetDevicePointer failed\n");
        hipHostFree(host);
        return 1;
    }

    // THE finding. On a stack with a real mapping these differ; when they are equal, every
    // "device" pointer derived from this allocation is host memory.
    const bool distinct = alias != reinterpret_cast<void*>(host);
    std::printf("\n  host  %p\n  alias %p\n  -> %s\n\n", static_cast<void*>(host), alias,
                distinct ? "a distinct device address (a real mapping)"
                         : "THE HOST POINTER: there is no device mapping for this allocation");
    report("the alias is a distinct device address", distinct);

    float* device = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&device), bytes) != hipSuccess) {
        hipHostFree(host);
        return 1;
    }
    if (hipMemcpy(device, pattern.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(device);
        hipHostFree(host);
        return 1;
    }

    // What tests/hip/handoff.cpp attempts. Deliberately NOT performed when the alias is the host
    // pointer: a DeviceToDevice copy with a host address in the destination faults the process, and
    // crashing the test run tells the maintainer less than this line does.
    if (distinct) {
        const hipError_t copy = hipMemcpy(alias, device, bytes, hipMemcpyDeviceToDevice);
        report("a D2D copy into the alias succeeds", copy == hipSuccess);
        if (hipDeviceSynchronize() == hipSuccess) {
            bool host_updated = true;
            for (int64_t i = 0; i < n; ++i)
                if (host[i] != pattern[size_t(i)]) { host_updated = false; break; }
            report("the host sees a D2D copy into the alias", host_updated);

            // And a kernel reading it - the other shape handoff uses.
            float* out = nullptr;
            if (hipMalloc(reinterpret_cast<void**>(&out), bytes) == hipSuccess) {
                strata::kernels::copy_from_mapped(out, reinterpret_cast<const float*>(alias), n, nullptr);
                if (hipDeviceSynchronize() == hipSuccess) {
                    bool read_ok = true;
                    for (int64_t i = 0; i < n; ++i)
                        if (out[i] != pattern[size_t(i)]) { read_ok = false; break; }
                    report("a kernel reads the alias correctly", read_ok);
                }
                hipFree(out);
            }
        }
    } else {
        std::printf("  skipped the D2D copy and the mapped read: with alias == host both would take a\n"
                    "  host address in a device pointer, which faults rather than reporting.\n");
    }

    std::printf("\n%s\n", distinct ? "VERDICT: mapped host memory is usable on this stack."
                                    : "VERDICT: hipHostGetDevicePointer returns the HOST pointer here, so a\n"
                                      "        mapped host allocation is NOT addressable by the GPU. That is why\n"
                                      "        tests/hip/handoff times out on gfx1201/Windows, and why\n"
                                      "        native_head.cpp keeps the token embedding in device memory there\n"
                                      "        instead of trusting a non-null alias.");
    std::printf("\n%d check(s) did not hold (a Windows host is expected to fail them)\n", not_held);

    hipFree(device);
    hipHostFree(host);
    return 0;
}
