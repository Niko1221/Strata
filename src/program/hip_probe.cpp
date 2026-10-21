// strata --hip-probe. Device 0 must stay on the nvcc host. The first HIP
// ordinal must run the hipcc host: hybrid_marker() returns 1000 plus that
// copy's device cc only when the HIP compile of moe_fused.cu ran.
#include <cuda_runtime.h>

#include <cstdio>

#include "strata/kernels/gfx_arch.hpp"

namespace strata::prefill::fused {
int hybrid_marker();
}

extern "C" int strata_hybrid_device_count();
extern "C" int strata_hybrid_is_hip_device(int ordinal);
extern "C" const char* strata_hybrid_arch(int ordinal);

int strata_hip_probe() {
    cudaError_t e = cudaSetDevice(0);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata hip probe: cudaSetDevice(0): %s\n", cudaGetErrorString(e));
        return 1;
    }
    const int cuda_marker = strata::prefill::fused::hybrid_marker();
    std::fprintf(stderr, "strata hip probe: device 0 marker %d\n", cuda_marker);
    if (cuda_marker >= 1000 || strata_hybrid_is_hip_device(0)) {
        std::fprintf(stderr, "strata hip probe: device 0 ran the HIP host\n");
        return 2;
    }

    int hip_ord = -1;
    const int n = strata_hybrid_device_count();
    if (n < 0) {
        std::fprintf(stderr, "strata hip probe: %d HIP devices; this build supports 8\n", -n);
        return 1;
    }
    for (int i = 1; i < n; ++i) {
        if (strata_hybrid_is_hip_device(i)) {
            hip_ord = i;
            break;
        }
    }
    if (hip_ord < 0) {
        std::fprintf(stderr, "strata hip probe: no HIP device\n");
        return 1;
    }
    e = cudaSetDevice(hip_ord);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata hip probe: cudaSetDevice(%d): %s\n", hip_ord, cudaGetErrorString(e));
        return 1;
    }
    const int hip_marker = strata::prefill::fused::hybrid_marker();
    std::fprintf(stderr, "strata hip probe: device %d marker %d\n", hip_ord, hip_marker);
    if (hip_marker < 1000) {
        std::fprintf(stderr, "strata hip probe: device %d ran the CUDA host\n", hip_ord);
        return 3;
    }
    const int cc = hip_marker - 1000;
    std::fprintf(stderr, "strata hip probe: HIP host ran, fused cc %d\n", cc);
    // dev_info() reports the gfx11 WMMA expert kernel. gfx12 and the other
    // AMD parts have no such kernel; the HIP host running is the check.
    const char* arch = strata_hybrid_arch(hip_ord);
    if (strata::kernels::gfx_arch_is_gfx11_wmma(arch) && cc < 75) {
        std::fprintf(stderr, "strata hip probe: fused expert kernel is not available\n");
        return 4;
    }
    return 0;
}
