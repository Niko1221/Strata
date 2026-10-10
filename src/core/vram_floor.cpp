#include "strata/core/vram_floor.hpp"

#include "strata/program/vram_cap.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace strata::core {
namespace {

double g_frac = 1.0;
std::atomic<int> g_armed{0};

struct Once {
    std::mutex mu;
    char labels[8][80]{};
    int n = 0;
    bool claim(const char* label) {
        std::lock_guard<std::mutex> lock(mu);
        for (int i = 0; i < n; ++i)
            if (std::strcmp(labels[i], label) == 0) return false;
        if (n >= 8 || label == nullptr) return false;
        std::snprintf(labels[n], sizeof labels[n], "%s", label);
        ++n;
        return true;
    }
};

Once& once() {
    static Once o;
    return o;
}

}  // namespace

void vram_floor_arm(double fraction) {
    g_frac = fraction;
    g_armed.store(fraction < 1.0 ? 1 : 0, std::memory_order_release);
}

bool vram_floor_armed() {
    return g_armed.load(std::memory_order_acquire) != 0;
}

bool vram_floor_allow(const char* where, std::string& err) {
    if (!vram_floor_armed()) return true;
    int dev = 0;
    cudaGetDevice(&dev);
    size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess || tb == 0) {
        std::fprintf(stderr, "strata generate: --vram-frac: CUDA%d VRAM telemetry failed (%s)\n", dev, where);
        err = std::string("--vram-frac: cannot read VRAM (") + where + ")";
        return false;
    }
    const int64_t floor = strata::program::vram_cap::floor_mib((uint64_t) tb, g_frac);
    if ((uint64_t) fb < ((uint64_t) floor << 20)) {
        std::fprintf(stderr, "strata generate: --vram-frac %.6g: CUDA%d has %llu MiB free, needs %lld MiB "
                             "kept free (%s). Use a smaller context/chunk or a larger fraction; cap not relaxed\n",
                     g_frac, dev, (unsigned long long) (fb >> 20), (long long) floor, where);
        std::fprintf(stderr, "strata vram: REFUSED %s free_mib=%llu floor_mib=%lld\n",
                     where, (unsigned long long) (fb >> 20), (long long) floor);
        std::fflush(stderr);
        err = std::string("free VRAM is under the floor (") + where + "); cap not relaxed";
        return false;
    }
    return true;
}

void vram_floor_log_once(const char* label) {
    if (!vram_floor_armed() || label == nullptr || !once().claim(label)) return;
    int dev = 0;
    cudaGetDevice(&dev);
    size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) {
        std::fprintf(stderr, "strata vram: %s free_mib=unavailable cuda=%d\n", label, dev);
    } else {
        std::fprintf(stderr, "strata vram: %s free_mib=%llu\n", label, (unsigned long long) (fb >> 20));
    }
    std::fflush(stderr);
}

}  // namespace strata::core
