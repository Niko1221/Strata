// Linux HIP interposition covers both Strata and dynamically linked BLAS
// allocations. Runtime/driver resources not allocated through these APIs are
// covered by a separate reserve, not misreported as tracked allocations.
#include <hip/hip_runtime_api.h>
#include "strata/platform/device_budget.hpp"

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {
constexpr uint64_t MiB = 1ull << 20, page = 65536;
template <typename Fn> Fn next(const char* name) {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) { std::fprintf(stderr, "HIP budget: cannot resolve %s\n", name); std::exit(2); }
    return reinterpret_cast<Fn>(symbol);
}
uint64_t setting(const char* name, uint64_t fallback) {
    const char* text = std::getenv(name);
    if (!text) return fallback;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (!*text || *text == '-' || *end || value > 10240) {
        std::fprintf(stderr, "HIP budget: invalid %s (expected 0..10240 MiB)\n", name);
        std::exit(2);
    }
    return value * MiB;
}
struct Allocation { uint64_t bytes; int device; };
struct State {
    const uint64_t limit = setting("STRATA_VRAM_BUDGET_MIB", 10240 * MiB);
    const uint64_t runtime = setting("STRATA_VRAM_RUNTIME_RESERVE_MIB", 1024 * MiB);
    const uint64_t slack = setting("STRATA_VRAM_SLACK_MIB", 256 * MiB);
    std::mutex mutex;
    std::array<strata::platform::DeviceBudget, 64> devices;
    std::unordered_map<void*, Allocation> allocations;
    State() {
        if (runtime < 1024 * MiB || slack < 256 * MiB || runtime + slack >= limit) {
            std::fprintf(stderr, "HIP budget: preserve >=1024 MiB runtime reserve and >=256 MiB slack\n");
            std::exit(2);
        }
        const uint64_t tracked = limit ? limit - runtime - slack : 0;
        for (auto& device : devices) device = strata::platform::DeviceBudget(tracked);
        if (limit) std::fprintf(stderr, "HIP budget: %llu MiB ceiling, %llu MiB runtime reserve, %llu MiB slack; "
                                      "explicit allocations capped at %llu MiB\n",
            (unsigned long long) (limit / MiB), (unsigned long long) (runtime / MiB),
            (unsigned long long) (slack / MiB), (unsigned long long) (tracked / MiB));
    }
    void report() {
        if (limit) for (size_t i = 0; i < devices.size(); ++i)
            if (devices[i].peak()) std::fprintf(stderr, "HIP budget: device %zu tracked peak %llu bytes\n",
                                               i, (unsigned long long) devices[i].peak());
    }
};
State& state() {
    // Keep the ledger alive through HIP/library static destructors.
    static State* instance = [] {
        auto* ledger = new State;
        std::atexit([] { state().report(); });
        return ledger;
    }();
    return *instance;
}
using InfoFn = hipError_t (*)(size_t*, size_t*);
InfoFn raw_info() { static auto fn = next<InfoFn>("hipMemGetInfo"); return fn; }

template <typename Allocate>
hipError_t allocate(void** pointer, size_t bytes, Allocate real) {
    State& s = state();
    if (!s.limit) return real(pointer, bytes);
    if (!pointer) return hipErrorInvalidValue;
    if (bytes > std::numeric_limits<uint64_t>::max() - (page - 1)) return hipErrorOutOfMemory;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= (int) s.devices.size())
        return hipErrorInvalidDevice;
    const uint64_t reservation = (bytes + page - 1) & ~(page - 1);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        size_t free = 0, total = 0;
        const auto info = raw_info()(&free, &total);
        if (info != hipSuccess) return info;
        if (free < s.slack || reservation > free - s.slack || !s.devices[device].reserve(reservation)) {
            std::fprintf(stderr, "HIP budget: refusing %llu bytes before allocation on device %d "
                                 "(%llu bytes available within ceiling)\n",
                         (unsigned long long) reservation, device,
                         (unsigned long long) s.devices[device].available());
            return hipErrorOutOfMemory;
        }
    }
    // Do not hold the ledger lock inside HIP: the runtime may call an exported
    // allocation API itself. Its nested reservations must also be admitted.
    const auto result = real(pointer, bytes);
    std::lock_guard<std::mutex> lock(s.mutex);
    if (result == hipSuccess && *pointer) {
        if (!s.allocations.emplace(*pointer, Allocation{reservation, device}).second)
            s.devices[device].release(reservation); // a nested API already registered this pointer
    }
    else s.devices[device].release(reservation);
    if (std::getenv("STRATA_VRAM_TRACE"))
        std::fprintf(stderr, "HIP budget allocation: device=%d requested=%zu reserved=%llu "
                             "used=%llu peak=%llu success=%d\n", device, bytes,
                     (unsigned long long) reservation, (unsigned long long) s.devices[device].used(),
                     (unsigned long long) s.devices[device].peak(), (int) (result == hipSuccess));
    return result;
}
}  // namespace

extern "C" hipError_t hipMalloc(void** pointer, size_t bytes) {
    using Fn = hipError_t (*)(void**, size_t);
    static auto real = next<Fn>("hipMalloc");
    return allocate(pointer, bytes, real);
}

extern "C" hipError_t hipExtMallocWithFlags(void** pointer, size_t bytes, unsigned int flags) {
    using Fn = hipError_t (*)(void**, size_t, unsigned int);
    static auto real = next<Fn>("hipExtMallocWithFlags");
    return allocate(pointer, bytes, [=](void** p, size_t n) { return real(p, n, flags); });
}

extern "C" hipError_t hipFree(void* pointer) {
    using Fn = hipError_t (*)(void*);
    static auto real = next<Fn>("hipFree");
    const auto result = real(pointer);
    State& s = state();
    if (s.limit && result == hipSuccess) {
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.allocations.find(pointer);
        if (found != s.allocations.end()) {
            s.devices[found->second.device].release(found->second.bytes);
            s.allocations.erase(found);
        }
    }
    return result;
}

extern "C" hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    const auto result = raw_info()(free, total);
    State& s = state();
    if (result != hipSuccess || !s.limit) return result;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= (int) s.devices.size())
        return hipErrorInvalidDevice;
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint64_t physical = *free > s.slack ? *free - s.slack : 0;
    *free = (size_t) std::min<uint64_t>(physical, s.devices[device].available());
    return result;
}

// Managed allocations can migrate independently of this reservation ledger.
// This deployment does not use them; fail before allocation if budgeting is on.
extern "C" hipError_t hipMallocManaged(void** pointer, size_t bytes, unsigned int flags) {
    if (state().limit) return hipErrorNotSupported;
    using Fn = hipError_t (*)(void**, size_t, unsigned int);
    static auto real = next<Fn>("hipMallocManaged");
    return real(pointer, bytes, flags);
}

// Pool allocations retain device memory after free and cannot be charged using
// this pointer ledger. Reject them before allocation instead of bypassing it.
extern "C" hipError_t hipMallocAsync(void** pointer, size_t bytes, hipStream_t stream) {
    if (state().limit) return hipErrorNotSupported;
    using Fn = hipError_t (*)(void**, size_t, hipStream_t);
    static auto real = next<Fn>("hipMallocAsync");
    return real(pointer, bytes, stream);
}
extern "C" hipError_t hipMallocFromPoolAsync(void** pointer, size_t bytes, hipMemPool_t pool,
                                            hipStream_t stream) {
    if (state().limit) return hipErrorNotSupported;
    using Fn = hipError_t (*)(void**, size_t, hipMemPool_t, hipStream_t);
    static auto real = next<Fn>("hipMallocFromPoolAsync");
    return real(pointer, bytes, pool, stream);
}
