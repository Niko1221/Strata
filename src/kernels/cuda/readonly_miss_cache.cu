#include "strata/kernels/readonly_miss_cache.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

bool compact_fill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_Q8_COMPACT_MISS_FILL");
        const bool on = value && std::strcmp(value, "1") == 0;
        if (on) std::fprintf(stderr, "strata readonly miss cache: compact fill enabled; miss payload ranges only\n");
        return on;
    }();
    return enabled;
}

void check(const char* where) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void plan_kernel(const int32_t* count, const int32_t* starts,
                            const int32_t* dst, const int32_t* ids,
                            uint8_t* staging, int64_t bytes,
                            ReadonlyMissCacheBank bank, ReadonlyMissCachePlan* plan,
                            bool compact) {
    // Reserve ALL hits before admitting any miss: an early miss must not evict
    // a later member of this same group. Excess misses use ordinary staging.
    bool used[kMissCacheMaxWays] = {};
    const int n = *count;
    plan->count = n;
    plan->fill_count = 0;
    for (int q = 0; q < n; ++q) {
        const int expert = ids[dst[starts[q]]];
        plan->expert[q] = expert;
        plan->slot[q] = -1;
        plan->fill[q] = 1;
        plan->target[q] = (unsigned long long)(staging + (int64_t)q * bytes);
        for (int s = 0; s < bank.ways; ++s) {
            if (bank.tags[s] != expert) continue;
            used[s] = true;
            plan->slot[q] = s;
            plan->fill[q] = 0;
            plan->target[q] = (unsigned long long)(bank.data + (int64_t)s * bank.stride);
            bank.ages[s] = ++*bank.clock;
            break;
        }
    }
    for (int q = 0; q < n; ++q) {
        if (!plan->fill[q]) continue;
        int victim = -1;
        for (int s = 0; s < bank.ways; ++s) {
            if (used[s]) continue;
            if (bank.tags[s] < 0) { victim = s; break; }
            if (victim < 0 || bank.ages[s] < bank.ages[victim]) victim = s;
        }
        if (victim < 0) continue;
        used[victim] = true;
        plan->slot[q] = victim;
        plan->target[q] = (unsigned long long)(bank.data + (int64_t)victim * bank.stride);
        // Do not publish the new key here. A plan abandoned before its complete
        // copy leaves an invalid slot, not a hit on partially copied weights.
        bank.tags[victim] = -1;
        bank.ages[victim] = ++*bank.clock;
    }
    if (compact) {
        for (int q = 0; q < n; ++q)
            if (plan->fill[q]) plan->fill_group[plan->fill_count++] = q;
    }
}

__global__ void fill_kernel(const unsigned long long* src, long long per,
                            const ReadonlyMissCachePlan* plan) {
    const long long total = (long long)plan->count * per;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long q = i / per, off = i - q * per;
        if (plan->fill[q]) ((uint4*)plan->target[q])[off] = ((const uint4*)src[q])[off];
    }
}

__global__ void compact_fill_kernel(const unsigned long long* src, long long per,
                                    const ReadonlyMissCachePlan* plan) {
    const long long total = (long long)plan->fill_count * per;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long group = i / per, off = i - group * per;
        const int q = plan->fill_group[group];
        ((uint4*)plan->target[q])[off] = ((const uint4*)src[q])[off];
    }
}

__global__ void publish_kernel(unsigned long long* ptrs, ReadonlyMissCacheBank bank,
                               const ReadonlyMissCachePlan* plan) {
    // Ordered after the entire fill kernel, not just one block's portion.
    for (int q = 0; q < plan->count; ++q) {
        ptrs[q] = plan->target[q];
        if (plan->fill[q] && plan->slot[q] >= 0)
            bank.tags[plan->slot[q]] = plan->expert[q];
        ++bank.counters[0];
        if (!plan->fill[q]) ++bank.counters[1];
        else ++bank.counters[2];
        if (plan->slot[q] < 0) ++bank.counters[3];
    }
}

}  // namespace

void plan_readonly_misses(const int32_t* count, const int32_t* starts,
                         const int32_t* dst, const int32_t* ids,
                         uint8_t* staging, int64_t bytes, int cap,
                         ReadonlyMissCacheBank bank, ReadonlyMissCachePlan* plan,
                         void* stream) {
    if (cap < 1 || cap > kMissCacheMaxGroups || bank.ways < 1 || bank.ways > kMissCacheMaxWays ||
        bytes <= 0 || bytes > bank.stride || bytes % 16 || bank.stride % 16) {
        std::fprintf(stderr, "readonly miss cache: invalid geometry\n");
        std::exit(1);
    }
    plan_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(count, starts, dst, ids, staging, bytes, bank, plan,
                                               compact_fill_enabled());
    check("readonly miss cache plan");
}

void fill_readonly_misses(const unsigned long long* src, int64_t bytes,
                         const ReadonlyMissCachePlan* plan, void* stream) {
    if (compact_fill_enabled())
        compact_fill_kernel<<<48 * 8, 256, 0, (cudaStream_t)stream>>>(src, bytes / 16, plan);
    else
        fill_kernel<<<48 * 8, 256, 0, (cudaStream_t)stream>>>(src, bytes / 16, plan);
    check("readonly miss cache fill");
}

void publish_readonly_misses(unsigned long long* ptrs, ReadonlyMissCacheBank bank,
                            const ReadonlyMissCachePlan* plan, void* stream) {
    publish_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(ptrs, bank, plan);
    check("readonly miss cache publish");
}

void fetch_readonly_misses(unsigned long long* ptrs, const int32_t* count,
                          const int32_t* starts, const int32_t* dst,
                          const int32_t* ids, uint8_t* staging, int64_t bytes,
                          int cap, ReadonlyMissCacheBank bank,
                          ReadonlyMissCachePlan* plan, void* stream) {
    plan_readonly_misses(count, starts, dst, ids, staging, bytes, cap, bank, plan, stream);
    fill_readonly_misses(ptrs, bytes, plan, stream);
    publish_readonly_misses(ptrs, bank, plan, stream);
}

}  // namespace strata::kernels
