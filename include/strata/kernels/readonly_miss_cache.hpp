#pragma once

#include <cstdint>

namespace strata::kernels {

constexpr int kMissCacheMaxWays = 16;
constexpr int kMissCacheMaxGroups = 64;

/// An immutable secondary copy of weights. It never owns the authoritative
/// copy and never writes an eviction back. One bank per layer, one stream.
/// tags starts at -1; ages, clock and counters start at zero.
struct ReadonlyMissCacheBank {
    uint8_t* data = nullptr;
    int32_t* tags = nullptr;
    uint64_t* ages = nullptr;
    uint64_t* clock = nullptr;
    uint64_t* counters = nullptr;  // completed groups, hits, fills, bypasses
    int ways = 0;
    int64_t stride = 0;
};

/// Device scratch for one group. Keep it alive until its consumers finish.
struct ReadonlyMissCachePlan {
    unsigned long long target[kMissCacheMaxGroups];
    int32_t slot[kMissCacheMaxGroups];
    int32_t fill[kMissCacheMaxGroups];
    int32_t expert[kMissCacheMaxGroups];
    int32_t count;
    // Optional compact-fill path: stable group IDs for misses only. Never
    // changes output pointer order or cache admission/publication.
    int32_t fill_group[kMissCacheMaxGroups];
    int32_t fill_count;
};

/// Preconditions: distinct expert IDs within a group, 0 <= *count <= cap <= 64,
/// 1 <= ways <= 16, 0 < bytes <= bank.stride, both sizes multiples of 16.
/// The graph plan's first dst entry identifies each group's expert in ids.
/// All calls, and every consumer of the returned pointers, use the same stream.
/// The three stages are public for cancellation/fill-publication component tests.
void plan_readonly_misses(const int32_t* count, const int32_t* starts,
                         const int32_t* dst, const int32_t* ids,
                         uint8_t* staging, int64_t bytes, int cap,
                         ReadonlyMissCacheBank bank, ReadonlyMissCachePlan* plan,
                         void* stream);
void fill_readonly_misses(const unsigned long long* src, int64_t bytes,
                         const ReadonlyMissCachePlan* plan, void* stream);
void publish_readonly_misses(unsigned long long* ptrs, ReadonlyMissCacheBank bank,
                            const ReadonlyMissCachePlan* plan, void* stream);
void fetch_readonly_misses(unsigned long long* ptrs, const int32_t* count,
                          const int32_t* starts, const int32_t* dst,
                          const int32_t* ids, uint8_t* staging, int64_t bytes,
                          int cap, ReadonlyMissCacheBank bank,
                          ReadonlyMissCachePlan* plan, void* stream);

}  // namespace strata::kernels
