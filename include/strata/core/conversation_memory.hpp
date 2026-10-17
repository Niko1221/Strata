#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// The RAM a runtime allocation (parking a conversation, saving or restoring a session file) may still take.
//
// available_host_bytes() is the smallest of
//   * MemAvailable (Windows: the available physical memory);
//   * the room under every memory limit of this process's cgroup and of each ancestor visible under
//     /sys/fs/cgroup (cgroup v2: memory.max and memory.high, each minus usage; cgroup v1:
//     memory.limit_in_bytes minus memory.usage_in_bytes);
//   * the operator's own cap, when one is set (--memory-limit-mib / STRATA_MEMORY_LIMIT_MIB): that many MiB minus
//     what the container uses now - the top visible cgroup's memory.current, or MemTotal - MemAvailable when the
//     cgroup files cannot be read.  For a container whose real limit sits on a parent cgroup it cannot see.
//
// Usage is memory.current as the kernel reports it, less one reclaimable kind: the clean inactive file cache of the
// same group (memory.stat inactive_file, v1 total_inactive_file, less dirty and writeback pages, never more than
// memory.current), which the kernel gives back at memory.high or the limit before it kills anything.  Nothing else is
// credited: active_file holds mlocked pages, and shmem is the pinned complements and the KV pool.  A memory.stat that
// is missing or unparsable gives no credit (it does not make the sample unknown).
// Pinned, locked and shared memory stays charged in full: a hard-limit kill is worse than a skipped park.
//
// Unknown telemetry is deliberately distinct from a measured zero.  A limit file that exists but cannot be parsed
// makes the sample unknown (fail closed); a file that is absent (no such controller, not mounted) is ignored.
// Not a reservation: other writers can take the room after the sample.

enum class MemorySource { meminfo, cgroup, flag };

const char* memory_source_name(MemorySource source);

/// One sample.  `available` is the smallest term; `source`, `limit` and `current` describe the term that was the
/// smallest (meminfo: MemTotal and MemTotal - MemAvailable; cgroup: that group's limit and memory.current; flag: the
/// operator's cap and the usage it was measured against).  `mem_available` is MemAvailable by itself (0 if unknown).
struct HostMemoryReading {
    uint64_t available = 0;
    MemorySource source = MemorySource::meminfo;
    uint64_t limit = 0;
    uint64_t current = 0;        ///< memory.current (meminfo: MemTotal - MemAvailable), before the credit
    uint64_t credit = 0;         ///< the clean inactive file cache counted as reclaimable (0 for meminfo)
    uint64_t mem_available = 0;
};

/// The files the sampler reads; the defaults are the real ones (tests point them at a fixture directory).
struct MemoryProbePaths {
    std::string meminfo = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
};

/// The operator's cap in bytes for the container/cgroup this engine runs in (0 = none).  Set once at start.
void set_memory_limit_bytes(uint64_t bytes);
uint64_t memory_limit_bytes();

/// Linux: reads `paths`.  `explicit_limit` 0 = no operator cap.  nullopt when the RAM cannot be determined.
std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit);

/// The real files (Windows: GlobalMemoryStatusEx) and the cap set by set_memory_limit_bytes().
std::optional<HostMemoryReading> sample_host_memory();

std::optional<uint64_t> available_host_bytes();

/// What the machine says about an allocation of `bytes` right now. Called from a bad_alloc handler, after the
/// failed call and before anything else allocates, to tell a size the machine cannot commit from one that was
/// refused by the moment. Reserve first (address space only), then commit: reserving without committing says
/// the commit limit is the cause. `probed` is false for a size of 0 - a park that reuses all of its K/V asks
/// for nothing new, and there is nothing to ask the machine about.
struct AllocationProbe {
    bool probed = false;
    bool reserved = false;
    bool committed = false;
    unsigned long error = 0;        ///< GetLastError() / errno of the failing step
};
AllocationProbe conversation_memory_probe(size_t bytes);

inline const char* conversation_memory_probe_text(const AllocationProbe& p) {
    if (!p.probed) return "nothing new was requested, so the size is not the cause";
    if (p.reserved && !p.committed) return "the address space reserves but will not commit: commit is exhausted";
    if (!p.reserved) return "the address space itself cannot be reserved";
    return "the same size allocates now, so the failure was transient or came from elsewhere";
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

/// What one park's eviction round decided, so the refusal line can say why it stopped.
struct EvictionRound {
    bool admitted = false;       ///< the last reading left room for the request
    size_t evicted = 0;
    bool hopeless = false;       ///< the request exceeds everything the cache could ever free
    bool stalled = false;        ///< freeing one entry did not make the short resource larger
    size_t freeable = 0;         ///< the bytes evict_oldest() could have taken
    uint64_t short_by = 0;       ///< how far the last reading fell short
};

/// Free parked conversations until `sample()` reports room for `additional` beside `floor`, and stop early when
/// continuing cannot help. `sample()` returns the short resource (free RAM or free commit, whichever the guard
/// reads) and may keep the full reading for the caller's report; the answer is always decided by a reading, never
/// by the arithmetic alone. Three bounds the entry count alone does not give: only what evict_oldest() can take is
/// payment, since the retained K/V and any pinned shared prefix sit in the cache but cannot be spent; a park that
/// reuses its K/V asks for nothing the cache can give; and if freeing one entry left the resource no larger, the
/// rest will not either - on Windows a pinned working set and WDDM device allocations keep the commit charge flat
/// while parked conversations are dropped. Templated on the cache so the decision is testable against the real
/// one without a model.
template <typename Cache, typename Sample>
inline EvictionRound evict_for_admission(Cache& conversations, size_t additional,
                                         uint64_t floor, Sample&& sample) {
    EvictionRound round;
    std::optional<uint64_t> available = sample();
    if (conversation_memory_admit(available, additional, floor)) { round.admitted = true; return round; }
    if (available) round.short_by = additional + floor - *available;
    round.freeable = conversations.evictable_bytes();
    round.hopeless = additional != 0 && additional > round.freeable;
    if (round.hopeless || additional == 0) return round;
    uint64_t best = available ? *available : 0;
    for (;;) {
        const size_t evictable_now = conversations.evictable_bytes();
        // For ConversationCache the last test cannot be what ends the round - evictable_bytes() > 0 means an
        // unpinned entry exists and evict_oldest() takes exactly such an entry. It is what terminates the round
        // for a cache type whose two predicates disagree, which is why it stays.
        if (evictable_now == 0 || additional > evictable_now || !conversations.evict_oldest()) break;
        ++round.evicted;
        available = sample();
        if (conversation_memory_admit(available, additional, floor)) { round.admitted = true; return round; }
        // This reading is after the eviction, so comparing it with the one before says whether freeing a
        // conversation gave the resource back.
        if (available && *available <= best) { round.stalled = true; break; }
        if (available) { round.short_by = additional + floor - *available; best = *available; }
    }
    return round;
}

} // namespace strata::core
