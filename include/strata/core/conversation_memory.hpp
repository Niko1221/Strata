#pragma once

#include <cstddef>
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

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

// --conversation-cache-park: where a parking capture goes.  ram is the behaviour before the flag; disk captures,
// hands the copy to the asynchronous writer and frees the RAM copy at once, so the RAM cache keeps nothing for it;
// auto resolves once at startup (conversation_park_fits_ram) against the RAM the arena and the model left free.
enum class ConversationParkTarget { ram, disk, automatic };

inline const char* conversation_park_target_name(ConversationParkTarget target) {
    switch (target) {
        case ConversationParkTarget::ram: return "ram";
        case ConversationParkTarget::disk: return "disk";
        case ConversationParkTarget::automatic: return "auto";
    }
    return "ram";
}

// What a parking capture is given when the physical-RAM admission gate is read.  ram: the parking floor holds and
// the capture is parked in RAM as before.  disk: the parking floor does not hold but the transient floor of a
// disk-only park does, and the disk tier is on: the capture is spilled and NO RAM entry is kept.  discard: neither
// holds - the refusal that, before this flag, dropped the conversation unmirrored even though the disk tier was
// right there (the hole the upstream #1529 "takes what RAM cannot hold" closes).  In one place so the rule is
// testable without the engine.
enum class ConversationParkPlacement { ram, disk, discard };

inline ConversationParkPlacement conversation_park_placement(std::optional<uint64_t> available,
                                                             uint64_t allocation, uint64_t floor,
                                                             uint64_t disk_floor, bool disk_target,
                                                             bool disk_capable) {
    if (!disk_target && conversation_memory_admit(available, allocation, floor))
        return ConversationParkPlacement::ram;
    if (disk_capable && conversation_memory_admit(available, allocation, disk_floor))
        return ConversationParkPlacement::disk;
    return ConversationParkPlacement::discard;
}

// --conversation-cache-park auto: park in RAM while the free RAM can hold one snapshot per cache slot beside the
// floor; below that the parking goes to disk.  Called once, after the arena and the model are resident, so
// `available` is already net of the arena (the spec's `arena + slots x ~2.5 GiB + piso`).
inline bool conversation_park_fits_ram(std::optional<uint64_t> available, size_t slots,
                                       uint64_t snapshot_bytes, uint64_t floor) {
    if (!available || *available < floor) return false;
    const uint64_t room = *available - floor;
    return slots == 0 || snapshot_bytes <= room / slots;
}

// The READ/save side of the same rule: conversation_park_placement with the destination fixed to disk.  An
// operation whose copy is destined to disk and whose RAM is transient - a conversation disk hit, a segment-tier
// hit, a system-prompt-cache hit, a streamed session restore, a session the API saves to disk - is admitted under
// the TRANSIENT floor when the parking floor does not hold; only when neither holds is it refused (discard).  The
// same fallback the park takes, so a copy already on disk is never ignored for RAM it would not keep.
inline ConversationParkPlacement conversation_disk_placement(std::optional<uint64_t> available, uint64_t allocation,
                                                             uint64_t floor, uint64_t disk_floor) {
    return conversation_park_placement(available, allocation, floor, disk_floor, /*disk_target=*/true,
                                       /*disk_capable=*/true);
}

} // namespace strata::core
