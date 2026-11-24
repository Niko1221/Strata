#pragma once
#include <cstdint>
#include <string>

namespace strata::hy4_flash {
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
    uint64_t file = 0;   ///< 149: memory.stat "file" (active + inactive page cache); 0 when not reported
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming the group's clean page cache, as
/// MemAvailable counts it on a bare host: every clean file page (active and inactive: memory.stat "file", else
/// inactive_file), less a low-watermark share the kernel keeps (min(half the clean cache, limit / 64)).  A process
/// whose own reads (a GPU cache fill, a read-ahead) left gigabytes of active cache in its group was judged short of
/// RAM before (149: 1.2 GiB "available" of a 32 GiB group, the file tier then switched to unbuffered reads).
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable (Windows: the available physical memory), lowered
/// to the room under the tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group
/// and its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
    uint64_t commit = ~uint64_t{0};   ///< available Windows commit capacity; ~0 when not reported
};

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; the
/// defaults are the real files).  cgroup v2 as before (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

}
