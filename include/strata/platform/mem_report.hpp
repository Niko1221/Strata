// include/strata/platform/mem_report.hpp - Q36 plan Phase 0 (docs/Q36_MEMORY_COMPACTION.md): the memory and
// compaction counters every later phase is judged on, read and printed by the engine itself instead of from another
// shell.  Read-only: nothing here locks, allocates or changes a setting.
//
// Linux reads /proc/meminfo, /proc/vmstat and /sys/kernel/mm/transparent_hugepage/{enabled,defrag}.  Everywhere
// else the snapshot comes back invalid, and a row a kernel does not export is reported as absent rather than as a
// zero - "94 compact_success" on a kernel that never exported `compact_success` would be a number that is not there.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::platform {

/// /proc/meminfo in bytes.  The HugePages_* rows are page counts, exactly as the kernel writes them, and
/// `hugepage_size` (Hugepagesize) is the size of one of those pages.  `present` carries one bit per row in the
/// order of the parser's table - `meminfo_has` asks it by name - so the report can print `-` for a row this kernel
/// does not have.
struct MemInfo {
    uint64_t mem_total = 0;
    uint64_t mem_free = 0;
    uint64_t mem_available = 0;
    uint64_t buffers = 0;
    uint64_t cached = 0;
    uint64_t mlocked = 0;
    uint64_t unevictable = 0;
    uint64_t shmem = 0;
    uint64_t anon_hugepages = 0;
    uint64_t swap_total = 0;
    uint64_t swap_free = 0;
    uint64_t hugepage_size = 0;
    uint64_t hugepages_total = 0;
    uint64_t hugepages_free = 0;
    uint64_t hugepages_surp = 0;
    uint32_t present = 0;
};

/// /proc/vmstat: the counters this plan is judged on (the compactor's work, its failures, page migration, and the
/// transparent-huge-page allocations that the arena and the mapping provoke).  Same presence rule as MemInfo.
struct Vmstat {
    uint64_t nr_free_pages = 0;
    uint64_t pgmigrate_success = 0;
    uint64_t pgmigrate_fail = 0;
    uint64_t compact_stall = 0;
    uint64_t compact_fail = 0;
    uint64_t compact_success = 0;
    uint64_t compact_migrate_scanned = 0;
    uint64_t compact_free_scanned = 0;
    uint64_t compact_isolated = 0;
    uint64_t compact_daemon_wake = 0;
    uint64_t compact_daemon_migrate_scanned = 0;
    uint64_t compact_daemon_free_scanned = 0;
    uint64_t thp_fault_alloc = 0;
    uint64_t thp_file_alloc = 0;
    uint64_t thp_file_mapped = 0;
    uint64_t thp_split_page = 0;
    uint32_t present = 0;
};

/// One point in time: the two counter files and the THP mode.  `valid` is false when /proc/meminfo could not be
/// read at all (not Linux), and the report says so instead of printing a page of zeroes.
struct MemSnapshot {
    bool valid = false;
    MemInfo mem;
    Vmstat vm;
    std::string thp_enabled;   ///< the active word of .../transparent_hugepage/enabled, "" when unreadable
    std::string thp_defrag;    ///< the same for .../transparent_hugepage/defrag
};

/// Read one now.
MemSnapshot mem_snapshot();

/// Parse the text of /proc/meminfo (or /proc/vmstat).  False when nothing in it was recognised.  Kept separate from
/// the file reads so a test can feed a fixture, including an older kernel's shorter file.
bool parse_meminfo(const char* text, MemInfo& out);
bool parse_vmstat(const char* text, Vmstat& out);

/// One row of a parsed file by its /proc name (no colon).  `*_has` is false when the kernel does not export it, and
/// `*_get` then returns 0.  meminfo_get is in bytes for the kB rows and a count for the HugePages_* rows.
bool meminfo_has(const MemInfo& info, const char* name);
uint64_t meminfo_get(const MemInfo& info, const char* name);
bool vmstat_has(const Vmstat& vm, const char* name);
uint64_t vmstat_get(const Vmstat& vm, const char* name);

/// "never" for the text "always madvise [never]": the bracketed word of a sysfs choice file, or "".
std::string parse_thp_choice(const std::string& text);

/// Bytes for a human: GiB with two decimals from 1 GiB up, then MiB, then KiB, then plain bytes.
std::string format_bytes(uint64_t bytes);

/// Print `when`, every counter, and - when `prev` is not null - the delta since it.  A row the kernel does not
/// export is printed as `-`; a counter that moved backwards shows a negative delta rather than being hidden.
/// This reads a snapshot and prints it; `mem_report_at` prints one already in hand.
void mem_report(std::FILE* f, const MemSnapshot* prev, const char* when);

/// The same for a snapshot the caller already has, so a test can print fixtures (and check the delta arithmetic)
/// without a /proc of its own.
void mem_report_at(std::FILE* f, const MemSnapshot& now, const MemSnapshot* prev, const char* when);

/// Whether STRATA_MEM_REPORT asks for the report: unset, empty and "0" all mean off, as elsewhere in the engine.
bool mem_report_enabled();

}  // namespace strata::platform
