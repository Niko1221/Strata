// src/platform/mem_report.cpp - Q36 plan Phase 0 (docs/Q36_MEMORY_COMPACTION.md).
//
// The compactor's work, the pinned arena and the mapped GGUF were all read from another shell while the engine
// ran.  This puts the same numbers, absolute and as a delta, into the engine's own log, so the user's run and the
// A/B harness carry them.  Read-only, Linux-only data; on any other host mem_snapshot() is invalid and
// mem_report() says so in one line rather than failing.
#include "strata/platform/mem_report.hpp"

#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <process.h>
#define STRATA_MEM_GETPID _getpid
#else
#include <unistd.h>
#define STRATA_MEM_GETPID getpid
#endif

namespace strata::platform {
namespace {

constexpr int kGiB = 30;
constexpr int kMiB = 20;
constexpr int kKiB = 10;

/// The whole file, or "" when it cannot be read (no /proc, or no permission).
std::string read_file(const char* path) {
    std::string out;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return out;
    char buf[1 << 16];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

/// kB (the unit /proc/meminfo writes) to bytes, saturating instead of wrapping on a nonsense value.
uint64_t kb_to_bytes(uint64_t kb) { return kb > (~0ull >> kKiB) ? ~0ull : kb << kKiB; }

/// The signed difference, or 0 when either side is absent.  Unsigned counters are compared as magnitudes first so
/// that a counter resetting (a reboot between two reports) reads as a negative delta, not a 2^64 one.
long long signed_delta(uint64_t now, bool has_now, uint64_t prev, bool has_prev) {
    if (!has_now || !has_prev) return 0;
    if (now >= prev) return (long long) (now - prev);
    return -(long long) (prev - now);
}

struct MemRow {
    const char* name;         ///< the /proc/meminfo key, without its colon
    uint64_t MemInfo::*field;
    bool in_kb;               ///< false for the HugePages_* rows, which are page counts
    bool as_bytes;            ///< print with format_bytes: the byte rows, not the huge-page counts
};

/// The rows the report prints, in the order it prints them.  The index in this table is the bit in MemInfo::present.
const MemRow kMemRows[] = {
    {"MemTotal", &MemInfo::mem_total, true, true},
    {"MemFree", &MemInfo::mem_free, true, true},
    {"MemAvailable", &MemInfo::mem_available, true, true},
    {"Buffers", &MemInfo::buffers, true, true},
    {"Cached", &MemInfo::cached, true, true},
    {"Mlocked", &MemInfo::mlocked, true, true},
    {"Unevictable", &MemInfo::unevictable, true, true},
    {"Shmem", &MemInfo::shmem, true, true},
    {"AnonHugePages", &MemInfo::anon_hugepages, true, true},
    {"SwapTotal", &MemInfo::swap_total, true, true},
    {"SwapFree", &MemInfo::swap_free, true, true},
    {"Hugepagesize", &MemInfo::hugepage_size, true, true},
    {"HugePages_Total", &MemInfo::hugepages_total, false, false},
    {"HugePages_Free", &MemInfo::hugepages_free, false, false},
    {"HugePages_Surp", &MemInfo::hugepages_surp, false, false},
};
constexpr int kMemRowCount = (int) (sizeof kMemRows / sizeof kMemRows[0]);
static_assert(kMemRowCount <= 32, "MemInfo::present is a uint32_t");

struct VmRow {
    const char* name;         ///< the /proc/vmstat key
    uint64_t Vmstat::*field;
};

/// The counters the report prints, in order.  The index in this table is the bit in Vmstat::present.
const VmRow kVmRows[] = {
    {"nr_free_pages", &Vmstat::nr_free_pages},
    {"pgmigrate_success", &Vmstat::pgmigrate_success},
    {"pgmigrate_fail", &Vmstat::pgmigrate_fail},
    {"compact_stall", &Vmstat::compact_stall},
    {"compact_fail", &Vmstat::compact_fail},
    {"compact_success", &Vmstat::compact_success},
    {"compact_migrate_scanned", &Vmstat::compact_migrate_scanned},
    {"compact_free_scanned", &Vmstat::compact_free_scanned},
    {"compact_isolated", &Vmstat::compact_isolated},
    {"compact_daemon_wake", &Vmstat::compact_daemon_wake},
    {"compact_daemon_migrate_scanned", &Vmstat::compact_daemon_migrate_scanned},
    {"compact_daemon_free_scanned", &Vmstat::compact_daemon_free_scanned},
    {"thp_fault_alloc", &Vmstat::thp_fault_alloc},
    {"thp_file_alloc", &Vmstat::thp_file_alloc},
    {"thp_file_mapped", &Vmstat::thp_file_mapped},
    {"thp_split_page", &Vmstat::thp_split_page},
};
constexpr int kVmRowCount = (int) (sizeof kVmRows / sizeof kVmRows[0]);
static_assert(kVmRowCount <= 32, "Vmstat::present is a uint32_t");

/// The row's name as it is written in the file, and its place in `name`.. True when `name` names a row.
int mem_row_of(const char* name) {
    for (int i = 0; i < kMemRowCount; ++i)
        if (std::strcmp(name, kMemRows[i].name) == 0) return i;
    return -1;
}
int vm_row_of(const char* name) {
    for (int i = 0; i < kVmRowCount; ++i)
        if (std::strcmp(name, kVmRows[i].name) == 0) return i;
    return -1;
}

/// The first integer on the line, skipping spaces and tabs; false when there is none.
bool first_number(const char* p, const char* end, uint64_t& out) {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    char* stop = nullptr;
    const unsigned long long v = std::strtoull(p, &stop, 10);
    if (stop == p || stop > end) return false;
    out = (uint64_t) v;
    return true;
}

/// The delta as " (+1.50 GiB)" / " (-256.0 MiB)", in the same unit the value itself is printed in.
std::string bytes_delta(uint64_t now, bool has_now, uint64_t prev, bool has_prev) {
    if (!has_now || !has_prev) return {};
    const long long d = signed_delta(now, has_now, prev, has_prev);
    const uint64_t mag = d < 0 ? (uint64_t) (-(d + 1)) + 1 : (uint64_t) d;
    return std::string(" (") + (d < 0 ? "-" : "+") + format_bytes(mag) + ")";
}

/// The delta as " (+250)" for a counter that is already a plain number.
std::string count_delta(uint64_t now, bool has_now, uint64_t prev, bool has_prev) {
    if (!has_now || !has_prev) return {};
    char buf[32];
    std::snprintf(buf, sizeof buf, " (%+lld)", signed_delta(now, has_now, prev, has_prev));
    return buf;
}

const char* or_dash(const std::string& s) { return s.empty() ? "-" : s.c_str(); }

}  // namespace

std::string parse_thp_choice(const std::string& text) {
    const size_t open = text.find('[');
    if (open == std::string::npos) return {};
    const size_t close = text.find(']', open + 1);
    if (close == std::string::npos) return {};
    return text.substr(open + 1, close - open - 1);
}

std::string format_bytes(uint64_t bytes) {
    char buf[64];
    if (bytes >= (1ull << kGiB)) std::snprintf(buf, sizeof buf, "%.2f GiB", (double) bytes / (double) (1ull << kGiB));
    else if (bytes >= (1ull << kMiB)) std::snprintf(buf, sizeof buf, "%.1f MiB", (double) bytes / (double) (1ull << kMiB));
    else if (bytes >= (1ull << kKiB)) std::snprintf(buf, sizeof buf, "%.1f KiB", (double) bytes / (double) (1ull << kKiB));
    else std::snprintf(buf, sizeof buf, "%llu B", (unsigned long long) bytes);
    return buf;
}

bool parse_meminfo(const char* text, MemInfo& out) {
    out = MemInfo{};
    if (text == nullptr) return false;
    for (const char* line = text; *line != 0;) {
        const char* eol = std::strchr(line, '\n');
        const char* end = eol != nullptr ? eol : line + std::strlen(line);
        const char* colon = (const char*) std::memchr(line, ':', (size_t) (end - line));
        if (colon != nullptr) {
            const size_t klen = (size_t) (colon - line);
            for (int i = 0; i < kMemRowCount; ++i) {
                if (std::strlen(kMemRows[i].name) != klen || std::strncmp(line, kMemRows[i].name, klen) != 0) continue;
                uint64_t v = 0;
                if (first_number(colon + 1, end, v)) {
                    out.*(kMemRows[i].field) = kMemRows[i].in_kb ? kb_to_bytes(v) : v;
                    out.present |= 1u << i;
                }
                break;
            }
        }
        if (eol == nullptr) break;
        line = eol + 1;
    }
    return out.present != 0;
}

bool parse_vmstat(const char* text, Vmstat& out) {
    out = Vmstat{};
    if (text == nullptr) return false;
    for (const char* line = text; *line != 0;) {
        const char* eol = std::strchr(line, '\n');
        const char* end = eol != nullptr ? eol : line + std::strlen(line);
        const char* sp = line;
        while (sp < end && *sp != ' ' && *sp != '\t') ++sp;
        const size_t klen = (size_t) (sp - line);
        for (int i = 0; i < kVmRowCount; ++i) {
            if (std::strlen(kVmRows[i].name) != klen || std::strncmp(line, kVmRows[i].name, klen) != 0) continue;
            uint64_t v = 0;
            if (first_number(sp, end, v)) {
                out.*(kVmRows[i].field) = v;
                out.present |= 1u << i;
            }
            break;
        }
        if (eol == nullptr) break;
        line = eol + 1;
    }
    return out.present != 0;
}

bool meminfo_has(const MemInfo& info, const char* name) {
    const int i = mem_row_of(name);
    return i >= 0 && (info.present & (1u << i)) != 0;
}

uint64_t meminfo_get(const MemInfo& info, const char* name) {
    const int i = mem_row_of(name);
    return i >= 0 ? info.*(kMemRows[i].field) : 0;
}

bool vmstat_has(const Vmstat& vm, const char* name) {
    const int i = vm_row_of(name);
    return i >= 0 && (vm.present & (1u << i)) != 0;
}

uint64_t vmstat_get(const Vmstat& vm, const char* name) {
    const int i = vm_row_of(name);
    return i >= 0 ? vm.*(kVmRows[i].field) : 0;
}

MemSnapshot mem_snapshot() {
    MemSnapshot s;
    std::string meminfo = read_file("/proc/meminfo");
    s.valid = parse_meminfo(meminfo.c_str(), s.mem);
    parse_vmstat(read_file("/proc/vmstat").c_str(), s.vm);
    s.thp_enabled = parse_thp_choice(read_file("/sys/kernel/mm/transparent_hugepage/enabled"));
    s.thp_defrag = parse_thp_choice(read_file("/sys/kernel/mm/transparent_hugepage/defrag"));
    return s;
}

void mem_report(std::FILE* f, const MemSnapshot* prev, const char* when) {
    const MemSnapshot now = mem_snapshot();
    mem_report_at(f, now, prev, when);
}

void mem_report_at(std::FILE* f, const MemSnapshot& snap, const MemSnapshot* prev, const char* when) {
    if (f == nullptr) return;
    if (when == nullptr) when = "";
    std::fprintf(f, "strata-mem: %s: pid %ld\n", when, (long) STRATA_MEM_GETPID());
    if (!snap.valid) {
        std::fprintf(f, "strata-mem:   /proc/meminfo is not readable (Linux only): no memory report\n");
        return;
    }
    std::fprintf(f, "strata-mem:   meminfo:");
    for (int i = 0; i < kMemRowCount; ++i) {
        const MemRow& r = kMemRows[i];
        const bool has = (snap.mem.present & (1u << i)) != 0;
        if (!r.as_bytes) continue;
        if (!has) {
            std::fprintf(f, " %s -", r.name);
            continue;
        }
        const uint64_t v = snap.mem.*(r.field);
        const std::string d = prev != nullptr ? bytes_delta(v, has, prev->mem.*(r.field), (prev->mem.present & (1u << i)) != 0) : std::string();
        std::fprintf(f, " %s %s%s", r.name, format_bytes(v).c_str(), d.c_str());
    }
    std::fprintf(f, "\n");
    std::fprintf(f, "strata-mem:   huge pages:");
    for (int i = 0; i < kMemRowCount; ++i) {
        const MemRow& r = kMemRows[i];
        const bool has = (snap.mem.present & (1u << i)) != 0;
        if (r.as_bytes) continue;
        if (!has) {
            std::fprintf(f, " %s -", r.name);
            continue;
        }
        const uint64_t v = snap.mem.*(r.field);
        const std::string d = prev != nullptr ? count_delta(v, has, prev->mem.*(r.field), (prev->mem.present & (1u << i)) != 0) : std::string();
        std::fprintf(f, " %s %llu%s", r.name, (unsigned long long) v, d.c_str());
    }
    std::fprintf(f, "\n");
    std::fprintf(f, "strata-mem:   vmstat:");
    for (int i = 0; i < kVmRowCount; ++i) {
        const VmRow& r = kVmRows[i];
        const bool has = (snap.vm.present & (1u << i)) != 0;
        if (!has) {
            std::fprintf(f, " %s -", r.name);
            continue;
        }
        const uint64_t v = snap.vm.*(r.field);
        const std::string d = prev != nullptr ? count_delta(v, has, prev->vm.*(r.field), (prev->vm.present & (1u << i)) != 0) : std::string();
        std::fprintf(f, " %s %llu%s", r.name, (unsigned long long) v, d.c_str());
    }
    std::fprintf(f, "\n");
    std::fprintf(f, "strata-mem:   thp: enabled %s, defrag %s\n", or_dash(snap.thp_enabled), or_dash(snap.thp_defrag));
}

bool mem_report_enabled() {
    const char* e = std::getenv("STRATA_MEM_REPORT");
    return e != nullptr && e[0] != 0 && e[0] != '0';
}

}  // namespace strata::platform
