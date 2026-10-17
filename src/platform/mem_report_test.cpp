// src/platform/mem_report_test.cpp - Q36 plan Phase 0 (docs/Q36_MEMORY_COMPACTION.md): the two parsers on fixtures
// (including an older kernel's shorter file), the delta arithmetic, the unit formatting and the absent-row rule.
// Fixtures only, so no GPU, no model - and the live section is skipped off Linux.
#include "strata/platform/mem_report.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

namespace P = strata::platform;

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "<-- WRONG");
    if (!ok) ++g_fail;
}

bool has(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

/// The report as text, so the printing can be checked the way the engine will emit it.
std::string report_text(const P::MemSnapshot& now, const P::MemSnapshot* prev) {
    std::FILE* f = std::tmpfile();
    if (f == nullptr) return {};
    P::mem_report_at(f, now, prev, "fixture");
    std::fflush(f);
    std::rewind(f);
    std::string out;
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

// The target PC's shape before the arena is built: /proc/meminfo has no AnonHugePages row (an older kernel's
// file), /proc/vmstat has no compact_daemon_* and no thp_file_alloc rows.
const char* kMeminfoBefore = R"(MemTotal:       65123512 kB
MemFree:        24242908 kB
MemAvailable:   52120628 kB
Buffers:          406872 kB
Cached:         29411220 kB
Mlocked:             104 kB
Unevictable:         104 kB
Shmem:           188300 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Hugepagesize:       2048 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Surp:        0
)";

const char* kVmstatBefore = R"(nr_free_pages 6060744
pgmigrate_success 2373960
pgmigrate_fail 71744161
compact_stall 146
compact_fail 52
compact_success 344
compact_migrate_scanned 108515522
compact_free_scanned 90629242
compact_isolated 76632858
thp_fault_alloc 1731877
thp_file_mapped 868
thp_split_page 45
)";

// The same host with the 18.75 GiB arena pinned, 4,000,000 kB of free memory gone, the compact counters moved on
// and two rows the "before" kernel did not have.
const char* kMeminfoAfter = R"(MemTotal:       65123512 kB
MemFree:        20242908 kB
MemAvailable:   48120628 kB
Buffers:          406872 kB
Cached:         29411220 kB
Mlocked:        19660800 kB
Unevictable:    19660800 kB
Shmem:           188300 kB
AnonHugePages:         0 kB
SwapTotal:             0 kB
SwapFree:              0 kB
Hugepagesize:       2048 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Surp:        0
)";

const char* kVmstatAfter = R"(nr_free_pages 5060744
pgmigrate_success 2374000
pgmigrate_fail 71800000
compact_stall 150
compact_fail 52
compact_success 94
compact_migrate_scanned 108515600
compact_free_scanned 90629242
compact_isolated 76632858
compact_daemon_wake 2200
compact_daemon_migrate_scanned 107916788
compact_daemon_free_scanned 90449289
thp_fault_alloc 1733000
thp_file_alloc 0
thp_file_mapped 868
)";

P::MemSnapshot snapshot_of(const char* meminfo, const char* vmstat) {
    P::MemSnapshot s;
    s.valid = P::parse_meminfo(meminfo, s.mem);
    P::parse_vmstat(vmstat, s.vm);
    s.thp_enabled = "never";
    s.thp_defrag = "never";
    return s;
}

void parsers() {
    std::printf("parsers\n");
    P::MemSnapshot before = snapshot_of(kMeminfoBefore, kVmstatBefore);
    check(before.valid, "meminfo of the before fixture parses");
    check(P::meminfo_get(before.mem, "MemTotal") == 65123512ull * 1024, "MemTotal's kB become bytes");
    check(P::meminfo_get(before.mem, "Mlocked") == 104ull * 1024, "Mlocked's kB become bytes");
    check(P::meminfo_get(before.mem, "Hugepagesize") == 2048ull * 1024, "Hugepagesize's kB become bytes");
    check(P::meminfo_get(before.mem, "HugePages_Total") == 0, "HugePages_Total stays a page count");
    check(P::meminfo_has(before.mem, "Mlocked") && !P::meminfo_has(before.mem, "AnonHugePages"),
          "a row the kernel does not export is absent, not zero");
    check(P::meminfo_get(before.mem, "AnonHugePages") == 0, "an absent row reads as 0 to meminfo_get");

    check(P::vmstat_get(before.vm, "compact_stall") == 146, "compact_stall parses");
    check(P::vmstat_get(before.vm, "pgmigrate_fail") == 71744161, "pgmigrate_fail parses");
    check(P::vmstat_has(before.vm, "compact_stall") && !P::vmstat_has(before.vm, "compact_daemon_wake"),
          "a counter the kernel does not export is absent, not zero");
    check(P::vmstat_get(before.vm, "compact_daemon_wake") == 0, "an absent counter reads as 0 to vmstat_get");
    check(P::vmstat_get(before.vm, "thp_file_mapped") == 868, "the last row of the file parses");
    check(!P::vmstat_has(before.vm, "no_such_counter"), "an unknown name is absent");

    // an older kernel's shorter file: fewer rows, and an empty one is not a parse
    P::MemInfo one;
    check(P::parse_meminfo("MemTotal: 100 kB\n", one) && P::meminfo_get(one, "MemTotal") == 102400 &&
              !P::meminfo_has(one, "Cached"),
          "a one-row meminfo parses, the rest stay absent");
    P::Vmstat none;
    check(!P::parse_vmstat("", none) && !P::parse_vmstat(nullptr, none), "an empty vmstat is not a parse");
    check(!P::parse_meminfo("", one) && !P::parse_meminfo(nullptr, one), "an empty meminfo is not a parse");
    check(P::vmstat_get(none, "compact_stall") == 0 && !P::vmstat_has(none, "compact_stall"),
          "an unparsed vmstat answers without a row");
}

void formatting() {
    std::printf("formatting\n");
    check(P::format_bytes(0) == "0 B", "0 bytes");
    check(P::format_bytes(104) == "104 B", "bytes stay bytes");
    check(P::format_bytes(1024) == "1.0 KiB", "1 KiB");
    check(P::format_bytes(4096) == "4.0 KiB", "4 KiB");
    check(P::format_bytes(3ull << 20) == "3.0 MiB", "3 MiB");
    check(P::format_bytes(1ull << 30) == "1.00 GiB", "1 GiB");
    check(P::format_bytes(20132659200ull) == "18.75 GiB", "the 18.75 GiB arena");
    check(P::format_bytes(65123512ull * 1024) == "62.11 GiB", "62 GB of RAM");
    check(P::parse_thp_choice("always madvise [never]\n") == "never", "the bracketed THP word");
    check(P::parse_thp_choice("always [madvise] never\n") == "madvise", "the other THP choice");
    check(P::parse_thp_choice("").empty(), "no THP text, no word");
}

void deltas() {
    std::printf("deltas\n");
    const P::MemSnapshot before = snapshot_of(kMeminfoBefore, kVmstatBefore);
    const P::MemSnapshot after = snapshot_of(kMeminfoAfter, kVmstatAfter);

    const std::string first = report_text(before, nullptr);
    check(has(first, "strata-mem: fixture: pid "), "the report says when and which process");
    check(has(first, "meminfo: MemTotal 62.11 GiB"), "absolute values are printed");
    check(has(first, "Mlocked 104.0 KiB"), "Mlocked is printed");
    check(has(first, "vmstat: nr_free_pages 6060744"), "the vmstat counters are printed");
    check(has(first, "AnonHugePages -"), "a row the kernel does not export is printed as -");
    check(has(first, "compact_daemon_wake -"), "a counter the kernel does not export is printed as -");
    check(has(first, "thp: enabled never, defrag never"), "the THP mode is printed");
    check(!has(first, " (+") && !has(first, " (-"), "no previous snapshot, no deltas");

    const std::string second = report_text(after, &before);
    check(has(second, "Mlocked 18.75 GiB (+18.75 GiB)"), "a pinned arena shows as its delta: +18.75 GiB");
    check(has(second, "MemFree 19.31 GiB (-3.81 GiB)"), "freed memory shows a negative delta (-3.81 GiB)");
    check(has(second, "Cached 28.05 GiB (+0 B)"), "an unmoved byte row shows (+0 B)");
    check(has(second, "HugePages_Total 0 (+0)"), "an unmoved count shows (+0)");
    check(has(second, "compact_success 94 (-250)"), "a counter that went backwards shows -250");
    check(has(second, "thp_fault_alloc 1733000 (+1123)"), "a counter that grew shows +1123");
    check(has(second, "compact_daemon_wake 2200") && !has(second, "compact_daemon_wake 2200 ("),
          "a row the previous kernel did not have gets no delta");
    check(has(second, "AnonHugePages 0 B"), "a row the previous kernel did not have is printed, without a delta");
    check(has(second, "thp_split_page -"), "a row the current kernel does not export stays -, no delta");

    // a counter resetting (a reboot between two reports) is negative, not 2^64
    P::MemSnapshot reboot = after;
    reboot.vm.pgmigrate_fail = 10;
    check(has(report_text(reboot, &after), "pgmigrate_fail 10 (-71799990)"), "a reset counter reads negative");

    const std::string unreadable = report_text(P::MemSnapshot{}, nullptr);
    check(has(unreadable, "not readable"), "an unreadable /proc is one line, not a crash");
}

void live() {
#if defined(__linux__)
    std::printf("live (this host)\n");
    const P::MemSnapshot s = P::mem_snapshot();
    check(s.valid, "mem_snapshot() reads this host's /proc/meminfo");
    check(P::meminfo_has(s.mem, "MemTotal") && P::vmstat_has(s.vm, "compact_stall"),
          "the live snapshot has the rows this plan needs");
    P::mem_report(stdout, nullptr, "live");
#else
    std::printf("live: skipped, not Linux\n");
#endif
}

void switch_state() {
#if !defined(_WIN32)
    std::printf("the switch\n");
    ::unsetenv("STRATA_MEM_REPORT");
    check(!P::mem_report_enabled(), "unset means off");
    ::setenv("STRATA_MEM_REPORT", "0", 1);
    check(!P::mem_report_enabled(), "0 means off");
    ::setenv("STRATA_MEM_REPORT", "", 1);
    check(!P::mem_report_enabled(), "empty means off");
    ::setenv("STRATA_MEM_REPORT", "1", 1);
    check(P::mem_report_enabled(), "1 means on");
    ::unsetenv("STRATA_MEM_REPORT");
#endif
}

}  // namespace

int main() {
    parsers();
    formatting();
    deltas();
    switch_state();
    live();
    std::printf("mem_report_test: %s\n", g_fail == 0 ? "OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
