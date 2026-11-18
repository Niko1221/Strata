// include/strata/platform/numa_probe.hpp - diag:numa-probe (STRATA_NUMA_AUDIT)
//
// Physical NUMA-node census of an allocation: which nodes the pages of [p, p+bytes) actually
// landed on.  Diagnostics only - every call site is env-gated and nothing here belongs on a
// hot path.  Windows reads the working set (QueryWorkingSetEx, taken from psapi at runtime:
// no new link dependency).  Pages the OS cannot place (not committed, not resident, no NUMA)
// are counted as '?'.  Non-Windows builds report everything '?' - the port point is
// get_mempolicy(2) page by page on Linux.
//
// One line per audited allocation:
//   numa-audit: arena[0] dev 0: 234.5 MiB -> n1:100% (117/117 pages)
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#endif

namespace strata::platform {

inline bool numa_audit_on() {
    return std::getenv("STRATA_NUMA_AUDIT") != nullptr;
}

struct NumaCensus {
    int nodes[16] = {};     // pages sampled per node (nodes >= 16 fold into nodes[15])
    int unknown = 0;
    int sampled = 0;
};

/// Node histogram of [p, p+bytes), sampled every `step` bytes (default 2 MiB), at most
/// `max_samples` pages.  All counts land in `sampled` = nodes[*] + unknown.
inline NumaCensus numa_page_census(const void* p, uint64_t bytes, uint64_t step = 2ull << 20,
                                   int max_samples = 1024) {
    NumaCensus c;
    if (p == nullptr || bytes == 0) return c;
    if (step == 0) step = 4096;
#if defined(_WIN32)
    static const auto qwse = [] {
        HMODULE ps = LoadLibraryExW(L"psapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        return ps ? (BOOL(WINAPI*)(HANDLE, PVOID, DWORD)) GetProcAddress(ps, "QueryWorkingSetEx") : nullptr;
    }();
    if (qwse == nullptr) { c.unknown = 1; c.sampled = 1; return c; }
    for (uint64_t off = 0; off < bytes && c.sampled < max_samples; off += step, ++c.sampled) {
        PSAPI_WORKING_SET_EX_INFORMATION w{};
        w.VirtualAddress = (PVOID) ((const uint8_t*) p + off);
        if (qwse(GetCurrentProcess(), &w, sizeof(w)) && w.VirtualAttributes.Valid)
            ++c.nodes[w.VirtualAttributes.Node < 16 ? w.VirtualAttributes.Node : 15];
        else
            ++c.unknown;
    }
#else
    c.unknown = 1;   // port point: Linux asks get_mempolicy(2) MPOL_F_NODE|MPOL_F_ADDR per page
    c.sampled = 1;
#endif
    return c;
}

/// Compact "n0:60% n1:40% ?:2" rendering of a census; "" when nothing was sampled.
inline std::string numa_census_str(const NumaCensus& c) {
    if (c.sampled <= 0) return "";
    std::string s;
    char b[32];
    for (int n = 0; n < 16; ++n)
        if (c.nodes[n]) {
            std::snprintf(b, sizeof b, "n%d:%.0f%% ", n, 100.0 * c.nodes[n] / c.sampled);
            s += b;
        }
    if (c.unknown) {
        std::snprintf(b, sizeof b, "?:%d", c.unknown);
        s += b;
    }
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

/// The one-line audit print every call site shares; no output unless STRATA_NUMA_AUDIT is set.
/// The sample step grows with the range so ~768 pages span even a 60 GiB arena.
inline void numa_audit(const char* what, int dev, const void* p, uint64_t bytes) {
    if (!numa_audit_on() || p == nullptr || bytes == 0) return;
    const uint64_t step = std::max<uint64_t>(2ull << 20, bytes / 768);
    const NumaCensus c = numa_page_census(p, bytes, step);
    std::fprintf(stderr, "numa-audit: %s dev %d: %.1f MiB -> %s (%d pages)\n",
                 what, dev, (double) bytes / 1048576.0,
                 numa_census_str(c).c_str(), c.sampled);
    std::fflush(stderr);
}

}  // namespace strata::platform
