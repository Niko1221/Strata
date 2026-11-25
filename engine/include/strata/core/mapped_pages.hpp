// Give the pages of a copied-out range of a read-only file mapping back to the OS (strata-inkling GGUF loader).
//
// The GGUF is mmapped read-only and every tensor is COPIED out of the mapping (trunk arena, VRAM upload buffer, host expert arena). The mapped
// pages stay in the process's resident set after the copy, so a 15.7 GB UD-IQ4_XS sat next to its copies: peak RSS 32-36 GiB with 11-13 GiB
// of experts in RAM. Dropping the pages of a range that is already copied removes them from the process (they stay in the OS page cache,
// reclaimable, and fault back in if the range is read again), so nothing is lost but the double count.
// STRATA_INK_KEEP_MAPPED_PAGES=1 turns it off.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace strata::core {

inline bool keep_mapped_pages() {
    static const bool keep = [] { const char* e = std::getenv("STRATA_INK_KEEP_MAPPED_PAGES"); return e && e[0] == '1'; }();
    return keep;
}

// Only whole pages inside [p, p + n) are released (rounded inward, so a neighbour's page that is still to be read is never dropped).
// Returns the number of bytes handed back (0 when off, too small, or the OS call failed).
inline size_t release_mapped_pages(const void* p, size_t n) noexcept {
    if (!p || n == 0 || keep_mapped_pages()) return 0;
    constexpr uintptr_t kPage = 4096;
    const uintptr_t b = (reinterpret_cast<uintptr_t>(p) + kPage - 1) & ~(kPage - 1);
    const uintptr_t e = (reinterpret_cast<uintptr_t>(p) + n) & ~(kPage - 1);
    if (e <= b) return 0;
#ifdef _WIN32
    // VirtualUnlock on pages that are not locked trims them from the working set (and reports ERROR_NOT_LOCKED, which is expected here).
    VirtualUnlock(reinterpret_cast<void*>(b), static_cast<SIZE_T>(e - b));
    return static_cast<size_t>(e - b);
#else
    return madvise(reinterpret_cast<void*>(b), static_cast<size_t>(e - b), MADV_DONTNEED) == 0 ? static_cast<size_t>(e - b) : 0;
#endif
}

}  // namespace strata::core
