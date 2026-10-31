// Strata. Optional huge-page layout for the DeepSeek RAM store's registered chunks.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace strata::ds4::store_pages {

constexpr size_t kPage = 2u << 20;
// A measured registration-range choice, not a CUDA API size limit. Keep the
// entire expert slot in one range, including its direct-I/O slack.
constexpr size_t kMaxRegistration = 0xffe00000ULL;   // 4 GiB - 2 MiB

struct Layout {
    size_t chunk_slots = 0, chunk_bytes = 0, slots = 0, arena_bytes = 0;
};

inline size_t rounded(size_t bytes) { return (bytes + kPage - 1) & ~(kPage - 1); }

// Padding consumes the same RAM budget as payload. A short final chunk is
// rounded too; no slot or registration crosses a chunk boundary.
inline bool layout(size_t slot_bytes, size_t requested_chunk_slots, size_t budget, Layout& out) {
    if (!slot_bytes || slot_bytes > kMaxRegistration || !requested_chunk_slots) return false;
    Layout next;
    next.chunk_slots = std::min(requested_chunk_slots, kMaxRegistration / slot_bytes);
    next.chunk_bytes = rounded(next.chunk_slots * slot_bytes);
    const size_t whole = budget / next.chunk_bytes;
    const size_t tail_budget = (budget % next.chunk_bytes) & ~(kPage - 1);
    const size_t tail_slots = tail_budget / slot_bytes;
    next.slots = whole * next.chunk_slots + tail_slots;
    next.arena_bytes = whole * next.chunk_bytes + rounded(tail_slots * slot_bytes);
    if (!next.slots || next.arena_bytes > std::numeric_limits<size_t>::max() - kPage) return false;
    out = next;
    return true;
}

// Reserve one aligned address range so the store's [lo, hi) device check still
// applies. Physical pages are committed by grow(), as on the ordinary path.
inline void* reserve(size_t bytes) {
#if defined(__linux__)
    if (!bytes || bytes % kPage || bytes > std::numeric_limits<size_t>::max() - kPage) return nullptr;
    void* raw = ::mmap(nullptr, bytes + kPage, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kPage - 1) & ~(uintptr_t) (kPage - 1);
    const size_t prefix = aligned - start, suffix = kPage - prefix;
    if (prefix) ::munmap(raw, prefix);
    if (suffix) ::munmap((void*) (aligned + bytes), suffix);
    return (void*) aligned;
#else
    (void) bytes;
    return nullptr;
#endif
}

// Advice is best effort, not proof of THP backing. Fault the pages before CUDA
// pins them; registration first can prevent later promotion to huge pages.
inline bool prefault(void* p, size_t bytes) {
#if defined(__linux__)
    const bool advised = ::madvise(p, bytes, MADV_HUGEPAGE) == 0;
    std::memset(p, 0, bytes);
    return advised;
#else
    (void) p;
    (void) bytes;
    return false;
#endif
}

}   // namespace strata::ds4::store_pages
