#pragma once
// CUDA virtual memory (the elastic K/V, --kv-grow): an address range reserved once, whose physical memory is mapped and
// unmapped in chunks of the device's granularity (2 MiB).  The K/V pools grow with the context by taking chunks the
// expert cache gives up, and neither moves: every captured graph and stored pointer stays valid.  The driver calls
// are reached through the runtime's entry-point query, so nothing links the driver library.
// The combined Linux HIP VRAM/K/V arena requires a successful runtime probe.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

/// A physical chunk: the driver's allocation handle (CUmemGenericAllocationHandle); 0 = none.
using VmmChunk = unsigned long long;

/// The current device supports it and the entry points resolved.
bool vmm_available();
/// The chunk size (0 when not available).
uint64_t vmm_granularity();
#if defined(STRATA_USE_HIP) && defined(__linux__)
/// The current device passed an allocation, touch, and reclaim check with the arena's chunks.
bool vmm_shared_arena_available();
/// Acknowledges physical release; failure leaves the detached handle tracked.
bool vmm_chunk_release(VmmChunk h);
/// Segment operations run inside the cache's saved HIP device scope.
uintptr_t vmm_segment_reserve(uint64_t bytes);
bool vmm_segment_free_address(uintptr_t va, uint64_t bytes);
bool vmm_segment_map(uintptr_t va, uint64_t bytes, VmmChunk& handle, bool& mapped);
bool vmm_segment_unmap(uintptr_t va, uint64_t bytes);
#endif
/// A new physical chunk on the current device (0: out of memory), and its release.
VmmChunk vmm_chunk_new();
void vmm_chunk_free(VmmChunk h);

/// HIP only: checked owning-device scope. Negative device and other backends are no-ops.
class VmmDeviceGuard {
public:
    explicit VmmDeviceGuard(int device);
    ~VmmDeviceGuard();
    VmmDeviceGuard(const VmmDeviceGuard&) = delete;
    VmmDeviceGuard& operator=(const VmmDeviceGuard&) = delete;
    bool ok() const { return ok_; }
    bool enter();
    bool finish(bool result);
private:
    int device_ = -1, previous_ = -1;
    bool ok_ = true;
};

class VmmRange {
public:
    VmmRange() = default;
    ~VmmRange() { release(); }
    VmmRange(const VmmRange&) = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    /// Reserves `bytes` (rounded up to whole chunks) of addresses, nothing mapped.
    bool reserve(uint64_t bytes);
    /// Unmaps and frees every chunk, then the addresses.
    void release();
    uint8_t* base() const { return (uint8_t*) (uintptr_t) base_; }
    int64_t chunks() const { return (int64_t) h_.size(); }
    bool mapped(int64_t i) const { return i >= 0 && i < chunks() && h_[(std::size_t) i] != 0; }
    int64_t mapped_count() const;
    /// Maps chunks [lo, hi) that are not mapped yet, each to `take()` or, when that returns 0, to a new chunk.
    /// Readable and writable when it returns; false: out of memory (what was mapped stays mapped).
    template <class Take> bool map_range(int64_t lo, int64_t hi, Take take);
    /// Unmaps chunk i and hands back its physical chunk (0 when it was not mapped).
    VmmChunk unmap(int64_t i);

private:
    bool map_one(int64_t i, VmmChunk h);
    bool set_access(int64_t lo, int64_t hi);
    /// Readable and writable [lo, hi) (all newly mapped); when that fails they are unmapped and freed again, so no
    /// chunk counts as mapped without access.
    bool commit_run(int64_t lo, int64_t hi);
    unsigned long long base_ = 0;
    int device_ = -1;          // HIP range owner; captured by reserve, never taken from a later current device
    uint64_t gran_ = 0;
    std::vector<VmmChunk> h_;   // per chunk: its physical chunk, 0 = unmapped
};

template <class Take> bool VmmRange::map_range(int64_t lo, int64_t hi, Take take) {
    VmmDeviceGuard on(device_);
    if (!on.ok()) return false;
    lo = lo < 0 ? 0 : lo;
    hi = hi > chunks() ? chunks() : hi;
    int64_t run = -1;   // access is set once per run of newly mapped chunks (it costs as much as the map)
    for (int64_t i = lo; i < hi; ++i) {
        if (mapped(i)) {
            if (run >= 0 && !commit_run(run, i)) return on.finish(false);
            run = -1;
            continue;
        }
        VmmChunk h = take();
        if (!on.enter()) {
            if (h != 0) vmm_chunk_free(h);
            return on.finish(false);
        }
        if (h == 0) h = vmm_chunk_new();
        if (h == 0 || !map_one(i, h)) {
            if (h != 0) vmm_chunk_free(h);
            if (run >= 0) commit_run(run, i);
            return on.finish(false);
        }
        if (run < 0) run = i;
    }
    return on.finish(run < 0 || commit_run(run, hi));
}

}  // namespace strata::core
