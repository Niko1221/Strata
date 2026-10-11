// SYCL port: upstream's include/strata/core/device.hpp with a chunked DeviceArena.  The include order
// (sycl/include before include/) makes this copy shadow upstream's for every file of the SYCL build.
//
// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE cudaMalloc per planner region with bump sub-allocation below it and no frees.  That is
// not a simplification for the first version: the memory plan from P1.S9 is fixed at startup, so the set of
// regions and their sizes is known before anything is allocated, and an allocator that can free would be
// solving a problem the engine does not have while adding fragmentation and failure modes it does.
//
// The reason to get this in early is that the VRAM budget is the binding constraint of the whole design
// (5.95 GB pooled between KV and the expert cache, 33.97 GB of experts in DRAM).  A runtime that discovers at
// token 4000 that it has overcommitted has already lost; the plan is printed against `cudaMemGetInfo` at
// startup so the discrepancy is visible immediately.
//
// SYCL port: on an Arc Pro B70 the one allocation can be what fails.  Under Linux one device allocation above a
// threshold that tracks host RAM plus swap is refused (compute-runtime#998, open; 19.3 GiB on a 16 GiB box), llama.cpp
// caps its buffers at 60% of max_mem_alloc_size on BMG-G31 (PR #28953), and native-Windows B70 reports keep every
// allocation under 4 GiB (UR errors 39/40 above that).  Measured on the Windows B70 PC (128 GiB of RAM, alloc_probe,
// 2026-10-04): one allocation of the whole card, 31.16 GiB, passed at VRAM speed.  So the arena is one
// sycl::malloc_device first, as upstream, and several chunks behind the same bump allocator only when the device
// refuses that: a sub-allocation never straddles two chunks, and the tail a chunk cannot hold counts as used.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;   // SYCL port: the device's version string (Level Zero "1.6"), not a compute capability
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
    std::string arch;              // SYCL port: the architecture enum's name ("intel_gpu_bmg_g31", ..., "unknown")
};

// SYCL port: whether Level Zero device `ordinal` can run this binary: 32-wide sub-groups (every kernel is built with
// -fsycl-default-sub-group-size=32), fp16, device and host USM; and, in an ahead-of-time build for bmg-g31, that
// architecture.  "" when it can (or when there is no such device: the caller's own device errors apply), else the
// reason in a sentence.  An integrated GPU beside a discrete one is not refused here: only the capability checks are.
std::string gpu_arch_problem(int ordinal);

// SYCL port: the ahead-of-time device this binary was compiled for ("bmg-g31", STRATA_SYCL_AOT_DEVICE); "" for a
// SPIR-V build that is compiled at run time.
const char* compiled_gpu_archs();

// How many Level Zero devices the runtime enumerates, numbered in SYCL's order (ONEAPI_DEVICE_SELECTOR=level_zero:N
// numbers them the same way); 0 when there is none (or no usable runtime).  `device_info` throws on a card this
// binary cannot run, so a caller that LISTS every card - the ones it has no code for included (`strata-device
// --list-devices`) - counts them with this first.
int device_count();

// One line per device for `strata-device --list-devices`, without the capability check that `device_info` applies:
// "arch intel_gpu_bmg_g31, 30.3 GiB" (info::device::global_mem_size), plus the device's name.  false when the runtime
// cannot describe it.
bool device_summary(int ordinal, std::string& name, std::string& detail);

// Throws when there is no Level Zero device, or when the device cannot run this engine (gpu_arch_problem).  A binary
// can be carried to a different machine: this is the run-time half of the architecture check.
DeviceInfo device_info(int ordinal = 0);

/// "" when this build has device code for the current device, else the runtime's error: a build for other GPUs would
/// otherwise fail at its first kernel launch, with nothing that names the cause.
std::string device_code_error();

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// Several sycl::malloc_device chunks, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern
// in a debug build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same
// reasoning as the harness work in Phase 1: a wrong value that looks right is the expensive kind.
//
// Chunk size: STRATA_ARENA_CHUNK_MIB when set (0 = one allocation for the whole arena, no fallback);
// else one allocation of the whole arena first and, only when the device refuses it, chunks of 60% of
// info::device::max_mem_alloc_size (on Windows at most 4096 MiB); STRATA_MAX_ALLOC_MIB overrides the reported
// maximum.  The last chunk may be smaller.  Every chunk is registered with the VRAM heartbeat
// (strata/sycl_heartbeat.hpp) for its lifetime.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.  The alignment is of the
    // absolute address.  A request that does not fit the rest of the current chunk moves to the next one, and the
    // skipped tail counts as used.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return chunks_.empty() ? nullptr : chunks_.front().ptr; }   // the first chunk

    // SYCL port: how the arena was cut, for the startup print and the probes.
    size_t chunk_count() const { return chunks_.size(); }
    uint64_t chunk_bytes() const { return chunk_bytes_; }

private:
    struct Chunk {
        void* ptr = nullptr;
        uint64_t bytes = 0, used = 0;
    };
    void release() noexcept;                       // unregister and free every chunk (the destructor, and a failed constructor)
    std::vector<Chunk> chunks_;
    size_t cur_ = 0;                               // the chunk alloc() is filling
    uint64_t capacity_ = 0, used_ = 0, chunk_bytes_ = 0;
    int ordinal_ = 0;
    unsigned dev_id_ = 0;                          // dpct's id of the device the chunks live on (the destructor frees there)
    bool poison_ = false;
};

}  // namespace strata::core
