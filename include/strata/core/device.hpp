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
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
    std::string arch;              // HIP: gcnArchName without its feature suffix ("gfx1201"); empty on CUDA
};

// HIP builds: whether GPU `ordinal` can run this binary - its architecture must be one the binary was COMPILED
// for (STRATA_HIP_ARCHS, set by cmake/hip_backend.cmake) and it must run wave32.  "" when it can (or when there
// is no such device: the caller's own device errors apply), else the reason in a sentence.  A binary carried to
// another card would otherwise fail later with "invalid device function".  CUDA builds: always "".
std::string gpu_arch_problem(int ordinal);

// The GPU architectures this binary was compiled for ("gfx1100,gfx1201"); "" on CUDA builds.
const char* compiled_gpu_archs();

// How many devices the runtime enumerates, numbered as HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES number them; 0 when
// there is none (or no usable runtime).  `device_info` throws on a card this binary cannot run, so a caller that LISTS
// every card - the ones it has no code for included (`strata-device --list-devices`) - counts them with this first.
int device_count();

// One line per device for `strata-device --list-devices`, without the arch check that `device_info` applies:
// "arch gfx1201, 15.9 GiB" (HIP) or "compute capability 12.0, 11.9 GiB" (CUDA), plus the device's name.  false when
// the runtime cannot describe it.
bool device_summary(int ordinal, std::string& name, std::string& detail);

// Throws when there is no CUDA device.  The engine targets sm_120 specifically and must say so rather than
// run slowly on something else: `CMakeLists.txt` already refuses to COMPILE for another architecture, and
// this is the matching check at run time (a binary can be carried to a different machine).
DeviceInfo device_info(int ordinal = 0);

// Fase 2 (hetero multi-GPU): what each VISIBLE device can do, recorded once per ordinal rather than asked
// repeatedly.  The kernel paths already make their own per-device queries through static per-device caches;
// the role decisions that follow (main-device / draft-device) need the same facts in one record: the compute
// capability the kernel selectors use (STRATA_EMULATE_CC is honoured, so a test rig answers as the emulated
// card would), the shared-memory ceiling the smem opt-in call sites must respect, and the peer-access matrix -
// queried, never enabled, because peer access stays optional by design (docs/MULTI_GPU.md).
struct DeviceCaps {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;    // effective cc; STRATA_EMULATE_CC answers as the emulated card would
    int multi_processor_count = 0;
    uint64_t total_bytes = 0, free_bytes = 0;
    int shared_mem_default = 0, shared_mem_optin = 0;
    bool dp4a = false;                 // cc 6.1+: the packed int8 dot-product instruction
    bool mma_int8 = false;             // cc 7.5+: int8 mma
    bool mma_tf32 = false;             // cc 8.0+: TF32 mma
    bool cp_async = false;             // cc 8.0+
    int driver_version = 0, runtime_version = 0;
    std::string arch;                  // HIP: gcn arch name; empty on CUDA
};

// Every visible device, in ordinal order.  Throws only when the count query itself fails; there is no global
// minimum here - gating per capability is up to the caller (device_info keeps the release-build floor).
std::vector<DeviceCaps> device_caps();

// device_caps().size() squared entries; entry [i * n + j] is 1 when device i can peer-access device j.  Read
// with cudaDeviceCanAccessPeer only - nothing gets enabled by a query (enabling is a later, explicit decision).
// A topology without peer support (PCIe root bridges) simply reports 0 everywhere.
std::vector<uint8_t> peer_access_matrix();

/// "" when this build has device code for the current device, else CUDA's error: a build for other GPUs would
/// otherwise fail at its first kernel launch, with nothing that names the cause.
std::string device_code_error();

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// One cudaMalloc, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a debug
// build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same reasoning as
// the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

}  // namespace strata::core
