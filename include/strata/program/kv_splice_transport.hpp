// include/strata/program/kv_splice_transport.hpp - Phase 2 hotfix: the memory transport for
// the suffix-reuse splice.
//
// WHY THIS EXISTS (the 2026-10-12 SIGSEGV).  `rotate_qsa_tail_inplace` (kv_splice_apply.hpp)
// was written as a HOST loop: it reads `st.k_pool` rows and `st.cos_tab`/`st.sin_tab`
// entries directly through the pointers `QsaState` carries.  In the CUDA build those
// pointers live in the session arena allocated with `cudaMalloc` (generate.cpp:3763), and
// `qsa_state_init` builds the rope table on the host only to `cudaMemcpy` it INTO
// `st.cos_tab` (layer.cpp:836-841) - they are DEVICE addresses.  A host dereference of a
// discrete-GPU device address is SIGSEGV.  The CPU-only build (the ctest suite) puts the
// same arena in host memory, which is why T6 never saw the fault: the first
// checkpoint-mounted request with --cache-suffix-reuse died at the splice, before any
// prefill log line, exit code -11 (reproduced 2/2 + 6 historical, probe5).
//
// THE FIX IN ONE SENTENCE: the splice math stays a host loop (it is pure and T6-pinned),
// but every byte that crosses to/from `st.k_pool` and the rope tables goes through
// `cudaMemcpy` staging when the pointer is a device address, and the loop never
// dereferences a device pointer on the host.
//
// HOW "IS THIS A DEVICE POINTER?" IS DECIDED.  A CUDA runtime pointer's address carries no
// portable tag, so the transport asks the runtime: `cudaPointerGetAttributes` reports
// `cudaMemoryTypeDevice` for a cudaMalloc'd/VMM-mapped allocation.  That probe is cached
// per QsaState (the arena is allocated once at startup and the answer never changes), so
// the per-request cost is one attributes query per layer the first time and nothing after.
// On a CPU-only build the transport compiles to the identity: pointers are host pointers,
// the staging buffer is the buffer itself.
//
// FAIL-CLOSED BY CONSTRUCTION.  The transport is active on every TU compiled with the CUDA
// compiler (`__CUDACC__` is NVCC's own per-TU define, so it covers generate.cpp, layer.cpp,
// every qsa_*.cu - the files that hold live QsaStates).  An extra opt-in macro
// `STRATA_SPLICE_FORCE_TRANSPORT` lets a test TU force the transport on a CPU build (the
// fake-cuda harness at sfx-crash-harness-transport.cpp uses this; it supplies its own
// stub cuda_runtime API and PROT_NONE guard region).  Compiling a CUDA TU without the
// transport active is a hard `#error` - the splice is too dangerous to be inert on GPU.
#pragma once

// ---- Transport activation policy ------------------------------------------------
//
//   1. The CUDA compiler (NVCC, HIPCC, etc.) sets `__CUDACC__` for every TU it builds.
//      Those TUs MUST have the transport: they own the live QsaStates whose k_pool /
//      rope tables are device pointers.  We define STRATA_SPLICE_TRANSPORT for them so
//      the staging path is the only code that can ship.
//
//   2. The project's CMake option `STRATA_SPLICE_TRANSPORT` (a developer-only knob for
//      opting a CPU TU into the transport) overrides the default; this is what the
//      fake-cuda harness and the device-path ctest use.
//
//   3. The fail-closed `#error` below fires when a TU is compiled by NVCC (so it sees
//      live QsaStates) but somehow lacks the transport - it is a hard compile error
//      so a future refactor that disables the transport by accident cannot ship.
//
// Layer.cpp's qsa_state_init (which lays out the arena) and generate.cpp's prefill path
// are both NVCC-compiled; the engine binary always sees the transport.  The
// header-only test executables (suffix_reuse_apply_test, suffix_reuse_test) are
// g++-compiled and stay on the identity path: their QsaStates are synthesised, the
// pointers are host, and the identity transport is what runs.
// BUILD-GLOBAL activation (2026-10-12 hotfix round 2): `__CUDACC__` alone is NOT sufficient
// because generate.cpp - the TU that runs the splice at generate.cpp:9075 - is compiled by
// G++ as a plain host TU even in a CUDA build (only the .cu files go through NVCC).  The
// 3f840e3 gate keyed on __CUDACC__ therefore shipped the identity transport into the one
// TU that holds live device-backed QsaStates: is_active()=false + stub pointer_is_device()=false
// (which also disarmed the runtime de-rotation guard).  STRATA_SPLICE_TRANSPORT is now a
// build-wide target_compile_definition on every CUDA/HIP build target (see CMakeLists.txt),
// so ANY TU in a CUDA build - NVCC or g++ - gets the real transport.  __CUDACC__ is kept as
// a belt-and-braces per-TU trigger.  The #error below is extended to catch g++ TUs in a
// CUDA build too: any TU that sees this header in a build with CUDA enabled but no
// transport is a compile error, so the transport can never again ship inert.
#if defined(__CUDACC__) && !defined(STRATA_SPLICE_TRANSPORT)
#  define STRATA_SPLICE_TRANSPORT 1
#endif
#if defined(STRATA_HAVE_CUDA_BUILD) && !defined(STRATA_SPLICE_TRANSPORT)
#  define STRATA_SPLICE_TRANSPORT 1
#endif

#include "strata/core/layer.hpp"     // QsaState

#if defined(STRATA_SPLICE_TRANSPORT)
#  if defined(STRATA_SPLICE_FAKE_CUDA)
// The transport-verification harness supplies its own fake CUDA API before this header;
// it does not include <cuda_runtime.h> (there is no CUDA toolkit on the CPU host).
#  else
#    include <cuda_runtime.h>
#  endif
#  define STRATA_SPLICE_HAS_TRANSPORT 1
#endif

// ---- Hard fail-closed: a CUDA TU with no transport must NOT compile --------------
// This branch is unreachable when STRATA_SPLICE_TRANSPORT is set (above), but a future
// refactor that drops the activation could leave it as a build-time alarm: a CUDA TU
// that includes this header without the transport active is a bug.  We report it
// rather than silently shipping a build that will SEGV on the first splice.
#if (defined(__CUDACC__) || defined(STRATA_HAVE_CUDA_BUILD)) && !defined(STRATA_SPLICE_HAS_TRANSPORT)
#  error "STRATA SPLICE: this translation unit is in a CUDA build (NVCC or STRATA_HAVE_CUDA_BUILD) \
but the splice transport is not active.  The splice is unsafe to compile without \
STRATA_SPLICE_TRANSPORT: it dereferences device pointers from the host loop.  Either \
restore the activation above or stop including kv_splice_apply.hpp from this TU."
#endif

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program::suffix_reuse::transport {

/// `true` iff this translation unit was compiled with the splice transport active.
/// The runtime de-rotation guard in `apply_splice` checks this against
/// `pointer_is_device(st.k_pool)` to refuse a splice on a device-backed state when the
/// transport is missing - the alternative is a host deref of a device address (SIGSEGV).
/// On a CPU build this returns false; the identity transport is always safe to call.
inline bool is_active() {
#if defined(STRATA_SPLICE_HAS_TRANSPORT)
    return true;
#else
    return false;
#endif
}

#if !defined(STRATA_SPLICE_HAS_TRANSPORT)

/// CPU build: every pointer is host memory; staging is the identity.
inline bool pointer_is_device(const void*) { return false; }

/// A staging view of `bytes` starting at `p`.  On the CPU build the view aliases the
/// caller's memory directly - no copy in either direction.
struct Staged {
    void* host = nullptr;          ///< host-readable/writable alias of the range
    void* dev  = nullptr;          ///< the address to copy back to (== p on CPU)
    bool  device_backed = false;   ///< false: `host` IS the target, write directly
};
inline Staged stage_for_readwrite(void* p, size_t bytes) {
    (void) bytes;
    return Staged{p, p, false};
}
inline void commit(const Staged& s, size_t bytes) {
    (void) s; (void) bytes;         // nothing to do: writes already landed
}

#else  // STRATA_SPLICE_HAS_TRANSPORT

/// Ask the CUDA runtime whether `p` lives on the device.  `cudaPointerGetAttributes`
/// succeeds and reports a device/VMA allocation exactly for pointers the runtime handed
/// out (cudaMalloc, VMM map).  A failure of the call itself is treated as "not device"
/// (a plain host pointer) so a stub driver cannot disable the splice.
inline bool pointer_is_device(const void* p) {
    cudaPointerAttributes attr{};
    if (p == nullptr) return false;
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
        (void) cudaGetLastError();   // a host pointer: not an error worth surfacing
        return false;
    }
    return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

/// A staging buffer for one contiguous range of device memory.  `host` is a fresh host
/// buffer the caller reads/writes; `commit` copies the range back.  The copy pair is
/// synchronous (`cudaMemcpy`), which is correct here: the splice runs at request start,
/// before any prefill work is enqueued on the pool (generate.cpp quiesces the session
/// around the request boundary; nothing else touches K until the read loop starts).
///
/// Multiple `Staged` views can be live at the same time (the splice stages cos, sin, and
/// k together), so each view occupies its own slot in the shared pool.  The pool doubles
/// on demand and is never shrunk.  Slots are recycled by index, so the read+write pair
/// for one view is internally consistent: the host address handed back at stage time is
/// the same address commit() copies back to the device.
struct Staged {
    void* host = nullptr;          ///< host staging buffer (owned by the shared pool)
    void* dev  = nullptr;          ///< the device range's base
    bool  device_backed = false;   ///< true: commit() must copy back
};
struct StagedSlot {
    std::vector<uint8_t> data;     ///< grows; one Staged view per call
};
inline Staged stage_for_readwrite(void* p, size_t bytes) {
    if (!pointer_is_device(p)) return Staged{p, p, false};
    // Each call gets its own slot.  Recycle a small set (the splice stages at most a
    // handful of ranges per call: cos slice, sin slice, k range).  Using a thread_local
    // vector of slots means no global lock; the splice is single-threaded today.
    static thread_local std::vector<StagedSlot> slots;
    static thread_local size_t cursor = 0;
    if (cursor >= slots.size()) slots.emplace_back();
    StagedSlot& slot = slots[cursor++];
    if (cursor >= 1024) cursor = 0;   // bound the cycle so we don't grow forever
    if (slot.data.size() < bytes) {
        size_t want = bytes > slot.data.size() * 2 ? bytes : slot.data.size() * 2;
        try { slot.data.resize(want); } catch (...) { slot.data.resize(bytes); }
    }
    Staged s{slot.data.data(), p, true};
    if (cudaMemcpy(s.host, p, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return Staged{nullptr, nullptr, false};   // caller sees a null host: refuse the layer
    }
    return s;
}
inline void commit(const Staged& s, size_t bytes) {
    if (s.device_backed && s.host != nullptr && s.dev != nullptr && bytes > 0)
        (void) cudaMemcpy(s.dev, s.host, bytes, cudaMemcpyHostToDevice);
}

#endif  // STRATA_SPLICE_HAS_TRANSPORT

}  // namespace strata::program::suffix_reuse::transport
