// include/strata/sycl_mem.hpp - the SYCL port: the free-VRAM figure the engine plans against.
//
// dpct migrated every cudaMemGetInfo to dpct's get_memory_info, which asks sycl_ext_intel_device_info's free_memory
// (Level Zero Sysman, ZES_ENABLE_SYSMAN=1) and prints to stderr when the card cannot answer.  On Windows that figure
// is the card's size minus this process's allocations: it never counts what the desktop or other programs hold, and
// WDDM moves whatever is allocated past the process's video memory budget out to system RAM (upstream measured the
// same on HIP: src/core/device.cu).  mem_get_info() lowers the figure by what the budget withholds, so a plan that
// fits "free" also stays resident.  On a card that drives no display the budget is nearly the whole card and the
// subtraction is close to zero, but it is the same query, so the code is the same.
#pragma once
#include <cstddef>

namespace strata {

/// Free and total VRAM of the current dpct device.  On Windows, `free_bytes` is lowered by what the DXGI local
/// budget withholds from this process (STRATA_WDDM_BUDGET=0 turns that off).  False when no free figure is
/// available (no ext_intel_free_memory aspect: Sysman off, or a runtime that cannot say): `total_bytes` is still
/// set when the device could be asked, `free_bytes` is 0 and must not be read as "nothing free".
bool mem_get_info(size_t& free_bytes, size_t& total_bytes);

/// The current dpct device's Windows adapter LUID (sycl_ext_intel_device_info `luid`, DPC++ 2025.3+), the 8 bytes
/// DXGI and cudaDeviceProp::luid use.  False when the runtime has no LUID for it, and always elsewhere than
/// Windows.  Shared by mem_get_info and pinned.cu's sliced-pin cap (strata::platform::gpu_shared_memory_budget).
bool device_luid(unsigned char (&luid)[8]);

}  // namespace strata
