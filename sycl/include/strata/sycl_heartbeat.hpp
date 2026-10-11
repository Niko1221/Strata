// include/strata/sycl_heartbeat.hpp - the SYCL port: keep the dedicated card's VRAM resident.
//
// Windows' display driver model evicts a GPU's idle memory to system RAM, and on a card that drives no display
// nothing else keeps it warm.  The engine registers every device range it owns here, and a thread in serve mode
// (src/program/generate.cpp) calls heartbeat_touch() every heartbeat_period_s() seconds: one small kernel per range
// reads a word every 64 KiB, which is a touch of every page the range spans.  The same submission keeps the Level
// Zero queue alive across idle minutes: a Windows L0 driver has returned DEVICE_LOST on the first submission after
// sustained work and then idle (intel/llvm#23209; a keep-alive every 60 s avoided it).
#pragma once
#include <cstddef>

namespace strata {

/// Add [ptr, ptr + bytes) of device USM on the CURRENT dpct device to the registry.  DeviceArena registers each of
/// its chunks; the engine's other device allocations register themselves.  A nullptr or 0 bytes is ignored.
void heartbeat_register(const void* ptr, size_t bytes);

/// Remove the range that starts at `ptr` (before it is freed: the touch kernel must not read freed memory).
void heartbeat_unregister(const void* ptr);

/// Touch every registered range: on its own lazily created in-order queue (one per dpct device the ranges live on,
/// in dpct's context, so the USM pointers are valid there), one kernel per range that reads one 32-bit word every
/// 64 KiB and sums them into a device scratch word, so the reads survive optimisation.  Waits for the kernels and
/// returns the bytes covered; 0 when nothing is registered or the submission failed (reported to stderr).  Thread
/// safe: the registry is under a mutex and the kernels only read.
size_t heartbeat_touch();

/// STRATA_SYCL_HEARTBEAT_S: the period in seconds (default 10 on Windows, 0 elsewhere; 0 = off).
unsigned heartbeat_period_s();

}  // namespace strata
