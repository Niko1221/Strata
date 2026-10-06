// include/strata/kernels/dp4a.hpp - the two instructions the kernels use that Pascal (sm_60) does not have.
//
// The engine is written for compute capability 7.5 and newer and refuses older cards at CMake configure time
// (a Pascal build passes -DSTRATA_EXPERIMENTAL_SM60=ON).  A GP100-class card is compute capability 6.0, which is
// missing exactly two things the i-quant kernels rely on:
//
//   * `__dp4a`, a byte-wise dot product, available from 6.1 (Pascal GP10x/GV11x).  Its fallback below is
//     llama.cpp's own (`ggml/src/ggml-cuda/common.cuh`), which is the reference for the kernels in this
//     directory: they are transcribed from llama.cpp's vecdotq.cuh, and that wrapper's operands are the same
//     sites. GP100 uses signed-byte VMAD with wrapping int32 accumulation. The retained scalar fallback
//     agrees for bounded quant sums; its signed additions must not overflow, including intermediate sums.
//   * `__nanosleep`, available from 7.0 (Volta).  It only paces single-thread doorbell waits, so a loop that
//     spins without it is correct, merely busier.
//
// Both are compile-time selections on `__CUDA_ARCH__`, so one source tree builds for Pascal and for RTX
// 20/30/40/50 without a #define at the call sites.  A HIP build never takes either fallback: `__CUDA_ARCH__` is
// undefined there, so `STRATA_DP4A` is `__dp4a` (which `hip_compat/intrinsics.hpp` provides) and the pause is
// HIP's own `__nanosleep`.
#pragma once

#include <cstdint>
#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>   // gfx906: the compat layer (__forceinline__, __nanosleep, __dp4a)
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int strata_dp4a(const int a, const int b, const int c) {
#if __CUDA_ARCH__ == 600
    // GP100 has byte-select VMAD, but no DP4A. Keep this integer operation exact.
    // Original GP100 VMAD idea: shinbunbun, llama-cpp-p100-patches/01 (MIT, 2026).
    // Attribution and license: docs/P100_VMAD.md. A scoped accumulator keeps
    // inputs intact even when a caller passes the same value as an input and c.
    int result;
    asm("{\n\t"
        ".reg .s32 acc;\n\t"
        "vmad.s32.s32.s32 acc, %1.b0, %2.b0, %3;\n\t"
        "vmad.s32.s32.s32 acc, %1.b1, %2.b1, acc;\n\t"
        "vmad.s32.s32.s32 acc, %1.b2, %2.b2, acc;\n\t"
        "vmad.s32.s32.s32 acc, %1.b3, %2.b3, acc;\n\t"
        "mov.b32 %0, acc;\n\t"
        "}"
        : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#else
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}
#define STRATA_DP4A(a, b, c) strata_dp4a((a), (b), (c))
#else
#define STRATA_DP4A(a, b, c) __dp4a((a), (b), (c))
#endif

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract, and only pre-Volta CUDA has no `__nanosleep` at all.
__device__ __forceinline__ void strata_spin_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    // sm_6x: the loop spins.  Every call site is a single-thread doorbell wait, so nothing else is delayed.
#else
    __nanosleep(100);
#endif
}
