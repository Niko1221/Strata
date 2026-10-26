// include/arm64_compat/immintrin.h - the x86 intrinsics the host loops use, on Linux arm64: the spin-wait hint and the
// store fence before a host flag is raised (verify.cpp, session.cpp, mtp.cpp, graph.cpp, pool.cpp).  The GPU reads
// those flags, so the fence orders the stores for the outer-shareable domain (Metal's shim uses ishst: one SoC).
#pragma once
static inline void _mm_pause() { __asm__ __volatile__("yield"); }
static inline void _mm_sfence() { __asm__ __volatile__("dmb oshst" ::: "memory"); }
