// engine/src/inkling2/cuda/ink_pdl_dev.cuh - the device half of programmatic dependent launch (PERF-STEP lane)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// INK_PDL_WAIT() is the FIRST statement of every kernel that may be started early (see ink_pdl.cuh for the host half). When the kernel is launched the plain
// way, or on a card below sm_90, it does nothing. When the graph holds a programmatic edge into the kernel, it returns once the kernel before it has completed
// and its writes are visible: a kernel may read constants (weights) before it, and must read or write nothing an earlier kernel of the step produces.
// This file and ink_pdl.cuh are written for Strata's graph rewrite.
#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDACC_VER_MAJOR__) && \
    ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)) && !defined(__HIP_PLATFORM_AMD__)
#define INK_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define INK_PDL_COMPILED 1
#else
#define INK_PDL_WAIT() ((void)0)
#endif
