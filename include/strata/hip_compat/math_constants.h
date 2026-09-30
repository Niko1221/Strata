#pragma once
// CUDA's <math_constants.h>.  CUDA-shaped sources include it for CUDART_INF_F; HIP names the same quantity
// HIP_INF_F.  cuda_runtime.h (force-included everywhere) maps it as well.
#include <hip/hip_math_constants.h>
#ifndef CUDART_INF_F
#define CUDART_INF_F HIP_INF_F
#endif
