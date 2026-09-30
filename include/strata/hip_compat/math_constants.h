#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA's header.
#include <hip/hip_math_constants.h>
// CUDA's math_constants.h: the CUDART_ prefixed floating point constants. The bit patterns are the same
// as HIP's, so this is a rename, not a re-derivation.
#define CUDART_INF_F HIP_INF_F
#define CUDART_NAN_F HIP_NAN_F
#define CUDART_MIN_DENORM_F HIP_MIN_DENORM_F
#define CUDART_MAX_NORMAL_F HIP_MAX_NORMAL_F
#define CUDART_NEG_ZERO_F HIP_NEG_ZERO_F
#define CUDART_SQRT2_F HIP_SQRT_TWO_F
#define CUDART_SQRT_HALF_F HIP_SQRT_HALF_F
#define CUDART_THIRD_F HIP_THIRD_F
#define CUDART_PIO4_F HIP_PIO4_F
#define CUDART_PIO2_F HIP_PIO2_F
#define CUDART_3PIO4_F HIP_3PIO4_F
#define CUDART_2_OVER_PI_F HIP_2_OVER_PI_F
#define CUDART_SQRT_2_OVER_PI_F HIP_SQRT_2_OVER_PI_F
#define CUDART_L2E_F HIP_L2E_F
#define CUDART_L2T_F HIP_L2T_F
#define CUDART_LG2_F HIP_LG2_F
#define CUDART_LGE_F HIP_LGE_F
#define CUDART_LN2_F HIP_LN2_F
#define CUDART_LNT_F HIP_LNT_F
#define CUDART_LNPI_F HIP_LNPI_F
#define CUDART_TWO_TO_M126_F HIP_TWO_TO_M126_F
#define CUDART_TWO_TO_126_F HIP_TWO_TO_126_F
#define CUDART_NORM_HUGE_F HIP_NORM_HUGE_F
#define CUDART_TWO_TO_23_F HIP_TWO_TO_23_F
#define CUDART_TWO_TO_24_F HIP_TWO_TO_24_F
#define CUDART_TWO_TO_31_F HIP_TWO_TO_31_F
#define CUDART_TWO_TO_32_F HIP_TWO_TO_32_F
