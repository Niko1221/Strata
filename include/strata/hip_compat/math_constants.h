// hip_compat/math_constants.h - the CUDA math constants Strata uses, for the
// HIP build (ROCm ships no math_constants.h). Values are the CUDA definitions.
#pragma once

#define CUDART_INF_F __int_as_float(0x7f800000u)
#define CUDART_NAN_F __int_as_float(0x7fffffffu)
#define CUDART_PI_F 3.14159265358979323846264338327950288f
#define CUDART_SQRT2_F 1.41421356237309504880168872420970f
#define CUDART_MAX_NORMAL_F 3.40282346638528859811704183484516925e38f
