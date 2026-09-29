// Minimal <math_constants.h> for ROCm installs that do not ship one (gfx1200 port).
#pragma once

#ifndef CUDART_INF_F
#define CUDART_INF_F __builtin_huge_valf()
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __builtin_nanf("")
#endif
#ifndef CUDART_PI_F
#define CUDART_PI_F 3.1415926535897932f
#endif
