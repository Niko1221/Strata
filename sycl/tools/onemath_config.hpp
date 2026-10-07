// Minimal oneapi/math/detail/config.hpp for the Windows installer-free oneMKL bring-up
// (sycl/CMakeLists.txt copies this over the generated header oneMKL's own CMake would write).
// Generated config.hpp only selects backends (ONEMATH_ENABLE_*_BACKEND); the only backend on this
// machine is Intel GPUs through mkl_sycl_blas (PyPI onemkl-sycl-blas). No CUDA/HIP/NETLIB here.
#pragma once
#define ONEMATH_ENABLE_MKLGPU_BACKEND
