// The rocWMMA include for the GDN WY WMMA arm - GLOBAL SCOPE ONLY (include this BEFORE
// namespace strata::prefill opens).  rocwmma's headers spell std:: names that would resolve
// against strata::prefill's anonymous namespace if the headers were pulled in inside it
// (amdclang: "no member named 'max' in namespace 'strata::prefill::(anonymous namespace)::std'").
// The macro lift/restore is the qsa_select.cu pattern: the CUDA-compat shim
// (hip_compat/intrinsics.hpp, force-included via <cuda_runtime.h>) renames the CUDA warp
// intrinsics with macros, and hip's own bf16/fp8 headers (which rocwmma pulls in) use those
// same tokens as function names - so the macros are lifted while rocwmma parses and put back
// after; they must match intrinsics.hpp exactly.  gdn_rec_wy_wmma.cuh (inside the namespace)
// then re-includes <rocwmma/rocwmma.hpp> as a guarded no-op and sees the real ::std entities.
#if defined(__HIPCC__) && defined(__has_include)
#if __has_include(<rocwmma/rocwmma.hpp>)
#ifndef STRATA_GDN_WY_HAVE_ROCWMMA
#define STRATA_GDN_WY_HAVE_ROCWMMA 1
#undef __dp4a
#undef __vsub4
#undef __vsubss4
#undef __vcmpne4
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#include <rocwmma/rocwmma.hpp>
#define __dp4a(a, b, c) (::strata::hip_compat::dp4a((a), (b), (c)))
#define __vsub4(a, b) (::strata::hip_compat::vsub4((a), (b)))
#define __vsubss4(a, b) (::strata::hip_compat::vsubss4((a), (b)))
#define __vcmpne4(a, b) (::strata::hip_compat::vcmpne4((a), (b)))
#define __shfl_xor_sync(...) (::strata::hip_compat::shfl_xor_sync(__VA_ARGS__))
#define __shfl_down_sync(...) (::strata::hip_compat::shfl_down_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (::strata::hip_compat::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (::strata::hip_compat::shfl_sync(__VA_ARGS__))
#define __ballot_sync(...) (::strata::hip_compat::ballot_sync(__VA_ARGS__))
#endif
#endif
#endif
