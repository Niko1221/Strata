#pragma once
#include <cstdint>
namespace strata::hy4_flash::cpu {
/// CUDA-layout Q8_1: half scale, half raw sum, 32 signed codes. No CUDA dependency.
void native_quant_q81_compatible(const float* x, void* dst, int n);
/// Scalar oracle/fallback: identical CUDA-layout quantization contract.
void native_quant_q81_scalar(const float* x, void* dst, int n);
/// Q4_K/Q6_K integer dots against that layout. CPU accumulation is not CUDA's warp tree.
float native_dot_q81_compatible(int type, const uint8_t* row, const void* act, int n);
float native_dot_q81_scalar(int type, const uint8_t* row, const void* act, int n);
/// Opt-in Q4_K GU pair. False leaves the caller on its original two-dot path.
bool native_dot_q81_pair_compatible(int type, const uint8_t* gate, const uint8_t* up,
                                    const void* act, int n, float& g, float& u);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
void native_quant_q81_avx2(const float* x, void* dst, int n);
float native_dot_q81_avx2(int type, const uint8_t* row, const void* act, int n);
void native_dot_q81_pair_q4_avx2(const uint8_t* gate, const uint8_t* up,
                                const void* act, int n, float& g, float& u);
#endif
} // namespace strata::hy4_flash::cpu
