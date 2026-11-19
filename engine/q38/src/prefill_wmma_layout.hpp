#pragma once
#include <cstddef>
#ifdef __CUDACC__
#define Q38_WMMA_HD __host__ __device__
#else
#define Q38_WMMA_HD
#endif
namespace strata::qwen38::wmma_layout {
inline constexpr int rows=64, tokens=32, depth=32, stride=40;
inline constexpr int warps=8, threads=warps*32;
inline constexpr std::size_t shared_bytes=(rows+tokens)*stride*2+rows*tokens*4;
Q38_WMMA_HD constexpr int a_index(int row,int k) {return row*stride+k;}
Q38_WMMA_HD constexpr int b_index(int token,int k) {return token*stride+k;}
Q38_WMMA_HD constexpr int c_index(int token,int row) {return token*rows+row;}
Q38_WMMA_HD constexpr int tiles(int n,int width) {return n<=0?0:1+(n-1)/width;}
constexpr bool eligible(int k,int n,int t,bool fp8_requested) {
    return !fp8_requested && k>=512 && n>=256 && t>=32;
}
} // namespace strata::qwen38::wmma_layout
#undef Q38_WMMA_HD
