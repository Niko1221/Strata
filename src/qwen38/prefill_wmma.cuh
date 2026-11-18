#pragma once
// Experimental native packed-weight -> shared FP16 -> WMMA kernel.
// No ggml/llama runtime dependency. Format decoding is supplied by the existing
// prefill_gemm.cu decoder. Validate on the target GPU before enabling dispatch.
#include "prefill_wmma_layout.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstddef>

namespace strata::qwen38::detail {
template<class Decode>
__global__ void __launch_bounds__(wmma_layout::threads)
packed_wmma_kernel(const uint8_t* __restrict__ packed,
                   const half* __restrict__ x,
                   float* __restrict__ y,
                   int K,int N,int T,std::size_t row_bytes,Decode decode) {
    namespace L=wmma_layout;
    namespace W=nvcuda::wmma;
    __shared__ __align__(32) half a[L::rows*L::stride];
    __shared__ __align__(32) half b[L::tokens*L::stride];
    __shared__ __align__(32) float c[L::rows*L::tokens];
    const int tid=threadIdx.x,warp=tid/32;
    const int row0=blockIdx.x*L::rows,token0=blockIdx.y*L::tokens;
    const int warp_row=(warp/2)*16,warp_token=(warp%2)*16;
    W::fragment<W::matrix_a,16,16,16,half,W::row_major> fa;
    W::fragment<W::matrix_b,16,16,16,half,W::col_major> fb;
    W::fragment<W::accumulator,16,16,16,float> acc;
    W::fill_fragment(acc,0.0f);
    // Never return individual threads/warps for a ragged tile: WMMA and block
    // barriers require uniform participation. Only global loads/stores are masked.
    const int ktiles=L::tiles(K,L::depth);
    for(int kt=0;kt<ktiles;++kt) {
        const int k0=kt*L::depth;
        for(int i=tid;i<L::rows*L::depth;i+=L::threads) {
            const int r=i/L::depth,k=i%L::depth,gr=row0+r,gk=k0+k;
            half value=__float2half_rn(0.0f);
            if(gr<N && gk<K)
                value=__float2half_rn(decode(packed+std::size_t(gr)*row_bytes,gk));
            a[L::a_index(r,k)]=value;
        }
        for(int i=tid;i<L::tokens*L::depth;i+=L::threads) {
            const int t=i/L::depth,k=i%L::depth,gt=token0+t,gk=k0+k;
            b[L::b_index(t,k)]=(gt<T && gk<K)?x[std::size_t(gt)*K+gk]:__float2half_rn(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<L::depth;k+=16) {
            W::load_matrix_sync(fa,a+L::a_index(warp_row,k),L::stride);
            W::load_matrix_sync(fb,b+L::b_index(warp_token,k),L::stride);
            W::mma_sync(acc,fa,fb,acc);
        }
        __syncthreads(); // every warp has finished reading a/b before overwrite
    }
    W::store_matrix_sync(c+L::c_index(warp_token,warp_row),acc,L::rows,W::mem_col_major);
    __syncthreads();
    for(int i=tid;i<L::rows*L::tokens;i+=L::threads) {
        const int t=i/L::rows,r=i%L::rows;
        if(token0+t<T && row0+r<N)
            y[std::size_t(token0+t)*N+row0+r]=c[L::c_index(t,r)];
    }
}
} // namespace strata::qwen38::detail
