// Packed scalar decoders follow Strata iq_kernels.cu (ggml MIT formats);
// PQ2_0 follows PrismML 0.4.1 commit 1965755ceee6738b9a04ceaf5d30e6305d6ccd00.
// See README.md and LICENSE-formats.txt. No model/graph/runtime code is linked.
#include "ops.cuh"
#include "iq_tables.hpp"
#include "iq_decode.hpp"
#include <cmath>
namespace strata::qwen38 {
namespace {
__device__ float h(const uint8_t* p){return __half2float(__ushort_as_half(uint16_t(p[0])|uint16_t(p[1])<<8));}
__device__ float value(int type,const uint8_t* p,int i) {
    if(type==0)return reinterpret_cast<const float*>(p)[i];
    if(type==1)return __half2float(reinterpret_cast<const half*>(p)[i]);
    if(type==30)return __uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(p)[i])<<16);
    if(type==142){p+=size_t(i/128)*34;i%=128;return h(p)*(int((p[2+i/4]>>(2*(i%4)))&3)-1);}
    if(type==8){p+=size_t(i/32)*34;i%=32;return h(p)*reinterpret_cast<const int8_t*>(p+2)[i];}
    if(type==21||type==22){p+=size_t(i/256)*(type==21?110:82);i%=256;return iq_value(type,p,i,h(p),q38_iq2s_grid,q38_iq3s_grid);}
    if(type==23){
        p+=size_t(i/256)*136;i%=256;int b=i/32,j=i%32;
        int scale=((p[4+b/2]>>(4*(b%2)))&15)|(((uint16_t(p[2])|uint16_t(p[3])<<8)>>(2*b)&3)<<4);
        const int table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        int code=(p[8+16*b+j%16]>>(j>=16?4:0))&15;return h(p)*(scale-32)*table[code];
    }
    if(type==12 || type==13){
        bool five=type==13;p+=size_t(i/256)*(five?176:144);i%=256;int b=i/32,j=i%32;
        const uint8_t* sc=p+4;int d,m;
        if(b<4){d=sc[b]&63;m=sc[b+4]&63;}else{d=(sc[b+4]&15)|((sc[b-4]>>6)<<4);m=(sc[b+4]>>4)|((sc[b]>>6)<<4);}
        const uint8_t* qs=p+(five?48:16);
        int code=(qs[(b/2)*32+j]>>(4*(b%2)))&15;
        if(five)code|=((p[16+j]>>b)&1)<<4;
        return h(p)*d*code-h(p+2)*m;
    }
    // Q6_K: ql[128], qh[64], signed scales[16], fp16 d.
    p+=size_t(i/256)*210;i%=256;int ip=i/128,j=i%32,group=(i%128)/32;
    int lo=(p[ip*64+(group%2)*32+j]>>(group>=2?4:0))&15;
    int hi=(p[128+ip*32+j]>>(2*group))&3;
    int scale=reinterpret_cast<const int8_t*>(p+192)[ip*8+group*2+j/16];
    return h(p+208)*scale*((lo|(hi<<4))-32);
}
size_t row_bytes(int type,int n){
    switch(type){case 0:return size_t(n)*4;case 1:case 30:return size_t(n)*2;case 142:return size_t(n/128)*34;
    case 8:return size_t(n/32)*34;case 12:return size_t(n/256)*144;case 13:return size_t(n/256)*176;case 14:return size_t(n/256)*210;case 21:return size_t(n/256)*110;case 22:return size_t(n/256)*82;case 23:return size_t(n/256)*136;default:throw std::runtime_error("unsupported matrix type");}
}
void shape(int type,int n,int rows){int block=type==142?128:type==8?32:(type==12||type==13||type==14||type==21||type==22||type==23)?256:1;
    if(n<=0||rows<=0||n%block)throw std::runtime_error("invalid packed matrix dimensions");(void)row_bytes(type,n);}
__global__ void deq(int type,const uint8_t* p,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=value(type,p,i);}
__device__ float sum(float v){for(int i=16;i;i/=2)v+=__shfl_down_sync(0xffffffffu,v,i);return v;}
__global__ void __launch_bounds__(256) mv(int type,const uint8_t* __restrict__ w,const float* __restrict__ x,float* __restrict__ y,int n,int rows,size_t stride){
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=rows)return;
    float s=0;w+=size_t(row)*stride;
    for(int j=lane;j<n;j+=32)s=fmaf(value(type,w,j),__ldg(x+j),s);
    s=sum(s);if(!lane)y[row]=s;
}
__global__ void __launch_bounds__(256) mv_q8_0(const uint8_t* __restrict__ w,const float* __restrict__ x,float* __restrict__ y,int n,int rows,size_t stride){
    const int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=rows)return;
    const uint8_t* wp=w+size_t(row)*stride;const int nb=n/32;float s=0;
    for(int b=0;b<nb;++b){
        const uint8_t* blk=wp+size_t(b)*34;
        const float d=h(blk);
        s=fmaf(d*float(reinterpret_cast<const int8_t*>(blk+2)[lane]),__ldg(x+b*32+lane),s);
    }
    s=sum(s);if(!lane)y[row]=s;
}
template <int U16Count>
__device__ __forceinline__ void load_block_u16(uint16_t* dst, const uint8_t* __restrict__ src, int lane) {
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
    #pragma unroll
    for (int idx = lane; idx < U16Count; idx += 32) {
        dst[idx] = __ldg(s16 + idx);
    }
    __syncwarp();
}

template <bool FIVE>
__device__ __forceinline__ float dot_block_qk45(const uint8_t* p, const float* xb, int lane, bool x_is_shared) {
    const float d0 = h(p), dmin0 = h(p + 2);
    const uint8_t* sc = p + 4;
    const uint8_t* qs = p + (FIVE ? 48 : 16);
    float sd = 0, sm = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        int d, m;
        if (b < 4) { d = sc[b] & 63; m = sc[b + 4] & 63; }
        else { d = (sc[b + 4] & 15) | ((sc[b - 4] >> 6) << 4); m = (sc[b + 4] >> 4) | ((sc[b] >> 6) << 4); }
        int code = (qs[(b / 2) * 32 + lane] >> (4 * (b % 2))) & 15;
        if (FIVE) code |= ((p[16 + lane] >> b) & 1) << 4;
        const float xv = x_is_shared ? xb[b * 32 + lane] : __ldg(xb + b * 32 + lane);
        sd = fmaf(float(d * code), xv, sd);
        sm = fmaf(float(m), xv, sm);
    }
    return d0 * sd - dmin0 * sm;
}

__device__ __forceinline__ float dot_block_q6_k(const uint8_t* p, const float* xb, int lane, bool x_is_shared) {
    const float d = h(p + 208);
    float s = 0;
    #pragma unroll
    for (int ip = 0; ip < 2; ++ip) {
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int lo = (p[ip * 64 + (group % 2) * 32 + lane] >> (group >= 2 ? 4 : 0)) & 15;
            const int hi = (p[128 + ip * 32 + lane] >> (2 * group)) & 3;
            const int scale = reinterpret_cast<const int8_t*>(p + 192)[ip * 8 + group * 2 + lane / 16];
            const float xv = x_is_shared ? xb[ip * 128 + group * 32 + lane] : __ldg(xb + ip * 128 + group * 32 + lane);
            s = fmaf(float(scale * ((lo | (hi << 4)) - 32)), xv, s);
        }
    }
    return d * s;
}

__device__ __forceinline__ float dot_block_iq2_s(const uint8_t* p, const float* xb, int lane, bool x_is_shared) {
    const int group = lane / 8, g_lane = lane % 8;
    const float d = h(p) * 0.25f;
    float s = 0;
    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
        const int index = p[2 + sub * 4 + group] | (((p[66 + sub] >> (2 * group)) & 3) << 8);
        const uint64_t grid = __ldg(&q38_iq2s_grid[index]);
        const int mag = int((grid >> (8 * g_lane)) & 255);
        const int smag = (p[34 + sub * 4 + group] & (1 << g_lane)) ? -mag : mag;
        const int scale = (p[74 + sub] >> (4 * (group / 2))) & 15;
        const float xv = x_is_shared ? xb[sub * 32 + lane] : __ldg(xb + sub * 32 + lane);
        s = fmaf((0.5f + float(scale)) * float(smag), xv, s);
    }
    return d * s;
}

__device__ __forceinline__ float dot_block_iq3_s(const uint8_t* p, const float* xb, int lane, bool x_is_shared) {
    const int group = lane / 8, g_lane = lane % 8, idx = lane / 4, idx_lane = lane % 4;
    const float d = h(p);
    float s = 0;
    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
        const int index = p[2 + sub * 8 + idx] | (((p[66 + sub] >> idx) & 1) << 8);
        const uint32_t grid = __ldg(&q38_iq3s_grid[index]);
        const int mag = int((grid >> (8 * idx_lane)) & 255);
        const int smag = (p[74 + sub * 4 + group] & (1 << g_lane)) ? -mag : mag;
        const int scale = (p[106 + sub / 2] >> (4 * (sub % 2))) & 15;
        const float xv = x_is_shared ? xb[sub * 32 + lane] : __ldg(xb + sub * 32 + lane);
        s = fmaf(float((1 + 2 * scale) * smag), xv, s);
    }
    return d * s;
}

enum class MvKind { Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_S };

template <MvKind KIND, int BLOCK_BYTES>
__global__ void __launch_bounds__(128) mv_8rows(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n, int rows, size_t stride) {
    __shared__ uint16_t sw[4][(BLOCK_BYTES + 1) / 2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    const uint8_t* wp = w + size_t(row) * stride;
    const int nb = n / 256;
    float s = 0;
    for (int blk_i = 0; blk_i < nb; ++blk_i) {
        load_block_u16<(BLOCK_BYTES + 1) / 2>(sw[warp], wp + size_t(blk_i) * BLOCK_BYTES, lane);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(sw[warp]);
        const float* xb = x + blk_i * 256;
        if constexpr (KIND == MvKind::Q4_K) s += dot_block_qk45<false>(p, xb, lane, false);
        else if constexpr (KIND == MvKind::Q5_K) s += dot_block_qk45<true>(p, xb, lane, false);
        else if constexpr (KIND == MvKind::Q6_K) s += dot_block_q6_k(p, xb, lane, false);
        else if constexpr (KIND == MvKind::IQ2_S) s += dot_block_iq2_s(p, xb, lane, false);
        else s += dot_block_iq3_s(p, xb, lane, false);
    }
    s = sum(s);
    if (!lane) y[row] = s;
}

template <MvKind KIND, int BLOCK_BYTES>
__global__ void __launch_bounds__(256) mv_1row(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n, size_t stride) {
    __shared__ uint16_t sw[8][(BLOCK_BYTES + 1) / 2];
    __shared__ float wsum[8];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    const uint8_t* wp = w + size_t(row) * stride;
    const int nb = n / 256;
    float s = 0;
    for (int blk_i = warp; blk_i < nb; blk_i += 8) {
        load_block_u16<(BLOCK_BYTES + 1) / 2>(sw[warp], wp + size_t(blk_i) * BLOCK_BYTES, lane);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(sw[warp]);
        const float* xb = x + blk_i * 256;
        if constexpr (KIND == MvKind::Q4_K) s += dot_block_qk45<false>(p, xb, lane, false);
        else if constexpr (KIND == MvKind::Q5_K) s += dot_block_qk45<true>(p, xb, lane, false);
        else if constexpr (KIND == MvKind::Q6_K) s += dot_block_q6_k(p, xb, lane, false);
        else if constexpr (KIND == MvKind::IQ2_S) s += dot_block_iq2_s(p, xb, lane, false);
        else s += dot_block_iq3_s(p, xb, lane, false);
    }
    s = sum(s);
    if (!lane) wsum[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? wsum[lane] : 0.f;
        for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (!lane) y[row] = v;
    }
}

__global__ void __launch_bounds__(256) quantize_q8_1_kernel(const float* __restrict__ x, BlockQ8_1* __restrict__ y, int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 8 + warp;
    if (blk * 32 >= n) return;
    const float xv = __ldg(x + blk * 32 + lane);
    float amax = fabsf(xv);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    const int q = __float2int_rn(xv * inv_d);
    int qsum = q;
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xffffffffu, qsum, o);
    const int b1 = __shfl_down_sync(0xffffffffu, q, 1) & 0xFF;
    const int b2 = __shfl_down_sync(0xffffffffu, q, 2) & 0xFF;
    const int b3 = __shfl_down_sync(0xffffffffu, q, 3) & 0xFF;
    if ((lane & 3) == 0) {
        y[blk].qs[lane >> 2] = (q & 0xFF) | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    if (lane == 0) {
        y[blk].d = d;
        y[blk].s = d * float(qsum);
    }
}

__device__ __forceinline__ uint32_t expand_sign4(uint32_t b) {
    return ((b & 1) ? 0x000000FFu : 0u) | ((b & 2) ? 0x0000FF00u : 0u) |
           ((b & 4) ? 0x00FF0000u : 0u) | ((b & 8) ? 0xFF000000u : 0u);
}

template <bool FIVE>
__device__ __forceinline__ float dot_q8_qk45(const uint8_t* p, const BlockQ8_1* xb, int lane) {
    const float d0 = h(p), dmin0 = h(p + 2);
    const uint8_t* sc = p + 4;
    const uint8_t* qs = p + (FIVE ? 48 : 16);
    const int b_pair = lane >> 3, j = (lane & 7) * 4;
    const int b0 = 2 * b_pair, b1 = b0 + 1;
    int d_lo, m_lo, d_hi, m_hi;
    if (b_pair < 2) {
        d_lo = sc[b0] & 63; m_lo = sc[b0 + 4] & 63;
        d_hi = sc[b1] & 63; m_hi = sc[b1 + 4] & 63;
    } else {
        d_lo = (sc[b0 + 4] & 15) | ((sc[b0 - 4] >> 6) << 4);
        m_lo = (sc[b0 + 4] >> 4) | ((sc[b0] >> 6) << 4);
        d_hi = (sc[b1 + 4] & 15) | ((sc[b1 - 4] >> 6) << 4);
        m_hi = (sc[b1 + 4] >> 4) | ((sc[b1] >> 6) << 4);
    }
    const uint32_t q4 = *reinterpret_cast<const uint32_t*>(qs + b_pair * 32 + j);
    uint32_t lo4 = q4 & 0x0F0F0F0Fu;
    uint32_t hi4 = (q4 >> 4) & 0x0F0F0F0Fu;
    if constexpr (FIVE) {
        const uint32_t qh = *reinterpret_cast<const uint32_t*>(p + 16 + j);
        lo4 |= ((qh >> b0) & 0x01010101u) << 4;
        hi4 |= ((qh >> b1) & 0x01010101u) << 4;
    }
    const BlockQ8_1& x0 = xb[b0];
    const BlockQ8_1& x1 = xb[b1];
    const int32_t dp0 = __dp4a(int32_t(lo4), x0.qs[lane & 7], 0);
    const int32_t dp1 = __dp4a(int32_t(hi4), x1.qs[lane & 7], 0);
    float sum = d0 * (float(d_lo * dp0) * x0.d + float(d_hi * dp1) * x1.d);
    if ((lane & 7) == 0) sum -= dmin0 * (float(m_lo) * x0.s + float(m_hi) * x1.s);
    return sum;
}

__device__ __forceinline__ float dot_q8_q6_k(const uint8_t* p, const BlockQ8_1* xb, int lane) {
    const float d = h(p + 208);
    const int group = lane >> 3, j = (lane & 7) * 4;
    float s = 0;
    #pragma unroll
    for (int ip = 0; ip < 2; ++ip) {
        const uint32_t ql = *reinterpret_cast<const uint32_t*>(p + ip * 64 + (group & 1) * 32 + j);
        const uint32_t qh = *reinterpret_cast<const uint32_t*>(p + 128 + ip * 32 + j);
        const uint32_t lo4 = (group >= 2) ? ((ql >> 4) & 0x0F0F0F0Fu) : (ql & 0x0F0F0F0Fu);
        const uint32_t hi2 = (qh >> (2 * group)) & 0x03030303u;
        const int32_t w4 = __vsub4(int32_t(lo4 | (hi2 << 4)), 0x20202020);
        const int scale = reinterpret_cast<const int8_t*>(p + 192)[ip * 8 + group * 2 + ((lane & 7) >> 2)];
        const BlockQ8_1& x0 = xb[ip * 4 + group];
        const int32_t dp = __dp4a(w4, x0.qs[lane & 7], 0);
        s += float(scale * dp) * x0.d;
    }
    return d * s;
}

__device__ __forceinline__ float dot_q8_iq2_s(const uint8_t* p, const BlockQ8_1* xb, int lane) {
    const int sub = lane >> 2, group = lane & 3;
    const float d = h(p) * 0.25f;
    const int index = p[2 + sub * 4 + group] | (((p[66 + sub] >> (2 * group)) & 3) << 8);
    const uint64_t grid = __ldg(&q38_iq2s_grid[index]);
    const uint32_t lo = uint32_t(grid), hi = uint32_t(grid >> 32);
    const uint8_t sbyte = p[34 + sub * 4 + group];
    const uint32_t m_lo = expand_sign4(sbyte & 0xF), m_hi = expand_sign4(sbyte >> 4);
    const BlockQ8_1& x0 = xb[sub];
    const int2 xv2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[group * 2]));
    const int32_t dp = __dp4a(int32_t(lo & ~m_lo), xv2.x, 0) - __dp4a(int32_t(lo & m_lo), xv2.x, 0) +
                       __dp4a(int32_t(hi & ~m_hi), xv2.y, 0) - __dp4a(int32_t(hi & m_hi), xv2.y, 0);
    const int scale = (p[74 + sub] >> (4 * (group >> 1))) & 15;
    return d * (0.5f + float(scale)) * float(dp) * __ldg(&x0.d);
}

__device__ __forceinline__ float dot_q8_iq2_s_sgrid(const uint8_t* __restrict__ p, const BlockQ8_1* __restrict__ xb,
                                                   const uint64_t* __restrict__ sgrid, int lane) {
    const int sub = lane >> 2, group = lane & 3;
    const uint16_t dh = __ldg(reinterpret_cast<const uint16_t*>(p));
    const float d = __half2float(__ushort_as_half(dh)) * 0.25f;
    const uint8_t qs_b = __ldg(p + 2 + lane);
    const uint8_t qh_b = __ldg(p + 66 + sub);
    const int index = qs_b | (((qh_b >> (2 * group)) & 3) << 8);
    const uint64_t grid = sgrid[index];
    const uint32_t lo = uint32_t(grid), hi = uint32_t(grid >> 32);
    const uint8_t sbyte = __ldg(p + 34 + lane);
    const uint32_t m_lo = expand_sign4(sbyte & 0xF), m_hi = expand_sign4(sbyte >> 4);
    const BlockQ8_1& x0 = xb[sub];
    const int2 xv2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[group * 2]));
    const int32_t dp = __dp4a(int32_t(lo & ~m_lo), xv2.x, 0) - __dp4a(int32_t(lo & m_lo), xv2.x, 0) +
                       __dp4a(int32_t(hi & ~m_hi), xv2.y, 0) - __dp4a(int32_t(hi & m_hi), xv2.y, 0);
    const int scale = (__ldg(p + 74 + sub) >> (4 * (group >> 1))) & 15;
    return d * (0.5f + float(scale)) * float(dp) * __ldg(&x0.d);
}

__global__ void __launch_bounds__(256) mv_q8_iq2s_8rows(const uint8_t* __restrict__ w, const BlockQ8_1* __restrict__ x, float* __restrict__ y, int n, int rows, size_t stride) {
    __shared__ uint64_t sgrid[1024];
    for (int i = threadIdx.x; i < 1024; i += 256) sgrid[i] = __ldg(&q38_iq2s_grid[i]);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint8_t* wp = w + size_t(row) * stride;
    const int nb = n / 256;
    float s = 0;
    int blk_i = 0;
    for (; blk_i + 1 < nb; blk_i += 2) {
        s += dot_q8_iq2_s_sgrid(wp + size_t(blk_i) * 82, x + blk_i * 8, sgrid, lane);
        s += dot_q8_iq2_s_sgrid(wp + size_t(blk_i + 1) * 82, x + (blk_i + 1) * 8, sgrid, lane);
    }
    if (blk_i < nb) {
        s += dot_q8_iq2_s_sgrid(wp + size_t(blk_i) * 82, x + blk_i * 8, sgrid, lane);
    }
    s = sum(s);
    if (!lane) y[row] = s;
}

__device__ __forceinline__ float dot_q8_iq3_s(const uint8_t* p, const BlockQ8_1* xb, int lane) {
    const int sub = lane >> 2, group = lane & 3;
    const int idx0 = group * 2, idx1 = group * 2 + 1;
    const float d = h(p);
    const int index0 = p[2 + sub * 8 + idx0] | (((p[66 + sub] >> idx0) & 1) << 8);
    const int index1 = p[2 + sub * 8 + idx1] | (((p[66 + sub] >> idx1) & 1) << 8);
    const uint32_t lo = __ldg(&q38_iq3s_grid[index0]), hi = __ldg(&q38_iq3s_grid[index1]);
    const uint8_t sbyte = p[74 + sub * 4 + group];
    const uint32_t m_lo = expand_sign4(sbyte & 0xF), m_hi = expand_sign4(sbyte >> 4);
    const BlockQ8_1& x0 = xb[sub];
    const int2 xv2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[group * 2]));
    const int32_t dp = __dp4a(int32_t(lo & ~m_lo), xv2.x, 0) - __dp4a(int32_t(lo & m_lo), xv2.x, 0) +
                       __dp4a(int32_t(hi & ~m_hi), xv2.y, 0) - __dp4a(int32_t(hi & m_hi), xv2.y, 0);
    const int scale = (p[106 + (sub >> 1)] >> (4 * (sub & 1))) & 15;
    return d * float(1 + 2 * scale) * float(dp) * __ldg(&x0.d);
}

__device__ __forceinline__ float dot_q8_iq3_s_direct(const uint8_t* __restrict__ p, const BlockQ8_1* __restrict__ xb, int lane) {
    const int sub = lane >> 2, group = lane & 3;
    const int idx0 = group * 2, idx1 = group * 2 + 1;
    const uint16_t dh = __ldg(reinterpret_cast<const uint16_t*>(p));
    const float d = __half2float(__ushort_as_half(dh));
    const uint16_t qs2 = __ldg(reinterpret_cast<const uint16_t*>(p + 2) + lane);
    const uint8_t qh_b = __ldg(p + 66 + sub);
    const int index0 = (qs2 & 0xFF) | (((qh_b >> idx0) & 1) << 8);
    const int index1 = (qs2 >> 8) | (((qh_b >> idx1) & 1) << 8);
    const uint32_t lo = __ldg(&q38_iq3s_grid[index0]), hi = __ldg(&q38_iq3s_grid[index1]);
    const uint8_t sbyte = __ldg(p + 74 + lane);
    const uint32_t m_lo = expand_sign4(sbyte & 0xF), m_hi = expand_sign4(sbyte >> 4);
    const BlockQ8_1& x0 = xb[sub];
    const int2 xv2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[group * 2]));
    const int32_t dp = __dp4a(int32_t(lo & ~m_lo), xv2.x, 0) - __dp4a(int32_t(lo & m_lo), xv2.x, 0) +
                       __dp4a(int32_t(hi & ~m_hi), xv2.y, 0) - __dp4a(int32_t(hi & m_hi), xv2.y, 0);
    const int scale = (__ldg(p + 106 + (sub >> 1)) >> (4 * (sub & 1))) & 15;
    return d * float(1 + 2 * scale) * float(dp) * __ldg(&x0.d);
}

template <MvKind KIND, int BLOCK_BYTES>
__global__ void __launch_bounds__(128) mv_q8_4rows(const uint8_t* __restrict__ w, const BlockQ8_1* __restrict__ x, float* __restrict__ y, int n, int rows, size_t stride) {
    __shared__ alignas(4) uint16_t sw[4][((BLOCK_BYTES + 3) / 4) * 2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    const uint8_t* wp = w + size_t(row) * stride;
    const int nb = n / 256;
    float s = 0;
    if constexpr (KIND == MvKind::IQ2_S) {
        for (int blk_i = 0; blk_i < nb; ++blk_i) {
            s += dot_q8_iq2_s_sgrid(wp + size_t(blk_i) * 82, x + blk_i * 8, q38_iq2s_grid, lane);
        }
    } else if constexpr (KIND == MvKind::IQ3_S) {
        for (int blk_i = 0; blk_i < nb; ++blk_i) {
            s += dot_q8_iq3_s_direct(wp + size_t(blk_i) * 110, x + blk_i * 8, lane);
        }
    } else {
        for (int blk_i = 0; blk_i < nb; ++blk_i) {
            load_block_u16<(BLOCK_BYTES + 1) / 2>(sw[warp], wp + size_t(blk_i) * BLOCK_BYTES, lane);
            const uint8_t* p = reinterpret_cast<const uint8_t*>(sw[warp]);
            const BlockQ8_1* xb = x + blk_i * 8;
            if constexpr (KIND == MvKind::Q4_K) s += dot_q8_qk45<false>(p, xb, lane);
            else if constexpr (KIND == MvKind::Q5_K) s += dot_q8_qk45<true>(p, xb, lane);
            else s += dot_q8_q6_k(p, xb, lane);
        }
    }
    s = sum(s);
    if (!lane) y[row] = s;
}

template <MvKind KIND, int BLOCK_BYTES>
__global__ void __launch_bounds__(256) mv_q8_1row(const uint8_t* __restrict__ w, const BlockQ8_1* __restrict__ x, float* __restrict__ y, int n, size_t stride) {
    __shared__ alignas(4) uint16_t sw[8][((BLOCK_BYTES + 3) / 4) * 2];
    __shared__ float wsum[8];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    const uint8_t* wp = w + size_t(row) * stride;
    const int nb = n / 256;
    float s = 0;
    for (int blk_i = warp; blk_i < nb; blk_i += 8) {
        load_block_u16<(BLOCK_BYTES + 1) / 2>(sw[warp], wp + size_t(blk_i) * BLOCK_BYTES, lane);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(sw[warp]);
        const BlockQ8_1* xb = x + blk_i * 8;
        if constexpr (KIND == MvKind::Q4_K) s += dot_q8_qk45<false>(p, xb, lane);
        else if constexpr (KIND == MvKind::Q5_K) s += dot_q8_qk45<true>(p, xb, lane);
        else if constexpr (KIND == MvKind::Q6_K) s += dot_q8_q6_k(p, xb, lane);
        else if constexpr (KIND == MvKind::IQ2_S) s += dot_q8_iq2_s(p, xb, lane);
        else s += dot_q8_iq3_s(p, xb, lane);
    }
    s = sum(s);
    if (!lane) wsum[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? wsum[lane] : 0.f;
        for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (!lane) y[row] = v;
    }
}
__global__ void norm(const float* x,const float* w,float* y,int n,float eps){
    __shared__ float sums[8];
    const float* xr=x+size_t(blockIdx.x)*n; float* yr=y+size_t(blockIdx.x)*n;
    float v=0;for(int i=threadIdx.x;i<n;i+=256)v+=xr[i]*xr[i];v=sum(v);
    if(!(threadIdx.x%32))sums[threadIdx.x/32]=v;__syncthreads();
    float s=0;for(int i=0;i<8;++i)s+=sums[i];float r=rsqrtf(s/n+eps);
    for(int i=threadIdx.x;i<n;i+=256)yr[i]=xr[i]*r*w[i];
}
__global__ void elt(float* x,const float* y,int n,bool sw){int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]=sw?x[i]/(1+expf(-x[i]))*y[i]:x[i]+y[i];}
__global__ void gb(float* al,float* be,const float* dt,const float* a,int n){
    int i=threadIdx.x;if(i<n){
        const size_t off=size_t(blockIdx.x)*n+i;
        float v=al[off]+dt[i];al[off]=(v>20?v:log1pf(expf(v)))*a[i];be[off]=1/(1+expf(-be[off]));
    }
}
__global__ void fin(const float* x,size_t n,int* err){size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n&&!isfinite(x[i]))atomicExch(err,1);}
__global__ void had(const float* x,const float* signs,float* y,bool inverse,bool grouped){
    __shared__ float b[1024];int j=threadIdx.x,index=blockIdx.x*1024+j,src=index;
    // tiled [128,16,3] -> grouped [128,3,16], before signs and transform.
    if(grouped){int head=index/128;src=((head%3)*16+head/3)*128+index%128;}
    b[j]=x[src]*(inverse?1.f:signs[index]);__syncthreads();
    for(int stride=1;stride<1024;stride*=2){float v=b[j],other=b[j^stride];__syncthreads();b[j]=(j&stride)?other-v:v+other;__syncthreads();}
    y[index]=b[j]*(1.f/32)*(inverse?signs[index]:1.f);
}
// Based on Strata q36 k_attn_prep: Q and output gates interleave per head.
__global__ void prep(const float* qg,const float* k,const float* v,const float* qw,const float* kw,float* q,float* gate,
                     half* kc,half* vc,int pos,int ctx,float eps,float base,float* normalized_k=nullptr){
    __shared__ float buf[256],part[8];int head=blockIdx.x,t=blockIdx.y,i=threadIdx.x;bool isq=head<24;
    const float* x=isq?(qg+size_t(t)*12288+head*512):(k+size_t(t)*1024+(head-24)*256);float z=x[i];float s=sum(z*z);
    if(!(i%32))part[i/32]=s;__syncthreads();s=0;for(int j=0;j<8;++j)s+=part[j];
    buf[i]=z*rsqrtf(s/256+eps)*(isq?qw[i]:kw[i]);__syncthreads();z=buf[i];
    if(i<64){int p=i%32;float angle=(pos+t)*powf(base,-float(p)/32),sn,cs;sincosf(angle,&sn,&cs);z=i<32?buf[i]*cs-buf[i+32]*sn:buf[i-32]*sn+buf[i]*cs;}
    if(isq){q[size_t(t)*6144+head*256+i]=z;gate[size_t(t)*6144+head*256+i]=qg[size_t(t)*12288+head*512+256+i];}
    else{int kh=head-24;if(normalized_k)normalized_k[size_t(t)*1024+kh*256+i]=z;
        else{kc[(size_t(kh)*ctx+pos+t)*256+i]=__float2half(z);vc[(size_t(kh)*ctx+pos+t)*256+i]=__float2half(v[size_t(t)*1024+kh*256+i]);}}
}
__global__ void score(const float* q,const half* k,float* s,int pos,int ctx,const int* status){
    if(status&&*status)return;
    int t=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y,lane=threadIdx.x%32;if(t>pos)return;int kh=head/6;
    float a=0;for(int i=lane;i<256;i+=32)a=fmaf(q[head*256+i],__half2float(k[(size_t(kh)*ctx+t)*256+i]),a);
    a=sum(a);if(!lane)s[size_t(head)*ctx+t]=a/16;
}
__global__ void attend(const float* scores,const half* v,const float* gate,float* out,int pos,int ctx,const int* status){
    if(status&&*status)return;
    // One block per query head; stable softmax then each lane's value component.
    __shared__ float tmp[256],maxi,denom;int head=blockIdx.x,i=threadIdx.x;const float* s=scores+size_t(head)*ctx;
    float mx=-INFINITY;for(int t=i;t<=pos;t+=256)mx=fmaxf(mx,s[t]);tmp[i]=mx;__syncthreads();
    for(int d=128;d;d/=2){if(i<d)tmp[i]=fmaxf(tmp[i],tmp[i+d]);__syncthreads();}if(!i)maxi=tmp[0];__syncthreads();
    float z=0;for(int t=i;t<=pos;t+=256)z+=expf(s[t]-maxi);tmp[i]=z;__syncthreads();
    for(int d=128;d;d/=2){if(i<d)tmp[i]+=tmp[i+d];__syncthreads();}if(!i)denom=tmp[0];__syncthreads();
    float y=0;for(int t=0;t<=pos;++t)y=fmaf(expf(s[t]-maxi)/denom,__half2float(v[(size_t(head/6)*ctx+t)*256+i]),y);
    out[head*256+i]=y/(1+expf(-gate[head*256+i]));
}
}
void preload_ops(){
    cudaFuncAttributes a{};
    ck(cudaFuncGetAttributes(&a,deq));ck(cudaFuncGetAttributes(&a,mv));ck(cudaFuncGetAttributes(&a,norm));
    ck(cudaFuncGetAttributes(&a,elt));ck(cudaFuncGetAttributes(&a,gb));ck(cudaFuncGetAttributes(&a,fin));
    ck(cudaFuncGetAttributes(&a,had));ck(cudaFuncGetAttributes(&a,prep));ck(cudaFuncGetAttributes(&a,score));ck(cudaFuncGetAttributes(&a,attend));
}
void decode(int type,const void* p,float* y,int n,cudaStream_t s){shape(type,n,1);if(!p||!y)throw std::runtime_error("null decoder input");deq<<<(n+255)/256,256,0,s>>>(type,(const uint8_t*)p,y,n);ck(cudaGetLastError());}
template <MvKind KIND, int BLOCK_BYTES>
void launch_mv_256(const uint8_t* wp, const float* x, float* y, int n, int rows, size_t rb, cudaStream_t s) {
    if (rows < 128 || (n >= 10240 && rows <= 8192)) {
        mv_1row<KIND, BLOCK_BYTES><<<rows, 256, 0, s>>>(wp, x, y, n, rb);
    } else {
        mv_8rows<KIND, BLOCK_BYTES><<<(rows + 3) / 4, 128, 0, s>>>(wp, x, y, n, rows, rb);
    }
}

template <MvKind KIND, int BLOCK_BYTES>
void launch_mv_q8_256(const uint8_t* wp, const BlockQ8_1* x, float* y, int n, int rows, size_t rb, cudaStream_t s) {
    if (rows < 256) {
        mv_q8_1row<KIND, BLOCK_BYTES><<<rows, 256, 0, s>>>(wp, x, y, n, rb);
    } else if constexpr (KIND == MvKind::IQ2_S) {
        mv_q8_iq2s_8rows<<<(rows + 7) / 8, 256, 0, s>>>(wp, x, y, n, rows, rb);
    } else {
        mv_q8_4rows<KIND, BLOCK_BYTES><<<(rows + 3) / 4, 128, 0, s>>>(wp, x, y, n, rows, rb);
    }
}

void quantize_q8_1(const float* input, BlockQ8_1* output, int n, cudaStream_t stream) {
    if (n <= 0 || n % 32 || !input || !output) throw std::runtime_error("invalid quantize_q8_1 args");
    const int nb = n / 32;
    quantize_q8_1_kernel<<<(nb + 7) / 8, 256, 0, stream>>>(input, output, n);
    ck(cudaGetLastError());
}

void matvec_q8_1(int type, const void* w, const BlockQ8_1* x_q8, const float* x_f32, float* y, int n, int rows, cudaStream_t s) {
    shape(type, n, rows);
    if (!w || !x_q8 || !x_f32 || !y) throw std::runtime_error("null matrix input");
    const size_t rb = row_bytes(type, n);
    const auto* wp = (const uint8_t*)w;
    switch (type) {
        case 12: launch_mv_q8_256<MvKind::Q4_K, 144>(wp, x_q8, y, n, rows, rb, s); break;
        case 13: launch_mv_q8_256<MvKind::Q5_K, 176>(wp, x_q8, y, n, rows, rb, s); break;
        case 14: launch_mv_q8_256<MvKind::Q6_K, 210>(wp, x_q8, y, n, rows, rb, s); break;
        case 21: launch_mv_q8_256<MvKind::IQ3_S, 110>(wp, x_q8, y, n, rows, rb, s); break;
        case 22: launch_mv_q8_256<MvKind::IQ2_S, 82>(wp, x_q8, y, n, rows, rb, s); break;
        default: matvec(type, w, x_f32, y, n, rows, s); break;
    }
    ck(cudaGetLastError());
}

void matvec(int type,const void* w,const float* x,float* y,int n,int rows,cudaStream_t s){
    shape(type,n,rows);if(!w||!x||!y)throw std::runtime_error("null matrix input");
    const dim3 grid((rows+7)/8);const size_t rb=row_bytes(type,n);const auto* wp=(const uint8_t*)w;
    switch(type){
        case 8:  mv_q8_0<<<grid,256,0,s>>>(wp,x,y,n,rows,rb); break;
        case 12: launch_mv_256<MvKind::Q4_K, 144>(wp,x,y,n,rows,rb,s); break;
        case 13: launch_mv_256<MvKind::Q5_K, 176>(wp,x,y,n,rows,rb,s); break;
        case 14: launch_mv_256<MvKind::Q6_K, 210>(wp,x,y,n,rows,rb,s); break;
        case 21: launch_mv_256<MvKind::IQ3_S, 110>(wp,x,y,n,rows,rb,s); break;
        case 22: launch_mv_256<MvKind::IQ2_S, 82>(wp,x,y,n,rows,rb,s); break;
        default: mv<<<grid,256,0,s>>>(type,wp,x,y,n,rows,rb); break;
    }
    ck(cudaGetLastError());
}
void rms(const float* x,const float* w,float* y,int n,float eps,cudaStream_t s){norm<<<1,256,0,s>>>(x,w,y,n,eps);ck(cudaGetLastError());}
void rms_batch(const float* x,const float* w,float* y,int rows,int n,float eps,cudaStream_t s){if(rows<=0)return;norm<<<rows,256,0,s>>>(x,w,y,n,eps);ck(cudaGetLastError());}
void add(float* x,const float* y,int n,cudaStream_t s){elt<<<(n+255)/256,256,0,s>>>(x,y,n,false);ck(cudaGetLastError());}
void swiglu(float* x,const float* y,int n,cudaStream_t s){elt<<<(n+255)/256,256,0,s>>>(x,y,n,true);ck(cudaGetLastError());}
void gdn_gates(float* al,float* be,const float* dt,const float* a,int n,cudaStream_t s){gb<<<1,256,0,s>>>(al,be,dt,a,n);ck(cudaGetLastError());}
void gdn_gates_batch(float* al,float* be,const float* dt,const float* a,int rows,int n,cudaStream_t s){if(rows<=0)return;gb<<<rows,256,0,s>>>(al,be,dt,a,n);ck(cudaGetLastError());}
void finite(const float* x,size_t n,int* e,cudaStream_t s){fin<<<unsigned((n+255)/256),256,0,s>>>(x,n,e);ck(cudaGetLastError());}
void hadamard(const float* x,const float* signs,float* y,int n,bool inverse,bool grouped,cudaStream_t s){
    if(n<=0||n%1024||!x||!signs||!y||x==y||(grouped&&(n!=6144||inverse)))throw std::runtime_error("invalid Hadamard arguments");
    had<<<n/1024,1024,0,s>>>(x,signs,y,inverse,grouped);ck(cudaGetLastError());}
void attention_prepare(const float* qg,const float* k,const float* v,const float* qw,const float* kw,float* q,float* gate,half* kc,half* vc,int pos,int ctx,float eps,float base,cudaStream_t s){
    if(pos<0||pos>=ctx)throw std::runtime_error("attention position out of range");prep<<<28,256,0,s>>>(qg,k,v,qw,kw,q,gate,kc,vc,pos,ctx,eps,base);ck(cudaGetLastError());}
void attention_apply(const float* q,const float* gate,const half* kc,const half* vc,float* out,float* scores,int pos,int ctx,cudaStream_t s,const int* status){
    if(pos<0||pos>=ctx)throw std::runtime_error("attention position out of range");score<<<dim3((pos+8)/8,24),256,0,s>>>(q,kc,scores,pos,ctx,status);attend<<<24,256,0,s>>>(scores,vc,gate,out,pos,ctx,status);ck(cudaGetLastError());}
void attention_normalize(const float* qg,const float* k,const float* qw,const float* kw,float* q,float* gate,float* nk,int pos,float eps,float base,cudaStream_t s){
    if(pos<0||!qg||!k||!qw||!kw||!q||!gate||!nk)throw std::runtime_error("invalid attention normalization");
    prep<<<28,256,0,s>>>(qg,k,nullptr,qw,kw,q,gate,nullptr,nullptr,pos,0,eps,base,nk);ck(cudaGetLastError());}
void attention_normalize_batch(const float* qg,const float* k,const float* qw,const float* kw,float* q,float* gate,float* nk,int pos0,int n_tokens,float eps,float base,cudaStream_t s){
    if(n_tokens<=0)return;
    if(pos0<0||!qg||!k||!qw||!kw||!q||!gate||!nk)throw std::runtime_error("invalid attention normalization batch");
    prep<<<dim3(28,unsigned(n_tokens)),256,0,s>>>(qg,k,nullptr,qw,kw,q,gate,nullptr,nullptr,pos0,0,eps,base,nk);ck(cudaGetLastError());}
}
