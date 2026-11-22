// Packed scalar decoders follow Strata iq_kernels.cu (ggml MIT formats);
// PQ2_0 follows PrismML 0.4.1 commit 1965755ceee6738b9a04ceaf5d30e6305d6ccd00.
// See README.md and LICENSE-formats.txt. No model/graph/runtime code is linked.
#include "ops.cuh"
#include "iq_tables.hpp"
#include "iq_decode.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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
__global__ void __launch_bounds__(128) mv_pq2_0(const uint8_t* __restrict__ w,const float* __restrict__ x,float* __restrict__ y,int n,int rows,size_t stride){
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    const int row=blockIdx.x*4+warp;if(row>=rows)return;
    const uint8_t* wp=w+size_t(row)*stride;const int nb=n/128;float s=0;
    for(int b=0;b<nb;++b){
        const uint8_t* blk=wp+size_t(b)*34;
        const float d=h(blk);
        const uint8_t qb=__ldg(blk+2+lane);
        const float4 xv=__ldg(reinterpret_cast<const float4*>(x+b*128+lane*4));
        const float c0=float(int(qb&3)-1);
        const float c1=float(int((qb>>2)&3)-1);
        const float c2=float(int((qb>>4)&3)-1);
        const float c3=float(int((qb>>6)&3)-1);
        const float acc=fmaf(c0,xv.x,fmaf(c1,xv.y,fmaf(c2,xv.z,c3*xv.w)));
        s=fmaf(d,acc,s);
    }
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

__device__ __forceinline__ void quantize_warp_q8_1(float xv, BlockQ8_1* dst, int lane) {
    float amax = fabsf(xv);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    const int q = __float2int_rn(xv * inv_d);
    int qsum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xffffffffu, qsum, o);
    const int b1 = __shfl_down_sync(0xffffffffu, q, 1) & 0xFF;
    const int b2 = __shfl_down_sync(0xffffffffu, q, 2) & 0xFF;
    const int b3 = __shfl_down_sync(0xffffffffu, q, 3) & 0xFF;
    if ((lane & 3) == 0) {
        dst->qs[lane >> 2] = (q & 0xFF) | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    if (lane == 0) {
        dst->d = d;
        dst->s = d * float(qsum);
    }
}

__global__ void __launch_bounds__(256) quantize_q8_1_kernel(const float* __restrict__ x, BlockQ8_1* __restrict__ y, int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 8 + warp;
    if (blk * 32 >= n) return;
    const float xv = __ldg(x + blk * 32 + lane);
    quantize_warp_q8_1(xv, y + blk, lane);
}

__global__ void __launch_bounds__(256) quantize_q8_1_batch_kernel(
    const float* __restrict__ x, BlockQ8_1* __restrict__ out, int n_tokens, int n) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 8 + warp, tok = blockIdx.y;
    const int nb = n >> 5;
    if (tok >= n_tokens || blk >= nb) return;
    quantize_warp_q8_1(x[size_t(tok) * n + blk * 32 + lane], out + size_t(tok) * nb + blk, lane);
}
__global__ void __launch_bounds__(256) rms_quantize_q8_1_kernel(
    const float* x, const float* w, float* out_f32, BlockQ8_1* out_q8, int n, float eps) {
    __shared__ float sums[8];
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) { const float xi = x[i]; v += xi * xi; }
    v = sum(v);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) s += sums[i];
    const float r = rsqrtf(s / float(n) + eps);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = n >> 5;
    for (int blk = warp; blk < nb; blk += 8) {
        const int idx = blk * 32 + lane;
        const float yv = x[idx] * r * w[idx];
        out_f32[idx] = yv;
        quantize_warp_q8_1(yv, out_q8 + blk, lane);
    }
}

__global__ void __launch_bounds__(256) swiglu_quantize_q8_1_kernel(float* gate, const float* up, BlockQ8_1* out_q8, int n) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, blk = blockIdx.x * 8 + warp;
    if (blk * 32 >= n) return;
    const int idx = blk * 32 + lane;
    const float g = gate[idx], yv = (g / (1.0f + expf(-g))) * up[idx];
    gate[idx] = yv;
    quantize_warp_q8_1(yv, out_q8 + blk, lane);
}

struct MvTargets {
    const uint8_t* w0; float* y0; int rows0;
    const uint8_t* w1; float* y1; int rows12;
    const uint8_t* w2; float* y2; int total_rows;
};

__device__ __forceinline__ const uint8_t* target_row_ptr(const MvTargets& t, int r, size_t stride, float*& y_out) {
    if (r < t.rows0) { y_out = t.y0 + r; return t.w0 + size_t(r) * stride; }
    if (r < t.rows12) { const int r1 = r - t.rows0; y_out = t.y1 + r1; return t.w1 + size_t(r1) * stride; }
    const int r2 = r - t.rows12; y_out = t.y2 + r2; return t.w2 + size_t(r2) * stride;
}

__device__ __forceinline__ uint32_t ldg_u32_from_u16(const uint8_t* __restrict__ ptr) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(ptr);
    return uint32_t(__ldg(p16)) | (uint32_t(__ldg(p16 + 1)) << 16);
}

// All codebook bytes in q38_iq2s_grid and q38_iq3s_grid lie in [1, 43], so
// two's-complement negation (mag ^ mask) + s01 never carries across byte lanes.
__device__ __forceinline__ int32_t apply_signs4(uint32_t mag4, uint32_t s4) {
    const uint32_t s01 = (s4 * 0x00204081u) & 0x01010101u;
    return int32_t((mag4 ^ (s01 * 0xFFu)) + s01);
}

template <bool FIVE>
__device__ __forceinline__ float eval_qk45_sublane(
    uint4 hdr, uint4 q16, uint4 qh16, int sub_lane,
    float2 ds0, float2 ds1, int2 x0_a, int2 x0_b, int2 x1_a, int2 x1_b) {
    const float d0 = __half2float(__ushort_as_half(uint16_t(hdr.x & 0xFFFFu)));
    const float dm0 = __half2float(__ushort_as_half(uint16_t(hdr.x >> 16)));
    const int b_pair = sub_lane >> 1;
    int d_lo, m_lo, d_hi, m_hi;
    if (b_pair < 2) {
        const int sh = b_pair * 16;
        const uint32_t sy = hdr.y >> sh, sz = hdr.z >> sh;
        d_lo = int(sy & 0x3Fu); d_hi = int((sy >> 8) & 0x3Fu);
        m_lo = int(sz & 0x3Fu); m_hi = int((sz >> 8) & 0x3Fu);
    } else {
        const int sh = (b_pair - 2) * 16;
        const uint32_t sy = hdr.y >> sh, sz = hdr.z >> sh, sw = hdr.w >> sh;
        d_lo = int((sw & 0x0Fu) | (((sy >> 6) & 0x03u) << 4));
        m_lo = int(((sw >> 4) & 0x0Fu) | (((sz >> 6) & 0x03u) << 4));
        d_hi = int(((sw >> 8) & 0x0Fu) | (((sy >> 14) & 0x03u) << 4));
        m_hi = int(((sw >> 12) & 0x0Fu) | (((sz >> 14) & 0x03u) << 4));
    }
    uint32_t lo0 = q16.x & 0x0F0F0F0Fu, hi0 = (q16.x >> 4) & 0x0F0F0F0Fu;
    uint32_t lo1 = q16.y & 0x0F0F0F0Fu, hi1 = (q16.y >> 4) & 0x0F0F0F0Fu;
    uint32_t lo2 = q16.z & 0x0F0F0F0Fu, hi2 = (q16.z >> 4) & 0x0F0F0F0Fu;
    uint32_t lo3 = q16.w & 0x0F0F0F0Fu, hi3 = (q16.w >> 4) & 0x0F0F0F0Fu;
    if constexpr (FIVE) {
        const int b0 = b_pair * 2, b1 = b0 + 1;
        lo0 |= ((qh16.x >> b0) & 0x01010101u) << 4; hi0 |= ((qh16.x >> b1) & 0x01010101u) << 4;
        lo1 |= ((qh16.y >> b0) & 0x01010101u) << 4; hi1 |= ((qh16.y >> b1) & 0x01010101u) << 4;
        lo2 |= ((qh16.z >> b0) & 0x01010101u) << 4; hi2 |= ((qh16.z >> b1) & 0x01010101u) << 4;
        lo3 |= ((qh16.w >> b0) & 0x01010101u) << 4; hi3 |= ((qh16.w >> b1) & 0x01010101u) << 4;
    }
    int32_t dp0 = __dp4a(int32_t(lo3), x0_b.y, __dp4a(int32_t(lo2), x0_b.x, __dp4a(int32_t(lo1), x0_a.y, __dp4a(int32_t(lo0), x0_a.x, 0))));
    int32_t dp1 = __dp4a(int32_t(hi3), x1_b.y, __dp4a(int32_t(hi2), x1_b.x, __dp4a(int32_t(hi1), x1_a.y, __dp4a(int32_t(hi0), x1_a.x, 0))));
    float res = d0 * (float(d_lo * dp0) * ds0.x + float(d_hi * dp1) * ds1.x);
    if ((sub_lane & 1) == 0) res -= dm0 * (float(m_lo) * ds0.y + float(m_hi) * ds1.y);
    return res;
}

template <bool FIVE>
__device__ __forceinline__ void step_qk45_4blk_2row(
    const uint8_t* wp0, const uint8_t* wp1, const BlockQ8_1* __restrict__ x,
    int blk_i, int sub_lane, float& s0, float& s1) {
    constexpr int BLOCK_BYTES = FIVE ? 176 : 144;
    constexpr int QS_OFF = FIVE ? 48 : 16;
    const int b_pair = sub_lane >> 1;
    const int half_off = (sub_lane & 1) * 4;
    const uint8_t* p0 = wp0 + size_t(blk_i) * BLOCK_BYTES;
    const uint8_t* p1 = wp1 + size_t(blk_i) * BLOCK_BYTES;
    const uint4 hdr0 = __ldg(reinterpret_cast<const uint4*>(p0));
    const uint4 q16_0 = __ldg(reinterpret_cast<const uint4*>(p0 + QS_OFF + sub_lane * 16));
    const uint4 hdr1 = __ldg(reinterpret_cast<const uint4*>(p1));
    const uint4 q16_1 = __ldg(reinterpret_cast<const uint4*>(p1 + QS_OFF + sub_lane * 16));
    uint4 qh0 = {0, 0, 0, 0}, qh1 = {0, 0, 0, 0};
    if constexpr (FIVE) {
        qh0 = __ldg(reinterpret_cast<const uint4*>(p0 + 16 + (sub_lane & 1) * 16));
        qh1 = __ldg(reinterpret_cast<const uint4*>(p1 + 16 + (sub_lane & 1) * 16));
    }
    const BlockQ8_1& x0 = x[blk_i * 8 + b_pair * 2];
    const BlockQ8_1& x1 = x[blk_i * 8 + b_pair * 2 + 1];
    const float2 ds0 = __ldg(reinterpret_cast<const float2*>(&x0.d));
    const float2 ds1 = __ldg(reinterpret_cast<const float2*>(&x1.d));
    const int2 x0_a = __ldg(reinterpret_cast<const int2*>(&x0.qs[half_off]));
    const int2 x0_b = __ldg(reinterpret_cast<const int2*>(&x0.qs[half_off + 2]));
    const int2 x1_a = __ldg(reinterpret_cast<const int2*>(&x1.qs[half_off]));
    const int2 x1_b = __ldg(reinterpret_cast<const int2*>(&x1.qs[half_off + 2]));
    s0 += eval_qk45_sublane<FIVE>(hdr0, q16_0, qh0, sub_lane, ds0, ds1, x0_a, x0_b, x1_a, x1_b);
    s1 += eval_qk45_sublane<FIVE>(hdr1, q16_1, qh1, sub_lane, ds0, ds1, x0_a, x0_b, x1_a, x1_b);
}

template <bool FIVE>
__global__ void __launch_bounds__(256, 4) mv_q8_qk45_4blk_2row(
    MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * 16 + warp * 2;
    if (r0 >= t.total_rows) return;
    const int r1 = r0 + 1;
    float *y_out0 = nullptr, *y_out1 = nullptr;
    const uint8_t* wp0 = target_row_ptr(t, r0, stride, y_out0);
    const bool has_r1 = (r1 < t.total_rows);
    const uint8_t* wp1 = has_r1 ? target_row_ptr(t, r1, stride, y_out1) : wp0;
    const int blk_off = lane >> 3, sub_lane = lane & 7;
    const int nb = n >> 8, nb4 = nb & ~3;
    float s0 = 0.0f, s1 = 0.0f;
    int blk_base = 0;
    for (; blk_base < nb4; blk_base += 4) {
        step_qk45_4blk_2row<FIVE>(wp0, wp1, x, blk_base + blk_off, sub_lane, s0, s1);
    }
    if (blk_base + blk_off < nb) {
        step_qk45_4blk_2row<FIVE>(wp0, wp1, x, blk_base + blk_off, sub_lane, s0, s1);
    }
    s0 = sum(s0);
    s1 = sum(s1);
    if (lane == 0) { *y_out0 = s0; if (has_r1) *y_out1 = s1; }
}

template <bool FIVE>
__global__ void __launch_bounds__(128) mv_q8_qk45_4blk_2row_128(
    MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * 8 + warp * 2;
    if (r0 >= t.total_rows) return;
    const int r1 = r0 + 1;
    float *y_out0 = nullptr, *y_out1 = nullptr;
    const uint8_t* wp0 = target_row_ptr(t, r0, stride, y_out0);
    const bool has_r1 = (r1 < t.total_rows);
    const uint8_t* wp1 = has_r1 ? target_row_ptr(t, r1, stride, y_out1) : wp0;
    const int blk_off = lane >> 3, sub_lane = lane & 7;
    const int nb = n >> 8, nb4 = nb & ~3;
    float s0 = 0.0f, s1 = 0.0f;
    int blk_base = 0;
    for (; blk_base < nb4; blk_base += 4) {
        step_qk45_4blk_2row<FIVE>(wp0, wp1, x, blk_base + blk_off, sub_lane, s0, s1);
    }
    if (blk_base + blk_off < nb) {
        step_qk45_4blk_2row<FIVE>(wp0, wp1, x, blk_base + blk_off, sub_lane, s0, s1);
    }
    s0 = sum(s0);
    s1 = sum(s1);
    if (lane == 0) { *y_out0 = s0; if (has_r1) *y_out1 = s1; }
}

template <bool FIVE, int NCOLS>
__global__ void __launch_bounds__(256, 4) mv_q8_qk45_batch(
    const uint8_t* __restrict__ wp, const BlockQ8_1* __restrict__ x, float* __restrict__ y,
    int n, int rows, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint8_t* row_ptr = wp + size_t(row) * stride;
    const int blk_off = lane >> 3, sub_lane = lane & 7;
    const int b_pair = sub_lane >> 1, half_off = (sub_lane & 1) * 4;
    const int nb = n >> 8, x_stride = n >> 5;
    constexpr int BLOCK_BYTES = FIVE ? 176 : 144;
    constexpr int QS_OFF = FIVE ? 48 : 16;
    float acc[NCOLS];
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) acc[c] = 0.0f;
    for (int blk_i = blk_off; blk_i < nb; blk_i += 4) {
        const uint8_t* p = row_ptr + size_t(blk_i) * BLOCK_BYTES;
        const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(p));
        const uint4 q16 = __ldg(reinterpret_cast<const uint4*>(p + QS_OFF + sub_lane * 16));
        uint4 qh16 = {0, 0, 0, 0};
        if constexpr (FIVE) qh16 = __ldg(reinterpret_cast<const uint4*>(p + 16 + (sub_lane & 1) * 16));
        const float d0 = __half2float(__ushort_as_half(uint16_t(hdr.x & 0xFFFFu)));
        const float dm0 = __half2float(__ushort_as_half(uint16_t(hdr.x >> 16)));
        int d_lo, m_lo, d_hi, m_hi;
        if (b_pair < 2) {
            const int sh = b_pair * 16;
            const uint32_t sy = hdr.y >> sh, sz = hdr.z >> sh;
            d_lo = int(sy & 0x3Fu); d_hi = int((sy >> 8) & 0x3Fu);
            m_lo = int(sz & 0x3Fu); m_hi = int((sz >> 8) & 0x3Fu);
        } else {
            const int sh = (b_pair - 2) * 16;
            const uint32_t sy = hdr.y >> sh, sz = hdr.z >> sh, sw = hdr.w >> sh;
            d_lo = int((sw & 0x0Fu) | (((sy >> 6) & 0x03u) << 4));
            m_lo = int(((sw >> 4) & 0x0Fu) | (((sz >> 6) & 0x03u) << 4));
            d_hi = int(((sw >> 8) & 0x0Fu) | (((sy >> 14) & 0x03u) << 4));
            m_hi = int(((sw >> 12) & 0x0Fu) | (((sz >> 14) & 0x03u) << 4));
        }
        uint32_t lo0 = q16.x & 0x0F0F0F0Fu, hi0 = (q16.x >> 4) & 0x0F0F0F0Fu;
        uint32_t lo1 = q16.y & 0x0F0F0F0Fu, hi1 = (q16.y >> 4) & 0x0F0F0F0Fu;
        uint32_t lo2 = q16.z & 0x0F0F0F0Fu, hi2 = (q16.z >> 4) & 0x0F0F0F0Fu;
        uint32_t lo3 = q16.w & 0x0F0F0F0Fu, hi3 = (q16.w >> 4) & 0x0F0F0F0Fu;
        if constexpr (FIVE) {
            const int b0 = b_pair * 2, b1 = b0 + 1;
            lo0 |= ((qh16.x >> b0) & 0x01010101u) << 4; hi0 |= ((qh16.x >> b1) & 0x01010101u) << 4;
            lo1 |= ((qh16.y >> b0) & 0x01010101u) << 4; hi1 |= ((qh16.y >> b1) & 0x01010101u) << 4;
            lo2 |= ((qh16.z >> b0) & 0x01010101u) << 4; hi2 |= ((qh16.z >> b1) & 0x01010101u) << 4;
            lo3 |= ((qh16.w >> b0) & 0x01010101u) << 4; hi3 |= ((qh16.w >> b1) & 0x01010101u) << 4;
        }
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const BlockQ8_1* xc = x + size_t(c) * x_stride + blk_i * 8 + b_pair * 2;
            const float2 ds0 = __ldg(reinterpret_cast<const float2*>(&xc[0].d));
            const float2 ds1 = __ldg(reinterpret_cast<const float2*>(&xc[1].d));
            const int2 x0_a = __ldg(reinterpret_cast<const int2*>(&xc[0].qs[half_off]));
            const int2 x0_b = __ldg(reinterpret_cast<const int2*>(&xc[0].qs[half_off + 2]));
            const int2 x1_a = __ldg(reinterpret_cast<const int2*>(&xc[1].qs[half_off]));
            const int2 x1_b = __ldg(reinterpret_cast<const int2*>(&xc[1].qs[half_off + 2]));
            const int32_t dp0 = __dp4a(int32_t(lo3), x0_b.y, __dp4a(int32_t(lo2), x0_b.x, __dp4a(int32_t(lo1), x0_a.y, __dp4a(int32_t(lo0), x0_a.x, 0))));
            const int32_t dp1 = __dp4a(int32_t(hi3), x1_b.y, __dp4a(int32_t(hi2), x1_b.x, __dp4a(int32_t(hi1), x1_a.y, __dp4a(int32_t(hi0), x1_a.x, 0))));
            float res = d0 * (float(d_lo * dp0) * ds0.x + float(d_hi * dp1) * ds1.x);
            if ((sub_lane & 1) == 0) res -= dm0 * (float(m_lo) * ds0.y + float(m_hi) * ds1.y);
            acc[c] += res;
        }
    }
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        const float s = sum(acc[c]);
        if (lane == 0) y[size_t(c) * rows + row] = s;
    }
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
    float acc = d0 * (float(d_lo * dp0) * x0.d + float(d_hi * dp1) * x1.d);
    if ((lane & 7) == 0) acc -= dmin0 * (float(m_lo) * x0.s + float(m_hi) * x1.s);
    return acc;
}

// Q6_K uses 210 bytes/block (210 % 4 == 2): odd blocks are 2-byte aligned, not 4-byte aligned.
__device__ __forceinline__ float dot_q8_q6_k_direct(const uint8_t* __restrict__ p, const BlockQ8_1* __restrict__ xb, int lane) {
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p + 208))));
    const int group = lane >> 3;
    const int sub = lane & 7;
    const int j = sub * 4;
    float s = 0.0f;
    #pragma unroll
    for (int ip = 0; ip < 2; ++ip) {
        const uint32_t ql = ldg_u32_from_u16(p + ip * 64 + (group & 1) * 32 + j);
        const uint32_t qh = ldg_u32_from_u16(p + 128 + ip * 32 + j);
        const uint32_t lo4 = (group >= 2) ? ((ql >> 4) & 0x0F0F0F0Fu) : (ql & 0x0F0F0F0Fu);
        const uint32_t hi2 = (qh >> (2 * group)) & 0x03030303u;
        const int32_t w4 = __vsub4(int32_t(lo4 | (hi2 << 4)), 0x20202020);
        const int scale = int(__ldg(reinterpret_cast<const int8_t*>(p + 192 + ip * 8 + group * 2 + (sub >> 2))));
        const BlockQ8_1& x0 = xb[ip * 4 + group];
        const int32_t dp = __dp4a(w4, __ldg(&x0.qs[sub]), 0);
        s += float(scale * dp) * __ldg(&x0.d);
    }
    return d * s;
}

__device__ __forceinline__ void accum_q8_q6_k_2rows(
    const uint8_t* __restrict__ p0, const uint8_t* __restrict__ p1,
    const BlockQ8_1* __restrict__ xb, int lane, float& s0, float& s1) {
    const float d0 = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p0 + 208))));
    const float d1 = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p1 + 208))));
    const int group = lane >> 3;
    const int sub = lane & 7;
    const int j = sub * 4;
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll
    for (int ip = 0; ip < 2; ++ip) {
        const BlockQ8_1& x0 = xb[ip * 4 + group];
        const int32_t xq = __ldg(&x0.qs[sub]);
        const float xd = __ldg(&x0.d);
        const int ql_off = ip * 64 + (group & 1) * 32 + j;
        const int qh_off = 128 + ip * 32 + j;
        const int sc_off = 192 + ip * 8 + group * 2 + (sub >> 2);

        const uint32_t ql0 = ldg_u32_from_u16(p0 + ql_off);
        const uint32_t qh0 = ldg_u32_from_u16(p0 + qh_off);
        const int sc0 = int(__ldg(reinterpret_cast<const int8_t*>(p0 + sc_off)));

        const uint32_t ql1 = ldg_u32_from_u16(p1 + ql_off);
        const uint32_t qh1 = ldg_u32_from_u16(p1 + qh_off);
        const int sc1 = int(__ldg(reinterpret_cast<const int8_t*>(p1 + sc_off)));

        const uint32_t lo4_0 = (group >= 2) ? ((ql0 >> 4) & 0x0F0F0F0Fu) : (ql0 & 0x0F0F0F0Fu);
        const uint32_t hi2_0 = (qh0 >> (2 * group)) & 0x03030303u;
        const int32_t w4_0 = __vsub4(int32_t(lo4_0 | (hi2_0 << 4)), 0x20202020);
        a0 += float(sc0 * __dp4a(w4_0, xq, 0)) * xd;

        const uint32_t lo4_1 = (group >= 2) ? ((ql1 >> 4) & 0x0F0F0F0Fu) : (ql1 & 0x0F0F0F0Fu);
        const uint32_t hi2_1 = (qh1 >> (2 * group)) & 0x03030303u;
        const int32_t w4_1 = __vsub4(int32_t(lo4_1 | (hi2_1 << 4)), 0x20202020);
        a1 += float(sc1 * __dp4a(w4_1, xq, 0)) * xd;
    }
    s0 += d0 * a0;
    s1 += d1 * a1;
}

__global__ void __launch_bounds__(256, 4) mv_q8_q6k_16rows(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = blockIdx.x * 16 + warp * 2;
    if (r0 >= t.total_rows) return;
    float *y_out0 = nullptr, *y_out1 = nullptr;
    const uint8_t* wp0 = target_row_ptr(t, r0, stride, y_out0);
    const bool has_r1 = (r0 + 1 < t.total_rows);
    const uint8_t* wp1 = has_r1 ? target_row_ptr(t, r0 + 1, stride, y_out1) : wp0;
    const int nb = n >> 8;
    float s0 = 0.0f, s1 = 0.0f;
    int blk_i = 0;
    for (; blk_i + 1 < nb; blk_i += 2) {
        accum_q8_q6_k_2rows(wp0 + size_t(blk_i) * 210, wp1 + size_t(blk_i) * 210, x + blk_i * 8, lane, s0, s1);
        accum_q8_q6_k_2rows(wp0 + size_t(blk_i + 1) * 210, wp1 + size_t(blk_i + 1) * 210, x + (blk_i + 1) * 8, lane, s0, s1);
    }
    if (blk_i < nb) accum_q8_q6_k_2rows(wp0 + size_t(blk_i) * 210, wp1 + size_t(blk_i) * 210, x + blk_i * 8, lane, s0, s1);
    s0 = sum(s0); s1 = sum(s1);
    if (lane == 0) { *y_out0 = s0; if (has_r1) *y_out1 = s1; }
}

template <int NCOLS>
__global__ void __launch_bounds__(256, 4) mv_q8_q6k_batch(
    const uint8_t* __restrict__ wp, const BlockQ8_1* __restrict__ x, float* __restrict__ y,
    int n, int rows, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint8_t* row_ptr = wp + size_t(row) * stride;
    const int nb = n >> 8, x_stride = n >> 5;
    const int ip = lane >> 4, group = (lane & 15) >> 2, sub = lane & 3, j = sub * 8;
    const int ql_off = ip * 64 + (group & 1) * 32 + j;
    const int qh_off = 128 + ip * 32 + j;
    const int sc_off = 192 + ip * 8 + group * 2 + (sub >> 1);
    const int x_blk = ip * 4 + group;
    const int x_qs_off = sub * 2;
    float acc[NCOLS];
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) acc[c] = 0.0f;
    for (int blk_i = 0; blk_i < nb; ++blk_i) {
        const uint8_t* p = row_ptr + size_t(blk_i) * 210;
        const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p + 208))));
        const uint32_t ql_a = ldg_u32_from_u16(p + ql_off);
        const uint32_t qh_a = ldg_u32_from_u16(p + qh_off);
        const int sc = int(__ldg(reinterpret_cast<const int8_t*>(p + sc_off)));
        const uint32_t ql_b = ldg_u32_from_u16(p + ql_off + 4);
        const uint32_t qh_b = ldg_u32_from_u16(p + qh_off + 4);
        const int32_t w4_a = __vsub4(int32_t(((group >= 2) ? ((ql_a >> 4) & 0x0F0F0F0Fu) : (ql_a & 0x0F0F0F0Fu)) | (((qh_a >> (2 * group)) & 0x03030303u) << 4)), 0x20202020);
        const int32_t w4_b = __vsub4(int32_t(((group >= 2) ? ((ql_b >> 4) & 0x0F0F0F0Fu) : (ql_b & 0x0F0F0F0Fu)) | (((qh_b >> (2 * group)) & 0x03030303u) << 4)), 0x20202020);
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const BlockQ8_1& x0 = x[size_t(c) * x_stride + blk_i * 8 + x_blk];
            const float xd = __ldg(&x0.d);
            const int2 xq2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[x_qs_off]));
            acc[c] += d * (float(sc * (__dp4a(w4_a, xq2.x, 0) + __dp4a(w4_b, xq2.y, 0))) * xd);
        }
    }
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        const float s = sum(acc[c]);
        if (lane == 0) y[size_t(c) * rows + row] = s;
    }
}

__device__ __forceinline__ float dot_q8_iq2_s_sgrid(
    const uint8_t* __restrict__ p, const BlockQ8_1* __restrict__ xb, const uint64_t* __restrict__ sgrid, int lane) {
    const int sub = lane >> 2, group = lane & 3;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p)))) * 0.25f;
    const int index = __ldg(p + 2 + lane) | (((__ldg(p + 66 + sub) >> (2 * group)) & 3) << 8);
    const uint64_t grid = sgrid[index];
    const uint8_t sbyte = __ldg(p + 34 + lane);
    const BlockQ8_1& x0 = xb[sub];
    const int2 xv2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[group * 2]));
    const int32_t dp = __dp4a(apply_signs4(uint32_t(grid >> 32), uint32_t(sbyte >> 4)), xv2.y,
                              __dp4a(apply_signs4(uint32_t(grid), sbyte & 0xFu), xv2.x, 0));
    const int scale = (__ldg(p + 74 + sub) >> (4 * (group >> 1))) & 15;
    return d * (0.5f + float(scale)) * float(dp) * __ldg(&x0.d);
}

// IQ2_S uses 82 bytes/block (82 % 4 == 2): strictly 1B and 2B-aligned loads, 4 blocks x 2 rows/warp.
__device__ __forceinline__ float eval_iq2s_subblock(
    uint16_t dh, uint32_t qs4, uint32_t sb4, uint32_t qh_b, uint32_t sc_b,
    const uint64_t* __restrict__ sgrid, float xd, int2 xq01, int2 xq23, int2 xq45, int2 xq67) {
    const uint64_t g0 = sgrid[(qs4 & 0xFFu) | (((qh_b >> 0) & 3u) << 8)];
    const uint64_t g1 = sgrid[((qs4 >> 8) & 0xFFu) | (((qh_b >> 2) & 3u) << 8)];
    const uint64_t g2 = sgrid[((qs4 >> 16) & 0xFFu) | (((qh_b >> 4) & 3u) << 8)];
    const uint64_t g3 = sgrid[(qs4 >> 24) | (((qh_b >> 6) & 3u) << 8)];
    int32_t dp01 = __dp4a(apply_signs4(uint32_t(g0 >> 32), (sb4 >> 4) & 0xFu), xq01.y,
                          __dp4a(apply_signs4(uint32_t(g0), sb4 & 0xFu), xq01.x, 0));
    dp01 = __dp4a(apply_signs4(uint32_t(g1 >> 32), (sb4 >> 12) & 0xFu), xq23.y,
                  __dp4a(apply_signs4(uint32_t(g1), (sb4 >> 8) & 0xFu), xq23.x, dp01));
    int32_t dp23 = __dp4a(apply_signs4(uint32_t(g2 >> 32), (sb4 >> 20) & 0xFu), xq45.y,
                          __dp4a(apply_signs4(uint32_t(g2), (sb4 >> 16) & 0xFu), xq45.x, 0));
    dp23 = __dp4a(apply_signs4(uint32_t(g3 >> 32), sb4 >> 28), xq67.y,
                  __dp4a(apply_signs4(uint32_t(g3), (sb4 >> 24) & 0xFu), xq67.x, dp23));
    const float wsum = (0.5f + float(sc_b & 0xFu)) * float(dp01) + (0.5f + float((sc_b >> 4) & 0xFu)) * float(dp23);
    return __half2float(__ushort_as_half(dh)) * xd * wsum;
}

__device__ __forceinline__ void step_iq2s_4blk_2row(
    const uint8_t* wp0, const uint8_t* wp1, const BlockQ8_1* __restrict__ x,
    const uint64_t* __restrict__ sgrid, int blk_i, int sub, float& s0, float& s1) {
    const uint8_t* p0 = wp0 + size_t(blk_i) * 82;
    const uint8_t* p1 = wp1 + size_t(blk_i) * 82;
    const uint16_t dh0 = __ldg(reinterpret_cast<const uint16_t*>(p0));
    const uint32_t qs0 = ldg_u32_from_u16(p0 + 2 + sub * 4);
    const uint32_t sb0 = ldg_u32_from_u16(p0 + 34 + sub * 4);
    const uint32_t qh0 = __ldg(p0 + 66 + sub), sc0 = __ldg(p0 + 74 + sub);

    const uint16_t dh1 = __ldg(reinterpret_cast<const uint16_t*>(p1));
    const uint32_t qs1 = ldg_u32_from_u16(p1 + 2 + sub * 4);
    const uint32_t sb1 = ldg_u32_from_u16(p1 + 34 + sub * 4);
    const uint32_t qh1 = __ldg(p1 + 66 + sub), sc1 = __ldg(p1 + 74 + sub);

    const BlockQ8_1& x0 = x[blk_i * 8 + sub];
    const float xd = __ldg(&x0.d) * 0.25f;
    const int2 xq01 = __ldg(reinterpret_cast<const int2*>(&x0.qs[0]));
    const int2 xq23 = __ldg(reinterpret_cast<const int2*>(&x0.qs[2]));
    const int2 xq45 = __ldg(reinterpret_cast<const int2*>(&x0.qs[4]));
    const int2 xq67 = __ldg(reinterpret_cast<const int2*>(&x0.qs[6]));

    s0 += eval_iq2s_subblock(dh0, qs0, sb0, qh0, sc0, sgrid, xd, xq01, xq23, xq45, xq67);
    s1 += eval_iq2s_subblock(dh1, qs1, sb1, qh1, sc1, sgrid, xd, xq01, xq23, xq45, xq67);
}

__global__ void __launch_bounds__(256, 4) mv_q8_iq2s_4blk_2row(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    __shared__ uint64_t sgrid[1024];
    #pragma unroll
    for (int i = threadIdx.x; i < 1024; i += 256) sgrid[i] = __ldg(&q38_iq2s_grid[i]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = blockIdx.x * 16 + warp * 2;
    if (r0 >= t.total_rows) return;
    float *y_out0 = nullptr, *y_out1 = nullptr;
    const uint8_t* wp0 = target_row_ptr(t, r0, stride, y_out0);
    const bool has_r1 = (r0 + 1 < t.total_rows);
    const uint8_t* wp1 = has_r1 ? target_row_ptr(t, r0 + 1, stride, y_out1) : wp0;
    const int blk_off = lane >> 3, sub = lane & 7, nb = n >> 8, nb4 = nb & ~3;
    float s0 = 0.0f, s1 = 0.0f;
    int blk_base = 0;
    for (; blk_base < nb4; blk_base += 4) step_iq2s_4blk_2row(wp0, wp1, x, sgrid, blk_base + blk_off, sub, s0, s1);
    if (blk_base + blk_off < nb) step_iq2s_4blk_2row(wp0, wp1, x, sgrid, blk_base + blk_off, sub, s0, s1);
    s0 = sum(s0); s1 = sum(s1);
    if (lane == 0) { *y_out0 = s0; if (has_r1) *y_out1 = s1; }
}

__device__ __forceinline__ float dot_q8_iq3_s_sgrid(
    const uint8_t* __restrict__ p, const BlockQ8_1* __restrict__ xb, const uint32_t* __restrict__ sgrid3, int lane) {
    const int sub = lane >> 2, group = lane & 3, idx0 = group * 2, idx1 = idx0 + 1;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p))));
    const uint16_t qs2 = __ldg(reinterpret_cast<const uint16_t*>(p + 2) + lane);
    const uint8_t qh_b = __ldg(p + 66 + sub), sbyte = __ldg(p + 74 + lane);
    const uint32_t lo = sgrid3[(qs2 & 0xFF) | (((qh_b >> idx0) & 1) << 8)];
    const uint32_t hi = sgrid3[(qs2 >> 8) | (((qh_b >> idx1) & 1) << 8)];
    const BlockQ8_1& x0 = xb[sub];
    const int2 xv2 = __ldg(reinterpret_cast<const int2*>(&x0.qs[group * 2]));
    const int32_t dp = __dp4a(apply_signs4(hi, uint32_t(sbyte >> 4)), xv2.y, __dp4a(apply_signs4(lo, sbyte & 0xFu), xv2.x, 0));
    const int scale = (__ldg(p + 106 + (sub >> 1)) >> (4 * (sub & 1))) & 15;
    return d * float(1 + 2 * scale) * float(dp) * __ldg(&x0.d);
}

// IQ3_S uses 110 bytes/block (110 % 4 == 2): strictly 1B and 2B-aligned loads, 4 blocks x 2 rows/warp.
__device__ __forceinline__ float eval_iq3s_subblock(
    uint16_t dh, uint32_t qs_lo, uint32_t qs_hi, uint32_t sb4, uint32_t qh_b, uint32_t sc_b, int sub,
    const uint32_t* __restrict__ sgrid3, float xd, int2 xq01, int2 xq23, int2 xq45, int2 xq67) {
    int32_t acc = __dp4a(apply_signs4(sgrid3[((qs_lo >> 8) & 0xFFu) | (((qh_b >> 1) & 1u) << 8)], (sb4 >> 4) & 0xFu), xq01.y,
                         __dp4a(apply_signs4(sgrid3[(qs_lo & 0xFFu) | (((qh_b >> 0) & 1u) << 8)], sb4 & 0xFu), xq01.x, 0));
    acc = __dp4a(apply_signs4(sgrid3[(qs_lo >> 24) | (((qh_b >> 3) & 1u) << 8)], (sb4 >> 12) & 0xFu), xq23.y,
                 __dp4a(apply_signs4(sgrid3[((qs_lo >> 16) & 0xFFu) | (((qh_b >> 2) & 1u) << 8)], (sb4 >> 8) & 0xFu), xq23.x, acc));
    acc = __dp4a(apply_signs4(sgrid3[((qs_hi >> 8) & 0xFFu) | (((qh_b >> 5) & 1u) << 8)], (sb4 >> 20) & 0xFu), xq45.y,
                 __dp4a(apply_signs4(sgrid3[(qs_hi & 0xFFu) | (((qh_b >> 4) & 1u) << 8)], (sb4 >> 16) & 0xFu), xq45.x, acc));
    acc = __dp4a(apply_signs4(sgrid3[(qs_hi >> 24) | (((qh_b >> 7) & 1u) << 8)], sb4 >> 28), xq67.y,
                 __dp4a(apply_signs4(sgrid3[((qs_hi >> 16) & 0xFFu) | (((qh_b >> 6) & 1u) << 8)], (sb4 >> 24) & 0xFu), xq67.x, acc));
    const int sc = int((sc_b >> (4 * (sub & 1))) & 0xFu);
    return __half2float(__ushort_as_half(dh)) * float(1 + 2 * sc) * float(acc) * xd;
}

__device__ __forceinline__ void step_iq3s_4blk_2row(
    const uint8_t* wp0, const uint8_t* wp1, const BlockQ8_1* __restrict__ x,
    const uint32_t* __restrict__ sgrid3, int blk_i, int sub, float& s0, float& s1) {
    const uint8_t* p0 = wp0 + size_t(blk_i) * 110;
    const uint8_t* p1 = wp1 + size_t(blk_i) * 110;
    const uint16_t dh0 = __ldg(reinterpret_cast<const uint16_t*>(p0));
    const uint32_t qslo0 = ldg_u32_from_u16(p0 + 2 + sub * 8), qshi0 = ldg_u32_from_u16(p0 + 6 + sub * 8);
    const uint32_t sb0 = ldg_u32_from_u16(p0 + 74 + sub * 4);
    const uint32_t qh0 = __ldg(p0 + 66 + sub), sc0 = __ldg(p0 + 106 + (sub >> 1));

    const uint16_t dh1 = __ldg(reinterpret_cast<const uint16_t*>(p1));
    const uint32_t qslo1 = ldg_u32_from_u16(p1 + 2 + sub * 8), qshi1 = ldg_u32_from_u16(p1 + 6 + sub * 8);
    const uint32_t sb1 = ldg_u32_from_u16(p1 + 74 + sub * 4);
    const uint32_t qh1 = __ldg(p1 + 66 + sub), sc1 = __ldg(p1 + 106 + (sub >> 1));

    const BlockQ8_1& x0 = x[blk_i * 8 + sub];
    const float xd = __ldg(&x0.d);
    const int2 xq01 = __ldg(reinterpret_cast<const int2*>(&x0.qs[0]));
    const int2 xq23 = __ldg(reinterpret_cast<const int2*>(&x0.qs[2]));
    const int2 xq45 = __ldg(reinterpret_cast<const int2*>(&x0.qs[4]));
    const int2 xq67 = __ldg(reinterpret_cast<const int2*>(&x0.qs[6]));

    s0 += eval_iq3s_subblock(dh0, qslo0, qshi0, sb0, qh0, sc0, sub, sgrid3, xd, xq01, xq23, xq45, xq67);
    s1 += eval_iq3s_subblock(dh1, qslo1, qshi1, sb1, qh1, sc1, sub, sgrid3, xd, xq01, xq23, xq45, xq67);
}

__global__ void __launch_bounds__(256, 4) mv_q8_iq3s_4blk_2row(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    __shared__ uint32_t sgrid3[512];
    sgrid3[threadIdx.x] = __ldg(&q38_iq3s_grid[threadIdx.x]);
    sgrid3[threadIdx.x + 256] = __ldg(&q38_iq3s_grid[threadIdx.x + 256]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = blockIdx.x * 16 + warp * 2;
    if (r0 >= t.total_rows) return;
    float *y_out0 = nullptr, *y_out1 = nullptr;
    const uint8_t* wp0 = target_row_ptr(t, r0, stride, y_out0);
    const bool has_r1 = (r0 + 1 < t.total_rows);
    const uint8_t* wp1 = has_r1 ? target_row_ptr(t, r0 + 1, stride, y_out1) : wp0;
    const int blk_off = lane >> 3, sub = lane & 7, nb = n >> 8, nb4 = nb & ~3;
    float s0 = 0.0f, s1 = 0.0f;
    int blk_base = 0;
    for (; blk_base < nb4; blk_base += 4) step_iq3s_4blk_2row(wp0, wp1, x, sgrid3, blk_base + blk_off, sub, s0, s1);
    if (blk_base + blk_off < nb) step_iq3s_4blk_2row(wp0, wp1, x, sgrid3, blk_base + blk_off, sub, s0, s1);
    s0 = sum(s0); s1 = sum(s1);
    if (lane == 0) { *y_out0 = s0; if (has_r1) *y_out1 = s1; }
}

// Q8_0 uses 34 bytes/block (34 % 4 == 2): assemble 4 int8_t weights from two uint16_t loads.
__device__ __forceinline__ float dot_q8_0_sublane(const uint8_t* wp, const BlockQ8_1* __restrict__ x, int b, int g) {
    const uint8_t* blk = wp + size_t(b) * 34;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(blk))));
    const int32_t w4 = int32_t(ldg_u32_from_u16(blk + 2 + g * 4));
    const BlockQ8_1& x0 = x[b];
    const int32_t dp = __dp4a(w4, __ldg(&x0.qs[g]), 0);
    return d * __ldg(&x0.d) * float(dp);
}

__global__ void __launch_bounds__(128, 8) mv_q8_q8_0_1row(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    __shared__ float wsum[4];
    const int row = blockIdx.x;
    if (row >= t.total_rows) return;
    float* y_out = nullptr;
    const uint8_t* wp = target_row_ptr(t, row, stride, y_out);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int sub_blk = lane >> 3, g = lane & 7, nb = n >> 5;
    float s = 0.0f;
    for (int blk_base = warp * 4; blk_base < nb; blk_base += 16) {
        if (blk_base + sub_blk < nb) s += dot_q8_0_sublane(wp, x, blk_base + sub_blk, g);
    }
    s = sum(s);
    if (lane == 0) wsum[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? wsum[lane] : 0.0f;
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
        if (lane == 0) *y_out = v;
    }
}

__global__ void __launch_bounds__(256, 6) mv_q8_q8_0_8rows(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= t.total_rows) return;
    float* y_out = nullptr;
    const uint8_t* wp = target_row_ptr(t, row, stride, y_out);
    const int sub_blk = lane >> 3, g = lane & 7, nb = n >> 5, nb4 = nb & ~3;
    float s = 0.0f;
    int blk_base = 0;
    for (; blk_base < nb4; blk_base += 4) s += dot_q8_0_sublane(wp, x, blk_base + sub_blk, g);
    if (blk_base + sub_blk < nb) s += dot_q8_0_sublane(wp, x, blk_base + sub_blk, g);
    s = sum(s);
    if (lane == 0) *y_out = s;
}

template <MvKind KIND, int BLOCK_BYTES>
__global__ void __launch_bounds__(128) mv_q8_4rows(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    __shared__ alignas(4) uint16_t sw[4][((BLOCK_BYTES + 3) / 4) * 2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= t.total_rows) return;
    float* y_out = nullptr;
    const uint8_t* wp = target_row_ptr(t, row, stride, y_out);
    const int nb = n / 256;
    float s = 0;
    for (int blk_i = 0; blk_i < nb; ++blk_i) {
        load_block_u16<(BLOCK_BYTES + 1) / 2>(sw[warp], wp + size_t(blk_i) * BLOCK_BYTES, lane);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(sw[warp]);
        const BlockQ8_1* xb = x + blk_i * 8;
        if constexpr (KIND == MvKind::Q4_K) s += dot_q8_qk45<false>(p, xb, lane);
        else s += dot_q8_qk45<true>(p, xb, lane);
    }
    s = sum(s);
    if (!lane) *y_out = s;
}

template <MvKind KIND, int BLOCK_BYTES>
__global__ void __launch_bounds__(256) mv_q8_1row(MvTargets t, const BlockQ8_1* __restrict__ x, int n, size_t stride) {
    struct alignas(8) Shared1Row {
        float wsum[8];
        uint64_t sgrid[KIND == MvKind::IQ2_S ? 1024 : (KIND == MvKind::IQ3_S ? 256 : 1)];
        uint16_t sw[8][(KIND == MvKind::Q4_K || KIND == MvKind::Q5_K) ? ((BLOCK_BYTES + 3) / 4) * 2 : 2];
    };
    __shared__ Shared1Row sm;
    if constexpr (KIND == MvKind::IQ2_S) {
        #pragma unroll
        for (int i = threadIdx.x; i < 1024; i += 256) sm.sgrid[i] = __ldg(&q38_iq2s_grid[i]);
        __syncthreads();
    } else if constexpr (KIND == MvKind::IQ3_S) {
        uint32_t* sg3 = reinterpret_cast<uint32_t*>(sm.sgrid);
        sg3[threadIdx.x] = __ldg(&q38_iq3s_grid[threadIdx.x]);
        sg3[threadIdx.x + 256] = __ldg(&q38_iq3s_grid[threadIdx.x + 256]);
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x;
    if (row >= t.total_rows) return;
    float* y_out = nullptr;
    const uint8_t* wp = target_row_ptr(t, row, stride, y_out);
    const int nb = n / 256;
    float s = 0;
    for (int blk_i = warp; blk_i < nb; blk_i += 8) {
        const uint8_t* gptr = wp + size_t(blk_i) * BLOCK_BYTES;
        const BlockQ8_1* xb = x + blk_i * 8;
        if constexpr (KIND == MvKind::Q4_K) {
            load_block_u16<(BLOCK_BYTES + 1) / 2>(sm.sw[warp], gptr, lane);
            s += dot_q8_qk45<false>(reinterpret_cast<const uint8_t*>(sm.sw[warp]), xb, lane);
        } else if constexpr (KIND == MvKind::Q5_K) {
            load_block_u16<(BLOCK_BYTES + 1) / 2>(sm.sw[warp], gptr, lane);
            s += dot_q8_qk45<true>(reinterpret_cast<const uint8_t*>(sm.sw[warp]), xb, lane);
        } else if constexpr (KIND == MvKind::Q6_K) {
            s += dot_q8_q6_k_direct(gptr, xb, lane);
        } else if constexpr (KIND == MvKind::IQ2_S) {
            s += dot_q8_iq2_s_sgrid(gptr, xb, sm.sgrid, lane);
        } else {
            s += dot_q8_iq3_s_sgrid(gptr, xb, reinterpret_cast<const uint32_t*>(sm.sgrid), lane);
        }
    }
    s = sum(s);
    if (!lane) sm.wsum[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? sm.wsum[lane] : 0.f;
        for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (!lane) *y_out = v;
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
__global__ void had_batch(const float* __restrict__ x,const float* __restrict__ signs,float* __restrict__ y,int n,bool inverse,bool grouped){
    __shared__ float b[1024];
    const int j=threadIdx.x,index=blockIdx.x*1024+j,tok=blockIdx.y;
    int src=index;
    if(grouped){int head=index/128;src=((head%3)*16+head/3)*128+index%128;}
    const size_t base=size_t(tok)*n;
    b[j]=x[base+src]*(inverse?1.f:signs[index]);__syncthreads();
    for(int stride=1;stride<1024;stride*=2){float v=b[j],other=b[j^stride];__syncthreads();b[j]=(j&stride)?other-v:v+other;__syncthreads();}
    y[base+index]=b[j]*(1.f/32)*(inverse?signs[index]:1.f);
}
// Based on Strata q36 k_attn_prep: Q and output gates interleave per head.
__global__ void prep(const float* qg,const float* k,const float* v,const float* qw,const float* kw,float* q,float* gate,
                     half* kc,half* vc,int pos,int ctx,int q_heads,int kv_heads,float eps,float base,float* normalized_k=nullptr){
    __shared__ float buf[256],part[8],inv_freq[32];int head=blockIdx.x,t=blockIdx.y,i=threadIdx.x;bool isq=head<q_heads;
    // inv_freq depends only on the 32 RoPE frequency slots, not on head, token
    // or position. Compute it once per block on the device with the same powf
    // so the angles stay bit-identical to the original expression.
    if(i<32)inv_freq[i]=powf(base,-float(i)/32);
    const float* x=isq?(qg+size_t(t)*(2*q_heads*256)+head*512):(k+size_t(t)*(kv_heads*256)+(head-q_heads)*256);float z=x[i];float s=sum(z*z);
    if(!(i%32))part[i/32]=s;__syncthreads();s=0;for(int j=0;j<8;++j)s+=part[j];
    buf[i]=z*rsqrtf(s/256+eps)*(isq?qw[i]:kw[i]);__syncthreads();z=buf[i];
    if(i<64){int p=i%32;float angle=(pos+t)*inv_freq[p],sn,cs;sincosf(angle,&sn,&cs);z=i<32?buf[i]*cs-buf[i+32]*sn:buf[i-32]*sn+buf[i]*cs;}
    if(isq){const size_t qo=size_t(t)*(q_heads*256)+head*256+i;const size_t go=size_t(t)*(2*q_heads*256)+head*512+256+i;q[qo]=z;gate[qo]=qg[go];}
    else{int kh=head-q_heads;if(normalized_k)normalized_k[size_t(t)*(kv_heads*256)+kh*256+i]=z;
        else{kc[(size_t(kh)*ctx+pos+t)*256+i]=__float2half(z);vc[(size_t(kh)*ctx+pos+t)*256+i]=__float2half(v[size_t(t)*(kv_heads*256)+kh*256+i]);}}
}
__global__ void score(const float* q,const half* k,float* s,int pos,int ctx,int heads_per_kv,const int* status){
    if(status&&*status)return;
    int t=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y,lane=threadIdx.x%32;if(t>pos)return;int kh=head/heads_per_kv;
    float a=0;for(int i=lane;i<256;i+=32)a=fmaf(q[head*256+i],__half2float(k[(size_t(kh)*ctx+t)*256+i]),a);
    a=sum(a);if(!lane)s[size_t(head)*ctx+t]=a/16;
}
__global__ void attend(const float* scores,const half* v,const float* gate,float* out,int pos,int ctx,int heads_per_kv,const int* status){
    if(status&&*status)return;
    // One block per query head; stable softmax then each lane's value component.
    __shared__ float tmp[256],maxi,denom;int head=blockIdx.x,i=threadIdx.x;const float* s=scores+size_t(head)*ctx;
    float mx=-INFINITY;for(int t=i;t<=pos;t+=256)mx=fmaxf(mx,s[t]);tmp[i]=mx;__syncthreads();
    for(int d=128;d;d/=2){if(i<d)tmp[i]=fmaxf(tmp[i],tmp[i+d]);__syncthreads();}if(!i)maxi=tmp[0];__syncthreads();
    float z=0;for(int t=i;t<=pos;t+=256)z+=expf(s[t]-maxi);tmp[i]=z;__syncthreads();
    for(int d=128;d;d/=2){if(i<d)tmp[i]+=tmp[i+d];__syncthreads();}if(!i)denom=tmp[0];__syncthreads();
    float y=0;for(int t=0;t<=pos;++t)y=fmaf(expf(s[t]-maxi)/denom,__half2float(v[(size_t(head/heads_per_kv)*ctx+t)*256+i]),y);
    out[head*256+i]=y/(1+expf(-gate[head*256+i]));
}
}
void preload_ops(){
    cudaFuncAttributes a{};
    ck(cudaFuncGetAttributes(&a,deq));ck(cudaFuncGetAttributes(&a,mv));ck(cudaFuncGetAttributes(&a,norm));
    ck(cudaFuncGetAttributes(&a,elt));ck(cudaFuncGetAttributes(&a,gb));ck(cudaFuncGetAttributes(&a,fin));
    ck(cudaFuncGetAttributes(&a,had));ck(cudaFuncGetAttributes(&a,prep));ck(cudaFuncGetAttributes(&a,score));ck(cudaFuncGetAttributes(&a,attend));
    ck(cudaFuncGetAttributes(&a,quantize_q8_1_kernel));
    ck(cudaFuncGetAttributes(&a,rms_quantize_q8_1_kernel));
    ck(cudaFuncGetAttributes(&a,swiglu_quantize_q8_1_kernel));
    ck(cudaFuncGetAttributes(&a,mv_q8_qk45_4blk_2row<false>));
    ck(cudaFuncGetAttributes(&a,mv_q8_qk45_4blk_2row<true>));
    ck(cudaFuncGetAttributes(&a,mv_q8_q6k_16rows));
    ck(cudaFuncGetAttributes(&a,mv_q8_iq2s_4blk_2row));
    ck(cudaFuncGetAttributes(&a,mv_q8_iq3s_4blk_2row));
    ck(cudaFuncGetAttributes(&a,mv_q8_q8_0_1row));ck(cudaFuncGetAttributes(&a,mv_q8_q8_0_8rows));
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

void quantize_q8_1(const float* input, BlockQ8_1* output, int n, cudaStream_t stream) {
    if (n <= 0 || n % 32 || !input || !output || (uintptr_t(output) & 7)) throw std::runtime_error("invalid quantize_q8_1 args");
    const int nb = n / 32;
    quantize_q8_1_kernel<<<(nb + 7) / 8, 256, 0, stream>>>(input, output, n);
    ck(cudaGetLastError());
}

void rms_quantize_q8_1(const float* x, const float* weight, float* out_f32, BlockQ8_1* out_q8, int n, float eps, cudaStream_t s) {
    if (n <= 0 || n % 32 || n > 17408 || !x || !weight || !out_f32 || !out_q8 || (uintptr_t(out_q8) & 7))
        throw std::runtime_error("invalid rms_quantize_q8_1 args");
    rms_quantize_q8_1_kernel<<<1, 256, 0, s>>>(x, weight, out_f32, out_q8, n, eps);
    ck(cudaGetLastError());
}

void swiglu_quantize_q8_1(float* gate, const float* up, BlockQ8_1* out_q8, int n, cudaStream_t s) {
    if (n <= 0 || n % 32 || !gate || !up || !out_q8 || (uintptr_t(out_q8) & 7))
        throw std::runtime_error("invalid swiglu_quantize_q8_1 args");
    const int nb = n / 32;
    swiglu_quantize_q8_1_kernel<<<(nb + 7) / 8, 256, 0, s>>>(gate, up, out_q8, n);
    ck(cudaGetLastError());
}

namespace {
void dispatch_matvec_q8_1(int type, const MvTargets& t, uintptr_t u_align, const BlockQ8_1* x_q8, const float* x_f32, int n, cudaStream_t s) {
    if (uintptr_t(x_q8) & 7) throw std::runtime_error("misaligned BlockQ8_1 input");
    const bool q8_supported = (type == 8 || type == 12 || type == 13 || type == 14 || type == 21 || type == 22);
    if (!q8_supported || (u_align & 1)) {
        matvec(type, t.w0, x_f32, t.y0, n, t.rows0, s);
        if (t.w1) matvec(type, t.w1, x_f32, t.y1, n, t.rows12 - t.rows0, s);
        if (t.w2) matvec(type, t.w2, x_f32, t.y2, n, t.total_rows - t.rows12, s);
        return;
    }
    const size_t rb = row_bytes(type, n);
    const int rows = t.total_rows;
    switch (type) {
        case 8:
            if (rows <= 256) mv_q8_q8_0_1row<<<rows, 128, 0, s>>>(t, x_q8, n, rb);
            else mv_q8_q8_0_8rows<<<(rows + 7) / 8, 256, 0, s>>>(t, x_q8, n, rb);
            break;
        case 12:
            if ((u_align & 15) == 0 && rows >= 65536) mv_q8_qk45_4blk_2row_128<false><<<(rows + 7) / 8, 128, 0, s>>>(t, x_q8, n, rb);
            else if ((u_align & 15) == 0 && rows >= 256) mv_q8_qk45_4blk_2row<false><<<(rows + 15) / 16, 256, 0, s>>>(t, x_q8, n, rb);
            else if (rows < 256) mv_q8_1row<MvKind::Q4_K, 144><<<rows, 256, 0, s>>>(t, x_q8, n, rb);
            else mv_q8_4rows<MvKind::Q4_K, 144><<<(rows + 3) / 4, 128, 0, s>>>(t, x_q8, n, rb);
            break;
        case 13:
            if ((u_align & 15) == 0 && rows >= 65536) mv_q8_qk45_4blk_2row_128<true><<<(rows + 7) / 8, 128, 0, s>>>(t, x_q8, n, rb);
            else if ((u_align & 15) == 0 && rows >= 256) mv_q8_qk45_4blk_2row<true><<<(rows + 15) / 16, 256, 0, s>>>(t, x_q8, n, rb);
            else if (rows < 256) mv_q8_1row<MvKind::Q5_K, 176><<<rows, 256, 0, s>>>(t, x_q8, n, rb);
            else mv_q8_4rows<MvKind::Q5_K, 176><<<(rows + 3) / 4, 128, 0, s>>>(t, x_q8, n, rb);
            break;
        case 14:
            if (rows >= 256) mv_q8_q6k_16rows<<<(rows + 15) / 16, 256, 0, s>>>(t, x_q8, n, rb);
            else mv_q8_1row<MvKind::Q6_K, 210><<<rows, 256, 0, s>>>(t, x_q8, n, rb);
            break;
        case 21:
            if (rows >= 256) mv_q8_iq3s_4blk_2row<<<(rows + 15) / 16, 256, 0, s>>>(t, x_q8, n, rb);
            else mv_q8_1row<MvKind::IQ3_S, 110><<<rows, 256, 0, s>>>(t, x_q8, n, rb);
            break;
        case 22:
            if (rows >= 256) mv_q8_iq2s_4blk_2row<<<(rows + 15) / 16, 256, 0, s>>>(t, x_q8, n, rb);
            else mv_q8_1row<MvKind::IQ2_S, 82><<<rows, 256, 0, s>>>(t, x_q8, n, rb);
            break;
    }
    ck(cudaGetLastError());
}
}

template <int NCOLS>
__global__ void __launch_bounds__(256, 4) mv_q8_q8_0_batch(
    const uint8_t* __restrict__ wp, const BlockQ8_1* __restrict__ x, float* __restrict__ y,
    int n, int rows, size_t stride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const uint8_t* row_ptr = wp + size_t(row) * stride;
    const int sub_blk = lane >> 3, g = lane & 7, nb = n >> 5, x_stride = n >> 5;
    float acc[NCOLS];
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) acc[c] = 0.0f;
    for (int blk_i = sub_blk; blk_i < nb; blk_i += 4) {
        const uint8_t* blk = row_ptr + size_t(blk_i) * 34;
        const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(blk))));
        const int32_t w4 = int32_t(ldg_u32_from_u16(blk + 2 + g * 4));
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const BlockQ8_1& x0 = x[size_t(c) * x_stride + blk_i];
            acc[c] += d * __ldg(&x0.d) * float(__dp4a(w4, __ldg(&x0.qs[g]), 0));
        }
    }
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        const float s = sum(acc[c]);
        if (lane == 0) y[size_t(c) * rows + row] = s;
    }
}

template <int NCOLS>
static void launch_mv_q8_batch_cols(int type, const uint8_t* wp, const BlockQ8_1* x_q8, float* y, int n, int rows, size_t rb, cudaStream_t s) {
    const dim3 grid((rows + 7) / 8);
    switch (type) {
        case 8:  mv_q8_q8_0_batch<NCOLS><<<grid, 256, 0, s>>>(wp, x_q8, y, n, rows, rb); break;
        case 12: mv_q8_qk45_batch<false, NCOLS><<<grid, 256, 0, s>>>(wp, x_q8, y, n, rows, rb); break;
        case 13: mv_q8_qk45_batch<true, NCOLS><<<grid, 256, 0, s>>>(wp, x_q8, y, n, rows, rb); break;
        case 14: mv_q8_q6k_batch<NCOLS><<<grid, 256, 0, s>>>(wp, x_q8, y, n, rows, rb); break;
        default: throw std::runtime_error("unsupported batch MMVQ type");
    }
}

bool matvec_q8_1_batch_supported(int type) {
    return type == 8 || type == 12 || type == 13 || type == 14;
}

void quantize_q8_1_batch(const float* input, BlockQ8_1* output, int n_tokens, int n, cudaStream_t stream) {
    if (n_tokens <= 0) return;
    if (!input || !output || n <= 0 || (n % 32) != 0) throw std::runtime_error("invalid quantize_q8_1_batch");
    const int nb = n >> 5;
    quantize_q8_1_batch_kernel<<<dim3(unsigned((nb + 7) / 8), unsigned(n_tokens)), 256, 0, stream>>>(input, output, n_tokens, n);
    ck(cudaGetLastError());
}

void matvec_q8_1_batch(int type, const void* w, const BlockQ8_1* x_q8, float* y, int n, int rows, int n_tokens, cudaStream_t s) {
    if (n_tokens <= 0) return;
    shape(type, n, rows);
    if (!w || !x_q8 || !y) throw std::runtime_error("null matrix input");
    const auto* wp = static_cast<const uint8_t*>(w);
    const size_t rb = row_bytes(type, n);
    const int x_stride = n >> 5;
    int t = 0;
    while (t < n_tokens) {
        const int rem = n_tokens - t;
        if (rem >= 8) {
            launch_mv_q8_batch_cols<8>(type, wp, x_q8 + size_t(t) * x_stride, y + size_t(t) * rows, n, rows, rb, s);
            t += 8;
        } else if (rem >= 4) {
            launch_mv_q8_batch_cols<4>(type, wp, x_q8 + size_t(t) * x_stride, y + size_t(t) * rows, n, rows, rb, s);
            t += 4;
        } else if (rem >= 2) {
            launch_mv_q8_batch_cols<2>(type, wp, x_q8 + size_t(t) * x_stride, y + size_t(t) * rows, n, rows, rb, s);
            t += 2;
        } else {
            launch_mv_q8_batch_cols<1>(type, wp, x_q8 + size_t(t) * x_stride, y + size_t(t) * rows, n, rows, rb, s);
            t += 1;
        }
    }
    ck(cudaGetLastError());
}

void matvec_q8_1(int type, const void* w, const BlockQ8_1* x_q8, const float* x_f32, float* y, int n, int rows, cudaStream_t s) {
    shape(type, n, rows);
    if (!w || !x_q8 || !x_f32 || !y) throw std::runtime_error("null matrix input");
    const auto* wp = static_cast<const uint8_t*>(w);
    dispatch_matvec_q8_1(type, {wp, y, rows, nullptr, nullptr, rows, nullptr, nullptr, rows}, uintptr_t(wp), x_q8, x_f32, n, s);
}

void matvec_q8_1_dual(int type, const void* w0, float* y0, int rows0,
                      const void* w1, float* y1, int rows1,
                      const BlockQ8_1* x_q8, const float* x_f32, int n, cudaStream_t s) {
    shape(type, n, rows0);
    shape(type, n, rows1);
    if (!w0 || !y0 || !w1 || !y1 || !x_q8 || !x_f32) throw std::runtime_error("null matrix input");
    if (int64_t(rows0) + int64_t(rows1) > std::numeric_limits<int>::max()) throw std::runtime_error("row count overflow");
    const auto* wp0 = static_cast<const uint8_t*>(w0);
    const auto* wp1 = static_cast<const uint8_t*>(w1);
    const int r12 = rows0 + rows1;
    dispatch_matvec_q8_1(type, {wp0, y0, rows0, wp1, y1, r12, nullptr, nullptr, r12}, uintptr_t(wp0) | uintptr_t(wp1), x_q8, x_f32, n, s);
}

void matvec_q8_1_triple(int type, const void* w0, float* y0, int rows0,
                        const void* w1, float* y1, int rows1,
                        const void* w2, float* y2, int rows2,
                        const BlockQ8_1* x_q8, const float* x_f32, int n, cudaStream_t s) {
    shape(type, n, rows0);
    shape(type, n, rows1);
    shape(type, n, rows2);
    if (!w0 || !y0 || !w1 || !y1 || !w2 || !y2 || !x_q8 || !x_f32) throw std::runtime_error("null matrix input");
    if (int64_t(rows0) + int64_t(rows1) + int64_t(rows2) > std::numeric_limits<int>::max()) throw std::runtime_error("row count overflow");
    const auto* wp0 = static_cast<const uint8_t*>(w0);
    const auto* wp1 = static_cast<const uint8_t*>(w1);
    const auto* wp2 = static_cast<const uint8_t*>(w2);
    const int r12 = rows0 + rows1, r123 = r12 + rows2;
    dispatch_matvec_q8_1(type, {wp0, y0, rows0, wp1, y1, r12, wp2, y2, r123},
                         uintptr_t(wp0) | uintptr_t(wp1) | uintptr_t(wp2), x_q8, x_f32, n, s);
}

void matvec(int type,const void* w,const float* x,float* y,int n,int rows,cudaStream_t s){
    shape(type,n,rows);if(!w||!x||!y)throw std::runtime_error("null matrix input");
    const dim3 grid((rows+7)/8);const size_t rb=row_bytes(type,n);const auto* wp=(const uint8_t*)w;
    // The vectorised packed GEMV kernels require 16 B-aligned rows.  Fall back
    // to the byte-load scalar kernel for deliberately misaligned inputs.
    if ((type==12||type==13||type==14||type==21||type==22) &&
        (((uintptr_t(wp) & 15) != 0) || ((rb & 15) != 0))) {
        mv<<<grid,256,0,s>>>(type,wp,x,y,n,rows,rb);
        ck(cudaGetLastError());
        return;
    }
    switch(type){
        case 8:  mv_q8_0<<<grid,256,0,s>>>(wp,x,y,n,rows,rb); break;
        case 12: launch_mv_256<MvKind::Q4_K, 144>(wp,x,y,n,rows,rb,s); break;
        case 13: launch_mv_256<MvKind::Q5_K, 176>(wp,x,y,n,rows,rb,s); break;
        case 14: launch_mv_256<MvKind::Q6_K, 210>(wp,x,y,n,rows,rb,s); break;
        case 21: launch_mv_256<MvKind::IQ3_S, 110>(wp,x,y,n,rows,rb,s); break;
        case 22: launch_mv_256<MvKind::IQ2_S, 82>(wp,x,y,n,rows,rb,s); break;
        case 142: mv_pq2_0<<<(rows+3)/4,128,0,s>>>(wp,x,y,n,rows,rb); break;
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
void hadamard_batch(const float* x,const float* signs,float* y,int n_tokens,int n,bool inverse,bool grouped,cudaStream_t s){
    if(n_tokens<=0)return;
    if(n<=0||n%1024||!x||!signs||!y||x==y||(grouped&&(n!=6144||inverse)))throw std::runtime_error("invalid Hadamard arguments");
    had_batch<<<dim3(unsigned(n/1024),unsigned(n_tokens)),1024,0,s>>>(x,signs,y,n,inverse,grouped);ck(cudaGetLastError());}
void attention_prepare(const float* qg,const float* k,const float* v,const float* qw,const float* kw,float* q,float* gate,half* kc,half* vc,int pos,int ctx,int q_heads,int kv_heads,float eps,float base,cudaStream_t s){
    if(pos<0||pos>=ctx||q_heads<=0||kv_heads<=0||q_heads%kv_heads)throw std::runtime_error("attention position or head geometry out of range");prep<<<q_heads+kv_heads,256,0,s>>>(qg,k,v,qw,kw,q,gate,kc,vc,pos,ctx,q_heads,kv_heads,eps,base);ck(cudaGetLastError());}
void attention_apply(const float* q,const float* gate,const half* kc,const half* vc,float* out,float* scores,int pos,int ctx,int q_heads,int kv_heads,cudaStream_t s,const int* status){
    if(pos<0||pos>=ctx||q_heads<=0||kv_heads<=0||q_heads%kv_heads)throw std::runtime_error("attention position or head geometry out of range");const int hp=q_heads/kv_heads;score<<<dim3((pos+8)/8,q_heads),256,0,s>>>(q,kc,scores,pos,ctx,hp,status);attend<<<q_heads,256,0,s>>>(scores,vc,gate,out,pos,ctx,hp,status);ck(cudaGetLastError());}
void attention_normalize(const float* qg,const float* k,const float* qw,const float* kw,float* q,float* gate,float* nk,int pos,int q_heads,int kv_heads,float eps,float base,cudaStream_t s){
    if(pos<0||!qg||!k||!qw||!kw||!q||!gate||!nk||q_heads<=0||kv_heads<=0||q_heads%kv_heads)throw std::runtime_error("invalid attention normalization");
    prep<<<q_heads+kv_heads,256,0,s>>>(qg,k,nullptr,qw,kw,q,gate,nullptr,nullptr,pos,0,q_heads,kv_heads,eps,base,nk);ck(cudaGetLastError());}
void attention_normalize_batch(const float* qg,const float* k,const float* qw,const float* kw,float* q,float* gate,float* nk,int pos0,int n_tokens,int q_heads,int kv_heads,float eps,float base,cudaStream_t s){
    if(n_tokens<=0)return;
    if(pos0<0||!qg||!k||!qw||!kw||!q||!gate||!nk||q_heads<=0||kv_heads<=0||q_heads%kv_heads)throw std::runtime_error("invalid attention normalization batch");
    prep<<<dim3(q_heads+kv_heads,unsigned(n_tokens)),256,0,s>>>(qg,k,nullptr,qw,kw,q,gate,nullptr,nullptr,pos0,0,q_heads,kv_heads,eps,base,nk);ck(cudaGetLastError());}
}
