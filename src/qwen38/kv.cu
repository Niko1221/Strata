// Native packed history. Turbo arithmetic reuses the pinned tqcodec primitives;
// its MIT attribution is retained in tqcodec/DIALECT_LICENSE.txt.
#include "kv.hpp"
#include "dialect_cuda_device.cuh"
#include <algorithm>
#include <cmath>

namespace strata::qwen38 {
namespace {
__device__ float reduce(float x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffffu,x,d);return x;}
__global__ void spin(const float* x,float* y,const float* r,int n,bool inverse){
    int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int b=i/8*8,row=i%8;float value=0;
    for(int j=0;j<8;++j)value=fmaf(r[inverse?j*8+row:row*8+j],x[b+j],value);y[i]=value;
}
__global__ void store_f16(const float* x,uint8_t* cache,int pos,int ctx,int* status){
    if(atomicAdd(status,0))return;
    int i=blockIdx.x*256+threadIdx.x;if(i<1024){half value=__float2half_rn(x[i]);if(!isfinite(__half2float(value))){atomicExch(status,1);return;}
        reinterpret_cast<half*>(cache)[(size_t(i/256)*ctx+pos)*256+i%256]=value;}
}
// GGML reference Q8_0: positive amax/127, roundf (ties away from zero).
// Q4_0: signed first absolute maximum / -8, low first16/high second16.
// Scale is rounded to FP16 for storage AFTER codes use the original FP32 scale.
__global__ void store_q(const float* x,uint8_t* cache,int pos,int ctx,bool q8,int* status){
    if(atomicAdd(status,0))return;
    int b=blockIdx.x*32+threadIdx.x;if(b>=32)return;const float* input=x+b*32;
    uint8_t* p=cache+(size_t(b/8)*ctx+pos)*(q8?272:144)+(b%8)*(q8?34:18);
    float amax=0,signed_max=0;for(int i=0;i<32;++i){float a=fabsf(input[i]);if(a>amax){amax=a;signed_max=input[i];}}
    float d=q8?amax/127.f:signed_max/-8.f,inv=d?1.f/d:0.f;
    if(!isfinite(inv)){atomicExch(status,1);return;}
    auto hd=__float2half_rn(d);if(!isfinite(__half2float(hd))){atomicExch(status,1);return;}
    tqc_cuda_detail::store16(p,__half_as_ushort(hd));
    if(q8)for(int i=0;i<32;++i)p[2+i]=uint8_t(int8_t(roundf(input[i]*inv)));
    else for(int i=0;i<16;++i){int lo=min(15,int(input[i]*inv+8.5f)),hi=min(15,int(input[i+16]*inv+8.5f));p[2+i]=uint8_t(lo|(hi<<4));}
}
__global__ void store_turbo(const float* v,uint8_t* cache,int pos,int ctx,int id,int* status){
    if(atomicAdd(status,0))return;
    int b=threadIdx.x;if(b>=8)return;float prepared[128];uint8_t packed[66];
    float norm=tqc_cuda_detail::prepare(id,v+b*128,prepared);tqc_cuda_detail::turbo_pack(id,prepared,norm,packed);
    if(!isfinite(tqc_cuda_detail::half_decode(tqc_cuda_detail::load16(packed)))){atomicExch(status,1);return;}
    int bytes=18+16*id;uint8_t* p=cache+(size_t(b/2)*ctx+pos)*2*bytes+(b%2)*bytes;
    for(int i=0;i<bytes;++i)p[i]=packed[i];
}
__device__ float load_key(const uint8_t* p,int i,bool q8){
    if(!q8)return __half2float(reinterpret_cast<const half*>(p)[i]);
    p+=i/32*34;return tqc_cuda_detail::half_decode(tqc_cuda_detail::load16(p))*int8_t(p[2+i%32]);
}
__device__ float load_value(const uint8_t* p,int i,int mode){
    if(mode==0)return __half2float(reinterpret_cast<const half*>(p)[i]);
    if(mode>=3){int id=mode-2;return tqc_cuda_detail::turbo_value(id,p+i/128*(18+16*id),i%128);}
    p+=i/32*18;i%=32;return tqc_cuda_detail::half_decode(tqc_cuda_detail::load16(p))*(int((p[2+i%16]>>(i>=16?4:0))&15)-8);
}
__global__ void scores_kernel(const float* q,const uint8_t* key,float* scores,int pos,int ctx,int stride,bool q8,const int* status){
    __shared__ int early_stop;
    if(threadIdx.x==0)early_stop=*status;
    __syncthreads();
    if(early_stop)return;
    int t=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y,lane=threadIdx.x%32;if(t>pos)return;
    const uint8_t* k=key+(size_t(head/6)*ctx+t)*stride;float z=0;
    for(int i=lane;i<256;i+=32)z=fmaf(q[head*256+i],load_key(k,i,q8),z);
    z=reduce(z);if(!lane)scores[size_t(head)*ctx+t]=z/16.f;
}
__global__ void aggregate_kernel(const float* scores,const uint8_t* value,float* out,int pos,int ctx,int stride,int mode,const int* status){
    __shared__ int early_stop;
    if(threadIdx.x==0)early_stop=*status;
    __syncthreads();
    if(early_stop)return;
    __shared__ float temp[256],maximum,denominator;int h=blockIdx.x,i=threadIdx.x;const float* s=scores+size_t(h)*ctx;
    float mx=-INFINITY;for(int t=i;t<=pos;t+=256)mx=fmaxf(mx,s[t]);temp[i]=mx;__syncthreads();
    for(int d=128;d;d/=2){if(i<d)temp[i]=fmaxf(temp[i],temp[i+d]);__syncthreads();}
    if(!i)maximum=temp[0];__syncthreads();float den=0;for(int t=i;t<=pos;t+=256)den+=expf(s[t]-maximum);temp[i]=den;__syncthreads();
    for(int d=128;d;d/=2){if(i<d)temp[i]+=temp[i+d];__syncthreads();}
    if(!i)denominator=temp[0];__syncthreads();float y=0;
    for(int t=0;t<=pos;++t)y=fmaf(expf(s[t]-maximum)/denominator,load_value(value+(size_t(h/6)*ctx+t)*stride,i,mode),y);
    out[h*256+i]=y;
}
__global__ void inverse_turbo(const float* in,float* out){
    int block=blockIdx.x*32+threadIdx.x;if(block>=48)return;float x[128];
    for(int i=0;i<128;++i)x[i]=in[block*128+i];tqc_cuda_detail::fwht(x,true);
    for(int i=0;i<128;++i)out[block*128+i]=x[i];
}
__global__ void apply_gate(float* out,const float* gate){int i=blockIdx.x*256+threadIdx.x;if(i<6144)out[i]/=1+expf(-gate[i]);}
__device__ __forceinline__ int smem_swiz(int elem){return elem^((elem&32)>>3);}
__global__ void spin8_store_q_batch_kernel(const float* in,uint8_t* cache,const float* r,int pos_start,int n_tokens,int ctx,bool q8,int* status){
    __shared__ int early_stop;
    __shared__ float sr[64];
    const int tok=blockIdx.x;
    const int b=threadIdx.x; // 32 blocks of 32 elems = 1024 elems (4 KV heads * 256)
    if(b==0)early_stop=atomicAdd(status,0);
    if(b<32){sr[b]=r[b];sr[b+32]=r[b+32];}
    __syncthreads();
    if(early_stop||tok>=n_tokens||b>=32)return;
    const float* src=in+size_t(tok)*1024+b*32;
    float rot[32];
    for(int g=0;g<4;++g){
        float sv[8];
        #pragma unroll
        for(int j=0;j<8;++j){
            sv[j]=src[g*8+j];
            if(!isfinite(sv[j])){atomicExch(status,1);return;}
        }
        for(int row=0;row<8;++row){
            float v=0.f;
            #pragma unroll
            for(int j=0;j<8;++j)v=fmaf(sr[row*8+j],sv[j],v);
            rot[g*8+row]=v;
        }
    }
    const int pos=pos_start+tok;
    uint8_t* p=cache+(size_t(b/8)*ctx+pos)*(q8?272:144)+(b%8)*(q8?34:18);
    float amax=0,signed_max=0;
    for(int i=0;i<32;++i){if(!isfinite(rot[i])){atomicExch(status,1);return;}float a=fabsf(rot[i]);if(a>amax){amax=a;signed_max=rot[i];}}
    float d=q8?amax/127.f:signed_max/-8.f,inv=d?1.f/d:0.f;
    if(!isfinite(inv)){atomicExch(status,1);return;}
    auto hd=__float2half_rn(d);if(!isfinite(__half2float(hd))){atomicExch(status,1);return;}
    tqc_cuda_detail::store16(p,__half_as_ushort(hd));
    if(q8)for(int i=0;i<32;++i)p[2+i]=uint8_t(int8_t(roundf(rot[i]*inv)));
    else for(int i=0;i<16;++i){int lo=min(15,int(rot[i]*inv+8.5f)),hi=min(15,int(rot[i+16]*inv+8.5f));p[2+i]=uint8_t(lo|(hi<<4));}
}
__global__ void __launch_bounds__(256) spin8_attend_batch_kernel(
    const float* __restrict__ q_batch,const float* __restrict__ gate_batch,
    const uint8_t* __restrict__ key_cache,const uint8_t* __restrict__ val_cache,
    const float* __restrict__ r,float* __restrict__ out_batch,
    int pos_start,int n_tokens,int ctx,int key_stride,int val_stride,int mode,int* status){
    constexpr int BK=16,BT=8,HG=6;
    __shared__ int early_stop;
    __shared__ float sr[64];
    __shared__ alignas(16) float sk[BK][256];
    __shared__ alignas(16) float sv[BK][256];
    const int kh=blockIdx.x,tok_base=blockIdx.y*BT,i=threadIdx.x;
    const int t_cnt=(n_tokens-tok_base<BT)?(n_tokens-tok_base):BT;
    if(i==0)early_stop=atomicAdd(status,0);
    if(i<64)sr[i]=r[i];
    __syncthreads();
    if(early_stop||t_cnt<=0)return;
    const int lane=i&31,warp=i>>5,b=lane*8;
    float q_rot[HG][8],acc[HG][8],mx[HG],denom[HG];
    #pragma unroll
    for(int s=0;s<HG;++s){
        mx[s]=-INFINITY;denom[s]=0.f;
        #pragma unroll
        for(int c=0;c<8;++c){q_rot[s][c]=0.f;acc[s][c]=0.f;}
        const int q_idx=warp+s*8;
        const int t=q_idx/6,h=q_idx%6;
        if(t<t_cnt){
            const size_t qoff=size_t(tok_base+t)*6144+size_t(kh*6+h)*256+b;
            const float4 q0=*reinterpret_cast<const float4*>(q_batch+qoff);
            const float4 q1=*reinterpret_cast<const float4*>(q_batch+qoff+4);
            const float qv[8]={q0.x,q0.y,q0.z,q0.w,q1.x,q1.y,q1.z,q1.w};
            #pragma unroll
            for(int j=0;j<8;++j){if(!isfinite(qv[j]))atomicExch(status,1);}
            #pragma unroll
            for(int row=0;row<8;++row){
                float qr=0.f;
                #pragma unroll
                for(int j=0;j<8;++j)qr=fmaf(sr[row*8+j],qv[j],qr);
                q_rot[s][row]=qr;
            }
        }
    }
    const int kp_max=pos_start+tok_base+t_cnt-1;
    const uint8_t* k_base=key_cache+size_t(kh)*ctx*key_stride;
    const uint8_t* v_base=val_cache+size_t(kh)*ctx*val_stride;
    const int sw_i=smem_swiz(i);
    const int sb0=smem_swiz(b),sb1=smem_swiz(b+4);
    for(int kp_base=0;kp_base<=kp_max;kp_base+=BK){
        const int kp_cnt=(kp_max+1-kp_base<BK)?(kp_max+1-kp_base):BK;
        for(int k=0;k<kp_cnt;++k){
            const int kp=kp_base+k;
            sk[k][sw_i]=load_key(k_base+size_t(kp)*key_stride,i,true);
            sv[k][sw_i]=load_value(v_base+size_t(kp)*val_stride,i,mode);
        }
        __syncthreads();
        for(int k=0;k<kp_cnt;++k){
            const int kp=kp_base+k;
            const float4 kv0=*reinterpret_cast<const float4*>(&sk[k][sb0]);
            const float4 kv1=*reinterpret_cast<const float4*>(&sk[k][sb1]);
            const float4 vv0=*reinterpret_cast<const float4*>(&sv[k][sb0]);
            const float4 vv1=*reinterpret_cast<const float4*>(&sv[k][sb1]);
            #pragma unroll
            for(int s=0;s<HG;++s){
                const int t=(warp+s*8)/6;
                if(t<t_cnt&&kp<=pos_start+tok_base+t){
                    float p=0.f;
                    p=fmaf(q_rot[s][0],kv0.x,p);p=fmaf(q_rot[s][1],kv0.y,p);
                    p=fmaf(q_rot[s][2],kv0.z,p);p=fmaf(q_rot[s][3],kv0.w,p);
                    p=fmaf(q_rot[s][4],kv1.x,p);p=fmaf(q_rot[s][5],kv1.y,p);
                    p=fmaf(q_rot[s][6],kv1.z,p);p=fmaf(q_rot[s][7],kv1.w,p);
                    #pragma unroll
                    for(int d=16;d>0;d>>=1)p+=__shfl_xor_sync(0xffffffffu,p,d);
                    const float sc=p*(1.f/16.f);
                    const float new_mx=fmaxf(mx[s],sc);
                    const float alpha=(kp==0)?0.f:expf(mx[s]-new_mx);
                    const float w=expf(sc-new_mx);
                    denom[s]=fmaf(denom[s],alpha,w);
                    mx[s]=new_mx;
                    acc[s][0]=fmaf(acc[s][0],alpha,w*vv0.x);acc[s][1]=fmaf(acc[s][1],alpha,w*vv0.y);
                    acc[s][2]=fmaf(acc[s][2],alpha,w*vv0.z);acc[s][3]=fmaf(acc[s][3],alpha,w*vv0.w);
                    acc[s][4]=fmaf(acc[s][4],alpha,w*vv1.x);acc[s][5]=fmaf(acc[s][5],alpha,w*vv1.y);
                    acc[s][6]=fmaf(acc[s][6],alpha,w*vv1.z);acc[s][7]=fmaf(acc[s][7],alpha,w*vv1.w);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int s=0;s<HG;++s){
        const int q_idx=warp+s*8;
        const int t=q_idx/6,h=q_idx%6;
        if(t<t_cnt){
            float norm_acc[8];
            #pragma unroll
            for(int j=0;j<8;++j)norm_acc[j]=acc[s][j]/denom[s];
            const size_t qoff=size_t(tok_base+t)*6144+size_t(kh*6+h)*256+b;
            const float4 g0=*reinterpret_cast<const float4*>(gate_batch+qoff);
            const float4 g1=*reinterpret_cast<const float4*>(gate_batch+qoff+4);
            const float gv[8]={g0.x,g0.y,g0.z,g0.w,g1.x,g1.y,g1.z,g1.w};
            float out_v[8];
            #pragma unroll
            for(int row=0;row<8;++row){
                float out_val=0.f;
                #pragma unroll
                for(int j=0;j<8;++j)out_val=fmaf(sr[j*8+row],norm_acc[j],out_val);
                const float res=out_val/(1.f+expf(-gv[row]));
                if(!isfinite(gv[row])||!isfinite(res))atomicExch(status,1);
                out_v[row]=res;
            }
            *reinterpret_cast<float4*>(out_batch+qoff)=make_float4(out_v[0],out_v[1],out_v[2],out_v[3]);
            *reinterpret_cast<float4*>(out_batch+qoff+4)=make_float4(out_v[4],out_v[5],out_v[6],out_v[7]);
        }
    }
}
__global__ void __launch_bounds__(192) spin8_attend_decode_split_kernel(
    const float* __restrict__ q_vec,const uint8_t* __restrict__ key_cache,const uint8_t* __restrict__ val_cache,
    const float* __restrict__ r,float* __restrict__ partial,int pos,int ctx,int key_stride,int val_stride,int mode,int n_splits,int* status){
    constexpr int BK=16;
    __shared__ int early_stop;
    __shared__ float sr[64];
    __shared__ alignas(16) float sk[BK][256];
    __shared__ alignas(16) float sv[BK][256];
    const int kh=blockIdx.x,sp=blockIdx.y,tid=threadIdx.x;
    if(tid==0)early_stop=atomicAdd(status,0);
    if(tid<64)sr[tid]=r[tid];
    __syncthreads();
    if(early_stop)return;
    const int lane=tid&31,h=tid>>5,b=lane*8;
    const size_t qoff=size_t(kh*6+h)*256+b;
    const float4 q0=*reinterpret_cast<const float4*>(q_vec+qoff);
    const float4 q1=*reinterpret_cast<const float4*>(q_vec+qoff+4);
    const float qv[8]={q0.x,q0.y,q0.z,q0.w,q1.x,q1.y,q1.z,q1.w};
    if(sp==0){
        #pragma unroll
        for(int j=0;j<8;++j){if(!isfinite(qv[j]))atomicExch(status,1);}
    }
    float q_rot[8],acc[8];
    #pragma unroll
    for(int row=0;row<8;++row){
        float qr=0.f;
        #pragma unroll
        for(int j=0;j<8;++j)qr=fmaf(sr[row*8+j],qv[j],qr);
        q_rot[row]=qr;acc[row]=0.f;
    }
    float mx=-INFINITY,denom=0.f;
    const int total=pos+1;
    const int chunk=(total+n_splits-1)/n_splits;
    const int kp_start=sp*chunk;
    const int kp_end=(kp_start+chunk<total)?(kp_start+chunk):total;
    const uint8_t* k_base=key_cache+size_t(kh)*ctx*key_stride;
    const uint8_t* v_base=val_cache+size_t(kh)*ctx*val_stride;
    const int sb0=smem_swiz(b),sb1=smem_swiz(b+4);
    for(int kp_base=kp_start;kp_base<kp_end;kp_base+=BK){
        const int kp_cnt=(kp_end-kp_base<BK)?(kp_end-kp_base):BK;
        const int total_elems=kp_cnt*256;
        for(int idx=tid;idx<total_elems;idx+=192){
            const int k=idx>>8,elem=idx&255;
            const int kp=kp_base+k,sw=smem_swiz(elem);
            sk[k][sw]=load_key(k_base+size_t(kp)*key_stride,elem,true);
            sv[k][sw]=load_value(v_base+size_t(kp)*val_stride,elem,mode);
        }
        __syncthreads();
        for(int k=0;k<kp_cnt;++k){
            const float4 kv0=*reinterpret_cast<const float4*>(&sk[k][sb0]);
            const float4 kv1=*reinterpret_cast<const float4*>(&sk[k][sb1]);
            const float4 vv0=*reinterpret_cast<const float4*>(&sv[k][sb0]);
            const float4 vv1=*reinterpret_cast<const float4*>(&sv[k][sb1]);
            float p=0.f;
            p=fmaf(q_rot[0],kv0.x,p);p=fmaf(q_rot[1],kv0.y,p);
            p=fmaf(q_rot[2],kv0.z,p);p=fmaf(q_rot[3],kv0.w,p);
            p=fmaf(q_rot[4],kv1.x,p);p=fmaf(q_rot[5],kv1.y,p);
            p=fmaf(q_rot[6],kv1.z,p);p=fmaf(q_rot[7],kv1.w,p);
            #pragma unroll
            for(int d=16;d>0;d>>=1)p+=__shfl_xor_sync(0xffffffffu,p,d);
            const float sc=p*(1.f/16.f);
            const float new_mx=fmaxf(mx,sc);
            const float alpha=(kp_base==kp_start&&k==0)?0.f:expf(mx-new_mx);
            const float w=expf(sc-new_mx);
            denom=fmaf(denom,alpha,w);
            mx=new_mx;
            acc[0]=fmaf(acc[0],alpha,w*vv0.x);acc[1]=fmaf(acc[1],alpha,w*vv0.y);
            acc[2]=fmaf(acc[2],alpha,w*vv0.z);acc[3]=fmaf(acc[3],alpha,w*vv0.w);
            acc[4]=fmaf(acc[4],alpha,w*vv1.x);acc[5]=fmaf(acc[5],alpha,w*vv1.y);
            acc[6]=fmaf(acc[6],alpha,w*vv1.z);acc[7]=fmaf(acc[7],alpha,w*vv1.w);
        }
        __syncthreads();
    }
    const int head=kh*6+h;
    float* dst=partial+(size_t(sp)*24+head)*258;
    #pragma unroll
    for(int c=0;c<8;++c)dst[b+c]=acc[c];
    if(lane==0){dst[256]=mx;dst[257]=denom;}
}
__global__ void __launch_bounds__(256) spin8_attend_decode_reduce_kernel(
    const float* __restrict__ partial,const float* __restrict__ r,const float* __restrict__ gate,
    float* __restrict__ output,int n_splits,int* status){
    __shared__ int early_stop;
    if(threadIdx.x==0)early_stop=atomicAdd(status,0);
    __syncthreads();
    if(early_stop)return;
    __shared__ float sacc[256],sr[64];
    const int head=blockIdx.x,i=threadIdx.x;
    const int b=(i>>3)<<3,row=i&7;
    if(i<64)sr[i]=r[i];
    __syncthreads();
    float mx=-INFINITY,denom=0.f,acc=0.f;
    for(int sp=0;sp<n_splits;++sp){
        const float* src=partial+(size_t(sp)*24+head)*258;
        const float sp_den=src[257];
        if(sp_den<=0.f)continue;
        const float sp_mx=src[256];
        const float new_mx=fmaxf(mx,sp_mx);
        const float alpha=(denom==0.f)?0.f:expf(mx-new_mx);
        const float beta=expf(sp_mx-new_mx);
        denom=fmaf(denom,alpha,beta*sp_den);
        acc=fmaf(acc,alpha,beta*src[i]);
        mx=new_mx;
    }
    sacc[i]=acc/denom;
    __syncthreads();
    float out_val=0.f;
    #pragma unroll
    for(int j=0;j<8;++j)out_val=fmaf(sr[j*8+row],sacc[b+j],out_val);
    const size_t qoff=size_t(head)*256+i;
    const float gv=gate[qoff];
    const float res=out_val/(1.f+expf(-gv));
    if(!isfinite(gv)||!isfinite(res))atomicExch(status,1);
    output[qoff]=res;
}
void demand(bool b,const char* message){if(!b)throw std::runtime_error(message);}
}
void preload_kv(){
    cudaFuncAttributes a{};
    ck(cudaFuncGetAttributes(&a,spin));ck(cudaFuncGetAttributes(&a,store_f16));ck(cudaFuncGetAttributes(&a,store_q));
    ck(cudaFuncGetAttributes(&a,store_turbo));ck(cudaFuncGetAttributes(&a,scores_kernel));ck(cudaFuncGetAttributes(&a,aggregate_kernel));
    ck(cudaFuncGetAttributes(&a,inverse_turbo));ck(cudaFuncGetAttributes(&a,apply_gate));
    ck(cudaFuncGetAttributes(&a,spin8_store_q_batch_kernel));ck(cudaFuncGetAttributes(&a,spin8_attend_batch_kernel));
    ck(cudaFuncGetAttributes(&a,spin8_attend_decode_split_kernel));ck(cudaFuncGetAttributes(&a,spin8_attend_decode_reduce_kernel));
}
void KvCache::check(){
    demand(!poisoned_,"KV session requires reset after error");
    try{int bad=0;ck(cudaMemcpyAsync(&bad,status_.data,4,cudaMemcpyDeviceToHost,stream_));
        ck(cudaStreamSynchronize(stream_));++fences_;demand(!bad,"non-finite KV input/storage/output; reset required");
    }catch(...){poisoned_=true;throw;}
}
void KvCache::step(int index,int pos,const float* q,const float* k,const float* v,const float* gate,float* output,float* scores){
    enqueue(index,pos,q,k,v,gate,output,scores);check();
}
void kv_spin_transform(const float* in,float* out,const float* matrix,int rows,bool inverse,cudaStream_t s){
    demand(in&&out&&matrix&&in!=out&&rows>0&&rows<=24,"invalid Spin8 transform");
    spin<<<rows,256,0,s>>>(in,out,matrix,rows*256,inverse);ck(cudaGetLastError());
}
KvCache::KvCache(const KvConfig& c,int capacity,int layers,cudaStream_t stream,bool conservative):config_(c),capacity_(capacity),stream_(stream),conservative_(conservative){
    demand(capacity>0&&capacity<=262144&&layers>0&&layers<=16&&stream,"invalid KV configuration");
    demand(!c.automatic&&int(c.mode)>=0&&int(c.mode)<=5,"KV cache requires selected effective mode");
    // Two owners avoid per-layer CUDA allocator granularity overhead. Views
    // remain valid for this cache lifetime and are never individually freed.
    const size_t kb=size_t(4)*capacity*c.key_row_bytes(),vb=size_t(4)*capacity*c.value_row_bytes();
    keys_.allocate(kb*layers);values_.allocate(vb*layers);layers_.resize(layers);
    for(int i=0;i<layers;++i){layers_[i].key={keys_.as<uint8_t>()+size_t(i)*kb,kb};layers_[i].value={values_.as<uint8_t>()+size_t(i)*vb,vb};}
    // These buffers hold at most ONE position, independent of context capacity.
    status_.allocate(4);if(c.mode!=KvMode::f16)aggregate_.allocate(6144*4);
    if(c.mode==KvMode::spin8_k8v4){matrix_.allocate(64*4);ck(cudaMemcpy(matrix_.data,c.matrix.data(),64*4,cudaMemcpyHostToDevice));q_.allocate(6144*4);k_.allocate(1024*4);v_.allocate(1024*4);}
    reset();
}
size_t KvCache::persistent_bytes()const{return keys_.bytes+values_.bytes;}
size_t KvCache::scratch_bytes()const{return q_.bytes+k_.bytes+v_.bytes+aggregate_.bytes+status_.bytes;}
void KvCache::reset(){
    ck(cudaStreamSynchronize(stream_));keys_.zero(stream_);values_.zero(stream_);for(auto& l:layers_)l.length=0;
    if(status_.data)status_.zero(stream_);ck(cudaStreamSynchronize(stream_));poisoned_=false;
}
void KvCache::rewind(int new_length){
    demand(!poisoned_&&new_length>=0&&new_length<=capacity_,"invalid KV rewind");
    for(auto& l:layers_){demand(new_length<=l.length,"KV rewind cannot advance");l.length=new_length;}
}
void KvCache::enqueue_batch(int index,int pos_start,int n_tokens,const float* q_batch,const float* k_batch,const float* v_batch,const float* gate_batch,float* output_batch,float* scores){
    if(n_tokens<=0)return;
    if(config_.mode==KvMode::spin8_k8v4){
        demand(!poisoned_,"KV session requires reset after error");
        demand(index>=0&&index<int(layers_.size()),"invalid attention layer");
        auto& l=layers_[index];
        demand(pos_start==l.length&&pos_start>=0&&pos_start+n_tokens<=capacity_,"KV batch position out of range");
        demand(q_batch&&k_batch&&v_batch&&gate_batch&&output_batch,"null KV operand");
        try{
            spin8_store_q_batch_kernel<<<n_tokens,32,0,stream_>>>(k_batch,l.key.as<uint8_t>(),matrix_.as(),pos_start,n_tokens,capacity_,true,status_.as<int>());
            spin8_store_q_batch_kernel<<<n_tokens,32,0,stream_>>>(v_batch,l.value.as<uint8_t>(),matrix_.as(),pos_start,n_tokens,capacity_,false,status_.as<int>());
            spin8_attend_batch_kernel<<<dim3(4,unsigned((n_tokens+7)/8)),256,0,stream_>>>(
                q_batch,gate_batch,l.key.as<uint8_t>(),l.value.as<uint8_t>(),matrix_.as(),output_batch,
                pos_start,n_tokens,capacity_,int(config_.key_row_bytes()),int(config_.value_row_bytes()),int(config_.mode),status_.as<int>());
            ck(cudaGetLastError());
            l.length+=n_tokens;
        }catch(...){poisoned_=true;throw;}
        return;
    }
    for(int t=0;t<n_tokens;++t){
        enqueue(index,pos_start+t,q_batch+size_t(t)*6144,k_batch+size_t(t)*1024,v_batch+size_t(t)*1024,gate_batch+size_t(t)*6144,output_batch+size_t(t)*6144,scores);
    }
}
void KvCache::enqueue(int index,int pos,const float* q,const float* k,const float* v,const float* gate,float* output,float* scores){
    demand(!poisoned_,"KV session requires reset after error");demand(index>=0&&index<int(layers_.size()),"invalid attention layer");auto& l=layers_[index];
    // Admission does not mutate cache; unsupported rewinds/restore/shift fail here.
    demand(pos==l.length&&pos>=0&&pos<capacity_,"KV position is not the next causal cell");
    demand(q&&k&&v&&gate&&output&&scores,"null KV operand");
    try{
        if(config_.mode==KvMode::spin8_k8v4){
            spin8_store_q_batch_kernel<<<1,32,0,stream_>>>(k,l.key.as<uint8_t>(),matrix_.as(),pos,1,capacity_,true,status_.as<int>());
            spin8_store_q_batch_kernel<<<1,32,0,stream_>>>(v,l.value.as<uint8_t>(),matrix_.as(),pos,1,capacity_,false,status_.as<int>());
            ck(cudaGetLastError());if(conservative_)check();
            const int max_splits=std::min(16,capacity_/258);
            const int want_splits=(pos>=128)?std::min(16,(pos+127)/128):1;
            const int n_splits=std::min(max_splits,want_splits);
            if(n_splits>=2){
                spin8_attend_decode_split_kernel<<<dim3(4,unsigned(n_splits)),192,0,stream_>>>(
                    q,l.key.as<uint8_t>(),l.value.as<uint8_t>(),matrix_.as(),scores,pos,capacity_,
                    int(config_.key_row_bytes()),int(config_.value_row_bytes()),int(config_.mode),n_splits,status_.as<int>());
                spin8_attend_decode_reduce_kernel<<<24,256,0,stream_>>>(scores,matrix_.as(),gate,output,n_splits,status_.as<int>());
            }else{
                spin8_attend_batch_kernel<<<dim3(4,1),256,0,stream_>>>(
                    q,gate,l.key.as<uint8_t>(),l.value.as<uint8_t>(),matrix_.as(),output,pos,1,capacity_,
                    int(config_.key_row_bytes()),int(config_.value_row_bytes()),int(config_.mode),status_.as<int>());
            }
            ck(cudaGetLastError());if(conservative_)check();++l.length;
            return;
        }
        ck(cudaMemsetAsync(output,0,6144*4,stream_));
        if(aggregate_.data)aggregate_.zero(stream_);
        finite(q,6144,status_.as<int>(),stream_);finite(k,1024,status_.as<int>(),stream_);finite(v,1024,status_.as<int>(),stream_);finite(gate,6144,status_.as<int>(),stream_);
        if(conservative_)check();
        if(config_.mode==KvMode::f16){
            store_f16<<<4,256,0,stream_>>>(k,l.key.as<uint8_t>(),pos,capacity_,status_.as<int>());store_f16<<<4,256,0,stream_>>>(v,l.value.as<uint8_t>(),pos,capacity_,status_.as<int>());
            ck(cudaGetLastError());if(conservative_)check();
            attention_apply(q,gate,l.key.as<half>(),l.value.as<half>(),output,scores,pos,capacity_,stream_,status_.as<int>());
        }else{
            if(config_.mode==KvMode::spin8_k8v4){
                kv_spin_transform(q,q_.as(),matrix_.as(),24,false,stream_);kv_spin_transform(k,k_.as(),matrix_.as(),4,false,stream_);kv_spin_transform(v,v_.as(),matrix_.as(),4,false,stream_);
                q=q_.as();k=k_.as();v=v_.as();
                finite(q,6144,status_.as<int>(),stream_);finite(k,1024,status_.as<int>(),stream_);finite(v,1024,status_.as<int>(),stream_);
                if(conservative_)check();
            }
            bool q8=config_.key_row_bytes()==272;
            if(q8)store_q<<<1,32,0,stream_>>>(k,l.key.as<uint8_t>(),pos,capacity_,true,status_.as<int>());
            else store_f16<<<4,256,0,stream_>>>(k,l.key.as<uint8_t>(),pos,capacity_,status_.as<int>());
            if(config_.turbo())store_turbo<<<1,32,0,stream_>>>(v,l.value.as<uint8_t>(),pos,capacity_,config_.turbo(),status_.as<int>());
            else store_q<<<1,32,0,stream_>>>(v,l.value.as<uint8_t>(),pos,capacity_,false,status_.as<int>());
            ck(cudaGetLastError());if(conservative_)check();
            scores_kernel<<<dim3((pos+8)/8,24),256,0,stream_>>>(q,l.key.as<uint8_t>(),scores,pos,capacity_,int(config_.key_row_bytes()),q8,status_.as<int>());
            aggregate_kernel<<<24,256,0,stream_>>>(scores,l.value.as<uint8_t>(),aggregate_.as(),pos,capacity_,int(config_.value_row_bytes()),int(config_.mode),status_.as<int>());
            if(config_.mode==KvMode::spin8_k8v4)kv_spin_transform(aggregate_.as(),output,matrix_.as(),24,true,stream_);
            else if(config_.turbo())inverse_turbo<<<2,32,0,stream_>>>(aggregate_.as(),output);
            else ck(cudaMemcpyAsync(output,aggregate_.data,6144*4,cudaMemcpyDeviceToDevice,stream_));
            apply_gate<<<24,256,0,stream_>>>(output,gate);
        }
        finite(output,6144,status_.as<int>(),stream_);ck(cudaGetLastError());if(conservative_)check();++l.length;
    }catch(...){poisoned_=true;throw;}
}
}
