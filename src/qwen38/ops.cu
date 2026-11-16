// Packed scalar decoders follow Strata iq_kernels.cu (ggml MIT formats);
// PQ2_0 follows PrismML 0.4.1 commit 1965755ceee6738b9a04ceaf5d30e6305d6ccd00.
// See README.md and LICENSE-formats.txt. No model/graph/runtime code is linked.
#include "ops.cuh"
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
    case 8:return size_t(n/32)*34;case 12:return size_t(n/256)*144;case 13:return size_t(n/256)*176;case 14:return size_t(n/256)*210;case 23:return size_t(n/256)*136;default:throw std::runtime_error("unsupported matrix type");}
}
void shape(int type,int n,int rows){int block=type==142?128:type==8?32:(type==12||type==13||type==14||type==23)?256:1;
    if(n<=0||rows<=0||n%block)throw std::runtime_error("invalid packed matrix dimensions");(void)row_bytes(type,n);}
__global__ void deq(int type,const uint8_t* p,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=value(type,p,i);}
__device__ float sum(float v){for(int i=16;i;i/=2)v+=__shfl_down_sync(0xffffffffu,v,i);return v;}
__global__ void mv(int type,const uint8_t* w,const float* x,float* y,int n,int rows,size_t stride){
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=rows)return;
    float s=0;w+=size_t(row)*stride;
    for(int j=lane;j<n;j+=32)s=fmaf(value(type,w,j),x[j],s);
    s=sum(s);if(!lane)y[row]=s;
}
__global__ void norm(const float* x,const float* w,float* y,int n,float eps){
    __shared__ float sums[8];float v=0;for(int i=threadIdx.x;i<n;i+=256)v+=x[i]*x[i];v=sum(v);
    if(!(threadIdx.x%32))sums[threadIdx.x/32]=v;__syncthreads();
    float s=0;for(int i=0;i<8;++i)s+=sums[i];float r=rsqrtf(s/n+eps);
    for(int i=threadIdx.x;i<n;i+=256)y[i]=x[i]*r*w[i];
}
__global__ void elt(float* x,const float* y,int n,bool sw){int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]=sw?x[i]/(1+expf(-x[i]))*y[i]:x[i]+y[i];}
__global__ void gb(float* al,float* be,const float* dt,const float* a,int n){int i=threadIdx.x;if(i<n){float v=al[i]+dt[i];al[i]=(v>20?v:log1pf(expf(v)))*a[i];be[i]=1/(1+expf(-be[i]));}}
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
                     half* kc,half* vc,int pos,int ctx,float eps,float base){
    __shared__ float buf[256],part[8];int head=blockIdx.x,i=threadIdx.x;bool isq=head<24;
    const float* x=isq?qg+head*512:k+(head-24)*256;float z=x[i];float s=sum(z*z);
    if(!(i%32))part[i/32]=s;__syncthreads();s=0;for(int j=0;j<8;++j)s+=part[j];
    buf[i]=z*rsqrtf(s/256+eps)*(isq?qw[i]:kw[i]);__syncthreads();z=buf[i];
    if(i<64){int p=i%32;float angle=pos*powf(base,-float(p)/32),sn,cs;sincosf(angle,&sn,&cs);z=i<32?buf[i]*cs-buf[i+32]*sn:buf[i-32]*sn+buf[i]*cs;}
    if(isq){q[head*256+i]=z;gate[head*256+i]=qg[head*512+256+i];}
    else{int kh=head-24;kc[(size_t(kh)*ctx+pos)*256+i]=__float2half(z);vc[(size_t(kh)*ctx+pos)*256+i]=__float2half(v[kh*256+i]);}
}
__global__ void score(const float* q,const half* k,float* s,int pos,int ctx){
    int t=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y,lane=threadIdx.x%32;if(t>pos)return;int kh=head/6;
    float a=0;for(int i=lane;i<256;i+=32)a=fmaf(q[head*256+i],__half2float(k[(size_t(kh)*ctx+t)*256+i]),a);
    a=sum(a);if(!lane)s[size_t(head)*ctx+t]=a/16;
}
__global__ void attend(const float* scores,const half* v,const float* gate,float* out,int pos,int ctx){
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
void decode(int type,const void* p,float* y,int n,cudaStream_t s){shape(type,n,1);if(!p||!y)throw std::runtime_error("null decoder input");deq<<<(n+255)/256,256,0,s>>>(type,(const uint8_t*)p,y,n);ck(cudaGetLastError());}
void matvec(int type,const void* w,const float* x,float* y,int n,int rows,cudaStream_t s){shape(type,n,rows);if(!w||!x||!y)throw std::runtime_error("null matrix input");mv<<<(rows+7)/8,256,0,s>>>(type,(const uint8_t*)w,x,y,n,rows,row_bytes(type,n));ck(cudaGetLastError());}
void rms(const float* x,const float* w,float* y,int n,float eps,cudaStream_t s){norm<<<1,256,0,s>>>(x,w,y,n,eps);ck(cudaGetLastError());}
void add(float* x,const float* y,int n,cudaStream_t s){elt<<<(n+255)/256,256,0,s>>>(x,y,n,false);ck(cudaGetLastError());}
void swiglu(float* x,const float* y,int n,cudaStream_t s){elt<<<(n+255)/256,256,0,s>>>(x,y,n,true);ck(cudaGetLastError());}
void gdn_gates(float* al,float* be,const float* dt,const float* a,int n,cudaStream_t s){gb<<<1,256,0,s>>>(al,be,dt,a,n);ck(cudaGetLastError());}
void finite(const float* x,size_t n,int* e,cudaStream_t s){fin<<<unsigned((n+255)/256),256,0,s>>>(x,n,e);ck(cudaGetLastError());}
void hadamard(const float* x,const float* signs,float* y,int n,bool inverse,bool grouped,cudaStream_t s){
    if(n<=0||n%1024||!x||!signs||!y||x==y||(grouped&&(n!=6144||inverse)))throw std::runtime_error("invalid Hadamard arguments");
    had<<<n/1024,1024,0,s>>>(x,signs,y,inverse,grouped);ck(cudaGetLastError());}
void attention_prepare(const float* qg,const float* k,const float* v,const float* qw,const float* kw,float* q,float* gate,half* kc,half* vc,int pos,int ctx,float eps,float base,cudaStream_t s){
    if(pos<0||pos>=ctx)throw std::runtime_error("attention position out of range");prep<<<28,256,0,s>>>(qg,k,v,qw,kw,q,gate,kc,vc,pos,ctx,eps,base);ck(cudaGetLastError());}
void attention_apply(const float* q,const float* gate,const half* kc,const half* vc,float* out,float* scores,int pos,int ctx,cudaStream_t s){
    if(pos<0||pos>=ctx)throw std::runtime_error("attention position out of range");score<<<dim3((pos+8)/8,24),256,0,s>>>(q,kc,scores,pos,ctx);attend<<<24,256,0,s>>>(scores,vc,gate,out,pos,ctx);ck(cudaGetLastError());}
}
