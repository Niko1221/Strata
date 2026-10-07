#include "strata/prefill/dense_q8_t8.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace strata::prefill {
namespace {
constexpr int K=2560, N=10240, T=8;
__global__ void repack_q8_1(const uint8_t* src,uint8_t* dst) {
    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    const int block=tid/32,lane=tid%32;
    if(block>=T*(K/128))return;
    const int group=block/T,row=block%T;
    uint8_t* output=dst+block*144;
    #pragma unroll
    for(int s=0;s<4;++s) {
        const uint8_t* input=src+(row*(K/32)+group*4+s)*36;
        output[16+s*32+lane]=input[4+lane];
    }
    if(lane<4) {
        const uint8_t* input=src+(row*(K/32)+group*4+lane)*36;
        reinterpret_cast<float*>(output)[lane]=__half2float(*reinterpret_cast<const __half*>(input));
    }
}
struct State {
    mmq::Context context;
    void* packed=nullptr;
    int32_t *bounds=nullptr,*ids=nullptr;
    cudaStream_t stream=nullptr;
    ~State() {
        if(stream)cudaStreamSynchronize(stream);
        if(ids)cudaFree(ids);
        if(bounds)cudaFree(bounds);
        if(packed)cudaFree(packed);
    }
    void product(const void* w,float* y) {
        mmq::Product p;
        p.w=w;p.type=8;p.w_rows=N;p.w_cols=K;
        p.expert_bytes=mmq::matrix_bytes(8,N,K);p.n=1;
        p.xq=packed;p.bounds=bounds;p.ids=ids;p.total_rows=T;p.max_rows=T;p.dst=y;p.ld_dst=N;
        context.run(p,stream);
    }
};
struct Allocation {
    void* p=nullptr;
    ~Allocation(){if(p)cudaFree(p);}
};
void check(cudaError_t e) {
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
}

DenseQ8T8::~DenseQ8T8(){close();}
void DenseQ8T8::close(){delete static_cast<State*>(state_);state_=nullptr;}
bool DenseQ8T8::open(void* stream,std::string& error) {
    error.clear();
    if(state_||!stream){error="dense T8 MMQ requires one unused owner and an explicit stream";return false;}
    try {
        int device=0;check(cudaGetDevice(&device));cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop,device));
        if(prop.major!=12||prop.minor!=0||!mmq::built()||!mmq::fits(8,N)) {
            error="dense T8 MMQ pilot requires SM120 and a supported Q8 MMQ tile";return false;
        }
        auto s=std::make_unique<State>();s->stream=static_cast<cudaStream_t>(stream);
        check(cudaMalloc(&s->packed,mmq::q8_bytes(T,K)));
        check(cudaMemsetAsync(s->packed,0,mmq::q8_bytes(T,K),s->stream));
        check(cudaMalloc(&s->bounds,2*sizeof(int32_t)));check(cudaMalloc(&s->ids,T*sizeof(int32_t)));
        const int32_t bounds[2]={0,T},ids[T]={0,1,2,3,4,5,6,7};
        check(cudaMemcpyAsync(s->bounds,bounds,sizeof(bounds),cudaMemcpyHostToDevice,s->stream));
        check(cudaMemcpyAsync(s->ids,ids,sizeof(ids),cudaMemcpyHostToDevice,s->stream));
        // Exercise the exact shape and stream before capture, so the MMQ
        // context's internal scratch pool is already populated.
        Allocation w,y;
        check(cudaMalloc(&w.p,mmq::matrix_bytes(8,N,K)+4096));
        check(cudaMalloc(&y.p,T*N*sizeof(float)));
        check(cudaMemsetAsync(w.p,0,mmq::matrix_bytes(8,N,K)+4096,s->stream));
        s->product(w.p,static_cast<float*>(y.p));check(cudaStreamSynchronize(s->stream));
        state_=s.release();return true;
    }catch(const std::exception& e){error=std::string("dense T8 MMQ: ")+e.what();return false;}
}
bool DenseQ8T8::run(int type,const void* w,const void* native_q8_1,float* y,
                    int k,int n,int t,void* stream) {
    auto* s=static_cast<State*>(state_);
    if(!s||type!=8||k!=K||n!=N||t!=T||stream!=s->stream||!w||!native_q8_1||!y)return false;
    constexpr int threads=T*(K/128)*32;
    repack_q8_1<<<(threads+255)/256,256,0,s->stream>>>(static_cast<const uint8_t*>(native_q8_1),static_cast<uint8_t*>(s->packed));
    check(cudaPeekAtLastError());s->product(w,y);return true;
}
}
