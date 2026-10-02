// Numerical screen of the actual FP16 prompt route on GP100, including the
// BF16 -> FP16 conversion and native Q4_K / Q5_K / Q5_1 / Q8_0 weights.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Dev {
    void* p=nullptr;
    Dev(size_t bytes) { ck(cudaMalloc(&p,bytes)); }
    ~Dev() { cudaFree(p); }
};
void check(strata::prefill::Gemm& gemm, cudaStream_t stream, ggml_type type, int N, int K) {
    const int T=7;
    std::vector<float> x(T*K), w(N*K);
    std::vector<uint16_t> xb(T*K);
    for(int i=0;i<T*K;++i) {
        float f = i >= (T-1)*K ? 0.0f : 0.35f*std::sin(i*.009f)+.17f*std::cos(i*.021f);
        if(type==GGML_TYPE_BF16) {
            auto b=ggml_fp32_to_bf16(f); xb[i]=b.bits;
            x[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(ggml_bf16_to_fp32(b)));
        } else { xb[i]=ggml_fp32_to_fp16(f); x[i]=ggml_fp16_to_fp32(xb[i]); }
    }
    for(int i=0;i<N*K;++i) w[i]=.025f*std::sin(i*.013f)+.0015f*((i*37)%101-50);
    std::vector<uint8_t> raw;
    if(type==GGML_TYPE_BF16) {
        raw.resize(w.size()*2);
        auto* b=(ggml_bf16_t*)raw.data();
        for(size_t i=0;i<w.size();++i) {
            b[i]=ggml_fp32_to_bf16(w[i]);
            w[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(ggml_bf16_to_fp32(b[i])));
        }
    } else {
        const auto* tr=ggml_get_type_traits(type);
        size_t row=ggml_row_size(type,K);
        raw.resize(N*row);
        for(int r=0;r<N;++r) {
            tr->from_float_ref(w.data()+r*K,raw.data()+r*row,K);
            tr->to_float(raw.data()+r*row,w.data()+r*K,K);
        }
    }
    Dev dx(xb.size()*2), dw(raw.size()), dy(T*N*4);
    ck(cudaMemcpy(dx.p,xb.data(),xb.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dw.p,raw.data(),raw.size(),cudaMemcpyHostToDevice));
    ck(cudaMemset(dy.p,0xff,T*N*4));
    ck(cudaDeviceSynchronize()); // default-stream copies before the nonblocking compute stream
    if(type==GGML_TYPE_BF16)
        gemm.bf16((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p,T,N,K);
    else gemm.native((uint16_t*)dx.p,(int)type,dw.p,(float*)dy.p,T,N,K);
    ck(cudaStreamSynchronize(stream));
    if(type!=GGML_TYPE_BF16) {
        std::vector<uint16_t> dq(N*K);
        ck(cudaMemcpy(dq.data(),gemm.scratch(),dq.size()*2,cudaMemcpyDeviceToHost));
        size_t diff=0;
        double de=0, dr=0, dm=0;
        for(size_t i=0;i<dq.size();++i) {
            auto expected=ggml_fp32_to_fp16(w[i]);
            diff+=dq[i]!=expected;
            double ref=ggml_fp16_to_fp32(expected), got=ggml_fp16_to_fp32(dq[i]);
            double d=got-ref; de+=d*d; dr+=ref*ref; dm=std::fmax(dm,std::fabs(d));
        }
        std::printf("%s GPU dequant FP16: %zu / %zu bits differ, rel_l2 %.6g max %.6g\n",ggml_type_name(type),diff,dq.size(),std::sqrt(de/dr),dm);
        if(std::sqrt(de/dr)>.001 || !std::isfinite(de)) throw std::runtime_error("GPU FP16 dequant outside rounding tolerance");
    }
    std::vector<float> got(T*N);
    ck(cudaMemcpy(got.data(),dy.p,T*N*4,cudaMemcpyDeviceToHost));
    double err2=0, ref2=0, maxerr=0, zero=0;
    for(int t=0;t<T;++t) for(int n=0;n<N;++n) {
        double ref=0;
        for(int k=0;k<K;++k) ref+=(double)x[t*K+k]*w[n*K+k];
        double v=got[t*N+n];
        if(!std::isfinite(v)) throw std::runtime_error("non-finite or unwritten FP16 output");
        double d=v-ref; err2+=d*d; ref2+=ref*ref;
        maxerr=std::fmax(maxerr,std::fabs(d));
        if(t==T-1) zero=std::fmax(zero,std::fabs(v));
    }
    double rel=std::sqrt(err2/ref2), rms=std::sqrt(ref2/(T*N));
    std::printf("%s %dx%d: rel_l2 %.6g max/rms %.6g zero %.6g\n",ggml_type_name(type),N,K,rel,maxerr/rms,zero);
    if(rel>.003 || maxerr/rms>.02 || zero!=0) throw std::runtime_error("FP16 GPU numerical screen failed");
}
int main() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
        prop.major != 6 || prop.minor != 0) {
        std::printf("prefill_p100_test: requires a GP100 (sm_60), skipped\n");
        return 77;
    }
    try {
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        {
            int32_t *host=nullptr; void* alias=nullptr;
            ck(cudaHostAlloc((void**)&host,8192*sizeof(int32_t),cudaHostAllocMapped));
            ck(cudaHostGetDevicePointer(&alias,host,0));
            Dev ids(8192*sizeof(int32_t)), logits(40*512*sizeof(float)), weights(40*10*sizeof(float));
            ck(cudaMemsetAsync(logits.p,0,40*512*sizeof(float),stream));
            strata::prefill::route((float*)logits.p,(int32_t*)ids.p,(float*)weights.p,40,512,stream);
            ck(cudaStreamSynchronize(stream));
            strata::prefill::copy_i32((int32_t*)alias,(int32_t*)ids.p,400,stream);
            ck(cudaStreamSynchronize(stream));
            for(int i=0;i<400;++i) if(host[i]!=i%10) throw std::runtime_error("mapped router ids differ");
            ck(cudaFreeHost(host));
            std::printf("P100 mapped host grouping and route passed\n");
        }
        {
            strata::prefill::Gemm gemm; std::string err;
            if(!gemm.init(stream,1280*2560,err)) throw std::runtime_error(err);
            check(gemm,stream,GGML_TYPE_Q4_K,1280,2560);
            check(gemm,stream,GGML_TYPE_Q5_K,1280,2560);
            check(gemm,stream,GGML_TYPE_Q5_1,2560,640);
            check(gemm,stream,GGML_TYPE_Q8_0,2560,640);
            check(gemm,stream,GGML_TYPE_BF16,320,10240);
        }
        ck(cudaStreamDestroy(stream));
        std::printf("P100 FP16 prompt route passed\n"); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
