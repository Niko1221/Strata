#include "prefill_gemm.cuh"
#include "ops.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace Q=strata::qwen38;
namespace {
void cb(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS test error="+std::to_string(int(e)));}
struct Blas {cublasHandle_t h=nullptr;Blas(){cb(cublasCreate(&h));}~Blas(){if(h)cublasDestroy(h);}};
struct Events {
    cudaEvent_t a=nullptr,b=nullptr;
    Events(){Q::ck(cudaEventCreate(&a));Q::ck(cudaEventCreate(&b));}
    ~Events(){if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}
    template<class F> float time(F&& fn){
        Q::ck(cudaEventRecord(a));for(int i=0;i<5;++i)fn();Q::ck(cudaEventRecord(b));
        Q::ck(cudaEventSynchronize(b));float ms=0;Q::ck(cudaEventElapsedTime(&ms,a,b));return ms/5;
    }
};
void scale16(uint8_t* p,float f){half v=__float2half_rn(f);uint16_t bits;std::memcpy(&bits,&v,2);p[0]=uint8_t(bits);p[1]=uint8_t(bits>>8);}
std::vector<uint8_t> packed_fixture(int type,int K,int N,size_t rb){
    std::vector<uint8_t> bytes(size_t(N)*rb);
    const auto qt=static_cast<Q::GgmlType>(type);
    const size_t block=Q::StrataQuantTensor::block_bytes(qt);
    for(int r=0;r<N;++r){
        auto* p=bytes.data()+size_t(r)*rb;
        if(type==0||type==1||type==30){
            for(int k=0;k<K;++k){
                float v=0.04f*std::cos(0.003f*float(r*K+k));
                if(type==0)std::memcpy(p+size_t(k)*4,&v,4);
                else if(type==1)scale16(p+size_t(k)*2,v);
                else {uint32_t u;std::memcpy(&u,&v,4);p[size_t(k)*2]=uint8_t(u>>16);p[size_t(k)*2+1]=uint8_t(u>>24);}
            }
        }else{
            for(size_t off=0;off<rb;off+=block){
                auto* b=p+off;
                for(size_t i=0;i<block;++i)b[i]=uint8_t((r*31+int(off)*17+int(i)*13)&255);
                scale16(b+(type==14?208:0),0.001f);
                if(type==12||type==13)scale16(b+2,0.001f);
            }
        }
    }
    return bytes;
}
void run_case(Blas& blas,int type,int K,int N,int T){
    auto ty=static_cast<Q::GgmlType>(type);
    size_t rb=size_t(K/Q::StrataQuantTensor::block_elements(ty))*Q::StrataQuantTensor::block_bytes(ty);
    auto hw=packed_fixture(type,K,N,rb);
    std::vector<half> hx(size_t(T)*K);
    for(size_t i=0;i<hx.size();++i)hx[i]=__float2half_rn(0.05f*std::sin(0.013f*float(i)));
    Q::Buffer dw(hw.size()),dx(hx.size()*2),wf(size_t(N)*K*2),ref(size_t(T)*N*4),got(size_t(T)*N*4);
    Q::ck(cudaMemcpy(dw.data,hw.data(),hw.size(),cudaMemcpyHostToDevice));
    Q::ck(cudaMemcpy(dx.data,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    const auto weight=Q::make_quant_tensor(dw.data,type,K,N);
    const float alpha=1.0f,beta=0.0f;
    auto reference=[&]{
        Q::dequant_rows_fp16(weight,0,N,wf.as<half>(),nullptr);
        cb(cublasGemmEx(blas.h,CUBLAS_OP_T,CUBLAS_OP_N,N,T,K,&alpha,
            wf.data,CUDA_R_16F,K,dx.data,CUDA_R_16F,K,&beta,ref.data,CUDA_R_32F,N,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto candidate=[&]{Q::matmul_fused_wmma_fp16(weight,dx.as<half>(),got.as<float>(),T,nullptr);};
    reference();candidate();Q::ck(cudaDeviceSynchronize());
    std::vector<float> a(size_t(T)*N),b(a.size());
    Q::ck(cudaMemcpy(a.data(),ref.data,a.size()*4,cudaMemcpyDeviceToHost));
    Q::ck(cudaMemcpy(b.data(),got.data,b.size()*4,cudaMemcpyDeviceToHost));
    double maxerr=0,maxref=0,se=0,sr=0;
    for(size_t i=0;i<a.size();++i){
        if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite test output");
        double e=double(a[i])-b[i];maxerr=std::max(maxerr,std::abs(e));maxref=std::max(maxref,std::abs(double(a[i])));se+=e*e;sr+=double(a[i])*a[i];
    }
    const double nrmse=std::sqrt(se/std::max(sr,1e-20));
    if(maxerr>1e-3+1e-3*maxref || nrmse>1e-3)throw std::runtime_error("fused WMMA numerical parity failure");
    Events ev;float reference_ms=ev.time(reference),candidate_ms=ev.time(candidate);
    std::printf("type=%d K=%d N=%d T=%d max_abs=%.8g nrmse=%.8g ref_dequant_gemm_ms=%.6f fused_ms=%.6f ratio=%.5f\n",
        type,K,N,T,maxerr,nrmse,reference_ms,candidate_ms,reference_ms/std::max(candidate_ms,1e-12f));
}
}
int main(){
    try{
        Q::ck(cudaSetDevice(0));cudaDeviceProp prop{};Q::ck(cudaGetDeviceProperties(&prop,0));
        if(prop.major!=12||prop.minor!=0)throw std::runtime_error("qualification target must be SM120");
        int runtime=0,driver=0;Q::ck(cudaRuntimeGetVersion(&runtime));Q::ck(cudaDriverGetVersion(&driver));
        std::printf("GPU=%s runtime=%d driver_api=%d cublas=%d\n",prop.name,runtime,driver,Q::PrefillGemmWorkspace::cublas_version());
        Blas blas;int count=0;
        const int types[]={0,1,8,12,13,14,21,22,23,30,142};
        const int shapes[][3]={{256,17,1},{512,63,15},{512,65,32},{5120,257,33},{6144,257,64},{17408,65,65},{512,513,128},{272,65,33},{514,65,33}};
        for(int type:types)for(const auto& s:shapes){
            if(s[0]%Q::StrataQuantTensor::block_elements(static_cast<Q::GgmlType>(type)))continue;
            run_case(blas,type,s[0],s[1],s[2]);++count;
        }
        std::printf("fused WMMA arithmetic tests PASS: %d cases; not an end-to-end model/llama.cpp qualification\n",count);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
