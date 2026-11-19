#include "strata/core/spin8_attention.hpp"
#include "spin8/spin8.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string error;
    if (strata::core::spin8_attention_configure("invalid", argv[1], error)) return 3;
    if (!strata::core::spin8_attention_configure("consensus", argv[1], error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 4;
    }
    constexpr size_t heads=2, keys=4, d=8;
    std::array<float,heads*d> q{}, actual{};
    std::array<half,keys*d> k{},v{};
    std::array<double,heads*d> qd{},expected{};
    std::array<double,keys*d> kd{},vd{};
    for(size_t i=0;i<q.size();++i) qd[i]=q[i]=float(int(i%7)-3)/8;
    for(size_t i=0;i<k.size();++i) {
        k[i]=__float2half(float(int(i%11)-5)/8); v[i]=__float2half(float(int(i%13)-6)/8);
        kd[i]=__half2float(k[i]); vd[i]=__half2float(v[i]);
    }
    spin8_plan* plan=nullptr;
    spin8_plane planes[]={{0,1,.37},{2,3,-.51},{0,7,.23}};
    if(spin8_plan_create(planes,3,&plan)!=SPIN8_OK) return 5;
    if(spin8_attention_f64(plan,qd.data(),heads,kd.data(),keys,vd.data(),d,d,nullptr,nullptr,expected.data())!=SPIN8_OK) return 6;
    spin8_plan_free(plan);
    float *dq=nullptr,*out=nullptr,*scores=nullptr;half *dk=nullptr,*dv=nullptr;
    int32_t *step=nullptr,*status=nullptr,*host=nullptr;
    cudaStream_t stream=nullptr;
    auto check=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));return false;}return true;};
    if(!check(cudaStreamCreate(&stream)) || !check(cudaMalloc(&dq,sizeof(q))) || !check(cudaMalloc(&out,sizeof(actual))) ||
       !check(cudaMalloc(&scores,keys*sizeof(float))) || !check(cudaMalloc(&dk,sizeof(k))) ||
       !check(cudaMalloc(&dv,sizeof(v))) || !check(cudaMalloc(&step,4*sizeof(int32_t))) ||
       !check(cudaMalloc(&status,sizeof(int32_t))) || !check(cudaMallocHost(&host,sizeof(int32_t)))) return 7;
    const int32_t counts[]={3,4,1,4};
    if(!check(cudaMemcpyAsync(dq,q.data(),sizeof(q),cudaMemcpyHostToDevice,stream)) ||
       !check(cudaMemcpyAsync(dk,k.data(),sizeof(k),cudaMemcpyHostToDevice,stream)) ||
       !check(cudaMemcpyAsync(dv,v.data(),sizeof(v),cudaMemcpyHostToDevice,stream)) ||
       !check(cudaMemcpyAsync(step,counts,sizeof(counts),cudaMemcpyHostToDevice,stream))) return 8;
    if(!strata::core::spin8_attention_step(dq,reinterpret_cast<uint16_t*>(dk),reinterpret_cast<uint16_t*>(dv),step,
        keys,heads,1,d,scores,out,status,host,stream,error)){std::fprintf(stderr,"%s\n",error.c_str());return 9;}
    if(!check(cudaMemcpyAsync(actual.data(),out,sizeof(actual),cudaMemcpyDeviceToHost,stream)) ||
        !check(cudaStreamSynchronize(stream)) || *host) return 10;
    double maximum=0;for(size_t i=0;i<actual.size();++i)maximum=std::fmax(maximum,std::abs(actual[i]-expected[i]));
    if(!strata::core::spin8_attention_check(error)) return 11;
    cudaFree(dq);cudaFree(out);cudaFree(scores);cudaFree(dk);cudaFree(dv);cudaFree(step);cudaFree(status);cudaFreeHost(host);cudaStreamDestroy(stream);
    std::printf("native GQA reference max_abs_error=%.17g\n",maximum);
    return maximum<1e-5?0:12;
}
