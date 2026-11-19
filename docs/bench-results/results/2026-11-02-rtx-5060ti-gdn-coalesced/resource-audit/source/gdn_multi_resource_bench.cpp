// Synthetic timing of the actual native verify GDN + Q8_1 commit branch.
// Compile this unchanged against baseline and experimental verify_kernels objects.
// Correctness is separately checked by the existing gdn_parity executable.
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
template<class T> static T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)));
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
int main() {
    constexpr int S=128, HK=16, HV=48, VD=S*HV, C=2*S*HK+VD, MT=8;
    constexpr size_t STATE=size_t(S)*HV*S;
    cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, 0));
    std::printf("{\"kind\":\"device\",\"name\":\"%s\",\"sm\":%d,\"state_bytes\":%zu}\n", p.name, p.multiProcessorCount, STATE*4);
    std::vector<float> h(MT*C), gate(MT*HV,-0.2f), beta(MT*HV,0.5f), z(MT*VD), gamma(S,1.0f);
    for (size_t i=0;i<h.size();++i) h[i]=0.05f*std::sin(float(i)*0.13f);
    for (int t=0;t<MT;++t) for (int head=0;head<2*HK;++head) {
        float sum=0; for(int c=0;c<S;++c) sum+=h[t*C+head*S+c]*h[t*C+head*S+c];
        for(int c=0;c<S;++c) h[t*C+head*S+c]/=std::sqrt(sum);
    }
    for (size_t i=0;i<z.size();++i) z[i]=0.1f*std::cos(float(i)*0.17f);
    float *dh=upload(h), *dg=upload(gate), *db=upload(beta), *dz=upload(z), *dgm=upload(gamma), *dy=nullptr;
    int32_t* keep=nullptr; void* xq=nullptr;
    ck(cudaMalloc(&dy,MT*VD*sizeof(float))); ck(cudaMalloc(&keep,sizeof(int32_t)));
    ck(cudaMalloc(&xq,size_t(MT)*VD/32*36));
    cudaStream_t stream=nullptr; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t begin,end; ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
    for(int buffers : {1,48}) {
        float* states=nullptr; ck(cudaMalloc(&states,buffers*STATE*4)); ck(cudaMemset(states,0,buffers*STATE*4));
        for(int tokens : {1,2,4,8}) {
            ck(cudaMemcpy(keep,&tokens,sizeof(tokens),cudaMemcpyHostToDevice));
            const int repeats=buffers==1?64:4, calls=buffers*repeats;
            cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int r=0;r<repeats;++r) for(int b=0;b<buffers;++b)
                strata::kernels::gdn_step_norm_multi(states+b*STATE,dh,C,dg,db,dz,dgm,1e-6f,dy,HK,HV,tokens,keep,stream,0,xq);
            ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            for(int r=0;r<3;++r) ck(cudaGraphLaunch(exec,stream));
            ck(cudaStreamSynchronize(stream));
            for(int r=0;r<15;++r) {
                ck(cudaEventRecord(begin,stream)); ck(cudaGraphLaunch(exec,stream)); ck(cudaEventRecord(end,stream));
                ck(cudaEventSynchronize(end)); float ms=0; ck(cudaEventElapsedTime(&ms,begin,end));
                std::printf("{\"kind\":\"timing\",\"buffers\":%d,\"tokens\":%d,\"round\":%d,\"calls\":%d,\"us_per_call\":%.6f}\n",buffers,tokens,r,calls,ms*1000/calls);
            }
            ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
        }
        ck(cudaFree(states));
    }
    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end)); ck(cudaStreamDestroy(stream));
    for(void* ptr : {static_cast<void*>(dh),static_cast<void*>(dg),static_cast<void*>(db),static_cast<void*>(dz),static_cast<void*>(dgm),static_cast<void*>(dy),static_cast<void*>(keep),xq}) ck(cudaFree(ptr));
    std::puts("{\"kind\":\"status\",\"completed\":true}");
}