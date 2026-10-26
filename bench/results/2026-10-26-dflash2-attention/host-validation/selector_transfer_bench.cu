#include "strata/kernels/dflash2.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
template<class T>T* dev(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
__global__ void read_eviction(const unsigned* p,unsigned* sums,size_t n){
    unsigned sum=0;for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)sum^=p[i];
    atomicXor(sums,sum);
}
unsigned rng(unsigned& x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
int main(){try{
    using namespace strata::kernels;
    constexpr int vocab=248320,rank=256,top=16;
    const size_t cells=size_t(vocab)*rank;
    uint16_t *hp,*hn,*mp,*mn;ck(cudaHostAlloc(&hp,cells*2,cudaHostAllocMapped));ck(cudaHostAlloc(&hn,cells*2,cudaHostAllocMapped));
    ck(cudaHostGetDevicePointer(&mp,hp,0));ck(cudaHostGetDevicePointer(&mn,hn,0));
    unsigned random=123456789;
    for(size_t i=0;i<cells;++i){hp[i]=__bfloat16_as_ushort(__float2bfloat16((int(rng(random)%129)-64)*0.0078125f));hn[i]=__bfloat16_as_ushort(__float2bfloat16((int(rng(random)%129)-64)*0.0078125f));}
    auto gp=dev<uint16_t>(cells),gn=dev<uint16_t>(cells);ck(cudaMemcpy(gp,hp,cells*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(gn,hn,cells*2,cudaMemcpyHostToDevice));
    constexpr size_t evn=64*1024*1024;std::vector<unsigned> ev(evn);for(auto& v:ev)v=rng(random);
    auto eviction=dev<unsigned>(evn),sum=dev<unsigned>(1);ck(cudaMemcpy(eviction,ev.data(),evn*4,cudaMemcpyHostToDevice));ck(cudaMemset(sum,0,4));ev.clear();ev.shrink_to_fit();
    auto ids=dev<int32_t>(8*top),anchor=dev<int32_t>(1),out=dev<int32_t>(8);auto confidence=dev<float>(8),unary=dev<float>(8*top),hidden=dev<float>(8*rank);
    const auto stage_bytes=dflash2_select_staging_bytes(8,top,rank);auto stage=dev<unsigned char>(stage_bytes);
    const auto scratch_bytes=dflash2_topk_scratch_bytes(8,vocab,top);auto scratch=dev<unsigned char>(scratch_bytes);
    cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    for(int rows:{2,4,8}){
        cudaGraph_t graph[3];cudaGraphExec_t exec[3];
        for(int mode=0;mode<3;++mode){
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            if(mode==2)dflash2_select_staged(ids,unary,hidden,mp,mn,anchor,out,confidence,rows,top,rank,vocab,stage,stage_bytes,stream,scratch,scratch_bytes);
            else dflash2_select_bf16(ids,unary,hidden,mode?mp:gp,mode?mn:gn,anchor,out,confidence,rows,top,rank,stream,false,scratch,scratch_bytes);
            ck(cudaStreamEndCapture(stream,&graph[mode]));ck(cudaGraphInstantiate(&exec[mode],graph[mode],0));
        }
        // New candidate IDs each frame; each mode sees identical values. Explicit
        // 256 MiB high-entropy reads precede timing, without claiming perfect L2 eviction.
        for(int round=0;round<6;++round)for(int frame=0;frame<50;++frame){
            std::vector<int32_t> hi(rows*top);std::vector<float> hu(rows*top),hh(rows*rank);
            int32_t ha=rng(random)%vocab;
            for(auto& v:hi)v=rng(random)%vocab;
            for(auto& v:hu)v=(int(rng(random)%1025)-512)*0.001953125f;
            for(auto& v:hh)v=(int(rng(random)%129)-64)*0.0078125f;
            ck(cudaMemcpyAsync(ids,hi.data(),hi.size()*4,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(anchor,&ha,4,cudaMemcpyHostToDevice,stream));
            ck(cudaMemcpyAsync(unary,hu.data(),hu.size()*4,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(hidden,hh.data(),hh.size()*4,cudaMemcpyHostToDevice,stream));
            int32_t ref[8];float refp[8];
            ck(cudaGraphLaunch(exec[0],stream));ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(ref,out,(rows-1)*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(refp,confidence,(rows-1)*4,cudaMemcpyDeviceToHost));
            for(int step=0;step<3;++step){
                int mode=round%2?2-step:step;
                read_eviction<<<1024,256,0,stream>>>(eviction,sum,evn);
                ck(cudaEventRecord(a,stream));ck(cudaGraphLaunch(exec[mode],stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));
                float ms;ck(cudaEventElapsedTime(&ms,a,b));int32_t got[8];float p[8];ck(cudaMemcpy(got,out,(rows-1)*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(p,confidence,(rows-1)*4,cudaMemcpyDeviceToHost));
                if(memcmp(got,ref,(rows-1)*4)||memcmp(p,refp,(rows-1)*4))throw std::runtime_error("selector bits differ");
                printf("sample rows=%d round=%d frame=%d mode=%d us=%.6f\n",rows,round,frame,mode,ms*1000);
            }
        }
        for(int i=0;i<3;++i){ck(cudaGraphExecDestroy(exec[i]));ck(cudaGraphDestroy(graph[i]));}
    }
    ck(cudaDeviceSynchronize());puts("PASS: all selector IDs and confidence bits equal");
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
