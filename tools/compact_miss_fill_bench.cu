// Component timings only: not a model or an RTX PRO performance claim.
#include "strata/kernels/readonly_miss_cache.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::kernels;
#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"line%d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(2); } } while(0)
#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr,"check failed line%d\n",__LINE__);std::exit(3); } } while(0)

void measure(int n, int misses, size_t bytes, bool mapped) {
    constexpr int replays=32;
    const size_t stride=bytes+16;
    unsigned char *host=nullptr, *alias=nullptr, *weights=nullptr, *target=nullptr;
    unsigned long long *sources=nullptr;
    ReadonlyMissCachePlan *device_plan=nullptr;
    CU(cudaHostAlloc((void**)&host,n*bytes,cudaHostAllocMapped));
    CU(cudaHostGetDevicePointer((void**)&alias,host,0));
    for(size_t i=0;i<n*bytes;++i) host[i]=(unsigned char)(i*37+(i>>8));
    CU(cudaMalloc((void**)&weights,n*bytes));
    CU(cudaMemcpy(weights,host,n*bytes,cudaMemcpyHostToDevice));
    CU(cudaMalloc((void**)&target,n*stride));
    CU(cudaMemset(target,0xcd,n*stride));
    CU(cudaMalloc((void**)&sources,n*sizeof(*sources)));
    CU(cudaMalloc((void**)&device_plan,sizeof(*device_plan)));
    ReadonlyMissCachePlan plan{};
    plan.count=n;
    std::vector<unsigned long long> ptrs(n);
    for(int q=0;q<n;++q) {
        plan.fill[q]=(q*7)%n<misses; // mixed hit/miss positions, including tails
        plan.target[q]=(unsigned long long)(target+q*stride);
        if(plan.fill[q]) {
            plan.fill_group[plan.fill_count++]=q;
            ptrs[q]=(unsigned long long)((mapped?alias:weights)+q*bytes);
        } else {
            ptrs[q]=0; // cache hits must never dereference their sources
            CU(cudaMemcpy(target+q*stride,host+q*bytes,bytes,cudaMemcpyHostToDevice));
        }
    }
    REQUIRE(plan.fill_count==misses);
    CU(cudaMemcpy(sources,ptrs.data(),n*sizeof(*sources),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(device_plan,&plan,sizeof(plan),cudaMemcpyHostToDevice));
    CU(cudaDeviceSynchronize());
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaEvent_t begin,end;
    CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
    CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    fill_readonly_misses(sources,bytes,device_plan,stream);
    CU(cudaStreamEndCapture(stream,&graph));
    CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    CU(cudaGraphLaunch(exec,stream));CU(cudaStreamSynchronize(stream));
    std::vector<unsigned char> got(n*stride);
    CU(cudaMemcpy(got.data(),target,got.size(),cudaMemcpyDeviceToHost));
    for(int q=0;q<n;++q) {
        REQUIRE(std::memcmp(got.data()+q*stride,host+q*bytes,bytes)==0);
        for(size_t i=bytes;i<stride;++i) REQUIRE(got[q*stride+i]==0xcd);
    }
    std::vector<float> times;
    for(int trial=0;trial<3;++trial) {
        CU(cudaEventRecord(begin,stream));
        for(int j=0;j<replays;++j) CU(cudaGraphLaunch(exec,stream));
        CU(cudaEventRecord(end,stream));CU(cudaEventSynchronize(end));
        float ms;CU(cudaEventElapsedTime(&ms,begin,end));times.push_back(ms*1000/replays);
    }
    std::sort(times.begin(),times.end());
    std::printf("RESULT n=%d misses=%d bytes=%zu source=%s min_us=%.3f median_us=%.3f max_us=%.3f exact_bytes=1\n",
                n,misses,bytes,mapped?"mapped_ram":"gpu",times[0],times[1],times[2]);
    CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
    CU(cudaEventDestroy(begin));CU(cudaEventDestroy(end));CU(cudaStreamDestroy(stream));
    CU(cudaFree(device_plan));CU(cudaFree(sources));CU(cudaFree(target));CU(cudaFree(weights));CU(cudaFreeHost(host));
}

int main() {
    for(size_t bytes:{size_t(144),size_t(5222400)})
        for(bool mapped:{false,true})
            for(int n:{16,64})
                for(int misses:{0,1,n/2,n}) measure(n,misses,bytes,mapped);
    std::puts("PASS fill-only byte and tail-guard checks; warmed synthetic graph timings");
}
