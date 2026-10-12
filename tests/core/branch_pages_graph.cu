// A captured kernel must follow the new physical mapping at the same address.
#include "strata/core/branch_pages.hpp"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void captured_write(unsigned char* p) { if (threadIdx.x==0) *p=71; }
int main() {
    using namespace strata::core;
    if (cudaFree(nullptr)!=cudaSuccess || !vmm_available()) return 2;
    const auto g=vmm_granularity();
    const auto before=BranchPages::counters();
    int failures=0;
    {
        BranchPages pages;
        if (!pages.reserve(2*g) || !pages.grow(2*g)) return 3;
        cudaStream_t stream{};
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        if (cudaStreamCreate(&stream)!=cudaSuccess ||
            cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)!=cudaSuccess) return 4;
        captured_write<<<1,1,0,stream>>>(pages.data()+g-1);
        if (cudaStreamEndCapture(stream,&graph)!=cudaSuccess ||
            cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0)!=cudaSuccess) return 5;
        auto root=pages.freeze();
        if (!root) return 6;
        const auto address=pages.data();
        for (int i=0;i<8;++i) {
            if (!pages.restore(root) || pages.data()!=address || !pages.prepare_write(g-1,1)) return 7;
            if (cudaGraphLaunch(executable,stream)!=cudaSuccess || cudaStreamSynchronize(stream)!=cudaSuccess) return 8;
            unsigned char value=0;
            if (cudaMemcpy(&value,pages.data()+g-1,1,cudaMemcpyDeviceToHost)!=cudaSuccess || value!=71) ++failures;
            if (!pages.restore(root) || pages.data()!=address) return 9;
            if (cudaMemcpy(&value,pages.data()+g-1,1,cudaMemcpyDeviceToHost)!=cudaSuccess || value!=0) ++failures;
        }
        if (cudaGraphExecDestroy(executable)!=cudaSuccess || cudaGraphDestroy(graph)!=cudaSuccess ||
            cudaStreamDestroy(stream)!=cudaSuccess) ++failures;
    }
    const auto after=BranchPages::counters();
    if (after.cow_pages-before.cow_pages!=8 || after.copied_bytes-before.copied_bytes!=8*g ||
        after.live_bytes!=before.live_bytes || after.allocations-before.allocations!=after.releases-before.releases) ++failures;
    std::printf("{\"status\":\"%s\",\"graph_replays\":8,\"cow_pages\":%llu,\"copied_bytes\":%llu,\"live_bytes\":%llu,\"failures\":%d}\n",
        failures?"FAIL":"PASS",(unsigned long long)(after.cow_pages-before.cow_pages),
        (unsigned long long)(after.copied_bytes-before.copied_bytes),(unsigned long long)after.live_bytes,failures);
    return failures?1:0;
}
