// Deliberately fault an actual GPU write in an isolated test process.
#include "strata/core/branch_pages.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void forbidden_write(unsigned char* p) { if (threadIdx.x==0) p[0]=71; }
int main() {
    using namespace strata::core;
    if (cudaFree(nullptr)!=cudaSuccess || !vmm_available()) return 2;
    BranchPages pages;
    if (!pages.reserve(vmm_granularity()) || !pages.grow(vmm_granularity())) return 3;
    auto snapshot=pages.freeze();
    if (!snapshot || cudaDeviceSynchronize()!=cudaSuccess) return 4;
    forbidden_write<<<1,1>>>(pages.data());
    const auto result=cudaDeviceSynchronize();
    const bool protected_write=result==cudaErrorIllegalAddress;
    std::printf("{\"status\":\"%s\",\"cuda_error\":%d,\"readonly_write_rejected\":%s}\n",
        protected_write?"PASS":"FAIL",(int)result,protected_write?"true":"false");
    std::fflush(stdout);
    // The expected sticky CUDA fault makes this context unusable; let process
    // teardown release it rather than invoking healthy-context destructors.
    std::_Exit(protected_write?0:1);
}
