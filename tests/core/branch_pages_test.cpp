// Real CUDA allocation/ownership test. Missing GPU support is a failure, never a skip.
#include "strata/core/branch_pages.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using strata::core::BranchPages;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); ++failures; } } while (0)

int main() {
    if (cudaFree(nullptr)!=cudaSuccess || !strata::core::vmm_available()) {
        std::fprintf(stderr,"BLOCKED: actual CUDA VMM device required\n");
        return 2;
    }
    const auto g = strata::core::vmm_granularity();
    const auto baseline = BranchPages::counters();
    {
        BranchPages original,a,b;
        CHECK(original.reserve(4*g));
        CHECK(original.grow(3*g));
        std::vector<unsigned char> input(3*g),output(3*g);
        for (size_t i=0;i<input.size();++i) input[i]=(i*71+13)%251;
        CHECK(cudaMemcpy(original.data(),input.data(),input.size(),cudaMemcpyHostToDevice)==cudaSuccess);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        auto root = original.freeze();
        CHECK(root!=nullptr);
        CHECK(a.restore(root));
        CHECK(b.restore(root));
        const auto parent = original.ownership();
        const auto sibling = b.ownership();
        CHECK(parent.size()==3 && sibling.size()==3);
        for (size_t i=0;i<parent.size();++i) {
            CHECK(parent[i].allocation_id==sibling[i].allocation_id);
            CHECK(parent[i].driver_handle==sibling[i].driver_handle);
            CHECK(parent[i].owners>=3);
        }
        CHECK(BranchPages::counters().live_bytes-baseline.live_bytes==3*g);
        const auto before = BranchPages::counters();
        CHECK(a.prepare_write(g-1,2));
        CHECK(cudaMemset(a.data()+g-1,252,2)==cudaSuccess);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        CHECK(BranchPages::counters().cow_pages-before.cow_pages==2);
        CHECK(BranchPages::counters().copied_bytes-before.copied_bytes==2*g);
        CHECK(cudaMemcpy(output.data(),b.data(),output.size(),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(output==input);
        CHECK(cudaMemcpy(output.data(),a.data(),output.size(),cudaMemcpyDeviceToHost)==cudaSuccess);
        auto expected = input;
        expected[g-1]=expected[g]=252;
        CHECK(output==expected);
        auto saved_a = a.freeze();
        CHECK(b.prepare_write(2*g+7,1));
        CHECK(cudaMemset(b.data()+2*g+7,253,1)==cudaSuccess);
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        const auto address=a.data();
        const auto reservations=BranchPages::counters().address_reservations;
        CHECK(a.restore(saved_a));
        CHECK(a.data()==address);
        CHECK(BranchPages::counters().address_reservations==reservations);
        CHECK(cudaMemcpy(output.data(),a.data(),output.size(),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(output==expected);
        CHECK(original.restore(root));
        CHECK(cudaMemcpy(output.data(),original.data(),output.size(),cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(output==input);
        CHECK(!a.prepare_write(4*g,1));
        CHECK(!a.prepare_write(UINT64_MAX,2));
        CHECK(BranchPages::counters().live_bytes-baseline.live_bytes==6*g);
    }
    const auto after = BranchPages::counters();
    CHECK(after.live_bytes==baseline.live_bytes);
    CHECK(after.allocations-baseline.allocations==after.releases-baseline.releases);
    CHECK(after.mappings-baseline.mappings==after.unmappings-baseline.unmappings);
    CHECK(cudaGetLastError()==cudaSuccess);
    std::printf("{\"status\":\"%s\",\"granularity\":%llu,\"allocations\":%llu,\"releases\":%llu,\"cow_pages\":%llu,\"copied_bytes\":%llu,\"live_bytes\":%llu,\"failures\":%d}\n",
        failures?"FAIL":"PASS",(unsigned long long)g,(unsigned long long)(after.allocations-baseline.allocations),
        (unsigned long long)(after.releases-baseline.releases),(unsigned long long)(after.cow_pages-baseline.cow_pages),
        (unsigned long long)(after.copied_bytes-baseline.copied_bytes),(unsigned long long)after.live_bytes,failures);
    return failures?1:0;
}
