#include "strata/core/branch_pages.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    using namespace strata::core;
    if (cudaFree(nullptr)!=cudaSuccess || !vmm_available()) return 2;
    const auto g=vmm_granularity();
    const auto before=BranchPages::counters();
    std::atomic<int> failures{0};
    {
        BranchPages parent;
        if (!parent.reserve(3*g) || !parent.grow(3*g)) return 3;
        if (cudaMemset(parent.data(),17,3*g)!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) return 4;
        auto root=parent.freeze();
        if (!root) return 5;
        std::vector<std::thread> workers;
        for (int worker=0;worker<4;++worker) workers.emplace_back([&,worker] {
            if (cudaSetDevice(0)!=cudaSuccess) { ++failures; return; }
            BranchPages child;
            for (int i=0;i<16;++i) {
                const auto offset=2*g+worker*32+i;
                if (!child.restore(root) || !child.prepare_write(offset,1)) { ++failures; return; }
                const unsigned char value=80+worker;
                unsigned char found=0;
                if (cudaMemcpy(child.data()+offset,&value,1,cudaMemcpyHostToDevice)!=cudaSuccess ||
                    cudaDeviceSynchronize()!=cudaSuccess ||
                    cudaMemcpy(&found,child.data()+offset,1,cudaMemcpyDeviceToHost)!=cudaSuccess || found!=value) ++failures;
                const auto own=child.ownership(), original=parent.ownership();
                if (own[0].allocation_id!=original[0].allocation_id ||
                    own[1].allocation_id!=original[1].allocation_id ||
                    own[2].allocation_id==original[2].allocation_id) ++failures;
            }
        });
        for (auto& worker:workers) worker.join();
        std::vector<unsigned char> bytes(3*g);
        if (cudaMemcpy(bytes.data(),parent.data(),bytes.size(),cudaMemcpyDeviceToHost)!=cudaSuccess) ++failures;
        for (const auto value:bytes) if (value!=17) { ++failures; break; }
    }
    const auto after=BranchPages::counters();
    if (after.cow_pages-before.cow_pages!=64 || after.copied_bytes-before.copied_bytes!=64*g ||
        after.live_bytes!=before.live_bytes || after.allocations-before.allocations!=after.releases-before.releases ||
        after.mappings-before.mappings!=after.unmappings-before.unmappings ||
        after.address_reservations-before.address_reservations!=after.address_releases-before.address_releases) ++failures;
    std::printf("{\"status\":\"%s\",\"workers\":4,\"fork_restore_cycles\":64,\"cow_pages\":%llu,\"copied_bytes\":%llu,\"allocations\":%llu,\"releases\":%llu,\"live_bytes\":%llu,\"failures\":%d}\n",
        failures?"FAIL":"PASS",(unsigned long long)(after.cow_pages-before.cow_pages),
        (unsigned long long)(after.copied_bytes-before.copied_bytes),(unsigned long long)(after.allocations-before.allocations),
        (unsigned long long)(after.releases-before.releases),(unsigned long long)after.live_bytes,failures.load());
    return failures?1:0;
}
