// Damage a real retained physical page through another CUDA mapping. A failed
// restore must preserve the current valid lease; no simulated device is used.
#include "strata/core/branch_pages.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace strata::core;
static int failures = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); ++failures; } } while (0)

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !vmm_available()) return 2;
    const auto granularity = vmm_granularity();
    const auto baseline = BranchPages::counters();
    {
        BranchPages original, active;
        CHECK(original.reserve(granularity));
        CHECK(original.grow(granularity));
        CHECK(cudaMemset(original.data(), 17, granularity) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        auto root = original.freeze();
        CHECK(root != nullptr);
        CHECK(active.restore(root));
        CHECK(active.prepare_write(0, 1));
        CHECK(cudaMemset(active.data(), 29, 1) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        auto child = active.freeze();
        CHECK(child != nullptr);
        if (failures) return 1;
        const auto address = active.data();
        const auto child_id = active.ownership()[0].allocation_id;
        VmmMapping corruptor;
        CHECK(corruptor.reserve(granularity));
        CHECK(corruptor.map(0, BranchPages::ownership(root)[0].driver_handle, true));
        CHECK(cudaMemset(corruptor.base(), 18, 1) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(corruptor.release());
        std::vector<uint8_t> damaged;
        CHECK(BranchPages::read_page(root, 0, damaged));
        CHECK(damaged.size() == granularity && damaged[0] == 18);
        CHECK(!active.restore(root));
        CHECK(active.data() == address);
        CHECK(active.ownership()[0].allocation_id == child_id);
        std::vector<uint8_t> unchanged(granularity);
        CHECK(cudaMemcpy(unchanged.data(), active.data(), unchanged.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        auto expected = std::vector<uint8_t>(granularity, 17);
        expected[0] = 29;
        CHECK(unchanged == expected);
        CHECK(active.restore(child));
        CHECK(active.prepare_write(1, 1));
        CHECK(cudaMemset(active.data() + 1, 31, 1) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(BranchPages::read_page(child, 0, unchanged));
        CHECK(unchanged == expected);
    }
    const auto after = BranchPages::counters();
    CHECK(after.live_bytes == baseline.live_bytes);
    CHECK(after.allocations - baseline.allocations == after.releases - baseline.releases);
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"granularity\":%llu}\n",
                failures ? "FAIL" : "PASS", checks, failures, (unsigned long long)granularity);
    return failures ? 1 : 0;
}
