// Explicitly scheduled test: exhausts the real GPU and must never share it with
// another test/job. CMake builds this binary without adding a parallel CTest.
#include "branch_pressure.hpp"
#include "strata/core/branch_pages.hpp"
#include <cstdio>
#include <cstring>

using namespace strata::core;
static int failures = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); ++failures; } } while (0)

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !vmm_available()) {
        std::fprintf(stderr, "BLOCKED: actual CUDA VMM device required\n");
        return 2;
    }
    const auto granularity = vmm_granularity();
    const auto baseline = BranchPages::counters();
    BranchPressure pressure;
    {
        BranchPages original, a, b;
        CHECK(original.reserve(granularity));
        CHECK(original.grow(granularity));
        std::vector<uint8_t> input(granularity), output(granularity);
        for (size_t i = 0; i < input.size(); ++i) input[i] = (i * 71 + 13) % 251;
        CHECK(cudaMemcpy(original.data(), input.data(), input.size(), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        auto root = original.freeze();
        CHECK(root != nullptr);
        CHECK(a.restore(root));
        CHECK(b.restore(root));
        if (failures) return 1;
        const auto address = a.data();
        const auto before_pages = BranchPages::counters();
        CHECK(pressure.fill());
        const auto exhausted = vmm_failure_counters();
        CHECK(std::strcmp(exhausted.last_operation, "cuMemCreate") == 0);
        CHECK(!a.prepare_write(0, 1));
        const auto refused = vmm_failure_counters();
        CHECK(refused.out_of_memory_failures > exhausted.out_of_memory_failures);
        CHECK(refused.driver_failures > exhausted.driver_failures);
        CHECK(refused.last_error == CUDA_ERROR_OUT_OF_MEMORY);
        CHECK(a.data() == address);
        CHECK(BranchPages::counters().cow_pages == before_pages.cow_pages);
        CHECK(pressure.release());
        CHECK(pressure.allocated_bytes == pressure.released_bytes);
        CHECK(cudaMemcpy(output.data(), b.data(), output.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(output == input);
        CHECK(a.restore(root));
        CHECK(a.data() == address);
        CHECK(a.prepare_write(0, 1));
        CHECK(cudaMemset(a.data(), 252, 1) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(cudaMemcpy(output.data(), a.data(), output.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        auto expected = input;
        expected[0] = 252;
        CHECK(output == expected);
        CHECK(cudaMemcpy(output.data(), b.data(), output.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(output == input);
        CHECK(BranchPages::counters().cow_pages == before_pages.cow_pages + 1);
    }
    const auto after = BranchPages::counters();
    CHECK(after.live_bytes == baseline.live_bytes);
    CHECK(after.allocations - baseline.allocations == after.releases - baseline.releases);
    CHECK(after.mappings - baseline.mappings == after.unmappings - baseline.unmappings);
    const auto errors = vmm_failure_counters();
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"granularity\":%llu,"
                "\"held_bytes\":%llu,\"released_bytes\":%llu,\"driver_failures\":%llu,"
                "\"out_of_memory_failures\":%llu,\"last_error\":%d,\"last_operation\":\"%s\"}\n",
                failures ? "FAIL" : "PASS", checks, failures,
                (unsigned long long)granularity, (unsigned long long)pressure.allocated_bytes,
                (unsigned long long)pressure.released_bytes, (unsigned long long)errors.driver_failures,
                (unsigned long long)errors.out_of_memory_failures, errors.last_error, errors.last_operation);
    return failures ? 1 : 0;
}
