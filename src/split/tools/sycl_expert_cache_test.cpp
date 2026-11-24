#include "../src/sycl/expert_cache.hpp"
#include "../src/sycl_memory.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>

using namespace fnx::sycl_backend;

static void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
template <typename F> static void refuses(F f) {
    try { f(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid expert working set accepted");
}

int main() try {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    check(devices.size() == 2, "select exactly two GPUs");
    Runtime runtime(devices);
    constexpr size_t ne = 8, span = 64;
    std::array<std::array<unsigned char, span>, ne> gu{}, dn{};
    std::vector<ExpertSource> sources;
    for (size_t i = 0; i < ne; ++i) {
        gu[i].fill((unsigned char) (i + 1));
        dn[i].fill((unsigned char) (i + 101));
        sources.push_back({gu[i].data(), dn[i].data(), i < 4 ? 32u : 64u, 32});
    }
    std::vector<std::pair<size_t, size_t>> sizes{{32,32}, {64,32}};
    const std::vector<fnx::SyclMemoryBudget> budgets{{448,128,0}, {448,128,0}};
    const auto plan = fnx::sycl_expert_placement({sizes, sizes}, std::vector<double>(ne, 0), 4, 2, budgets);
    std::vector<char> resident;
    for (const auto& layer : plan.resident) resident.insert(resident.end(), layer.begin(), layer.end());
    std::vector<std::unique_ptr<ExpertCache>> caches;
    for (size_t c = 0; c < runtime.cards(); ++c)
        caches.push_back(std::make_unique<ExpertCache>(runtime, c, sources, resident, 2));
    for (const auto& picks : {std::vector<size_t>{0,1,4,5}, {2,3,6,7}, {0,3,4,7}, {2,2,5,5}}) {
        for (auto& cache : caches) cache->prepare(picks);
        for (size_t c = 0; c < runtime.cards(); ++c) {
            DeviceBuffer output(runtime, c, picks.size() * sizeof(unsigned));
            auto* values = (unsigned*) output.data();
            auto* ids = sycl::malloc_shared<size_t>(picks.size(), runtime.queue(c));
            if (!ids) throw std::bad_alloc();
            auto free_ids = [&](size_t* p) { sycl::free(p, runtime.context()); };
            std::unique_ptr<size_t, decltype(free_ids)> ids_owner(ids, free_ids);
            std::copy(picks.begin(), picks.end(), ids);
            const char* const* gp = caches[c]->gate_up();
            const char* const* dp = caches[c]->down();
            runtime.queue(c).parallel_for(sycl::range<1>(picks.size()), [=](sycl::id<1> id) {
                const size_t e = ids[id[0]];
                values[id[0]] = (unsigned char) gp[e][0] + (unsigned char) dp[e][31];
            });
            std::vector<unsigned> got(picks.size());
            runtime.queue(c).memcpy(got.data(), values, got.size() * sizeof(unsigned)).wait_and_throw();
            for (size_t j = 0; j < picks.size(); ++j)
                check(got[j] == 2 * picks[j] + 102, "expert eviction/copy/pointer table mismatch");
            check(caches[c]->bytes() == 448, "cache VRAM accounting omits slots or pointer tables");
        }
    }
    refuses([&] { caches[0]->prepare({0,1,2}); });
    refuses([&] { caches[0]->prepare({ne}); });
    caches[0]->prepare({1,3});
    std::puts("SYCL two-card expert cache replacement: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
