#include "strata/wait_timeout_registry.hpp"

#include <array>
#include <cassert>
#include <thread>
#include <vector>

int main() {
    strata::WaitTimeoutRegistry registry;
    constexpr size_t owners = 8;
    std::array<std::array<uint32_t, strata::kWaitKinds>, owners> flags{}, counters{};
    std::vector<std::thread> threads;
    for (size_t owner = 0; owner < owners; ++owner) {
        threads.emplace_back([&, owner] {
            for (int round = 0; round < 1000; ++round) {
                for (int kind = 1; kind < strata::kWaitKinds; ++kind)
                    registry.add(&flags[owner][kind], &counters[owner][kind]);
                for (int kind = 1; kind < strata::kWaitKinds; ++kind) {
                    auto* counter = registry.counter(&flags[owner][kind]);
                    assert(counter == &counters[owner][kind]);
                    ++*counter;
                }
                for (int kind = 1; kind < strata::kWaitKinds; ++kind)
                    registry.remove(&flags[owner][kind]);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    for (const auto& counts : counters)
        for (int kind = 1; kind < strata::kWaitKinds; ++kind) assert(counts[kind] == 1000);

    bool rejected = false;
    try { registry.counter(&flags[0][1]); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { registry.add(&flags[0][1], nullptr); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    registry.add(&flags[0][1], &counters[0][1]);
    rejected = false;
    try { registry.add(&flags[0][1], &counters[1][1]); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(registry.counter(&flags[0][1]) == &counters[0][1]);
}
