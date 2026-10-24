// SPDX-License-Identifier: MIT
// Original fixture contributed to this repository; no external code copied.
// Finite external CPU work for a controlled desktop fairness experiment.
// Independent unsigned dependency chains prevent compile-time folding and do
// not allocate model-sized RAM. This is a CPU workload, not a GPU simulation.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const std::filesystem::path gate(argv[1]), output(argv[2]);
    const uint64_t iterations = std::stoull(argv[3]);
    const unsigned workers = std::stoul(argv[4]);
    if (!iterations || iterations > 16000000000ull || !workers || workers > 16) return 2;
    std::atomic<bool> begin{false}, cancel{false};
    std::vector<uint64_t> results(workers);
    std::vector<std::thread> pool;
    for (unsigned worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
            while (!begin.load(std::memory_order_acquire)) {
                if (cancel.load(std::memory_order_relaxed)) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            uint64_t x = 0x9e3779b97f4a7c15ull + worker;
            for (uint64_t n = 0; n < iterations; ++n) {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                x *= 0x2545f4914f6cdd1dull;
                x += n ^ (x >> 33);
            }
            results[worker] = x;
        });
    }
    std::ofstream(output.string() + ".ready") << "ready\n";
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::filesystem::exists(gate)) {
        if (std::chrono::steady_clock::now() > deadline) {
            cancel = true;
            for (auto& t : pool) t.join();
            return 3;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const auto start = std::chrono::steady_clock::now();
    const double started = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    begin.store(true, std::memory_order_release);
    for (auto& t : pool) t.join();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream out(output);
    out.precision(17);
    out << "{\"start_unix\":" << started << ",\"wall_s\":" << wall
        << ",\"workers\":" << workers << ",\"iterations\":" << iterations << ",\"checksums\":[";
    for (unsigned i = 0; i < workers; ++i) {
        if (i) out << ',';
        out << '"' << results[i] << '"';
    }
    out << "]}\n";
    return out.good() ? 0 : 4;
}
