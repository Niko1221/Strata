// A drained foreground-yield interval must sleep the workers, not merely the host.
// Force a five-second normal spin in this process so observing sleep within one
// second exercises the scoped override rather than the normal 20-ms fallback.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
int main() {
    if (!cpu::cpu_features().usable()) { std::puts("pool_background_test: SKIP (CPU instruction set)"); return 0; }
#if defined(_WIN32)
    _putenv_s("STRATA_POOL_SPIN_US", "5000000");
#else
    setenv("STRATA_POOL_SPIN_US", "5000000", 1);
#endif
    try {
        cpu::ExpertPool pool(2, false);
        auto check = [](bool ok, const char* why) { if (!ok) throw std::runtime_error(why); };
        auto sleeping = [&] {
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (pool.sleeping_workers() != pool.workers() && std::chrono::steady_clock::now() < end)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return pool.sleeping_workers() == pool.workers();
        };
        check(!pool.background_idle(), "default background override must be disabled");
        std::vector<uint8_t> blob(cpu::BLOB, 0);
        std::vector<float> x(cpu::H, 0.25f), out(cpu::H);
        cpu::ActQ act;
        cpu::act_quant_q8_1(x.data(), cpu::H, act);
        cpu::ExpertJob job{};
        job.blob = blob.data(); job.act = &act; job.out = out.data(); job.slot = 0;
        for (int turn = 0; turn < 40; ++turn) {
            {
                cpu::ExpertPool::BackgroundIdleScope idle(pool);
                check(pool.background_idle() && sleeping(), "foreground wait must actually sleep every worker");
                { cpu::ExpertPool::BackgroundIdleScope nested(pool); }
                check(pool.background_idle(), "nested scope must preserve the outer wait");
            }
            check(!pool.background_idle(), "completed or cancelled wait restores the normal policy");
            const float sentinel = -1.2345e33f;
            std::fill(out.begin(), out.end(), sentinel);
            if (turn & 1) pool.run(&job, 1); else pool.run_split(&job, 1);
            check(out.front() != sentinel && out.back() != sentinel, "next published work must wake sleepers and complete");
        }
        try {
            cpu::ExpertPool::BackgroundIdleScope idle(pool);
            check(sleeping(), "cancellation fixture must begin with sleeping workers");
            throw 1; // cancellation/error unwinds the same scope used around native waits
        } catch (int) {}
        check(!pool.background_idle(), "exception unwinding restores the default idle policy");
        pool.run(&job, 1);
        std::puts("pool_background_test: 40 sleep/wake batches and scoped restoration passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pool_background_test: %s\n", e.what()); return 1;
    }
}
