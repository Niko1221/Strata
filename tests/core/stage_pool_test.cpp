// The stage pool (include/strata/core/stage_pool.hpp): the per-stage copies run at the same time, one worker per
// stage.  Header-only, CPU.
#include "strata/core/stage_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    using strata::core::StagePool;
    using clock = std::chrono::steady_clock;
    // every stage's body runs, and they run at the same time
    {
        StagePool pool(4);
        check(pool.size() == 4, "the pool has one worker per stage");
        std::atomic<int> running{0}, peak{0};
        std::vector<std::function<bool(std::string&)>> bodies(4);
        for (size_t k = 0; k < bodies.size(); ++k)
            bodies[k] = [&](std::string&) {
                const int now = running.fetch_add(1) + 1;
                int old = peak.load();
                while (now > old && !peak.compare_exchange_weak(old, now)) {}
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                running.fetch_sub(1);
                return true;
            };
        const auto t0 = clock::now();
        std::string error;
        check(pool.run(bodies, error), "every stage succeeded");
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
        check(peak.load() == 4, "the four stages ran at the same time");
        check(ms < 600, "the stages did not queue (they cost the slowest one, not the sum)");
    }
    // a stage that fails: its error is reported, the others still ran
    {
        StagePool pool(3);
        std::atomic<int> ran{0};
        std::vector<std::function<bool(std::string&)>> bodies(3);
        for (size_t k = 0; k < bodies.size(); ++k)
            bodies[k] = [&, k](std::string& e) {
                ran.fetch_add(1);
                if (k == 1) { e = "stage 1 failed"; return false; }
                return true;
            };
        std::string error;
        check(!pool.run(bodies, error), "the failing stage is reported");
        check(error == "stage 1 failed", "the failing stage's error is the one reported");
        check(ran.load() == 3, "the other stages ran anyway");
    }
    // the pool is reusable: a second round runs the bodies again
    {
        StagePool pool(2);
        std::atomic<int> ran{0};
        std::vector<std::function<bool(std::string&)>> bodies(2);
        for (size_t k = 0; k < bodies.size(); ++k)
            bodies[k] = [&](std::string&) { ran.fetch_add(1); return true; };
        std::string error;
        check(pool.run(bodies, error), "the first round succeeded");
        check(pool.run(bodies, error), "the second round succeeded");
        check(ran.load() == 4, "both rounds ran every stage");
        // a body that reports no error but fails is an error too
        bodies[0] = [](std::string& e) { e.clear(); return false; };
        check(!pool.run(bodies, error), "a failure without a message is still a failure");
        check(!error.empty(), "a failure without a message gets one");
    }
    // one stage: the pool is the serial path with one worker
    {
        StagePool pool(1);
        std::vector<std::function<bool(std::string&)>> bodies(1);
        bodies[0] = [](std::string&) { return true; };
        std::string error;
        check(pool.run(bodies, error), "a single stage succeeded");
    }
    if (failures == 0) std::fprintf(stderr, "stage_pool_test: ok\n");
    return failures == 0 ? 0 : 1;
}
