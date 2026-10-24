#include "strata/core/stage_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <new>
#include <stdexcept>
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
}

int main() {
    using strata::core::StagePool;
    {
        StagePool pool(4);
        check(pool.size() == 4, "one worker per stage");
        std::atomic<int> running{0}, peak{0};
        std::vector<std::function<bool(std::string&)>> bodies(4);
        for (size_t k = 0; k < bodies.size(); ++k)
            bodies[k] = [&](std::string&) {
                const int now = running.fetch_add(1) + 1;
                int old = peak.load();
                while (now > old && !peak.compare_exchange_weak(old, now)) {}
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                running.fetch_sub(1);
                return true;
            };
        std::string error;
        std::vector<double> timings;
        check(pool.run(bodies, error, &timings), "all stages succeed");
        check(peak.load() == 4, "all stages execute concurrently");
        check(timings.size() == 4, "every stage reports a duration");
        check(std::all_of(timings.begin(), timings.end(), [](double ms) { return ms >= 40; }),
              "stage durations cover the body, not only the join");
    }
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
        check(!pool.run(bodies, error), "stage failure is returned");
        check(error == "stage 1 failed", "stage error is preserved");
        check(ran.load() == 3, "failure does not cancel other stages");
        bodies[1] = [&](std::string&) { ran.fetch_add(1); return true; };
        check(pool.run(bodies, error), "pool is reusable");
        check(ran.load() == 6, "second run executes every stage");

        bodies[0] = [](std::string&) { return false; };
        check(!pool.run(bodies, error), "failure without a message is returned");
        check(!error.empty(), "failure without a message gets a default");

        bodies[0] = [](std::string&) -> bool { throw std::runtime_error("worker exception"); };
        try {
            (void) pool.run(bodies, error);
            check(false, "worker exception is rethrown");
        } catch (const std::runtime_error& e) {
            check(std::string(e.what()).find("worker exception") != std::string::npos,
                  "exception type and message are preserved");
        }
        bodies[0] = [](std::string&) -> bool { throw std::bad_alloc(); };
        try {
            (void) pool.run(bodies, error);
            check(false, "bad_alloc is rethrown");
        } catch (const std::bad_alloc&) {
        }

        bodies.resize(2);
        check(!pool.run(bodies, error), "wrong body count is rejected");
        check(error.find("wrong number") != std::string::npos, "wrong body count explains the error");
    }
    {
        StagePool pool(1);
        std::weak_ptr<int> weak;
        {
            auto marker = std::make_shared<int>(1);
            weak = marker;
            std::vector<std::function<bool(std::string&)>> bodies(1);
            bodies[0] = [marker](std::string&) { return true; };
            std::string error;
            check(pool.run(bodies, error), "resource-owning body succeeds");
        }
        check(weak.expired(), "run returns only after the worker destroys its callable");
    }
    if (failures == 0) std::fprintf(stderr, "stage_pool_test: ok\n");
    return failures == 0 ? 0 : 1;
}
