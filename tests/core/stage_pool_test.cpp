#include "strata/core/stage_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <latch>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
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
    {
        StagePool pool(0);
        std::string error;
        std::vector<double> timings{123.0};
        check(pool.run({}, error, &timings), "zero stages complete without waiting");
        check(timings.empty(), "zero stages clear the timing output");
    }
    {
        StagePool idle(2); // destruction also joins workers that never received a body
    }
    {
        StagePool pool(2);
        std::vector<std::function<bool(std::string&)>> bodies(2);
        std::string error;
        check(pool.run(bodies, error), "empty stage bodies complete successfully");
        bodies[0] = [](std::string&) -> bool { throw std::runtime_error("recoverable"); };
        bool caught = false;
        try {
            (void) pool.run(bodies, error);
        } catch (const std::runtime_error&) {
            caught = true;
        }
        check(caught, "worker exception reaches the caller before reuse");
        int calls = 0;
        bodies[0] = [&](std::string&) { ++calls; return true; };
        for (int k = 0; k < 3; ++k)
            check(pool.run(bodies, error), "pool remains reusable after an exception");
        check(calls == 3, "each post-exception run executes its own body once");
    }
    {
        StagePool pool(2);
        std::latch start(3);
        std::mutex counters;
        int active = 0, owner = 0, calls = 0;
        bool mixed_runs = false;
        std::atomic<bool> ok{true};
        auto caller = [&](int tag) {
            std::vector<std::function<bool(std::string&)>> bodies(2);
            for (auto& body : bodies)
                body = [&, tag](std::string&) {
                    {
                        std::lock_guard<std::mutex> lk(counters);
                        if (active && owner != tag) mixed_runs = true;
                        owner = tag;
                        ++active;
                        ++calls;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    {
                        std::lock_guard<std::mutex> lk(counters);
                        --active;
                    }
                    return true;
                };
            start.arrive_and_wait();
            std::string error;
            if (!pool.run(bodies, error)) ok = false;
        };
        std::thread first(caller, 1), second(caller, 2);
        start.arrive_and_wait();
        first.join();
        second.join();
        check(ok.load(), "simultaneous callers both complete");
        check(calls == 4 && active == 0, "simultaneous callers finish every stage once");
        check(!mixed_runs, "different callers cannot mix their stage bodies");
    }
    {
        // Regression (TSan 50-round timeouts): stop_ was set and notified outside every
        // slot's mutex, so a worker that had checked its predicate and was about to sleep
        // could miss the wakeup and sleep forever, hanging stop_and_join's join. Hammer
        // the construct/run/destroy cycle; a lost wakeup shows up as this test hanging
        // past the ctest timeout.
        for (int round = 0; round < 300; ++round) {
            StagePool pool(4);
            std::vector<std::function<bool(std::string&)>> bodies(4);
            for (auto& body : bodies)
                body = [](std::string&) { return true; };
            std::string error;
            check(pool.run(bodies, error), "hammer round completes");
        }
    }
    {
        // Regression: destroying the pool while a run() is in flight must not hang; the
        // destructor waits out the in-flight run (run_mu_) and then stops the workers.
        for (int round = 0; round < 100; ++round) {
            auto pool = std::make_unique<StagePool>(4);
            std::vector<std::function<bool(std::string&)>> bodies(4);
            for (auto& body : bodies)
                body = [](std::string&) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); return true; };
            std::latch done(1);
            std::thread caller([&] {
                std::string error;
                pool->run(bodies, error);
                done.count_down();
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            const auto started = std::chrono::steady_clock::now();
            pool.reset();   // waits out the in-flight run, then stops the workers
            const auto waited = std::chrono::steady_clock::now() - started;
            check(waited < std::chrono::seconds(30), "destruction with an in-flight run completes");
            caller.join();
            check(done.try_wait(), "the in-flight run finished before destruction returned");
        }
    }
    if (failures == 0) std::fprintf(stderr, "stage_pool_test: ok\n");
    return failures == 0 ? 0 : 1;
}
