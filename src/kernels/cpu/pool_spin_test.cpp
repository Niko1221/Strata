// src/kernels/cpu/pool_spin_test.cpp - issue #4624: what `STRATA_POOL_SPIN_US` does to the park loop, read without a clock.
//
// A parked worker spins for `kSpinBeforeSleep` and then sleeps in the kernel; the knob overrides that budget.  It used
// to override it badly: the pause stage was unconditional and the clock was read once every 1,024 pauses, so a budget
// of 0 still cost 1,024 pauses per park (`spin_hot`, src/kernels/cpu/pool.cpp).  This test settles what the budgets do
// from two readings, neither of which is a threshold on a duration:
//
//   * `strata_pool_pause_count` (include/strata/kernels/cpu/pool.hpp) counts the times a park loop executed its pause
//     stage.  The increment is compiled only when `STRATA_POOL_PAUSE_COUNT` is defined, and cmake/backends/cpu.cmake
//     defines it only for a build with `STRATA_BUILD_TESTS=ON`, so the count is a direct observation of the stage
//     rather than something inferred from CPU time.
//   * the pool's own state report (`park_state_of`, src/kernels/cpu/pool_sleep_diag.hpp) says how many workers the pool
//     counts as inside the park wait and how many `diag` says are asleep in the kernel, which is where a park is
//     supposed to end up.  Those two together say where a worker is: the park loop has exactly one blocking state, the
//     condition-variable wait that stamps a worker as sleeping (`src/kernels/cpu/pool.cpp:1037-1044`), so a worker the
//     pool counts at the park and `diag` does not call asleep is still turning through the loop - with a budget still
//     open, that is the pause stage.  The counted arrival is per worker as well: a worker adds itself once when it gets
//     to the park and takes itself off when it leaves (`:983`, `:1049`), so a count of `workers` means every worker is
//     there, not two workers twice.  (The exception is the MSVC-only `_umwait` branch at `:1016-1025`, a wait the label
//     does not cover: a worker there is still a worker the destructor has to reach, which is what the last phase asks.)
//
// Nothing here needs the expert kernels or a particular instruction set, and the pool is given no work: a parked worker
// with nothing to do is the whole subject.
//
// No phase sleeps for a fixed time and then assumes a state: that is what made this test flaky on a loaded runner,
// because a delay says nothing about whether a worker was scheduled.  Each phase *waits*, with a bound, for a state the
// pool reports about itself, and judges a count and a state at the moment that state was seen.  Every wait is bounded,
// so an implementation that never gets there fails with what it reported instead of hanging, and so does a destructor
// that cannot stop its workers (`shutdown_guard`).  On a loaded machine the cost of a bound that is too tight is a
// wrong answer, so the bounds below are hundreds of times the 20 ms they have to cover, and the phase that has to catch
// a state inside the 20 ms window gets many bounded attempts at it rather than one.
//
// So the test is registered wherever the pool is built, including the CPU configuration CI builds, where the AVX-512
// expert tests and the cost measurement are not.
//
// The decision itself is also checked at compile time, so a build cannot be written that gets it wrong.
#include "strata/kernels/cpu/pool.hpp"
#include "pool_sleep_diag.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

namespace cpu = strata::kernels::cpu;

static_assert(!cpu::ExpertPool::park_warm_stage(std::chrono::microseconds(0)),
              "a zero spin budget must leave the park loop no hot stage at all");
static_assert(cpu::ExpertPool::park_warm_stage(std::chrono::microseconds(1)),
              "a one microsecond budget must keep the hot stage");
static_assert(cpu::ExpertPool::park_warm_stage(cpu::ExpertPool::kSpinBeforeSleep),
              "the built-in 20 ms budget must keep the hot stage");

static void spin_env(const char* v) {
#ifdef _WIN32
    _putenv_s("STRATA_POOL_SPIN_US", v);
#else
    setenv("STRATA_POOL_SPIN_US", v, 1);
#endif
}
static void spin_unset() {   // back to the built-in default: an empty value would parse as 0, and 0 is not the default
#ifdef _WIN32
    _putenv_s("STRATA_POOL_SPIN_US", "");
#else
    unsetenv("STRATA_POOL_SPIN_US");
#endif
}

// Bounds, all of them hundreds of times the thing they bound, and none of them a threshold on measured performance: no
// assertion below compares a measured duration or a measured count against a target.  `kWaitMs` is how long a state the
// pool must reach on its own may take - 250 times the built-in 20 ms budget, and inside the 120 s ctest timeout the
// target carries (cmake/backends/cpu.cmake:504).  `kAttemptWaitMs` bounds one attempt of the phase that has to catch a
// state *inside* that 20 ms window, which a loaded runner can miss; `kAttempts` is how many tries it gets, each on a
// fresh pool, because a worker that has slept stays asleep and parks only once per pool.  `kShutdownMs` bounds the
// destructor of a pool whose workers are asleep or mid-pause.
constexpr int kWaitMs = 5000;
constexpr int kAttemptWaitMs = 500;
constexpr int kCloseWaitMs = 2000;
constexpr int kAttempts = 25;
constexpr int kShutdownMs = 5000;

// Turn `ready()` until it says yes, with the bound handed to it.  Each turn reads the pool's own report and yields, so
// waiting costs a report and a context switch rather than a guess about when the workers were scheduled.
template <class F>
static bool wait_until(int limit_ms, F&& ready) {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        if (ready()) return true;
        const long long waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - start).count();
        if (waited > limit_ms) return false;
        std::this_thread::yield();
    }
}

// Elapsed milliseconds, for the evidence a failure prints - never for a judgement about the pool.
class clock_ms {
  public:
    clock_ms() : start_(std::chrono::steady_clock::now()) {}
    long long operator()() const {
        return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_).count();
    }

  private:
    std::chrono::steady_clock::time_point start_;
};

// A pool that cannot stop its workers has to fail with a reason, not sit until ctest's timeout.  Armed over the
// destructor alone: `arm` before the pool goes away, `disarm` after.  The watching thread touches one flag and nothing
// else, so it can outlive a stuck destructor without itself becoming a problem, and it is test-only by construction.
class shutdown_guard {
  public:
    ~shutdown_guard() { disarm(); }
    void arm(int limit_ms) {
        if (watch_.joinable()) return;   // already armed
        done_.store(false, std::memory_order_release);
        watch_ = std::thread([this, limit_ms] {
            for (int waited = 0; waited < limit_ms && !done_.load(std::memory_order_acquire); waited += 20)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (done_.load(std::memory_order_acquire)) return;
            std::printf("pool_spin_test: destroying a pool with its workers parked or mid-pause did not finish within "
                        "%d ms\n", limit_ms);
            std::fflush(stdout);
            std::abort();
        });
    }
    void disarm() {
        done_.store(true, std::memory_order_release);
        if (watch_.joinable()) watch_.join();
    }

  private:
    std::atomic<bool> done_{true};
    std::thread watch_;
};

// Destroy a pool under the guard, whatever state its workers are in.
static void destroy_pool(std::unique_ptr<cpu::ExpertPool>& p) {
    shutdown_guard guard;
    guard.arm(kShutdownMs);
    p.reset();
}

int main() {
    constexpr int NW = 4;

    // The three ways the knob can ask for no hot stage: zero, a negative, and something that is not a number at all.
    // The parse in `pool.cpp` (`atoll`, clamped at 0, read the way the other knobs in that file are read) puts all
    // three on the same path, and on that path the pause stage must never be reached.
    for (const char* cold : {"0", "-5", "not-a-number"}) {
        spin_env(cold);
        cpu::strata_pool_pause_count.store(0, std::memory_order_relaxed);
        std::unique_ptr<cpu::ExpertPool> p{new cpu::ExpertPool(NW)};
        // A worker has to reach the park loop before it can sleep in the kernel, and when that happens is not something
        // a delay can promise: wait for the report that says all four are asleep.
        const bool reached = wait_until(kWaitMs, [&] { return cpu::park_state_of(*p, NW).asleep == p->workers(); });
        const cpu::park_state st = cpu::park_state_of(*p, NW);
        const unsigned long long pauses = cpu::strata_pool_pause_count.load(std::memory_order_relaxed);
        if (!reached || st.asleep != p->workers() || pauses != 0) {
            std::printf("pool_spin_test: at STRATA_POOL_SPIN_US=%s the park loop executed its pause stage %llu times "
                        "and %d of %d workers reached the kernel sleep (%d of %d reported at the park) within %d ms; "
                        "expected no pauses at all and every worker asleep\n",
                        cold, pauses, st.asleep, p->workers(), st.at_park, p->workers(), kWaitMs);
            p->diag(stdout);
            destroy_pool(p);   // a failure path leaves no unguarded destructor behind, either
            return 1;
        }
        std::printf("pool_spin_test: STRATA_POOL_SPIN_US=%s: %llu pauses, %d of %d workers asleep\n",
                    cold, pauses, st.asleep, p->workers());
        destroy_pool(p);   // the pool goes away while every worker is asleep: shutdown out of a park, #29's other half
    }

    // A non-zero budget keeps the stage.  One microsecond is far short of the wait above, so by the time the state is
    // read the workers have spun and slept: the pause stage ran, and the park still ended in the kernel.
    spin_env("1");
    {
        cpu::strata_pool_pause_count.store(0, std::memory_order_relaxed);
        std::unique_ptr<cpu::ExpertPool> p{new cpu::ExpertPool(NW)};
        const bool reached = wait_until(kWaitMs, [&] { return cpu::park_state_of(*p, NW).asleep == p->workers(); });
        const cpu::park_state st = cpu::park_state_of(*p, NW);
        const unsigned long long pauses = cpu::strata_pool_pause_count.load(std::memory_order_relaxed);
        if (!reached || st.asleep != p->workers() || pauses == 0) {
            std::printf("pool_spin_test: at STRATA_POOL_SPIN_US=1 the park loop executed its pause stage %llu times and "
                        "%d of %d workers reached the kernel sleep within %d ms; expected the stage to have run and "
                        "every worker to have slept\n", pauses, st.asleep, p->workers(), kWaitMs);
            p->diag(stdout);
            destroy_pool(p);
            return 1;
        }
        std::printf("pool_spin_test: STRATA_POOL_SPIN_US=1: %llu pauses, %d of %d workers asleep\n",
                    pauses, st.asleep, p->workers());
        destroy_pool(p);
    }

    // The built-in default, both halves of it: the pause stage runs while the window is open and none of the workers has
    // slept yet, and later all of them have slept.  This is the behaviour the fix must not move.
    //
    // The half that used to be a `sleep_for(5ms)` and a reading is here a *caught* state: one report of the pool that
    // says all four workers at the park, none of them asleep, and the pause stage already run.  One report is one
    // snapshot, so "none asleep while the window is open" is the pool's own statement at the moment the stage was seen,
    // not the test's guess about where 5 ms of sleep had landed.  A runner loaded by other work can miss that state -
    // the window is 20 ms and the test process may not be scheduled inside it - so the phase retries on a fresh pool up
    // to `kAttempts` times and fails only if no attempt ever saw it.  That failure is the right one: a pool that sends a
    // worker to the kernel before its budget closes can never be caught in that state, however many tries it gets.
    spin_unset();
    {
        struct sample {
            long long t;
            unsigned long long pauses;
            int at_park, asleep;
        };
        sample caught{0, 0, -1, -1}, ended{0, 0, -1, -1}, miss{0, 0, -1, -1};   // the snapshot, and the end of that pool
        int attempts = 0;
        bool all_asleep = false;
        for (; attempts < kAttempts; ++attempts) {
            cpu::strata_pool_pause_count.store(0, std::memory_order_relaxed);
            const clock_ms started;
            std::unique_ptr<cpu::ExpertPool> p{new cpu::ExpertPool(NW)};
            // One report at a turn: the catch is the pool saying all four at the park, none asleep, stage already run.
            const bool open = wait_until(kAttemptWaitMs, [&] {
                const cpu::park_state st = cpu::park_state_of(*p, NW);
                miss = {started(), cpu::strata_pool_pause_count.load(std::memory_order_acquire), st.at_park, st.asleep};
                return miss.at_park == NW && miss.asleep == 0 && miss.pauses > 0;
            });
            if (!open) {   // this attempt never got inside the window with the workers still spinning: try a new pool
                destroy_pool(p);
                continue;
            }
            caught = miss;
            // Same pool, same attempt: the window has to close and every worker has to end up in the kernel sleep.
            const bool closed = wait_until(kCloseWaitMs, [&] { return cpu::park_state_of(*p, NW).asleep == NW; });
            const cpu::park_state st = cpu::park_state_of(*p, NW);
            ended = {started(), cpu::strata_pool_pause_count.load(std::memory_order_acquire), st.at_park, st.asleep};
            destroy_pool(p);
            if (closed && ended.asleep == NW) { all_asleep = true; break; }   // both halves seen on one pool
        }
        if (caught.pauses == 0) {
            std::printf("pool_spin_test: in %d attempts to catch the built-in %lld ms budget with every worker at the "
                        "park and none of them asleep, the pool never reported that state; the last attempt read %d of "
                        "%d at the park, %d asleep, %llu pauses after %lld ms\n",
                        attempts, (long long) cpu::ExpertPool::kSpinBeforeSleep.count(), miss.at_park, NW, miss.asleep,
                        miss.pauses, miss.t);
            return 1;
        }
        if (!all_asleep) {
            std::printf("pool_spin_test: %d attempts caught the built-in %lld ms budget with the workers spinning "
                        "(%llu pauses, %d of %d at the park, none asleep) but never saw all of them reach the kernel "
                        "sleep within %d ms of that; the last attempt read %d of %d at the park, %d asleep, %llu pauses "
                        "after %lld ms\n",
                        attempts + 1, (long long) cpu::ExpertPool::kSpinBeforeSleep.count(), caught.pauses, NW, NW,
                        kCloseWaitMs, ended.at_park, NW, ended.asleep, ended.pauses, ended.t);
            return 1;
        }
        if (ended.pauses < caught.pauses) {
            std::printf("pool_spin_test: the pause stage went backwards: %llu pauses when the stage was caught with all "
                        "%d workers at the park and none asleep, %llu when every worker had slept\n", caught.pauses, NW,
                        ended.pauses);
            return 1;
        }
        std::printf("pool_spin_test: default %lld ms, attempt %d of %d: all %d workers at the park, none asleep, %llu "
                    "pauses at the caught report; %llu pauses and %d of %d workers asleep %lld ms into that pool's "
                    "life\n", (long long) cpu::ExpertPool::kSpinBeforeSleep.count(), attempts + 1, kAttempts, NW,
                    caught.pauses, ended.pauses, ended.asleep, NW, ended.t);
    }

    // A pool destroyed while its workers are inside a spin budget must not hang: the stop flag and the epoch bump have
    // to reach a worker that is mid-pause, not only one that has slept.  The wait below reads the pool's own count of
    // workers at the park, none of them asleep, together with a pause stage that has run - on one report, so it is the
    // state the destructor is going to find them in.  The guard is the bound on the destruction itself.
    spin_env("250000");
    {
        cpu::strata_pool_pause_count.store(0, std::memory_order_relaxed);   // so the number printed below is this phase's
        std::unique_ptr<cpu::ExpertPool> p{new cpu::ExpertPool(NW)};
        const bool in_spin = wait_until(kWaitMs, [&] {
            const cpu::park_state st = cpu::park_state_of(*p, NW);
            return st.at_park == p->workers() && st.asleep == 0 &&
                   cpu::strata_pool_pause_count.load(std::memory_order_relaxed) > 0;
        });
        const cpu::park_state st = cpu::park_state_of(*p, NW);
        const unsigned long long pauses = cpu::strata_pool_pause_count.load(std::memory_order_relaxed);
        if (!in_spin) {
            std::printf("pool_spin_test: 250 ms into a park the pool reported %d of %d workers at the park, %d asleep, "
                        "%llu pauses; expected every worker inside the pause stage and none asleep\n",
                        st.at_park, p->workers(), st.asleep, pauses);
            p->diag(stdout);
            destroy_pool(p);
            return 1;
        }
        std::printf("pool_spin_test: 250 ms budget: %d of %d workers at the park, none asleep, %llu pauses - destroying "
                    "the pool with every worker mid-pause\n", st.at_park, p->workers(), pauses);
        std::fflush(stdout);
        destroy_pool(p);
    }
    spin_unset();
    std::printf("pool_spin_test OK\n");
    return 0;
}
