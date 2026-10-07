// include/strata/glm/pool.hpp - a persistent worker pool with one operation: parallel_for over a range.
//
// The GLM forward pass is a few hundred short data-parallel loops per token (every projection, every attention
// head group, every routed expert), so the cost that matters is the dispatch, not the scheduling: workers spin
// briefly on an epoch counter before they sleep, and a loop hands out fixed-size chunks through one fetch_add.
// The caller's thread takes chunks too, so a pool of N threads uses N cores, not N + 1.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::glm {

class Pool {
public:
    explicit Pool(int threads) {
        n_ = std::max(1, threads);
        for (int i = 1; i < n_; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            epoch_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    int size() const { return n_; }

    /// Run fn(begin, end) over [0, n) in chunks of `grain`.  Returns when every chunk is done.  Not reentrant: a
    /// loop body must not call parallel_for on the same pool (it runs the nested loop inline instead).
    void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn) {
        if (n <= 0) return;
        grain = std::max<int64_t>(1, grain);
        if (n_ == 1 || n <= grain || thread_inside()) {
            fn(0, n);
            return;
        }
        std::lock_guard<std::mutex> serial(run_m_);
        fn_ = &fn;
        total_ = n;
        grain_ = grain;
        next_.store(0, std::memory_order_relaxed);
        active_.store(n_ - 1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(m_);
            epoch_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        thread_inside() = true;
        drain();
        thread_inside() = false;
        while (active_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
        fn_ = nullptr;
    }

private:
    void drain() {
        for (;;) {
            const int64_t b = next_.fetch_add(grain_, std::memory_order_relaxed);
            if (b >= total_) break;
            (*fn_)(b, std::min(total_, b + grain_));
        }
    }
    void loop() {
        // epoch 0 is the one the pool was built with: a worker that starts after the first loop was posted must still
        // see that loop as new (reading the epoch here would skip it, and the caller would wait for it forever)
        uint64_t seen = 0;
        for (;;) {
            // spin first: back-to-back loops arrive microseconds apart during a token
            auto t0 = std::chrono::steady_clock::now();
            uint64_t e;
            for (;;) {
                e = epoch_.load(std::memory_order_acquire);
                if (e != seen) break;
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(200)) {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_.wait(lk, [&] { return epoch_.load(std::memory_order_acquire) != seen; });
                    e = epoch_.load(std::memory_order_acquire);
                    break;
                }
                std::this_thread::yield();
            }
            seen = e;
            if (quit_) return;
            thread_inside() = true;
            drain();
            thread_inside() = false;
            active_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    static bool& thread_inside() {
        static thread_local bool v = false;
        return v;
    }

    int n_ = 1;
    std::vector<std::thread> workers_;
    std::mutex m_, run_m_;
    std::condition_variable cv_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int64_t> next_{0};
    std::atomic<int> active_{0};
    const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
    int64_t total_ = 0, grain_ = 1;
    bool quit_ = false;
};

}  // namespace strata::glm
