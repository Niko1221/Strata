#pragma once

// One worker per stage.  A layer split gives every stage its own card, its own sessions and its own PCIe bus, and
// the copies that move a conversation between the sessions and the batch slots (or into the conversation cache)
// are per stage: each stage moves its own share of the K/V on its own device.  Run one after another, the stages
// queue: the copy costs the sum of the stages instead of the slowest one.  Here they run at the same time.
//
// The workers are persistent (one per stage, created once): the snapshot copies ride a private non-blocking CUDA
// stream that is thread-local (conversation_copy.hpp), so a thread per copy would leak a stream per stage per copy.
//
// The stages are independent, so no lock is needed: every call site runs on the engine's serial orchestration
// thread, and a call waits for all of its stages.  A stage that fails does not stop the others (their work is
// discarded with the image); the first failing stage's error is reported, as the serial loop's was.
//
// Header-only, CPU (the workers touch the GPU only in the bodies they are given).

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <memory>
#include <vector>

namespace strata::core {

class StagePool {
public:
    /// One worker per stage.  The stages' bodies are given to run(), one per stage.
    explicit StagePool(size_t stages) {
        for (size_t k = 0; k < stages; ++k) {
            slots_.push_back(std::make_unique<Slot>());
            workers_.emplace_back([this, k] { work(k); });
        }
    }
    ~StagePool() {
        stop_ = true;
        for (auto& s : slots_) s->cv.notify_all();
        for (auto& w : workers_) w.join();
    }

    StagePool(const StagePool&) = delete;
    StagePool& operator=(const StagePool&) = delete;

    size_t size() const { return slots_.size(); }

    /// Run body(k) on the worker of stage k - all of them at the same time - and wait for all of them.
    /// The body reports its failure through its error string (an empty one is a success).
    bool run(const std::vector<std::function<bool(std::string&)>>& bodies, std::string& error) {
        if (bodies.size() != slots_.size()) { error = "stage pool: wrong number of stages"; return false; }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stop_.load()) { error = "stage pool: shut down"; return false; }
        }
        for (size_t k = 0; k < bodies.size(); ++k) {
            std::lock_guard<std::mutex> lk(slots_[k]->mu);
            slots_[k]->body = bodies[k];
            slots_[k]->error.clear();
            slots_[k]->done = false;
            slots_[k]->busy = true;
        }
        for (auto& s : slots_) s->cv.notify_all();
        std::string first;
        for (size_t k = 0; k < bodies.size(); ++k) {
            std::unique_lock<std::mutex> lk(slots_[k]->mu);
            slots_[k]->cv.wait(lk, [&] { return slots_[k]->done; });
            if (first.empty() && !slots_[k]->error.empty()) first = slots_[k]->error;
        }
        if (first.empty()) return true;
        error = std::move(first);
        return false;
    }

private:
    struct Slot {
        std::mutex mu;
        std::condition_variable cv;
        std::function<bool(std::string&)> body;
        std::string error;
        bool busy = false;
        bool done = false;
    };

    void work(size_t k) {
        for (;;) {
            std::function<bool(std::string&)> body;
            {
                std::unique_lock<std::mutex> lk(slots_[k]->mu);
                slots_[k]->cv.wait(lk, [&] { return stop_ || slots_[k]->busy; });
                if (stop_) return;
                body = slots_[k]->body;
                slots_[k]->busy = false;
            }
            std::string error;
            const bool ok = body ? body(error) : true;
            {
                std::lock_guard<std::mutex> lk(slots_[k]->mu);
                slots_[k]->error = ok ? std::string() : (error.empty() ? std::string("stage failed") : error);
                slots_[k]->done = true;
            }
            slots_[k]->cv.notify_all();
        }
    }

    std::mutex mu_;
    std::atomic<bool> stop_{false};
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::thread> workers_;
};

}  // namespace strata::core
