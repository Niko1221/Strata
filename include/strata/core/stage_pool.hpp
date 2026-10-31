#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::core {

class StagePool {
public:
    explicit StagePool(size_t stages) {
        slots_.reserve(stages);
        workers_.reserve(stages);
        for (size_t k = 0; k < stages; ++k) slots_.push_back(std::make_unique<Slot>());
        try {
            for (size_t k = 0; k < stages; ++k) workers_.emplace_back([this, k] { work(k); });
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    ~StagePool() { stop_and_join(); }

    StagePool(const StagePool&) = delete;
    StagePool& operator=(const StagePool&) = delete;

    size_t size() const { return slots_.size(); }

    bool run(const std::vector<std::function<bool(std::string&)>>& bodies, std::string& error,
             std::vector<double>* timings_ms = nullptr) {
        if (bodies.size() != slots_.size()) {
            error = "stage pool: wrong number of stages";
            return false;
        }
        std::vector<std::function<bool(std::string&)>> pending = bodies;
        std::lock_guard<std::mutex> run_lock(run_mu_);
        if (stop_.load()) {
            error = "stage pool: shut down";
            return false;
        }
        if (timings_ms) timings_ms->assign(slots_.size(), 0.0);
        for (size_t k = 0; k < pending.size(); ++k) {
            std::lock_guard<std::mutex> lk(slots_[k]->mu);
            slots_[k]->body = std::move(pending[k]);
            slots_[k]->error.clear();
            slots_[k]->exception = nullptr;
            slots_[k]->measure = timings_ms != nullptr;
            slots_[k]->done = false;
            slots_[k]->ready = true;
        }
        for (auto& s : slots_) s->cv.notify_all();
        std::string first;
        std::exception_ptr first_exception;
        for (size_t k = 0; k < slots_.size(); ++k) {
            auto& slot = slots_[k];
            std::unique_lock<std::mutex> lk(slot->mu);
            slot->cv.wait(lk, [&] { return slot->done || stop_.load(); });
            if (!slot->done) {   // stop_ won the race; report it instead of reading unset fields
                error = "stage pool: shut down while stages were pending";
                return false;
            }
            if (timings_ms) (*timings_ms)[k] = slot->elapsed_ms;
            if (!first_exception && slot->exception) first_exception = slot->exception;
            if (first.empty() && !slot->error.empty()) first = slot->error;
        }
        if (first_exception) std::rethrow_exception(first_exception);
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
        std::exception_ptr exception;
        double elapsed_ms = 0.0;
        bool measure = false;
        bool ready = false;
        bool done = false;
    };

    void stop_and_join() {
        // Taking run_mu_ makes destruction safe against an in-flight run(): the run
        // completes first (this only waits out its bodies), then the workers stop.
        std::lock_guard<std::mutex> run_lock(run_mu_);
        // Set stop_ under every slot's mutex so a worker that checked its predicate and
        // is about to sleep cannot miss the wakeup (notify_all alone races that window).
        for (auto& s : slots_) {
            std::lock_guard<std::mutex> lk(s->mu);
            stop_ = true;
        }
        for (auto& s : slots_) s->cv.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
    }

    void work(size_t k) {
        Slot& slot = *slots_[k];
        for (;;) {
            std::function<bool(std::string&)> body;
            {
                std::unique_lock<std::mutex> lk(slot.mu);
                slot.cv.wait(lk, [&] { return stop_ || slot.ready; });
                if (stop_) {
                    // A run() in its completion wait, or a later one, must observe this
                    // slot as finished instead of waiting for a worker that is gone.
                    slot.error = "stage pool: stopped before this stage ran";
                    slot.done = true;
                    slot.cv.notify_all();
                    return;
                }
                body = std::move(slot.body);
                slot.ready = false;
            }
            std::exception_ptr exception;
            const auto started = slot.measure ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
            try {
                std::string error;
                const bool ok = body ? body(error) : true;
                slot.error = ok ? std::string() : (error.empty() ? std::string("stage failed") : error);
            } catch (...) {
                exception = std::current_exception();
            }
            body = nullptr;
            {
                std::lock_guard<std::mutex> lk(slot.mu);
                slot.exception = exception;
                if (slot.measure)
                    slot.elapsed_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                slot.done = true;
            }
            slot.cv.notify_all();
        }
    }

    std::mutex run_mu_;
    std::atomic<bool> stop_{false};
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::thread> workers_;
};

}
