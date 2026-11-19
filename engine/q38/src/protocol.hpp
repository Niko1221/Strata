#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

// One request owns the output stream through its terminal reply. STOP targets
// the currently accepted queued or running request. Without request IDs,
// a delayed old STOP after a new GEN is indistinguishable from a new STOP.
class Q38Protocol {
public:
    enum class Kind { command, busy, overflow };
    struct Event { Kind kind; std::string line; };

    void submit(std::string line) {
        std::lock_guard<std::mutex> lock(mu_);
        // QUIT is a terminal control signal, not a bounded queue entry. Drain
        // earlier replies, then deliver it once; later input cannot postpone it.
        if (quit_requested_) return;
        if (line == "QUIT") {
            quit_requested_ = true;
            if (phase_ != Phase::idle) stop_.store(true);
            cv_.notify_one();
            return;
        }
        if (line == "STOP") {
            if (phase_ != Phase::idle) stop_.store(true);
            return;
        }
        // After overflow, coalesce further rejections until that marker has
        // been emitted. This preserves bounded storage and response order.
        if (overflow_ || queue_.size() >= capacity_) {
            overflow_ = true;
        } else if (line.rfind("GEN ", 0) == 0) {
            if (phase_ != Phase::idle || reply_pending_ || !queue_.empty()) queue_.push_back({Kind::busy, {}});
            else {
                stop_.store(false);
                phase_ = Phase::queued;
                queue_.push_back({Kind::command, std::move(line)});
            }
        } else {
            queue_.push_back({Kind::command, std::move(line)});
        }
        cv_.notify_one();
    }

    void end_input() {
        std::lock_guard<std::mutex> lock(mu_);
        eof_ = true;
        if (phase_ == Phase::running) stop_.store(true);
        cv_.notify_one();
    }

    bool next(Event &event) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return eof_ || quit_requested_ || overflow_ || !queue_.empty(); });
        if (!queue_.empty()) {
            event = std::move(queue_.front());
            queue_.pop_front();
            if (event.kind == Kind::command && event.line.rfind("GEN ", 0) == 0) {
                phase_ = Phase::running;
                if (eof_) stop_.store(true);
            } else {
                reply_pending_ = true;
            }
            return true;
        }
        if (overflow_) { overflow_ = false; reply_pending_ = true; event = {Kind::overflow, {}}; return true; }
        if (quit_requested_ && !quit_delivered_) {
            quit_delivered_ = true;
            reply_pending_ = true;
            event = {Kind::command, "QUIT"};
            return true;
        }
        return false;
    }

    // Publish and flush while admission and STOP are excluded. The callable
    // runs only on the output-owning thread and must not call this object.
    template<class Publish> void finish(Publish publish) {
        std::lock_guard<std::mutex> lock(mu_);
        // Decide cancellation while STOP submission is excluded, then publish
        // the terminal reply before another request may be admitted.
        publish(stop_.load());
        phase_ = Phase::idle;
    }
    template<class Publish> void finish_reply(Publish publish) {
        std::lock_guard<std::mutex> lock(mu_);
        publish();
        reply_pending_ = false;
    }
    std::atomic<bool> &stop_flag() { return stop_; }
    bool input_ended() const { std::lock_guard<std::mutex> lock(mu_); return eof_; }

private:
    enum class Phase { idle, queued, running };
    static constexpr size_t capacity_ = 2;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Event> queue_;
    Phase phase_ = Phase::idle;
    bool eof_ = false;
    bool overflow_ = false;
    bool reply_pending_ = false;
    bool quit_requested_ = false, quit_delivered_ = false;
    std::atomic<bool> stop_{false};
};
