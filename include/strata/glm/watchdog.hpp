#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace strata::glm {
// Active only inside a forward pass, so idle servers never time out. Destruction joins the monitor.
class Watchdog {
public:
    explicit Watchdog(double seconds) : seconds_(seconds) {
        if (seconds <= 0) return;
        thread_ = std::thread([this] {
            std::unique_lock<std::mutex> lock(m_);
            while (!quit_) {
                cv_.wait_for(lock, std::chrono::milliseconds(100));
                if (active_ && std::chrono::duration<double>(Clock::now() - last_).count() > seconds_) {
                    std::fprintf(stderr, "strata-glm: watchdog: no forward progress for %.1f seconds\n", seconds_);
                    std::fflush(stderr);
                    std::_Exit(124);
                }
            }
        });
    }
    ~Watchdog() {
        { std::lock_guard<std::mutex> lock(m_); quit_ = true; }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    void touch() { if (seconds_ > 0) { std::lock_guard<std::mutex> lock(m_); last_ = Clock::now(); } }
    void active(bool on) { std::lock_guard<std::mutex> lock(m_); active_ = on; last_ = Clock::now(); }
    struct Guard { Watchdog& w; Guard(Watchdog& x):w(x){w.active(true);} ~Guard(){w.active(false);} };
private:
    using Clock = std::chrono::steady_clock;
    double seconds_;
    Clock::time_point last_ = Clock::now();
    bool active_ = false, quit_ = false;
    std::mutex m_; std::condition_variable cv_; std::thread thread_;
};
} // namespace strata::glm
