#pragma once

#include <exception>
#include <new>
#include <system_error>
#include <thread>
#include <utility>

namespace strata::program {

// A completed batch releases its thread-local I/O scratch when wait joins.
// Task-pool async implementations do not provide that lifetime boundary.
class JoinedPrefetch {
    std::thread worker_;
    std::exception_ptr error_;
    bool active_ = false;

    template<class Task> void invoke(Task& task) noexcept {
        try { task(); } catch (...) { error_ = std::current_exception(); }
    }

public:
    JoinedPrefetch() = default;
    JoinedPrefetch(const JoinedPrefetch&) = delete;
    JoinedPrefetch& operator=(const JoinedPrefetch&) = delete;
    ~JoinedPrefetch() { if (worker_.joinable()) worker_.join(); }

    bool valid() const noexcept { return active_; }
    void get() {
        if (worker_.joinable()) worker_.join();
        active_ = false;
        if (auto error = std::exchange(error_, {})) std::rethrow_exception(error);
    }

    // Launcher must either return a joinable owned thread or throw before
    // starting work. The overload permits CPU injection of launch failures.
    template<class Task, class Launcher> void start(Task task, Launcher launch) {
        get();
        active_ = true;
        try {
            worker_ = launch([this, task]() mutable { invoke(task); });
        } catch (const std::bad_alloc&) {
            invoke(task);
        } catch (const std::system_error&) {
            invoke(task);
        }
    }
    template<class Task> void start(Task task) {
        start(task, [](auto work) { return std::thread(std::move(work)); });
    }
};

} // namespace strata::program
