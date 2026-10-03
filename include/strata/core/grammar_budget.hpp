// Cooperative bounds for the pinned native grammar backend. No model or I/O calls.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::grammar::detail {

struct WorkBudget {
    uint64_t remaining;
    uint64_t used = 0;
    std::chrono::steady_clock::time_point deadline{};
};

inline thread_local WorkBudget* active_budget = nullptr;

inline void check_deadline() {
    if (active_budget && std::chrono::steady_clock::now() > active_budget->deadline)
        throw std::runtime_error("grammar resource limit: operation deadline exceeded");
}

inline void work(uint64_t units = 1) {
    if (!active_budget) return;
    if (units > active_budget->remaining)
        throw std::runtime_error("grammar resource limit: work budget exhausted");
    active_budget->remaining -= units;
    const uint64_t before = active_budget->used;
    active_budget->used += units;
    if ((before >> 8) != (active_budget->used >> 8) || before == 0) check_deadline();
}

inline void bound(size_t count, size_t maximum) {
    if (active_budget && count > maximum)
        throw std::runtime_error("grammar resource limit: parser or automaton state limit exceeded");
}

// Compiler and matcher calls are single-threaded. A sequence retains its work
// count across tokens; the deadline covers each native grammar operation only.
class WorkScope {
public:
    WorkScope(WorkBudget& budget, std::chrono::milliseconds duration)
        : previous_(active_budget) {
        budget.deadline = std::chrono::steady_clock::now() + duration;
        active_budget = &budget;
    }
    ~WorkScope() { active_budget = previous_; }
    WorkScope(const WorkScope&) = delete;
    WorkScope& operator=(const WorkScope&) = delete;
    void finish() const { check_deadline(); }
private:
    WorkBudget* previous_;
};

} // namespace strata::grammar::detail
