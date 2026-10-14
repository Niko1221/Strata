// include/strata/core/optional_alloc.hpp - optional device copies are placed after every required allocation (no GPU
// needed to check).
//
// NativeDense::load keeps a second, packed copy of some matrices when STRATA_Q8_PACKED=1.  A copy is only a faster layout
// of bytes the device already holds (the kernel reads the GGUF layout when a matrix has none), so it must never cost a
// required allocation.  `load` runs its allocations through one RequiredThenOptional: each required allocation (a
// matrix, a STRATA_HC_Q8 copy, the q8_1 scratch) runs at once through `required`, each packed copy is queued through
// `want` where load decides it, and `finish` makes the queued copies, in order, after the last required one.  It stops
// at the first copy that does not fit: the copies placed before it stay, the rest keep the GGUF layout.  The steps are
// the caller's own allocate-and-upload code, so a test runs the same sequencer against a fake device
// (tests/core/optional_alloc_test.cpp).
#pragma once

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace strata::core {

/// The result of one allocation attempt.
enum class AllocTry { ok, out_of_memory, failed };

struct OptionalPlaced {
    size_t placed = 0;           ///< copies [0, placed) are allocated
    bool out_of_memory = false;  ///< copy `placed` did not fit: it and the ones after it were not tried
    bool failed = false;         ///< copy `placed` failed another way (an upload): the caller reports an error
};

/// Tries copies 0 .. n-1 in order with `try_one(i)` (returns AllocTry) and stops at the first that is not `ok`.
template <class Try>
OptionalPlaced place_optional(size_t n, Try&& try_one) {
    OptionalPlaced r;
    for (; r.placed < n; ++r.placed) {
        const AllocTry t = try_one(r.placed);
        if (t == AllocTry::ok) continue;
        if (t == AllocTry::out_of_memory) r.out_of_memory = true;
        else r.failed = true;
        break;
    }
    return r;
}

/// The order of a load's device allocations: required ones as they come, optional copies after all of them.
class RequiredThenOptional {
public:
    /// A required allocation: runs `step()` now; it returns false when it failed (and sets the caller's error).  After
    /// `finish` it is refused: returns false without running `step`, and `late_required()` turns true.
    template <class Step>
    bool required(Step&& step) {
        if (finished_) { late_required_ = true; return false; }
        return std::forward<Step>(step)();
    }
    /// An optional copy: queued, not run here.  `finish` runs it after every required allocation.  Returns its index.
    size_t want(std::function<AllocTry()> step) {
        wanted_.push_back(std::move(step));
        return wanted_.size() - 1;
    }
    /// Makes the queued copies in order through place_optional.  Runs them once; a second call returns the same result.
    OptionalPlaced finish() {
        if (!finished_) {
            finished_ = true;
            result_ = place_optional(wanted_.size(), [&](size_t i) { return wanted_[i](); });
        }
        return result_;
    }
    size_t wanted() const { return wanted_.size(); }
    /// Copy `i` (an index `want` returned) was made.  The others keep the GGUF layout only.
    bool placed(size_t i) const { return finished_ && i < result_.placed; }
    bool late_required() const { return late_required_; }

private:
    std::vector<std::function<AllocTry()>> wanted_;
    OptionalPlaced result_;
    bool finished_ = false, late_required_ = false;
};

}  // namespace strata::core
