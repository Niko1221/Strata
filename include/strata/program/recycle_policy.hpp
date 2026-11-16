// Adaptive slot lifetime for `--parallel-recycle`.
//
// The engine reserves no batch slot session until a request needs one, and recycles the sessions
// once a burst is over: a solo request runs on the solo path and needs no session, so an idle engine
// should have the whole expert cache.  The policy here decides how long the engine waits for the
// next request before it recycles:
//
//   * the grace starts at STRATA_PARALLEL_RECYCLE_MIN (2 s by default) and is clamped to the maximum;
//   * a batch request that follows a recycle within the maximum doubles the grace, once per recycle
//     cycle, up to STRATA_PARALLEL_RECYCLE_MAX (600 s by default), so a burst every few seconds does not
//     keep paying a shrink+grow pair;
//   * 3 * max with no batch activity at all resets the grace to the minimum.
//
// Setting min == max is the fixed mode (the grace is constant, the old fixed-S behaviour).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cerrno>
#include <cstdlib>

namespace strata::program::adaptive_slots {

// A seconds value from text: fallback when it is not a whole number of at least 1.
inline int64_t seconds_value(const char* v, int64_t fallback) {
    if (v == nullptr || *v == '\0') return fallback;
    errno = 0;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE || n < 1 || n > 86400) return fallback;
    return (int64_t) n;
}

// The env knob's value, with its default (STRATA_PARALLEL_RECYCLE_MIN/MAX).
inline int64_t env_seconds(const char* name, int64_t fallback) {
    return seconds_value(std::getenv(name), fallback);
}

// The grace the policy starts with: min(base, max); equal values mean the fixed mode (no backoff).
inline int64_t initial_grace(int64_t max_s, int64_t base_s) {
    if (max_s <= 0) return 0;
    return std::min<int64_t>(std::max<int64_t>(base_s, 1), max_s);
}

class RecyclePolicy {
 public:
    RecyclePolicy(int64_t max_s, int64_t base_s)
        : max_s_(max_s > 0 ? max_s : 0),
          backoff_(max_s > 0 && base_s < max_s),
          base_s_(max_s > 0 ? std::min<int64_t>(std::max<int64_t>(base_s, 1), max_s) : 0),
          grace_s_(initial_grace(max_s, base_s)) {}

    bool enabled() const { return max_s_ > 0; }
    int64_t max_s() const { return max_s_; }
    int64_t base_s() const { return base_s_; }
    int64_t grace_s() const { return grace_s_; }
    bool backoff_on() const { return backoff_; }

    // Any batch request (BGEN) keeps the quiet timer alive and feeds the observed inter-burst gap
    // (an EWMA): the decay below uses it to tell a burst cluster from a real idle stretch.
    void on_activity(int64_t now_s) {
        if (!enabled()) return;
        if (last_activity_s_ >= 0) {
            int64_t gap = now_s - last_activity_s_;
            if (gap < 0) gap = 0;
            observed_gap_s_ = observed_gap_s_ <= 0 ? gap : (observed_gap_s_ * 7 + gap) / 8;
            if (observed_gap_s_ < 1) observed_gap_s_ = 1;
        }
        last_activity_s_ = now_s;
        last_decay_s_ = -1;
    }

    // A successful recycle to zero sessions starts a new backoff cycle.  It does not decay the
    // grace: a burst cluster keeps its window, and only a real idle stretch shrinks it (maybe_decay).
    void on_recycle(int64_t now_s) {
        if (!enabled()) return;
        last_recycle_s_ = now_s;
        last_activity_s_ = now_s;
        last_decay_s_ = -1;
        backoff_used_ = false;
    }

    // A successful grow (added slot sessions) that follows a recycle: double the grace once.
    bool on_grow(int64_t added, int64_t now_s) {
        if (!enabled()) return false;
        last_activity_s_ = now_s;
        if (!backoff_ || added <= 0 || backoff_used_ || last_recycle_s_ < 0) return false;
        if (now_s - last_recycle_s_ > max_s_) return false;
        backoff_used_ = true;
        const int64_t next = grace_s_ > max_s_ / 2 ? max_s_ : std::min<int64_t>(max_s_, grace_s_ * 2);
        if (next == grace_s_) return false;
        grace_s_ = next;
        return true;
    }

    // Idle decay: once the engine has been quiet for twice the observed burst gap, halve the grace
    // (floored at the minimum), and keep halving on every further such stretch.  A burst arriving
    // inside that stretch keeps the grace, so a cluster does not churn through shrink+grow.
    bool maybe_decay(int64_t now_s) {
        if (!enabled() || !backoff_ || grace_s_ <= base_s_) return false;
        if (last_activity_s_ < 0) return false;
        const int64_t gap = observed_gap_s_ > 0 ? observed_gap_s_ : base_s_;
        int64_t threshold = gap > INT64_MAX / 2 ? INT64_MAX : 2 * gap;
        if (threshold < base_s_) threshold = base_s_;
        const int64_t since = std::max(last_activity_s_, last_decay_s_);
        if (now_s - since < threshold) return false;
        last_decay_s_ = now_s;
        grace_s_ = std::max(base_s_, grace_s_ / 2);
        return true;
    }

    // 4*max with no batch activity resets the grace to the minimum.  Returns true when it changed.
    bool maybe_reset(int64_t now_s) {
        if (!enabled() || !backoff_) return false;
        const int64_t quiet = max_s_ > INT64_MAX / 4 ? INT64_MAX : 4 * max_s_;
        const int64_t last = std::max(last_recycle_s_, last_activity_s_);
        if (last < 0 || now_s - last < quiet) return false;
        last_recycle_s_ = -1;
        backoff_used_ = false;
        const bool changed = grace_s_ != base_s_;
        grace_s_ = base_s_;
        return changed;
    }

 private:
    int64_t max_s_ = 0;
    bool backoff_ = false;
    int64_t base_s_ = 0;
    int64_t grace_s_ = 0;
    int64_t last_recycle_s_ = -1;
    int64_t last_activity_s_ = -1;
    int64_t last_decay_s_ = -1;
    int64_t observed_gap_s_ = 0;
    bool backoff_used_ = false;
};

}  // namespace strata::program::adaptive_slots
