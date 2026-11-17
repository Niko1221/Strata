// Unit tests for the adaptive recycle policy (header-only, no CUDA):
//   g++ -std=c++17 -O1 -Wall -Wextra -I include src/program/recycle_policy_test.cpp
#include "strata/program/recycle_policy.hpp"

#include <cstdio>

namespace rp = strata::program::adaptive_slots;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    {   // max = 0 disables everything
        rp::RecyclePolicy p(0, 2);
        check(!p.enabled(), "max=0 is off");
        check(p.grace_s() == 0, "off keeps grace 0");
    }
    {   // defaults: 2 s grace, doubling up to 600
        rp::RecyclePolicy p(600, 2);
        check(p.enabled() && p.grace_s() == 2 && p.backoff_on(), "defaults 2/600");
        check(p.on_grow(4, 5) == false, "grow without a recycle does not double");
        check(p.grace_s() == 2, "grace untouched");
    }
    {   // min is clamped to max; min == max is the fixed mode
        rp::RecyclePolicy p(10, 30);
        check(p.grace_s() == 10 && !p.backoff_on(), "min clamped to max");
        rp::RecyclePolicy q(30, 30);
        check(q.grace_s() == 30 && !q.backoff_on(), "min=max is fixed");
        q.on_recycle(0);
        check(!q.on_grow(4, 1), "fixed mode never doubles");
        check(!q.maybe_reset(1LL << 40), "fixed mode never resets");
    }
    {   // one doubling per recycle cycle, window = max
        rp::RecyclePolicy p(300, 2);
        p.on_recycle(100);
        check(p.on_grow(1, 103), "first grow after recycle doubles");
        check(p.grace_s() == 4, "grace 2 -> 4");
        check(!p.on_grow(1, 104), "second grow in the same cycle does not double");
        p.on_recycle(200);
        check(p.grace_s() == 4, "recycle keeps the grace");
        check(p.on_grow(4, 207), "new recycle cycle doubles again");
        check(p.grace_s() == 8, "grace 4 -> 8");
        check(!p.on_grow(1, 900), "grow after max does not double");
    }
    {   // exact window boundary and cap
        rp::RecyclePolicy p(300, 2);
        p.on_recycle(0);
        check(p.on_grow(1, 300), "exact max boundary doubles");
        check(p.grace_s() == 4, "grace 2 -> 4");
        rp::RecyclePolicy c(16, 2);
        int64_t t = 0;
        c.on_recycle(t);
        for (int i = 0; i < 10; ++i) {
            ++t;
            c.on_grow(1, t);
            c.on_recycle(t);
        }
        check(c.grace_s() <= c.max_s(), "grace capped at max");
    }
    {   // reset needs 4*max without batch activity
        rp::RecyclePolicy p(10, 2);
        p.on_recycle(0);
        check(p.on_grow(1, 3), "grace 2 -> 4");
        p.on_activity(20);
        check(!p.maybe_reset(59), "activity keeps the reset away");
        check(p.maybe_reset(60), "4*max after the last activity resets");
        check(p.grace_s() == 2, "grace back to the minimum");
        check(!p.maybe_reset(1000), "no second reset without a new recycle");
    }
    {   // large max: no overflow and the cap holds
        rp::RecyclePolicy p(1LL << 62, 2);
        p.on_recycle(0);
        check(p.on_grow(1, 3), "huge max doubles");
        check(p.grace_s() == 4, "huge max: 2 -> 4");
        check(!p.maybe_reset(1LL << 62), "huge max: quiet window not reached, no overflow");
    }
    {   // a custom minimum grace
        rp::RecyclePolicy p(30, 5);
        check(p.grace_s() == 5, "starts at the min");
        p.on_recycle(0);
        check(p.on_grow(1, 3), "doubles from the min");
        check(p.grace_s() == 10, "grace 5 -> 10");
        p.on_activity(20);
        check(!p.maybe_reset(110), "still inside 4*max");
        check(p.maybe_reset(140), "reset after the 4*max quiet window");
        check(p.grace_s() == 5, "grace back to the min");
    }
    {   // seconds parsing for the env knobs
        check(rp::seconds_value("30", 600) == 30, "seconds value");
        check(rp::seconds_value(nullptr, 600) == 600, "unset falls back");
        check(rp::seconds_value("", 600) == 600, "empty falls back");
        check(rp::seconds_value("0", 600) == 600, "0 falls back");
        check(rp::seconds_value("-5", 600) == 600, "negative falls back");
        check(rp::seconds_value("12s", 600) == 600, "trailing text falls back");
        check(rp::seconds_value("abc", 600) == 600, "not a number falls back");
    }
    {   // a burst cluster keeps the grace; only idle > 2*observed gap halves it
        rp::RecyclePolicy p(300, 2);
        p.on_activity(0);
        p.on_activity(10);                      // observed gap = 10
        p.on_recycle(10);                       // recycle does not decay
        check(p.on_grow(1, 11), "grow after recycle doubles");
        check(p.grace_s() == 4, "grace 2 -> 4");
        check(!p.maybe_decay(25), "14 s < 2*observed gap: keep the grace");
        check(p.grace_s() == 4, "grace kept");
        check(p.maybe_decay(35), "24 s >= 2*observed gap: decay");
        check(p.grace_s() == 2, "grace 4 -> 2");
        check(!p.maybe_decay(100), "already at the minimum");
    }
    {   // a larger observed gap keeps the grace longer
        rp::RecyclePolicy p(300, 2);
        p.on_activity(0);
        p.on_activity(30);                      // observed gap = 30
        p.on_recycle(30);
        p.on_grow(1, 31);
        check(p.grace_s() == 4, "grace 2 -> 4");
        check(!p.maybe_decay(80), "49 s < 2*30: keep");
        check(p.maybe_decay(95), "65 s >= 60: decay");
        check(p.grace_s() == 2, "grace 4 -> 2");
    }
    {   // 4*max with no batch activity resets the grace to the minimum directly
        rp::RecyclePolicy p(300, 2);
        p.on_recycle(0);
        p.on_grow(1, 1);
        check(p.grace_s() == 4, "grace raised");
        check(p.maybe_reset(1 + 4 * 300 + 10), "quiet > 4*max resets to min");
        check(p.grace_s() == 2, "grace back to the minimum");
    }
    if (failures == 0) std::printf("recycle_policy_test: all passed\n");
    return failures == 0 ? 0 : 1;
}
