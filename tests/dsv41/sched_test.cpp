// Strata. The DeepSeek-V4.1 step scheduler (src/dsv41/ds4_sched.hpp, M1 of docs/DSV41_BATCHING.md): admission
// order, prefill chunking, the token budget, round-robin decode and retirement.  No GPU, no model.
#include "ds4_sched.hpp"

#include <cstdio>

using strata::ds4::Scheduler;
using strata::ds4::StepPlan;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

static int32_t tokens_of(const StepPlan& p, int64_t id) {
    for (const auto& e : p.entries)
        if (e.id == id) return e.tokens;
    return -1;
}

int main() {
    {   // three short prompts: all admitted in one step, each reads its whole prompt
        Scheduler s;
        const int64_t a = s.submit(40, 100, 0.0), b = s.submit(60, 100, 0.1), c = s.submit(20, 100, 0.2);
        CHECK(s.waiting() == 3);
        StepPlan p = s.plan();
        CHECK(p.entries.size() == 3);
        CHECK(p.decodes == 0);
        CHECK(p.prefill_tokens == 120);
        CHECK(tokens_of(p, a) == 40 && tokens_of(p, b) == 60 && tokens_of(p, c) == 20);
        // arrival order among short prompts: a, b, c
        CHECK(p.entries[0].id == a && p.entries[1].id == b && p.entries[2].id == c);
        for (const auto& e : p.entries) s.done(e.id, e.tokens, true);
        CHECK(s.running() == 3 && s.waiting() == 0);
        // prompts read: the next steps decode, one token each, round robin
        StepPlan d1 = s.plan(), d2 = s.plan();
        CHECK(d1.entries.size() == 3 && d1.decodes == 3 && d1.prefill_tokens == 0);
        CHECK(d1.entries[0].id != d2.entries[0].id);   // the first decode moves to the back
        for (const auto& e : d1.entries) s.done(e.id, 1, false);
    }
    {   // a long prompt reads in chunks, the sequence stays running until its prompt is read
        Scheduler::Options o;
        o.prefill_chunk = 2048;
        Scheduler s(o);
        const int64_t id = s.submit(5000, 10, 0.0);
        StepPlan p1 = s.plan();
        CHECK(tokens_of(p1, id) == 2048);
        s.done(id, 2048, true);
        StepPlan p2 = s.plan();
        CHECK(tokens_of(p2, id) == 2048);       // still prefilling, and it is running, not waiting
        CHECK(p2.decodes == 0);
        s.done(id, 2048, true);
        StepPlan p3 = s.plan();
        CHECK(tokens_of(p3, id) == 904);        // the rest
        s.done(id, 904, true);
        StepPlan p4 = s.plan();
        CHECK(tokens_of(p4, id) == 1 && p4.decodes == 1);
    }
    {   // the token budget caps a step (4096 over 3 x 5000-token prompts: two chunks, not three)
        Scheduler::Options o;
        o.token_budget = 4096;
        o.prefill_chunk = 2048;
        Scheduler s(o);
        for (int i = 0; i < 3; ++i) s.submit(5000, 10, 0.0);
        StepPlan p = s.plan();
        CHECK(p.entries.size() == 2);
        CHECK(p.prefill_tokens == 4096);
    }
    {   // short-first admission: the chat question is admitted before the queued 40K-token agent prompt
        Scheduler::Options o;
        o.token_budget = 30;            // room for the short prompt only, so the order decides who runs
        o.prefill_chunk = 2048;
        Scheduler s(o);
        const int64_t long_id = s.submit(40000, 10, 0.0);
        const int64_t short_id = s.submit(30, 10, 1.0);
        StepPlan p = s.plan();
        CHECK(p.entries.size() == 1);
        CHECK(p.entries[0].id == short_id && tokens_of(p, long_id) == -1);
        // with room for both, both are admitted and the short one comes first
        Scheduler::Options o2;
        o2.token_budget = 4096;
        Scheduler s2(o2);
        const int64_t long2 = s2.submit(40000, 10, 0.0);
        const int64_t short2 = s2.submit(30, 10, 1.0);
        StepPlan p2 = s2.plan();
        CHECK(p2.entries.size() == 2);
        CHECK(p2.entries[0].id == short2 && p2.entries[1].id == long2);
        // FIFO when the rule is off: the long one queued first is admitted first
        Scheduler::Options o3;
        o3.token_budget = 2048;
        o3.short_first = 0;
        Scheduler f(o3);
        const int64_t first = f.submit(40000, 10, 0.0);
        f.submit(30, 10, 1.0);
        StepPlan q = f.plan();
        CHECK(q.entries.size() == 1 && q.entries[0].id == first);
    }
    {   // max_batch caps the step; cancel works waiting and running
        Scheduler::Options o;
        o.max_batch = 2;
        Scheduler s(o);
        s.submit(10, 5, 0.0);
        s.submit(10, 5, 0.0);
        const int64_t third = s.submit(10, 5, 0.0);
        StepPlan p = s.plan();
        CHECK(p.entries.size() == 2);
        CHECK(s.cancel(third));
        CHECK(!s.cancel(third));
    }
    {   // retirement: max_new decoded tokens finish a sequence
        Scheduler s;
        const int64_t id = s.submit(5, 2, 0.0);
        StepPlan p = s.plan();
        s.done(id, 5, true);
        CHECK(!s.finished(id));
        for (int i = 0; i < 2; ++i) {
            StepPlan d = s.plan();
            CHECK(tokens_of(d, id) == 1);
            s.done(id, 1, false);
        }
        CHECK(s.finished(id));
        s.retire(id);
        CHECK(s.running() == 0);
        CHECK(s.plan().empty());
    }
    if (fails == 0) std::printf("ds4_sched_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
