// Strata. The batch plan (src/dsv41/ds4_batch_plan.hpp, M2 of docs/DSV41_BATCHING.md): scheduler step -> per-run
// arena blocks, offsets, eviction and errors.  No GPU.
#include "ds4_batch_plan.hpp"

#include <cstdio>

using strata::ds4::BatchKv;
using strata::ds4::BatchPlanner;
using strata::ds4::KvBatch;
using strata::ds4::StepPlan;

static int fails = 0;
#define CHECK(x)                                                                                    \
    do {                                                                                            \
        if (!(x)) {                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

static BatchKv cfg(int32_t blocks, int32_t seqs) {
    BatchKv c;
    c.block_tokens = 4;
    c.blocks = blocks;
    c.max_seqs = seqs;
    return c;
}

int main() {
    {   // a decode run and a 6-token prefill run in one step
        KvBatch kv(cfg(8, 2));
        CHECK(kv.open(1) && kv.open(2));
        BatchPlanner plan(kv);
        StepPlan step;
        step.entries.push_back({1, 1, false});
        step.entries.push_back({2, 6, true});
        std::vector<strata::ds4::SeqRun> runs;
        int32_t ev = -1;
        std::string err;
        CHECK(plan.build(step, runs, &ev, err));
        CHECK(runs.size() == 2 && ev == -1);
        CHECK(runs[0].id == 1 && runs[0].tokens == 1 && runs[0].pos == 0 && runs[0].blocks.size() == 1);
        CHECK(runs[1].id == 2 && runs[1].tokens == 6 && runs[1].pos == 0 && runs[1].blocks.size() == 2);
        CHECK(runs[1].blocks[0] != runs[1].blocks[1]);      // 6 tokens over blocks of 4
        // advance: sequence 2 now sits at position 6, the next decode lands in its second block at offset 2
        plan.done(2, 6);
        plan.done(1, 1);
        StepPlan step2;
        step2.entries.push_back({2, 1, false});
        CHECK(plan.build(step2, runs, &ev, err));
        CHECK(runs.size() == 1 && runs[0].pos == 6 && runs[0].offset == 2);
        CHECK(runs[0].blocks.size() == 1);
    }
    {   // a sequence the arena does not hold is an error, not a crash
        KvBatch kv(cfg(4, 2));
        CHECK(kv.open(1));
        BatchPlanner plan(kv);
        StepPlan step;
        step.entries.push_back({9, 1, false});
        std::vector<strata::ds4::SeqRun> runs;
        int32_t ev = -1;
        std::string err;
        CHECK(!plan.build(step, runs, &ev, err));
        CHECK(!err.empty() && runs.empty());
    }
    {   // the arena being full evicts the other sequence and says so
        KvBatch kv(cfg(2, 2));
        CHECK(kv.open(1) && kv.open(2));
        BatchPlanner plan(kv);
        StepPlan step;
        step.entries.push_back({1, 1, false});
        step.entries.push_back({2, 1, false});
        std::vector<strata::ds4::SeqRun> runs;
        int32_t ev = -1;
        std::string err;
        CHECK(plan.build(step, runs, &ev, err));       // two blocks for two sequences: fits
        CHECK(runs.size() == 2 && ev == -1);
        plan.done(1, 4);                              // each sequence fills its block
        plan.done(2, 4);
        // the next token of sequence 1 crosses into a new block; the arena is empty, so sequence 2 goes
        StepPlan step2;
        step2.entries.push_back({1, 1, false});
        ev = -1;
        CHECK(plan.build(step2, runs, &ev, err));
        CHECK(ev == 2);
        CHECK(runs.size() == 1 && runs[0].id == 1);
    }
    if (fails == 0) std::printf("ds4_batch_plan_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
