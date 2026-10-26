// Strata. From a scheduler step to what the attention and MoE kernels need (M2/M3 of docs/DSV41_BATCHING.md):
// for every sequence in the step, the arena blocks its tokens live in and the offset inside the first one.
//
// The kernels take one run at a time today (one kernel call per entry); a varlen launch takes the whole vector,
// with per-run block tables.  Pure CPU: the test beside it needs no GPU.
#pragma once

#include "ds4_batch_kv.hpp"
#include "ds4_sched.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata {
namespace ds4 {

struct SeqRun {
    int64_t id = 0;
    int64_t pos = 0;              // position of the first token of this run
    int32_t tokens = 0;           // 1 for a decode step, the chunk size for a prefill
    bool prefill = false;
    int32_t offset = 0;           // pos % block_tokens: where blocks[0] starts
    std::vector<int32_t> blocks;  // arena blocks holding [pos, pos + tokens)
};

class BatchPlanner {
public:
    explicit BatchPlanner(KvBatch& kv) : kv_(kv) {}

    /// Turn a step into runs.  A sequence the arena does not hold is an error; an eviction (the arena was full)
    /// is reported in *evicted (-1: none), and the caller must drop that sequence.
    bool build(const StepPlan& step, std::vector<SeqRun>& runs, int32_t* evicted, std::string& err) {
        runs.clear();
        *evicted = -1;
        for (const auto& e : step.entries) {
            const int32_t seq = static_cast<int32_t>(e.id);
            const KvSeq* s = kv_.find(seq);
            if (s == nullptr) {
                err = "sequence " + std::to_string(e.id) + " is not in the KV batch";
                return false;
            }
            SeqRun r;
            r.id = e.id;
            r.pos = s->tokens();
            r.tokens = e.tokens;
            r.prefill = e.prefill;
            r.offset = static_cast<int32_t>(r.pos % kv_.cfg().block_tokens);
            int32_t did = -1;
            const int64_t slot = kv_.touch(seq, r.pos + r.tokens - 1, &did);
            if (slot < 0) {
                err = "no KV room for sequence " + std::to_string(e.id);
                return false;
            }
            if (did >= 0) *evicted = did;
            s = kv_.find(seq);
            if (s == nullptr) {
                err = "the arena dropped sequence " + std::to_string(e.id);
                return false;
            }
            const int32_t first = static_cast<int32_t>(r.pos / kv_.cfg().block_tokens);
            const int32_t last = static_cast<int32_t>((r.pos + r.tokens - 1) / kv_.cfg().block_tokens);
            for (int32_t b = first; b <= last; ++b) {
                if (static_cast<size_t>(b) >= s->blocks().size()) {
                    err = "block table short for sequence " + std::to_string(e.id);
                    return false;
                }
                r.blocks.push_back(s->blocks()[static_cast<size_t>(b)]);
            }
            runs.push_back(std::move(r));
        }
        return true;
    }

    /// The sequence wrote `tokens` more tokens (after its run ran).
    void done(int64_t id, int32_t tokens) { kv_.advance(static_cast<int32_t>(id), tokens); }

private:
    KvBatch& kv_;
};

}  // namespace ds4
}  // namespace strata
