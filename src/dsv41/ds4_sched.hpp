// Strata. The step scheduler of the DeepSeek-V4.1 engine: which sequences take the next step, and how much
// prompt each one reads.  M1 of docs/DSV41_BATCHING.md - it owns the order and the token budget, while the
// engine still runs its existing single-sequence kernel path once per entry (the batched kernels are M2/M3).
//
// The policy, in the order a plan is built:
//   1. running sequences take the batch first, round robin: a sequence still reading its prompt gets its next
//      chunk, a sequence whose prompt is read decodes one token;
//   2. then waiting sequences fill the rest, shortest prompt first (a chat question does not wait behind a
//      40K-token agent prompt), each with one prefill chunk;
//   3. the per-step token budget caps the prompt tokens read, so prefill never starves decode.
// No GPU and no model: a plan is ids and chunk sizes, so the test beside this header runs anywhere.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <vector>

namespace strata {
namespace ds4 {

struct SeqSpec {
    int64_t id = 0;
    int64_t prompt_tokens = 0;   // tokens the prompt has, in all
    int64_t prefilled = 0;       // read so far
    int64_t max_new = 0;
    int64_t generated = 0;
    double arrived = 0.0;        // monotonic seconds, for the ordering
};

struct StepPlan {
    struct Entry {
        int64_t id;
        int32_t tokens;          // prompt tokens to read, or 1 for a decode step
        bool prefill;
    };
    std::vector<Entry> entries;
    int32_t prefill_tokens = 0;
    int32_t decodes = 0;
    bool empty() const { return entries.empty(); }
};

class Scheduler {
public:
    struct Options {
        int32_t token_budget = 4096;   // prompt tokens one step may read in all
        int32_t max_batch = 3;         // sequences in one step
        int32_t prefill_chunk = 2048;  // at most this many prompt tokens per sequence per step
        int32_t short_first = 2048;    // prompts at most this long are admitted before longer ones (0: FIFO)
    };

    Scheduler() = default;
    explicit Scheduler(Options o) : o_(o) {}

    int64_t submit(int64_t prompt_tokens, int64_t max_new, double now) {
        SeqSpec s;
        s.id = ++next_id_;
        s.prompt_tokens = std::max<int64_t>(1, prompt_tokens);
        s.max_new = std::max<int64_t>(1, max_new);
        s.arrived = now;
        waiting_.push_back(s);
        return s.id;
    }

    bool cancel(int64_t id) {
        for (auto it = running_.begin(); it != running_.end(); ++it)
            if (it->id == id) { running_.erase(it); return true; }
        for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
            if (it->id == id) { waiting_.erase(it); return true; }
        return false;
    }

    size_t waiting() const { return waiting_.size(); }
    size_t running() const { return running_.size(); }
    const std::vector<SeqSpec>& running_seqs() const { return running_; }

    // One step's work.
    StepPlan plan() {
        StepPlan p;
        const int32_t batch_cap = std::max<int32_t>(1, o_.max_batch);
        int32_t prompt_left = std::max<int32_t>(0, o_.token_budget);

        // 1. running sequences, round robin: next prefill chunk or one decoded token
        const size_t n = running_.size();
        for (size_t k = 0; k < n && static_cast<int32_t>(p.entries.size()) < batch_cap; ++k) {
            SeqSpec& s = running_[(cursor_ + k) % n];
            if (s.prefilled < s.prompt_tokens) {
                int32_t chunk = static_cast<int32_t>(std::min<int64_t>(s.prompt_tokens - s.prefilled, o_.prefill_chunk));
                chunk = std::min(chunk, prompt_left);
                if (chunk <= 0) continue;
                p.entries.push_back({s.id, chunk, true});
                p.prefill_tokens += chunk;
                prompt_left -= chunk;
            } else {
                p.entries.push_back({s.id, 1, false});
                ++p.decodes;
            }
        }
        cursor_ = n ? (cursor_ + 1) % n : 0;

        // 2. admit, shortest prompt first (arrival order among equals)
        std::stable_sort(waiting_.begin(), waiting_.end(), [this](const SeqSpec& a, const SeqSpec& b) {
            if (o_.short_first > 0) {                     // a chat question jumps a queued agent prompt
                const bool a_long = a.prompt_tokens > o_.short_first;
                const bool b_long = b.prompt_tokens > o_.short_first;
                if (a_long != b_long) return !a_long;
            }
            return a.arrived < b.arrived;                 // FIFO among equals
        });
        while (static_cast<int32_t>(p.entries.size()) < batch_cap && !waiting_.empty() && prompt_left > 0) {
            SeqSpec s = waiting_.front();
            int32_t chunk = static_cast<int32_t>(std::min<int64_t>(s.prompt_tokens - s.prefilled, o_.prefill_chunk));
            chunk = std::min(chunk, prompt_left);
            if (chunk <= 0) break;
            p.entries.push_back({s.id, chunk, true});
            p.prefill_tokens += chunk;
            prompt_left -= chunk;
            waiting_.pop_front();
            running_.push_back(s);
        }
        return p;
    }

    // The engine ran one plan entry: advance that sequence.
    void done(int64_t id, int32_t tokens, bool prefill) {
        for (auto& s : running_) {
            if (s.id != id) continue;
            if (prefill) {
                s.prefilled = std::min<int64_t>(s.prompt_tokens, s.prefilled + tokens);
            } else {
                ++s.generated;
                if (s.prefilled < s.prompt_tokens) s.prefilled = s.prompt_tokens;
            }
            return;
        }
    }

    bool finished(int64_t id) const {
        for (const auto& s : running_)
            if (s.id == id) return s.generated >= s.max_new;
        return true;
    }

    void retire(int64_t id) {
        running_.erase(std::remove_if(running_.begin(), running_.end(),
                                      [id](const SeqSpec& s) { return s.id == id; }),
                       running_.end());
    }

private:
    Options o_{};
    int64_t next_id_ = 0;
    size_t cursor_ = 0;
    std::deque<SeqSpec> waiting_;
    std::vector<SeqSpec> running_;
};

}  // namespace ds4
}  // namespace strata
