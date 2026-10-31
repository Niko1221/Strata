// include/strata/prefill/score.hpp - teacher-forced scoring through the prompt path.
//
// WHAT THIS IS FOR.  MeZO (and any zeroth-order method) needs f(theta) for one fixed token sequence: the sum of
// log p(x_t | x_<t) over every position, for two weight vectors theta +/- eps.  Generation cannot answer that: it
// reports the log-probability of the token it SAMPLED, after temperature and truncation, and nothing for the prompt
// positions.  This is the other thing - a raw, full-vocabulary log-softmax score for a sequence someone else chose.
//
// HOW IT STAYS CHEAP.  It rides the prompt path the engine already has: `Prefill::on_chunk` hands every chunk's
// final multi-stream residual rows (`T x hc*n_embd`, device) to a callback, and nothing in the chunk loop changes.
// The head then runs ONCE PER ROW BLOCK instead of once per row - see score_rows.hpp for why the difference is
// 40 TB of weight traffic at 128K versus ~2 GB.
//
// The state it consumes.  `run` advances the session exactly as a request's prompt would (KV pools, GDN recurrence,
// PLE history, indexer), so a Scorer is not re-entrant with a live conversation: a caller gives it a session it can
// spend.  Nothing else about the engine's behaviour changes.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {
class ExpertCache;
class ExpertSource;
class NativeHead;
struct ModelGeometry;   // struct, and the spelling has to match: MSVC mangles a class tag differently
struct SessionState;
class WeightTable;
}  // namespace strata::core

namespace strata::prefill {

class Prefill;

class Scorer {
public:
    Scorer();
    ~Scorer();
    Scorer(const Scorer&) = delete;
    Scorer& operator=(const Scorer&) = delete;

    /// Bind to an already-initialized prompt path and the weights it reads.  `pf` must outlive the Scorer and must
    /// not be running anything else.  Its `on_chunk` is taken over for the duration of `run` and restored after.
    /// `head`: the native output projection the engine uses when `--native` is on (the one case where
    /// `output.weight` in the table is a canonical tensor with no GGUF blocks); null reads the table's own.
    /// Allocates the head's working set (see the .cpp): the FP16 image of `output.weight` when it fits, a
    /// `rows x vocab` logits block, and per-token mixer scratch.
    bool attach(Prefill& pf, const core::WeightTable& wt, const core::ModelGeometry& g, const core::NativeHead* head,
                void* stream, std::string& err);

    /// Score positions [0, n) of `tokens` against `targets`: `out[t]` = log p(targets[t] | tokens[0..t]).  `targets`
    /// and `out` are HOST arrays of n entries; a negative target marks a position with nothing to score and `out`
    /// receives NaN.  For next-token training `targets[t] = tokens[t+1]` and the last position is negative.
    bool run(const int64_t* tokens, int64_t n, const int32_t* targets, float* out, std::string& err);

    /// Positions scored so far in the last `run`, and the time the head's own work took (the prompt path's time is
    /// `Prefill::stats()`).
    int64_t scored() const;
    double ms_head() const;
    /// Row block size the head settled on (it halves when a block does not fit).
    int64_t rows_per_block() const;
    /// What the head's layout cost: whether `output.weight` fits as an FP16 image or is dequantized per row block,
    /// and why.  A fallback is 4x-40x slower and silent otherwise, so the caller prints this once.
    const std::string& note() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::prefill
