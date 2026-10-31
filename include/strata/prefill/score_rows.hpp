// include/strata/prefill/score_rows.hpp - teacher-forced scoring on the prompt path.
//
// The prompt path leaves a chunk's final multi-stream residual rows (`Prefill::on_chunk`'s `R_rows`, T x hc*n_embd)
// and then throws them away: `lm_head` is only ever called for the one position a decode step needs.  Scoring a
// fixed token sequence needs the opposite - a log-probability for EVERY position - and doing it with the decode
// path's `lm_head` would read `output.weight` once per position: 318 MB x 131072 positions = 40 TB at 128K, about
// four hours.  These kernels turn the same head into one batched GEMM per row block, so the weight is read once per
// block instead.
//
// Two pieces, both here so a non-CUDA build still links (see the .cu):
//
//   score_rows      log p(target) = logits[target] - logsumexp(logits[row, :]) over the WHOLE vocabulary.  The
//                   log-sum-exp is a single online pass (the max and the scaled sum travel together), so the row is
//                   read once; `targets[row] < 0` marks a position with nothing to score and writes NaN.
//   f32_to_f16      the head's GEMM takes FP16 activations (`Gemm::native`); the mixer's output is FP32.
//
// The numbers are the engine's own: the same BF16 head weights, the same dequantization, the same FP32 logits.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Device scratch `score_rows` needs: rows x 2 floats (one (max, sum) pair per row).  Zero for invalid sizes.
size_t score_rows_scratch_bytes(int rows);
/// `rows` rows of `stride` floats starting at `logits`; `targets` (device, `rows` int32).  `out` (device, `rows`
/// floats) receives log p(target).  Every row must hold at least `vocab` finite logits.
bool score_rows(const float* logits, int rows, int vocab, int64_t stride, const int32_t* targets, void* scratch,
                float* out, void* stream);

/// `n` FP32 values -> their FP16 image (round to nearest even), for the head's GEMM.
bool f32_to_f16(const float* in, uint16_t* out, int64_t n, void* stream);

}  // namespace strata::prefill
