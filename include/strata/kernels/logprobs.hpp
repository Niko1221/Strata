#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// One chosen token followed by K highest-scoring alternatives for each row.
// Scores are natural log probabilities under the unmodified, full-vocabulary
// softmax of the head logits. The chosen token need not be in the top K.
struct LogprobItem {
    int32_t id;
    float logprob;
};

constexpr int kLogprobsMaxTop = 20;

// Device scratch for logprobs_extract. The head vocabulary must contain at most
// 262144 entries, as in the split sampler. Returns zero for unsupported sizes.
size_t logprobs_scratch_bytes(int rows, int vocab, int top_k);

// All pointers, including chosen and out, are device pointers. Scratch is at
// least logprobs_scratch_bytes(rows, vocab, top_k) bytes. The caller preserves
// logits until the stream has finished, and transfers only rows*(K+1) items.
// Returns false without launching for invalid dimensions.
bool logprobs_extract(const float* logits, int rows, int vocab, const int32_t* chosen,
                      int top_k, void* scratch, LogprobItem* out, void* stream);

}  // namespace strata::kernels
