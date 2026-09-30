// include/strata/core/split_head.hpp - the verify window's output head across two GPUs.
//
// The head streams the whole output matrix after the last layer (IQ3_S's Q6_K 521 MB, UD-Q4_K_XL's Q8_0 675 MB): the
// main GPU keeps rows [0, split) and a second GPU holds rows [split, n_vocab), so each streams part of it.  The main
// GPU's window graph writes the head's input rows (the final hyper-connection mix, T x n_in floats) to mapped memory
// and raises a count; the host then launches this GPU's part (the same quantization, projection and per-row argmax
// kernels, so every logit is the one-GPU head's), which writes each token's pick and logit to mapped memory and raises
// a count of its own.  The host keeps the larger logit, on equality the main GPU's (the lower index).  It never waits
// on host memory on this GPU (a display may draw there).  With `logits`, a window's rows are copied to mapped memory
// too, for sampled requests and --window-logits.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class SplitHead {
public:
    SplitHead() = default;
    ~SplitHead();
    SplitHead(const SplitHead&) = delete;
    SplitHead& operator=(const SplitHead&) = delete;

    /// Rows [split, n_vocab) of `output.weight` in the model's shards on `device`, for windows of up to `max_t` tokens.
    /// `main_device` is current again after every call.
    bool init(int device, int main_device, const std::vector<std::string>& shards, int64_t n_in, int64_t n_vocab,
              int64_t split, int max_t, std::string& err);
    bool on() const { return dev_ >= 0; }
    int64_t split() const { return split_; }
    uint64_t bytes() const { return bytes_; }
    /// Mapped: the head's input rows the main GPU writes (max_t x n_in floats) and the count it raises after them.
    float* input() const { return h_in_; }
    uint32_t* handoff() const { return h_hand_; }
    /// Launches this GPU's part for a window of T tokens (after the handoff count has risen); `logits` copies its
    /// rows to logits() too.
    bool submit(int T, bool logits, std::string& err);
    /// Until this GPU's count reaches `n` (the submits so far).
    bool wait(uint32_t n, std::string& err);
    /// Per token: the pick's row within this part, its logit; T x (n_vocab - split) logits after a `logits` submit.
    const int32_t* idx() const { return h_idx_; }
    const float* val() const { return h_val_; }
    const float* logits() const { return h_logits_; }

private:
    bool capture(int T, bool logits, std::string& err);

    int dev_ = -1, main_ = 0, max_t_ = 0, type_ = -1;
    int64_t n_in_ = 0, split_ = 0, rows_ = 0;
    uint64_t bytes_ = 0;
    cudaStream_t s_ = nullptr;
    void* w_ = nullptr;                                         // the rows' GGUF blocks
    float *mixed_ = nullptr, *out_ = nullptr;                   // device: the input, the logits
    uint8_t *xq_ = nullptr, *xil_ = nullptr, *scratch_ = nullptr;
    float* h_in_ = nullptr;                                     // mapped (portable)
    uint32_t *h_hand_ = nullptr, *h_done_ = nullptr;
    int32_t* h_idx_ = nullptr;
    float *h_val_ = nullptr, *h_logits_ = nullptr;
    cudaGraphExec_t exec_[9][2] = {};
};

}  // namespace strata::core
