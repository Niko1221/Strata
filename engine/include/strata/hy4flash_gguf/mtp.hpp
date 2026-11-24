#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4_flash {
struct Geometry;
struct DenseResidency;

// One hy4-flash nextn layer. Shares the target embedding and output head. Its experts
// can stay in mapped host memory; latent KV, indexer and scratch stay on the GPU.
class Mtp {
public:
    Mtp(const std::string& gguf, const Geometry& geometry, int context);
    ~Mtp();
    Mtp(const Mtp&) = delete;
    Mtp& operator=(const Mtp&) = delete;
    void bind(DenseResidency& target);
    void checkpoint();
    void rewind(int position);
    int position() const;
    // Committed prefix rows, accessed by serving between requests.
    struct KvCache& kv_cache();
    struct IndexerCache& indexer_cache();
    void forget_speculation();
    std::size_t weight_bytes() const;
    // h_previous is device [tokens, hidden], the target/draft state one token
    // BEFORE each token. Returned hidden rows are device memory owned by Mtp.
    const float* forward(const std::vector<int32_t>& tokens, const float* h_previous,
                         std::vector<float>* last_logits = nullptr);
    void catchup(const std::vector<int32_t>& tokens,const float* h_previous);
    std::vector<int32_t> draft(int32_t token,const float* h_previous,int count);
    // Queue accepted-prefix cache repair and the next round's first draft before
    // host reads the verdict. Returns false for a not-yet-warmed shape; no work
    // is queued in that case. CUDA13 conditional graphs, separately opt-in.
    bool continue_device(const struct RoundVerdict* verdict,void* verdict_stream,
        const std::vector<int32_t>& window,const float* target_hidden,int base,int remaining);
    void finish_continue(const struct RoundVerdict& verdict,bool ran);
    bool loop_ready(int tokens,bool sparse);
    void record_loop_draft(struct DeviceLoopState* state,int32_t* window,const float* hidden,int tokens,bool sparse,void* stream);
    void record_loop_commit(const struct DeviceLoopState* state,const struct RoundVerdict* verdict,
        const int32_t* window,const float* hidden,int tokens,void* stream);
    void loop_mirror(int base);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::hy4_flash
