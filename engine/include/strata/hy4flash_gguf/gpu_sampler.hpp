#pragma once
#include "strata/hy4flash_gguf/sampler.hpp"
#include "strata/hy4flash_gguf/round_state.hpp"
#include <memory>

namespace strata::hy4_flash {
// Full-vocabulary sampling follows the host filter order and SplitMix64 RNG sequence.
class GpuSampler {
public:
    GpuSampler();
    ~GpuSampler();
    GpuSampler(const GpuSampler&)=delete;
    GpuSampler& operator=(const GpuSampler&)=delete;
    void configure(const SamplerConfig& config);
    int32_t pick(const float* device_logits,int vocab,const std::vector<int32_t>& history);
    // Queue sampling and acceptance after logits are ready. Result pointers remain valid until the next sampler
    // call. Restore RNG to the accepted prefix.
    const RoundVerdict* begin_verify(const float* logits,int vocab,const std::vector<int32_t>& drafts,
        const std::vector<int32_t>& history,int base,int remaining,const std::vector<int32_t>& stop);
    void* stream() const;
    RoundVerdict finish_verify();
    void prepare_loop(int vocab,int rows,const std::vector<int32_t>& stop);
    const RoundVerdict* record_loop(const float* logits,int vocab,int rows,const int32_t* window,
        const int32_t* history,struct DeviceLoopState* state,void* stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
