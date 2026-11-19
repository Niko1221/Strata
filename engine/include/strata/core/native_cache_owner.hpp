#pragma once
#include "strata/core/native_cache_config.hpp"
namespace strata::core {
struct NativeCacheQueryBatch {
  const float *src = nullptr;
  size_t src_bytes = 0, src_head_stride = 0, src_token_stride = 0;
  float *dst = nullptr;
  size_t dst_bytes = 0, dst_head_stride = 0, dst_token_stride = 0;
  size_t tokens = 0;
};
// One session/layer, one ordered explicit nondefault stream, one borrowed
// 256-aligned GPU region. Owns ONLY host metadata and a shared immutable plan.
// Caller selects the device and allocates/frees region/stream. Never concurrent
// calls, hidden full-FP32 history, cudaMalloc, device switching or hot-call
// sync.
//
// Lifetime for future SessionState/QsaState integration: SessionState must own
// the owner container; a layer may borrow its pointer. Retain the plan, owners,
// GPU region, parameter bank, persistent commands and external buffers through
// completion AND every graph replay. Quiesce/destroy graphs before freeing any
// of them. Destructor performs no CUDA work and cannot establish quiescence.
// This API never claims that other RNN/indexer/session state remains valid.
class NativeCacheOwner final {
public:
  // Explicit quiescent init/upload/audit/sync. No graph capture. On failure,
  // caller must still quiesce its stream before reclaiming any borrowed buffer.
  static std::unique_ptr<NativeCacheOwner>
  attach(std::shared_ptr<const NativeCachePlan>, void *arena,
         size_t arena_bytes, int device, void *stream, int &result);
  ~NativeCacheOwner() = default;
  NativeCacheOwner(const NativeCacheOwner &) = delete;
  NativeCacheOwner &operator=(const NativeCacheOwner &) = delete;
  int initialize_quiescent();
  int append(const tqc_cache_cuda_append &, size_t accessible_tokens);
  int attend(const tqc_cache_cuda_attention &, size_t accessible_tokens);
  // Input is ALREADY Q-normalized; caller places this before original RoPE.
  // tokens declares actual accessible input, not persistent GPU query count.
  // No adapter uses the same v3 zero-output identity bank, never extra RMS.
  int query(const NativeCacheQueryBatch &);
  int control(uint32_t operation);
  int audit();
  // Only invoke after successful caller completion/readback on the owning
  // stream. Output consumption requires both cache AND query status, plus
  // runtime completion. Any failure sticks on host even if GPU flag writes
  // failed; subsequent calls refuse until explicit quiescent reinitialisation.
  bool observe_completion(bool runtime_succeeded, const tqc_cache_cuda_state &,
                          int query_status);
  bool host_usable() const { return usable_; } // not an unobserved GPU status
  const tqc_cache_cuda_context &context() const { return context_; }
  const NativeCachePlan &plan() const { return *plan_; }
  tqc_cache_cuda_command *command_device() const;
  tqc_cache_cuda_state *cache_status_device() const;
  int *query_status_device() const;

private:
  NativeCacheOwner(std::shared_ptr<const NativeCachePlan>, void *, int, void *);
  int ready();
  int result(int code);
  std::shared_ptr<const NativeCachePlan> plan_;
  tqc_cache_cuda_context context_{};
  bool usable_ = false;
};
} // namespace strata::core