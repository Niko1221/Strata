#include "strata/core/native_cache_owner.hpp"
#include "spin8/query_v3_cuda.h"
#include <cuda_runtime_api.h>
#include <new>
namespace strata::core {
namespace {
bool range(const void *p, size_t bytes) {
  return p && bytes && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p);
}
bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a),
             y = reinterpret_cast<uintptr_t>(b);
  return x < y + bn && y < x + an;
}
} // namespace
NativeCacheOwner::NativeCacheOwner(std::shared_ptr<const NativeCachePlan> p,
                                   void *arena, int device, void *stream)
    : plan_(std::move(p)) {
  context_ = {1,
              sizeof(context_),
              plan_->cache_plan(),
              arena,
              plan_->layout().total_bytes,
              device,
              stream};
}
std::unique_ptr<NativeCacheOwner>
NativeCacheOwner::attach(std::shared_ptr<const NativeCachePlan> plan,
                         void *arena, size_t bytes, int device, void *stream,
                         int &code) {
  code = TQC_D_INVALID;
  if (!plan || !range(arena, bytes) ||
      reinterpret_cast<uintptr_t>(arena) % 256 || device < 0 || !stream ||
      stream == reinterpret_cast<void *>(1) ||
      stream == reinterpret_cast<void *>(2))
    return {};
  if (bytes < plan->layout().total_bytes) {
    code = TQC_D_CAPACITY;
    return {};
  }
  try {
    std::unique_ptr<NativeCacheOwner> owner(
        new NativeCacheOwner(std::move(plan), arena, device, stream));
    code = owner->initialize_quiescent();
    if (code)
      return {};
    return owner;
  } catch (const std::bad_alloc &) {
    code = TQC_D_CAPACITY;
    return {};
  }
}
int NativeCacheOwner::result(int code) {
  if (code == TQC_DC_RUNTIME)
    usable_ = false;
  return code;
}
int NativeCacheOwner::ready() {
  if (!usable_)
    return TQC_D_INVALID;
  int current = -1, stream_device = -1;
  if (cudaGetDevice(&current) != cudaSuccess ||
      cudaStreamGetDevice(static_cast<cudaStream_t>(context_.stream),
                          &stream_device) != cudaSuccess)
    return result(TQC_DC_RUNTIME);
  if (current != context_.device_ordinal || stream_device != current)
    return TQC_D_UNSUPPORTED;
  return 0;
}
int NativeCacheOwner::initialize_quiescent() {
  usable_ = false;
  auto stream = static_cast<cudaStream_t>(context_.stream);
  int code = tqc_cache_cuda_initialize(&context_);
  if (code) {
    // Library runtime failures can follow queued uploads. Attempt completion
    // before an attach failure releases this owner's shared host plan.
    if (code == TQC_DC_RUNTIME)
      (void)cudaStreamSynchronize(stream);
    return code;
  }
  const auto &l = plan_->layout();
  const auto v = plan_->parameters();
  const void *sources[4] = {v.win, v.bin, v.wout, v.bout};
  bool failed = false;
  for (size_t i = 0; i < 4 && !failed; ++i)
    failed = cudaMemcpyAsync(static_cast<char *>(context_.arena) +
                                 l.parameter_offsets[i],
                             sources[i], l.parameter_bytes[i],
                             cudaMemcpyHostToDevice, stream) != cudaSuccess;
  if (!failed)
    failed = cudaMemsetAsync(query_status_device(), 0, sizeof(int), stream) !=
             cudaSuccess;
  tqc_cache_cuda_state state{};
  int query_status = -1;
  if (!failed)
    failed = cudaMemcpyAsync(&state, cache_status_device(), sizeof(state),
                             cudaMemcpyDeviceToHost, stream) != cudaSuccess;
  if (!failed)
    failed = cudaMemcpyAsync(&query_status, query_status_device(),
                             sizeof(query_status), cudaMemcpyDeviceToHost,
                             stream) != cudaSuccess;
  // Unconditional completion attempt: never return early with an enqueued
  // D2H copy still borrowing this function's stack, even if a later copy fails.
  const auto completion = cudaStreamSynchronize(stream);
  if (failed || completion != cudaSuccess)
    return TQC_DC_RUNTIME;
  usable_ = state.valid == 1 && state.status == 0 && query_status == 0;
  return usable_ ? 0 : TQC_D_INVALID;
}
int NativeCacheOwner::append(const tqc_cache_cuda_append &r, size_t tokens) {
  int code = ready();
  return code ? code
              : result(
                    tqc_cache_cuda_append_rows_bounded(&context_, &r, tokens));
}
int NativeCacheOwner::attend(const tqc_cache_cuda_attention &r, size_t tokens) {
  int code = ready();
  return code ? code
              : result(tqc_cache_cuda_attend_bounded(&context_, &r, tokens));
}
int NativeCacheOwner::query(const NativeCacheQueryBatch &r) {
  if (!usable_)
    return TQC_D_INVALID;
  if (!r.tokens || r.tokens > plan_->max_queries() ||
      !range(r.src, r.src_bytes) || !range(r.dst, r.dst_bytes))
    return TQC_D_INVALID;
  if (overlaps(r.src, r.src_bytes, context_.arena, context_.arena_bytes) ||
      overlaps(r.dst, r.dst_bytes, context_.arena, context_.arena_bytes))
    return TQC_D_OVERLAP;
  int code = ready();
  if (code)
    return code;
  const auto &l = plan_->layout();
  const auto &g = plan_->geometry();
  auto ptr = [&](size_t i) {
    return reinterpret_cast<const float *>(
        static_cast<const char *>(context_.arena) + l.parameter_offsets[i]);
  };
  const spin8_query_v3_cuda_bank_f32 bank{SPIN8_QUERY_V3_CUDA_BANK_VERSION,
                                          g.q_heads,
                                          g.head_width,
                                          ptr(0),
                                          l.parameter_bytes[0],
                                          size_t(8) * g.head_width * 4,
                                          ptr(1),
                                          l.parameter_bytes[1],
                                          size_t(8) * 4,
                                          ptr(2),
                                          l.parameter_bytes[2],
                                          size_t(56) * 4,
                                          ptr(3),
                                          l.parameter_bytes[3],
                                          size_t(7) * 4};
  auto status = spin8_query_v3_cuda_apply_batched_f32(
      plan_->query_plan(), &bank, r.src, r.src_bytes, r.src_head_stride,
      r.src_token_stride, r.dst, r.dst_bytes, r.dst_head_stride,
      r.dst_token_stride, g.q_heads, r.tokens, g.head_width,
      query_status_device(), static_cast<cudaStream_t>(context_.stream));
  // The CUDA API can also report an earlier asynchronous failure. Even a
  // structural CUDA rejection conservatively stops this owner, not the model.
  return status == cudaSuccess ? 0 : result(TQC_DC_RUNTIME);
}
int NativeCacheOwner::control(uint32_t operation) {
  int code = ready();
  if (code)
    return code;
  code = result(tqc_cache_cuda_control(&context_, operation));
  if (!code && operation == TQC_CC_INVALIDATE)
    usable_ = false;
  return code;
}
int NativeCacheOwner::audit() {
  int code = ready();
  return code ? code : result(tqc_cache_cuda_audit_device(&context_));
}
bool NativeCacheOwner::observe_completion(bool runtime_succeeded,
                                          const tqc_cache_cuda_state &state,
                                          int query_status) {
  usable_ = usable_ && runtime_succeeded && state.valid == 1 &&
            state.status == 0 && query_status == 0;
  return usable_;
}
tqc_cache_cuda_command *NativeCacheOwner::command_device() const {
  return reinterpret_cast<tqc_cache_cuda_command *>(
      static_cast<char *>(context_.arena) + plan_->layout().cache.command);
}
tqc_cache_cuda_state *NativeCacheOwner::cache_status_device() const {
  return reinterpret_cast<tqc_cache_cuda_state *>(
      static_cast<char *>(context_.arena) + plan_->layout().cache.state);
}
int *NativeCacheOwner::query_status_device() const {
  return reinterpret_cast<int *>(static_cast<char *>(context_.arena) +
                                 plan_->layout().query_status);
}
} // namespace strata::core