#pragma once
// Opt-in host counters only: no CUDA events, synchronization or additional I/O.
#include "strata/qwen35/startup_trace.hpp"
#include <array>

namespace strata::qwen35 {
enum class Q36Phase { load, warmup, prefill, decode, snapshot, cancel, unload };
inline const char *q36_phase_name(Q36Phase phase) {
  switch (phase) {
  case Q36Phase::load:
    return "load";
  case Q36Phase::warmup:
    return "warmup";
  case Q36Phase::prefill:
    return "prefill";
  case Q36Phase::decode:
    return "decode";
  case Q36Phase::snapshot:
    return "snapshot";
  case Q36Phase::cancel:
    return "cancel";
  case Q36Phase::unload:
    return "unload";
  }
  return "unknown";
}
inline bool q36_phase_metrics_enabled() {
  const char *value = std::getenv("STRATA_Q36_PHASE_METRICS");
  return value && std::strcmp(value, "1") == 0;
}
struct Q36PhaseCounters {
  uint64_t read_calls = 0, read_requested_bytes = 0, read_completed_bytes = 0,
           read_failed_calls = 0;
  uint64_t expert_h2d_submitted_bytes = 0, expert_d2d_submitted_bytes = 0,
           cpu_jobs = 0;
  double read_wall_ms = 0, cpu_compute_ms = 0, route_wait_ms = 0;
};
class Q36PhaseTrace;
inline thread_local Q36PhaseTrace *q36_active_phase = nullptr;
inline thread_local int q36_active_phase_layer = -1;

class Q36PhaseTrace {
  using Clock = std::chrono::steady_clock;
  Q36Phase phase_;
  uint64_t request_;
  bool enabled_, finished_ = false;
  FILE *output_;
  Clock::time_point start_{};
  Q36PhaseTrace *previous_;
  int previous_layer_;
  std::array<Q36PhaseCounters, 65>
      counters_{}; // last entry: no layer attribution

  void emit(const char *state, int layer, const Q36PhaseCounters &c,
            double wall) const {
    std::fprintf(
        output_,
        "Q36_PHASE schema=1 generation=%s request=%llu phase=%s state=%s "
        "layer=%d wall_ms=%.6f "
        "read_calls=%llu read_requested_bytes=%llu read_completed_bytes=%llu "
        "read_failed_calls=%llu "
        "expert_h2d_submitted_bytes=%llu expert_d2d_submitted_bytes=%llu "
        "cpu_jobs=%llu "
        "read_wall_ms=%.6f cpu_compute_ms=%.6f route_wait_ms=%.6f "
        "input_tokens=%llu reused_tokens=%llu processed_tokens=%llu "
        "committed_tokens=%llu "
        "instrumentation=host_counters gpu_time=unavailable "
        "disk_physical_bytes=unavailable\n",
        q36_startup_generation(), (unsigned long long)request_,
        q36_phase_name(phase_), state, layer, wall,
        (unsigned long long)c.read_calls,
        (unsigned long long)c.read_requested_bytes,
        (unsigned long long)c.read_completed_bytes,
        (unsigned long long)c.read_failed_calls,
        (unsigned long long)c.expert_h2d_submitted_bytes,
        (unsigned long long)c.expert_d2d_submitted_bytes,
        (unsigned long long)c.cpu_jobs, c.read_wall_ms, c.cpu_compute_ms,
        c.route_wait_ms, (unsigned long long)input_tokens,
        (unsigned long long)reused_tokens, (unsigned long long)processed_tokens,
        (unsigned long long)committed_tokens);
  }

public:
  uint64_t input_tokens = 0, reused_tokens = 0, processed_tokens = 0,
           committed_tokens = 0;
  explicit Q36PhaseTrace(Q36Phase phase, uint64_t request = 0,
                         bool enabled = q36_phase_metrics_enabled(),
                         FILE *output = stderr)
      : phase_(phase), request_(request), enabled_(enabled), output_(output),
        previous_(q36_active_phase), previous_layer_(q36_active_phase_layer) {
    // A disabled nested scope masks the parent; batch/warmup work must not
    // accidentally be attributed to a surrounding single-request phase.
    q36_active_phase = enabled_ ? this : nullptr;
    q36_active_phase_layer = -1;
    if (enabled_)
      start_ = Clock::now();
  }
  Q36PhaseTrace(const Q36PhaseTrace &) = delete;
  Q36PhaseTrace &operator=(const Q36PhaseTrace &) = delete;
  Q36PhaseCounters &counters() {
    return counters_[(size_t)(q36_active_phase_layer >= 0 &&
                                      q36_active_phase_layer < 64
                                  ? q36_active_phase_layer
                                  : 64)];
  }
  Q36PhaseCounters total() const {
    Q36PhaseCounters result;
    for (const auto &c : counters_) {
      result.read_calls += c.read_calls;
      result.read_requested_bytes += c.read_requested_bytes;
      result.read_completed_bytes += c.read_completed_bytes;
      result.read_failed_calls += c.read_failed_calls;
      result.expert_h2d_submitted_bytes += c.expert_h2d_submitted_bytes;
      result.expert_d2d_submitted_bytes += c.expert_d2d_submitted_bytes;
      result.cpu_jobs += c.cpu_jobs;
      result.read_wall_ms += c.read_wall_ms;
      result.cpu_compute_ms += c.cpu_compute_ms;
      result.route_wait_ms += c.route_wait_ms;
    }
    return result;
  }
  void finish(const char *state = "complete") {
    if (finished_)
      return;
    if (enabled_) {
      const double wall =
          std::chrono::duration<double, std::milli>(Clock::now() - start_)
              .count();
      emit(state, -1, total(), wall);
      for (size_t i = 0; i < counters_.size(); ++i) {
        const auto &c = counters_[i];
        if (c.read_calls || c.expert_h2d_submitted_bytes ||
            c.expert_d2d_submitted_bytes || c.cpu_jobs || c.route_wait_ms)
          emit(state, (int)i, c, wall);
      }
      std::fflush(output_);
    }
    finished_ = true;
    q36_active_phase = previous_;
    q36_active_phase_layer = previous_layer_;
  }
  ~Q36PhaseTrace() { finish("incomplete"); }
};
class Q36PhaseLayer {
  int previous_;

public:
  explicit Q36PhaseLayer(int layer) : previous_(q36_active_phase_layer) {
    q36_active_phase_layer = layer;
  }
  ~Q36PhaseLayer() { q36_active_phase_layer = previous_; }
};
// A failed call can have performed partial reads. Only fully successful API
// bytes enter read_completed_bytes; failure counts prohibit treating it as
// total I/O.
class Q36ReadTrace {
  Q36PhaseTrace *phase_ = q36_active_phase;
  int layer_ = q36_active_phase_layer;
  std::chrono::steady_clock::time_point start_{};
  uint64_t bytes_;
  bool completed_ = false;

public:
  explicit Q36ReadTrace(uint64_t bytes) : bytes_(bytes) {
    if (phase_)
      start_ = std::chrono::steady_clock::now();
  }
  void complete(bool success) { completed_ = success; }
  ~Q36ReadTrace() {
    if (!phase_)
      return;
    Q36PhaseLayer layer(layer_);
    auto &c = phase_->counters();
    ++c.read_calls;
    c.read_requested_bytes += bytes_;
    if (completed_)
      c.read_completed_bytes += bytes_;
    else
      ++c.read_failed_calls;
    c.read_wall_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start_)
                          .count();
  }
};
inline void q36_phase_h2d(uint64_t bytes) {
  if (q36_active_phase)
    q36_active_phase->counters().expert_h2d_submitted_bytes += bytes;
}
inline void q36_phase_d2d(uint64_t bytes) {
  if (q36_active_phase)
    q36_active_phase->counters().expert_d2d_submitted_bytes += bytes;
}
} // namespace strata::qwen35
