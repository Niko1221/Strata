#pragma once
// One stderr line per Qwen3.6 startup stage. stdout stays the serve protocol.
// elapsed_ms and stage_elapsed_ms are steady-clock milliseconds for the stage
// that just finished. They are not a clock since process start.
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::qwen35 {
inline const auto q36_process_started = std::chrono::steady_clock::now();
inline const char* q36_startup_generation() {
    const char* value = std::getenv("STRATA_Q36_STARTUP_GENERATION");
    if (!value || !*value || std::strlen(value) > 80) return "null";
    for (const char* p = value; *p; ++p)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == '-')) return "null";
    return value;
}
inline bool q36_startup_detail_enabled() {
    const char* value = std::getenv("STRATA_Q36_STARTUP_DETAIL");
    return value && std::strcmp(value, "1") == 0;
}
class Q36StartupDetail;
inline thread_local Q36StartupDetail* q36_active_detail = nullptr;

// Host clock intervals only. No CUDA calls, extra file reads, or diagnostic synchronisation.
class Q36StartupDetail {
    const char* stage_;
    bool enabled_, finished_ = false;
    uint64_t completed_ = 0, total_;
    int layer_ = -1;
    std::chrono::steady_clock::time_point start_{}, last_{};
    Q36StartupDetail* previous_;
    void emit(const char* state, const char* error = "null") {
        if (!enabled_) return;
        const auto now = std::chrono::steady_clock::now();
        const auto process_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - q36_process_started).count();
        const auto stage_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
        std::fprintf(stderr, "Q36_STARTUP_DETAIL schema=1 generation=%s stage=%s state=%s process_elapsed_ms=%lld "
                     "stage_elapsed_ms=%lld completed_bytes=%llu total_bytes=%llu layer_index=%d error_code=%s\n",
                     q36_startup_generation(), stage_, state, (long long) process_ms, (long long) stage_ms,
                     (unsigned long long) completed_, (unsigned long long) total_, layer_, error);
        std::fflush(stderr);
        last_ = now;
    }
public:
    explicit Q36StartupDetail(const char* stage, uint64_t total = 0)
        : stage_(stage), enabled_(stage && q36_startup_detail_enabled()), total_(total), previous_(q36_active_detail) {
        if (enabled_) { start_ = last_ = std::chrono::steady_clock::now(); q36_active_detail = this; emit("begin"); }
    }
    void progress(uint64_t completed, int layer = -1) {
        if (!enabled_ || finished_) return;
        completed_ = completed; layer_ = layer;
        if (std::chrono::steady_clock::now() - last_ >= std::chrono::seconds(1)) emit("progress");
    }
    void end() {
        if (finished_) return;
        if (enabled_) { completed_ = total_; emit("end"); }
        finished_ = true;
    }
    void total(uint64_t bytes) { total_ = bytes; }
    void error(const char* code) {
        if (finished_) return;
        emit("error", code); finished_ = true;
    }
    ~Q36StartupDetail() {
        if (enabled_) { if (!finished_) error("stage_failed"); q36_active_detail = previous_; }
    }
};
inline void q36_startup_detail_error(const char* code) {
    if (q36_active_detail) q36_active_detail->error(code);
}
// Implemented beside the existing platform-specific expert-store memory code.
// Called only when detail tracing is enabled. Unknown physical memory is omitted.
void q36_startup_physical_memory(const char* stage);
inline void q36_startup(const char* stage, const char* state, long long elapsed_ms, long long stage_elapsed_ms,
                        const char* error_code) {
    std::fprintf(stderr,
                 "Q36_STARTUP schema=1 stage=%s state=%s elapsed_ms=%lld stage_elapsed_ms=%lld error_code=%s\n",
                 stage ? stage : "unknown", state ? state : "unknown", elapsed_ms, stage_elapsed_ms,
                 error_code && error_code[0] ? error_code : "null");
}
}  // namespace strata::qwen35
