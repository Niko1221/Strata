#pragma once
// Failure-only diagnostic for the startup profile fill. No CUDA/Win32 calls, heap, worker or class ABI changes.
// OFF by default. A terminal caller writes one bounded snapshot; a killed process may produce no snapshot.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace strata::perf131k::blob_failure {
enum class Source { unknown, file, arena };
enum class Step { pin_alloc, heap_alloc, stage_claim, stage_key, stage_fill,
                  direct_alloc, direct_event, direct_issue, direct_wait, direct_short };
enum class Domain { unknown, cuda, win32 };
inline constexpr unsigned capacity = 32;
// The terminal failure has a dedicated logical slot within the fixed budget.
inline constexpr unsigned event_capacity = capacity - 1;
struct Context { uint64_t request = 0; int64_t layer = -1, expert = -1; Source source = Source::unknown; };
struct Record {
    Context context;
    Step step{};
    Domain domain{};
    uint64_t bytes = 0;
    int64_t code = 0, batch = -1;
    bool code_known = false;
};
struct Slot { Record record; std::atomic<bool> ready{false}; };
inline Slot slots[event_capacity];
inline std::atomic<uint64_t> seen{0}, requests{0};
inline std::atomic<bool> saturated{false};
inline thread_local Context context;

inline bool enabled() noexcept {
    static const bool on = [] { const char* p = std::getenv("STRATA_PERF131K_BLOB_DIAG");
        return p != nullptr && p[0] == '1' && p[1] == '\0'; }();
    return on;
}
inline uint64_t increment(std::atomic<uint64_t>& counter) noexcept {
    uint64_t value = counter.load(std::memory_order_relaxed);
    while (value != std::numeric_limits<uint64_t>::max()) {
        if (counter.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
    saturated.store(true, std::memory_order_relaxed);
    return value;
}
class Scope {
    Context previous_{};
    bool active_ = false;
public:
    Scope(Source source, int64_t layer, int64_t expert) noexcept : active_(enabled()) {
        if (active_) {
            previous_ = context;
            const uint64_t id = increment(requests);
            context = {id == std::numeric_limits<uint64_t>::max() ? 0 : id + 1, layer, expert, source};
        }
    }
    ~Scope() { if (active_) context = previous_; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
inline void record(Step step, uint64_t bytes, Domain domain = Domain::unknown,
                   int64_t code = 0, bool code_known = false, int64_t batch = -1) noexcept {
    if (!enabled()) return;
    const uint64_t index = increment(seen);
    if (index >= event_capacity) return;
    slots[index].record = {context, step, domain, bytes, code, batch, code_known};
    slots[index].ready.store(true, std::memory_order_release);
}
inline const char* name(Source source) noexcept {
    return source == Source::file ? "file" : source == Source::arena ? "arena" : "unknown";
}
inline const char* name(Domain domain) noexcept {
    return domain == Domain::cuda ? "cuda" : domain == Domain::win32 ? "win32" : "unknown";
}
inline const char* name(Step step) noexcept {
    switch (step) {
    case Step::pin_alloc: return "cudaHostAlloc";
    case Step::heap_alloc: return "new_nothrow";
    case Step::stage_claim: return "stage_claim_failed";
    case Step::stage_key: return "stage_key_changed";
    case Step::stage_fill: return "stage_fill_failed";
    case Step::direct_alloc: return "VirtualAlloc";
    case Step::direct_event: return "CreateEventW";
    case Step::direct_issue: return "ReadFile";
    case Step::direct_wait: return "GetOverlappedResult";
    case Step::direct_short: return "short_read";
    }
    return "unknown";
}
// One caller-owned buffer and one fwrite. No allocation in an outstanding-I/O failure path.
// Reserved-but-unpublished slots are skipped: never wait on an I/O worker merely to collect diagnostics.
inline void flush_terminal(uint64_t requested_bytes) noexcept {
    if (!enabled()) return;
    char output[32768];
    size_t used = 0;
    bool truncated = false;
    auto add = [&](const char* format, auto... args) {
        if (truncated) return;
        const int count = std::snprintf(output + used, sizeof(output) - used, format, args...);
        if (count < 0 || (size_t) count >= sizeof(output) - used) { truncated = true; return; }
        used += (size_t) count;
    };
    const uint64_t count = seen.load(std::memory_order_acquire);
    auto unsigned_value = [&](uint64_t value) {
        if (value == 0) add("null"); else add("%llu", (unsigned long long) value);
    };
    auto signed_value = [&](int64_t value) {
        if (value < 0) add("null"); else add("%lld", (long long) value);
    };
    add("BLOB_DIAG {\"schema\":1,\"status\":\"SNAPSHOT\",\"capacity\":%u,\"event_capacity\":%u,\"seen_at_snapshot\":%llu,"
        "\"terminal\":{\"stage\":\"profile_blob_null\",\"source\":\"%s\",\"request\":",
        capacity, event_capacity, (unsigned long long) count, name(context.source));
    unsigned_value(context.request);
    add(",\"layer\":"); signed_value(context.layer);
    add(",\"expert\":"); signed_value(context.expert);
    add(",\"requested_bytes\":%llu},\"records\":[", (unsigned long long) requested_bytes);
    unsigned published = 0;
    for (uint64_t i = 0; i < count && i < event_capacity; ++i) {
        if (!slots[i].ready.load(std::memory_order_acquire)) continue;
        const Record& r = slots[i].record;
        add("%s{\"sequence\":%llu,\"stage\":\"%s\",\"domain\":\"%s\",\"source\":\"%s\","
            "\"request\":", published ? "," : "", (unsigned long long) i,
            name(r.step), name(r.domain), name(r.context.source));
        unsigned_value(r.context.request);
        add(",\"layer\":"); signed_value(r.context.layer);
        add(",\"expert\":"); signed_value(r.context.expert);
        add(",\"batch_index\":"); signed_value(r.batch);
        add(",\"same_terminal_request\":%s",
            context.request != 0 && r.context.request == context.request ? "true" : "false");
        add(",\"requested_bytes\":%llu,\"code\":", (unsigned long long) r.bytes);
        if (r.code_known) add("%lld}", (long long) r.code); else add("null}");
        ++published;
    }
    const uint64_t reserved = count < event_capacity ? count : event_capacity;
    add("],\"pending_at_snapshot\":%llu,\"dropped_at_snapshot\":",
        (unsigned long long) (reserved - published));
    const bool overflow = saturated.load(std::memory_order_relaxed);
    if (overflow) add("null"); else add("%llu", (unsigned long long) (count - reserved));
    add(",\"counter_saturated\":%s,\"later_events\":\"unknown\"}\n", overflow ? "true" : "false");
    if (truncated) { std::fputs("BLOB_DIAG_UNAVAILABLE serialization_limit\n", stderr); return; }
    (void) std::fwrite(output, 1, used, stderr);
}
} // namespace strata::perf131k::blob_failure
