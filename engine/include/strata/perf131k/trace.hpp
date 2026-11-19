// P01-A: bounded host observations only. No CUDA API, GPU timing or disk inference.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::perf131k {

enum class Stage {
    request,
    prefill_host,
    verify_host,
    graph_launch_issue,
    graph_enqueue_host,
    host_wait,
    h2d_issue,
    expert_copy_host,
    expert_direct_read_host
};
inline const char *stage_name(Stage stage) noexcept {
    switch (stage) {
    case Stage::request:
        return "request";
    case Stage::prefill_host:
        return "prefill_host";
    case Stage::verify_host:
        return "verify_host";
    case Stage::graph_launch_issue:
        return "graph_launch_issue";
    case Stage::graph_enqueue_host:
        return "graph_enqueue_host";
    case Stage::host_wait:
        return "host_wait";
    case Stage::h2d_issue:
        return "h2d_issue";
    case Stage::expert_copy_host:
        return "expert_copy_host";
    case Stage::expert_direct_read_host:
        return "expert_direct_read_host";
    }
    return "unknown";
}
inline uint64_t monotonic_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
inline int64_t range_end(int64_t start, int64_t count) noexcept {
    return start >= 0 && count >= 0 && start <= INT64_MAX - count ? start + count : -1;
}

struct Context {
    uint64_t parent = 0, generation = 0;
    int64_t first = -1, last = -1;
};
inline thread_local Context context;
inline std::atomic<uint64_t> next_thread{1};
inline thread_local uint64_t thread_number = 0;
inline std::atomic<uint64_t> diagnostic_failures{0};
struct Span {
    uint64_t id = 0, parent = 0, generation = 0, thread = 0, start = 0, end = 0;
    uint64_t stream = 0, graph = 0, replay = 0, graph_instance = 0;
    int64_t first = -1, last = -1, layer = -1, logical_bytes = -1, copy_bytes = -1;
    Stage stage = Stage::request;
    bool complete = false;
};

class Recorder {
  public:
    static constexpr size_t capacity = 4096;
    using Clock = uint64_t (*)();
    explicit Recorder(Clock clock = monotonic_ns) : clock_(clock) {}
    Recorder(const Recorder &) = delete;
    Recorder &operator=(const Recorder &) = delete;

    uint64_t begin(Stage stage, int64_t first, int64_t last, int64_t layer, bool generation, const Context &previous) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || used_ == capacity) {
            ++dropped_;
            return 0;
        }
        Span &s = spans_[used_++];
        s.id = used_;
        s.parent = previous.parent;
        s.generation = generation ? s.id : previous.generation;
        s.first = first >= 0 ? first : previous.first;
        s.last = first >= 0 ? last : previous.last;
        s.layer = layer;
        s.stage = stage;
        // Monotonic IDs survive OS thread-id/TLS-address reuse.
        if (!thread_number)
            thread_number = next_thread.fetch_add(1);
        s.thread = thread_number;
        s.start = clock_();
        return s.id;
    }
    void finish(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_)
            return; // shutdown footer already counted this as unfinished
        auto &s = spans_[id - 1];
        s.end = clock_();
        s.complete = true;
    }
    void metadata(uint64_t id, const void *stream, const void *graph, int64_t logical, int64_t copy) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_)
            return;
        auto &s = spans_[id - 1];
        s.stream = stream ? identity_locked(stream, 1) : 0;
        s.graph = graph ? identity_locked(graph, 2) : 0;
        s.replay = s.graph ? id : 0;
        if (graph) {
            for (const auto &entry : graphs_)
                if (entry.key == graph) {
                    s.graph_instance = entry.instance;
                    break;
                }
            if (!s.graph_instance)
                ++unregistered_graphs_;
        }
        s.logical_bytes = logical;
        s.copy_bytes = copy;
    }
    void graph_created(const void *handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || !handle)
            return;
        for (auto &entry : graphs_) {
            if (entry.key == handle) {
                // A duplicate live handle means a retirement hook was missed.
                ++unregistered_graphs_;
                entry.instance = next_graph_++;
                return;
            }
        }
        for (auto &entry : graphs_) {
            if (!entry.key) {
                entry = {handle, next_graph_++};
                return;
            }
        }
        ++identity_overflow_;
    }
    void graph_retired(const void *handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || !handle)
            return;
        for (auto &entry : graphs_)
            if (entry.key == handle) {
                entry = {};
                return;
            }
    }
    // One shutdown export, after the engine's owned workers have joined. Not periodic I/O.
    bool write_and_close(std::FILE *output) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_)
            return false;
        closed_ = true;
        size_t complete = 0;
        bool ok = std::fprintf(output,
                               "{\"type\":\"header\",\"schema\":1,\"source\":\"host_only\",\"clock\":\"steady_ns\","
                               "\"capacity\":%zu}\n",
                               capacity) > 0;
        for (size_t i = 0; i < used_; ++i) {
            const auto &s = spans_[i];
            if (!s.complete)
                continue;
            ++complete;
            char instance[32] = "null";
            if (s.graph_instance)
                std::snprintf(instance, sizeof(instance), "%llu", (unsigned long long)s.graph_instance);
            const int n = std::fprintf(
                output,
                "{\"type\":\"span\",\"id\":%llu,\"parent\":%llu,\"generation\":%llu,\"thread\":%llu,"
                "\"stage\":\"%s\",\"start_ns\":%llu,\"end_ns\":%llu,\"token_start\":%lld,\"token_end\":%lld,"
                "\"layer\":%lld,\"stream\":%llu,\"graph_handle_id\":%llu,\"graph_instance\":%s,\"graph_replay\":%llu,"
                "\"logical_read_requested_bytes\":%lld,\"copy_requested_bytes\":%lld,"
                "\"gpu_elapsed_ns\":null,\"physical_read_bytes\":null,\"event_generation\":null,\"allocation_id\":null}"
                "\n",
                (unsigned long long)s.id, (unsigned long long)s.parent, (unsigned long long)s.generation,
                (unsigned long long)s.thread, stage_name(s.stage), (unsigned long long)s.start,
                (unsigned long long)s.end, (long long)s.first, (long long)s.last, (long long)s.layer,
                (unsigned long long)s.stream, (unsigned long long)s.graph, instance, (unsigned long long)s.replay,
                (long long)s.logical_bytes, (long long)s.copy_bytes);
            ok = n > 0 && ok;
        }
        ok = std::fprintf(output,
                          "{\"type\":\"footer\",\"records\":%zu,\"dropped\":%llu,\"unfinished\":%zu,\"identity_"
                          "overflow\":%llu,\"diagnostic_failures\":%llu,\"unregistered_graphs\":%llu}\n",
                          complete, (unsigned long long)dropped_, used_ - complete,
                          (unsigned long long)identity_overflow_, (unsigned long long)diagnostic_failures.load(),
                          (unsigned long long)unregistered_graphs_) > 0 &&
             ok;
        return std::fflush(output) == 0 && ok;
    }

  private:
    struct Identity {
        const void *key = nullptr;
        int kind = 0;
    };
    struct Graph {
        const void *key = nullptr;
        uint64_t instance = 0;
    };
    uint64_t identity_locked(const void *key, int kind) {
        for (size_t i = 0; i < identities_used_; ++i)
            if (identities_[i].key == key && identities_[i].kind == kind)
                return i + 1;
        if (identities_used_ == identities_.size()) {
            ++identity_overflow_;
            return 0;
        }
        identities_[identities_used_++] = {key, kind};
        return identities_used_;
    }
    Clock clock_;
    std::mutex mutex_;
    std::array<Span, capacity> spans_{};
    std::array<Identity, 256> identities_{};
    std::array<Graph, 256> graphs_{};
    size_t used_ = 0, identities_used_ = 0;
    uint64_t dropped_ = 0, identity_overflow_ = 0;
    uint64_t next_graph_ = 1, unregistered_graphs_ = 0;
    bool closed_ = false;
};

inline std::atomic<Recorder *> active{nullptr};
inline void graph_created(const void *handle) noexcept {
    if (auto *recorder = active.load(std::memory_order_acquire)) {
        try {
            recorder->graph_created(handle);
        } catch (...) {
            diagnostic_failures.fetch_add(1);
        }
    }
}
inline void graph_retired(const void *handle) noexcept {
    if (auto *recorder = active.load(std::memory_order_acquire)) {
        try {
            recorder->graph_retired(handle);
        } catch (...) {
            diagnostic_failures.fetch_add(1);
        }
    }
}
class Scope {
  public:
    explicit Scope(Stage stage, int64_t first = -1, int64_t last = -1, int64_t layer = -1,
                   bool new_generation = false) noexcept {
        recorder_ = active.load(std::memory_order_acquire);
        if (!recorder_)
            return; // OFF: no TLS, lock, clock, allocation, file or CUDA access.
        previous_ = context;
        try {
            id_ = recorder_->begin(stage, first, last, layer, new_generation, previous_);
        } catch (...) {
            diagnostic_failures.fetch_add(1);
            return;
        }
        if (id_) {
            context = {id_, new_generation ? id_ : previous_.generation, first >= 0 ? first : previous_.first,
                       first >= 0 ? last : previous_.last};
        }
    }
    ~Scope() {
        if (id_) {
            try {
                recorder_->finish(id_);
            } catch (...) {
                diagnostic_failures.fetch_add(1);
            }
            context = previous_;
        }
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    void metadata(const void *stream = nullptr, const void *graph = nullptr, int64_t logical_bytes = -1,
                  int64_t copy_requested_bytes = -1) {
        if (id_) {
            try {
                recorder_->metadata(id_, stream, graph, logical_bytes, copy_requested_bytes);
            } catch (...) {
                diagnostic_failures.fetch_add(1);
            }
        }
    }

  private:
    Recorder *recorder_ = nullptr;
    uint64_t id_ = 0;
    Context previous_{};
};

inline std::FILE *exclusive_output(const char *path) noexcept {
#if defined(_WIN32)
    int fd = -1;
    if (_sopen_s(&fd, path, _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY, _SH_DENYRW, _S_IREAD | _S_IWRITE) != 0)
        return nullptr;
    std::FILE *file = _fdopen(fd, "wb");
    if (!file)
        _close(fd);
#else
    int fd = ::open(path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0)
        return nullptr;
    std::FILE *file = fdopen(fd, "wb");
    if (!file)
        ::close(fd);
#endif
    return file;
}

// Instantiate first in main, so its destructor follows the existing worker lifetimes.
class Runtime {
  public:
    Runtime() noexcept {
        const char *path = std::getenv("STRATA_PERF131K_HOST_TRACE");
        if (!path || !*path)
            return;
        // An explicit absolute destination is required. Never overwrite an existing file.
#if defined(_WIN32)
        if (!((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) || path[1] != ':' ||
            (path[2] != '\\' && path[2] != '/')) {
            warn();
            return;
        }
#else
        if (path[0] != '/') {
            warn();
            return;
        }
#endif
        try {
            // A process has one engine session. Refuse repeated/concurrent owners.
            static std::atomic_flag claimed = ATOMIC_FLAG_INIT;
            if (claimed.test_and_set()) {
                warn();
                return;
            }
            // Storage lives until process exit; shutdown cannot invalidate an in-flight scope.
            static std::unique_ptr<Recorder> storage = std::make_unique<Recorder>();
            if (active.load(std::memory_order_acquire)) {
                warn();
                return;
            }
            file_ = exclusive_output(path);
            if (!file_) {
                warn();
                return;
            }
            recorder_ = storage.get();
            active.store(recorder_, std::memory_order_release);
        } catch (...) {
            warn();
        }
    }
    ~Runtime() {
        if (!file_)
            return;
        active.store(nullptr, std::memory_order_release);
        bool ok = false;
        try {
            ok = recorder_->write_and_close(file_);
        } catch (...) {
            diagnostic_failures.fetch_add(1);
        }
        if (std::fclose(file_) != 0 || !ok)
            warn();
    }
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

  private:
    static void warn() noexcept {
        std::fprintf(stderr, "strata: host trace unavailable or incomplete; no performance claim\n");
    }
    std::FILE *file_ = nullptr;
    Recorder *recorder_ = nullptr;
};
} // namespace strata::perf131k
