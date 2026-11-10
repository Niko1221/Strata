#pragma once

// The GPU streams a host thread may sit waiting on, for the watchdog's stall report (#5718 follow-up).  A prompt read
// waits in cudaStreamSynchronize on its compute stream; when that wait never returns, the report used to say nothing
// about the stream itself, so a GPU that never finished the work could not be told from a host that was never woken
// after the GPU had finished.  The prompt path names its stream here while it reads (WatchedStream, RAII); the report
// asks each named stream, without blocking, whether its work is done.  Header-only and without the GPU runtime: the
// query is the caller's (generate.cpp's stall report passes cudaStreamQuery), so the list and the lines are tested on
// the CPU (tests/core/stream_watch_test.cpp).

#include <cstdio>
#include <mutex>
#include <string>

namespace strata::core {

enum class StreamState { finished, working, failed };

class StreamWatch {
public:
    static constexpr int kSlots = 16;

    /// A slot for (what, stream, device); -1 when all are taken (the report then simply does not list it).
    int add(const char* what, void* stream, int device) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < kSlots; ++i)
            if (!slots_[i].used) { slots_[i] = {true, what, stream, device}; return i; }
        return -1;
    }
    void remove(int slot) {
        if (slot < 0 || slot >= kSlots) return;
        std::lock_guard<std::mutex> lk(mu_);
        slots_[slot].used = false;
    }
    int count() {
        std::lock_guard<std::mutex> lk(mu_);
        int n = 0;
        for (const Slot& s : slots_) n += s.used ? 1 : 0;
        return n;
    }
    /// One line per watched stream; `query(stream, device, why)` must not block (cudaStreamQuery).  Nothing when
    /// no stream is watched (the stall is elsewhere).  The lines' wording is a DRAFT (RESULTS-GH-FIX.md, robust 3).
    template <class Query>
    void report(std::FILE* f, Query&& query) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const Slot& s : slots_) {
            if (!s.used) continue;
            std::string why;
            const StreamState st = query(s.stream, s.device, why);
            std::fprintf(f, "  [DRAFT] %s (GPU %d): %s\n", s.what, s.device,
                         st == StreamState::finished ? "its work is finished (a host still waiting on it was not woken)"
                         : st == StreamState::working ? "its work is still running on the GPU"
                                                      : ("the query failed: " + why).c_str());
        }
    }

private:
    struct Slot {
        bool used = false;
        const char* what = nullptr;
        void* stream = nullptr;
        int device = -1;
    };
    std::mutex mu_;
    Slot slots_[kSlots];
};

inline StreamWatch& stream_watch() {
    static StreamWatch w;
    return w;
}

/// Names `stream` for the stall report while this object lives (the stream must outlive it; nullptr is the device's
/// default stream, which can be asked too).
class WatchedStream {
public:
    WatchedStream(const char* what, void* stream, int device) : slot_(stream_watch().add(what, stream, device)) {}
    ~WatchedStream() { stream_watch().remove(slot_); }
    WatchedStream(const WatchedStream&) = delete;
    WatchedStream& operator=(const WatchedStream&) = delete;

private:
    int slot_;
};

}  // namespace strata::core
