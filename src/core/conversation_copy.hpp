#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace strata::core { void conversation_checkpoint_join(const void* only); }
namespace strata::core::conversation_detail {

/// The private copy streams and ordering events (below) belong to a thread, but a thread can be short-lived: the layer
/// split's later stages run each prompt chunk on a fresh std::async thread, and a checkpoint part is saved on it.  The
/// handles used to be created per thread and never destroyed, so every such thread left one stream and one event per
/// device behind - a few MiB of VRAM per long cold prompt on the later cards, until a stream could not be created any
/// more and the copy failed with "out of memory" (#3350, #3470).  Now a thread hands its handles back to a process-wide
/// pool when it ends and the next thread takes them: the number of handles is the number of threads copying at once.
/// The pool is leaked on purpose (a thread may end after the statics are gone).
struct CopyHandlePool {
    std::mutex mu;
    std::vector<std::vector<cudaStream_t>> streams;   // free ones, by device
    std::vector<std::vector<cudaEvent_t>> events;
    long streams_created = 0, events_created = 0;
};
inline CopyHandlePool& copy_handle_pool() {
    static CopyHandlePool* p = new CopyHandlePool;
    return *p;
}
/// How many streams / events the process has created for these copies (tests: it stays at the peak number of threads
/// copying at once, not at the number of threads that ever copied).
inline long copy_streams_created() {
    CopyHandlePool& p = copy_handle_pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return p.streams_created;
}
inline long copy_events_created() {
    CopyHandlePool& p = copy_handle_pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return p.events_created;
}
struct ThreadCopyHandles {
    std::vector<cudaStream_t> streams;   // by device, nullptr = none yet
    std::vector<cudaEvent_t> events;
    ThreadCopyHandles() = default;
    ThreadCopyHandles(const ThreadCopyHandles&) = delete;
    ThreadCopyHandles& operator=(const ThreadCopyHandles&) = delete;
    ~ThreadCopyHandles() {
        CopyHandlePool& p = copy_handle_pool();
        std::lock_guard<std::mutex> lk(p.mu);
        if (p.streams.size() < streams.size()) p.streams.resize(streams.size());
        if (p.events.size() < events.size()) p.events.resize(events.size());
        for (size_t d = 0; d < streams.size(); ++d)
            if (streams[d]) p.streams[d].push_back(streams[d]);
        for (size_t d = 0; d < events.size(); ++d)
            if (events[d]) p.events[d].push_back(events[d]);
    }
};
inline ThreadCopyHandles& thread_copy_handles() {
    thread_local ThreadCopyHandles t;
    return t;
}

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
/// private_stream: this thread's stream on the current device: one a finished thread handed back, else a new one on
/// first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    std::vector<cudaStream_t>& streams = thread_copy_handles().streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);
    if (!streams[(size_t) dev]) {
        CopyHandlePool& p = copy_handle_pool();
        std::lock_guard<std::mutex> lk(p.mu);
        if ((size_t) dev < p.streams.size() && !p.streams[(size_t) dev].empty()) {
            streams[(size_t) dev] = p.streams[(size_t) dev].back();
            p.streams[(size_t) dev].pop_back();
        } else {
            e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
            if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
            ++p.streams_created;
        }
    }
    s = streams[(size_t) dev];
    return cudaSuccess;
}

/// #2409: true when a copy on `dev` from this thread needs no new stream: the thread has one, or a thread that ended left
/// one in the pool. false: the next copy creates one (a small device allocation, which can fail on a full card).
inline bool copy_stream_ready(int dev) {
    if (dev < 0) dev = 0;
    const std::vector<cudaStream_t>& streams = thread_copy_handles().streams;
    if ((size_t) dev < streams.size() && streams[(size_t) dev]) return true;
    CopyHandlePool& p = copy_handle_pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return (size_t) dev < p.streams.size() && !p.streams[(size_t) dev].empty();
}

/// STRATA_TEST_COPY_FAIL_AT=N (tests only, #3350): the N-th snapshot / checkpoint copy of the process fails with
/// cudaErrorMemoryAllocation, as when a card has no memory left for the copy.  Unset: nothing.
inline long test_copy_fail_at() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_COPY_FAIL_AT"); return e ? std::atol(e) : 0L; }();
    return v;
}

/// First thing a background copy thread does (STRATA_CKPT_ASYNC, STRATA_PARK_ASYNC): a thread the host creates
/// inherits the host's one CPU (strata/platform/aux_cpus.hpp), where its memcpys would take turns with the decode.
/// Wide first: the mask the aux placement keeps as this thread's own is the one restore() puts back (a request with
/// aux_cpus=0), and the host's one CPU must not be it.  No output depends on where a thread runs.
inline void background_thread_placement() {
#if defined(__linux__)
    cpu_set_t all;
    CPU_ZERO(&all);
    for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
    (void) sched_setaffinity(0, sizeof all, &all);   // the kernel keeps the CPUs the process may use
    strata::aux_cpus::pin_current_thread();          // (a no-op while the feature is off)
#endif
}

/// #879-family (Windows/WDDM): the C++ heap can land inside a virtual range the HIP runtime tracks as pinned
/// host memory.  The pointer attributes then claim type=host/dev=0 for an ordinary pageable allocation and
/// hipMemcpyAsync refuses the copy with "invalid argument" - deterministically at one session size (the heap
/// layout), clean at another, and never on Linux, where the heap placement differs.  And a GB-scale synchronous
/// copy that is accepted can still interleave with expert-cache H2D streaming on the copy engine and land bytes
/// wrong (decode then reads garbage as expert weights and the grouped GEMM writes NaN).  Stage the copy through
/// a genuinely pinned buffer: the device leg sees a real HIP allocation, the per-chunk sync is the ordering
/// WDDM honors at this size, and the host leg is a plain memcpy.  Same bytes, same order.
inline bool copy_ptr_is_device(const void* p) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) { cudaGetLastError(); return false; }
    return a.type == hipMemoryTypeDevice;
#elif defined(STRATA_GPU_METAL)
    (void) p;   // no pointer-attribute query in the Metal shim: the staged path takes no part there
    return false;
#else
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeDevice;
#endif
}

inline cudaError_t staged_copy(void* dst, const void* src, size_t bytes, bool to_host) {
    cudaStream_t s = nullptr;
    const cudaError_t se = private_stream(s);
    void* stage = nullptr;
    constexpr size_t kStage = (size_t) 16 << 20;
    const cudaError_t he = cudaHostAlloc(&stage, bytes < kStage ? bytes : kStage, cudaHostAllocDefault);
    if (se != cudaSuccess) return se;
    if (he != cudaSuccess) return he;
    cudaError_t staging = cudaSuccess;
    for (size_t at = 0; at < bytes; at += kStage) {
        const size_t n = std::min(kStage, bytes - at);
        const cudaError_t st = to_host
            ? cudaMemcpyAsync(stage, (const char*) src + at, n, cudaMemcpyDeviceToHost, s)
            : (std::memcpy(stage, (const char*) src + at, n), cudaMemcpyAsync((char*) dst + at, stage, n, cudaMemcpyHostToDevice, s));
        if (st != cudaSuccess) { staging = st; break; }
        const cudaError_t sy = cudaStreamSynchronize(s);
        if (sy != cudaSuccess) { staging = sy; break; }
        if (to_host) std::memcpy((char*) dst + at, stage, n);
        core::progress_beat();
    }
    cudaFreeHost(stage);   // the pinned buffer is released right away: a held one shifts the WDDM
                           // allocation map again and can break the next graph capture
    return staging;
}

/// `queued`: the copy is only queued on the private stream, and the caller synchronizes the device (or orders a
/// later launch after it with order_after_queued) before anything reads the target or writes the source.
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes, bool queued = false) {
    if (const long at = test_copy_fail_at(); at > 0) {
        static std::atomic<long> copies{0};
        if (++copies == at) return cudaErrorMemoryAllocation;
    }
    // #879 (Windows/WDDM): every synchronous host<->device snapshot copy (running-state and K/V alike) goes
    // staged-first through a pinned 16 MiB buffer; measured on an RX 9070 XT at 262144: direct copies ->
    // non-finite logits every park round, staged-first -> clean.  STRATA_CKPT_STAGE_FIRST=0 opts back out for
    // A/B (then: direct, and on a refusal consume it, retry with the direction explicit, stage as safety net).
    // Queued copies keep the direct path (their ordering is the caller's event), as do device-to-device and
    // host-to-host copies (the VRAM-slot and pool-alias paths).
    if (!queued) {
        const bool src_dev = copy_ptr_is_device(src), dst_dev = copy_ptr_is_device(dst);
        if (src_dev != dst_dev) {
            const bool to_host = src_dev;
            static const bool stage_first = [] {
                const char* v = std::getenv("STRATA_CKPT_STAGE_FIRST");
                return v ? (v[0] != '0') : true;
            }();
            if (stage_first) return staged_copy(dst, src, bytes, to_host);
            cudaStream_t s = nullptr;
            cudaError_t e = private_stream(s);
            if (e == cudaSuccess) e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
            if (e == cudaSuccess) { core::progress_beat(); return cudaSuccess; }
            // The Default-kind copy was refused.  On Windows/WDDM HIP classifies the C++ heap as pinned host
            // memory (hipPointerGetAttributes type 1), so the Default kind infers the wrong direction and the
            // copy is rejected outright - the same bytes copy fine when the direction is stated.  Consume the
            // refusal (a stale last-error would surface as a bogus launch failure in the next check_launch),
            // retry with the direction explicit, and keep the staged path as the safety net.
            cudaGetLastError();
            s = nullptr;
            if (private_stream(s) == cudaSuccess && s != nullptr) {
                cudaError_t r = cudaMemcpyAsync(dst, src, bytes,
                    to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, s);
                if (r == cudaSuccess) r = cudaStreamSynchronize(s);
                if (r == cudaSuccess) { core::progress_beat(); return cudaSuccess; }
                cudaGetLastError();
            }
            return staged_copy(dst, src, bytes, to_host);
        }
    }
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess || queued) return e;
    e = cudaStreamSynchronize(s);
    core::progress_beat();   // #2049: a parked conversation is many of these; each one that lands is progress
    return e;
}

/// #2049 #2065 #1971: a host-to-host copy that tells the watchdog it is moving.  Parked state (the running
/// checkpoints, the K/V segments) sits in ordinary pageable memory for the whole session; when the OS has written
/// it to the pagefile, copying it back runs at the drive's page-in speed (measured in the reports: ~190 hard faults
/// a second, under 1 MB/s) and one memcpy of it stays silent past the 60 s limit, so the watchdog ended an engine
/// that was only slow.  One beat per 1 MiB slice keeps a copy that moves alive and still lets a real stall show.
/// STRATA_TEST_SLOW_COPY_US=N (tests only): every 1 MiB slice of a checkpoint copy takes N microseconds more, as when
/// the OS has paged the checkpoints out and each slice is read back from the drive.  Unset: no delay.
inline long test_slow_copy_us() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_SLOW_COPY_US"); return e ? std::atol(e) : 0L; }();
    return v;
}

template<class T>
inline std::vector<T> copy_beating(const std::vector<T>& src) {
    std::vector<T> out;
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(T));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
    return out;
}

inline ConversationCheckpoint copy_checkpoint_beating(const ConversationCheckpoint& c) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = copy_beating(c.imgs);
    o.gdn = copy_beating(c.gdn);
    o.dev_tag = c.dev_tag;   // the same bytes
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
    o.used = c.used;
    o.pinned = c.pinned;
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

/// copy_beating into `out`'s storage (its capacity kept: a vector whose pages are in takes the copy without faults)
template<class V>
inline void assign_beating(V& out, const V& src) {
    out.clear();
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(typename V::value_type));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
}

/// copy_checkpoint_beating into `o`, a dropped checkpoint's storage (STRATA_CKPT_POOL): the same bytes
inline void copy_checkpoint_into(ConversationCheckpoint& o, const ConversationCheckpoint& c) {
    assign_beating(o.ids, c.ids);
    assign_beating(o.imgs, c.imgs);
    assign_beating(o.gdn, c.gdn);
    o.dev_tag = c.dev_tag;
    assign_beating(o.ple, c.ple);
    assign_beating(o.tails, c.tails);
    assign_beating(o.dead, c.dead);
    assign_beating(o.block_pos, c.block_pos);
    o.used = c.used;
    o.pinned = c.pinned;
    o.stage_parts.clear();
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
}

/// `storage` (optional): dropped checkpoints whose vectors the copies go into, taken from its back while it lasts
inline std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs,
                                                                    std::vector<ConversationCheckpoint>* storage = nullptr) {
    for (const auto& c : cs) core::conversation_checkpoint_join(c.gdn.data());
    std::vector<ConversationCheckpoint> out;
    out.reserve(cs.size());
    for (const auto& c : cs) {
        if (storage != nullptr && !storage->empty()) {
            out.push_back(std::move(storage->back()));
            storage->pop_back();
            copy_checkpoint_into(out.back(), c);
        } else {
            out.push_back(copy_checkpoint_beating(c));
        }
    }
    return out;
}

/// Work queued on the legacy stream after this call starts after every copy this thread queued on the current
/// device's private stream (an event the legacy stream waits for; no host wait).
inline cudaError_t order_after_queued() {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    int dev = 0;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    std::vector<cudaEvent_t>& events = thread_copy_handles().events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, nullptr);
    if (!events[(size_t) dev]) {
        CopyHandlePool& p = copy_handle_pool();
        std::lock_guard<std::mutex> lk(p.mu);
        if ((size_t) dev < p.events.size() && !p.events[(size_t) dev].empty()) {
            events[(size_t) dev] = p.events[(size_t) dev].back();
            p.events[(size_t) dev].pop_back();
        } else {
            e = cudaEventCreateWithFlags(&events[(size_t) dev], cudaEventDisableTiming);
            if (e != cudaSuccess) { events[(size_t) dev] = nullptr; return e; }
            ++p.events_created;
        }
    }
    e = cudaEventRecord(events[(size_t) dev], s);
    if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(nullptr, events[(size_t) dev], 0);
}

} // namespace strata::core::conversation_detail
