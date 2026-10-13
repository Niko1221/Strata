#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/state_codec.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    cudaStream_t s = streams[(size_t) dev];
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess) return e;
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
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
    o.state_codec = c.state_codec;
    o.used = c.used;
    o.pinned = c.pinned;
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

/// The same fp32 values `src` holds under `from`, held instead under `to`. Sliced and beaten like every other copy
/// of parked state, because a coded conversion of the running state is tens of megabytes of pageable memory. An
/// array that is already under `to` is copied, never converted: coding an already-coded array would mean widening
/// it first, which is what a mount does and not what a park does.
inline bool convert_state(const std::vector<uint8_t>& src, StateCodec from, std::vector<uint8_t>& dst, StateCodec to,
                          std::string& error) {
    if (from == to) { dst = copy_beating(src); return true; }
    const size_t fp32 = state_codec_fp32_bytes(src.size(), from);
    if (!fp32 && !src.empty()) { error = "state codec: an array that is not a whole number of values"; return false; }
    const size_t floats = fp32 / 4;
    // built beside the source and swapped in: widening an array inside its own buffer would free it while reading it,
    // and this is called with one vector on both sides when a mount widens a checkpoint in place
    std::vector<uint8_t> out(state_codec_size(fp32, to), 0);
    // one pass, source bytes to destination bytes: an fp32 scratch in between would be a full-size copy in and a
    // full-size copy out of a conversion that is a round to eight significant bits one way and a shift the other
    for (size_t at = 0; at < floats; ) {
        const size_t n = std::min(kStateCodecChunkFloats, floats - at);
        if (!state_transcode_bytes(src.data() + state_codec_size(at * 4, from), n, from, to,
                                   out.data() + state_codec_size(at * 4, to), error)) return false;
        at += n;
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
    dst.swap(out);
    return true;
}

/// A stored copy of one checkpoint under `codec`. `Fp32` and a chain already under `codec` both take the plain
/// beating copy; `block_pos` is int32 positions and is copied whatever the codec says.
inline bool copy_checkpoint_storing(const ConversationCheckpoint& c, StateCodec codec,
                                    ConversationCheckpoint& out, std::string& error) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = copy_beating(c.imgs);
    o.block_pos = copy_beating(c.block_pos);
    o.used = c.used;
    o.pinned = c.pinned;
    StateCodec from = StateCodec::Fp32;
    if (!state_codec_by_code(c.state_codec, from)) {
        error = "state codec: a checkpoint names a codec this build does not know";
        return false;
    }
    o.state_codec = c.state_codec;
    if (from == codec) {   // nothing to convert: the plain beating copy, byte for byte
        o.gdn = copy_beating(c.gdn); o.ple = copy_beating(c.ple);
        o.tails = copy_beating(c.tails); o.dead = copy_beating(c.dead);
    } else {
        o.state_codec = (uint8_t) codec;
        if (!convert_state(c.gdn, from, o.gdn, codec, error) ||
            !convert_state(c.ple, from, o.ple, codec, error) ||
            !convert_state(c.tails, from, o.tails, codec, error) ||
            !convert_state(c.dead, from, o.dead, codec, error)) return false;
    }
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts)
        if (!copy_checkpoint_storing(part, codec, o.stage_parts.emplace_back(), error)) return false;
    out = std::move(o);
    return true;
}

inline std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs) {
    std::vector<ConversationCheckpoint> out;
    out.reserve(cs.size());
    for (const auto& c : cs) out.push_back(copy_checkpoint_beating(c));
    return out;
}

} // namespace strata::core::conversation_detail
